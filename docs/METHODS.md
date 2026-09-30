# 方法與訓練參數

## 1. 整體流程

```
原始資料 → 轉換為 nnU-Net 格式（data_conversion/）
        → nnU-Net 規劃與前處理（官方 ExperimentPlanner）
        → 訓練 baseline（anchor）
        → 以 anchor 的 out-of-fold 預測計算 Ψ（psi/run_all_psi.py）
        → 病例層級篩選顯著 proxy（psi/recompute_caselevel.py）
        → 產生加權清單（psi/make_weight_lists.py）
        → 訓練 Ψ 加權專家模型（nnunet_extension/）
        → 測試集推論與評估（官方 nnUNetv2_predict / evaluate_folder）
        → 結果整理（analysis/）
```

---

## 2. 資料與切分

### 2.1 BTCV

| 項目 | 設定 |
|---|---|
| 資料 | 30 例腹部增強 CT（有標註） |
| 切分 | 18 訓練 / 12 測試，沿用 TransUNet 公開清單（亦為 SAMA-UNet 採用） |
| 測試病例（原始編號） | 0001, 0002, 0003, 0004, 0008, 0022, 0025, 0029, 0032, 0035, 0036, 0038 |
| 評估結構 | Aorta, Gallbladder, Kidney L, Kidney R, Liver, Pancreas, Spleen, Stomach（8 / 13 類） |
| 5 折 | nnU-Net 預設切分（`splits_final.json`），驗證折 4 / 4 / 4 / 3 / 3 例 |

測試集標註的檔名（`case0001` 等）與 nnU-Net 編號（`BTCV_019` 等）以**影像內容逐像素比對**建立對應，不依檔名推測（對應表見 `splits/`）。

### 2.2 ACDC

| 項目 | 設定 |
|---|---|
| 資料 | 官方 training 的 100 位病人，每位含 ED / ES 兩個時相 |
| 切分 | 以**病人為單位**隨機切分 80 / 20（種子 42），得 160 / 40 個影像 |
| 評估結構 | RV, Myo, LV |
| 5 折（補做 OOF 用） | 以病人為單位分組，同一病人的 ED / ES 永遠在同一折，每折 128 / 32 個影像 |

nnU-Net 預設的 5 折會逐影像打散，同一病人的 ED / ES 可能跨折，造成資訊洩漏。因此 ACDC 的 5 折由 `data_conversion/make_acdc_grouped_splits.py` 產生，先以標註內容比對建立「影像 → 病人」對應，再以病人為單位切分。

### 2.3 其他資料集（保留供後續研究）

REFUGE、BUSI、CAMUS 的 Ψ 計算與篩選程式已包含在 `psi/`，目前僅 REFUGE 完成 Ψ 計算與 proxy 篩選。

---

## 3. nnU-Net 設定

### 3.1 規劃參數（2D，由 `ExperimentPlanner` 產生）

| 參數 | BTCV | ACDC |
|---|---|---|
| batch size | 4 | 56 |
| patch size | 256 × 256 | 256 × 224 |
| spacing | 1.0 × 1.0 mm | 1.5625 × 1.5625 mm |
| 影像中位尺寸 | 414 × 510.5 | 240 × 206 |
| 強度正規化 | CTNormalization | ZScoreNormalization |
| batch_dice | True | True |
| 網路 | PlainConvUNet | PlainConvUNet |
| 階層數 | 7 | 6 |
| 各階通道 | 32, 64, 128, 256, 512, 512, 512 | 32, 64, 128, 256, 512, 512 |
| 每階卷積 | 2 | 2 |
| 正規化層 / 激活 | InstanceNorm2d / LeakyReLU | InstanceNorm2d / LeakyReLU |

**與官方規劃器重算的比對**（`verification/plan_ensemble_report.sh`）：ACDC 所有 configuration 完全一致；BTCV 的 3D 設定完全一致，2D 的 batch size 與 patch size 不同（重算結果為 14 / 448 × 512），其餘欄位相同。anchor 與所有專家模型共用同一份 plans，模型間比較不受影響。

### 3.2 訓練

| 項目 | BTCV | ACDC |
|---|---|---|
| trainer（baseline） | `nnUNetTrainer` | `nnUNetTrainer_500epochs` |
| epochs | 1000（nnU-Net 預設） | 500（SAMA-UNet 設定） |
| 每 epoch 迭代數 | 250 | 250 |
| 訓練方式 | 5 折交叉驗證 | `all` 模式（全部訓練資料訓練單一模型） |
| 補做 | — | 5 折 OOF（僅 anchor，供 proxy 篩選） |
| optimizer / 排程 | nnU-Net 預設（SGD + poly） | 同左 |
| 資料增強 | nnU-Net 預設 | 同左 |
| 每 epoch 時間 | baseline 12.7 秒 / 專家 15.2 秒 | baseline 44 秒 / 專家約 100 秒 |

各資料集內部，anchor 與所有專家模型使用完全相同的 plans、訓練長度、切分與資料增強。

`all` 模式為 nnU-Net 官方支援的選項（見官方 `documentation/pretraining_and_finetuning.md`），但無法產生 out-of-fold 預測，因此 ACDC 另補做 5 折。

### 3.3 推論

| 項目 | 設定 |
|---|---|
| 程式 | 官方 `nnUNetv2_predict`（未修改） |
| BTCV | 5 折 ensemble：各折機率圖平均後取 argmax |
| ACDC | `all` 模式單一模型 |
| 滑動視窗步長 | 0.5 |
| 高斯加權 | 啟用 |
| 鏡像測試時增強 | 啟用 |
| checkpoint | `checkpoint_final.pth` |
| 後處理 | **不使用**（避免移除碎片而掩蓋 Ψ6 相關差異） |

---

## 4. loss 與加權

### 4.1 官方 loss（未修改）

由 `nnUNetTrainer._build_loss()` 建立：

```python
DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
               {}, weight_ce=1, weight_dice=1,
               dice_class=MemoryEfficientSoftDiceLoss)
```

外層以 `DeepSupervisionWrapper` 包裝：各解析度權重為 1/2^i，最低一層設為 0 後正規化為總和 1。

$$L_{\text{base}} = L_{CE} + L_{Dice}$$

### 4.2 本研究的唯一修改：`train_step`

官方（`nnUNetTrainer.py` 第 1037 行）：

```python
l = self.loss(output, target)
```

本研究（`nnUNetTrainer_PsiWeighted.py` 第 139–149 行）：

```python
if torch.allclose(w, torch.ones_like(w)):
    l = self.loss(output, target)            # batch 中無清單病例：同官方
else:
    total = 0.0
    for i in range(n):
        li = self.loss(self._sample_slice(output, i),
                       self._sample_slice(target, i))
        total = total + w[i] * li            # 清單病例 w = 3
    l = total / w.sum()                      # 分母為權重和
```

$$L = \frac{\sum_i w_i \cdot L_{\text{base+DS}}(x_i)}{\sum_i w_i}, \qquad w_i \in \{1, 3\}$$

- 乘的是每張切片的**完整** loss（CE + Dice + deep supervision）
- 以**病人**為單位：同一病人的所有切片權重相同
- 分母為權重和，只改變樣本間相對比重，不改變 loss 尺度
- optimizer、梯度裁剪（12）、混合精度等其餘部分與官方相同

### 4.3 副作用：逐樣本 Dice

拆開 batch 後，含清單病例的 batch 其 Dice 由「批次聚合」變為「逐樣本計算」，為加權之外的第二個變因。

| | BTCV | ACDC |
|---|---|---|
| batch size | 4 | 56 |
| 走逐樣本路徑的 batch 比例 | 約 63% | 幾乎 100% |

分離兩者需要「走相同路徑、權重約為 1」的對照組（倍率 1.0001，`nnUNetTrainer_*_ps1ctrl`）。先前的 `nnUNetTrainer_btcv_w1ctrl` 倍率為 1.0，使 `allclose` 恆成立而每個 batch 都走官方路徑，實際上等同 baseline 重訓，僅能作為重訓變異的參考。

---

## 5. Ψ proxy 與加權清單

### 5.1 十個 Ψ

| Ψ | 名稱 | 定義 |
|---|---|---|
| Ψ1 | 凸包比 | Area(ConvexHull) / Area，逐切片取最大輪廓，面積加權平均 |
| Ψ2 | 聲影一致性 | 需重新推論模型 |
| Ψ3 | 包含關係 | 結構間包含 |
| Ψ4 | 鄰接不重疊 | 相鄰結構不重疊 |
| Ψ5 | 器官互斥 | 對其他器官取最大重疊 |
| Ψ6 | 連通元件 | 3D 連通元件數 − 1 |
| Ψ7 | 拓撲環 | \|洞數 − 1\|，逐切片平均 |
| Ψ8 | 旋轉一致性 | 1 − Dice(原始預測, 旋轉→推論→轉回)，需重新推論 |
| Ψ9 | 巢狀關係 | 巢狀結構一致性 |
| Ψ10 | 中心偏移 | 結構質心距離，正規化 |

算法見 `psi/run_all_psi.py`。Ψ 只由預測遮罩計算，**不使用 GT**。BTCV 的 Ψ8 以**同一折**模型重新推論（`psi/run_psi8_btcv.py`），確保仍為 OOF。

### 5.2 病例層級聚合

- Dice：各結構取平均（nanmean）
- Ψ：各結構取 nanmax（避免單一結構的崩壞被其他正常結構稀釋）

### 5.3 proxy 篩選

1. 以 anchor 的**訓練集 OOF 預測**計算 Ψ
2. 低分組：病例 Dice 最低 30%（含並列）
3. 以 Mann-Whitney U 計算各 Ψ 區分低分組的 AUC 與 p 值
4. p < 0.05 者視為顯著；AUC < 0.5 表示方向相反（Ψ 越低越危險）

### 5.4 加權清單規則

| 資料集 | 專家 | 規則 | 例數 | 預測來源 |
|---|---|---|---|---|
| BTCV | Ψ1 | 前 30% | 5 | 5 折 OOF |
| BTCV | Ψ6 | 前 30%，並列全部納入 | 6 | 5 折 OOF |
| BTCV | Ψ8 | 前 30% | 5 | 5 折 OOF |
| ACDC | Ψ1 | 前 30% | 48 | `all` 模式訓練集預測 |
| ACDC | Ψ6 | Ψ6 > 0 | 46 | `all` 模式訓練集預測 |
| ACDC | Ψ7 | Ψ7 > 0 | 20 | `all` 模式訓練集預測 |
| ACDC | Ψ10 | **最低** 30%（反向） | 48 | 5 折 OOF |

Ψ6 / Ψ7 為離散值且大量為 0，取「前 30%」會使門檻塌陷為 ≥ 0 而全選，因此改取 > 0。

ACDC 的 Ψ1 / Ψ6 / Ψ7 清單是在補做 5 折之前，以 `all` 模式的訓練集預測產生的；該預測 Dice 接近飽和（0.9890），為本研究的已知限制。Ψ10 則以補做後的 OOF 篩選與產生。

清單可由 `psi/make_weight_lists.py` 重新產生並與倉庫中的版本比對。

---

## 6. 評估

- 指標：Dice（nnU-Net `evaluate_folder`）
- 病例分數：各結構 Dice 的 nanmean
- 與 baseline 比較時，任一方為 NaN 的「病例 × 結構」兩邊同時排除，確保分母一致（GT 無該結構且預測為空時 Dice 為 NaN）
- 統計：Wilcoxon signed-rank（配對）
- 另報告 baseline 最差 30% 病例的改善

---

## 7. 已知限制

1. ACDC 最初以 `all` 模式訓練，proxy 篩選基於飽和的訓練集預測；已補做 5 折 OOF，但 Ψ1 / Ψ6 / Ψ7 專家仍是依舊清單訓練。
2. 逐樣本 Dice 為第二個變因，對照組實驗（ps1ctrl）進行中。
3. BTCV 2D 的 batch / patch 與官方規劃器重算結果不同（見 3.1）。
4. 兩個資料集訓練長度不同（1000 / 500 epochs），各資料集內部一致。
5. ACDC 為自行隨機切分，SAMA-UNet 原文未公開切分，絕對數值不直接比較。
6. BTCV 18 / 12 小型測試集與 8 器官評估，nnU-Net Revisited 指出其統計雜訊偏高（折間標準差約 2.6 個百分點）。
