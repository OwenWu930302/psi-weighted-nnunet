# 程式碼說明：方法與對 nnU-Net 的修改

本文件逐一說明本研究的核心程式：每支程式的角色、每個函式的功能，以及與官方 nnU-Net v2.8.1 的關係。

**標記說明**

| 標記 | 意義 |
|---|---|
| `[新增]` | 官方 nnU-Net 沒有，本研究新增 |
| `[覆寫]` | 覆寫官方類別的同名方法 |
| `[修改]` | 覆寫的方法中，與官方不同的部分 |
| `[沿用]` | 與官方完全相同 |
| `[官方檔修改]` | 直接改動官方 nnU-Net 的原始檔 |

---

## 0. 總覽

### 0.1 與 nnU-Net 的關係

| 項目 | 狀態 |
|---|---|
| 官方 nnU-Net 原始檔 | **僅 1 個被修改**：`nnunetv2/utilities/ddp_allgather.py`（單卡訓練時不執行） |
| 訓練、loss、推論核心 | 與官方 v2.8.1 逐位元組相同（git hash 比對） |
| 本研究新增的 nnU-Net 程式 | `nnUNetTrainer_PsiWeighted.py`（繼承官方 `nnUNetTrainer`） |
| 其餘程式 | 在 nnU-Net 之外執行，只讀取 nnU-Net 的輸出 |

### 0.2 流程與對應程式

```
convert_btcv_to_nnunet.py ─┐
convert_acdc_to_nnunet.py ─┴→ nnU-Net 規劃、前處理、訓練 baseline（官方）
                                   │  OOF 預測（fold_*/validation/）
                                   ▼
               run_psi8_btcv.py ─┐
               run_all_psi.py ───┴→ recompute_caselevel.py → 加權清單 wcases_*.csv
                                                                  │  環境變數
                                                                  ▼
                                          nnUNetTrainer_PsiWeighted.py（專家訓練）
```

---

## 1. `nnUNetTrainer_PsiWeighted.py` —— 本研究唯一與 nnU-Net 互動的程式

**位置**：`nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/`
nnU-Net 會自動掃描此資料夾中的類別，因此以 `-tr <類別名>` 即可呼叫。

### 1.1 模組層級函式

| 函式 | 類型 | 功能 | 輸入 → 輸出 |
|---|---|---|---|
| `_load_case_list(path)` | [新增] | 讀取加權清單 | csv（取 `case` 欄，無則取第一欄）或 json 陣列 → 病例 id 的 `set`；檔案不存在回傳空 set |
| `_normalise(key)` | [新增] | 將 dataloader 的 key 轉為病例 id | 去除 `.b2nd` / `.npz` / `.npy` / `.pkl` 副檔名與 `_seg` 後綴；**刻意不剝除數字後綴**，因病例 id 本身以數字結尾（`ACDC_001`） |

以 `set` 做完全比對，`BTCV_001` 不會誤中 `BTCV_010`。

### 1.2 `class nnUNetTrainer_PsiWeighted(nnUNetTrainer)` [新增]

**類別屬性**

| 屬性 | 值 | 說明 |
|---|---|---|
| `CASE_LIST_ENV` | `'PSI_WEIGHT_CASES'` | 清單路徑的環境變數名稱 |
| `FACTOR_ENV` | `'PSI_WEIGHT_FACTOR'` | 倍率的環境變數名稱 |
| `DEFAULT_FACTOR` | `3.0` | 未指定倍率時的預設值 |
| `NUM_EPOCHS` | `500` | 訓練長度，子類別可覆寫 |

**方法**

| 方法 | 類型 | 與官方的差異 |
|---|---|---|
| `__init__` | [覆寫] | [沿用] 第一行呼叫官方 `__init__`，所有官方設定照常建立。[新增] 其後設定 `num_epochs`、讀取清單與倍率、初始化統計計數器 `_psi_hit` / `_psi_total` |
| `on_train_start` | [覆寫] | [沿用] 先呼叫官方。[新增] 將清單路徑、例數、倍率寫入 `training_log`；清單為空時另加警告 |
| `_sample_slice(x, i)` | [新增] | 從 output / target 取出第 i 個樣本。deep supervision 時兩者是 list（每個解析度一層），**每一層都要切**；用 `i:i+1` 保留 batch 維度，官方 loss 才能直接使用 |
| `_weights_for(keys, n)` | [新增] | 每個樣本的權重：病例在清單中為倍率，否則為 1。2D 模式下 key 是病例 id，同一病例所有切片權重相同。同時累計加權樣本數 |
| `train_step(batch)` | [覆寫] | **唯一改變訓練行為的地方**，見 1.3 |
| `on_train_epoch_end` | [覆寫] | [沿用] 先呼叫官方。[新增] 將本輪加權比例寫入 log 並歸零計數器 |

**未覆寫（完全沿用官方）**：`_build_loss`（CE + Dice + deep supervision）、optimizer 與學習率排程、資料增強、網路架構、`validation_step`、推論、checkpoint 儲存。

### 1.3 `train_step` 與官方的逐段對照

| 段落 | 官方 `nnUNetTrainer.train_step` | 本研究 | 標記 |
|---|---|---|---|
| 讀取資料 | `data`、`target` | 另讀 `keys = batch.get('keys')` | [新增] |
| 搬到 GPU | 相同 | 相同 | [沿用] |
| 計算權重 | — | `w = self._weights_for(keys, n)` | [新增] |
| 前向傳播 | `output = self.network(data)` | 相同 | [沿用] |
| **loss** | `l = self.loss(output, target)`（第 1037 行） | 見下方 | **[修改]** |
| 反向傳播、混合精度、梯度裁剪（12）、optimizer | — | 相同 | [沿用] |

**loss 的修改**

```python
if torch.allclose(w, torch.ones_like(w)):
    l = self.loss(output, target)                 # 路徑 A：同官方
else:
    total = 0.0
    for i in range(n):
        li = self.loss(self._sample_slice(output, i),
                       self._sample_slice(target, i))
        total = total + w[i] * li
    l = total / w.sum()                           # 路徑 B：逐樣本加權
```

$$L = \frac{\sum_i w_i \cdot L_{CE+Dice+DS}(x_i)}{\sum_i w_i}$$

| 設計 | 原因 |
|---|---|
| 乘在**完整** loss 上 | `self.loss` 已含 CE、Dice 與 deep supervision，逐樣本呼叫即可，不需改動官方 loss |
| 分母為 `w.sum()` 而非 `n` | 只改變樣本間相對比重，loss 尺度不變；若用 `n`，等同暗中提高學習率 |
| 路徑 A 的捷徑 | batch 中無清單病例時與官方完全相同，且較快 |

**副作用**：路徑 B 中，Dice 由「整個 batch 聚合」變為「逐樣本計算」，是加權之外的第二個變因。

| | BTCV（batch 4） | ACDC（batch 56） |
|---|---|---|
| 走路徑 B 的 batch 比例 | 約 63% | 幾乎 100% |

**倍率 1.0 的陷阱**：倍率為 1.0 時 `w` 恆為 1，永遠走路徑 A，等同 baseline 重訓。因此要分離逐樣本 Dice 的效果，對照組必須用 **1.0001**。

### 1.4 子類別 [新增]

子類別只設定三項：清單路徑、倍率、訓練長度。清單資料夾 `_TF = $PSI_ROOT/tree_features`（未設定 `PSI_ROOT` 時為 `~/桌面/論文/tree_features`）。

| 類別 | 清單 | 倍率 | epochs | 設定方式 |
|---|---|---|---|---|
| `nnUNetTrainer_500epochs_psi1w3` | `wcases_psi1_convex.csv`（48） | 3.0（預設） | 500（繼承） | `setdefault` |
| `nnUNetTrainer_500epochs_psi6w3` | `wcases_psi6_cc.csv`（46） | 3.0（預設） | 500（繼承） | `setdefault` |
| `nnUNetTrainer_500epochs_psi7w3` | `wcases_psi7_genus.csv`（20） | 3.0（預設） | 500（繼承） | `setdefault` |
| `nnUNetTrainer_500epochs_psi10w3` | `wcases_acdc_psi10_centroid.csv`（48，反向） | 3.0 | 500 | 直接賦值 |
| `nnUNetTrainer_500epochs_ps1ctrl` | `wcases_psi1_convex.csv`（48） | **1.0001** | 500 | 直接賦值 |
| `nnUNetTrainer_btcv_psi1w3` | `wcases_btcv_psi1_convex.csv`（5） | 3.0 | 1000 | 直接賦值 |
| `nnUNetTrainer_btcv_psi6w3` | `wcases_btcv_psi6_cc.csv`（6） | 3.0 | 1000 | 直接賦值 |
| `nnUNetTrainer_btcv_psi8w3` | `wcases_btcv_psi8_rotation.csv`（5） | 3.0 | 1000 | 直接賦值 |
| `nnUNetTrainer_btcv_w1ctrl` | 同 psi8 | 1.0 | 1000 | 直接賦值 |

- `ps1ctrl`：真正的對照組，走與 Ψ1 專家相同的逐樣本路徑，但權重約為 1
- `w1ctrl`：倍率 1.0，因 1.3 所述陷阱而**非有效對照組**，僅代表 baseline 重訓的雜訊

**訓練長度的等價性**：ACDC baseline 使用官方 `nnUNetTrainer_500epochs`，專家繼承 `nnUNetTrainer` 並以 `num_epochs = 500` 覆寫，兩者訓練長度相同。

---

## 2. `ddp_allgather.py` —— 唯一被修改的官方檔案 [官方檔修改]

**位置**：`nnunetv2/utilities/ddp_allgather.py`

**修改內容**：`AllGatherGrad` 的 `forward` 與 `backward` 各加一段「未初始化分散式訓練時直接回傳」的判斷。

| 方法 | 新增的判斷 |
|---|---|
| `forward` | 未啟動分散式時，回傳 `tensor.unsqueeze(0)`，模擬單一裝置的 `stack(dim=0)` |
| `backward` | 對應地去除該維度，回傳 `grad_output[0]` |

**對本研究的影響：無。** 此函式只在官方 Dice loss 的 `ddp=True` 分支被呼叫；單卡訓練時 `is_ddp = False`，該分支不會執行。此修改推測源於先前嘗試 SAMA-UNet 整合時的相容性處理。

---

## 3. `run_all_psi.py` —— 十個 Ψ 的計算

**角色**：讀取各資料集的預測遮罩，逐「病例 × 結構」計算 Ψ1–Ψ10，輸出 `all_psi_<dataset>.csv`。Ψ 只由預測遮罩計算，**不使用 GT**（Dice 僅作為標記記錄，不參與 Ψ 計算）。

### 3.1 Ψ 函式

| 函式 | 定義 | 實作細節 |
|---|---|---|
| `psi1_convex(mask)` | 凸包面積 / 實際面積 | 3D 沿第 3 軸逐切片，取**最大外輪廓**；切片與輪廓面積 < 10 px 略過；以面積加權平均。空遮罩回傳 NaN |
| `psi2_shadow()` | 聲影一致性 | 需重新推論，固定回傳 NaN |
| `psi3_inclusion(inner, outer)` | 內層落在外層之外的比例 | (inner ∧ ¬outer) / inner |
| `psi4_adjacency(a, b)` | 兩結構重疊 | 交集 / 聯集（IoU） |
| `psi5_exclusion(target, others)` | 與其他結構的最大重疊 | 對每個其他結構取 IoU 的最大值 |
| `psi6_cc(mask)` | 連通元件數 − 1 | 對**整個 3D volume** 計算；空遮罩回傳 −1（比碎裂更嚴重） |
| `psi7_genus(ring)` | \|內部洞數 − 1\| | 逐切片以 `binary_fill_holes` 求洞，切片 < 20 px 略過，算術平均 |
| `psi8_rotation()` | 旋轉一致性 | 需重新推論，固定回傳 NaN；由 `run_psi8_btcv.py` 另行計算 |
| `psi9_nesting(inner, outer)` | 巢狀關係 | **直接呼叫 `psi3_inclusion`**，同式 |
| `psi10_centroid(target, ref, diag)` | 質心距離 / 影像對角線 | 以體素座標計算質心，除以影像對角線長度 |

### 3.2 資料載入函式

| 函式 | 資料 | 預測來源 |
|---|---|---|
| `load_refuge()` | REFUGE | `FunduSegmenter/repro_notes/segmap_designed/` |
| `load_busi()` | BUSI | `nnUNet/HA-Net/results/masks/` |
| `load_nnunet_3d(...)` | 通用：讀 nnU-Net 預測與 `summary.json` | 由下列函式指定 |
| `load_btcv()` | BTCV | anchor 5 折 OOF（`fold_*/validation/`） |
| `load_acdc()` | ACDC | **測試集**預測（`acdc_test_pred/`），僅供事後分析 |
| `load_acdc_oof()` | ACDC | anchor 5 折 OOF（`fold_[0-4]/`，排除 `fold_all`） |
| `load_camus()` | CAMUS | 預測遮罩未保存，回傳 None |

各資料集的結構設定：

| 資料集 | 結構 | Ψ3 / Ψ9 巢狀 | Ψ7 環狀 | Ψ10 參考 |
|---|---|---|---|---|
| BTCV | 8 個器官 | 無 | 無 | Liver |
| ACDC | RV / Myo / LV | (LV, Myo) | Myo | LV |
| REFUGE | OD / OC | (OC, OD) | 視盤環 | OD |

`load_acdc_oof` 的 glob 使用 `fold_[0-4]` 而非 `fold_*`，是為了排除 `fold_all`（模型看過全部訓練資料，Dice 飽和）。

### 3.3 統計與輸出

| 函式 | 功能 |
|---|---|
| `mutual_info(z, zh)` | 失敗標記與預測標記的互資訊 |
| `f1_of(z, zh)` | F1 |
| `best_threshold(v, z)` | 以 1–99 百分位為候選門檻，雙向（> / <）搜尋互資訊最大者 |
| `analyse(name, rows)` | 輸出 `all_psi_*.csv`；以 Dice < 0.80 為失敗，做樹狀級聯分析，輸出 `cascade_*.csv` |
| `main()` | 依序執行各資料集的載入與分析 |

`analyse` 的級聯分析使用絕對門檻 δ = 0.80，屬探索性輸出；**實際 proxy 篩選由 `recompute_caselevel.py` 以相對門檻進行**。

---

## 4. `run_psi8_btcv.py` —— Ψ8 旋轉一致性

**定義**

$$\Psi_8 = 1 - \text{Dice}\big(\hat{y},\ R_{-\varphi}(f(R_{\varphi}(x)))\big), \quad \varphi = 15°$$

**流程（逐折進行）**

1. 讀取 fold k 的 OOF 預測 ŷ
2. 將對應影像在平面內旋轉 15°（`order=1`，邊角以最小值填補）
3. **以同一折 k 的模型**重新推論（`nnUNetv2_predict -f k`）
4. 將預測轉回 −15°（`order=0`，保持標籤為整數）
5. 逐器官計算 1 − Dice

**為什麼必須逐折**：若用看過該病例的其他折模型推論，模型對它的預測過於穩定，Ψ8 會被系統性低估，也就不再是 OOF。

| 函式 / 常數 | 說明 |
|---|---|
| `dice(a, b)` | 兩者皆空時回傳 1.0 |
| `ANGLE = 15` | 旋轉角度 |
| 輸出 | `tree_features/psi8_btcv.csv`（case、structure、fold、psi8_rotation，共 144 筆） |

---

## 5. `recompute_caselevel.py` —— proxy 篩選

**角色**：將結構層級的 Ψ 聚合到病例層級，以相對低分組計算每個 Ψ 的區分能力。

### 5.1 三項方法學決定（程式開頭已說明）

| 決定 | 內容 | 原因 |
|---|---|---|
| 分析單位為病例 | 同一病例的結構合併 | 以結構為單位時，REFUGE 失敗幾乎全是 OC，任何能區分 OD / OC 的量都會被誤判為失敗偵測器 |
| 相對門檻 | 最低 20% / 30% | δ = 0.80 使部分資料集幾乎沒有低分樣本，無法檢定 |
| Ψ 取 nanmax | 病例風險 = 各結構中最高者 | 取平均會稀釋單一結構的崩壞（例：BTCV 膽囊失敗被其餘 7 個正常器官平均掉） |

### 5.2 函式

| 函式 | 功能 |
|---|---|
| `load(name)` | 讀 `all_psi_<name>.csv`，合併另算的 Ψ8（BTCV：`psi8_btcv.csv`；ACDC：`psi8_acdc.csv`，僅對應測試集）。`acdc_oof` 不合併任何 Ψ8，故為全 NaN |
| `to_case(d)` | 病例層級聚合：Dice 取平均（pandas 自動略過 NaN）、Ψ 取 nanmax；另計 `n_nan`（只計「該資料集有部分有效值」的欄位，排除結構性缺失） |
| `best_f1(v, y)` | 雙向搜尋 F1 最大的門檻 |
| `cv_f1(v, y)` | 5 折分層交叉驗證的 F1（門檻在訓練折決定、在驗證折評估），低分組 < 5 例時不計算 |
| `report(...)` | 每個 Ψ 的 AUC（`roc_auc_score`）、Mann-Whitney U p 值、低分組與其餘平均、F1、CV F1、Spearman ρ；全距 < 1e-6 視為恆定 |
| `main()` | 各資料集依序輸出 `caselevel_*.csv`，並以 δ = 0.80、最低 20%、最低 30% 三種口徑報告 |

**低分組定義**：`dice <= cd.dice.quantile(q)`（含並列）。`quantile` 使用線性插值，BTCV 18 例取 30% 時實際為 6 例（33%）。

**AUC 方向**：AUC > 0.5 表示 Ψ 越高越可能在低分組；AUC < 0.5 表示反向（Ψ10 在 BTCV 與 ACDC 皆為此情況）。

---

## 6. 資料轉換

### 6.1 `convert_btcv_to_nnunet.py`

| 項目 | 內容 |
|---|---|
| 來源 | `$BTCV_NIFTI`（已依 TransUNet 清單分為 imagesTr / imagesTs） |
| 編號 | 依檔名排序：訓練 18 例 → `BTCV_001`–`018`；測試 12 例 → `BTCV_019`–`030` |
| 測試標註 | 一併複製到 `labelsTs/`。原始實驗中是事後以影像內容逐像素比對後複製，比對結果與依檔名排序完全一致（`splits/btcv_test_label_mapping.csv`） |
| dataset.json | 由此腳本寫出，與 `configs/btcv_dataset.json` 相同 |

### 6.2 `convert_acdc_to_nnunet.py`

| 函式 | 功能 |
|---|---|
| `find_frame_pairs(patient_dir)` | 找出每位病人有標註的時相（ED / ES），排除無標註的 4D 檔 |
| `main()` | 以**病人為單位**隨機切分 80 / 20（`SEED = 42`），再展開時相；訓練 `ACDC_001`–`160`、測試 `ACDC_161`–`200`；輸出 `dataset.json` 與 `patient_split.json` |

先切病人、再展開時相，確保同一病人的 ED / ES 不會分屬訓練與測試集。

---

## 7. 程式審查中發現的事項

以下是整理文件時從程式碼本身發現的問題，建議在論文或後續版本中處理。

### 7.1 Ψ3 / Ψ9、Ψ4 / Ψ5 在 nnU-Net 資料集上結構性恆定

nnU-Net 的輸出是**互斥**的標籤圖（每個像素只有一個標籤），`load_nnunet_3d` 以 `a == k` 取出各結構，因此任兩個結構的遮罩**永遠不重疊**：

| Ψ | 算式 | 在互斥標籤上的結果 |
|---|---|---|
| Ψ4、Ψ5 | 結構間的 IoU | **恆為 0** |
| Ψ3（ACDC：LV 在 Myo 內） | (LV ∧ ¬Myo) / LV | LV 與 Myo 不重疊 → **恆為 1** |
| Ψ9 | 直接呼叫 Ψ3 | 與 Ψ3 **完全相同** |

這解釋了結果表中 Ψ3 / Ψ4 / Ψ5 / Ψ9「恆定，無變異」的原因：**並非模型從不犯錯，而是算式在此資料格式下無法產生變異**。此外，在 `load_nnunet_3d` 中 Ψ4 取「與其他結構 IoU 的最大值」，與 Ψ5 的算法等價，兩者本就重複。

若要讓包含關係有意義，外層應改用「填滿後」的區域（例如 ACDC 以 `binary_fill_holes(Myo ∪ LV)` 作為外層），而非 Myo 本身。

### 7.2 Ψ10 以體素座標計算

`psi10_centroid` 以 `np.argwhere` 的體素索引計算質心，並除以影像對角線（體素數）。BTCV 與 ACDC 的切片間距遠大於平面解析度，**z 方向的距離被相對低估**，且不同病例的影像尺寸不同，正規化尺度也不一致。這可能是 Ψ10 呈現反向的原因之一。改以 mm 為單位（乘上 spacing）會更一致。

### 7.3 ACDC 舊專家使用 `setdefault` 設定清單

`psi1w3`、`psi6w3`、`psi7w3` 以 `os.environ.setdefault` 設定清單、倍率沿用預設 3.0。若同一個終端機先前設過 `PSI_WEIGHT_CASES` 或 `PSI_WEIGHT_FACTOR`，會沿用舊值而非預期的清單。BTCV 與後來新增的類別已改用直接賦值。實際使用的清單與倍率以各次訓練 log 的 `[Ψ加權] 清單 ...` 為準。

### 7.4 路徑設定（已處理）

原本所有程式皆寫死 `~/桌面/論文`。現已統一改為讀取環境變數 `PSI_ROOT`（由 `env.sh` 設定），未設定時仍預設為原作者路徑，因此原環境的行為不變。

### 7.5 ACDC 的 Ψ8 只有測試集

`psi8_acdc.csv` 由另一支腳本以測試集產生，`recompute_caselevel.py` 只在 `acdc`（測試集）合併它。ACDC 的 OOF 分析因此沒有 Ψ8，無法用於篩選。
