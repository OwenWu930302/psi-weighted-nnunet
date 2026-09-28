# Ψ-Weighted Expert Models on nnU-Net

以解剖結構約束（Ψ，structural proxy）找出分割模型容易出錯的病例，並在 nnU-Net 訓練時提高這些病例的 loss 權重，訓練「專家模型」。

- **不修改 nnU-Net 本體**：只新增一個繼承官方 `nnUNetTrainer` 的 trainer，覆寫 `train_step` 中「loss 套用到 batch」的方式。
- **loss 函數完全沿用官方**（CE + Dice + deep supervision）。
- **proxy 篩選只使用訓練集**（5 折 out-of-fold 預測），測試集不參與任何決策。

> 目前進度：BTCV 與 ACDC 的 baseline、專家模型皆已訓練並完成測試集評估。
> 分離「加權」與「逐樣本 loss」兩個變因的對照實驗規劃中。詳見 [docs/RESULTS.md](docs/RESULTS.md)。

---

## 使用的資料集

| 資料集 | 模態 | 結構 | 本倉庫狀態 |
|---|---|---|---|
| **BTCV**（Synapse 多器官） | 腹部 CT | 8 個器官 | 已完成訓練與評估 |
| **ACDC** | 心臟 MRI | RV / Myo / LV | 已完成訓練與評估 |
| REFUGE | 眼底彩照 | 視盤 OD / 視杯 OC | 已計算 Ψ 與 proxy 篩選，尚未訓練專家 |
| BUSI | 乳房超音波 | 腫塊 | Ψ 計算程式已就緒，預測遮罩待補 |
| CAMUS | 心臟超音波 | LV / Myo / LA | Ψ 計算程式已就緒，預測遮罩待補 |

`psi/run_all_psi.py` 與 `psi/recompute_caselevel.py` 包含全部五個資料集的載入與分析程式碼；REFUGE / BUSI / CAMUS 保留供後續研究使用。

**資料本身不包含在本倉庫中**，請向各資料集官方取得。

---

## 環境

| 項目 | 版本 |
|---|---|
| nnU-Net | v2.8.1（官方 MIC-DKFZ/nnUNet，commit `468cf80`） |
| PyTorch | 2.5.1 + CUDA 12 |
| Python | 3.10 |
| GPU | 單張 NVIDIA RTX 4080（16 GB） |

```bash
# 1. 安裝官方 nnU-Net v2.8.1
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet && git checkout v2.8.1 && pip install -e . && cd ..

# 2. 其他套件
pip install -r requirements.txt

# 3. 安裝加權 trainer（放入 nnU-Net 會自動掃描的資料夾）
cp nnunet_extension/nnUNetTrainer_PsiWeighted.py \
   nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/
```

程式中的預設路徑為 `~/桌面/論文`，請依自己的環境修改（`grep -rn "桌面/論文" .` 可列出所有位置）。

---

## 倉庫結構

```
nnunet_extension/   加權 trainer（唯一與 nnU-Net 互動的程式）
data_conversion/    BTCV / ACDC 轉 nnU-Net 格式、ACDC 依病人分組 5 折
psi/                十個 Ψ 的算法、計算、病例層級 proxy 篩選、加權清單產生
weight_lists/       各專家實際使用的加權清單（僅 case id 與分數）
splits/             資料切分紀錄（可重現）
configs/            nnU-Net 自動產生的 plans（規劃參數）
analysis/           結果整理（CSV）與作圖
results/            已整理好的結果 CSV（不含影像）
verification/       與官方 nnU-Net 一致性的驗證腳本與報告
patches/            對 nnU-Net 的唯一修改（ddp_allgather，單卡不執行）
docs/               方法、參數、結果、重現步驟
```

---

## 文件

| 文件 | 內容 |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | 方法、訓練參數、loss、加權實作、proxy 篩選、資料切分、推論設定 |
| [docs/RESULTS.md](docs/RESULTS.md) | 所有數據：測試集、OOF、proxy 篩選、5 折 vs `all`、加權病人分析 |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 從原始資料到結果的完整指令 |

---

## 結果摘要（測試集 Dice）

| 資料集 | baseline | 專家 | Δ |
|---|---|---|---|
| BTCV（n=12） | 0.8533 | Ψ1 0.8477 / Ψ6 0.8412 / Ψ8 0.8506 | −0.0056 / −0.0121 / −0.0027 |
| ACDC（n=40） | 0.9171 | Ψ1 0.9241 / Ψ6 0.9213 / Ψ7 0.9239 / Ψ10 0.9222 | +0.0070 / +0.0042 / +0.0068 / +0.0051 |

- BTCV baseline 與 SAMA-UNet 論文報告之 nnUNet（0.8493）相近，baseline 重現良好。
- ACDC 四個專家改善幅度相近，且以反向篩選的 Ψ10 亦有改善，提示改善可能主要來自逐樣本 loss 機制而非 proxy 的病例選擇，對照實驗規劃中。

---

## 參考文獻

- Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods 18, 203–211 (2021).
- Isensee F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* MICCAI (2024).
- Qamar S. et al. *UNet with Self-Adaptive Mamba-Like Attention and Causal-Resonance Learning for Medical Image Segmentation.* arXiv:2505.15234 (2025).（僅沿用其資料切分協定）
- Maier-Hein L. et al. *Metrics reloaded: recommendations for image analysis validation.* Nature Methods 21, 195–212 (2024).

## License

Apache-2.0（與 nnU-Net 相同）
