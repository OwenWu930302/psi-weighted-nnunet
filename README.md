# Ψ-Weighted Expert Models on nnU-Net

**以解剖結構約束（Ψ）找出分割模型容易出錯的病例，並在 nnU-Net 訓練時提高這些病例的 loss 權重，訓練「專家模型」。**

---

## 這個倉庫在做什麼

醫學影像分割模型的錯誤，常常表現為**不合解剖常理的形狀**：器官碎成好幾塊、心肌環出現缺口、輪廓凹凸不平。這些錯誤不需要標準答案就能從預測結果本身偵測出來。

本研究定義了十個結構約束 Ψ1–Ψ10（例如 Ψ6 = 連通元件數、Ψ7 = 環狀結構的洞數），用它們找出 baseline 模型最可能出錯的訓練病例，再讓模型在訓練時**多注意這些病例**：

```
① 訓練 baseline（官方 nnU-Net）
        │  5 折交叉驗證，取得每個訓練病例「沒被看過時」的預測（OOF）
        ▼
② 從預測遮罩計算 Ψ（不使用標準答案）
        │
        ▼
③ 篩選 proxy：哪個 Ψ 最能區分低分病例？（只用訓練集）
        │
        ▼
④ 取該 Ψ 最高的 30% 訓練病例，作為加權清單
        │
        ▼
⑤ 訓練專家模型：清單病例的 loss × 3，其餘不變
        │
        ▼
⑥ 在獨立測試集上比較 baseline 與專家
```

---

## 對 nnU-Net 做了什麼修改

**官方 nnU-Net 的訓練、loss、推論程式一行都沒改。** 本研究只新增一個繼承官方 `nnUNetTrainer` 的類別，並覆寫其中計算 loss 的那一步：

| | 官方 nnU-Net | 本研究 |
|---|---|---|
| loss 定義 | CE + Dice + deep supervision | **相同** |
| optimizer、學習率、資料增強、網路 | 自動配置 | **相同** |
| loss 如何套用到 batch | 整個 batch 算一次 | batch 含清單病例時，**逐張計算、清單病例 × 3、以權重和正規化** |

$$L = \frac{\sum_i w_i \cdot L_{\text{CE+Dice}}(x_i)}{\sum_i w_i}, \qquad w_i \in \{1,\ 3\}$$

完整說明見 [docs/CODE.md](docs/CODE.md)；與官方的逐檔比對見 `verification/`。

---

## 資料集

| 資料集 | 模態 | 結構 | 切分 | 狀態 |
|---|---|---|---|---|
| **BTCV** | 腹部 CT | 8 個器官 | 18 訓練 / 12 測試（TransUNet 公開切分） | 完成 |
| **ACDC** | 心臟 MRI | RV / Myo / LV | 80 / 20 位病人（160 / 40 影像） | 完成 |
| REFUGE | 眼底彩照 | 視盤 / 視杯 | — | 已計算 Ψ，尚未訓練專家 |
| BUSI、CAMUS | 超音波 | — | — | 程式已就緒，預測待補 |

資料本身不包含在倉庫中，請向各資料集官方取得（見 [docs/REPRODUCE.md](docs/REPRODUCE.md)）。

---

## 主要結果（測試集 Dice）

| 資料集 | baseline | 專家模型 |
|---|---|---|
| BTCV（n = 12） | **0.8533** | Ψ1 0.8477、Ψ6 0.8412、Ψ8 0.8506 |
| ACDC（n = 40） | **0.9171** | Ψ1 0.9241、Ψ6 0.9213、Ψ7 0.9239、Ψ10 0.9222 |

1. **baseline 重現了文獻數值**：BTCV baseline 0.8533，SAMA-UNet 論文報告的 nnUNet 為 0.8493。
2. **BTCV 上專家未優於 baseline**，差距小於 BTCV 的折間雜訊（約 2.6 個百分點，nnU-Net Revisited）。
3. **ACDC 上四個專家都進步，但幅度相近**（+0.004 至 +0.007），連反向篩選的 Ψ10 也進步。這提示改善可能主要來自「逐樣本計算 loss」本身，而非 Ψ 選到的病例。對照實驗（ps1ctrl：相同路徑、權重約為 1）用來分離這兩個效果。
4. **proxy 篩選必須使用 OOF 預測**：同一批 ACDC 訓練影像，用看過它們的模型預測時 Dice 為 0.9890（飽和），OOF 預測為 0.9162；篩選出的 proxy 排名也因此改變。

完整數據見 [docs/RESULTS.md](docs/RESULTS.md) 與 `results/`。

---

## 快速開始

```bash
export PSI_ROOT=$HOME/psi_workspace
git clone https://github.com/OwenWu930302/psi-weighted-nnunet.git $PSI_ROOT/psi-weighted-nnunet
cd $PSI_ROOT/psi-weighted-nnunet
source env.sh
```

之後依 **[docs/REPRODUCE.md](docs/REPRODUCE.md)** 的第 1–10 步執行：環境安裝、資料準備、前處理、訓練、推論、分析。每一步都附有檢查指令與預期結果。

所有程式的路徑都由環境變數 `PSI_ROOT` 決定，只需修改 `env.sh` 的一行。

---

## 倉庫結構

```
env.sh               路徑設定（每個終端機先 source）
nnunet_extension/    Ψ 加權 trainer（唯一與 nnU-Net 互動的程式）
data_conversion/     BTCV / ACDC 轉為 nnU-Net 格式、ACDC 依病人分組的 5 折
psi/                 十個 Ψ 的計算、Ψ8 重新推論、proxy 篩選、加權清單產生
weight_lists/        各專家實際使用的加權清單
splits/              資料切分紀錄（可重現）
configs/             nnU-Net plans 與 dataset.json（本研究實際使用）
analysis/            結果整理（CSV）與作圖
results/             原始實驗的結果 CSV、圖、報告（不含影像）
verification/        與官方 nnU-Net 一致性的驗證腳本與報告
patches/             原作者環境對官方 nnU-Net 的唯一修改（單卡不執行，重現不需要）
docs/                方法、程式說明、結果、重現步驟
```

| 文件 | 內容 |
|---|---|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 從零開始的完整重現步驟 |
| [docs/METHODS.md](docs/METHODS.md) | 方法、訓練參數、資料切分、推論設定、已知限制 |
| [docs/CODE.md](docs/CODE.md) | 每支程式、每個函式的說明，與 nnU-Net 的差異 |
| [docs/RESULTS.md](docs/RESULTS.md) | 所有數據與解讀 |

---

## 環境

nnU-Net v2.8.1（官方 MIC-DKFZ/nnUNet，commit `468cf80`）、PyTorch 2.5.1、Python 3.10、單張 RTX 4080。

---

## 參考文獻

- Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods 18, 203–211 (2021).
- Isensee F. et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* MICCAI (2024).
- Qamar S. et al. *UNet with Self-Adaptive Mamba-Like Attention and Causal-Resonance Learning for Medical Image Segmentation.* arXiv:2505.15234 (2025).（僅沿用資料切分協定）
- Chen J. et al. *TransUNet.*（BTCV 切分來源）
- Maier-Hein L. et al. *Metrics reloaded.* Nature Methods 21, 195–212 (2024).

## License

Apache-2.0，見 [LICENSE](LICENSE)。
