# 完整重現步驟

本文件從零開始，逐步重現本研究的所有結果。每一步都附有**檢查指令**與**預期結果**，請確認通過後再進行下一步。

> **重要**：所有指令都在「專案工作區」執行，工作區由環境變數 `PSI_ROOT` 指定。
> 每開一個新終端機，都要先 `conda activate` 並 `source env.sh`。

---

## 0. 硬體與時間

| 項目 | 本研究環境 |
|---|---|
| GPU | 單張 NVIDIA RTX 4080（16 GB） |
| 作業系統 | Ubuntu（Linux） |
| 硬碟 | 約 100 GB（前處理資料、模型、預測） |

| 訓練項目 | 每 epoch | 總時間（約） |
|---|---|---|
| BTCV baseline，5 折 × 1000 epochs | 12.7 秒 | 18–20 小時 |
| BTCV 專家，每個 5 折 × 1000 epochs | 15.2 秒 | 22–26 小時 |
| ACDC baseline，`all` × 500 epochs | 44 秒 | 6–7 小時 |
| ACDC baseline 5 折 OOF × 500 epochs | 44 秒 | 31 小時 |
| ACDC 專家 / 對照組，每個 `all` × 500 epochs | 約 100 秒 | 14 小時 |

長時間訓練一律用 `nohup ... &` 在背景執行，關閉終端機不會中斷。

---

## 1. 建立工作區與環境

```bash
# 1-1 指定工作區（改成你要的位置）
export PSI_ROOT=$HOME/psi_workspace
mkdir -p $PSI_ROOT && cd $PSI_ROOT

# 1-2 下載本倉庫
git clone https://github.com/OwenWu930302/psi-weighted-nnunet.git
cd $PSI_ROOT/psi-weighted-nnunet

# 1-3 Python 環境
conda create -n expertree python=3.10 -y
conda activate expertree
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # 依 CUDA 版本選擇
pip install -r requirements.txt

# 1-4 安裝官方 nnU-Net v2.8.1（放在工作區內）
git clone https://github.com/MIC-DKFZ/nnUNet.git $PSI_ROOT/nnUNet
cd $PSI_ROOT/nnUNet && git checkout v2.8.1 && pip install -e . && cd -

# 1-5 放入 Ψ 加權 trainer（nnU-Net 會自動掃描此資料夾）
cp nnunet_extension/nnUNetTrainer_PsiWeighted.py \
   $PSI_ROOT/nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/

# 1-6 載入路徑設定
source env.sh
```

**檢查**

```bash
cd ~        # 不要在 nnUNet 原始碼資料夾內執行 Python
python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))"
python3 -c "from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_PsiWeighted \
import nnUNetTrainer_btcv_psi8w3 as T; print('trainer OK', T.NUM_EPOCHS)"
cd $PSI_ROOT/nnUNet && git describe --tags && cd -
```

預期：路徑為 `$PSI_ROOT/nnUNet/nnunetv2`、印出 `trainer OK 1000`、版本 `v2.8.1`。

`patches/ddp_allgather.diff` 是原作者環境中對官方檔案的修改，**單卡訓練不會執行到，重現時不需套用**。

---

## 2. 準備資料

### 2-1 ACDC

1. 至 ACDC 官方網站下載 training 資料（100 位病人）：
   https://humanheart-project.creatis.insa-lyon.fr/database/
2. 解壓縮後，使 `$ACDC_RAW` 底下有 `patient001/` … `patient100/`：

```bash
ls $ACDC_RAW | head -3        # 應看到 patient001 patient002 patient003
ls $ACDC_RAW | wc -l          # 應為 100（或另含 MANDATORY_CITATION 等說明檔）
```

若放在其他位置，執行 `export ACDC_RAW=<實際路徑>` 後再繼續。

3. 轉換（以病人為單位 80 / 20，種子 42）：

```bash
cd $PSI_ROOT/psi-weighted-nnunet
python3 data_conversion/convert_acdc_to_nnunet.py
```

**檢查**：切分必須與原始實驗相同。

```bash
python3 -c "
import json
a = json.load(open('$nnUNet_raw/Dataset116_ACDC/patient_split.json'))
b = json.load(open('splits/acdc_patient_split.json'))
print('ACDC 切分一致' if a == b else '✗ 切分不一致，停止')"
```

### 2-2 BTCV

> ⚠ **待補**：本研究使用的 `BTCV_nifti` 是由 TransUNet 公開的 Synapse 前處理資料轉換而來
> （強度已截斷並縮放到 0–1，體素間距為 1 mm）。其下載與轉換腳本將補入 `data_conversion/`。
> 在此之前，請確認 `$BTCV_NIFTI` 具有下列結構：

```
$BTCV_NIFTI/
  imagesTr/case0005.nii.gz … （18 例）
  labelsTr/case0005.nii.gz … （18 例，標籤 0–8）
  imagesTs/case0001.nii.gz … （12 例）
  labelsTs/case0001.nii.gz … （12 例，標籤 0–8）
```

轉換：

```bash
python3 data_conversion/convert_btcv_to_nnunet.py
```

**檢查**：

```bash
ls $nnUNet_raw/Dataset115_BTCV/imagesTs | sed 's/_0000.nii.gz//' | diff - splits/btcv_test_cases.txt && echo "BTCV 測試集一致"
ls $nnUNet_raw/Dataset115_BTCV/labelsTs | wc -l      # 應為 12
```

---

## 3. 前處理（使用本研究的 plans）

nnU-Net 自動規劃的 BTCV 2D 參數（batch 14 / patch 448×512）與本研究使用的（batch 4 / patch 256×256）不同。**為了完全重現，必須使用倉庫中的 plans**：

```bash
nnUNetv2_extract_fingerprint -d 115 116 --verify_dataset_integrity
nnUNetv2_plan_experiment -d 115 116

# 以本研究的 plans 覆蓋
cp configs/btcv_nnUNetPlans.json $nnUNet_preprocessed/Dataset115_BTCV/nnUNetPlans.json
cp configs/acdc_nnUNetPlans.json $nnUNet_preprocessed/Dataset116_ACDC/nnUNetPlans.json

nnUNetv2_preprocess -d 115 116 -c 2d

# 5 折切分（BTCV 為 nnU-Net 預設切分；ACDC 為依病人分組的切分）
cp splits/btcv_splits_final.json $nnUNet_preprocessed/Dataset115_BTCV/splits_final.json
cp splits/acdc_splits_final.json $nnUNet_preprocessed/Dataset116_ACDC/splits_final.json
```

**檢查**：

```bash
python3 -c "
import json
for d in ['Dataset115_BTCV', 'Dataset116_ACDC']:
    c = json.load(open('$nnUNet_preprocessed/' + d + '/nnUNetPlans.json'))['configurations']['2d']
    print(d, 'batch', c['batch_size'], 'patch', c['patch_size'])"
```

預期：`Dataset115_BTCV batch 4 patch [256, 256]`、`Dataset116_ACDC batch 56 patch [256, 224]`。

ACDC 的 5 折切分也可由 `data_conversion/make_acdc_grouped_splits.py` 重新產生（以標註內容比對病人，再依病人分組），結果與 `splits/acdc_splits_final.json` 相同。

---

## 4. 放入加權清單

專家模型的加權清單直接使用倉庫中的版本：

```bash
cp weight_lists/*.csv $PSI_ROOT/tree_features/
ls $PSI_ROOT/tree_features/wcases_*.csv | wc -l      # 應為 7
```

清單的產生方式見 `docs/METHODS.md` 第 5 節；也可在第 6 步後以 `psi/make_weight_lists.py` 重新產生並比對。

---

## 5. 訓練 baseline

```bash
cd ~ && source $PSI_ROOT/psi-weighted-nnunet/env.sh

# BTCV：5 折，1000 epochs
nohup bash -c 'for f in 0 1 2 3 4; do nnUNetv2_train 115 2d $f --npz; done' \
  > $PSI_ROOT/train_btcv_baseline.log 2>&1 &

# ACDC：all 模式，500 epochs（上一個結束後再執行，或另開 GPU）
nohup bash -c 'nnUNetv2_train 116 2d all -tr nnUNetTrainer_500epochs --npz' \
  > $PSI_ROOT/train_acdc_baseline.log 2>&1 &

# ACDC：5 折 OOF，供 proxy 篩選（500 epochs）
nohup bash -c 'for f in 0 1 2 3 4; do nnUNetv2_train 116 2d $f -tr nnUNetTrainer_500epochs --npz; done' \
  > $PSI_ROOT/train_acdc_oof.log 2>&1 &
```

GPU 只有一張時，請一個跑完再啟動下一個。

**檢查**（ACDC 5 折必須讀到分組切分）：

```bash
grep -h "split file\|This split has" $nnUNet_results/Dataset116_ACDC/nnUNetTrainer_500epochs__nnUNetPlans__2d/fold_0/training_log_*.txt
```

預期：`Using splits from existing split file` 與 `This split has 128 training and 32 validation cases`。

---

## 6. Ψ 計算與 proxy 篩選（驗證用，可選）

加權清單已在第 4 步提供；此步驟用來重現篩選過程。

```bash
cd $PSI_ROOT/psi-weighted-nnunet
python3 psi/run_psi8_btcv.py          # BTCV Ψ8：逐折旋轉 15° 重新推論（約 1–2 小時）
python3 psi/run_all_psi.py            # 所有資料集的 Ψ；缺少的資料集會自動略過
python3 psi/recompute_caselevel.py    # 病例層級聚合、AUC、p 值
python3 psi/make_weight_lists.py      # 依規則重新產生清單，與第 4 步的清單比對
```

**預期**（`recompute_caselevel.py`，最低 30% 口徑）：

- BTCV：Ψ1 AUC 0.833、Ψ6 0.861、Ψ8 0.903，皆 p < 0.05
- ACDC_OOF：Ψ7 AUC 0.733、Ψ6 0.647、Ψ10 0.295（反向）顯著，Ψ1 0.593 邊緣

`make_weight_lists.py` 預期 BTCV 三份與 ACDC Ψ10 顯示 ✓；ACDC Ψ1 / Ψ6 / Ψ7 顯示「未驗證」（其來源為 `all` 模式訓練集預測，Ψ 表未保存）。

---

## 7. 訓練專家模型與對照組

```bash
cd ~ && source $PSI_ROOT/psi-weighted-nnunet/env.sh

# BTCV 專家（每個 5 折 × 1000 epochs，依序執行）
nohup bash -c 'for t in psi1w3 psi6w3 psi8w3; do
  for f in 0 1 2 3 4; do nnUNetv2_train 115 2d $f -tr nnUNetTrainer_btcv_$t --npz; done; done' \
  > $PSI_ROOT/train_btcv_experts.log 2>&1 &

# ACDC 專家與對照組（每個 all × 500 epochs）
nohup bash -c 'for t in psi1w3 psi6w3 psi7w3 psi10w3 ps1ctrl; do
  nnUNetv2_train 116 2d all -tr nnUNetTrainer_500epochs_$t --npz; done' \
  > $PSI_ROOT/train_acdc_experts.log 2>&1 &
```

**檢查**（每個模型啟動後都要確認加權生效）：

```bash
grep -h "Ψ加權" $nnUNet_results/Dataset116_ACDC/nnUNetTrainer_500epochs_psi10w3__nnUNetPlans__2d/fold_all/training_log_*.txt | head -2
```

| 模型 | 清單 | case 數 | 倍率 | 每輪加權比例 |
|---|---|---|---|---|
| BTCV psi1w3 / psi6w3 / psi8w3 | btcv 清單 | 5 / 6 / 5 | 3.0 | 約 20–35%（依折） |
| ACDC psi1w3 / psi6w3 / psi7w3 | 舊清單 | 48 / 46 / 20 | 3.0 | 約 30% / 29% / 12% |
| ACDC psi10w3 | Ψ10 清單 | 48 | 3.0 | 約 30% |
| ACDC ps1ctrl | Ψ1 清單 | 48 | **1.0001** | 約 30% |

`case 數 0` 或比例 0% 代表清單沒有讀到，**立即停止**並檢查第 4 步。

ps1ctrl 的每 epoch 時間必須約 100 秒（與專家相同）；若約 44 秒，代表未走逐樣本路徑。

---

## 8. 測試集推論與評估

輸出資料夾名稱會被分析程式讀取，**請勿更改**。

```bash
cd ~ && source $PSI_ROOT/psi-weighted-nnunet/env.sh

# BTCV：5 折 ensemble
for t in nnUNetTrainer nnUNetTrainer_btcv_psi1w3 nnUNetTrainer_btcv_psi6w3 nnUNetTrainer_btcv_psi8w3; do
  nnUNetv2_predict -i $nnUNet_raw/Dataset115_BTCV/imagesTs -o $PSI_ROOT/btcv_test_pred/$t \
    -d 115 -c 2d -tr $t -f 0 1 2 3 4
done

# ACDC：單一模型（all）
nnUNetv2_predict -i $nnUNet_raw/Dataset116_ACDC/imagesTs -o $PSI_ROOT/nnUNet/acdc_test_pred \
  -d 116 -c 2d -tr nnUNetTrainer_500epochs -f all
for t in psi1w3 psi6w3 psi7w3 psi10w3 ps1ctrl; do
  nnUNetv2_predict -i $nnUNet_raw/Dataset116_ACDC/imagesTs -o $PSI_ROOT/nnUNet/acdc_test_pred_$t \
    -d 116 -c 2d -tr nnUNetTrainer_500epochs_$t -f all
done

# 評估（產生 summary.json）
B=$nnUNet_results/Dataset115_BTCV/nnUNetTrainer__nnUNetPlans__2d
for d in $PSI_ROOT/btcv_test_pred/*/; do
  nnUNetv2_evaluate_folder $nnUNet_raw/Dataset115_BTCV/labelsTs $d -djfile $B/dataset.json -pfile $B/plans.json
done
A=$nnUNet_results/Dataset116_ACDC/nnUNetTrainer_500epochs__nnUNetPlans__2d
for d in $PSI_ROOT/nnUNet/acdc_test_pred*/; do
  nnUNetv2_evaluate_folder $nnUNet_raw/Dataset116_ACDC/labelsTs $d -djfile $A/dataset.json -pfile $A/plans.json
done
```

不執行任何後處理（`nnUNetv2_find_best_configuration` / `apply_postprocessing`），以保留 Ψ6 所量測的分割碎片。

---

## 9. 結果整理

```bash
cd $PSI_ROOT/psi-weighted-nnunet
python3 analysis/dice_tables.py       # → $PSI_ROOT/tree_features/dice_tables/
python3 analysis/final_report.py      # → $PSI_ROOT/tree_features/final_report/
python3 analysis/plot_dice.py         # → $PSI_ROOT/figs/
```

**預期**（測試集，病例平均 Dice）：

| 資料集 | baseline | Ψ1 | Ψ6 | Ψ7 | Ψ8 | Ψ10 |
|---|---|---|---|---|---|---|
| BTCV | 0.8533 | 0.8477 | 0.8412 | — | 0.8506 | — |
| ACDC | 0.9171 | 0.9241 | 0.9213 | 0.9239 | — | 0.9222 |

GPU 運算具非確定性，重新訓練的數值可能有約 ±0.01 的差異；倉庫 `results/` 中為原始實驗的輸出。

---

## 10. 驗證與官方 nnU-Net 的一致性（可選）

```bash
bash verification/verify_nnunet.sh           # nnU-Net 版本與核心檔案逐位元組比對
bash verification/plan_ensemble_report.sh    # 規劃參數與官方重算比對、推論設定
bash verification/checklist.sh               # 全部進度一頁檢查表
```
