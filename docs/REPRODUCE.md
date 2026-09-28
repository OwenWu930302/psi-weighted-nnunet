# 重現步驟

以下指令假設專案根目錄為 `~/桌面/論文`，請依自己的環境調整。

```bash
conda activate expertree
export nnUNet_raw=~/桌面/論文/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=~/桌面/論文/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=~/桌面/論文/nnUNet_data/nnUNet_results
```

長時間訓練一律使用 `nohup ... &`，並在啟動後確認 log 中的 `[Ψ加權]` 訊息（清單路徑、例數、倍率、每輪加權比例）。

---

## 1. 資料轉換

```bash
python3 data_conversion/convert_btcv_to_nnunet.py     # Dataset115_BTCV，18 / 12
python3 data_conversion/convert_acdc_to_nnunet.py     # Dataset116_ACDC，80 / 20 位病人
```

BTCV 測試集標註需另外複製到 `nnUNet_raw/Dataset115_BTCV/labelsTs/`，對應表見 `splits/`。

## 2. 規劃與前處理

```bash
nnUNetv2_plan_and_preprocess -d 115 116 -c 2d --verify_dataset_integrity
```

實際使用的 plans 見 `configs/`。

## 3. baseline

```bash
# BTCV：5 折，1000 epochs
for f in 0 1 2 3 4; do nnUNetv2_train 115 2d $f --npz; done

# ACDC：all 模式，500 epochs
nnUNetv2_train 116 2d all -tr nnUNetTrainer_500epochs --npz

# ACDC：依病人分組的 5 折（補做 OOF）
python3 data_conversion/make_acdc_grouped_splits.py            # 檢查
python3 data_conversion/make_acdc_grouped_splits.py --write    # 寫入 splits_final.json
for f in 0 1 2 3 4; do nnUNetv2_train 116 2d $f -tr nnUNetTrainer_500epochs --npz; done
```

## 4. Ψ 計算與 proxy 篩選

```bash
python3 psi/run_psi8_btcv.py          # BTCV Ψ8（逐折重新推論）
python3 psi/run_all_psi.py            # 所有資料集的 Ψ
python3 psi/recompute_caselevel.py    # 病例層級聚合、AUC、p 值
python3 psi/make_weight_lists.py      # 依規則重新產生加權清單並與 weight_lists/ 比對
```

## 5. 專家模型

```bash
# BTCV
for t in psi1w3 psi6w3 psi8w3; do
  for f in 0 1 2 3 4; do nnUNetv2_train 115 2d $f -tr nnUNetTrainer_btcv_$t --npz; done
done

# ACDC
for t in psi1w3 psi6w3 psi7w3 psi10w3; do
  nnUNetv2_train 116 2d all -tr nnUNetTrainer_500epochs_$t --npz
done
```

## 6. 測試集推論與評估

```bash
# BTCV（5 折 ensemble）
nnUNetv2_predict -i $nnUNet_raw/Dataset115_BTCV/imagesTs -o btcv_test_pred/<trainer> \
  -d 115 -c 2d -tr <trainer> -f 0 1 2 3 4

# ACDC（單一模型）
nnUNetv2_predict -i $nnUNet_raw/Dataset116_ACDC/imagesTs -o nnUNet/acdc_test_pred_<name> \
  -d 116 -c 2d -tr <trainer> -f all

# 評估
nnUNetv2_evaluate_folder <labelsTs> <預測資料夾> -djfile <dataset.json> -pfile <plans.json>
```

## 7. 結果整理

```bash
python3 analysis/final_report.py      # → tree_features/final_report/*.csv
python3 analysis/dice_tables.py       # → tree_features/dice_tables/*.csv
python3 analysis/plot_dice_hist.py    # Dice 分布圖
```

## 8. 驗證

```bash
bash verification/verify_nnunet.sh           # 與官方 v2.8.1 逐檔比對
bash verification/plan_ensemble_report.sh    # 規劃參數與官方重算比對、ensemble 設定
```
