# ============================================================
# 環境設定：每開一個新終端機都要先執行
#     source env.sh
#
# 只需修改 PSI_ROOT 這一行，其餘路徑都由它推導。
# 所有程式皆讀取 PSI_ROOT；未設定時預設為 ~/桌面/論文（原作者環境）。
# ============================================================

export PSI_ROOT="${PSI_ROOT:-$HOME/桌面/論文}"

# nnU-Net 需要的三個路徑
export nnUNet_raw="$PSI_ROOT/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PSI_ROOT/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PSI_ROOT/nnUNet_data/nnUNet_results"

# 原始資料位置（可依實際存放位置修改）
export BTCV_NIFTI="${BTCV_NIFTI:-$PSI_ROOT/BTCV_nifti}"
export ACDC_RAW="${ACDC_RAW:-$PSI_ROOT/SAMA-UNet_repo/ACDC/database/training}"

# 避免殘留的加權設定影響訓練（ACDC 舊專家以 setdefault 讀取這兩個變數）
unset PSI_WEIGHT_CASES PSI_WEIGHT_FACTOR

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$PSI_ROOT/tree_features"

echo "PSI_ROOT            = $PSI_ROOT"
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "BTCV_NIFTI          = $BTCV_NIFTI"
echo "ACDC_RAW            = $ACDC_RAW"
