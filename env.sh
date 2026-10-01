# ============================================================
# 環境設定：每開一個新終端機都要先執行
#     source env.sh
#
# PSI_ROOT 是「工作區」：倉庫、nnUNet、nnUNet_data 都放在它底下。
# 預設為本倉庫的上一層資料夾（依 REPRODUCE.md 第 1 步的放法即正確），
# 因此在任何新終端機中，只要 source 這個檔案即可，不需事先設定任何變數。
# 若工作區在別處，先 export PSI_ROOT=<路徑> 再 source。
# ============================================================

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PSI_ROOT="${PSI_ROOT:-$(dirname "$_ENV_DIR")}"

# nnU-Net 需要的三個路徑
export nnUNet_raw="$PSI_ROOT/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PSI_ROOT/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PSI_ROOT/nnUNet_data/nnUNet_results"

# 原始資料位置（可依實際存放位置修改）
export TRANSUNET_SYNAPSE="${TRANSUNET_SYNAPSE:-$PSI_ROOT/SAMA-UNet_repo/project_TransUNet/data/Synapse}"
export BTCV_NIFTI="${BTCV_NIFTI:-$PSI_ROOT/BTCV_nifti}"
export ACDC_RAW="${ACDC_RAW:-$PSI_ROOT/SAMA-UNet_repo/ACDC/database/training}"

# 避免殘留的加權設定影響訓練（ACDC 舊專家以 setdefault 讀取這兩個變數）
unset PSI_WEIGHT_CASES PSI_WEIGHT_FACTOR

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$PSI_ROOT/tree_features"

echo "PSI_ROOT            = $PSI_ROOT"
[ -d "$PSI_ROOT/nnUNet/nnunetv2" ] || echo "  ⚠ 找不到 $PSI_ROOT/nnUNet（尚未安裝 nnU-Net，或 PSI_ROOT 設錯）"
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "TRANSUNET_SYNAPSE   = $TRANSUNET_SYNAPSE"
echo "BTCV_NIFTI          = $BTCV_NIFTI"
echo "ACDC_RAW            = $ACDC_RAW"
