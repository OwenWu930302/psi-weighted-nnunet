#!/bin/bash
# ============================================================
# BTCV：用「拿該病例訓練過」的每一折模型，預測該折的訓練病例
# ------------------------------------------------------------
# 目的：比較被加權病例在「同一種 loss（逐張）」下，加權前後的 Dice
#   加權前 = nnUNetTrainer_btcv_allctrl（權重等同 1）
#   加權後 = nnUNetTrainer_btcv_psi1w3 / psi6w3 / psi8w3（清單病例 ×3）
# 5 折 OOF 不能回答這個問題：OOF 中每個病例由「它當驗證集、沒拿它訓練」的那折預測。
# 這裡改為：fold k 的模型只預測 fold k 的訓練病例（每個病例會被它參與訓練的 4 折各預測一次）。
#
# 兩種模式：
#   weighted（預設）只推被加權的病例
#       專家：自己清單上的病例；allctrl：三份清單的聯集（作為加權前的對照）
#       約 1 小時
#   all     推每一折的全部訓練病例（已推過的會略過，只補缺的）
#       約再 1.5–2 小時
#
# 只做推論與評估，不訓練、不改動任何模型。可中斷後重跑（已推過的病例會略過）。
# 輸出：$PSI_ROOT/btcv_trainfit_pred/<trainer>/fold_<k>/（預測 + summary.json）
#
# 用法：
#   conda activate expertree
#   source <倉庫>/env.sh
#   bash analysis/predict_btcv_trainfit.sh            # 先推被加權的病例
#   bash analysis/predict_btcv_trainfit.sh all        # 之後補齊全部訓練病例
# ============================================================
set -e
: "${PSI_ROOT:?請先 source env.sh}"
MODE=${1:-weighted}
[ "$MODE" = weighted ] || [ "$MODE" = all ] || { echo "模式只能是 weighted 或 all"; exit 1; }
RAW=$nnUNet_raw/Dataset115_BTCV
SPLIT=$nnUNet_preprocessed/Dataset115_BTCV/splits_final.json
B=$nnUNet_results/Dataset115_BTCV/nnUNetTrainer__nnUNetPlans__2d   # 取 dataset.json / plans.json 供評估
TF=$PSI_ROOT/tree_features
IN=$PSI_ROOT/btcv_trainfit_input
OUT=$PSI_ROOT/btcv_trainfit_pred

# 每個模型要推的病例（weighted 模式）
declare -A LIST=(
  [psi1w3]="$TF/wcases_btcv_psi1_convex.csv"
  [psi6w3]="$TF/wcases_btcv_psi6_cc.csv"
  [psi8w3]="$TF/wcases_btcv_psi8_rotation.csv"
  [allctrl]="$TF/wcases_btcv_psi1_convex.csv $TF/wcases_btcv_psi6_cc.csv $TF/wcases_btcv_psi8_rotation.csv"
)

for k in 0 1 2 3 4; do
  TRAIN=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT'))[$k]['train']))")
  for t in allctrl psi1w3 psi6w3 psi8w3; do
    tr=nnUNetTrainer_btcv_$t
    if [ "$MODE" = all ]; then
      CASES=$TRAIN
    else   # 清單病例 ∩ 本折訓練病例
      WANT=$(tail -q -n +2 ${LIST[$t]} | cut -d, -f1 | tr -d '\r' | sort -u)
      CASES=$(for c in $TRAIN; do echo "$WANT" | grep -qx "$c" && echo $c; done; true)
    fi
    d=$IN/$tr/fold_$k
    mkdir -p $d
    find $d -maxdepth 1 -type l -delete            # 只清除本腳本建立的符號連結
    for c in $CASES; do ln -sf $RAW/imagesTr/${c}_0000.nii.gz $d/${c}_0000.nii.gz; done
    n=$(ls $d | wc -l)
    [ "$n" -eq 0 ] && { echo "（$tr fold $k 沒有要推的病例）"; continue; }
    o=$OUT/$tr/fold_$k
    echo "===== [$MODE] $tr　fold $k　$n 例：$(echo $CASES | tr '\n' ' ')"
    nnUNetv2_predict -i $d -o $o -d 115 -c 2d -tr $tr -f $k --continue_prediction
    nnUNetv2_evaluate_folder $RAW/labelsTr $o -djfile $B/dataset.json -pfile $B/plans.json --chill
  done
done
echo "完成（$MODE）：$OUT"