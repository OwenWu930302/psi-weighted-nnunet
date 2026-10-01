#!/bin/bash
# ============================================================
# 實驗進度檢查表（每項皆由終端機即時驗證）
# 用法：conda activate expertree && bash checklist.sh
# ============================================================
export PSI_ROOT="${PSI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # 預設：倉庫的上一層
set -u
H="${PSI_ROOT}"
R="$H/nnUNet_data/nnUNet_results"
RAW="$H/nnUNet_data/nnUNet_raw"
PRE="$H/nnUNet_data/nnUNet_preprocessed"
OUT="$H/checklist_$(date +%Y%m%d_%H%M%S).txt"

row() { printf "%-2s %-34s %s\n" "$1" "$2" "$3"; }
yn() { [ "$1" = "$2" ] && echo "✓" || echo "✗"; }

{
echo "=============================================================="
echo " 實驗進度檢查表   $(date '+%Y-%m-%d %H:%M')   主機 $(hostname)"
echo " 每一項都是本機即時執行的結果，未經手動編輯"
echo "=============================================================="
echo
echo "【A. 程式碼來源】"

cd "$H/nnUNet" 2>/dev/null || exit 1
REMOTE=$(git remote get-url origin)
HEADH=$(git rev-parse --short HEAD)
TAGOK=$(yn "$(git rev-parse HEAD)" "$(git rev-parse v2.8.1^{commit})")
row "$TAGOK" "nnU-Net 為官方 v2.8.1" "$REMOTE @ $HEADH"

CORE="nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py nnunetv2/training/loss/dice.py nnunetv2/training/loss/compound_losses.py nnunetv2/training/loss/deep_supervision.py nnunetv2/run/run_training.py nnunetv2/inference/predict_from_raw_data.py"
SAME=0; TOT=0
for f in $CORE; do TOT=$((TOT+1)); [ "$(git hash-object $f)" = "$(git rev-parse v2.8.1:$f)" ] && SAME=$((SAME+1)); done
row "$(yn $SAME $TOT)" "訓練/loss/推論核心未修改" "$SAME/$TOT 檔逐位元組相同"

MOD=$(git diff --name-only v2.8.1 | wc -l)
row "i" "官方檔案改動數" "$MOD 個（詳見 nnunet_verification 報告）"

echo
echo "【B. 資料與前處理】"
for D in Dataset115_BTCV Dataset116_ACDC; do
  PN=$(python3 -c "import json;p=json.load(open('$PRE/$D/nnUNetPlans.json'));print(p['plans_name'],p.get('experiment_planner_used'))" 2>/dev/null)
  row "$([ "$PN" = "nnUNetPlans ExperimentPlanner" ] && echo ✓ || echo ✗)" "$D 前處理為官方預設" "$PN"
done

BTR=$(ls $RAW/Dataset115_BTCV/imagesTr 2>/dev/null | wc -l)
BTS=$(ls $RAW/Dataset115_BTCV/imagesTs 2>/dev/null | wc -l)
BLS=$(ls $RAW/Dataset115_BTCV/labelsTs 2>/dev/null | wc -l)
row "$(yn $BTR 18)" "BTCV 18 訓練 / 12 測試（TransUNet）" "訓練 $BTR、測試影像 $BTS、測試標註 $BLS"

ATR=$(ls $RAW/Dataset116_ACDC/imagesTr 2>/dev/null | wc -l)
ATS=$(ls $RAW/Dataset116_ACDC/imagesTs 2>/dev/null | wc -l)
row "$(yn $ATR 160)" "ACDC 80/20 病人（160/40 影像）" "訓練 $ATR、測試 $ATS"

python3 - <<'PY'
import json, os, csv
H=os.environ['PSI_ROOT']
def row(a,b,c): print(f"{a:<2} {b:<34} {c}")
# BTCV 5 折
try:
    sp=json.load(open(f'{H}/nnUNet_data/nnUNet_preprocessed/Dataset115_BTCV/splits_final.json'))
    v=[c for f in sp for c in f['val']]
    row('✓' if len(sp)==5 and len(set(v))==18 else '✗','BTCV 5 折切分完整',f'{len(sp)} 折，驗證合計 {len(set(v))} 例')
except Exception as e: row('✗','BTCV 5 折切分完整',str(e)[:40])
# ACDC 依病人分組
try:
    sp=json.load(open(f'{H}/nnUNet_data/nnUNet_preprocessed/Dataset116_ACDC/splits_final.json'))
    m={r['case']:r['patient'] for r in csv.DictReader(open(f'{H}/tree_features/acdc_case_patient_map.csv'))}
    bad=sum(1 for f in sp if {m[c] for c in f['train']} & {m[c] for c in f['val']})
    v=[c for f in sp for c in f['val']]
    row('✓' if bad==0 else '✗','ACDC 5 折依病人分組（ED/ES 不拆）',f'{len(sp)} 折，跨組病人 {bad} 位，驗證合計 {len(set(v))}')
except Exception as e: row('i','ACDC 5 折依病人分組','尚未建立或無對應表')
PY

echo
echo "【C. 訓練完成度】（f=完成折數，e=最後 epoch）"
chk() { # $1 資料集 $2 trainer $3 期望折數 $4 期望最後epoch $5 說明
  d="$R/$1/$2__nnUNetPlans__2d"; n=0; last="-"
  for k in $(ls $d 2>/dev/null | grep '^fold_'); do
    [ -f "$d/$k/checkpoint_final.pth" ] && n=$((n+1))
    l=$(ls -t $d/$k/training_log_*.txt 2>/dev/null | head -1)
    [ -n "$l" ] && last=$(grep -o "Epoch [0-9]*" "$l" | tail -1 | awk '{print $2}')
  done
  row "$(yn $n $3)" "$5" "f=$n/$3  e=$last"
}
chk Dataset115_BTCV nnUNetTrainer 5 999 "BTCV anchor（1000 ep）"
chk Dataset115_BTCV nnUNetTrainer_btcv_psi8w3 5 999 "BTCV Ψ8 專家（1000 ep）"
chk Dataset115_BTCV nnUNetTrainer_btcv_psi6w3 5 999 "BTCV Ψ6 專家（1000 ep）"
chk Dataset115_BTCV nnUNetTrainer_btcv_w1ctrl 5 999 "BTCV 對照組 w1.0"
chk Dataset116_ACDC nnUNetTrainer_500epochs 5 499 "ACDC OOF 5 折（500 ep，補做中）"

echo
echo "【D. Ψ 加權設定】"
for c in psi1_convex psi6_cc psi8_rotation; do
  f="$H/tree_features/wcases_btcv_$c.csv"
  [ -f "$f" ] && row "✓" "BTCV 加權清單 $c" "$(( $(wc -l < $f) - 1 )) 例（Ψ 前 30%）" \
               || row "✗" "BTCV 加權清單 $c" "不存在"
done
for t in psi8w3 psi6w3; do
  l=$(grep -h "本輪加權樣本" $R/Dataset115_BTCV/nnUNetTrainer_btcv_${t}__nnUNetPlans__2d/fold_*/training_log_*.txt 2>/dev/null | tail -1 | sed 's/.*本輪/本輪/')
  row "$([ -n "$l" ] && echo ✓ || echo ✗)" "$t 加權確實生效（訓練 log）" "${l:-無紀錄}"
done

echo
echo "【E. 推論與評估】"
for t in nnUNetTrainer nnUNetTrainer_btcv_psi8w3 nnUNetTrainer_btcv_psi6w3; do
  n=$(ls $H/btcv_test_pred/$t/*.nii.gz 2>/dev/null | wc -l)
  s=$([ -f "$H/btcv_test_pred/$t/summary.json" ] && echo "已評估" || echo "未評估")
  row "$(yn $n 12)" "BTCV 測試集預測 $t" "$n/12 例，$s"
done
PP=$(find $R -name "postprocessing.pkl" 2>/dev/null | wc -l)
row "$(yn $PP 0)" "未套用後處理（公平比較）" "找到 $PP 個 postprocessing.pkl"

echo
echo "【F. 已知限制（主動揭露）】"
echo "  1. ACDC 原以 -f all 訓練，無 OOF，proxy 篩選基於飽和的訓練集預測；"
echo "     目前補做 5 折 OOF 以解決此問題。"
echo "  2. 對照組 w1ctrl 因倍率 1.0 觸發原始路徑，等同 anchor 重訓，"
echo "     未測到逐樣本 loss 的影響；需倍率 1.0001 的對照組才能分離。"
echo "  3. 訓練長度：BTCV 1000 ep（nnU-Net 預設）、ACDC 500 ep（SAMA-UNet）；"
echo "     各資料集內部 anchor 與專家完全一致。"
echo "  4. BTCV 僅評估 8 個器官、採 18/12 切分（沿用 TransUNet / SAMA-UNet），"
echo "     nnU-Net Revisited 指出此協定有統計雜訊偏高的疑慮。"
echo "  5. 逐樣本加權使含加權樣本的 batch 由 batch Dice 改為 per-sample Dice，"
echo "     為加權之外的第二個變因，論文中會揭露。"
echo
echo "=============================================================="
} 2>&1 | tee "$OUT"

echo
echo "已存檔：$OUT"
sha256sum "$OUT"
