#!/bin/bash
# ============================================================
# 驗證：規劃參數與 ensemble 預測是否與官方 nnU-Net 相同
# 用法：conda activate expertree && bash verify_plan_ensemble.sh
# ============================================================
set -u
H="${PSI_ROOT:-$HOME/桌面/論文}"
PRE="$H/nnUNet_data/nnUNet_preprocessed"
OUT="$H/verify_plan_ensemble_$(date +%Y%m%d_%H%M%S).txt"
export nnUNet_raw="$H/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PRE"
export nnUNet_results="$H/nnUNet_data/nnUNet_results"

{
echo "############################################################"
echo "# 規劃參數 / Ensemble 預測 驗證   $(date '+%Y-%m-%d %H:%M')"
echo "############################################################"

echo
echo "===== 1. 現有 plans 是誰產生的 ====="
for d in Dataset115_BTCV Dataset116_ACDC; do
  python3 -c "
import json
p = json.load(open('$PRE/$d/nnUNetPlans.json'))
print('$d : plans_name=%s  planner=%s' % (p['plans_name'], p.get('experiment_planner_used')))"
done

echo
echo "===== 2. 重新執行官方規劃器，與現有 plans 比對 ====="
echo "（只重跑規劃、不重跑前處理；結果寫到另一個名稱，不覆蓋原檔）"
for d in 115 116; do
  case $d in 115) D=Dataset115_BTCV;; 116) D=Dataset116_ACDC;; esac
  cp "$PRE/$D/nnUNetPlans.json" "/tmp/${D}_orig.json"
  nnUNetv2_plan_experiment -d $d -pl ExperimentPlanner \
      -overwrite_plans_name nnUNetPlans_verify > /tmp/plan_$d.log 2>&1
  if [ ! -f "$PRE/$D/nnUNetPlans_verify.json" ]; then
    echo "  $D : 規劃失敗，詳見 /tmp/plan_$d.log"; tail -3 /tmp/plan_$d.log; continue
  fi
  python3 - <<PY
import json
a = json.load(open('/tmp/${D}_orig.json'))
b = json.load(open('$PRE/$D/nnUNetPlans_verify.json'))
ca, cb = a['configurations']['2d'], b['configurations']['2d']
keys = ['batch_size','patch_size','spacing','normalization_schemes','batch_dice',
        'median_image_size_in_voxels','use_mask_for_norm','data_identifier']
diff = [k for k in keys if ca.get(k) != cb.get(k)]
archa, archb = ca['architecture'], cb['architecture']
same_arch = archa == archb
print('  $D : 2d 設定差異 %s ；網路架構 %s' %
      ('無' if not diff else diff, '相同' if same_arch else '不同'))
if diff:
    for k in diff: print('     ', k, '現有', ca.get(k), '/ 重算', cb.get(k))
PY
  rm -f "$PRE/$D/nnUNetPlans_verify.json"
done
echo "  （驗證用的 plans 已刪除，原 nnUNetPlans.json 未被更動）"

echo
echo "===== 3. 完整規劃參數（論文用） ====="
python3 - <<'PY'
import json, os
PRE = os.path.join(os.environ.get('PSI_ROOT', os.path.expanduser('~/桌面/論文')), 'nnUNet_data/nnUNet_preprocessed')
for d in ['Dataset115_BTCV', 'Dataset116_ACDC']:
    p = json.load(open(f'{PRE}/{d}/nnUNetPlans.json'))
    c = p['configurations']['2d']; a = c['architecture']['arch_kwargs']
    print(f"--- {d}")
    for k in ['batch_size','patch_size','spacing','median_image_size_in_voxels',
              'normalization_schemes','use_mask_for_norm','batch_dice']:
        print(f"    {k:<28}{c[k]}")
    print(f"    {'network':<28}{c['architecture']['network_class_name'].split('.')[-1]}")
    print(f"    {'n_stages':<28}{a['n_stages']}")
    print(f"    {'features_per_stage':<28}{a['features_per_stage']}")
    print(f"    {'strides':<28}{a['strides']}")
    print(f"    {'n_conv_per_stage':<28}{a['n_conv_per_stage']}")
    print(f"    {'norm_op':<28}{a['norm_op'].split('.')[-1]}")
    print(f"    {'nonlin':<28}{a['nonlin'].split('.')[-1]}")
PY

echo
echo "===== 4. 官方推論程式的預設值（未修改，見版本驗證報告） ====="
cd "$H/nnUNet"
echo "--- 推論入口 nnUNetv2_predict 的參數預設"
grep -A2 -E "'-f'|'-step_size'|'--disable_tta'|'-chk'|'--save_probabilities'" \
  nnunetv2/inference/predict_from_raw_data.py | grep -E "default=|action=|help=" | head -20
echo
echo "--- nnUNetPredictor 建構子預設"
grep -A8 "def __init__(self," nnunetv2/inference/predict_from_raw_data.py | head -12

echo
echo "===== 5. Ensemble 如何合併各折（官方程式碼） ====="
grep -n -B2 -A6 "for params in self.list_of_parameters" \
  nnunetv2/inference/predict_from_raw_data.py | head -30

echo
echo "===== 6. 本研究實際執行的推論設定 ====="
for f in "$H"/btcv_test_pred/*/predict_from_raw_data_args.json \
         "$H"/nnUNet/acdc_test_pred*/predict_from_raw_data_args.json; do
  [ -f "$f" ] || continue
  d=$(basename $(dirname "$f"))
  n=$(ls $(dirname "$f")/*.nii.gz 2>/dev/null | wc -l)
  sp=$(python3 -c "import json;print(json.load(open('$f')).get('save_probabilities'))")
  echo "  $d : 預測 $n 例   save_probabilities=$sp   (未列出的參數皆為官方預設)"
done

echo
echo "===== 7. 各模型使用的折數（由推論 log 的進度條數推算） ====="
for l in "$H"/btcv_test_pred.log "$H"/nnUNet/*.log; do
  [ -f "$l" ] || continue
  c=$(grep -c "^100%" "$l" 2>/dev/null)
  [ "$c" = "0" ] && continue
  echo "  $(basename $l) : 進度條 $c 條"
done
echo "  （BTCV 每例 5 條 = 5 折 ensemble；ACDC 每例 1 條 = fold_all 單一模型）"

echo
echo "############################################################"
} 2>&1 | tee "$OUT"

echo
echo "已存檔：$OUT"
sha256sum "$OUT"
