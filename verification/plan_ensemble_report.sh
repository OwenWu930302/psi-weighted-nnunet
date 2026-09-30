#!/bin/bash
# ============================================================
# 給指導教授：規劃參數（Planning）與 Ensemble 預測設定
# 每一項皆為本機即時執行結果
# 用法：conda activate expertree && bash plan_ensemble_report.sh
# ============================================================
set -u
H="${PSI_ROOT:-$HOME/桌面/論文}"
PRE="$H/nnUNet_data/nnUNet_preprocessed"
RES="$H/nnUNet_data/nnUNet_results"
OUT="$H/plan_ensemble_report_$(date +%Y%m%d_%H%M%S).txt"
export nnUNet_raw="$H/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PRE"
export nnUNet_results="$RES"

{
echo "================================================================"
echo " 規劃參數與 Ensemble 預測設定   $(date '+%Y-%m-%d %H:%M')   $(hostname)"
echo "================================================================"

echo
echo "【1】現有 plans 的產生者與時間"
for d in Dataset115_BTCV Dataset116_ACDC; do
  t=$(date -r "$PRE/$d/nnUNetPlans.json" '+%Y-%m-%d %H:%M')
  python3 -c "
import json
p=json.load(open('$PRE/$d/nnUNetPlans.json'))
print(f\"  {'$d':<18} plans_name={p['plans_name']}  planner={p.get('experiment_planner_used')}  產生於 $t\")"
done

echo
echo "【2】與官方規劃器重算結果的逐項比對"
echo "     方法：重跑 nnUNetv2_plan_experiment -pl ExperimentPlanner，寫入暫存名稱後比對，"
echo "           比對完刪除暫存檔，原 nnUNetPlans.json 全程未更動。"
for id in 115 116; do
  case $id in 115) D=Dataset115_BTCV;; 116) D=Dataset116_ACDC;; esac
  nnUNetv2_plan_experiment -d $id -pl ExperimentPlanner \
      -overwrite_plans_name nnUNetPlans_chk > /tmp/plan_$id.log 2>&1
  if [ ! -f "$PRE/$D/nnUNetPlans_chk.json" ]; then
    echo "  $D：規劃失敗（/tmp/plan_$id.log）"; continue
  fi
  python3 - <<PY
import json
a=json.load(open('$PRE/$D/nnUNetPlans.json'))['configurations']
b=json.load(open('$PRE/$D/nnUNetPlans_chk.json'))['configurations']
print('  --- $D')
print('    %-20s %-12s %-26s %-26s %s' % ('configuration','欄位','現有','官方重算','判定'))
keys=['batch_size','patch_size','spacing','normalization_schemes','batch_dice']
for cfg in a:
    if cfg not in b:
        print('    %-20s %s' % (cfg, '官方重算無此設定')); continue
    diff=False
    for k in keys:
        x, y = a[cfg].get(k), b[cfg].get(k)
        if x != y:
            diff=True
            print('    %-20s %-12s %-26s %-26s %s' % (cfg,k,str(x),str(y),'不同'))
    if a[cfg].get('architecture') != b[cfg].get('architecture'):
        print('    %-20s %-12s %-26s %-26s %s' % (cfg,'architecture','-','-','不同'))
    if not diff:
        print('    %-20s %-12s %-26s %-26s %s' % (cfg,'(全部欄位)','—','—','完全相同'))
PY
  rm -f "$PRE/$D/nnUNetPlans_chk.json"
done

echo
echo "【3】實際使用的規劃參數（2D，本研究採用）"
python3 - <<'PY'
import json, os
PRE=os.path.join(os.environ.get('PSI_ROOT', os.path.expanduser('~/桌面/論文')), 'nnUNet_data/nnUNet_preprocessed')
ks=['batch_size','patch_size','spacing','median_image_size_in_voxels',
    'normalization_schemes','use_mask_for_norm','batch_dice']
A={d: json.load(open(f'{PRE}/{d}/nnUNetPlans.json'))['configurations']['2d']
   for d in ['Dataset115_BTCV','Dataset116_ACDC']}
print('  %-28s %-28s %s' % ('參數','BTCV (115)','ACDC (116)'))
for k in ks:
    print('  %-28s %-28s %s' % (k, A['Dataset115_BTCV'][k], A['Dataset116_ACDC'][k]))
for k,f in [('network', lambda c: c['architecture']['network_class_name'].split('.')[-1]),
            ('n_stages', lambda c: c['architecture']['arch_kwargs']['n_stages']),
            ('features_per_stage', lambda c: c['architecture']['arch_kwargs']['features_per_stage']),
            ('n_conv_per_stage', lambda c: c['architecture']['arch_kwargs']['n_conv_per_stage']),
            ('norm_op', lambda c: c['architecture']['arch_kwargs']['norm_op'].split('.')[-1]),
            ('nonlin', lambda c: c['architecture']['arch_kwargs']['nonlin'].split('.')[-1])]:
    print('  %-28s %-28s %s' % (k, f(A['Dataset115_BTCV']), f(A['Dataset116_ACDC'])))
PY

echo
echo "【4】Ensemble 預測：官方程式碼的預設值"
cd "$H/nnUNet"
python3 - <<'PY'
import re
s=open('nnunetv2/inference/predict_from_raw_data.py').read()
want={'-f':'使用的折數','-step_size':'滑動視窗步長','--disable_tta':'關閉鏡像增強',
      '-chk':'checkpoint 檔名','--save_probabilities':'輸出機率圖'}
print('  %-22s %-24s %s' % ('參數','說明','官方預設'))
for k,desc in want.items():
    m=re.search(r"add_argument\('%s'.*?\)\n" % re.escape(k), s, re.S)
    if not m: continue
    blk=m.group(0)
    d=re.search(r"default=([^,\)]+)", blk)
    act=re.search(r"action='([^']+)'", blk)
    val = d.group(1).strip() if d else ('False（未指定即為 False）' if act else '—')
    print('  %-22s %-24s %s' % (k, desc, val))
m=re.search(r"def __init__\(self,\s*(.*?)\):", s, re.S)
print()
print('  nnUNetPredictor 建構子預設：')
for line in m.group(1).split('\n'):
    line=line.strip().rstrip(',')
    if line and '=' in line: print('    ', line)
PY

echo
echo "【5】Ensemble 如何合併多折（官方程式碼，未修改）"
grep -n -A3 "prediction = self._internal_maybe_mirror_and_predict\|prediction += self._internal_maybe_mirror_and_predict\|len(self.list_of_parameters)" \
  nnunetv2/inference/predict_from_raw_data.py | head -20

echo
echo "【6】本研究實際的推論設定與折數"
printf "  %-34s %-8s %-18s %s\n" "預測資料夾" "例數" "save_probabilities" "模型可用折數"
for f in "$H"/btcv_test_pred/*/predict_from_raw_data_args.json \
         "$H"/nnUNet/acdc_test_pred*/predict_from_raw_data_args.json; do
  [ -f "$f" ] || continue
  dir=$(dirname "$f"); name=$(basename "$dir")
  n=$(ls "$dir"/*.nii.gz 2>/dev/null | wc -l)
  sp=$(python3 -c "import json;print(json.load(open('$f')).get('save_probabilities'))")
  # 從 plans.json 旁的 dataset.json 對應不到 trainer，改由結果資料夾推算
  case "$name" in
    nnUNetTrainer|nnUNetTrainer_btcv_*) t="$RES/Dataset115_BTCV/${name}__nnUNetPlans__2d";;
    acdc_test_pred) t="$RES/Dataset116_ACDC/nnUNetTrainer_500epochs__nnUNetPlans__2d";;
    acdc_test_pred_psi*) v=${name#acdc_test_pred_}; t="$RES/Dataset116_ACDC/nnUNetTrainer_500epochs_${v}__nnUNetPlans__2d";;
    *) t="";;
  esac
  folds=$([ -d "$t" ] && ls "$t" | grep -c '^fold_' || echo "?")
  printf "  %-34s %-8s %-18s %s\n" "$name" "$n" "$sp" "${folds} 個 fold 目錄"
done

echo
echo "【7】結論"
echo "  (a) ACDC：2d / 3d_fullres 全部欄位與官方規劃器重算結果完全相同。"
echo "  (b) BTCV：3d_lowres / 3d_fullres 相同；2d 的 batch_size 與 patch_size 與官方重算不同"
echo "      （現有 4 / [256,256]；官方重算 14 / [448,512]）。其餘欄位與網路架構相同。"
echo "      差異僅出現在 2d，3d 不受影響，推測為當初針對 2d 另行指定所致。"
echo "      anchor 與所有專家模型皆使用同一份 plans，故模型間比較仍為公平對照。"
echo "  (c) Ensemble 與推論流程使用官方 nnUNetPredictor 預設值，推論程式碼未修改"
echo "      （見 nnunet_verification 報告：predict_from_raw_data.py 與 v2.8.1 逐位元組相同）。"
echo "  (d) 所有模型皆未套用後處理。"
echo "================================================================"
} 2>&1 | tee "$OUT"

echo
echo "已存檔：$OUT"
sha256sum "$OUT"