#!/bin/bash
# 當場示範：6 項，每項只印 1-3 行，全部來自 nnU-Net 自己產生的原始檔案
export PSI_ROOT="${PSI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # 預設：倉庫的上一層
H="${PSI_ROOT}"
R=$H/nnUNet_data/nnUNet_results
P=$H/nnUNet_data/nnUNet_preprocessed

echo "【1】程式碼來源與版本（git，即時向 GitHub 查詢）"
cd $H/nnUNet
echo "    $(git remote get-url origin)"
echo "    本機 HEAD = $(git rev-parse --short HEAD)   官方 v2.8.1 = $(git rev-parse --short v2.8.1^{commit})"

echo
echo "【2】核心訓練/loss/推論程式碼是否被改過"
for f in nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py \
         nnunetv2/training/loss/dice.py \
         nnunetv2/inference/predict_from_raw_data.py; do
  [ "$(git hash-object $f)" = "$(git rev-parse v2.8.1:$f)" ] \
    && echo "    ✓ 與官方相同  $(basename $f)" || echo "    ✗ 已修改  $(basename $f)"
done

echo
echo "【3】我唯一改的地方（官方 1 行 → 我的 11 行）"
sed -n '/if torch.allclose/,/l = total \/ w.sum()/p' nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_PsiWeighted.py \
  | sed 's/^/    /'

echo
echo "【4】加權確實生效（nnU-Net 訓練時自己寫的 log，非事後編輯）"
L=$(ls -t $R/Dataset115_BTCV/nnUNetTrainer_btcv_psi8w3__nnUNetPlans__2d/fold_0/training_log_*.txt | head -1)
echo "    檔案：$(basename $L)"
grep -m1 "Ψ加權\] 清單" $L | sed 's/^/    /'
grep -m2 "本輪加權樣本" $L | sed 's/^/    /'

echo
echo "【5】規劃參數（nnU-Net 自動產生的 nnUNetPlans.json）"
python3 -c "
import json
for d in ['Dataset115_BTCV','Dataset116_ACDC']:
    c=json.load(open('$P/'+d+'/nnUNetPlans.json'))
    x=c['configurations']['2d']
    print(f\"    {d:<16} planner={c.get('experiment_planner_used')}  batch={x['batch_size']}  patch={x['patch_size']}\")"

echo
echo "【6】測試集結果（nnU-Net 自己算的 summary.json）"
python3 -c "
import json, numpy as np
for tag,d in [('anchor','nnUNetTrainer'),('Psi8x3','nnUNetTrainer_btcv_psi8w3'),('Psi6x3','nnUNetTrainer_btcv_psi6w3')]:
    j=json.load(open('$H/btcv_test_pred/'+d+'/summary.json'))
    a=np.array([[c['metrics'][k]['Dice'] for k in sorted(c['metrics'],key=int)] for c in j['metric_per_case']],float)
    print(f'    {tag:<8} n={len(a)}  Dice {np.nanmean(np.nanmean(a,1)):.4f}')
print('    （參考：SAMA-UNet 論文 Table 2 的 nnUNet = 0.8493）')"
