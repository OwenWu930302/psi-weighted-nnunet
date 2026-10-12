"""
被加權病例：同一種 loss（逐張計算）下，加權前後的 Dice
============================================================
加權前 = 對照組（權重等同 1，與專家走同一條逐張 loss 路徑）
加權後 = Ψ 專家（清單病例 ×3）
兩者只差在「有沒有 ×3 加權」，因此 Δ = 加權本身對該病例的影響。

不與 baseline 比較：baseline 走官方路徑（整批 Dice），與專家的 loss 算法不同，
差異會混入「逐張 vs 整批 Dice」的效果（例如 ACDC 對照組本身在訓練集上就比 baseline 低約 0.03）。

預測來源（模型訓練時看過該病例 → 反映加權是否讓模型更擬合該病例）：
  ACDC  fold_all 模型對 160 個訓練影像的預測（nnU-Net fold_all/validation/summary.json）
        對照組：nnUNetTrainer_500epochs_ps1ctrl
  BTCV  每個病例由「拿它訓練」的 4 折模型各預測一次，取平均
        （先執行 analysis/predict_btcv_trainfit.sh 產生預測）
        對照組：nnUNetTrainer_btcv_allctrl
  註：5 折 OOF 不適用 —— OOF 中每個病例由沒拿它訓練的那折預測。

輸出目錄：$PSI_ROOT/tree_features/weighted_cases/
  weighted_cases_same_loss_ACDC.csv      每個被加權病例：加權前 / 後 Dice（整體與各結構）、Δ、提升或下降
  weighted_cases_same_loss_BTCV.csv      同上
  weighted_cases_same_loss_summary.csv   每個專家一列：例數、平均 Dice、Δ、提升 / 下降例數、Wilcoxon p

聚合規則（與 final_report.py 相同）：
  - 任一方為 nan 的「病例 × 結構」兩邊都排除，再取各結構平均
  - BTCV 先在每一折配對，再對該病例參與訓練的各折取平均

用法：
  conda activate expertree
  source <倉庫>/env.sh
  python3 analysis/weighted_cases_same_loss.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

H = os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.environ.get('nnUNet_results') or f'{H}/nnUNet_data/nnUNet_results'
PRE = os.environ.get('nnUNet_preprocessed') or f'{H}/nnUNet_data/nnUNet_preprocessed'
TF = f'{H}/tree_features'
OUT = f'{TF}/weighted_cases'
TRAINFIT = f'{H}/btcv_trainfit_pred'

ORG = {'ACDC': ['RV', 'Myo', 'LV'],
       'BTCV': ['Aorta', 'Gallbl', 'KidL', 'KidR', 'Liver', 'Pancr', 'Spleen', 'Stomach']}
CTRL = {'ACDC': 'nnUNetTrainer_500epochs_ps1ctrl', 'BTCV': 'nnUNetTrainer_btcv_allctrl'}
EXPERTS = {
    'ACDC': [('psi1w3', 'nnUNetTrainer_500epochs_psi1w3', 'wcases_psi1_convex.csv'),
             ('psi6w3', 'nnUNetTrainer_500epochs_psi6w3', 'wcases_psi6_cc.csv'),
             ('psi7w3', 'nnUNetTrainer_500epochs_psi7w3', 'wcases_psi7_genus.csv'),
             ('psi10w3', 'nnUNetTrainer_500epochs_psi10w3', 'wcases_acdc_psi10_centroid.csv')],
    'BTCV': [('psi1w3', 'nnUNetTrainer_btcv_psi1w3', 'wcases_btcv_psi1_convex.csv'),
             ('psi6w3', 'nnUNetTrainer_btcv_psi6w3', 'wcases_btcv_psi6_cc.csv'),
             ('psi8w3', 'nnUNetTrainer_btcv_psi8w3', 'wcases_btcv_psi8_rotation.csv')],
}


def read_summary(f):
    """summary.json → {病例: [各結構 Dice]}"""
    out = {}
    for c in json.load(open(f))['metric_per_case']:
        name = os.path.basename(c['reference_file']).replace('.nii.gz', '')
        out[name] = np.array([c['metrics'][k]['Dice'] for k in sorted(c['metrics'], key=int)], float)
    return out


def paired(a, b):
    """任一方為 nan 的結構兩邊都排除"""
    a, b = a.copy(), b.copy()
    both = np.isnan(a) | np.isnan(b)
    a[both] = np.nan
    b[both] = np.nan
    return a, b


def load_acdc(trainer):
    f = f'{RES}/Dataset116_ACDC/{trainer}__nnUNetPlans__2d/fold_all/validation/summary.json'
    return {'all': read_summary(f)} if os.path.exists(f) else None


def load_btcv(trainer):
    folds = {}
    for k in range(5):
        f = f'{TRAINFIT}/{trainer}/fold_{k}/summary.json'
        if os.path.exists(f):
            folds[k] = read_summary(f)
    return folds if len(folds) == 5 else None


def case_scores(ctrl, expt, case, n_org):
    """回傳（加權前各結構、加權後各結構、使用的折數）；各折先配對再平均"""
    bs, as_ = [], []
    for k in ctrl:
        if case in ctrl[k] and k in expt and case in expt[k]:
            b, a = paired(ctrl[k][case], expt[k][case])
            bs.append(b)
            as_.append(a)
    if not bs:
        return None, None, 0
    with warnings.catch_warnings():                     # 兩邊皆無該結構（如 BTCV_007 膽囊）→ nan，不警告
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(bs, 0), np.nanmean(as_, 0), len(bs)


def main():
    os.makedirs(OUT, exist_ok=True)
    loader = {'ACDC': load_acdc, 'BTCV': load_btcv}
    summary = []
    for ds in ('ACDC', 'BTCV'):
        org = ORG[ds]
        C = loader[ds](CTRL[ds])
        if C is None:
            hint = '（請先執行 analysis/predict_btcv_trainfit.sh）' if ds == 'BTCV' else ''
            print(f'⚠ {ds}：找不到對照組 {CTRL[ds]} 的預測，略過{hint}')
            continue
        rows = []
        for name, trainer, lf in EXPERTS[ds]:
            E = loader[ds](trainer)
            if E is None:
                print(f'⚠ {ds} {name}：找不到預測，略過')
                continue
            cases = sorted(pd.read_csv(f'{TF}/{lf}')['case'].astype(str))
            deltas = []
            for c in cases:
                b, a, nf = case_scores(C, E, c, len(org))
                if b is None:
                    print(f'  ⚠ {ds} {name} {c}：沒有可配對的預測')
                    continue
                bm, am = np.nanmean(b), np.nanmean(a)
                d = am - bm
                deltas.append(d)
                r = {'expert': name, 'list_file': lf, 'case': c,
                     'control': CTRL[ds].replace('nnUNetTrainer_', ''),
                     'n_folds': nf if ds == 'BTCV' else 'fold_all',
                     'before_control': bm, 'after_expert': am, 'delta': d,
                     'result': '提升' if d > 0 else ('下降' if d < 0 else '持平')}
                for i, o in enumerate(org):
                    r[f'{o}_before'] = b[i]
                    r[f'{o}_after'] = a[i]
                    r[f'{o}_delta'] = a[i] - b[i]
                rows.append(r)
            d = np.array(deltas)
            p = float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else np.nan
            sub = [r for r in rows if r['expert'] == name]
            summary.append({'dataset': ds, 'expert': name, 'list_file': lf,
                            'control': CTRL[ds].replace('nnUNetTrainer_', ''),
                            'prediction_source': 'fold_all 訓練集預測' if ds == 'ACDC'
                            else '拿該病例訓練的各折預測（平均）',
                            'n_weighted': len(d),
                            'before_control': np.mean([r['before_control'] for r in sub]),
                            'after_expert': np.mean([r['after_expert'] for r in sub]),
                            'delta': d.mean(), '提升': int((d > 0).sum()), '下降': int((d < 0).sum()),
                            'wilcoxon_p': p})
        f = f'{OUT}/weighted_cases_same_loss_{ds}.csv'
        pd.DataFrame(rows).to_csv(f, index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f'→ {f}（{len(rows)} 列）')

    S = pd.DataFrame(summary)
    S.to_csv(f'{OUT}/weighted_cases_same_loss_summary.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    pd.set_option('display.width', 200)
    print('\n被加權病例：加權前（對照組，權重 1）→ 加權後（專家 ×3），同一種逐張 loss')
    if len(S):
        print(S[['dataset', 'expert', 'control', 'n_weighted', 'before_control', 'after_expert',
                 'delta', '提升', '下降', 'wilcoxon_p']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\n輸出目錄：{OUT}')


if __name__ == '__main__':
    main()