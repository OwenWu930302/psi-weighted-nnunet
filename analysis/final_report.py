"""
最終數據整理：所有比較輸出成 CSV
============================================================
輸出目錄：~/桌面/論文/tree_features/final_report/

  1_test_overall.csv        測試集：各模型 vs anchor（整體、最差 30%、Wilcoxon）
  2_test_per_structure.csv  測試集：逐器官 / 逐結構
  3_test_worst30.csv        測試集：anchor 最差 30% 病例逐例
  4_test_per_case.csv       測試集：所有病例 × 所有模型（寬表）
  5_oof_overall.csv         5 折 OOF：各模型 vs anchor（BTCV 18 例）
  6_cv_vs_all.csv           ACDC：5 折 OOF 與 -f all 的訓練集預測比較
  7_proxy_selection.csv     proxy 篩選（OOF，最低 30%）：AUC、p、方向
  8_acdc_list_overlap.csv   ACDC：舊清單（-f all 預測）vs 新清單（OOF）

聚合規則：
  - 病例分數 = 該病例各結構 Dice 的 nanmean
  - 與 anchor 比較時，任一方為 nan 的「病例 × 結構」兩邊都排除（分母一致）
  - 最差 30% 以 anchor 的病例分數排序，k = round(0.3 n)

用法：
  conda activate expertree
  cd ~/桌面/論文
  python3 final_report.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

H = os.path.expanduser('~/桌面/論文')
RES = f'{H}/nnUNet_data/nnUNet_results'
TF = f'{H}/tree_features'
OUT = f'{TF}/final_report'
os.makedirs(OUT, exist_ok=True)

BTCV_ORG = ['Aorta', 'Gallbl', 'KidL', 'KidR', 'Liver', 'Pancr', 'Spleen', 'Stomach']
ACDC_ORG = ['RV', 'Myo', 'LV']

# ---------------- 模型清單（不存在者自動跳過） ----------------
BTCV_TEST = {
    'anchor': f'{H}/btcv_test_pred/nnUNetTrainer',
    'psi1w3': f'{H}/btcv_test_pred/nnUNetTrainer_btcv_psi1w3',
    'psi6w3': f'{H}/btcv_test_pred/nnUNetTrainer_btcv_psi6w3',
    'psi8w3': f'{H}/btcv_test_pred/nnUNetTrainer_btcv_psi8w3',
    'w1ctrl': f'{H}/btcv_test_pred/nnUNetTrainer_btcv_w1ctrl',
}
ACDC_TEST = {
    'anchor':  f'{H}/nnUNet/acdc_test_pred',
    'psi1w3':  f'{H}/nnUNet/acdc_test_pred_psi1w3',
    'psi6w3':  f'{H}/nnUNet/acdc_test_pred_psi6w3',
    'psi7w3':  f'{H}/nnUNet/acdc_test_pred_psi7w3',
    'psi10w3': f'{H}/nnUNet/acdc_test_pred_psi10w3',
}
BTCV_OOF = {
    'anchor': f'{RES}/Dataset115_BTCV/nnUNetTrainer__nnUNetPlans__2d',
    'psi1w3': f'{RES}/Dataset115_BTCV/nnUNetTrainer_btcv_psi1w3__nnUNetPlans__2d',
    'psi6w3': f'{RES}/Dataset115_BTCV/nnUNetTrainer_btcv_psi6w3__nnUNetPlans__2d',
    'psi8w3': f'{RES}/Dataset115_BTCV/nnUNetTrainer_btcv_psi8w3__nnUNetPlans__2d',
    'w1ctrl': f'{RES}/Dataset115_BTCV/nnUNetTrainer_btcv_w1ctrl__nnUNetPlans__2d',
}
ACDC_ANCHOR_DIR = f'{RES}/Dataset116_ACDC/nnUNetTrainer_500epochs__nnUNetPlans__2d'


# ---------------- 讀取工具 ----------------
def read_summaries(files):
    out = {}
    for f in files:
        for c in json.load(open(f))['metric_per_case']:
            name = os.path.basename(c['reference_file']).replace('.nii.gz', '')
            out[name] = [c['metrics'][k]['Dice'] for k in sorted(c['metrics'], key=int)]
    return out


def load_test(d):
    f = f'{d}/summary.json'
    return read_summaries([f]) if os.path.exists(f) else None


def load_folds(d, pattern):
    files = sorted(glob.glob(f'{d}/{pattern}/validation/summary.json'))
    return read_summaries(files) if files else None


def pval(x, y):
    d = np.asarray(y) - np.asarray(x)
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


# ---------------- 比較核心 ----------------
def compare(dataset, split, models, org, rows_overall, rows_struct, rows_worst, wide):
    if 'anchor' not in models or models['anchor'] is None:
        print(f'  ⚠ {dataset} {split}：找不到 anchor，跳過')
        return
    A = models['anchor']
    ks = sorted(A)
    a_all = np.array([A[k] for k in ks], float)
    a_case_raw = np.nanmean(a_all, 1)
    k30 = int(round(len(ks) * 0.30))
    worst_idx = np.argsort(a_case_raw)[:k30]

    wide_df = pd.DataFrame({'dataset': dataset, 'split': split, 'case': ks,
                            'anchor': a_case_raw})

    for m, B in models.items():
        if m == 'anchor' or B is None:
            continue
        miss = set(ks) - set(B)
        if miss:
            print(f'  ⚠ {dataset} {split} {m}：缺 {len(miss)} 例，跳過')
            continue
        a = a_all.copy()
        b = np.array([B[k] for k in ks], float)
        both = np.isnan(a) | np.isnan(b)                  # 分母一致
        a[both] = np.nan
        b[both] = np.nan
        am, bm = np.nanmean(a, 1), np.nanmean(b, 1)
        wide_df[m] = bm

        wd = bm[worst_idx] - am[worst_idx]
        rows_overall.append(dict(
            dataset=dataset, split=split, model=m, n=len(ks),
            anchor_mean=am.mean(), model_mean=bm.mean(), delta=bm.mean() - am.mean(),
            anchor_std=am.std(ddof=1), model_std=bm.std(ddof=1),
            anchor_min=am.min(), model_min=bm.min(),
            up=int((bm > am).sum()), down=int((bm < am).sum()),
            wilcoxon_p=pval(am, bm),
            worst30_n=k30, worst30_delta=wd.mean(), worst30_up=int((wd > 0).sum()),
            worst30_ratio=(wd.mean() / (bm.mean() - am.mean())
                           if abs(bm.mean() - am.mean()) > 1e-9 else np.nan)))

        for i, o in enumerate(org):
            mask = ~np.isnan(a[:, i])
            x, y = a[mask, i], b[mask, i]
            rows_struct.append(dict(
                dataset=dataset, split=split, model=m, structure=o, n=int(mask.sum()),
                anchor_mean=x.mean(), model_mean=y.mean(), delta=y.mean() - x.mean(),
                up=int((y > x).sum()), down=int((y < x).sum()), wilcoxon_p=pval(x, y)))

        for j in worst_idx:
            rows_worst.append(dict(
                dataset=dataset, split=split, model=m, case=ks[j],
                anchor=am[j], model_score=bm[j], delta=bm[j] - am[j],
                improved=bool(bm[j] > am[j])))
    wide.append(wide_df)


# ---------------- proxy AUC ----------------
def proxy_table(dataset, csv, rows):
    if not os.path.exists(csv):
        print(f'  ⚠ 找不到 {csv}')
        return
    d = pd.read_csv(csv)
    k = int(round(len(d) * 0.30))
    thr = np.sort(d['dice'].values)[k - 1]
    low = d['dice'] <= thr                                 # 含並列
    for c in [c for c in d.columns if c.startswith('psi')]:
        x = d[c]
        if x.isna().all() or x.nunique(dropna=True) <= 1:
            rows.append(dict(dataset=dataset, proxy=c, n=len(d), low_n=int(low.sum()),
                             status='恆定或全 NaN'))
            continue
        g1, g0 = x[low].dropna(), x[~low].dropna()
        u, p = mannwhitneyu(g1, g0, alternative='two-sided')
        auc = u / (len(g1) * len(g0))
        rows.append(dict(dataset=dataset, proxy=c, n=len(d), low_n=int(low.sum()),
                         dice_threshold=thr, auc=auc, p=p,
                         direction='高=危險' if auc >= 0.5 else '低=危險（反向）',
                         low_mean=g1.mean(), rest_mean=g0.mean(),
                         status='顯著' if p < 0.05 else ('邊緣' if p < 0.10 else '不顯著')))


# ============================================================
def main():
    ov, st, wo, wide = [], [], [], []

    print('【測試集】')
    compare('BTCV', 'test (n=12, 5-fold ensemble)',
            {m: load_test(p) for m, p in BTCV_TEST.items()}, BTCV_ORG, ov, st, wo, wide)
    compare('ACDC', 'test (n=40, fold_all)',
            {m: load_test(p) for m, p in ACDC_TEST.items()}, ACDC_ORG, ov, st, wo, wide)
    pd.DataFrame(ov).to_csv(f'{OUT}/1_test_overall.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(st).to_csv(f'{OUT}/2_test_per_structure.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(wo).to_csv(f'{OUT}/3_test_worst30.csv', index=False, encoding='utf-8-sig')
    pd.concat(wide).to_csv(f'{OUT}/4_test_per_case.csv', index=False, encoding='utf-8-sig')

    print('【BTCV 5 折 OOF】')
    ov2, st2, wo2, wide2 = [], [], [], []
    compare('BTCV', 'OOF (n=18)',
            {m: load_folds(p, 'fold_[0-4]') for m, p in BTCV_OOF.items()},
            BTCV_ORG, ov2, st2, wo2, wide2)
    pd.DataFrame(ov2).to_csv(f'{OUT}/5_oof_overall.csv', index=False, encoding='utf-8-sig')

    print('【ACDC：5 折 OOF vs -f all】')
    rows = []
    for tag, pat in [('5-fold OOF', 'fold_[0-4]'), ('-f all（模型看過）', 'fold_all')]:
        S = load_folds(ACDC_ANCHOR_DIR, pat)
        if not S:
            continue
        m = np.array([np.nanmean(v) for v in S.values()])
        rows.append(dict(source=tag, n=len(m), mean=m.mean(), std=m.std(ddof=1),
                         min=m.min(), p10=np.percentile(m, 10), median=np.median(m),
                         n_below_0_90=int((m < 0.90).sum()), n_below_0_80=int((m < 0.80).sum())))
    pd.DataFrame(rows).to_csv(f'{OUT}/6_cv_vs_all.csv', index=False, encoding='utf-8-sig')

    print('【proxy 篩選】')
    pr = []
    proxy_table('BTCV (OOF, n=18)', f'{TF}/caselevel_btcv.csv', pr)
    proxy_table('ACDC (OOF, n=160)', f'{TF}/caselevel_acdc_oof.csv', pr)
    pd.DataFrame(pr).to_csv(f'{OUT}/7_proxy_selection.csv', index=False, encoding='utf-8-sig')

    print('【ACDC 舊清單 vs 新清單】')
    ol = []
    trainer = f'{H}/nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_PsiWeighted.py'
    src = open(trainer).read()
    oof = pd.read_csv(f'{TF}/caselevel_acdc_oof.csv')
    k = int(round(len(oof) * 0.30))
    for cls, csvname in re.findall(r"class (nnUNetTrainer_500epochs_psi\d+w3)\(.*?'(wcases[^']+\.csv)'",
                                   src, re.S):
        num = re.search(r'psi(\d+)w3', cls).group(1)
        col = next((c for c in oof.columns if c.startswith(f'psi{num}_')), None)
        path = f'{TF}/{csvname}'
        if col is None or not os.path.exists(path):
            continue
        old = set(pd.read_csv(path)['case'].astype(str))
        asc = (num == '10')                                # Ψ10 反向：取最低
        s = oof.sort_values(col, ascending=asc)
        thr = s[col].iloc[k - 1]
        new = set(oof[(oof[col] <= thr) if asc else (oof[col] >= thr)]['case'].astype(str))
        ol.append(dict(trainer=cls, proxy=col, list_file=csvname,
                       old_n=len(old), new_oof_n=len(new), overlap=len(old & new),
                       jaccard=len(old & new) / max(len(old | new), 1),
                       same_list=(old == new)))
    pd.DataFrame(ol).to_csv(f'{OUT}/8_acdc_list_overlap.csv', index=False, encoding='utf-8-sig')

    # ---------------- 終端摘要 ----------------
    pd.set_option('display.width', 200)
    print('\n' + '=' * 100)
    print('測試集摘要（主要結果）')
    print('=' * 100)
    t = pd.DataFrame(ov)
    if len(t):
        print(t[['dataset', 'model', 'n', 'anchor_mean', 'model_mean', 'delta', 'up', 'down',
                 'wilcoxon_p', 'worst30_n', 'worst30_delta', 'worst30_up']]
              .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nACDC：5 折 OOF vs -f all')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n最佳改善（測試集，依 delta 排序）')
    if len(t):
        print(t.sort_values('delta', ascending=False)
              [['dataset', 'model', 'delta', 'worst30_delta', 'wilcoxon_p']]
              .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print(f'\n所有 CSV 已輸出至 {OUT}')


if __name__ == '__main__':
    main()