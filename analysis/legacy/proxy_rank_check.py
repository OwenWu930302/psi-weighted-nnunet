#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失敗組定義有兩種（Dice 最低 20% / 30%），
每種失敗組定義下，各顯著 proxy 再各自排序取前 20% / 30%，
故每個顯著 proxy 產生 2(失敗組門檻) × 2(proxy取樣) = 4 組結果。

用法:
    conda activate expertree
    cd ~/桌面/論文
    python3 proxy_topN.py
"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

TF = 'tree_features'
PSI = ['psi1_convex','psi2_shadow','psi3_inclusion','psi4_adjacency','psi5_exclusion',
       'psi6_cc','psi7_genus','psi8_rotation','psi9_nesting','psi10_centroid']

DICE_Q_LIST = (0.20, 0.30)   # 失敗組定義：Dice 最低的百分比
TOP_LIST    = (0.20, 0.30)   # proxy 排序後各取前多少百分比


def main():
    for name in ['busi', 'camus', 'acdc', 'btcv', 'refuge']:
        d = pd.read_csv(f'{TF}/caselevel_{name}.csv')
        n_total = len(d)
        print(f'\n{"="*105}')
        print(f'{name.upper()}　總病例 {n_total}')
        print(f'{"="*105}')

        for dq in DICE_Q_LIST:
            dice_thr = d.dice.quantile(dq)
            true_fail = (d.dice <= dice_thr)
            n_fail = int(true_fail.sum())

            print(f'\n--- 失敗組定義：Dice ≤ {dice_thr:.4f}（最低 {int(dq*100)}%），'
                  f'共 {n_fail} 例；正常組 {n_total - n_fail} 例 ---')
            print(f'{"proxy":<16}{"proxy取前":>10}{"抓到總數":>9}{"失敗組TP":>10}'
                  f'{"正常組FP":>10}{"漏掉FN":>9}{"F1":>9}')

            for c in PSI:
                v = d[c].values.astype(float)
                m = ~np.isnan(v)
                if m.sum() == 0 or np.ptp(v[m]) < 1e-6:
                    continue
                yy = true_fail.values[m]
                if yy.sum() == 0:
                    continue

                auc = roc_auc_score(yy, v[m])
                pv = mannwhitneyu(v[m][yy == 1], v[m][yy == 0])[1]
                if pv >= 0.05:
                    continue

                direction = 'high' if auc >= 0.5 else 'low'
                sub = d.loc[m]

                for top in TOP_LIST:
                    n_pick = int(round(n_total * top))
                    picked = (sub.nlargest(n_pick, c) if direction == 'high'
                              else sub.nsmallest(n_pick, c))

                    tp = int((picked.dice <= dice_thr).sum())
                    fp = len(picked) - tp
                    fn = n_fail - tp

                    prec = tp / (tp + fp) if (tp + fp) else 0
                    rec = tp / (tp + fn) if (tp + fn) else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

                    print(f'{c:<16}{int(top*100):>9}%{len(picked):>9}{tp:>10}{fp:>10}'
                          f'{fn:>9}{f1:>9.4f}')


if __name__ == '__main__':
    main()