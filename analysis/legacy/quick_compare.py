#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDC 專家模型快速比較（僅輸出 CSV）
================================================================
只印三件事：
  1. anchor 最差 30% 的 case，改變前後與是否提升
  2. 各結構（RV / Myo / LV）與整體的 Dice 平均變化
  3. 整體 Dice 的 mean 與 std 變化

註：ACDC 有 4 個標籤，但 label 0 是背景，不計入 Dice，
    故實際評分結構為 RV / Myo / LV 三個。

用法:
    python3 quick_compare.py                              # 比 psi1w3
    python3 quick_compare.py --exp test_psi6w3
    python3 quick_compare.py --exp test_psi1w3 test_psi6w3 test_psi7w3   # 多個一起比
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TF = os.path.expanduser('~/桌面/論文/tree_features')
ORDER = ['RV', 'Myo', 'LV']
WORST_Q = 0.30


def load_case(s):
    p = f'{TF}/acdc_psi167_{s}.csv'
    if not os.path.exists(p):
        raise SystemExit(f'[錯誤] 找不到 {p}')
    return pd.read_csv(p)[['case', 'dice']]


def load_struct(s):
    x = f'{TF}/acdc_psi167_{s}.xlsx'
    if not os.path.exists(x):
        return None
    d = pd.read_excel(x, sheet_name='struct_level')
    return d[['case', 'structure', 'dice']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='test')
    ap.add_argument('--exp', nargs='+', default=['test_psi1w3'])
    a = ap.parse_args()

    A = load_case(a.base)
    SA = load_struct(a.base)
    thr = A.dice.quantile(WORST_Q)
    worst = set(A[A.dice <= thr].case)

    overall, per_struct, worst_rows = [], [], []

    for e in a.exp:
        tag = e.replace('test_', '')
        B = load_case(e)
        m = A.merge(B, on='case', suffixes=('_a', '_b'))
        d = m.dice_b - m.dice_a

        # --- 整體
        overall.append({
            '模型': tag, 'n': len(m),
            'anchor_mean': m.dice_a.mean(), '專家_mean': m.dice_b.mean(),
            'mean變化': d.mean(),
            'anchor_std': m.dice_a.std(), '專家_std': m.dice_b.std(),
            'std變化': m.dice_b.std() - m.dice_a.std(),
            'anchor_min': m.dice_a.min(), '專家_min': m.dice_b.min(),
            '升高數': int((d > 0).sum()), '降低數': int((d < 0).sum()),
            'p值': wilcoxon(m.dice_a, m.dice_b).pvalue,
        })

        # --- 各結構
        SB = load_struct(e)
        if SA is not None and SB is not None:
            sm = SA.merge(SB, on=['case', 'structure'], suffixes=('_a', '_b'))
            for s in ORDER:
                g = sm[sm.structure == s]
                if not len(g):
                    continue
                gd = g.dice_b - g.dice_a
                per_struct.append({
                    '模型': tag, '結構': s, 'n': len(g),
                    'anchor_mean': g.dice_a.mean(), '專家_mean': g.dice_b.mean(),
                    'mean變化': gd.mean(),
                    'anchor_std': g.dice_a.std(), '專家_std': g.dice_b.std(),
                    'anchor_min': g.dice_a.min(), '專家_min': g.dice_b.min(),
                    '升高數': int((gd > 0).sum()), '降低數': int((gd < 0).sum()),
                    'p值': wilcoxon(g.dice_a, g.dice_b).pvalue,
                })
        per_struct.append({
            '模型': tag, '結構': '整體', 'n': len(m),
            'anchor_mean': m.dice_a.mean(), '專家_mean': m.dice_b.mean(),
            'mean變化': d.mean(),
            'anchor_std': m.dice_a.std(), '專家_std': m.dice_b.std(),
            'anchor_min': m.dice_a.min(), '專家_min': m.dice_b.min(),
            '升高數': int((d > 0).sum()), '降低數': int((d < 0).sum()),
            'p值': wilcoxon(m.dice_a, m.dice_b).pvalue,
        })

        # --- 最差 30%
        w = m[m.case.isin(worst)].copy()
        w['變化'] = w.dice_b - w.dice_a
        w['是否提升'] = np.where(w.變化 > 0, '提升', np.where(w.變化 < 0, '下降', '持平'))
        w['模型'] = tag
        worst_rows.append(w[['模型', 'case', 'dice_a', 'dice_b', '變化', '是否提升']])

    ov = pd.DataFrame(overall)
    ps = pd.DataFrame(per_struct)
    wo = pd.concat(worst_rows).rename(columns={'dice_a': 'anchor', 'dice_b': '專家'})
    wo = wo.sort_values(['模型', 'anchor'])

    # ================================================== 輸出
    print(f'\n【1】anchor 最差 {int(WORST_Q*100)}%（Dice ≤ {thr:.4f}，共 {len(worst)} 例）')
    for tag, g in wo.groupby('模型', sort=False):
        up = (g.變化 > 0).sum()
        print(f'\n--- {tag} ---　提升 {up}／{len(g)} 例　'
              f'平均 {g.變化.mean():+.4f}')
        print(g[['case', 'anchor', '專家', '變化', '是否提升']].to_string(
            index=False, float_format=lambda x: f'{x:8.4f}'))

    print(f'\n\n【2】各結構 Dice 平均（背景 label 0 不計分）')
    print(ps[['模型', '結構', 'anchor_mean', '專家_mean', 'mean變化',
              '升高數', '降低數', 'p值']].to_string(
        index=False, float_format=lambda x: f'{x:9.4f}'))

    print(f'\n\n【3】整體 Dice mean 與 std')
    print(ov[['模型', 'anchor_mean', '專家_mean', 'mean變化',
              'anchor_std', '專家_std', 'std變化',
              'anchor_min', '專家_min', 'p值']].to_string(
        index=False, float_format=lambda x: f'{x:9.4f}'))

    # ================================================== CSV
    n = '_'.join(t.replace('test_', '') for t in a.exp)
    f1 = f'{TF}/cmp_worst30_{n}.csv'
    f2 = f'{TF}/cmp_structure_{n}.csv'
    f3 = f'{TF}/cmp_overall_{n}.csv'
    wo.to_csv(f1, index=False, encoding='utf-8-sig')
    ps.to_csv(f2, index=False, encoding='utf-8-sig')
    ov.to_csv(f3, index=False, encoding='utf-8-sig')
    print(f'\nCSV 輸出：\n  {f1}\n  {f2}\n  {f3}')


if __name__ == '__main__':
    main()