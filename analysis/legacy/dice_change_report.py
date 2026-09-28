#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDC 測試集 Dice 變化列表：anchor(baseline) vs Ψ 加權專家
================================================================
只做一件事：把每一個 case、每一個器官的 Dice 全部列出來。

  case | 器官 | anchor | 專家 | 變化 | 變化%

器官欄位自動偵測（任何以 dice 開頭的欄，例如
dice / dice_lv / dice_rv / dice_myo / dice_LV_ED ...）。

用法:
    conda activate expertree
    cd ~/桌面/論文
    python3 dice_change_list.py                      # 預設比 psi1w3
    python3 dice_change_list.py --exp test_psi6w3
    python3 dice_change_list.py --sort delta         # 依變化量排序
"""
"""
import os
import argparse

import pandas as pd

TF = os.path.expanduser('~/桌面/論文/tree_features')


def load(split):
    p = os.path.join(TF, f'acdc_psi167_{split}.csv')
    if not os.path.exists(p):
        raise SystemExit(f'[錯誤] 找不到 {p}')
    return pd.read_csv(p)


def dice_cols(df):
    cols = [c for c in df.columns if c.lower().startswith('dice')]
    if not cols:
        raise SystemExit(f'[錯誤] 找不到任何 dice 欄位，現有欄位：{list(df.columns)}')
    return cols


def organ_name(col):
    n = col[4:].lstrip('_')
    return n.upper() if n else '整體'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='test', help='基準（anchor / baseline）的 split 名')
    ap.add_argument('--exp', default='test_psi1w3', help='加權專家的 split 名')
    ap.add_argument('--tag', default=None, help='專家組顯示名稱')
    ap.add_argument('--sort', default='case', choices=['case', 'delta'],
                    help='排序方式：case（預設）或 delta（變化量）')
    a = ap.parse_args()
    tag = a.tag or a.exp.replace('test_', '')

    A, B = load(a.base), load(a.exp)
    cols = [c for c in dice_cols(A) if c in B.columns]
    m = A.merge(B, on='case', suffixes=('_a', '_b'))
    if len(m) == 0:
        raise SystemExit('[錯誤] 兩份資料沒有共同的 case')

    # 攤平成長表：一列 = 一個 case 的一個器官
    rows = []
    for _, r in m.iterrows():
        for c in cols:
            x, y = r[c + '_a'], r[c + '_b']
            rows.append({
                'case': r['case'],
                '器官': organ_name(c),
                'anchor': x,
                tag: y,
                '變化': y - x,
                '變化%': (y - x) / x * 100 if x else float('nan'),
                '方向': '↑' if y > x else ('↓' if y < x else '='),
            })
    T = pd.DataFrame(rows)
    if a.sort == 'delta':
        T = T.sort_values('變化', ascending=False)
    else:
        T = T.sort_values(['case', '器官'])
    T = T.reset_index(drop=True)

    print(f'\nanchor({a.base}) vs {tag}({a.exp})　'
          f'{m["case"].nunique()} 例 × {len(cols)} 器官 = {len(T)} 筆\n')
    print(T.to_string(index=False, float_format=lambda x: f'{x:8.4f}'))

    # 每個器官一行小計，方便一眼看完
    print('\n各器官平均')
    s = T.groupby('器官').agg(n=('變化', 'size'),
                              anchor=('anchor', 'mean'),
                              專家=(tag, 'mean'),
                              平均變化=('變化', 'mean'),
                              升高數=('變化', lambda v: (v > 0).sum()))
    print(s.round(4).to_string())
    print(f'\n全部 {len(T)} 筆　平均變化 {T["變化"].mean():+.4f}　'
          f'升高 {(T["變化"] > 0).sum()} 筆／降低 {(T["變化"] < 0).sum()} 筆')

    out = os.path.join(TF, f'dice_list_{tag}.csv')
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        T.to_excel(w, sheet_name='全部Dice變化', index=False)
        # 另外附一張寬表：一列一個 case，器官橫向展開
        wide = T.pivot(index='case', columns='器官',
                       values=['anchor', tag, '變化'])
        wide.columns = [f'{b}_{c}' for c, b in wide.columns]
        wide.sort_index(axis=1).to_excel(w, sheet_name='寬表')
        s.to_excel(w, sheet_name='器官小計')
    print(f'\n輸出 → {out}')


if __name__ == '__main__':
    main() 

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDC 測試集 逐結構 Dice 變化報告：anchor vs Ψ 加權專家
================================================================
ACDC 前景標籤：1 = RV（右心室腔）、2 = Myo（左心室心肌）、3 = LV（左心室腔）
背景（0）不計入 Dice。

case-level 的 dice 是三個結構的算術平均，會把單一結構的崩壞稀釋掉。
本報告拆到結構層級，逐一檢視三個結構各自的變化。

資料來源為 acdc_psi167.py 產生之 xlsx 的 struct_level 工作表（每 case 三列）。

用法:
    conda activate expertree
    cd ~/桌面/論文
    python3 dice_change_by_structure.py
    python3 dice_change_by_structure.py --exp test_psi6w3
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TF = os.path.expanduser('~/桌面/論文/tree_features')
ORDER = ['RV', 'Myo', 'LV']
PSI = ['psi1_convex', 'psi6_cc', 'psi7_genus']


def load(split):
    x = os.path.join(TF, f'acdc_psi167_{split}.xlsx')
    if not os.path.exists(x):
        raise SystemExit(f'[錯誤] 找不到 {x}，請先跑 acdc_psi167.py')
    d = pd.read_excel(x, sheet_name='struct_level')
    return d[['case', 'structure', 'dice', 'voxels'] +
             [c for c in PSI if c in d.columns]]


def sec(t):
    print(f'\n{"="*80}\n{t}\n{"="*80}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='test')
    ap.add_argument('--exp', default='test_psi1w3')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    tag = a.tag or a.exp.replace('test_', '')

    A, B = load(a.base), load(a.exp)
    m = A.merge(B, on=['case', 'structure'], suffixes=('_a', '_b'))
    m['delta'] = m.dice_b - m.dice_a
    m['structure'] = pd.Categorical(m.structure, ORDER, ordered=True)
    m = m.sort_values(['structure', 'delta'], ascending=[True, False])

    n_case = m.case.nunique()
    print(f'\nanchor({a.base}) vs {tag}({a.exp})　'
          f'{n_case} 例 × 3 結構 = {len(m)} 筆')

    # ---------------------------------------------------- 各結構總覽
    sec('一、各結構總覽')
    rows = []
    for s in ORDER:
        g = m[m.structure == s]
        if not len(g):
            continue
        p = wilcoxon(g.dice_a, g.dice_b).pvalue if g.delta.abs().sum() > 0 else np.nan
        rows.append({
            '結構': s, 'n': len(g),
            'anchor': g.dice_a.mean(), tag: g.dice_b.mean(),
            '變化': g.delta.mean(),
            'anchor_sd': g.dice_a.std(), f'{tag}_sd': g.dice_b.std(),
            'anchor_min': g.dice_a.min(), f'{tag}_min': g.dice_b.min(),
            '升高': int((g.delta > 0).sum()), '降低': int((g.delta < 0).sum()),
            'p值': p,
        })
    ov = pd.DataFrame(rows)
    print(ov.round(4).to_string(index=False))
    print('\n注意 min 欄：case-level 平均會把單一結構的崩壞稀釋掉，'
          '\n此處可看出哪個結構才是真正的瓶頸。')

    # ---------------------------------------------------- 分層
    sec('二、各結構 × anchor 原始表現分層')
    m['層'] = pd.cut(m.dice_a, [-.001, .5, .75, .85, .95, 1.0],
                    labels=['<0.50 崩壞', '0.50-0.75', '0.75-0.85',
                            '0.85-0.95', '>0.95'])
    t = m.groupby(['structure', '層'], observed=True).agg(
        n=('delta', 'size'), anchor=('dice_a', 'mean'),
        專家=('dice_b', 'mean'), 平均變化=('delta', 'mean'),
        升高=('delta', lambda x: int((x > 0).sum())))
    print(t.round(4).to_string())

    # ---------------------------------------------------- 崩壞案例
    sec('三、anchor 表現最差的結構層級案例（前 15 筆）')
    w = m.nsmallest(15, 'dice_a')[
        ['case', 'structure', 'dice_a', 'dice_b', 'delta', 'voxels_a']]
    w.columns = ['case', '結構', 'anchor', tag, '變化', 'voxels']
    print(w.to_string(index=False, float_format=lambda x: f'{x:9.4f}'))

    # ---------------------------------------------------- 逐 case
    for s in ORDER:
        g = m[m.structure == s]
        if not len(g):
            continue
        sec(f'四-{s}、{s} 逐 case 明細（依變化排序）')
        d = g[['case', 'dice_a', 'dice_b', 'delta']].copy()
        d['變化%'] = d.delta / d.dice_a.replace(0, np.nan) * 100
        d['方向'] = np.where(d.delta > 0, '↑', np.where(d.delta < 0, '↓', '='))
        d.columns = ['case', 'anchor', tag, '變化', '變化%', '方向']
        print(d.to_string(index=False, float_format=lambda x: f'{x:9.4f}'))

    # ---------------------------------------------------- 輸出
    out = os.path.join(TF, f'dice_change_by_structure_{tag}.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as wr:
        ov.to_excel(wr, sheet_name='各結構總覽', index=False)
        t.to_excel(wr, sheet_name='分層分析')
        m[['case', 'structure', 'dice_a', 'dice_b', 'delta']].to_excel(
            wr, sheet_name='逐case逐結構', index=False)
        for s in ORDER:
            g = m[m.structure == s]
            if len(g):
                g[['case', 'dice_a', 'dice_b', 'delta']].to_excel(
                    wr, sheet_name=s, index=False)
    print(f'\n輸出 → {out}')


if __name__ == '__main__':
    main()