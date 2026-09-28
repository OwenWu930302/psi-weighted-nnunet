#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDC Ψ1 / Ψ6 / Ψ7 計算 → Excel
================================================================
只計算測試集上已驗證為顯著的三個 proxy（Ψ1 凸包比、Ψ6 連通元件、Ψ7 拓撲環），
沿用 psi_core.py 的算法（與 run_all_psi.py 完全一致）。

ACDC 設定（沿用 run_all_psi.py 的 load_acdc）：
    labels = {1: RV, 2: Myo, 3: LV}
    ring   = Myo          ← Ψ7 只算在心肌上，同一 case 三列共用同一個值

聚合方式（沿用 recompute_caselevel.py）：
    dice → 三結構平均
    Ψ    → nanmax（風險取最高者，不取平均，避免健康結構稀釋崩潰訊號）

輸出 Excel 兩張工作表：
    case_level    每個 case 一列，供依 Ψ 排名挑樣本
    struct_level  每個 case × 結構一列，保留 RV/Myo/LV 明細供回查

用法:
    conda activate expertree
    cd ~/桌面/論文
    python3 acdc_psi167.py --pred nnUNet/acdc_train_pred --split train
    python3 acdc_psi167.py --pred nnUNet/acdc_test_pred  --split test
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from psi_core import psi1_convex, psi6_cc, psi7_genus

ROOT = os.path.expanduser('~/桌面/論文')
OUT_DIR = os.path.join(ROOT, 'tree_features')

LABELS = {1: 'RV', 2: 'Myo', 3: 'LV'}
RING = 'Myo'                     # Ψ7 的環狀結構
PSI3 = ['psi1_convex', 'psi6_cc', 'psi7_genus']


def load_mask(path):
    """讀 nii.gz 成整數標籤陣列 (H, W, Z)。"""
    a = np.asarray(nib.load(path).dataobj)
    if a.ndim == 4:
        a = a[..., 0]
    return np.rint(a).astype(np.int16)


def load_dice(pred_dir):
    """若 pred_dir 下有 summary.json 就讀出 per-case per-structure Dice。"""
    sf = os.path.join(pred_dir, 'summary.json')
    if not os.path.exists(sf):
        return {}
    d = {}
    for c in json.load(open(sf))['metric_per_case']:
        n = os.path.basename(c['reference_file']).replace('.nii.gz', '')
        d[n] = {LABELS[k]: c['metrics'][str(k)]['Dice'] for k in LABELS}
    return d


def compute_struct_level(pred_dir):
    """逐 case 逐結構計算 Ψ1/6/7。"""
    files = sorted(glob.glob(os.path.join(pred_dir, '*.nii.gz')))
    if not files:
        raise SystemExit(f'[錯誤] {pred_dir} 裡沒有 .nii.gz')

    dice = load_dice(pred_dir)
    print(f'讀到 {len(files)} 個預測檔'
          f'{"，含 Dice" if dice else "，無 summary.json（Dice 欄留空）"}')

    rows = []
    for i, f in enumerate(files, 1):
        case = os.path.basename(f).replace('.nii.gz', '')
        a = load_mask(f)
        msk = {nm: (a == k) for k, nm in LABELS.items()}

        # Ψ7 只定義在心肌環上，同一 case 的三列共用
        p7 = psi7_genus(msk.get(RING))

        for nm, m in msk.items():
            rows.append({
                'case': case,
                'structure': nm,
                'psi1_convex': psi1_convex(m),
                'psi6_cc': psi6_cc(m),
                'psi7_genus': p7,
                'voxels': int(m.sum()),
                'dice': dice.get(case, {}).get(nm, np.nan),
            })

        if i % 20 == 0 or i == len(files):
            print(f'  {i}/{len(files)}', flush=True)

    return pd.DataFrame(rows)


def to_case_level(sd):
    """
    聚合到 case 層級：dice 取平均，Ψ 取 nanmax。
    n_nan 記錄該 case 有幾個 (結構 × Ψ) 無法計算，本身即為失敗訊號。
    """
    recs = []
    for case, g in sd.groupby('case', sort=True):
        r = {'case': case, 'n_struct': len(g)}
        r['dice'] = g.dice.mean() if g.dice.notna().any() else np.nan

        n_nan = 0
        for c in PSI3:
            v = g[c].values.astype(float)
            ok = ~np.isnan(v)
            r[c] = np.nanmax(v) if ok.any() else np.nan
            n_nan += int((~ok).sum())
        r['n_nan'] = n_nan

        # 額外旗標：是否有結構被完全預測不出來（Ψ6 == -1，nanmax 會蓋掉）
        r['n_empty'] = int((g.psi6_cc == -1).sum())
        recs.append(r)

    cols = ['case', 'dice'] + PSI3 + ['n_struct', 'n_nan', 'n_empty']
    return pd.DataFrame(recs)[cols]


def summarise(cd, split):
    print(f'\n{"="*70}\nACDC {split} — case 層級摘要（n = {len(cd)}）\n{"="*70}')
    for c in ['dice'] + PSI3:
        v = cd[c].dropna()
        if not len(v):
            print(f'{c:<15} 全空')
            continue
        print(f'{c:<15} n={len(v):<4} mean={v.mean():>8.4f} sd={v.std():>8.4f} '
              f'min={v.min():>8.4f} med={v.median():>8.4f} max={v.max():>8.4f}')

    if cd.n_empty.sum():
        print(f'\n有 {int((cd.n_empty > 0).sum())} 個 case 至少一個結構完全未被預測出來')

    if cd.dice.notna().any():
        v = cd.dice.dropna()
        print(f'\nDice 最低 20% 門檻 ≤ {v.quantile(.20):.4f}'
              f'　最低 30% 門檻 ≤ {v.quantile(.30):.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='預測遮罩資料夾')
    ap.add_argument('--split', default='train', help='輸出檔名後綴')
    ap.add_argument('--out', default=None, help='自訂輸出 xlsx 路徑')
    a = ap.parse_args()

    pred_dir = a.pred if os.path.isabs(a.pred) else os.path.join(ROOT, a.pred)

    sd = compute_struct_level(pred_dir)
    cd = to_case_level(sd)
    summarise(cd, a.split)

    os.makedirs(OUT_DIR, exist_ok=True)
    xlsx = a.out or os.path.join(OUT_DIR, f'acdc_psi167_{a.split}.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
        cd.to_excel(w, sheet_name='case_level', index=False)
        sd.to_excel(w, sheet_name='struct_level', index=False)

    # 一併存 csv，方便接到既有的 proxy_rank_check.py 流程
    csv = os.path.join(OUT_DIR, f'acdc_psi167_{a.split}.csv')
    cd.to_csv(csv, index=False)

    print(f'\n輸出 Excel → {xlsx}')
    print(f'　　　 csv → {csv}')


if __name__ == '__main__':
    main()