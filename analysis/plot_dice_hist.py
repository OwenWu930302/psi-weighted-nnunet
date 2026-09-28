#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDC 測試集 Dice 分布直方圖（每個模型一張，RV / Myo / LV 三聯）
================================================================
沿用 anchor 既有圖的版面：一列三欄，每欄一個結構，
標題含 n 與 sd，圖內標示 mean（紅虛線）與 median（黑點線）。

另輸出一張四模型疊圖，方便直接比較分布位移。

輸出（PNG，300 dpi）：
    figs/dice_hist_anchor.png
    figs/dice_hist_psi1w3.png
    figs/dice_hist_psi6w3.png
    figs/dice_hist_psi7w3.png
    figs/dice_hist_ALL.png          四模型疊圖

用法:
    conda activate expertree
    cd ~/桌面/論文
    python3 plot_dice_hist.py
    python3 plot_dice_hist.py --bins 40 --xmin 0.4
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TF = os.path.expanduser('~/桌面/論文/tree_features')
FIG = os.path.expanduser('~/桌面/論文/figs')
ORDER = ['RV', 'Myo', 'LV']
TITLE = {'test': 'ACDC (nnU-Net)  anchor',
         'test_psi1w3': r'ACDC (nnU-Net)  $\Psi$1 expert (convex, x3)',
         'test_psi6w3': r'ACDC (nnU-Net)  $\Psi$6 expert (components, x3)',
         'test_psi7w3': r'ACDC (nnU-Net)  $\Psi$7 expert (genus, x3)'}


def load_struct(split):
    """逐結構 Dice：優先 csv，沒有就從 xlsx 的 struct_level 轉出。"""
    c = os.path.join(TF, f'acdc_psi167_{split}_struct.csv')
    if os.path.exists(c):
        return pd.read_csv(c)
    x = os.path.join(TF, f'acdc_psi167_{split}.xlsx')
    if os.path.exists(x):
        d = pd.read_excel(x, sheet_name='struct_level')
        d.to_csv(c, index=False)
        return d
    return None


def find_splits():
    out = ['test']
    for f in sorted(glob.glob(os.path.join(TF, 'acdc_psi167_test_*.csv'))):
        n = os.path.basename(f)[len('acdc_psi167_'):-len('.csv')]
        if not n.endswith('_struct'):
            out.append(n)
    return out


def one_figure(d, split, bins, xmin, xmax):
    """單一模型的三聯直方圖。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    edges = np.linspace(xmin, xmax, bins + 1)

    for ax, s in zip(axes, ORDER):
        v = d.loc[d.structure == s, 'dice'].dropna().values
        if not len(v):
            ax.set_visible(False)
            continue
        ax.hist(v, bins=edges, color='#6b8fc4', edgecolor='white', linewidth=.4)
        ax.axvline(v.mean(), color='red', ls='--', lw=1.2,
                   label=f'mean {v.mean():.3f}')
        ax.axvline(np.median(v), color='black', ls=':', lw=1.2,
                   label=f'med {np.median(v):.3f}')
        ax.set_title(f'{s}  (n={len(v)}, sd={v.std():.3f})', fontsize=11)
        ax.set_xlabel('Dice')
        ax.set_ylabel('count')
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=8, loc='upper left', framealpha=.9)

    fig.suptitle(TITLE.get(split, split), fontsize=13, y=1.02)
    fig.tight_layout()
    tag = split.replace('test_', '').replace('test', 'anchor')
    p = os.path.join(FIG, f'dice_hist_{tag}.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return p


def overlay(data, bins, xmin, xmax):
    """四模型疊在同一張圖上，逐結構比較分布位移。"""
    colors = {'test': '#444444', 'test_psi1w3': '#d62728',
              'test_psi6w3': '#2ca02c', 'test_psi7w3': '#1f77b4'}
    label = {'test': 'anchor', 'test_psi1w3': 'Psi1 x3',
             'test_psi6w3': 'Psi6 x3', 'test_psi7w3': 'Psi7 x3'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    edges = np.linspace(xmin, xmax, bins + 1)

    for ax, s in zip(axes, ORDER):
        for split, d in data.items():
            v = d.loc[d.structure == s, 'dice'].dropna().values
            if not len(v):
                continue
            ax.hist(v, bins=edges, histtype='step', linewidth=1.6,
                    color=colors.get(split), label=f'{label.get(split, split)}'
                    f'  μ={v.mean():.3f}')
        ax.set_title(s, fontsize=11)
        ax.set_xlabel('Dice')
        ax.set_ylabel('count')
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=.9)

    fig.suptitle('ACDC test set — Dice distribution by model', fontsize=13, y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG, 'dice_hist_ALL.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins', type=int, default=50)
    ap.add_argument('--xmin', type=float, default=0.0)
    ap.add_argument('--xmax', type=float, default=1.0)
    a = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    data, made = {}, []
    for split in find_splits():
        d = load_struct(split)
        if d is None:
            print(f'[略過] 找不到 {split} 的逐結構資料')
            continue
        data[split] = d
        p = one_figure(d, split, a.bins, a.xmin, a.xmax)
        made.append(p)
        line = '  '.join(
            f'{s} {d.loc[d.structure==s,"dice"].mean():.4f}' for s in ORDER)
        print(f'{split:<14} {line}')

    if len(data) > 1:
        made.append(overlay(data, a.bins, a.xmin, a.xmax))

    print('\n輸出：')
    for p in made:
        print(' ', p)


if __name__ == '__main__':
    main()