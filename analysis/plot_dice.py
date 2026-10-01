"""
Dice 分布圖（取代 plot_dice_hist.py）
============================================================
直接讀 dice_tables.py 的輸出，不依賴其他中間檔。
兩個資料集、所有模型、測試集。

輸出（PNG，300 dpi）：$PSI_ROOT/figs/
  dice_hist_<資料集>_<模型>.png   每個模型一張，一列一個結構
  dice_overlay_<資料集>.png       所有模型的病例平均 Dice 分布疊圖
  dice_paired_<資料集>.png        每個專家對 baseline 的逐例差異（配對）

用法：
  python3 dice_tables.py        # 先產生 CSV
  python3 plot_dice.py
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H = os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = f'{H}/tree_features/dice_tables'
OUT = f'{H}/figs'
os.makedirs(OUT, exist_ok=True)
ORG = {'BTCV': ['Aorta', 'Gallbl', 'KidL', 'KidR', 'Liver', 'Pancr', 'Spleen', 'Stomach'],
       'ACDC': ['RV', 'Myo', 'LV']}


def load(ds):
    out = {}
    for f in sorted(glob.glob(f'{SRC}/dice_{ds}_test_*.csv')):
        m = os.path.basename(f)[len(f'dice_{ds}_test_'):-4]
        out[m] = pd.read_csv(f).set_index('case')
    def natural(k):                       # psi1 < psi6 < psi10，而非字串排序
        import re
        n = re.search(r'\d+', k)
        return (0, int(n.group())) if (k.startswith('psi') and n) else (1, k)
    order = ['baseline'] + sorted((k for k in out if k != 'baseline'), key=natural)
    return {k: out[k] for k in order if k in out}


def hist_per_model(ds, D):
    org = ORG[ds]
    for m, d in D.items():
        n = len(org)
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
        for i, o in enumerate(org):
            ax = axes[i // cols][i % cols]
            v = d[o].dropna().values
            ax.hist(v, bins=12, range=(0, 1) if v.min() < 0.5 else None,
                    color='#4C72B0', edgecolor='white')
            ax.axvline(v.mean(), color='red', ls='--', lw=1.2, label=f'mean {v.mean():.3f}')
            ax.axvline(np.median(v), color='black', ls=':', lw=1.2, label=f'median {np.median(v):.3f}')
            ax.set_title(f'{o}  (n={len(v)}, sd={v.std(ddof=1):.3f})', fontsize=9)
            ax.legend(fontsize=7)
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis('off')
        fig.suptitle(f'{ds} test — {m}', fontsize=11)
        fig.tight_layout()
        fig.savefig(f'{OUT}/dice_hist_{ds}_{m}.png', dpi=300)
        plt.close(fig)


def overlay(ds, D):
    fig, ax = plt.subplots(figsize=(6, 4))
    all_v = np.concatenate([d['mean'].values for d in D.values()])
    bins = np.linspace(all_v.min(), all_v.max(), 15)
    for m, d in D.items():
        ax.hist(d['mean'], bins=bins, histtype='step', lw=1.8 if m == 'baseline' else 1.2,
                label=f"{m} ({d['mean'].mean():.4f})")
    ax.set_xlabel('case mean Dice')
    ax.set_ylabel('count')
    ax.set_title(f'{ds} test — case mean Dice distribution')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/dice_overlay_{ds}.png', dpi=300)
    plt.close(fig)


def paired(ds, D):
    base = D['baseline']['mean']
    experts = [m for m in D if m != 'baseline']
    fig, ax = plt.subplots(figsize=(1.4 * len(experts) + 2, 4))
    for i, m in enumerate(experts):
        delta = (D[m]['mean'] - base).dropna()
        x = np.full(len(delta), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(delta))
        ax.scatter(x, delta, s=14, alpha=0.7,
                   c=np.where(delta > 0, '#2a9d8f', '#e76f51'))
        ax.hlines(delta.mean(), i - 0.3, i + 0.3, color='black', lw=2)
        ax.text(i, ax.get_ylim()[1] if i == 0 else delta.max(), f'{delta.mean():+.4f}',
                ha='center', va='bottom', fontsize=8)
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(experts)
    ax.set_ylabel('expert − baseline (case mean Dice)')
    ax.set_title(f'{ds} test — paired difference vs baseline')
    fig.tight_layout()
    fig.savefig(f'{OUT}/dice_paired_{ds}.png', dpi=300)
    plt.close(fig)


def main():
    for ds in ['BTCV', 'ACDC']:
        D = load(ds)
        if 'baseline' not in D:
            print(f'⚠ 找不到 {ds} baseline，請先執行 dice_tables.py')
            continue
        hist_per_model(ds, D)
        overlay(ds, D)
        paired(ds, D)
        print(f'{ds}：{len(D)} 個模型 → {", ".join(D)}')
    print(f'圖檔輸出至 {OUT}')


if __name__ == '__main__':
    main()