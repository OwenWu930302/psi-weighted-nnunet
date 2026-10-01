#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五資料集 × 十個 Ψ — 病例層級 + 相對低分組口徑 重算

============ 三項方法學決定 ============

【一、分析單位改為「病例」，不拆結構】
    先前以「結構」為單位（如 REFUGE 的 OD/OC 各算一筆）會產生結構混淆：
    REFUGE 88 個失敗中 87 個是 OC（OC 失敗率 21.75%，OD 僅 0.25%），
    導致任何能區分 OD/OC 的量都會被誤判為失敗偵測器。
    改以病例為單位後，該混淆自動消失。
        REFUGE  病例分數 = (OD_dice + OC_dice) / 2      → 400 張影像
        ACDC    病例分數 = (RV + Myo + LV) / 3          → 40 個 case
        BTCV    病例分數 = 8 個器官平均                  → 18 個 case
        CAMUS   病例分數 = 10 影格平均                   → 200 段影片
        BUSI    單一結構，直接使用                       → 80 張影像

【二、失敗組改用相對門檻（最低 20% / 30%）】
    δ = 0.80 的絕對門檻使 BUSI 零失敗、CAMUS 僅 2 段失敗，無法進行統計檢定。
    相對門檻確保每個資料集都有足夠的低分樣本。
    註：低分組不等同「臨床失敗」，措辭應為「相對低分組」。
        δ = 0.80 的結果一併保留作為對照。

【三、Ψ 聚合到病例層級採「取最大值」】
    依論文 §III-A：Ψt 為 scalar risk value，數值越高代表退化越嚴重。
    故病例風險 = 各結構中風險最高者，取 max 而非平均。
    取平均會以健康結構稀釋單一結構的崩潰，例如 BTCV 四個 Dice=0 的膽囊
    Ψ8 為 0.83，平均掉其餘七個正常器官後僅剩約 0.13，訊號被抹除。
    實務上亦然：任一結構失敗即應送人工複核。
    例外：BTCV 的 Ψ10 實測為反向（AUC 0.427），與風險定義相悖，
    故門檻搜尋仍同時嘗試 > 與 < 兩個方向。

【NaN 的處理】
    NaN 分三種成因，皆非計算錯誤：
      (a) 結構層級：該資料集無此解剖結構，整欄皆 NaN
      (b) 樣本層級：模型完全未預測出該結構，遮罩為空，Ψ 的分母為零
      (c) 數值穩定性：結構面積過小（< 10 像素），凸包等幾何量無意義
    聚合時以 np.nanmax 忽略 NaN；若某病例全部結構皆 NaN 則該病例為 NaN 並排除。
    另新增 n_nan 欄位（該病例有幾個結構無法計算）——成因 (b) 本身即為最嚴重的
    失敗訊號，可作為獨立的零成本偵測器。

用法:
    conda activate expertree
    source <倉庫>/env.sh
    python3 recompute_caselevel.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

TF   = os.path.join(os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'tree_features')
PSI  = ['psi1_convex','psi2_shadow','psi3_inclusion','psi4_adjacency','psi5_exclusion',
        'psi6_cc','psi7_genus','psi8_rotation','psi9_nesting','psi10_centroid']
QUANTILES = [0.20, 0.30]
DELTA = 0.80


def load(name):
    """讀入並合併後續補跑的 Ψ 欄位。"""
    d = pd.read_csv(f'{TF}/all_psi_{name}.csv')
    def merge(f, cols, keys):
        nonlocal d
        p = f'{TF}/{f}'
        if not os.path.exists(p):
            print(f'  [略過] 找不到 {f}'); return
        e = pd.read_csv(p)
        use = [c for c in cols if c in e.columns]
        d = d.drop(columns=[c for c in use if c in d.columns]).merge(
            e[keys + use], on=keys, how='left')
        print(f'  [合併] {f} → {use}')
    if name == 'busi':
        p = f'{TF}/psi2_psi8_busi.csv'
        if os.path.exists(p):
            e = pd.read_csv(p); e['case'] = e['case'].astype(str)
            d['case'] = d['case'].astype(str)
            d = d.drop(columns=[c for c in ['psi2_shadow','psi8_rotation'] if c in d.columns]).merge(
                e[['case','psi2_shadow','psi8_rotation']], on='case', how='left')
            print('  [合併] psi2_psi8_busi.csv')
    elif name == 'acdc':
        merge('psi8_acdc.csv', ['psi8_rotation'], ['case','structure'])
    elif name == 'btcv':
        merge('psi8_btcv.csv', ['psi8_rotation'], ['case','structure'])
    elif name == 'camus':
        p = f'{TF}/psi2_psi8_camus.csv'
        if os.path.exists(p):
            e = pd.read_csv(p)
            e['case'] = e.patient + '_' + e.view
            d = d.drop(columns=[c for c in ['psi2_shadow','psi8_rotation'] if c in d.columns]).merge(
                e[['case','frame','psi2_shadow','psi8_rotation']], on=['case','frame'], how='left')
            print('  [合併] psi2_psi8_camus.csv')
    elif name == 'refuge':
        p = f'{TF}/psi2_psi8_refuge.csv'
        if os.path.exists(p):
            e = pd.read_csv(p)
            d = d.drop(columns=[c for c in ['psi2_shadow','psi8_rotation'] if c in d.columns]).merge(
                e[['case','psi2_shadow','psi8_rotation']], on='case', how='left')
            print('  [合併] psi2_psi8_refuge.csv（僅 OC，OD 為 NaN）')
    return d


def to_case(d):
    """
    聚合到病例層級：dice 取平均，Ψ 取 nanmax。

    n_nan 僅計入「該資料集整體有部分有效值」的欄位，
    以排除整欄 NaN 的結構性缺失（如 BTCV 無心臟故 Ψ3 全欄 NaN）。
    如此 n_nan 才真正代表「模型未預測出該結構」的樣本層級失敗訊號。
    """
    live = [c for c in PSI if c in d.columns and not d[c].isna().all()]
    recs = []
    for case, g in d.groupby('case', sort=True):
        r = {'case': case, 'dice': g.dice.mean(),
             'n_struct': len(g), 'n_nan': 0}
        for c in PSI:
            if c not in g.columns:
                r[c] = np.nan; continue
            v = g[c].values.astype(float)
            ok = ~np.isnan(v)
            r[c] = np.nanmax(v) if ok.any() else np.nan
            if c in live:
                r['n_nan'] += int((~ok).sum())
        recs.append(r)
    return pd.DataFrame(recs)


def best_f1(v, y):
    b, t = 0.0, None
    for d in ('>', '<'):
        for x in np.unique(np.percentile(v, np.linspace(1, 99, 99))):
            m = (v > x) if d == '>' else (v < x)
            if m.sum() in (0, len(v)): continue
            s = f1_score(y, m.astype(int), zero_division=0)
            if s > b: b, t = s, (d, x)
    return b, t


def cv_f1(v, y):
    if y.sum() < 5: return np.nan
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pr = np.zeros(len(y), int)
    for tr, te in skf.split(v.reshape(-1, 1), y):
        if y[tr].sum() == 0: continue
        _, t = best_f1(v[tr], y[tr])
        if t: pr[te] = ((v[te] > t[1]) if t[0] == '>' else (v[te] < t[1])).astype(int)
    return f1_score(y, pr, zero_division=0)


def report(name, cd, y, label):
    n, npos = len(cd), int(y.sum())
    p = npos / n
    base = 2 * p / (1 + p) if npos else 0.0
    print(f'\n{"="*100}')
    print(f'{name} — {label}')
    print(f'病例數 {n}　低分組 {npos} ({p:.1%})　基準線 F1 = {base:.4f}')
    print('='*100)
    print(f'{"特徵":<16}{"AUC":>8}{"p 值":>11}{"低分均":>11}{"其餘均":>11}{"F1":>8}{"CV F1":>8}{"ρ":>9}  狀態')
    for c in PSI + ['n_nan']:
        v = cd[c].values.astype(float)
        m = ~np.isnan(v)
        if m.sum() == 0:
            print(f'{c:<16}{"—":>8}{"—":>11}{"—":>11}{"—":>11}{"—":>8}{"—":>8}{"—":>9}  全 NaN')
            continue
        vv, yy = v[m], y[m]
        # 全距小於 1e-6 視為恆定：ACDC 的 Ψ3／Ψ9 全距僅 7.3e-10，
        # 屬浮點運算誤差而非真實訊號，若不擋掉會算出無意義的 AUC 0.092
        if np.ptp(vv) < 1e-6:
            print(f'{c:<16}{"—":>8}{"—":>11}{vv[0]:>11.4f}{vv[0]:>11.4f}'
                  f'{"—":>8}{"—":>8}{"—":>9}  恆定，無變異')
            continue
        if yy.sum() == 0 or yy.sum() == len(yy):
            print(f'{c:<16}  子集無低分組，無法檢定'); continue
        auc = roc_auc_score(yy, vv)
        pv = mannwhitneyu(vv[yy == 1], vv[yy == 0])[1]
        rho = spearmanr(vv, cd.dice.values[m])[0]
        f1, _ = best_f1(vv, yy)
        cv = cv_f1(vv, yy)
        sig = '顯著' if pv < 0.05 else ('邊緣' if pv < 0.10 else '')
        cvs = f'{cv:.4f}' if not np.isnan(cv) else '—'
        print(f'{c:<16}{auc:>8.3f}{pv:>11.2e}{vv[yy==1].mean():>11.4f}{vv[yy==0].mean():>11.4f}'
              f'{f1:>8.4f}{cvs:>8}{rho:>+9.3f}  {sig}')


def main():
    for name, disp in [('busi','BUSI'), ('camus','CAMUS'), ('acdc','ACDC'),
                       ('acdc_oof','ACDC_OOF'),
                       ('btcv','BTCV'), ('refuge','REFUGE')]:
        f = f'{TF}/all_psi_{name}.csv'
        if not os.path.exists(f):
            print(f'\n[跳過] 找不到 {f}'); continue
        print(f'\n\n{"#"*100}\n# {disp}\n{"#"*100}')
        d = load(name)
        cd = to_case(d)
        cd = cd[~cd.dice.isna()].reset_index(drop=True)
        print(f'  病例層級：{len(d)} 筆結構 → {len(cd)} 個病例'
              f'（每病例 {cd.n_struct.iloc[0]} 個結構）')
        print(f'  病例分數 dice: min={cd.dice.min():.4f} max={cd.dice.max():.4f} '
              f'mean={cd.dice.mean():.4f}')
        cd.to_csv(f'{TF}/caselevel_{name}.csv', index=False)

        # 對照：絕對門檻
        y_abs = (cd.dice < DELTA).astype(int).values
        if y_abs.sum() > 0:
            report(disp, cd, y_abs, f'對照口徑：Dice < {DELTA}')
        else:
            print(f'\n[對照口徑 Dice < {DELTA}] 零低分樣本，無法檢定')

        # 相對門檻
        for q in QUANTILES:
            thr = cd.dice.quantile(q)
            y = (cd.dice <= thr).astype(int).values
            report(disp, cd, y,
                   f'相對口徑：最低 {int(q*100)}%（Dice ≤ {thr:.4f}，含並列）')

    print(f'\n\n病例層級資料已輸出至 {TF}/caselevel_<dataset>.csv')


if __name__ == '__main__':
    main()