"""
Dice 數據表輸出
============================================================
輸出目錄：~/桌面/論文/tree_features/dice_tables/

一、每個模型、每位病人的 Dice（baseline 與專家都有）
    dice_<資料集>_<資料來源>_<模型>.csv
    欄位：case, 各結構 Dice, mean, weighted（此病例是否在該專家的加權清單）

    資料來源：
      test      測試集（BTCV 12 例、ACDC 40 例）—— 皆未參與訓練，也不會被加權
      trainOOF  BTCV 訓練集 5 折 OOF（18 例）
      trainALL  ACDC 訓練集 fold_all 預測（160 例，模型看過這些影像）
      trainOOF  ACDC anchor 訓練集 5 折 OOF（160 例，僅 anchor 有）

二、加權病人對照表（每個專家一個區塊，baseline 不單獨列）
    weighted_patients_<資料集>.csv
    每個專家區塊內：先列加權病人，再列未加權病人；各組依 case 排序
    欄位：expert, case, weighted, anchor_mean, expert_mean, delta, 各結構 anchor/expert

注意：加權病人都是「訓練集」病人，測試集沒有任何加權病人。
      因此加權對照表使用訓練集預測（BTCV：OOF；ACDC：fold_all）。

用法：
  conda activate expertree
  cd ~/桌面/論文
  python3 dice_tables.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

H = os.path.expanduser('~/桌面/論文')
RES = f'{H}/nnUNet_data/nnUNet_results'
TF = f'{H}/tree_features'
OUT = f'{TF}/dice_tables'
os.makedirs(OUT, exist_ok=True)
TRAINER_PY = (f'{H}/nnUNet/nnunetv2/training/nnUNetTrainer/variants/'
              'training_length/nnUNetTrainer_PsiWeighted.py')

ORG = {'BTCV': ['Aorta', 'Gallbl', 'KidL', 'KidR', 'Liver', 'Pancr', 'Spleen', 'Stomach'],
       'ACDC': ['RV', 'Myo', 'LV']}

# 模型名稱 → (trainer 類別, 測試集預測資料夾)
MODELS = {
    'BTCV': {
        'baseline': ('nnUNetTrainer',              f'{H}/btcv_test_pred/nnUNetTrainer'),
        'psi1w3':   ('nnUNetTrainer_btcv_psi1w3',  f'{H}/btcv_test_pred/nnUNetTrainer_btcv_psi1w3'),
        'psi6w3':   ('nnUNetTrainer_btcv_psi6w3',  f'{H}/btcv_test_pred/nnUNetTrainer_btcv_psi6w3'),
        'psi8w3':   ('nnUNetTrainer_btcv_psi8w3',  f'{H}/btcv_test_pred/nnUNetTrainer_btcv_psi8w3'),
        'w1ctrl':   ('nnUNetTrainer_btcv_w1ctrl',  f'{H}/btcv_test_pred/nnUNetTrainer_btcv_w1ctrl'),
    },
    'ACDC': {
        'baseline': ('nnUNetTrainer_500epochs',         f'{H}/nnUNet/acdc_test_pred'),
        'psi1w3':   ('nnUNetTrainer_500epochs_psi1w3',  f'{H}/nnUNet/acdc_test_pred_psi1w3'),
        'psi6w3':   ('nnUNetTrainer_500epochs_psi6w3',  f'{H}/nnUNet/acdc_test_pred_psi6w3'),
        'psi7w3':   ('nnUNetTrainer_500epochs_psi7w3',  f'{H}/nnUNet/acdc_test_pred_psi7w3'),
        'psi10w3':  ('nnUNetTrainer_500epochs_psi10w3', f'{H}/nnUNet/acdc_test_pred_psi10w3'),
    },
}
DSID = {'BTCV': 'Dataset115_BTCV', 'ACDC': 'Dataset116_ACDC'}


# ---------------- 讀取 ----------------
def read_summaries(files):
    out = {}
    for f in files:
        for c in json.load(open(f))['metric_per_case']:
            name = os.path.basename(c['reference_file']).replace('.nii.gz', '')
            out[name] = [c['metrics'][k]['Dice'] for k in sorted(c['metrics'], key=int)]
    return out


def load(ds, model, source):
    trainer, test_dir = MODELS[ds][model]
    rdir = f'{RES}/{DSID[ds]}/{trainer}__nnUNetPlans__2d'
    if source == 'test':
        f = f'{test_dir}/summary.json'
        return read_summaries([f]) if os.path.exists(f) else None
    pat = {'trainOOF': 'fold_[0-4]', 'trainALL': 'fold_all'}[source]
    files = sorted(glob.glob(f'{rdir}/{pat}/validation/summary.json'))
    return read_summaries(files) if files else None


def weighted_lists():
    """從 trainer 原始碼找出每個專家實際使用的清單與倍率"""
    src = open(TRAINER_PY).read()
    out = {}
    for block in re.split(r'\nclass ', src)[1:]:
        cls = block.split('(')[0].strip()
        m_csv = re.search(r"'(wcases[^']+\.csv)'", block)
        m_fac = re.search(r"PSI_WEIGHT_FACTOR'\]\s*=\s*'([\d.]+)'", block)
        if not m_csv:
            continue
        factor = float(m_fac.group(1)) if m_fac else 3.0
        path = f'{TF}/{m_csv.group(1)}'
        if os.path.exists(path):
            out[cls] = (set(pd.read_csv(path)['case'].astype(str)), factor, m_csv.group(1))
    return out


# ---------------- 主程式 ----------------
def main():
    WL = weighted_lists()
    print('【各專家使用的加權清單】')
    for cls, (s, f, name) in WL.items():
        print(f'  {cls:<36} {name:<36} {len(s):>3} 例  倍率 {f}')

    # ---------- 一、每個模型的 Dice ----------
    print('\n【一、每個模型的 Dice 表】')
    sources = {'BTCV': ['test', 'trainOOF'], 'ACDC': ['test', 'trainALL', 'trainOOF']}
    for ds, models in MODELS.items():
        org = ORG[ds]
        for src in sources[ds]:
            for m, (trainer, _) in models.items():
                D = load(ds, m, src)
                if D is None:
                    continue
                wl = WL.get(trainer)
                wset = wl[0] if (wl and wl[1] != 1.0) else set()
                rows = []
                for case in sorted(D):
                    v = D[case]
                    r = {'case': case}
                    r.update({o: v[i] for i, o in enumerate(org)})
                    r['mean'] = np.nanmean(v)
                    r['weighted'] = ('是' if case in wset else '否') if src != 'test' else '—（測試集）'
                    rows.append(r)
                f = f'{OUT}/dice_{ds}_{src}_{m}.csv'
                pd.DataFrame(rows).to_csv(f, index=False, encoding='utf-8-sig', float_format='%.4f')
                print(f'  {os.path.basename(f):<40} {len(rows):>3} 例  平均 {np.mean([r["mean"] for r in rows]):.4f}')

    # ---------- 二、加權病人對照表 ----------
    print('\n【二、加權病人對照表】')
    src_for = {'BTCV': 'trainOOF', 'ACDC': 'trainALL'}
    for ds, models in MODELS.items():
        org = ORG[ds]
        src = src_for[ds]
        base = load(ds, 'baseline', src)
        if base is None:
            print(f'  ⚠ {ds} baseline 無 {src} 預測，跳過')
            continue
        rows = []
        for m, (trainer, _) in models.items():
            if m == 'baseline' or trainer not in WL:
                continue
            wset, factor, listname = WL[trainer]
            if factor == 1.0:
                continue                       # w1ctrl 倍率 1.0，沒有真正加權
            E = load(ds, m, src)
            if E is None:
                print(f'  ⚠ {ds} {m} 無 {src} 預測，跳過')
                continue
            cases = sorted(set(base) & set(E))
            order = [c for c in cases if c in wset] + [c for c in cases if c not in wset]
            for c in order:
                a = np.array(base[c], float)
                e = np.array(E[c], float)
                both = np.isnan(a) | np.isnan(e)
                a[both] = np.nan
                e[both] = np.nan
                r = {'expert': m, 'list_file': listname, 'case': c,
                     'weighted': '是' if c in wset else '否',
                     'baseline_mean': np.nanmean(a), 'expert_mean': np.nanmean(e),
                     'delta': np.nanmean(e) - np.nanmean(a)}
                for i, o in enumerate(org):
                    r[f'{o}_baseline'] = a[i]
                    r[f'{o}_expert'] = e[i]
                rows.append(r)
            # 區塊小計
            blk = pd.DataFrame([x for x in rows if x['expert'] == m])
            for flag in ['是', '否']:
                g = blk[blk.weighted == flag]
                if len(g):
                    print(f'  {ds:<5} {m:<8} 加權={flag}  n={len(g):>3}  '
                          f'baseline {g.baseline_mean.mean():.4f} → expert {g.expert_mean.mean():.4f}  '
                          f'Δ {g.delta.mean():+.4f}  升/降 {(g.delta>0).sum()}/{(g.delta<0).sum()}')
        f = f'{OUT}/weighted_patients_{ds}.csv'
        pd.DataFrame(rows).to_csv(f, index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f'  → {os.path.basename(f)}')

    print(f'\n全部輸出於 {OUT}')
    print('\n提醒：')
    print('  BTCV 對照表用 OOF：加權病人在「預測它的那一折」是驗證資料，當折並未被加權；')
    print('        它只在其餘 4 折被加權。因此加權組的變化反映的是泛化，而非模型直接記住它。')
    print('  ACDC 對照表用 fold_all：模型訓練時看過這 160 個影像，Dice 接近飽和，')
    print('        反映的是訓練集擬合程度，不代表泛化能力。')


if __name__ == '__main__':
    main()