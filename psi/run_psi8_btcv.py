#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ψ8 旋轉一致性（BTCV）

Eq.(8):  Ψ_rot(x, ŷ) = 1 − Dice(R_φ(ŷ), I(σ(f(R_φ(x))) > 0.5))

重要：BTCV 採 5-fold 交叉驗證，每個案例的原始預測來自「未見過該案例」的那一折模型。
      重新推論必須沿用同一折，否則等同用訓練過該案例的模型推論，Ψ8 會被系統性低估。
      本腳本因此逐折分別旋轉、推論、比對。

用法:
    conda activate expertree
    source <倉庫>/env.sh
    python3 run_psi8_btcv.py
"""
import os, glob, subprocess, csv
import numpy as np, nibabel as nib
from scipy.ndimage import rotate

R      = os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW    = f'{R}/nnUNet_data/nnUNet_raw/Dataset115_BTCV'
RESULT = f'{R}/nnUNet_data/nnUNet_results/Dataset115_BTCV/nnUNetTrainer__nnUNetPlans__2d'
WORK   = f'{R}/psi8_btcv_work'
ANGLE  = 15
LABELS = {1:'Aorta', 2:'Gallbl', 3:'KidL', 4:'KidR',
          5:'Liver', 6:'Pancr', 7:'Spleen', 8:'Stomach'}

DATASET_ID = '115'
CONFIG     = '2d'
TRAINER    = 'nnUNetTrainer'
FOLDS      = [0, 1, 2, 3, 4]


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


src = {}
for d in (f'{RAW}/imagesTr', f'{RAW}/imagesTs'):
    for f in glob.glob(f'{d}/*.nii.gz'):
        base = os.path.basename(f).replace('.nii.gz', '')
        key = base[:-5] if base.endswith('_0000') else base
        src.setdefault(key, f)
print(f'影像索引 {len(src)} 個')

rows, skipped = [], []

for fold in FOLDS:
    vdir = f'{RESULT}/fold_{fold}/validation'
    preds = {os.path.basename(f).replace('.nii.gz', ''): f
             for f in glob.glob(f'{vdir}/*.nii.gz')}
    if not preds:
        print(f'[fold {fold}] 無 validation 預測，略過')
        continue

    todo = [k for k in preds if k in src]
    print(f'\n[fold {fold}] 預測 {len(preds)} 個，可對應影像 {len(todo)} 個')
    if not todo:
        print(f'  預測範例: {list(preds)[:3]}')
        print(f'  影像範例: {list(src)[:3]}')
        skipped.extend(preds)
        continue

    fin  = f'{WORK}/fold{fold}/imagesRot'
    fout = f'{WORK}/fold{fold}/predRot'
    os.makedirs(fin, exist_ok=True)
    os.makedirs(fout, exist_ok=True)

    print(f'  旋轉 {ANGLE}°', end=' ', flush=True)
    for k in todo:
        img = nib.load(src[k])
        a = np.asarray(img.dataobj).astype(np.float32)
        ar = rotate(a, ANGLE, axes=(0, 1), reshape=False, order=1,
                    mode='constant', cval=float(a.min()))
        nib.save(nib.Nifti1Image(ar, img.affine, img.header),
                 f'{fin}/{k}_0000.nii.gz')
        print('.', end='', flush=True)
    print()

    print(f'  推論 fold {fold}')
    subprocess.run([
        'nnUNetv2_predict',
        '-i', fin, '-o', fout,
        '-d', DATASET_ID, '-c', CONFIG,
        '-tr', TRAINER, '-f', str(fold),
    ], check=True)

    for k in todo:
        fr = f'{fout}/{k}.nii.gz'
        if not os.path.exists(fr):
            skipped.append(k); continue
        p0 = np.asarray(nib.load(preds[k]).dataobj)
        pr = np.asarray(nib.load(fr).dataobj)
        pr_back = rotate(pr, -ANGLE, axes=(0, 1), reshape=False,
                         order=0, mode='constant', cval=0)
        for lab, nm in LABELS.items():
            rows.append(dict(case=k, structure=nm, fold=fold,
                             psi8_rotation=round(1.0 - dice(p0 == lab, pr_back == lab), 6)))

if skipped:
    print(f'\n警告: {len(skipped)} 個案例未處理: {skipped[:5]}')

out = f'{R}/tree_features/psi8_btcv.csv'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', newline='') as fo:
    w = csv.DictWriter(fo, ['case', 'structure', 'fold', 'psi8_rotation'])
    w.writeheader(); w.writerows(rows)

v = np.array([r['psi8_rotation'] for r in rows])
print(f'\n完成 → {out}　共 {len(rows)} 筆（預期 144）')
print(f'Ψ8  mean={v.mean():.4f}  sd={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}')
print('\n依器官:')
for nm in LABELS.values():
    vv = np.array([r['psi8_rotation'] for r in rows if r['structure'] == nm])
    if len(vv):
        print(f'  {nm:<9} mean={vv.mean():.4f}  n={len(vv)}')
print('\n若數值幾乎全為 0，代表模型對旋轉免疫，需調高 ANGLE 後重跑。')