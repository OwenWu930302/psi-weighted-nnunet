#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ψ8 旋轉一致性（ACDC）
Ψ8 = 1 - Dice(原始預測, 旋轉→推論→轉回的預測)
值越大 = 模型對旋轉越不穩定
"""
import os, glob, subprocess, csv
import numpy as np, nibabel as nib
from scipy.ndimage import rotate

R      = os.path.expanduser('~/桌面/論文')
IMG_IN = f'{R}/nnUNet_data/nnUNet_raw/Dataset116_ACDC/imagesTs'
PRED0  = f'{R}/nnUNet/acdc_test_pred'
WORK   = f'{R}/psi8_work'
ANGLE  = 15
LABELS = {1:'RV', 2:'Myo', 3:'LV'}

os.makedirs(f'{WORK}/imagesRot', exist_ok=True)
os.makedirs(f'{WORK}/predRot',   exist_ok=True)

files = sorted(glob.glob(f'{IMG_IN}/*.nii.gz'))
print(f'[1/4] 旋轉 {len(files)} 個影像 {ANGLE}°')
for f in files:
    img = nib.load(f)
    a = np.asarray(img.dataobj).astype(np.float32)
    ar = rotate(a, ANGLE, axes=(0,1), reshape=False, order=1,
                mode='constant', cval=0)
    nib.save(nib.Nifti1Image(ar, img.affine, img.header),
             f'{WORK}/imagesRot/{os.path.basename(f)}')
    print('.', end='', flush=True)
print()

print('[2/4] nnU-Net 推論中（最耗時）')
subprocess.run([
    'nnUNetv2_predict',
    '-i', f'{WORK}/imagesRot',
    '-o', f'{WORK}/predRot',
    '-d', '116',
    '-c', '2d',
    '-tr', 'nnUNetTrainer_500epochs',
    '-f', 'all',
], check=True)

print('[3/4] 旋轉還原並計算 Dice')
def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s

rows, missing = [], []
for f in sorted(glob.glob(f'{PRED0}/*.nii.gz')):
    n  = os.path.basename(f)
    fr = f'{WORK}/predRot/{n}'
    if not os.path.exists(fr):
        missing.append(n); continue
    p0 = np.asarray(nib.load(f).dataobj)
    pr = np.asarray(nib.load(fr).dataobj)
    pr_back = rotate(pr, -ANGLE, axes=(0,1), reshape=False,
                     order=0, mode='constant', cval=0)
    for k, nm in LABELS.items():
        rows.append(dict(case=n.replace('.nii.gz',''), structure=nm,
                         psi8_rotation=round(1.0 - dice(p0==k, pr_back==k), 6)))

if missing:
    print(f'  略過 {len(missing)} 個無旋轉預測的 case（測試模式下屬正常）')

out = f'{R}/tree_features/psi8_acdc.csv'
with open(out, 'w', newline='') as fo:
    w = csv.DictWriter(fo, ['case','structure','psi8_rotation'])
    w.writeheader(); w.writerows(rows)
print(f'[4/4] 完成 → {out}　共 {len(rows)} 筆')