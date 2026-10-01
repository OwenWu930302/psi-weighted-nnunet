"""
TransUNet 前處理的 Synapse（BTCV）資料 -> NIfTI
==========================================
本研究 BTCV 的實際資料來源。

輸入：$TRANSUNET_SYNAPSE（預設 $PSI_ROOT/SAMA-UNet_repo/project_TransUNet/data/Synapse）
    train_npz/caseXXXX_sliceYYY.npz   18 例訓練資料，以 2D 切片儲存（keys: image, label）
    test_vol_h5/caseXXXX.npy.h5       12 例測試資料，以 3D 體積儲存（keys: image, label）
    取得方式依 TransUNet 官方倉庫說明：https://github.com/Beckschen/TransUNet

TransUNet 的前處理（非本研究所做）：
    CT 強度截斷至 [-125, 275] 後縮放到 [0, 1]；標籤已整理為 8 個器官（1–8）。

輸出：$BTCV_NIFTI（預設 $PSI_ROOT/BTCV_nifti）
    imagesTr/ labelsTr/  18 例（由切片依編號重組為 3D）
    imagesTs/ labelsTs/  12 例

注意：
    1. 以 affine = 單位矩陣儲存，體素間距為 1 × 1 × 1 mm（原始物理間距未保留）
    2. 陣列由 (D, H, W) 轉為 (H, W, D)
    3. 與原始實驗所用版本的差異僅在路徑改由環境變數設定、並加入數量檢查
"""
import glob
import os
import re

import h5py
import nibabel as nib
import numpy as np

ROOT = os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_ROOT = os.environ.get('TRANSUNET_SYNAPSE',
                          os.path.join(ROOT, 'SAMA-UNet_repo/project_TransUNet/data/Synapse'))
DST_ROOT = os.environ.get('BTCV_NIFTI', os.path.join(ROOT, 'BTCV_nifti'))

for sub in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
    os.makedirs(os.path.join(DST_ROOT, sub), exist_ok=True)

# ---------- 測試集：h5 -> nii.gz（本身即為 3D） ----------
test_files = sorted(glob.glob(os.path.join(SRC_ROOT, 'test_vol_h5', '*.npy.h5')))
print(f'找到 {len(test_files)} 個 test volume')
for fpath in test_files:
    case_name = os.path.basename(fpath).replace('.npy.h5', '')          # case0001
    with h5py.File(fpath, 'r') as f:
        image = f['image'][:]                                            # (D, H, W)
        label = f['label'][:]
    nib.save(nib.Nifti1Image(image.transpose(1, 2, 0).astype(np.float32), affine=np.eye(4)),
             os.path.join(DST_ROOT, 'imagesTs', f'{case_name}.nii.gz'))
    nib.save(nib.Nifti1Image(label.transpose(1, 2, 0).astype(np.uint8), affine=np.eye(4)),
             os.path.join(DST_ROOT, 'labelsTs', f'{case_name}.nii.gz'))
    print(f'  test : {case_name}  shape={image.shape}')

# ---------- 訓練集：npz 切片 -> 依編號重組 3D -> nii.gz ----------
train_files = glob.glob(os.path.join(SRC_ROOT, 'train_npz', '*.npz'))
print(f'找到 {len(train_files)} 個 train 切片')
case_slices = {}
pattern = re.compile(r'(case\d+)_slice(\d+)\.npz')
for fpath in train_files:
    m = pattern.match(os.path.basename(fpath))
    if m:
        case_slices.setdefault(m.group(1), []).append((int(m.group(2)), fpath))

for case_id, slices in sorted(case_slices.items()):
    slices.sort(key=lambda x: x[0])
    images, labels = [], []
    for _, fpath in slices:
        data = np.load(fpath)
        images.append(data['image'])
        labels.append(data['label'])
    image_vol = np.stack(images, axis=0)                                 # (D, H, W)
    label_vol = np.stack(labels, axis=0)
    nib.save(nib.Nifti1Image(image_vol.transpose(1, 2, 0).astype(np.float32), affine=np.eye(4)),
             os.path.join(DST_ROOT, 'imagesTr', f'{case_id}.nii.gz'))
    nib.save(nib.Nifti1Image(label_vol.transpose(1, 2, 0).astype(np.uint8), affine=np.eye(4)),
             os.path.join(DST_ROOT, 'labelsTr', f'{case_id}.nii.gz'))
    print(f'  train: {case_id}  {len(slices)} 片  shape={image_vol.shape}')

print(f'\n訓練 {len(case_slices)} 例、測試 {len(test_files)} 例 → {DST_ROOT}')
assert len(case_slices) == 18 and len(test_files) == 12, '預期 18 訓練 + 12 測試（TransUNet 切分）'
print('完成')
