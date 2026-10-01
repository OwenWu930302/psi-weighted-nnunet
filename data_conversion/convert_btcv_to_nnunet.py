"""
BTCV -> nnU-Net v2 格式轉換（Dataset115_BTCV）
==========================================
輸入：$BTCV_NIFTI（預設 $PSI_ROOT/BTCV_nifti），需有：
    imagesTr/caseXXXX.nii.gz   18 例訓練影像（TransUNet 公開切分）
    labelsTr/caseXXXX.nii.gz   18 例訓練標註（標籤 0–8）
    imagesTs/caseXXXX.nii.gz   12 例測試影像
    labelsTs/caseXXXX.nii.gz   12 例測試標註（標籤 0–8）

編號規則（與原始實驗完全相同）：
    依檔名排序，訓練 → BTCV_001 … BTCV_018，測試 → BTCV_019 … BTCV_030

與原始版本的差異（不影響資料內容）：
    1. 路徑改由環境變數設定
    2. 自動建立輸出資料夾
    3. 一併複製測試標註到 labelsTs/（原始實驗是事後以影像內容比對後複製，
       對應結果與依檔名排序完全一致，見 splits/btcv_test_label_mapping.csv）
    4. 寫出 dataset.json（與 configs/btcv_dataset.json 相同）
"""
import json
import os
import shutil

ROOT = os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.environ.get('BTCV_NIFTI', os.path.join(ROOT, 'BTCV_nifti'))
DST = os.path.join(ROOT, 'nnUNet_data/nnUNet_raw/Dataset115_BTCV')

for sub in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
    os.makedirs(os.path.join(DST, sub), exist_ok=True)

case_id_map = {}
counter = 1

# 訓練集
for fname in sorted(os.listdir(os.path.join(SRC, 'imagesTr'))):
    if not fname.endswith('.nii.gz'):
        continue
    case = fname.replace('.nii.gz', '')
    new_id = f'BTCV_{counter:03d}'
    case_id_map[case] = new_id
    shutil.copy(os.path.join(SRC, 'imagesTr', fname), os.path.join(DST, 'imagesTr', f'{new_id}_0000.nii.gz'))
    shutil.copy(os.path.join(SRC, 'labelsTr', fname), os.path.join(DST, 'labelsTr', f'{new_id}.nii.gz'))
    print(f'Train: {case} -> {new_id}')
    counter += 1

# 測試集
n_test_labels = 0
for fname in sorted(os.listdir(os.path.join(SRC, 'imagesTs'))):
    if not fname.endswith('.nii.gz'):
        continue
    case = fname.replace('.nii.gz', '')
    new_id = f'BTCV_{counter:03d}'
    case_id_map[case] = new_id
    shutil.copy(os.path.join(SRC, 'imagesTs', fname), os.path.join(DST, 'imagesTs', f'{new_id}_0000.nii.gz'))
    lab = os.path.join(SRC, 'labelsTs', fname)
    if os.path.exists(lab):
        shutil.copy(lab, os.path.join(DST, 'labelsTs', f'{new_id}.nii.gz'))
        n_test_labels += 1
    print(f'Test : {case} -> {new_id}')
    counter += 1

n_train = sum(1 for v in case_id_map.values() if int(v[5:]) <= 18)
dataset_json = {
    'channel_names': {'0': 'CT'},
    'labels': {'background': 0, 'aorta': 1, 'gallbladder': 2, 'kidney_left': 3,
               'kidney_right': 4, 'liver': 5, 'pancreas': 6, 'spleen': 7, 'stomach': 8},
    'numTraining': n_train,
    'file_ending': '.nii.gz',
}
with open(os.path.join(DST, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f'\n訓練 {n_train} 例、測試 {counter - 1 - n_train} 例（測試標註 {n_test_labels} 個）')
assert n_train == 18 and counter - 1 == 30, '預期 18 訓練 + 12 測試'
print('對應表：')
for k, v in case_id_map.items():
    print(f'  {k} -> {v}')
