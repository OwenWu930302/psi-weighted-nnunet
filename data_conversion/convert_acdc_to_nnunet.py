"""
ACDC -> nnU-Net v2 格式轉換
==========================================
論文設定：100 例心臟 MRI，隨機切 80 train / 20 test
每例病人有 ED / ES 兩個時相，各含影像與標註，
因此 80 位病人 -> 160 個訓練 case，20 位 -> 40 個測試 case。

標註類別（ACDC 官方定義）：
    0 = 背景
    1 = 右心室腔 (RV)
    2 = 心肌     (Myo)
    3 = 左心室腔 (LV)

輸出結構：
    Dataset116_ACDC/
        imagesTr/ACDC_XXX_0000.nii.gz
        labelsTr/ACDC_XXX.nii.gz
        imagesTs/ACDC_XXX_0000.nii.gz
        labelsTs/ACDC_XXX.nii.gz
        dataset.json
"""
import json
import os
import random
import re
import shutil
from pathlib import Path

# ------------------------- 路徑設定 -------------------------
ROOT = Path(os.environ.get('PSI_ROOT', os.path.expanduser('~/桌面/論文')))
ACDC_ROOT = Path(os.environ.get('ACDC_RAW', ROOT / 'SAMA-UNet_repo/ACDC/database/training'))
OUT_ROOT = ROOT / 'nnUNet_data/nnUNet_raw/Dataset116_ACDC'

N_TRAIN_PATIENTS = 80      # 論文設定
SEED = 42                  # 固定亂數種子，確保可重現
# -----------------------------------------------------------


def find_frame_pairs(patient_dir: Path):
    """
    找出該病人的所有 (影像, 標註) 配對。
    ACDC 每位病人有兩個標註時相（ED / ES），檔名形如：
        patientXXX_frameNN.nii.gz      <- 影像
        patientXXX_frameNN_gt.nii.gz   <- 標註
    需排除 patientXXX_4d.nii.gz（完整心動週期，無標註）。
    """
    pairs = []
    for gt_path in sorted(patient_dir.glob('*_gt.nii.gz')):
        img_path = Path(str(gt_path).replace('_gt.nii.gz', '.nii.gz'))
        if img_path.exists():
            frame = re.search(r'frame(\d+)', gt_path.name).group(1)
            pairs.append((img_path, gt_path, frame))
    return pairs


def main():
    # 建立輸出目錄
    for sub in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # 取得所有病人並依論文設定隨機切分
    patients = sorted([p for p in ACDC_ROOT.iterdir()
                       if p.is_dir() and p.name.startswith('patient')])
    print(f'找到 {len(patients)} 位病人')
    assert len(patients) == 100, f'預期 100 位病人，實際 {len(patients)}'

    random.seed(SEED)
    shuffled = patients[:]
    random.shuffle(shuffled)
    train_patients = sorted(shuffled[:N_TRAIN_PATIENTS], key=lambda p: p.name)
    test_patients = sorted(shuffled[N_TRAIN_PATIENTS:], key=lambda p: p.name)

    print(f'訓練病人：{len(train_patients)} 位')
    print(f'測試病人：{len(test_patients)} 位')

    case_id = 0
    n_train_cases = 0
    n_test_cases = 0
    split_record = {'train_patients': [], 'test_patients': []}

    for split, plist in [('Tr', train_patients), ('Ts', test_patients)]:
        for pdir in plist:
            pairs = find_frame_pairs(pdir)
            if not pairs:
                print(f'  [警告] {pdir.name} 找不到標註配對，跳過')
                continue
            for img_path, gt_path, frame in pairs:
                case_id += 1
                name = f'ACDC_{case_id:03d}'
                # nnUNet 影像需帶 modality 後綴 _0000
                shutil.copy2(img_path, OUT_ROOT / f'images{split}' / f'{name}_0000.nii.gz')
                shutil.copy2(gt_path, OUT_ROOT / f'labels{split}' / f'{name}.nii.gz')
                if split == 'Tr':
                    n_train_cases += 1
                else:
                    n_test_cases += 1
            key = 'train_patients' if split == 'Tr' else 'test_patients'
            split_record[key].append(pdir.name)

    # dataset.json（nnUNet v2 格式）
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "RV": 1,      # 右心室腔
            "Myo": 2,     # 心肌
            "LV": 3       # 左心室腔
        },
        "numTraining": n_train_cases,
        "file_ending": ".nii.gz",
        "name": "ACDC",
        "description": "Automated Cardiac Diagnosis Challenge, ED/ES frames"
    }
    with open(OUT_ROOT / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)

    # 記錄病人層級的切分（方便論文說明與重現）
    with open(OUT_ROOT / 'patient_split.json', 'w') as f:
        json.dump(split_record, f, indent=4)

    print('\n===== 轉換完成 =====')
    print(f'訓練 case：{n_train_cases}（{len(train_patients)} 位病人 × ED/ES）')
    print(f'測試 case：{n_test_cases}（{len(test_patients)} 位病人 × ED/ES）')
    print(f'輸出位置：{OUT_ROOT}')
    print(f'病人切分記錄：{OUT_ROOT / "patient_split.json"}')


if __name__ == '__main__':
    main()