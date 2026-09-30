"""
ACDC：依病人分組的 5 折切分
============================================================
目的：補做 OOF 時，確保同一位病人的 ED / ES 兩個影像永遠在同一折，
      避免「訓練折看過 ED、驗證折預測 ES」造成資訊洩漏。

做法：
  1. 讀 patient_split.json，取得 80 位訓練病人
  2. 將 labelsTr 的 160 個標註，逐一與原始 ACDC 標註做內容比對，
     找出每個 ACDC_XXX 屬於哪位病人、哪個時相（不靠檔名猜測）
  3. 以病人為單位做 5 折（KFold，固定種子 12345，與 nnU-Net 預設相同）
  4. 寫出 splits_final.json，並逐項檢查

用法：
  conda activate expertree
  cd ~/桌面/論文
  python3 make_acdc_grouped_splits.py            # 只檢查、不寫檔
  python3 make_acdc_grouped_splits.py --write    # 確認無誤後寫檔
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import KFold

HOME = os.environ.get('PSI_ROOT', os.path.expanduser('~/桌面/論文'))
RAW = f'{HOME}/nnUNet_data/nnUNet_raw/Dataset116_ACDC'
PRE = f'{HOME}/nnUNet_data/nnUNet_preprocessed/Dataset116_ACDC'
SRC = os.environ.get('ACDC_RAW', f'{HOME}/SAMA-UNet_repo/ACDC/database/training')
MAP_OUT = f'{HOME}/tree_features/acdc_case_patient_map.csv'
SEED = 12345
N_FOLDS = 5


def read(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(p))


def main(write):
    # ---------- 1. 訓練病人 ----------
    train_patients = json.load(open(f'{RAW}/patient_split.json'))['train_patients']
    print(f'訓練病人 {len(train_patients)} 位')
    assert len(train_patients) == 80, '訓練病人應為 80 位'

    # ---------- 2. 讀取原始標註 ----------
    orig = {}
    for p in train_patients:
        for g in sorted(glob.glob(f'{SRC}/{p}/*_gt.nii.gz')):
            orig[(p, os.path.basename(g))] = read(g)
    print(f'原始標註 {len(orig)} 個（應為 160）')

    # ---------- 3. 內容比對 ----------
    cases = sorted(os.path.basename(f).replace('.nii.gz', '')
                   for f in glob.glob(f'{RAW}/labelsTr/*.nii.gz'))
    print(f'nnU-Net 訓練影像 {len(cases)} 個')

    case2pat, problems = {}, []
    for c in cases:
        x = read(f'{RAW}/labelsTr/{c}.nii.gz')
        hits = [k for k, v in orig.items() if v.shape == x.shape and np.array_equal(v, x)]
        if len(hits) == 1:
            case2pat[c] = hits[0]
        else:
            problems.append((c, len(hits)))
    if problems:
        print('✗ 以下影像無法唯一對應：', problems)
        sys.exit(1)
    print('✓ 160 個影像全部唯一對應到原始病人')

    pat2cases = defaultdict(list)
    for c, (p, _) in case2pat.items():
        pat2cases[p].append(c)
    bad = {p: v for p, v in pat2cases.items() if len(v) != 2}
    assert not bad, f'以下病人影像數不是 2：{bad}'
    print('✓ 每位病人剛好 2 個影像（ED / ES）')

    print('\n前 4 例對應：')
    for c in cases[:4]:
        print(f'  {c} ← {case2pat[c][0]} / {case2pat[c][1]}')

    # ---------- 4. 依病人分組切 5 折 ----------
    pats = sorted(pat2cases)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = []
    for tr_idx, va_idx in kf.split(pats):
        tr = sorted(c for i in tr_idx for c in pat2cases[pats[i]])
        va = sorted(c for i in va_idx for c in pat2cases[pats[i]])
        splits.append({'train': tr, 'val': va})

    # ---------- 5. 檢查 ----------
    all_val = [c for s in splits for c in s['val']]
    assert len(all_val) == len(set(all_val)) == 160, '驗證集應互不重疊且合計 160'
    for k, s in enumerate(splits):
        assert not set(s['train']) & set(s['val']), f'fold {k} 訓練與驗證重疊'
        ptr = {case2pat[c][0] for c in s['train']}
        pva = {case2pat[c][0] for c in s['val']}
        assert not ptr & pva, f'fold {k} 有病人同時出現在訓練與驗證'
        print(f'fold {k}: 訓練 {len(s["train"])} 個（{len(ptr)} 位）'
              f'  驗證 {len(s["val"])} 個（{len(pva)} 位）  病人不跨組 ✓')
    print('✓ 5 折驗證集互不重疊，合計 160')

    # ---------- 6. 寫檔 ----------
    if not write:
        print('\n（檢查模式，未寫檔。確認無誤後加上 --write）')
        return
    out = f'{PRE}/splits_final.json'
    if os.path.exists(out):
        sys.exit(f'✗ {out} 已存在，為避免覆蓋，請先手動確認')
    json.dump(splits, open(out, 'w'), indent=2)
    print(f'\n已寫出 {out}')

    os.makedirs(os.path.dirname(MAP_OUT), exist_ok=True)
    with open(MAP_OUT, 'w') as f:
        f.write('case,patient,frame_file,fold\n')
        fold_of = {c: k for k, s in enumerate(splits) for c in s['val']}
        for c in cases:
            f.write(f'{c},{case2pat[c][0]},{case2pat[c][1]},{fold_of[c]}\n')
    print(f'已寫出對應表 {MAP_OUT}')


if __name__ == '__main__':
    main('--write' in sys.argv)