import os
import shutil

SRC = "/home/tkyin/桌面/論文/BTCV_nifti"
DST = "/home/tkyin/桌面/論文/nnUNet_data/nnUNet_raw/Dataset115_BTCV"

# 處理 train (imagesTr + labelsTr)
train_img_src = os.path.join(SRC, "imagesTr")
train_lbl_src = os.path.join(SRC, "labelsTr")

case_id_map = {}  # 記錄 case0005 -> BTCV_001 這種對應關係
counter = 1

for fname in sorted(os.listdir(train_img_src)):
    if fname.endswith(".nii.gz"):
        case_name = fname.replace(".nii.gz", "")  # case0005
        new_id = f"BTCV_{counter:03d}"
        case_id_map[case_name] = new_id

        shutil.copy(
            os.path.join(train_img_src, fname),
            os.path.join(DST, "imagesTr", f"{new_id}_0000.nii.gz")
        )
        shutil.copy(
            os.path.join(train_lbl_src, fname),
            os.path.join(DST, "labelsTr", f"{new_id}.nii.gz")
        )
        print(f"Train: {case_name} -> {new_id}")
        counter += 1

# 處理 test (imagesTs)，nnU-Net的imagesTs只需要影像，不需要label
test_img_src = os.path.join(SRC, "imagesTs")

for fname in sorted(os.listdir(test_img_src)):
    if fname.endswith(".nii.gz"):
        case_name = fname.replace(".nii.gz", "")
        new_id = f"BTCV_{counter:03d}"
        case_id_map[case_name] = new_id

        shutil.copy(
            os.path.join(test_img_src, fname),
            os.path.join(DST, "imagesTs", f"{new_id}_0000.nii.gz")
        )
        print(f"Test: {case_name} -> {new_id}")
        counter += 1

print(f"\n總共處理 {counter-1} 個病人")
print("\n對應表 (原始case編號 -> nnU-Net編號):")
for k, v in case_id_map.items():
    print(f"  {k} -> {v}")
