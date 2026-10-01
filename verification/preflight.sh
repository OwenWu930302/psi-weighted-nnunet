#!/bin/bash
# ============================================================
# 訓練前總檢查：確認環境、資料、前處理、清單都正確
# 用法：source env.sh && bash verification/preflight.sh
# 任一項出現 ✗ 就不要開始訓練
# ============================================================
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0; FAIL=0
ok()  { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
chk() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo "【1】環境變數"
for v in PSI_ROOT nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  [ -n "${!v:-}" ] && ok "$v = ${!v}" || bad "$v 未設定（請先 source env.sh）"
done
[ -z "${PSI_WEIGHT_CASES:-}" ] && ok "PSI_WEIGHT_CASES 未殘留" || bad "PSI_WEIGHT_CASES 殘留舊值：$PSI_WEIGHT_CASES"

echo "【2】nnU-Net 與 trainer"
cd ~
NN=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))" 2>/dev/null)
[ "$NN" = "$PSI_ROOT/nnUNet/nnunetv2" ] && ok "載入的 nnU-Net：$NN" || bad "載入的 nnU-Net 不在 \$PSI_ROOT/nnUNet（實際：$NN）"
TAG=$(cd "$PSI_ROOT/nnUNet" 2>/dev/null && git describe --tags 2>/dev/null)
[ "$TAG" = "v2.8.1" ] && ok "nnU-Net 版本 v2.8.1" || bad "nnU-Net 版本為 $TAG（應為 v2.8.1）"
T="$PSI_ROOT/nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_PsiWeighted.py"
cmp -s "$T" "$REPO/nnunet_extension/nnUNetTrainer_PsiWeighted.py" && ok "trainer 已安裝且與倉庫版本相同" || bad "trainer 未安裝或與倉庫版本不同"
chk "trainer 可匯入（10 個類別）" "python3 -c 'import nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_PsiWeighted as m; assert len([c for c in dir(m) if c.startswith(\"nnUNetTrainer_\")])>=10'"
chk "其他套件（cv2, h5py, nibabel, sklearn）" "python3 -c 'import cv2, h5py, nibabel, sklearn'"
cd "$REPO"

echo "【3】原始資料（nnU-Net 格式）"
R=$nnUNet_raw
n() { ls "$1" 2>/dev/null | wc -l; }
[ "$(n $R/Dataset115_BTCV/imagesTr)" = 18 ] && ok "BTCV imagesTr 18" || bad "BTCV imagesTr $(n $R/Dataset115_BTCV/imagesTr)（應為 18）"
[ "$(n $R/Dataset115_BTCV/labelsTs)" = 12 ] && ok "BTCV labelsTs 12" || bad "BTCV labelsTs $(n $R/Dataset115_BTCV/labelsTs)（應為 12）"
[ "$(n $R/Dataset116_ACDC/imagesTr)" = 160 ] && ok "ACDC imagesTr 160" || bad "ACDC imagesTr $(n $R/Dataset116_ACDC/imagesTr)（應為 160）"
[ "$(n $R/Dataset116_ACDC/labelsTs)" = 40 ] && ok "ACDC labelsTs 40" || bad "ACDC labelsTs $(n $R/Dataset116_ACDC/labelsTs)（應為 40）"
chk "ACDC 病人切分與原始實驗相同" "python3 -c 'import json; assert json.load(open(\"$R/Dataset116_ACDC/patient_split.json\"))==json.load(open(\"splits/acdc_patient_split.json\"))'"
chk "BTCV 測試集編號相同" "diff <(ls $R/Dataset115_BTCV/imagesTs | sed 's/_0000.nii.gz//') splits/btcv_test_cases.txt"

echo "【4】前處理與切分"
P=$nnUNet_preprocessed
chk "BTCV plans 為 batch 4 / patch 256×256" "python3 -c 'import json; c=json.load(open(\"$P/Dataset115_BTCV/nnUNetPlans.json\"))[\"configurations\"][\"2d\"]; assert c[\"batch_size\"]==4 and c[\"patch_size\"]==[256,256]'"
chk "ACDC plans 為 batch 56 / patch 256×224" "python3 -c 'import json; c=json.load(open(\"$P/Dataset116_ACDC/nnUNetPlans.json\"))[\"configurations\"][\"2d\"]; assert c[\"batch_size\"]==56 and c[\"patch_size\"]==[256,224]'"
[ "$(n $P/Dataset115_BTCV/nnUNetPlans_2d)" -gt 0 ] && ok "BTCV 已前處理（2d）" || bad "BTCV 尚未前處理"
[ "$(n $P/Dataset116_ACDC/nnUNetPlans_2d)" -gt 0 ] && ok "ACDC 已前處理（2d）" || bad "ACDC 尚未前處理"
chk "BTCV splits_final.json 與倉庫相同" "cmp -s $P/Dataset115_BTCV/splits_final.json splits/btcv_splits_final.json"
chk "ACDC splits_final.json 與倉庫相同" "cmp -s $P/Dataset116_ACDC/splits_final.json splits/acdc_splits_final.json"

echo "【5】加權清單"
for f in weight_lists/wcases_*.csv; do
  b=$(basename $f)
  cmp -s "$f" "$PSI_ROOT/tree_features/$b" && ok "$b" || bad "$b 未放入 \$PSI_ROOT/tree_features"
done

echo
echo "通過 $PASS 項，失敗 $FAIL 項"
[ $FAIL -eq 0 ] && echo "→ 可以開始訓練（docs/REPRODUCE.md 第 5 步）" || echo "→ 請先修正 ✗ 項目"
