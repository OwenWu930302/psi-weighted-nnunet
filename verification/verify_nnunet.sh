#!/bin/bash
# ============================================================
# nnU-Net 版本驗證報告
# 目的：以 git 證明本機 nnU-Net 與官方 v2.8.1 的一致性，
#       並列出所有差異與 Ψ 加權 trainer 的 train_step。
# 用法：conda activate expertree && bash verify_nnunet.sh
# 輸出：~/桌面/論文/nnunet_verification_<時間>.txt
# ============================================================
set -u
REPO="${PSI_ROOT:-$HOME/桌面/論文}/nnUNet"
TAG="v2.8.1"
PSI="nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_PsiWeighted.py"
OUT="${PSI_ROOT:-$HOME/桌面/論文}/nnunet_verification_$(date +%Y%m%d_%H%M%S).txt"
CORE=(
  nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
  nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py
  nnunetv2/training/loss/compound_losses.py
  nnunetv2/training/loss/dice.py
  nnunetv2/training/loss/robust_ce_loss.py
  nnunetv2/training/loss/deep_supervision.py
  nnunetv2/training/dataloading/data_loader.py
  nnunetv2/run/run_training.py
  nnunetv2/inference/predict_from_raw_data.py
  nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py
)

cd "$REPO" || exit 1
{
echo "############################################################"
echo "# nnU-Net 版本驗證報告"
echo "# 產生時間：$(date '+%Y-%m-%d %H:%M:%S')"
echo "# 主機：$(hostname)"
echo "############################################################"

echo; echo "===== 1. 程式碼來源（git remote）====="
git remote -v

echo; echo "===== 2. 本機 commit 與官方標籤 $TAG 是否為同一個 commit ====="
HEAD_HASH=$(git rev-parse HEAD)
TAG_HASH=$(git rev-parse "$TAG^{commit}")
echo "本機 HEAD      : $HEAD_HASH"
echo "本機標籤 $TAG  : $TAG_HASH"
echo "GitHub 官方即時查詢（git ls-remote，不依賴本機資料）："
git ls-remote --tags origin "refs/tags/$TAG" "refs/tags/$TAG^{}"
if [ "$HEAD_HASH" = "$TAG_HASH" ]; then
  echo "結果：✓ 本機 HEAD 就是官方 $TAG 發布版本"
else
  echo "結果：✗ 本機 HEAD 與 $TAG 不同"
fi

echo; echo "===== 3. 相對官方 $TAG，所有被改動的官方檔案 ====="
echo "（M = 修改, D = 刪除；此處只列官方原本就有的檔案）"
git diff --name-status "$TAG"

echo; echo "===== 4. 被修改（M）檔案的完整差異 ====="
git diff --diff-filter=M "$TAG"

echo; echo "===== 5. 核心訓練 / loss / 推論檔案逐檔比對 ====="
echo "（比對方式：本機檔案的 git blob hash 與官方 $TAG 中同一檔案的 hash；相同代表逐位元組完全一致）"
for f in "${CORE[@]}"; do
  if [ ! -f "$f" ]; then echo "  ?  不存在     $f"; continue; fi
  LOCAL=$(git hash-object "$f")
  OFFICIAL=$(git rev-parse "$TAG:$f" 2>/dev/null || echo "官方無此檔")
  if [ "$LOCAL" = "$OFFICIAL" ]; then
    echo "  ✓  完全相同   $f   ($LOCAL)"
  else
    echo "  ✗  不同       $f   本機 $LOCAL / 官方 $OFFICIAL"
  fi
done

echo; echo "===== 6. 新增的檔案（官方沒有，git 未追蹤）====="
git status --short | grep '^??' || echo "（無）"

echo; echo "===== 7. Python 實際載入的 nnU-Net ====="
python3 -c "import nnunetv2, os; print('載入路徑:', os.path.dirname(nnunetv2.__file__))"
pip show nnunetv2 2>/dev/null | grep -E "^Version|^Editable"

echo; echo "===== 8. Ψ 加權 trainer：檔案指紋 ====="
sha256sum "$PSI"
echo "繼承關係與覆寫的方法："
grep -n -E "^class |^    def " "$PSI"

echo; echo "===== 9. Ψ 加權 trainer：train_step 完整內容（含行號）====="
awk '/^    def train_step/{p=1} p && /^    def / && !/train_step/{exit} p{printf "%4d  %s\n", NR, $0}' "$PSI"

echo; echo "===== 10. 官方 nnUNetTrainer.train_step（$TAG 原版，供對照）====="
git show "$TAG:nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py" \
  | awk '/^    def train_step/{p=1} p && /^    def / && !/train_step/{exit} p{printf "%4d  %s\n", NR, $0}'

echo; echo "############################################################"
echo "# 報告結束"
echo "############################################################"
} 2>&1 | tee "$OUT"

echo
echo "報告已存到：$OUT"
echo "報告指紋（sha256，可證明報告產生後未被修改）："
sha256sum "$OUT"
