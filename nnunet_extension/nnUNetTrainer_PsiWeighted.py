# ============================================================================
# nnUNetTrainer_PsiWeighted.py
# ----------------------------------------------------------------------------
# 用途：讓 nnU-Net 在訓練時，把「清單上的病例」的 loss 放大（預設 3 倍）。
# 安裝位置：$PSI_ROOT/nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/
# 使用方式：nnUNetv2_train 116 2d all -tr nnUNetTrainer_500epochs_psi1w3
#           （-tr 後面接的是本檔案中某個類別的名稱，nnU-Net 會自動搜尋到它）
# ============================================================================

import os                      # 讀取環境變數、處理檔案路徑
import json                    # 讀取 .json 格式的清單
import numpy as np             # （本檔案未直接使用，保留原始匯入）
import torch                   # PyTorch：張量運算、反向傳播
from torch import autocast     # 混合精度：讓 GPU 自動在合適的運算上改用 float16

# 官方 nnU-Net 的訓練器基底類別；本檔案的所有類別都繼承它，
# 因此「沒有覆寫的部分」全部沿用官方行為（優化器、學習率、資料增強、驗證…）
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# dummy_context：一個「什麼都不做」的 with 區塊。
# 不在 GPU 上時用它代替 autocast，讓 with 語法依然成立。
from nnunetv2.utilities.helpers import dummy_context


# ============================================================
# Ψ 加權 Trainer
# ============================================================
# 對「被指定 proxy 標記為結構風險偏高」的 case，其訓練樣本的 loss
# 乘上一個放大係數（預設 3.0），其餘樣本維持原權重。
#
# 實作重點：
#   nnU-Net 預設的 loss 是對整個 batch 聚合後的純量（Dice loss 尤其
#   是在 batch 維度上算統計量），無法直接做逐樣本加權。因此這裡
#   把 batch 拆開，逐樣本各算一次 loss，再做加權平均：
#
#       L = Σ_i w_i · L(x_i) / Σ_i w_i
#
#   分母用權重和而非樣本數，確保 loss 的尺度與未加權時可比，
#   梯度不會因為 batch 內恰好有很多加權樣本而整體放大。
#
#   deep supervision 由 self.loss（DeepSupervisionWrapper）自行處理，
#   拆 batch 時只需對 output/target 的每一層同步取出第 i 個樣本。
#
# case 清單來源：
#   環境變數 PSI_WEIGHT_CASES 指向一個 csv（需有 case 欄）或 json 陣列。
#   放大倍率由環境變數 PSI_WEIGHT_FACTOR 指定，預設 3.0。
# ============================================================


# ----------------------------------------------------------------------------
# 輔助函式 1：讀取清單
# ----------------------------------------------------------------------------
def _load_case_list(path):
    """讀取要加權的 case 清單，支援 csv（含 case 欄）與 json 陣列。

    輸入：path —— 清單檔的路徑（字串），也可能是 None
    輸出：一個 set，例如 {'ACDC_003', 'ACDC_017', ...}
          用 set 是因為之後要大量查詢「某病例在不在清單裡」，set 查詢最快。
    """
    # 沒給路徑，或檔案不存在 → 回傳空集合。
    # 注意：這裡「不會報錯」，訓練會照常進行但沒有任何加權，
    # 等同訓練 baseline。所以 on_train_start 會在 log 裡印警告。
    if path is None or not os.path.exists(path):
        return set()

    # .json 檔：內容應為陣列，例如 ["ACDC_003", "ACDC_017"]
    if path.endswith('.json'):
        return set(json.load(open(path)))

    # 其餘視為 .csv 檔
    cases = []
    with open(path) as f:
        # 第一行是標題列，例如 "case,dice,psi1_convex"
        header = f.readline().strip().split(',')
        try:
            # 找出名為 case 的欄位在第幾欄
            idx = header.index('case')
        except ValueError:
            # 沒有 case 欄 → 退而使用第 0 欄
            idx = 0
        # 逐行讀取，取出 case 那一欄
        for line in f:
            line = line.strip()          # 去掉行尾換行與空白
            if line:                     # 跳過空行
                cases.append(line.split(',')[idx])
    return set(cases)


# ----------------------------------------------------------------------------
# 輔助函式 2：整理病例名稱
# ----------------------------------------------------------------------------
def _normalise(key):
    """
    把 dataloader 的 key 正規化成 case id。

    preprocessed 資料夾裡的檔名即為 case id（例如 ACDC_001.b2nd），
    dataloader 傳來的 key 已經是不含副檔名的 ACDC_001，故只需處理
    可能出現的副檔名。這裡刻意不去猜測、剝除任何 _NNNN 後綴，
    因為 case id 本身就以數字結尾（ACDC_001），剝掉會直接對不上清單。

    例：'ACDC_001.b2nd' → 'ACDC_001'
        'ACDC_001_seg'  → 'ACDC_001'
        'ACDC_001'      → 'ACDC_001'（v2.8.1 實際情況，原樣回傳）
    """
    k = str(key)                                   # 確保是字串（key 可能是 numpy 字串）
    for ext in ('.b2nd', '.npz', '.npy', '.pkl'):  # nnU-Net 前處理可能用到的副檔名
        if k.endswith(ext):
            k = k[:-len(ext)]                      # 去掉副檔名
            break                                  # 只會有一個副檔名，找到就停
    if k.endswith('_seg'):                         # 標註檔可能帶 _seg 後綴
        k = k[:-4]                                 # '_seg' 長度為 4
    return k


# ============================================================================
# 基底類別：所有專家模型與對照組都繼承這個類別
# ============================================================================
class nnUNetTrainer_PsiWeighted(nnUNetTrainer):
    """Ψ 加權訓練器基底。子類別指定 CASE_LIST_ENV 或直接覆寫 psi_cases。"""

    # ---- 類別屬性：子類別可以覆寫 ----
    CASE_LIST_ENV = 'PSI_WEIGHT_CASES'   # 清單路徑存在哪個環境變數
    FACTOR_ENV = 'PSI_WEIGHT_FACTOR'     # 倍率存在哪個環境變數
    DEFAULT_FACTOR = 3.0                 # 沒設定倍率時用 3.0
    NUM_EPOCHS = 500                     # 訓練幾個 epoch（BTCV 子類別改成 1000）

    # ------------------------------------------------------------------------
    # 建構子：建立 trainer 時執行一次
    # ------------------------------------------------------------------------
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 先讓官方 nnUNetTrainer 完成所有初始化：
        # 讀 plans、決定 batch size 與 patch size、建立 loss 函式、設定 epoch 數…
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 官方預設 1000 epochs，這裡改成類別屬性 NUM_EPOCHS 指定的值
        self.num_epochs = self.NUM_EPOCHS

        # 從環境變數讀清單路徑 → 讀成 set
        # （子類別會在呼叫 super().__init__ 之前先把環境變數設好）
        self.psi_cases = _load_case_list(os.environ.get(self.CASE_LIST_ENV))

        # 從環境變數讀倍率；沒設定就用 DEFAULT_FACTOR（3.0）
        # 環境變數一定是字串，所以要轉成 float
        self.psi_factor = float(os.environ.get(self.FACTOR_ENV, self.DEFAULT_FACTOR))

        # 統計用：確認加權真的有生效，避免清單對不上而靜默失效
        self._psi_hit = 0     # 本 epoch 有幾張切片被加權
        self._psi_total = 0   # 本 epoch 總共處理幾張切片

    # ------------------------------------------------------------------------
    # 訓練開始時執行一次：在 log 寫下加權設定
    # ------------------------------------------------------------------------
    def on_train_start(self):
        super().on_train_start()   # 先執行官方的準備工作（建立網路、dataloader 等）

        # 寫入 log，例如：
        # [Ψ加權] 清單 .../wcases_psi1_convex.csv　case 數 48　倍率 3.0
        self.print_to_log_file(
            f'[Ψ加權] 清單 {os.environ.get(self.CASE_LIST_ENV)}　'
            f'case 數 {len(self.psi_cases)}　倍率 {self.psi_factor}')

        # 清單是空的（路徑錯誤或檔案不存在）→ 額外寫一行警告
        if not self.psi_cases:
            self.print_to_log_file(
                '[Ψ加權] 警告：清單為空，本次訓練等同未加權的基準模型')

    # ------------------------------------------------------------------------
    # 從一個 batch 中取出「第 i 張」，但保留 batch 維度
    # ------------------------------------------------------------------------
    def _sample_slice(self, x, i):
        """從 output/target 取出第 i 個樣本，保留 batch 維度與 deep supervision 結構。

        x 可能是：
          (a) 單一張量，形狀 (B, C, H, W)
          (b) list，deep supervision 時每個解析度一個張量：
              [(B, C, 256, 224), (B, C, 128, 112), (B, C, 64, 56), ...]

        用 x[i:i+1] 而不是 x[i]：
          x[i]     → 形狀 (C, H, W)       batch 維度消失，loss 函式無法處理
          x[i:i+1] → 形狀 (1, C, H, W)    保留 batch 維度，等於「大小為 1 的 batch」
        """
        if isinstance(x, (list, tuple)):
            # deep supervision：每個解析度都取出第 i 張，保持各層一一對應
            return [t[i:i + 1] for t in x]
        return x[i:i + 1]

    # ------------------------------------------------------------------------
    # 產生權重向量 w
    # ------------------------------------------------------------------------
    def _weights_for(self, keys, n):
        """
        輸入：
          keys —— 長度 n 的病例名稱，例如 ['ACDC_017', 'ACDC_003', 'ACDC_017', ...]
          n    —— batch 裡有幾張切片（batch size）
        輸出：
          w —— 長度 n 的權重向量；在清單上的切片 = 倍率，其餘 = 1.0
          例：n=4，清單含 BTCV_004，keys=['BTCV_004','BTCV_008','BTCV_011','BTCV_004']
              → w = [3.0, 1.0, 1.0, 3.0]
        """
        # 先建立全部是 1 的向量（預設不加權）
        w = torch.ones(n, dtype=torch.float32)

        # 清單是空的，或 dataloader 沒給 keys → 直接回傳全 1（不加權）
        if not self.psi_cases or keys is None:
            return w

        # 逐張檢查病例名稱是否在清單裡
        # min(n, len(keys))：防止 keys 長度與 n 不一致時索引越界
        for i in range(min(n, len(keys))):
            if _normalise(keys[i]) in self.psi_cases:
                w[i] = self.psi_factor     # 在清單上 → 權重改成倍率
                self._psi_hit += 1         # 命中數 +1

        # 總張數 +n（只有在清單非空、且有 keys 時才會執行到這裡）
        self._psi_total += n
        return w

    # ------------------------------------------------------------------------
    # 訓練一步（核心）：取代官方 nnUNetTrainer.train_step
    # 由官方迴圈呼叫：每個 epoch 呼叫 250 次（num_iterations_per_epoch）
    # ------------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        # ===== 步驟 1：從 batch 取出資料 =====
        # 以 ACDC 2D（batch 56、patch 256×224）為例：
        #   data   : (56, 1, 256, 224)  56 張切片、1 個通道（灰階）、高、寬
        #   target : list，每個解析度一份標註：(56,1,256,224), (56,1,128,112), ...
        #   keys   : 長度 56，每張切片來自哪個病例；同一病例可能出現多次
        data = batch['data']
        target = batch['target']
        keys = batch.get('keys')   # 用 .get：沒有 keys 時回傳 None 而不是報錯

        # ===== 步驟 2：搬到 GPU =====（與官方相同）
        # non_blocking=True：搬運與其他運算可以重疊，較快
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            # deep supervision：每個解析度的標註都要搬
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # ===== 步驟 3：n 與 w =====（本研究新增）
        # n：這個 batch 有幾張切片 = data 的第 0 維（ACDC 56、BTCV 4）
        n = data.shape[0]
        # w：長度 n 的權重向量（1.0 或倍率），搬到 GPU 以便和 loss 相乘
        w = self._weights_for(keys, n).to(self.device)

        # ===== 步驟 4：清空上一步的梯度 =====（與官方相同）
        # PyTorch 的梯度預設會累加，所以每步開始前要清空
        # set_to_none=True：直接設成 None，比填 0 省記憶體
        self.optimizer.zero_grad(set_to_none=True)

        # ===== 步驟 5：混合精度區塊 + 前向傳播 =====
        # 在 GPU 上 → 用 autocast（自動混合精度）
        # 不在 GPU 上 → 用 dummy_context（什麼都不做）
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # 網路「只做一次」前向傳播，56 張一起算
            # output 是 deep supervision 的 list，最高解析度那層形狀為 (56, 4, 256, 224)，
            # 其餘各層依序縮小；其中 4 = 類別數（ACDC：背景、RV、Myo、LV）
            output = self.network(data)

            # ===== 步驟 6：判斷走哪一條路徑 =====
            # torch.allclose 判斷 w 是否「幾乎全部等於 1」
            # 容許誤差：|w_i - 1| <= 1e-8 + 1e-5 × 1 ≈ 0.00001
            #   倍率 1.0    （w1ctrl）  → 永遠全 1 → 永遠走原始路徑
            #   倍率 1.0001 （ps1ctrl） → 差 0.0001 > 0.00001 → 走逐張路徑
            if torch.allclose(w, torch.ones_like(w)):
                # ----- 路徑 A：原始路徑（與官方完全相同）-----
                # 這個 batch 沒有任何加權樣本，走原始路徑（較快且數值一致）
                # 整個 batch 一起算一個 loss；
                # Dice 是 batch dice：把 56 張的像素合併後算一個整體 Dice
                l = self.loss(output, target)
            else:
                # ----- 路徑 B：逐張計算 + 加權平均（本研究新增）-----
                # 逐樣本算 loss 後加權平均，分母為權重和以維持尺度
                total = 0.0
                for i in range(n):                       # i = 0, 1, ..., n-1
                    # 取出第 i 張的預測與標註（每個解析度都取），
                    # 用「官方的 loss 函式」算這一張的 loss：
                    #   DeepSupervisionWrapper( CE + Dice )
                    # 因為送進去的 batch 只有 1 張，Dice 變成「這張自己的 Dice」
                    li = self.loss(self._sample_slice(output, i),
                                   self._sample_slice(target, i))
                    # 乘上權重後累加
                    total = total + w[i] * li
                # 除以「權重總和」而不是 n → 得到加權平均
                # 例：w=[3,1,1,3]、各張 loss=[0.5,0.8,0.4,0.6]
                #     total = 1.5+0.8+0.4+1.8 = 4.5；w.sum() = 8；l = 0.5625
                # 結果一定介於最小與最大單張 loss 之間，
                # 不會因為 batch 裡加權切片比較多就整體變大
                l = total / w.sum()

        # ===== 步驟 7：反向傳播與更新 =====（與官方相同）
        if self.grad_scaler is not None:
            # 混合精度（GPU）的流程
            self.grad_scaler.scale(l).backward()      # loss 先放大再反向傳播，避免 float16 梯度下溢成 0
            self.grad_scaler.unscale_(self.optimizer) # 把梯度除回原本大小
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # 梯度總長度超過 12 就等比縮小
            self.grad_scaler.step(self.optimizer)     # SGD 更新權重（梯度有 inf/NaN 時跳過這步）
            self.grad_scaler.update()                 # 調整下一步的放大倍數
        else:
            # 一般精度（例如 CPU）的流程，邏輯相同
            l.backward()                                                    # 反向傳播
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # 梯度裁剪
            self.optimizer.step()                                           # 更新權重

        # ===== 步驟 8：回傳 loss 給官方迴圈記錄 =====
        # detach：切斷計算圖；cpu：搬回 CPU；numpy：轉成數字
        # 注意：專家的 loss 是加權平均且 Dice 逐張計算，不能直接和 baseline 的 loss 曲線比高低
        return {'loss': l.detach().cpu().numpy()}

    # ------------------------------------------------------------------------
    # 每個 epoch 結束時執行：印出本輪加權比例
    # ------------------------------------------------------------------------
    def on_train_epoch_end(self, train_outputs):
        # 先執行官方版本（計算並記錄本 epoch 的平均 train loss）
        super().on_train_epoch_end(train_outputs)

        # 有統計資料才印（清單為空時 _psi_total 一直是 0）
        # 例：[Ψ加權] 本輪加權樣本 4183/14000 (29.9%)
        #     14000 = 250 步 × 56 張
        if self._psi_total:
            self.print_to_log_file(
                f'[Ψ加權] 本輪加權樣本 {self._psi_hit}/{self._psi_total} '
                f'({self._psi_hit / self._psi_total:.1%})')

        # 計數器歸零，準備下一個 epoch
        self._psi_hit = 0
        self._psi_total = 0


# ============================================================
# 各專家模型與對照組：清單路徑寫死，避免忘記設環境變數
# ============================================================

# 清單資料夾的位置：
#   有設定 PSI_ROOT → $PSI_ROOT/tree_features
#   沒設定 → 從本檔案位置往上推 7 層，即工作區：
#     $PSI_ROOT/nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/本檔案
#         7       6       5        4            3            2             1
# 這行在「匯入本檔案時」就執行，之後再改環境變數不會影響 _TF
_TF = os.path.join(os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))), 'tree_features')

# ----------------------------------------------------------------------------
# 子類別的共同寫法：
#   1. 先設定環境變數（清單路徑、倍率）
#   2. 再呼叫 super().__init__ → 基底類別在 __init__ 中讀取這些環境變數
#   順序不能反，否則基底類別會讀到舊值或空值。
#
# 兩種設定方式的差別：
#   os.environ.setdefault(...)  只有「原本沒設定」時才寫入；若終端機殘留舊值會沿用舊值
#   os.environ[...] = ...        一律覆蓋
# 因此 env.sh 會先 unset PSI_WEIGHT_CASES PSI_WEIGHT_FACTOR，避免殘留值影響下面三個 ACDC 類別。
# ----------------------------------------------------------------------------


# ===================== ACDC（500 epochs）=====================

class nnUNetTrainer_500epochs_psi1w3(nnUNetTrainer_PsiWeighted):
    """Ψ1 凸包比 前 30%（48 例）加權 3 倍"""
    # 沒有設定倍率 → 使用 DEFAULT_FACTOR = 3.0
    # NUM_EPOCHS 沿用基底類別的 500
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        os.environ.setdefault('PSI_WEIGHT_CASES',
                              os.path.join(_TF, 'wcases_psi1_convex.csv'))
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_500epochs_psi6w3(nnUNetTrainer_PsiWeighted):
    """Ψ6 連通元件 > 0（46 例）加權 3 倍"""
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        os.environ.setdefault('PSI_WEIGHT_CASES',
                              os.path.join(_TF, 'wcases_psi6_cc.csv'))
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_500epochs_psi7w3(nnUNetTrainer_PsiWeighted):
    """Ψ7 拓撲環 > 0（20 例）加權 3 倍"""
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        os.environ.setdefault('PSI_WEIGHT_CASES',
                              os.path.join(_TF, 'wcases_psi7_genus.csv'))
        super().__init__(plans, configuration, fold, dataset_json, device)


# ===================== BTCV（1000 epochs，對齊 anchor）=====================
# 這四個類別直接覆蓋環境變數，並把 NUM_EPOCHS 改成 1000（與 BTCV baseline 相同）

class nnUNetTrainer_btcv_psi1w3(nnUNetTrainer_PsiWeighted):
    """BTCV Ψ1 凸包比 加權 3 倍"""
    NUM_EPOCHS = 1000
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_btcv_psi1_convex.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '3.0'
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_btcv_psi6w3(nnUNetTrainer_PsiWeighted):
    """BTCV Ψ6 連通元件 加權 3 倍"""
    NUM_EPOCHS = 1000
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_btcv_psi6_cc.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '3.0'
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_btcv_psi8w3(nnUNetTrainer_PsiWeighted):
    """BTCV Ψ8 旋轉 加權 3 倍"""
    NUM_EPOCHS = 1000
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_btcv_psi8_rotation.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '3.0'
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_btcv_w1ctrl(nnUNetTrainer_PsiWeighted):
    """BTCV 對照組：拆 batch 逐樣本 loss，倍率 1.0

    注意：倍率 1.0 時 w 永遠全部等於 1，train_step 的 allclose 永遠成立，
    所以實際上「永遠走原始路徑」，並沒有拆 batch。
    這個模型等於 baseline 重訓一次，只能用來估計重訓的隨機誤差。
    """
    NUM_EPOCHS = 1000
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_btcv_psi8_rotation.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '1.0'
        super().__init__(plans, configuration, fold, dataset_json, device)


# ===================== ACDC：OOF 篩選的專家與對照組 =====================

class nnUNetTrainer_500epochs_psi10w3(nnUNetTrainer_PsiWeighted):
    """ACDC Ψ10 中心偏移 加權 3 倍（500 epochs）
    Ψ10 於 OOF 分析中為反向（AUC 0.295），清單取最低 30%。"""
    NUM_EPOCHS = 500
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_acdc_psi10_centroid.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '3.0'
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_500epochs_ps1ctrl(nnUNetTrainer_PsiWeighted):
    """ACDC 對照組：與 Ψ1 專家同一份清單、同一條逐樣本路徑，權重約為 1。
    倍率 1.0001 是為了避開 train_step 中 allclose(w, 1) 的原始路徑。

    與 Ψ1 專家比較：只差在「有沒有加權」
    與 baseline 比較：只差在「Dice 是逐張計算還是 batch 計算」
    （僅限含清單病例的 batch；ACDC batch 有 56 張，幾乎每個 batch 都符合）
    → 用來分辨 ACDC 的改善來自加權本身，還是來自逐張計算 loss。
    """
    NUM_EPOCHS = 500
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_psi1_convex.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '1.0001'
        super().__init__(plans, configuration, fold, dataset_json, device)
