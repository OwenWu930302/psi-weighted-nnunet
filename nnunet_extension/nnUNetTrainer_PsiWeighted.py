import os
import json
import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


# ====================================== ======================
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


def _load_case_list(path):
    """讀取要加權的 case 清單，支援 csv（含 case 欄）與 json 陣列。"""
    if path is None or not os.path.exists(path):
        return set()
    if path.endswith('.json'):
        return set(json.load(open(path)))
    cases = []
    with open(path) as f:
        header = f.readline().strip().split(',')
        try:
            idx = header.index('case')
        except ValueError:
            idx = 0
        for line in f:
            line = line.strip()
            if line:
                cases.append(line.split(',')[idx])
    return set(cases)


def _normalise(key):
    """
    把 dataloader 的 key 正規化成 case id。

    preprocessed 資料夾裡的檔名即為 case id（例如 ACDC_001.b2nd），
    dataloader 傳來的 key 已經是不含副檔名的 ACDC_001，故只需處理
    可能出現的副檔名。這裡刻意不去猜測、剝除任何 _NNNN 後綴，
    因為 case id 本身就以數字結尾（ACDC_001），剝掉會直接對不上清單。
    """
    k = str(key)
    for ext in ('.b2nd', '.npz', '.npy', '.pkl'):
        if k.endswith(ext):
            k = k[:-len(ext)]
            break
    if k.endswith('_seg'):
        k = k[:-4]
    return k


class nnUNetTrainer_PsiWeighted(nnUNetTrainer):
    """Ψ 加權訓練器基底。子類別指定 CASE_LIST_ENV 或直接覆寫 psi_cases。"""

    CASE_LIST_ENV = 'PSI_WEIGHT_CASES'
    FACTOR_ENV = 'PSI_WEIGHT_FACTOR'
    DEFAULT_FACTOR = 3.0
    NUM_EPOCHS = 500

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.NUM_EPOCHS

        self.psi_cases = _load_case_list(os.environ.get(self.CASE_LIST_ENV))
        self.psi_factor = float(os.environ.get(self.FACTOR_ENV, self.DEFAULT_FACTOR))

        # 統計用：確認加權真的有生效，避免清單對不上而靜默失效
        self._psi_hit = 0
        self._psi_total = 0

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f'[Ψ加權] 清單 {os.environ.get(self.CASE_LIST_ENV)}　'
            f'case 數 {len(self.psi_cases)}　倍率 {self.psi_factor}')
        if not self.psi_cases:
            self.print_to_log_file(
                '[Ψ加權] 警告：清單為空，本次訓練等同未加權的基準模型')

    def _sample_slice(self, x, i):
        """從 output/target 取出第 i 個樣本，保留 batch 維度與 deep supervision 結構。"""
        if isinstance(x, (list, tuple)):
            return [t[i:i + 1] for t in x]
        return x[i:i + 1]

    def _weights_for(self, keys, n):
        w = torch.ones(n, dtype=torch.float32)
        if not self.psi_cases or keys is None:
            return w
        for i in range(min(n, len(keys))):
            if _normalise(keys[i]) in self.psi_cases:
                w[i] = self.psi_factor
                self._psi_hit += 1
        self._psi_total += n
        return w

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch.get('keys')

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        n = data.shape[0]
        w = self._weights_for(keys, n).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            
            if torch.allclose(w, torch.ones_like(w)):
                # 這個 batch 沒有任何加權樣本，走原始路徑（較快且數值一致）
                l = self.loss(output, target)
            else:
                # 逐樣本算 loss 後加權平均，分母為權重和以維持尺度
                total = 0.0
                for i in range(n):
                    li = self.loss(self._sample_slice(output, i),
                                   self._sample_slice(target, i))
                    total = total + w[i] * li
                l = total / w.sum()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if self._psi_total:
            self.print_to_log_file(
                f'[Ψ加權] 本輪加權樣本 {self._psi_hit}/{self._psi_total} '
                f'({self._psi_hit / self._psi_total:.1%})')
        self._psi_hit = 0
        self._psi_total = 0


# ============================================================
# 三個 proxy 各自的專家：清單路徑寫死，避免忘記設環境變數
# ============================================================

_TF = os.path.expanduser('~/桌面/論文/tree_features')


class nnUNetTrainer_500epochs_psi1w3(nnUNetTrainer_PsiWeighted):
    """Ψ1 凸包比 前 30%（48 例）加權 3 倍"""
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
    """BTCV 對照組：拆 batch 逐樣本 loss，倍率 1.0"""
    NUM_EPOCHS = 1000
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_btcv_psi8_rotation.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '1.0'
        super().__init__(plans, configuration, fold, dataset_json, device)


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
    倍率 1.0001 是為了避開 train_step 中 allclose(w, 1) 的原始路徑。"""
    NUM_EPOCHS = 500
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        os.environ['PSI_WEIGHT_CASES'] = os.path.join(_TF, 'wcases_psi1_convex.csv')
        os.environ['PSI_WEIGHT_FACTOR'] = '1.0001'
        super().__init__(plans, configuration, fold, dataset_json, device)
