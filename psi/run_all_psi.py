#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
十個符號約束（Ψ1–Ψ10）通用計算 + 樹狀級聯分析

用法:
    conda activate expertree
    source <倉庫>/env.sh
    python3 run_all_psi.py

輸出:
    tree_features/all_psi_<dataset>.csv     每個樣本-結構的十個 Ψ 值
    tree_features/cascade_<dataset>.csv     樹狀級聯：每層挑出多少
    終端會印出完整的級聯表

說明:
    Ψ2（陰影一致性）與 Ψ8（旋轉一致性）需要把影像擾動後重新推論模型，
    無法只從預測遮罩計算。程式會將其標為 NaN 並在報告中註明。
"""
import os, sys, glob, csv, json, itertools
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('缺少 opencv，請執行: pip install opencv-python')
try:
    import nibabel as nib
except ImportError:
    nib = None
from scipy.ndimage import label as cclabel, binary_fill_holes
from PIL import Image

R = os.environ.get('PSI_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(R, 'tree_features')
os.makedirs(OUT, exist_ok=True)
DELTA = 0.80
EPS = 1e-6

# ============================================================
# 十個 Ψ 的通用實作
# ============================================================

def psi1_convex(mask):
    """Ψ1 凸包比 = 凸包面積 / 實際面積。3D 逐切片以面積加權平均。"""
    if mask.sum() == 0:
        return np.nan
    if mask.ndim == 2:
        slices = [mask]
    else:
        slices = [mask[:, :, z] for z in range(mask.shape[2])]
    ta = th = 0.0
    for s in slices:
        if s.sum() < 10:
            continue
        cs, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        b = max(cs, key=cv2.contourArea)
        a = cv2.contourArea(b)
        if a < 10:
            continue
        ta += a
        th += cv2.contourArea(cv2.convexHull(b))
    return float(th / ta) if ta > 0 else np.nan


def psi2_shadow():
    """Ψ2 聲影一致性 — 需重新推論模型，無法由遮罩計算。"""
    return np.nan


def psi3_inclusion(inner, outer):
    """Ψ3 包含關係 = 落在外層之外的內層像素 / 內層像素。"""
    if inner is None or outer is None or inner.sum() == 0:
        return np.nan
    return float((inner & (~outer)).sum() / (inner.sum() + EPS))


def psi4_adjacency(a, b):
    """Ψ4 鄰接不重疊 = 交集 / 聯集。"""
    if a is None or b is None:
        return np.nan
    u = (a | b).sum()
    return float((a & b).sum() / (u + EPS)) if u > 0 else np.nan


def psi5_exclusion(target, others):
    """Ψ5 互斥 = 該結構與其他所有結構的最大重疊比。"""
    if target is None or target.sum() == 0:
        return np.nan
    m = 0.0
    for o in others:
        if o is None:
            continue
        u = (target | o).sum()
        if u > 0:
            m = max(m, (target & o).sum() / u)
    return float(m)


def psi6_cc(mask):
    """Ψ6 連通元件數 - 1。完全無預測回傳 -1（比碎裂更嚴重）。"""
    if mask is None:
        return np.nan
    if mask.sum() == 0:
        return -1.0
    return float(cclabel(mask)[1] - 1)


def psi7_genus(ring):
    """Ψ7 拓撲環 = |內部洞數 - 1|。3D 逐切片平均。"""
    if ring is None or ring.sum() == 0:
        return np.nan
    if ring.ndim == 2:
        slices = [ring]
    else:
        slices = [ring[:, :, z] for z in range(ring.shape[2])]
    vals = []
    for s in slices:
        if s.sum() < 20:
            continue
        holes = binary_fill_holes(s) & (~s)
        vals.append(abs(cclabel(holes)[1] - 1))
    return float(np.mean(vals)) if vals else np.nan


def psi8_rotation():
    """Ψ8 旋轉一致性 — 需重新推論模型，無法由遮罩計算。"""
    return np.nan


def psi9_nesting(inner, outer):
    """Ψ9 巢狀關係 — 與 Ψ3 同式，套用於不同解剖結構。"""
    return psi3_inclusion(inner, outer)


def psi10_centroid(target, ref, diag):
    """Ψ10 中心偏移 = 兩結構質心距離 / 正規化尺度。"""
    if target is None or ref is None or target.sum() == 0 or ref.sum() == 0:
        return np.nan
    ct = np.argwhere(target).mean(axis=0)
    cr = np.argwhere(ref).mean(axis=0)
    return float(np.linalg.norm(ct - cr) / (diag + EPS))


PSI_NAMES = ['psi1_convex', 'psi2_shadow', 'psi3_inclusion', 'psi4_adjacency',
             'psi5_exclusion', 'psi6_cc', 'psi7_genus', 'psi8_rotation',
             'psi9_nesting', 'psi10_centroid']

PSI_DESC = {
    'psi1_convex':   'Ψ1  凸包比',
    'psi2_shadow':   'Ψ2  聲影一致性（需模型）',
    'psi3_inclusion':'Ψ3  包含關係',
    'psi4_adjacency':'Ψ4  鄰接不重疊',
    'psi5_exclusion':'Ψ5  器官互斥',
    'psi6_cc':       'Ψ6  連通元件',
    'psi7_genus':    'Ψ7  拓撲環',
    'psi8_rotation': 'Ψ8  旋轉一致性（需模型）',
    'psi9_nesting':  'Ψ9  巢狀關係',
    'psi10_centroid':'Ψ10 中心偏移',
}

# ============================================================
# 資料集設定
# ============================================================
# structures : {label 值: 結構名}
# nested     : (內層名, 外層名) 或 None — 供 Ψ3/Ψ9 使用
# ring       : 環狀結構名 或 None    — 供 Ψ7 使用
# ref        : Ψ10 的參考結構名


def load_refuge():
    src = os.path.join(R, 'FunduSegmenter/repro_notes')
    dice_csv = os.path.join(src, 'per_image_dice_designed.csv')
    if not os.path.exists(dice_csv):
        return None
    dice = {}
    for r in csv.DictReader(open(dice_csv)):
        n = r['filename'].replace('.png', '')
        dice[n] = {'OD': float(r['od_dice']), 'OC': float(r['oc_dice'])}
    rows = []
    for f in sorted(glob.glob(os.path.join(src, 'segmap_designed/map/*.png'))):
        n = os.path.basename(f)[:-4]
        if n not in dice:
            continue
        a = np.asarray(Image.open(f))
        if a.ndim == 3:
            a = a[:, :, 0]
        cup = (a == 0)
        disc = (a == 0) | (a == 128)
        rim = (a == 128)                      # 視盤環，是包住視杯的環狀結構
        yield_masks = {'OC': cup, 'OD': disc}
        diag = np.sqrt(sum(s ** 2 for s in a.shape))
        for name, m in yield_masks.items():
            rows.append(dict(
                case=n, structure=name,
                psi1_convex=psi1_convex(m),
                psi2_shadow=psi2_shadow(),
                psi3_inclusion=psi3_inclusion(cup, disc) if name == 'OC' else np.nan,
                psi4_adjacency=psi4_adjacency(cup, rim),
                psi5_exclusion=psi5_exclusion(m, [rim] if name == 'OC' else [cup]),
                psi6_cc=psi6_cc(m),
                psi7_genus=psi7_genus(rim),
                psi8_rotation=psi8_rotation(),
                psi9_nesting=psi9_nesting(cup, disc) if name == 'OC' else np.nan,
                psi10_centroid=psi10_centroid(m, disc, diag),
                voxels=int(m.sum()),
                dice=dice[n][name], Z=int(dice[n][name] < DELTA)))
    return rows


def load_busi():
    d = os.path.join(R, 'nnUNet/HA-Net/results')
    dice_csv = os.path.join(d, 'percase_dice.csv')
    if not os.path.exists(dice_csv):
        return None
    dice = {r['filename'].rsplit('.', 1)[0]: float(r['dice'])
            for r in csv.DictReader(open(dice_csv))}
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'masks/*'))):
        n = os.path.basename(f).rsplit('.', 1)[0]
        if n not in dice:
            continue
        a = np.asarray(Image.open(f))
        if a.ndim == 3:
            a = a[:, :, 0]
        m = a > (a.max() / 2) if a.max() > 1 else a > 0
        diag = np.sqrt(sum(s ** 2 for s in a.shape))
        rows.append(dict(
            case=n, structure='Tumour',
            psi1_convex=psi1_convex(m),
            psi2_shadow=psi2_shadow(),
            psi3_inclusion=np.nan,          # 單一結構，無巢狀關係
            psi4_adjacency=np.nan,          # 單一結構，無相鄰結構
            psi5_exclusion=np.nan,          # 單一結構，無互斥對象
            psi6_cc=psi6_cc(m),
            psi7_genus=psi7_genus(m),       # 腫塊非環狀，通常為 1（無洞）
            psi8_rotation=psi8_rotation(),
            psi9_nesting=np.nan,
            psi10_centroid=np.nan,          # 單一結構，無參考對象
            voxels=int(m.sum()),
            dice=dice[n], Z=int(dice[n] < DELTA)))
    return rows


def load_nnunet_3d(pred_glob, summary_glob, labels, ring=None, nested=None, ref=None):
    if nib is None:
        return None
    files = sorted(glob.glob(pred_glob))
    summaries = sorted(glob.glob(summary_glob))
    if not files or not summaries:
        return None
    dice = {}
    for sf in summaries:
        for cse in json.load(open(sf))['metric_per_case']:
            n = os.path.basename(cse['reference_file']).replace('.nii.gz', '')
            dice[n] = {labels[k]: cse['metrics'][str(k)]['Dice'] for k in labels}
    rows = []
    for f in files:
        n = os.path.basename(f).replace('.nii.gz', '')
        if n not in dice:
            continue
        a = np.asarray(nib.load(f).dataobj)
        msk = {nm: (a == k) for k, nm in labels.items()}
        diag = np.sqrt(sum(s ** 2 for s in a.shape))
        ring_m = msk.get(ring) if ring else None
        p7 = psi7_genus(ring_m)
        ref_m = msk.get(ref) if ref else None
        for nm, m in msk.items():
            others = [msk[o] for o in msk if o != nm]
            if nested and nm == nested[0]:
                p3 = psi3_inclusion(msk[nested[0]], msk[nested[1]])
            else:
                p3 = np.nan
            p4 = max([psi4_adjacency(m, o) for o in others]) if others else np.nan
            rows.append(dict(
                case=n, structure=nm,
                psi1_convex=psi1_convex(m),
                psi2_shadow=psi2_shadow(),
                psi3_inclusion=p3,
                psi4_adjacency=p4,
                psi5_exclusion=psi5_exclusion(m, others),
                psi6_cc=psi6_cc(m),
                psi7_genus=p7,
                psi8_rotation=psi8_rotation(),
                psi9_nesting=p3,
                psi10_centroid=psi10_centroid(m, ref_m, diag),
                voxels=int(m.sum()),
                dice=dice[n][nm], Z=int(dice[n][nm] < DELTA)))
        print('.', end='', flush=True)
    print()
    return rows


def load_btcv():
    base = os.path.join(R, 'nnUNet_data/nnUNet_results/Dataset115_BTCV/nnUNetTrainer__nnUNetPlans__2d')
    return load_nnunet_3d(
        os.path.join(base, 'fold_*/validation/*.nii.gz'),
        os.path.join(base, 'fold_*/validation/summary.json'),
        {1: 'Aorta', 2: 'Gallbl', 3: 'KidL', 4: 'KidR',
         5: 'Liver', 6: 'Pancr', 7: 'Spleen', 8: 'Stomach'},
        ring=None, nested=None, ref='Liver')


def load_acdc():
    d = os.path.join(R, 'nnUNet/acdc_test_pred')
    return load_nnunet_3d(
        os.path.join(d, '*.nii.gz'),
        os.path.join(d, 'summary.json'),
        {1: 'RV', 2: 'Myo', 3: 'LV'},
        ring='Myo', nested=('LV', 'Myo'), ref='LV')


def load_acdc_oof():
    """ACDC 5 折 out-of-fold 預測（每例由沒看過它的模型預測）。
    與 load_acdc() 的差別：後者讀的是 -f all 模型對測試集的預測。"""
    base = os.path.join(R, 'nnUNet_data/nnUNet_results/Dataset116_ACDC/'
                           'nnUNetTrainer_500epochs__nnUNetPlans__2d')
    return load_nnunet_3d(
        os.path.join(base, 'fold_[0-4]/validation/*.nii.gz'),
        os.path.join(base, 'fold_[0-4]/validation/summary.json'),
        {1: 'RV', 2: 'Myo', 3: 'LV'},
        ring='Myo', nested=('LV', 'Myo'), ref='LV')


def load_camus():
    return None     # 預測遮罩未保存，需重跑 test_video.py（opt.visual=True）


LOADERS = [('REFUGE', load_refuge), ('BUSI', load_busi),
           ('BTCV', load_btcv), ('ACDC', load_acdc),
           ('ACDC_OOF', load_acdc_oof), ('CAMUS', load_camus)]

# ============================================================
# 統計工具
# ============================================================

def mutual_info(z, zh):
    m = 0.0
    for i in (0, 1):
        for j in (0, 1):
            p = np.mean((z == i) & (zh == j))
            if p > 0:
                m += p * np.log2(p / (np.mean(z == i) * np.mean(zh == j) + 1e-12))
    return m


def f1_of(z, zh):
    tp = ((z == 1) & (zh == 1)).sum()
    fp = ((z == 0) & (zh == 1)).sum()
    fn = ((z == 1) & (zh == 0)).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def best_threshold(v, z):
    """回傳 (MI, 方向, 門檻, 旗標)。無變異時回傳 None。"""
    ok = ~np.isnan(v)
    # 全距 < 1e-6 視為恆定（與 recompute_caselevel.py、final_report.py 一致）：
    # ACDC 的 Ψ3／Ψ9 = n_LV / (n_LV + 1e-6)，相異值很多但全距僅約 1e-9，
    # 若只看相異值數量，會把 LV 大小誤當成 Ψ3 的訊號（見 docs/CODE.md 第 7.1 節）
    if ok.sum() < 10 or len(np.unique(v[ok])) < 3 or np.ptp(v[ok]) < 1e-6:
        return None
    cand = np.unique(np.percentile(v[ok], np.linspace(1, 99, 99)))
    best = None
    for d in ('>', '<'):
        for tau in cand:
            zh = np.zeros(len(v), int)
            zh[ok] = (v[ok] > tau) if d == '>' else (v[ok] < tau)
            s = zh.sum()
            if s == 0 or s == ok.sum():
                continue
            mi = mutual_info(z, zh)
            if best is None or mi > best[0]:
                best = (mi, d, tau, zh)
    return best


# ============================================================
# 主流程
# ============================================================

def analyse(name, rows):
    if not rows:
        return None
    keys = list(rows[0].keys())
    path = os.path.join(OUT, 'all_psi_%s.csv' % name.lower())
    with open(path, 'w', newline='') as fo:
        w = csv.DictWriter(fo, keys)
        w.writeheader()
        w.writerows(rows)

    Z = np.array([r['Z'] for r in rows])
    n, npos = len(Z), int(Z.sum())
    base_f1 = 2 * (npos / n) / (1 + npos / n) if npos else 0.0

    print('\n' + '=' * 78)
    print('%s   樣本 %d   失敗 %d (%.1f%%)   基準線 F1 = %.4f' %
          (name, n, npos, 100 * npos / n, base_f1))
    print('=' * 78)

    print('\n[1] 十個特徵的可用性')
    print('%-26s %10s %10s %8s %8s  %s' % ('特徵', 'mean', 'sd', '相異值', 'NaN', '狀態'))
    usable = {}
    for k in PSI_NAMES:
        v = np.array([r[k] for r in rows], float)
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            reason = '需重新推論模型' if k in ('psi2_shadow', 'psi8_rotation') else '此資料集無對應結構'
            print('%-26s %10s %10s %8s %8d  %s' % (PSI_DESC[k], '—', '—', '—', len(v), reason))
            continue
        vv = v[ok]
        u = len(np.unique(vv))
        bt = best_threshold(v, Z) if npos else None
        st = '可用' if bt else ('恆定，無訊號' if u == 1 or np.ptp(vv) < 1e-6 else '變異不足')
        if bt:
            usable[k] = bt
        print('%-26s %10.5f %10.5f %8d %8d  %s' %
              (PSI_DESC[k], vv.mean(), vv.std(), u, (~ok).sum(), st))

    if not usable or npos == 0:
        print('\n[2] 無法進行級聯分析：%s' %
              ('無失敗樣本（Z 全為 0）' if npos == 0 else '無任何可用特徵'))
        return dict(name=name, n=n, npos=npos, base_f1=base_f1, cascade=[])

    order = sorted(usable, key=lambda k: -usable[k][0])

    print('\n[3] 樹狀級聯（OR 邏輯：任一層觸發即標記）')
    hdr = '%-5s %-26s %8s %8s %8s %8s %8s %8s' % (
        '層', '特徵', '本層標記', '累積標記', '累積TP', '新增TP', '精確率', 'F1')
    print(hdr)
    print('-' * len(hdr))
    cum = np.zeros(n, int)
    prev_tp = 0
    cascade = []
    for lv, k in enumerate(order, 1):
        mi, d, tau, zh = usable[k]
        cum = ((cum == 1) | (zh == 1)).astype(int)
        tp = int(((Z == 1) & (cum == 1)).sum())
        fp = int(((Z == 0) & (cum == 1)).sum())
        pre = tp / (tp + fp + 1e-9)
        f1 = f1_of(Z, cum)
        print('%-5d %-26s %8d %8d %8d %8d %8.4f %8.4f' %
              (lv, PSI_DESC[k], int(zh.sum()), int(cum.sum()), tp, tp - prev_tp, pre, f1))
        cascade.append(dict(dataset=name, level=lv, psi=k, desc=PSI_DESC[k],
                            direction=d, threshold=float(tau), mi=float(mi),
                            layer_flagged=int(zh.sum()), cum_flagged=int(cum.sum()),
                            cum_tp=tp, new_tp=tp - prev_tp,
                            precision=round(pre, 4), f1=round(f1, 4),
                            baseline_f1=round(base_f1, 4)))
        prev_tp = tp
    print('%-5s %-26s %8s %8d %8d %8s %8.4f %8.4f' %
          ('—', '基準線（全部標記）', '—', n, npos, '—', npos / n, base_f1))

    with open(os.path.join(OUT, 'cascade_%s.csv' % name.lower()), 'w', newline='') as fo:
        w = csv.DictWriter(fo, list(cascade[0].keys()))
        w.writeheader()
        w.writerows(cascade)

    print('\n[4] 各層門檻（畫樹狀圖用）')
    for cc in cascade:
        print('    L%d  %-26s  %s %.5f   MI=%.4f' %
              (cc['level'], cc['desc'], cc['direction'], cc['threshold'], cc['mi']))
    return dict(name=name, n=n, npos=npos, base_f1=base_f1, cascade=cascade)


def main():
    print('十個符號約束通用計算　δ = %.2f' % DELTA)
    results = []
    for name, fn in LOADERS:
        print('\n>>> %s' % name, end=' ')
        try:
            rows = fn()
        except Exception as e:
            print('讀取失敗:', type(e).__name__, e)
            continue
        if rows is None:
            print('— 資料不可用（預測遮罩未保存或路徑不存在）')
            continue
        print('讀入 %d 個樣本-結構組合' % len(rows))
        res = analyse(name, rows)
        if res:
            results.append(res)

    print('\n\n' + '=' * 78)
    print('總結')
    print('=' * 78)
    print('%-10s %8s %8s %10s %10s %10s' %
          ('資料集', '樣本', '失敗', '基準線F1', '最佳F1', '最佳層'))
    for r in results:
        if r['cascade']:
            best = max(r['cascade'], key=lambda x: x['f1'])
            mark = '  ← 超越基準線' if best['f1'] > r['base_f1'] else ''
            print('%-10s %8d %8d %10.4f %10.4f %10d%s' %
                  (r['name'], r['n'], r['npos'], r['base_f1'], best['f1'], best['level'], mark))
        else:
            print('%-10s %8d %8d %10.4f %10s %10s' %
                  (r['name'], r['n'], r['npos'], r['base_f1'], '—', '—'))
    print('\n輸出目錄: %s' % OUT)
    print('  all_psi_<dataset>.csv   每個樣本的十個 Ψ 值')
    print('  cascade_<dataset>.csv   樹狀級聯，含每層門檻與標記數')


if __name__ == '__main__':
    main()