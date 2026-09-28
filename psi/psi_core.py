#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psi_core.py — 十個符號約束（Ψ1–Ψ10）的算法核心
================================================================
從 run_all_psi.py 抽出，與資料集載入邏輯解耦。

數值行為與原始 run_all_psi.py 完全一致，未做任何修改：
  Ψ1  逐切片取「最大輪廓」，以面積加權平均，切片與輪廓面積門檻皆為 10 px
  Ψ6  對整個 3D volume 計算連通元件（非逐切片），空遮罩回傳 -1
  Ψ7  以 binary_fill_holes 求內部洞，逐切片算術平均，切片門檻 20 px
  Ψ2 / Ψ8 需要重新推論模型，無法由遮罩計算，回傳 NaN

遮罩約定：所有 mask 參數皆為 bool 型別的 numpy 陣列（2D 或 3D）。
3D 陣列的切片軸為最後一軸 (H, W, Z)。

用法:
    from psi_core import psi1_convex, psi6_cc, psi7_genus
    v = psi1_convex(pred == 3)
"""

import numpy as np
import cv2
from scipy.ndimage import label as cclabel, binary_fill_holes

EPS = 1e-6

# 面積 / 切片門檻（沿用原始設定，勿隨意更動，會改變歷史結果）
MIN_AREA_PSI1 = 10
MIN_AREA_PSI7 = 20


# ---------------------------------------------------------------- 內部工具
def _slices(mask):
    """把 2D 或 3D 遮罩統一成 2D 切片串列。3D 的切片軸為最後一軸。"""
    if mask.ndim == 2:
        return [mask]
    return [mask[:, :, z] for z in range(mask.shape[2])]


# ---------------------------------------------------------------- Ψ1
def psi1_convex(mask):
    """
    Ψ1 凸包比 = 凸包面積 / 實際面積。3D 逐切片以面積加權平均。

    每張切片只取「最大」外輪廓，避免小碎片主導比值；
    加權方式為 sum(凸包面積) / sum(輪廓面積)，故大切片權重較高。
    值 ≥ 1，越大代表形狀越不凸（邊界碎裂或凹陷）。
    """
    if mask is None or mask.sum() == 0:
        return np.nan
    ta = th = 0.0
    for s in _slices(mask):
        if s.sum() < MIN_AREA_PSI1:
            continue
        cs, _ = cv2.findContours(s.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        b = max(cs, key=cv2.contourArea)
        a = cv2.contourArea(b)
        if a < MIN_AREA_PSI1:
            continue
        ta += a
        th += cv2.contourArea(cv2.convexHull(b))
    return float(th / ta) if ta > 0 else np.nan


# ---------------------------------------------------------------- Ψ2
def psi2_shadow():
    """Ψ2 聲影一致性 — 需重新推論模型，無法由遮罩計算。"""
    return np.nan


# ---------------------------------------------------------------- Ψ3
def psi3_inclusion(inner, outer):
    """Ψ3 包含關係 = 落在外層之外的內層像素 / 內層像素。"""
    if inner is None or outer is None or inner.sum() == 0:
        return np.nan
    return float((inner & (~outer)).sum() / (inner.sum() + EPS))


# ---------------------------------------------------------------- Ψ4
def psi4_adjacency(a, b):
    """Ψ4 鄰接不重疊 = 交集 / 聯集。"""
    if a is None or b is None:
        return np.nan
    u = (a | b).sum()
    return float((a & b).sum() / (u + EPS)) if u > 0 else np.nan


# ---------------------------------------------------------------- Ψ5
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


# ---------------------------------------------------------------- Ψ6
def psi6_cc(mask):
    """
    Ψ6 連通元件數 - 1。對整個 3D volume 計算，非逐切片。

    完全無預測時回傳 -1（原始設計視為「比碎裂更嚴重」的狀態）。
    注意：case-level 以 nanmax 聚合時，-1 會被其他結構的正值蓋過，
    此類「整個結構消失」的樣本需另以 n_nan 或 psi6 == -1 另行辨識。
    """
    if mask is None:
        return np.nan
    if mask.sum() == 0:
        return -1.0
    return float(cclabel(mask)[1] - 1)


# ---------------------------------------------------------------- Ψ7
def psi7_genus(ring):
    """
    Ψ7 拓撲環 = |內部洞數 - 1|。3D 逐切片算術平均。

    以 binary_fill_holes 填滿後與原遮罩相減得到「洞」，再數連通元件。
    心肌在短軸切面應恰為一個環（1 個洞），故理想值為 0。
    """
    if ring is None or ring.sum() == 0:
        return np.nan
    vals = []
    for s in _slices(ring):
        if s.sum() < MIN_AREA_PSI7:
            continue
        holes = binary_fill_holes(s) & (~s)
        vals.append(abs(cclabel(holes)[1] - 1))
    return float(np.mean(vals)) if vals else np.nan


# ---------------------------------------------------------------- Ψ8
def psi8_rotation():
    """Ψ8 旋轉一致性 — 需重新推論模型，無法由遮罩計算。"""
    return np.nan


# ---------------------------------------------------------------- Ψ9
def psi9_nesting(inner, outer):
    """Ψ9 巢狀關係 — 與 Ψ3 同式，套用於不同解剖結構。"""
    return psi3_inclusion(inner, outer)


# ---------------------------------------------------------------- Ψ10
def psi10_centroid(target, ref, diag):
    """Ψ10 中心偏移 = 兩結構質心距離 / 正規化尺度。"""
    if target is None or ref is None or target.sum() == 0 or ref.sum() == 0:
        return np.nan
    ct = np.argwhere(target).mean(axis=0)
    cr = np.argwhere(ref).mean(axis=0)
    return float(np.linalg.norm(ct - cr) / (diag + EPS))


# ---------------------------------------------------------------- 名稱表
PSI_NAMES = ['psi1_convex', 'psi2_shadow', 'psi3_inclusion', 'psi4_adjacency',
             'psi5_exclusion', 'psi6_cc', 'psi7_genus', 'psi8_rotation',
             'psi9_nesting', 'psi10_centroid']

PSI_DESC = {
    'psi1_convex':    'Ψ1  凸包比',
    'psi2_shadow':    'Ψ2  聲影一致性（需模型）',
    'psi3_inclusion': 'Ψ3  包含關係',
    'psi4_adjacency': 'Ψ4  鄰接不重疊',
    'psi5_exclusion': 'Ψ5  器官互斥',
    'psi6_cc':        'Ψ6  連通元件',
    'psi7_genus':     'Ψ7  拓撲環',
    'psi8_rotation':  'Ψ8  旋轉一致性（需模型）',
    'psi9_nesting':   'Ψ9  巢狀關係',
    'psi10_centroid': 'Ψ10 中心偏移',
}

# 不需要模型、純遮罩幾何即可計算的 Ψ
MASK_ONLY = ['psi1_convex', 'psi3_inclusion', 'psi4_adjacency',
             'psi5_exclusion', 'psi6_cc', 'psi7_genus',
             'psi9_nesting', 'psi10_centroid']