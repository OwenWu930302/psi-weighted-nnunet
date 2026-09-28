"""
加權清單產生與驗證
============================================================
依 docs/METHODS.md 第 5.4 節的規則，從病例層級 Ψ 表重新產生加權清單，
並與實際訓練使用的清單（tree_features/wcases_*.csv）逐一比對。

規則：
  連續值 proxy：取前 30%（並列全部納入）
  離散值 proxy（大量為 0）：取 > 0
  反向 proxy（AUC < 0.5）：取最低 30%（並列全部納入）

用法：
  python3 make_weight_lists.py            # 只比對，不覆蓋既有清單
  python3 make_weight_lists.py --write    # 將重新產生的清單寫到 regenerated/
"""
import os
import sys

import pandas as pd

H = os.path.expanduser('~/桌面/論文')
TF = f'{H}/tree_features'
OUT = f'{TF}/regenerated'

# (資料集, 病例層級 Ψ 表, 欄位, 規則, 實際使用的清單檔, 預測來源說明)
SPECS = [
    ('BTCV', 'caselevel_btcv.csv',       'psi1_convex',    'top30',    'wcases_btcv_psi1_convex.csv',    '5 折 OOF'),
    ('BTCV', 'caselevel_btcv.csv',       'psi6_cc',        'top30',    'wcases_btcv_psi6_cc.csv',        '5 折 OOF'),
    ('BTCV', 'caselevel_btcv.csv',       'psi8_rotation',  'top30',    'wcases_btcv_psi8_rotation.csv',  '5 折 OOF'),
    ('ACDC', 'caselevel_acdc_oof.csv',   'psi10_centroid', 'bottom30', 'wcases_acdc_psi10_centroid.csv', '5 折 OOF'),
    # ACDC Ψ1/6/7：以 all 模式的訓練集預測產生（補做 OOF 之前）
    ('ACDC', 'caselevel_acdc_train.csv', 'psi1_convex',    'top30',    'wcases_psi1_convex.csv',         'all 模式訓練集預測'),
    ('ACDC', 'caselevel_acdc_train.csv', 'psi6_cc',        'gt0',      'wcases_psi6_cc.csv',             'all 模式訓練集預測'),
    ('ACDC', 'caselevel_acdc_train.csv', 'psi7_genus',     'gt0',      'wcases_psi7_genus.csv',          'all 模式訓練集預測'),
]


def select(d, col, rule):
    k = int(round(len(d) * 0.30))
    if rule == 'top30':
        thr = d[col].sort_values(ascending=False).iloc[k - 1]
        return d[d[col] >= thr], f'>= {thr:.6g}'
    if rule == 'bottom30':
        thr = d[col].sort_values(ascending=True).iloc[k - 1]
        return d[d[col] <= thr], f'<= {thr:.6g}'
    if rule == 'gt0':
        return d[d[col] > 0], '> 0'
    raise ValueError(rule)


def main(write):
    if write:
        os.makedirs(OUT, exist_ok=True)
    print(f'{"資料集":<5} {"proxy":<16} {"規則":<9} {"門檻":<14} {"重產":>4} {"實際":>4} {"一致":<6} 來源')
    for ds, table, col, rule, used, src in SPECS:
        tp = f'{TF}/{table}'
        up = f'{TF}/{used}'
        if not os.path.exists(tp):
            n_used = len(pd.read_csv(up)) if os.path.exists(up) else '-'
            print(f'{ds:<5} {col:<16} {rule:<9} {"—":<14} {"—":>4} {n_used:>4} {"未驗證":<6} '
                  f'{src}（缺 {table}）')
            continue
        d = pd.read_csv(tp)
        s, thr = select(d, col, rule)
        new = set(s['case'].astype(str))
        old = set(pd.read_csv(up)['case'].astype(str)) if os.path.exists(up) else set()
        ok = '✓' if new == old else f'✗ 差 {len(new ^ old)}'
        print(f'{ds:<5} {col:<16} {rule:<9} {thr:<14} {len(new):>4} {len(old):>4} {ok:<6} {src}')
        if write:
            s[['case', col, 'dice']].sort_values(col, ascending=(rule == 'bottom30')) \
                .to_csv(f'{OUT}/{used}', index=False)
    if write:
        print(f'\n重新產生的清單已寫入 {OUT}（未覆蓋實際使用的清單）')


if __name__ == '__main__':
    main('--write' in sys.argv)
