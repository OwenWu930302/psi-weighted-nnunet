import json, glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ORG=['Aorta','Gallbl','KidL','KidR','Liver','Pancr','Spleen','Stomach']
R='/home/tkyin/桌面/論文/nnUNet_data/nnUNet_results/Dataset115_BTCV'

def load(d):
    out={}
    for f in sorted(glob.glob(f'{R}/{d}/fold_*/validation/summary.json')):
        for c in json.load(open(f))['metric_per_case']:
            n=c['reference_file'].split('/')[-1].replace('.nii.gz','')
            out[n]=[c['metrics'][k]['Dice'] for k in sorted(c['metrics'], key=int)]
    return out

base='nnUNetTrainer__nnUNetPlans__2d'
exp=f'nnUNetTrainer_btcv_{sys.argv[1]}__nnUNetPlans__2d'
A,B=load(base),load(exp)
ks=sorted(set(A)&set(B)); print(f'{exp}  n={len(ks)}')
a=np.array([A[k] for k in ks]); b=np.array([B[k] for k in ks])
am,bm=np.nanmean(a,1),np.nanmean(b,1)
print(f'anchor {am.mean():.4f}  exp {bm.mean():.4f}  Δ={bm.mean()-am.mean():+.4f}')
print(f'升/降 {(bm>am).sum()}/{(bm<am).sum()}  p={wilcoxon(am,bm).pvalue:.4f}')
for i,o in enumerate(ORG):
    m=~(np.isnan(a[:,i])|np.isnan(b[:,i]))
    print(f'  {o:<9}{np.nanmean(a[:,i]):.4f} → {np.nanmean(b[:,i]):.4f}  Δ={np.nanmean(b[:,i])-np.nanmean(a[:,i]):+.4f}  p={wilcoxon(a[m,i],b[m,i]).pvalue:.4f}')
o=sorted(zip(ks,am,bm), key=lambda t:t[1]); k=int(round(len(ks)*0.30))
d=np.array([y-x for _,x,y in o[:k]])
print(f'最差 {k} 例 平均 {d.mean():+.4f}  改善 {(d>0).sum()}/{k}')
for kk,x,y in o: print(f'  {kk:<12}{x:.4f} → {y:.4f}  {y-x:+.4f}')
