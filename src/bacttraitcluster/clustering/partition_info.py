
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

def feature_partition_nmi(df: pd.DataFrame, labels: np.ndarray, id_col: str='Strain_ID') -> pd.DataFrame:
    lab=np.asarray(labels)
    rows=[]
    for c in df.columns:
        s=df[c]
        m=s.notna()
        if m.sum()<5 or s[m].nunique()<2: continue
        x=s[m].astype(int).to_numpy(); y=lab[m.to_numpy()]
        rows.append({'Feature':c,'N':int(m.sum()),'NMI_C_feature':float(normalized_mutual_info_score(y,x)),'MI':float(mutual_info_score(y,x))})
    out=pd.DataFrame(rows)
    return out.sort_values('NMI_C_feature', ascending=False) if not out.empty else out


def partition_info_summary(labels: np.ndarray, X: np.ndarray) -> dict:
    lab=np.asarray(labels); X=np.asarray(X,float)
    unexpl=0.0
    for j in range(X.shape[1]):
        s=X[:,j]
        m=~np.isnan(s)
        if m.sum()<2: continue
        for cl in np.unique(lab[m]):
            v=s[m][lab[m]==cl]
            if len(v)==0: continue
            p=np.clip(np.nanmean(v),1e-9,1-1e-9)
            unexpl += -(len(v))*(p*np.log2(p)+(1-p)*np.log2(1-p))
    return {'Partition_Unexplained_Bits':float(expl if (expl:=unexpl) or True else unexpl)}
