
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering


def _center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    return H @ K @ H


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    Kc, Lc = _center(K), _center(L)
    return float(np.sum(Kc*Lc) / max((K.shape[0]-1)**2,1))


def cka(K: np.ndarray, L: np.ndarray) -> float:
    num = hsic(K,L)
    den = np.sqrt(max(hsic(K,K),1e-12)*max(hsic(L,L),1e-12))
    return float(num/den)


def cka_matrix(kernels: dict[str,np.ndarray]) -> pd.DataFrame:
    names=list(kernels)
    m=np.zeros((len(names),len(names)),float)
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            m[i,j]=cka(kernels[a], kernels[b]) if i!=j else 1.0
    return pd.DataFrame(m,index=names,columns=names)


def _spectral_gap_quality(K: np.ndarray, k: int) -> float:
    K=(K+K.T)/2
    vals=eigh(K, eigvals_only=True)
    vals=np.sort(vals)[::-1]
    if len(vals)<=k: return float(vals[0]) if len(vals) else 0.0
    return float(vals[k-1]/(abs(vals[k])+1e-9))


def optimize_kernel_weights(kernels: dict[str,np.ndarray], k: int=2, cka_df: pd.DataFrame|None=None):
    names=list(kernels)
    Ks=[(np.asarray(kernels[n],float)+np.asarray(kernels[n],float).T)/2 for n in names]
    L=len(Ks)
    if cka_df is None:
        cka_df = cka_matrix(kernels)
    inform = np.maximum(0, np.nanmean(cka_df.values, axis=1))
    inform = inform / max(inform.sum(),1e-12)
    def fused(w):
        K=np.zeros_like(Ks[0])
        for wi,Ki in zip(w,Ks): K += wi*Ki
        return (K+K.T)/2
    def obj(w):
        w=np.clip(w,0,None); w=w/max(w.sum(),1e-12)
        q=_spectral_gap_quality(fused(w),k)
        reg=0.05*np.sum((w-inform)**2)
        return -(q-reg)
    x0=np.ones(L)/L
    cons=({'type':'eq','fun':lambda w: np.sum(w)-1.0},)
    bnds=[(0.0,1.0)]*L
    try:
        r=minimize(obj,x0,method='SLSQP',bounds=bnds,constraints=cons,options={'maxiter':200,'ftol':1e-8})
        w=r.x if r.success else x0
    except Exception:
        w=x0
    w=np.clip(w,0,None); w=w/max(w.sum(),1e-12)
    Kf=fused(w)
    wdf=pd.DataFrame({'Layer':names,'Weight':w,'CKA_mean':np.nanmean(cka_df.values,axis=1)}).sort_values('Weight',ascending=False)
    return Kf, wdf, cka_df


def fused_spectral_clusters(Kf: np.ndarray, k: int, seed: int=42) -> np.ndarray:
    Kf=(Kf+Kf.T)/2
    try:
        lab = SpectralClustering(n_clusters=int(k), affinity='precomputed', random_state=int(seed), assign_labels='kmeans').fit_predict(Kf)
    except Exception:
        # fallback: thresholded connected comps / signless kmeans on eigvecs
        vals, vecs = eigh(Kf)
        X = vecs[:, np.argsort(vals)[::-1][:max(1,int(k))]]
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10).fit_predict(X)
    return np.asarray(lab, dtype=int)
