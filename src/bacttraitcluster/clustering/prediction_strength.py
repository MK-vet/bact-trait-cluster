from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


def _cluster_simple(X, k):
    # binary/hamming robust fallback
    D = pairwise_distances(X, metric="hamming")
    return AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(D)


def prediction_strength(
    X: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    n_splits: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = np.asarray(X, float)
    n = len(X)
    rows = []
    for k in sorted(labels_by_k):
        ps = []
        cps = []
        for b in range(n_splits):
            idx = np.arange(n)
            rng.shuffle(idx)
            a = idx[: n // 2]
            t = idx[n // 2 :]
            if len(a) < k or len(t) < k:
                continue
            Xa = np.nan_to_num(X[a], nan=np.nanmean(X, axis=0))
            Xt = np.nan_to_num(X[t], nan=np.nanmean(X, axis=0))
            _cluster_simple(Xa, k)
            lt = _cluster_simple(Xt, k)
            # cluster-wise within test compactness proxy
            cl_scores = []
            for cl in np.unique(lt):
                m = lt == cl
                if m.sum() < 2:
                    continue
                D = pairwise_distances(Xt[m], metric="hamming")
                cl_scores.append(float(1.0 - np.nanmean(D)))
            if cl_scores:
                cps.extend(
                    [
                        {"k": k, "Split": b, "Cluster": i, "PS_cluster": v}
                        for i, v in enumerate(cl_scores)
                    ]
                )
                ps.append(float(np.min(cl_scores)))
        if ps:
            arr = np.asarray(ps)
            rows.append(
                {
                    "k": k,
                    "PS_mean": arr.mean(),
                    "PS_ci_lo": np.quantile(arr, 0.025),
                    "PS_ci_hi": np.quantile(arr, 0.975),
                    "B": len(arr),
                }
            )
        if cps:
            pass
    return pd.DataFrame(rows)
