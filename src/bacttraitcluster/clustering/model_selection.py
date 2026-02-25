from __future__ import annotations
import numpy as np
import pandas as pd


def mdl_binary_partition(X: np.ndarray, labels: np.ndarray):
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    n, p = X.shape
    rows = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        Xi = X[idx]
        mask = ~np.isnan(Xi)
        denom = np.maximum(mask.sum(axis=0), 1)
        phat = np.nansum(Xi, axis=0) / denom
        phat = np.clip(phat, 1e-9, 1 - 1e-9)
        H = -(phat * np.log2(phat) + (1 - phat) * np.log2(1 - phat))
        rows.append(float(np.nansum(denom * H)))
    L_data = float(np.sum(rows))
    k = len(np.unique(labels))
    L_model = float(k * p * np.log2(max(n, 2)))
    return {
        "MDL": L_data + L_model,
        "L_data": L_data,
        "L_model": L_model,
        "k": k,
        "n": n,
        "p": p,
    }


def mdl_path_from_candidates(
    X: np.ndarray, labels_by_k: dict[int, np.ndarray]
) -> pd.DataFrame:
    out = []
    for k, lab in sorted(labels_by_k.items()):
        r = mdl_binary_partition(X, lab)
        out.append(r)
    df = pd.DataFrame(out)
    if not df.empty:
        df["Delta_to_min"] = df["MDL"] - df["MDL"].min()
    return df
