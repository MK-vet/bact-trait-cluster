
"""Robust CSV loader for wide (matrix) and long (ID,feature[,value]) layers.

Design goals:
1) Do NOT silently convert missing data to 0. Missing stays as NA.
2) Long format is assumed to be a *positive list* (features present).
   - For strains present in the file: unlisted features are treated as 0 (absent), because the strain was observed.
   - For strains absent from the file: all features are NA (unobserved / missing).
3) Provide explicit coverage/observed flags for defensible reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

@dataclass
class LayerLoadResult:
    name: str
    data: pd.DataFrame            # index: IDs, columns: features (Int8 with NA)
    observed: pd.Series           # boolean, True if strain observed in this layer
    meta: Dict[str, object]       # misc metadata

def _is_long_format(df: pd.DataFrame, id_col: str) -> bool:
    """Heuristic for positive-list long format, avoiding categorical metadata misclassification."""
    if id_col not in df.columns:
        return False
    cols = [c for c in df.columns if c != id_col]
    if len(cols) == 0:
        return False
    dup_ids = float(df[id_col].astype(str).duplicated().mean()) if len(df) else 0.0
    if dup_ids <= 0:
        return False
    if len(cols) == 1:
        c = cols[0]
        return not pd.api.types.is_numeric_dtype(df[c])
    if len(cols) == 2:
        feat_col, val_col = cols[0], cols[1]
        return (not pd.api.types.is_numeric_dtype(df[feat_col])) and pd.api.types.is_numeric_dtype(df[val_col])
    return False

def _coerce_binary_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce wide matrix columns to binary where appropriate, preserving NA.

    Binary-like columns (0/1, bool) -> Int8 binary.
    Categorical metadata columns (e.g., MLST, Serotype, Species) -> one-hot columns.
    """
    out = pd.DataFrame(index=df.index)

    def _add_dummies(base_name: str, cat_series: pd.Series) -> None:
        nonlocal out
        cat = cat_series.astype("string")
        miss = cat.isna()
        dummies = pd.get_dummies(cat, prefix=str(base_name), dtype="Int8")
        if miss.any() and dummies.shape[1] > 0:
            dummies.loc[miss, :] = pd.NA
        if dummies.shape[1] == 0:
            out[base_name] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int8")
        else:
            out = pd.concat([out, dummies], axis=1)

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("Int8")
            continue

        num = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
        if pd.api.types.is_numeric_dtype(s) or num.notna().any():
            vals = pd.Series(num).dropna()
            uniq = pd.unique(vals)
            if len(uniq) > 0 and set(pd.Series(uniq).astype(float).round(12)).issubset({0.0, 1.0}):
                arr = pd.Series(num, index=df.index).astype("Float64")
                out[c] = arr.where(arr.isna(), (arr > 0).astype("Int8"))
            else:
                def _fmt(v):
                    if pd.isna(v):
                        return pd.NA
                    try:
                        fv = float(v)
                        return str(int(fv)) if fv.is_integer() else str(fv)
                    except Exception:
                        return str(v)
                _add_dummies(c, pd.Series(num, index=df.index).map(_fmt))
            continue

        _add_dummies(c, s)

    for c in out.columns:
        try:
            if str(out[c].dtype) in {"int64", "int32", "int16", "int8", "uint8", "bool"}:
                out[c] = out[c].astype("Int8")
        except Exception:
            pass
    return out

def load_layer_csv(
    name: str,
    path: str | Path,
    id_col: str = "Strain_ID",
    all_ids: Optional[List[str]] = None,
    feature_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> LayerLoadResult:
    p = Path(path)
    df = pd.read_csv(p)
    if id_col not in df.columns:
        raise ValueError(f"[{name}] Missing id_column '{id_col}' in {p}")

    df[id_col] = df[id_col].astype(str)

    if feature_col is not None and feature_col not in df.columns:
        raise ValueError(f"[{name}] feature_col='{feature_col}' not found in {p}")
    if value_col is not None and value_col not in df.columns:
        raise ValueError(f"[{name}] value_col='{value_col}' not found in {p}")

    meta: Dict[str, object] = {"path": str(p), "format": "wide"}

    if feature_col is None and _is_long_format(df, id_col):
        meta["format"] = "long_auto"
        cols = [c for c in df.columns if c != id_col]
        feature_col = cols[0]
        if len(cols) == 2:
            value_col = cols[1]

    if feature_col is not None:
        # Long format: rows are detections (ID, feature[, value]).
        meta["format"] = "long"
        feat = df[feature_col].astype(str)
        ids = df[id_col].astype(str)

        if value_col is None:
            tab = pd.crosstab(ids, feat)
        else:
            # treat any positive/ non-null as presence
            val = pd.to_numeric(df[value_col], errors="coerce")
            tmp = pd.DataFrame({id_col: ids, feature_col: feat, "__val": val})
            tmp = tmp[tmp["__val"].notna()]
            tmp["__val"] = (tmp["__val"] > 0).astype(int)
            tab = tmp.pivot_table(index=id_col, columns=feature_col, values="__val", aggfunc="max").fillna(0)

        tab = (tab > 0).astype("Int8")
        observed_ids = tab.index.astype(str)
        observed = pd.Series(True, index=observed_ids, name="__OBSERVED")

        # Expand to all IDs: unobserved strains get NA, not 0.
        if all_ids is not None:
            all_ids = [str(x) for x in all_ids]
            tab = tab.reindex(index=all_ids)
            observed = observed.reindex(index=all_ids).fillna(False).astype(bool)

        return LayerLoadResult(name=name, data=tab, observed=observed, meta=meta)

    # Wide format
    wide = df.set_index(id_col)
    wide.index = wide.index.astype(str)

    # Coerce to binary (Int8) where possible, but never impute NA to 0.
    wide = _coerce_binary_wide(wide)
    observed = wide.notna().any(axis=1)
    observed.name = "__OBSERVED"

    if all_ids is not None:
        all_ids = [str(x) for x in all_ids]
        wide = wide.reindex(index=all_ids)
        observed = observed.reindex(index=all_ids).fillna(False).astype(bool)

    return LayerLoadResult(name=name, data=wide, observed=observed, meta=meta)

def layer_coverage(res: LayerLoadResult) -> Dict[str, object]:
    n_total = int(res.data.shape[0])
    n_obs = int(res.observed.sum())
    n_feat = int(res.data.shape[1])
    miss_cells = int(res.data.isna().sum().sum())
    return {
        "Layer": res.name,
        "Format": res.meta.get("format", "wide"),
        "N_total": n_total,
        "N_observed": n_obs,
        "Coverage": round(n_obs / max(n_total, 1), 4),
        "N_features": n_feat,
        "Missing_cells": miss_cells,
    }

def qc_binary_features(
    df: pd.DataFrame,
    observed: Optional[pd.Series] = None,
    min_prev: float = 0.01,
    max_prev: float = 0.99,
    max_missing_frac: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Feature QC for binary matrices with NA.

    - prevalence computed among observed rows and non-missing values
    - features with prevalence outside [min_prev, max_prev] are dropped
    - features with missingness > max_missing_frac are dropped
    """
    if observed is None:
        observed = df.notna().any(axis=1)
    obs_df = df.loc[observed]
    if obs_df.empty:
        return df.iloc[0:0], pd.DataFrame()

    miss_frac = obs_df.isna().mean(axis=0)
    valid = obs_df.notna().sum(axis=0)
    prev = obs_df.mean(axis=0, skipna=True)  # mean of 0/1 ignoring NA
    keep = (valid > 0) & (prev >= min_prev) & (prev <= max_prev) & (miss_frac <= max_missing_frac)

    report = pd.DataFrame(index=pd.Index(df.columns, name="Feature"))
    report["Valid_N"] = valid.reindex(df.columns).fillna(0).astype(int)
    report["Prevalence"] = prev.reindex(df.columns)
    report["Missing_Frac"] = miss_frac.reindex(df.columns)
    report["Keep"] = keep.reindex(df.columns).fillna(False)
    report["Dropped_reason"] = ""
    report.loc[(~report["Keep"]) & (report["Valid_N"] == 0), "Dropped_reason"] = "all_missing"
    report.loc[(~report["Keep"]) & (report["Prevalence"] < min_prev), "Dropped_reason"] = "too_rare"
    report.loc[(~report["Keep"]) & (report["Prevalence"] > max_prev), "Dropped_reason"] = "too_common"
    report.loc[(~report["Keep"]) & (report["Missing_Frac"] > max_missing_frac), "Dropped_reason"] = "too_many_missing"

    keep_mask = report["Keep"].to_numpy(dtype=bool)
    return df.loc[:, keep_mask], report.reset_index()

def sha256_file(path: str | Path) -> str:
    import hashlib
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
