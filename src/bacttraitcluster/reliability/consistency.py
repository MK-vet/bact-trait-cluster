from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    _SKLEARN_OK = True
except Exception:
    adjusted_rand_score = None
    normalized_mutual_info_score = None
    _SKLEARN_OK = False

from .core import detect_id_column, read_table, save_json


def _load_label_table(
    path: str | Path, label_col: str, id_col: Optional[str] = None
) -> pd.DataFrame:
    df = read_table(path)
    idc = detect_id_column(df, explicit=id_col)
    if idc is None:
        raise ValueError(f"Could not detect ID column in {path}")
    if label_col not in df.columns:
        # attempt heuristic
        candidates = [c for c in df.columns if c != idc]
        if len(candidates) == 1:
            label_col = candidates[0]
        else:
            raise ValueError(
                f"Label column '{label_col}' not found in {path}; candidates={candidates}"
            )
    out = df[[idc, label_col]].copy()
    out.columns = ["id", "label"]
    out["id"] = out["id"].astype(str)
    return out.dropna(subset=["id", "label"])


def _cramers_v_bias_corrected(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    if tab.empty:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    if n == 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def compare_labelings(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(tables.keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            merged = tables[a].merge(tables[b], on="id", suffixes=(f"_{a}", f"_{b}"))
            n = len(merged)
            if n == 0:
                rows.append(
                    {
                        "tool_a": a,
                        "tool_b": b,
                        "n_overlap": 0,
                        "ari": np.nan,
                        "nmi": np.nan,
                        "cramers_v_bc": np.nan,
                    }
                )
                continue
            x = merged[f"label_{a}"].astype(str)
            y = merged[f"label_{b}"].astype(str)
            if _SKLEARN_OK:
                ari = adjusted_rand_score(x, y)
                nmi = normalized_mutual_info_score(x, y)
            else:
                ari = np.nan
                nmi = np.nan
            cv = _cramers_v_bias_corrected(x, y)
            rows.append(
                {
                    "tool_a": a,
                    "tool_b": b,
                    "n_overlap": int(n),
                    "ari": float(ari),
                    "nmi": float(nmi),
                    "cramers_v_bc": float(cv),
                }
            )
    return pd.DataFrame(rows)


def build_consistency_report(
    tool_label_specs: Mapping[str, Mapping[str, str]],
    out_json: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    tool_label_specs:
      {
        "bact-trait-cluster": {"path": ".../cluster_labels.csv", "label_col": "cluster", "id_col": "strain"},
        ...
      }
    """
    tables = {}
    skipped = {}
    for tool, spec in tool_label_specs.items():
        p = Path(spec["path"])
        if not p.exists():
            skipped[tool] = {"reason": "file_missing", "path": str(p)}
            continue
        try:
            tbl = _load_label_table(
                p, label_col=spec.get("label_col", "label"), id_col=spec.get("id_col")
            )
            if tbl["label"].nunique() <= 1:
                skipped[tool] = {
                    "reason": "degenerate_labels",
                    "path": str(p),
                    "n_unique_labels": int(tbl["label"].nunique()),
                }
                continue
            tables[tool] = tbl
        except Exception as e:
            skipped[tool] = {"reason": "parse_error", "path": str(p), "error": str(e)}
    pairwise = (
        compare_labelings(tables)
        if len(tables) >= 2
        else pd.DataFrame(
            columns=["tool_a", "tool_b", "n_overlap", "ari", "nmi", "cramers_v_bc"]
        )
    )
    flags = []
    for _, row in pairwise.iterrows():
        if pd.notna(row["nmi"]) and row["nmi"] < 0.05:
            flags.append(
                f"Near-zero NMI between {row['tool_a']} and {row['tool_b']} (n={int(row['n_overlap'])})"
            )
        if pd.notna(row["cramers_v_bc"]) and row["cramers_v_bc"] < 0.05:
            flags.append(
                f"Near-zero Cramér's V between {row['tool_a']} and {row['tool_b']} (n={int(row['n_overlap'])})"
            )
    status = "PASS"
    if flags:
        status = "WARN"
    if len(tables) < 2:
        status = "WARN"
        flags.append(
            "Insufficient non-degenerate label tables for cross-tool consistency comparison"
        )
    report = {
        "status": status,
        "sklearn_metrics_available": bool(_SKLEARN_OK),
        "tools_compared": sorted(list(tables.keys())),
        "tools_skipped": skipped,
        "pairwise_metrics": pairwise.to_dict(orient="records"),
        "flags": flags,
    }
    if out_json is not None:
        save_json(report, out_json)
    return report
