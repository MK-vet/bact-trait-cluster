"""
Cluster profiling via predictive feature importance, topology, and effect sizes.

Novel contributions
-------------------
1.  **SHAP predictive feature importance with bootstrap CI** — TreeSHAP on
    LightGBM for cluster label prediction; bootstrap provides CI bands.
    Replaces ad-hoc RF importance with model-agnostic, additive explanations
    (Lundberg & Lee 2017).

    Interpretation caveat: SHAP values quantify each feature's contribution
    to *predicting the cluster label* by LightGBM.  They reflect predictive
    relevance, not direct biological causation: a gene with high SHAP may be
    co-located with the cluster-defining gene rather than mechanistically
    responsible for resistance.  Results should be interpreted as
    "predictive feature importance" and not as causal AMR determinants.

2.  **Persistent homology of binary feature space** — Vietoris–Rips
    complex on Hamming distances.  Betti numbers and persistence entropy
    capture multi-scale topological shape that PCA / MCA cannot detect
    (Otter et al. 2017).
3.  **Cliff's delta effect size** — ordinal non-parametric effect size for
    binary cluster-vs-rest comparisons with bootstrap CI.  Addresses the
    "everything significant at large n" chi-square problem (Cliff 1993).
4.  **Enrichment z-scores / Fisher exact** — standardised cluster enrichment
    relative to global prevalence, with Fisher exact fallback for small
    clusters or rare features, and FDR correction.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# ── SHAP importance ──────────────────────────────────────────────────────

def shap_importance(
    X: pd.DataFrame,
    labels: np.ndarray,
    n_bg: int = 100,
    n_boot: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Predictive feature importance per feature × cluster via TreeSHAP with bootstrap CI.

    Trains a LightGBM classifier to predict cluster labels and computes mean
    absolute SHAP values per feature per cluster across bootstrap resamples.

    Interpretation
    --------------
    SHAP values here measure **predictive feature importance**: the contribution
    of each AMR feature to the LightGBM model's ability to predict cluster
    membership.  This is distinct from causal or mechanistic importance.  A
    feature can have high SHAP purely because it co-occurs with the
    cluster-defining features (e.g., on the same plasmid) without being an
    independent resistance determinant.  Results should not be interpreted as
    direct causal AMR mechanisms.

    Parameters
    ----------
    X : pd.DataFrame
        Binary feature matrix (rows = isolates, columns = AMR features).
    labels : np.ndarray
        Cluster label per isolate.
    n_bg : int
        Background sample size for TreeExplainer.
    n_boot : int
        Number of bootstrap resamples for CI estimation.
    seed : int

    Returns
    -------
    pd.DataFrame with columns Feature, Cluster, SHAP_Mean, CI_Lo, CI_Hi.
    """
    import shap
    from lightgbm import LGBMClassifier

    rng = np.random.RandomState(seed)
    feats = X.columns.tolist()
    store: Dict[Tuple[str, int], List[float]] = {
        (f, c): [] for f in feats for c in np.unique(labels)
    }
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        Xb, yb = X.values[idx], labels[idx]
        if len(np.unique(yb)) < 2:
            continue
        mdl = LGBMClassifier(n_estimators=100, max_depth=5,
                             random_state=seed + b, verbosity=-1, n_jobs=1)
        mdl.fit(Xb, yb)
        bg = Xb[rng.choice(len(Xb), min(n_bg, len(Xb)), replace=False)]
        ex = shap.TreeExplainer(mdl, bg)
        sv = ex.shap_values(Xb)
        if isinstance(sv, list):
            for ci, cl in enumerate(mdl.classes_):
                ma = np.mean(np.abs(sv[ci]), axis=0)
                for fi, f in enumerate(feats):
                    store[(f, cl)].append(ma[fi])
        else:
            ma = np.mean(np.abs(sv), axis=0)
            cl = mdl.classes_[-1]
            for fi, f in enumerate(feats):
                store[(f, cl)].append(ma[fi])

    rows = []
    for (f, c), vals in store.items():
        if not vals: continue
        a = np.array(vals)
        rows.append({"Feature": f, "Cluster": c,
                      "SHAP_Mean": round(a.mean(), 5),
                      "CI_Lo": round(np.percentile(a, 2.5), 5),
                      "CI_Hi": round(np.percentile(a, 97.5), 5)})
    return pd.DataFrame(rows).sort_values(["Cluster", "SHAP_Mean"], ascending=[True, False])


# ── Topological Data Analysis ────────────────────────────────────────────

def persistent_homology(
    X: np.ndarray,
    max_dim: int = 1,
    n_sub: int = 200,
    seed: int = 42,
) -> Dict:
    """Persistent homology via Vietoris–Rips on Hamming distances."""
    from ripser import ripser
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.RandomState(seed)
    if X.shape[0] > n_sub:
        X = X[rng.choice(X.shape[0], n_sub, replace=False)]
    D = squareform(pdist(X, "hamming"))
    res = ripser(D, maxdim=max_dim, distance_matrix=True)

    betti, total_pers, pers_ent, pairs = {}, {}, {}, []
    for dim, dgm in enumerate(res["dgms"]):
        fin = dgm[np.isfinite(dgm[:, 1])]
        betti[dim] = len(fin)
        if len(fin):
            lt = fin[:, 1] - fin[:, 0]; lt = lt[lt > 0]
            tp = float(lt.sum()); total_pers[dim] = tp
            if tp > 0:
                p = lt / tp
                pers_ent[dim] = float(-np.sum(p * np.log(p + 1e-15)))
            else:
                pers_ent[dim] = 0.0
        else:
            total_pers[dim] = 0.0; pers_ent[dim] = 0.0
        for b, d in dgm:
            pairs.append({"Dim": dim, "Birth": float(b),
                          "Death": float(d) if np.isfinite(d) else np.inf})
    return {"betti": betti, "total_persistence": total_pers,
            "persistence_entropy": pers_ent, "diagram": pd.DataFrame(pairs)}


# ── Cliff's delta ────────────────────────────────────────────────────────

def _cliff(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0: return 0.0
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return (gt - lt) / (len(x) * len(y))

def cliff_delta_table(
    X: pd.DataFrame,
    labels: np.ndarray,
    n_boot: int = 500,
    conf: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Cliff's delta for every feature × cluster-vs-rest."""
    rng = np.random.RandomState(seed)
    alpha = 1 - conf
    rows = []
    for f in X.columns:
        for cl in np.unique(labels):
            inc = X[f].values[labels == cl]
            exc = X[f].values[labels != cl]
            d = _cliff(inc, exc)
            ds = []
            for _ in range(n_boot):
                ds.append(_cliff(rng.choice(inc, len(inc), True),
                                 rng.choice(exc, len(exc), True)))
            da = np.array(ds)
            mag = ("large" if abs(d) >= .474 else
                   "medium" if abs(d) >= .33 else
                   "small" if abs(d) >= .147 else "negligible")
            rows.append({"Feature": f, "Cluster": cl,
                         "Delta": round(d, 4),
                         "CI_Lo": round(np.percentile(da, alpha/2*100), 4),
                         "CI_Hi": round(np.percentile(da, (1-alpha/2)*100), 4),
                         "Magnitude": mag})
    return pd.DataFrame(rows)


# ── enrichment z-scores ──────────────────────────────────────────────────

def enrichment_z(
    X: pd.DataFrame,
    labels: np.ndarray,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Standardised enrichment z per feature × cluster, FDR-corrected.

    Test selection
    --------------
    For each (feature, cluster) pair:
    - If ``n_cluster >= 30`` AND ``0.05 <= p_global <= 0.95``:
        z-score under normal approximation (central limit theorem applies).
    - Otherwise (small cluster OR rare/near-universal feature):
        two-sided Fisher exact test on the 2×2 contingency table
        [in_cluster × resistant/susceptible].  The z-score is back-calculated
        from the Fisher p-value for display consistency (Z = sign(Δp) × |Φ⁻¹(p/2)|).

    This fallback avoids inflated false-positive rates that arise when the
    normal approximation is applied to small n or extreme prevalences.
    """
    from scipy.stats import fisher_exact as _fisher_exact

    pg = X.mean()
    rows = []
    for f in X.columns:
        for cl in np.unique(labels):
            m = labels == cl
            nc = int(m.sum())
            pc = float(X[f].values[m].mean()) if nc else 0.0
            p_global = float(pg[f])

            use_normal = (nc >= 30) and (0.05 <= p_global <= 0.95)

            if use_normal:
                se = np.sqrt(p_global * (1 - p_global) / max(nc, 1))
                z = (pc - p_global) / se if se > 0 else 0.0
                pv = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
                test_used = "z"
            else:
                # Fisher exact on 2×2 table: [cluster, rest] × [resistant, susceptible]
                n_total = len(labels)
                n_rest = n_total - nc
                a = int(round(pc * nc))            # resistant in cluster
                b = nc - a                          # susceptible in cluster
                c = int(round(p_global * n_total)) - a  # resistant outside
                c = max(0, c)
                d = n_rest - c                      # susceptible outside
                d = max(0, d)
                table = np.array([[a, b], [c, d]], dtype=int)
                try:
                    _, pv = _fisher_exact(table)
                    pv = float(pv)
                except Exception:
                    pv = np.nan
                # back-calculate z for display (signed by direction)
                sign = 1.0 if pc >= p_global else -1.0
                z = sign * abs(float(norm.ppf(pv / 2 + 1e-15))) if (np.isfinite(pv) and pv > 0) else 0.0
                test_used = "fisher"

            rows.append({
                "Feature": f,
                "Cluster": cl,
                "Prev_Global": round(p_global, 4),
                "Prev_Cluster": round(pc, 4),
                "Z": round(float(z), 4),
                "P": pv,
                "Test": test_used,
            })

    df = pd.DataFrame(rows)
    ok = df["P"].notna()
    if ok.sum():
        _, padj, _, _ = multipletests(df.loc[ok, "P"], alpha=alpha, method=method)
        df.loc[ok, "P_adj"] = padj
        df["Significant"] = df["P_adj"] < alpha
    return df
