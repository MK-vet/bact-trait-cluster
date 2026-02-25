"""Pipeline: load (wide/long, with explicit missingness) → consensus cluster → profile → export.

Key innovations introduced in v1.0.0:
- robust handling of missing data (never coerced to 0)
- long-format layers (Plasmid/MGE) with explicit observed vs unobserved strata
- per-layer QC (prevalence, missingness) and coverage reporting
- layer concordance (ARI/NMI/VI + Hungarian label matching)
- assignment confidence from the consensus co-association matrix
- optional phylogeny-aware validation (purity vs. permutation), without duplicating phylo-comparative methods
- run_manifest.json (hashes, versions, config) for reproducibility
"""
from __future__ import annotations
from . import __version__

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config, LayerSpec
from .io import load_layer_csv, layer_coverage, qc_binary_features, sha256_file
from .clustering.consensus import (
    stability_path, select_k, variation_of_information,
    validate_algorithms,
)
from .clustering.multiview import optimize_kernel_weights, fused_spectral_clusters
from .clustering.model_selection import mdl_path_from_candidates
from .clustering.prediction_strength import prediction_strength
from .clustering.partition_info import feature_partition_nmi, partition_info_summary
from .profiling.importance import (
    shap_importance, persistent_homology, cliff_delta_table, enrichment_z,
)

logger = logging.getLogger(__name__)


def _save(df: pd.DataFrame, p: str) -> None:
    num = df.select_dtypes(include=[np.number]).columns
    out = df.copy()
    if len(num):
        out[num] = out[num].round(6)
    out.to_csv(p, index=False)
    logger.info("  → %s (%d rows)", p, len(out))


def _hungarian_match(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """Best one-to-one mapping between cluster labels (Hungarian on overlap counts)."""
    from scipy.optimize import linear_sum_assignment
    ua = np.unique(a); ub = np.unique(b)
    mat = np.zeros((len(ua), len(ub)), dtype=int)
    for i, la in enumerate(ua):
        for j, lb in enumerate(ub):
            mat[i, j] = int(np.sum((a == la) & (b == lb)))
    # maximise overlap => minimise negative
    r, c = linear_sum_assignment(-mat)
    rows = []
    for i, j in zip(r, c):
        rows.append({"Label_A": int(ua[i]), "Label_B": int(ub[j]), "Overlap": int(mat[i, j])})
    return pd.DataFrame(rows)


def _assignment_confidence(M: np.ndarray, labels: np.ndarray, ids: List[str]) -> pd.DataFrame:
    """Per-sample confidence from the consensus matrix."""
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    n = len(labs)
    out = []
    for i in range(n):
        li = labs[i]
        in_mask = (labs == li)
        in_mask[i] = False
        in_mean = float(np.nanmean(M[i, in_mask])) if in_mask.sum() > 0 else np.nan
        other_means = []
        for lj in uniq:
            if lj == li:
                continue
            m = (labs == lj)
            other_means.append(float(np.nanmean(M[i, m])) if m.sum() > 0 else -np.inf)
        out_best = float(np.max(other_means)) if other_means else np.nan
        conf = in_mean - out_best if (np.isfinite(in_mean) and np.isfinite(out_best)) else np.nan
        out.append({"Strain_ID": ids[i], "Cluster": int(li), "InCluster_Mean": in_mean,
                    "BestOther_Mean": out_best, "Confidence": conf})
    return pd.DataFrame(out)


def _tree_distance_matrix(tree, tips: List[str]) -> np.ndarray:
    # O(n^2) via Bio.Phylo.distance; acceptable for n~1e2 and used only if enabled.
    import numpy as np
    n = len(tips)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(tree.distance(tips[i], tips[j]))
            D[i, j] = D[j, i] = d
    return D


def _phylo_purity(labels: np.ndarray, D: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    """Purity score: ratio of mean between-cluster distance to mean within-cluster distance + permutation p-value."""
    rng = np.random.RandomState(seed)
    labs = np.asarray(labels)
    n = len(labs)

    def score(l):
        within = []
        between = []
        for i in range(n):
            for j in range(i + 1, n):
                if l[i] == l[j]:
                    within.append(D[i, j])
                else:
                    between.append(D[i, j])
        w = float(np.mean(within)) if within else np.nan
        b = float(np.mean(between)) if between else np.nan
        return b / (w + 1e-12)

    s0 = score(labs)
    if not np.isfinite(s0) or n_perm <= 0:
        return s0, np.nan

    ge = 0
    for _ in range(n_perm):
        lp = labs.copy()
        rng.shuffle(lp)
        if score(lp) >= s0:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return s0, p


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self, used_layers: List[LayerSpec]) -> None:
        import platform
        import sys
        import dataclasses as dc
        try:
            config_snapshot = dc.asdict(self.cfg)
        except Exception:
            config_snapshot = None
        manifest = {
            "tool": "bact-trait-cluster",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seed": self.cfg.seed,
            "tool_version": __version__,
            "algorithms_requested": getattr(self, '_algorithms_requested', None),
            "algorithms_used": getattr(self, '_algorithms_used', None),
            "algorithms_missing": getattr(self, '_algorithms_missing', None),
            "config_sha256": sha256_file(self.out / "config_used.yaml") if (self.out / "config_used.yaml").exists() else None,
            "inputs": [{"name": sp.name, "path": sp.path, "sha256": sha256_file(sp.path)} for sp in used_layers],
        }
        manifest["config_snapshot"] = config_snapshot
        try:
            import numpy
            import pandas
            manifest["package_versions"]={"numpy":numpy.__version__,"pandas":pandas.__version__}
            try:
                import scipy; manifest["package_versions"]["scipy"]=scipy.__version__
            except Exception: pass
            try:
                import networkx; manifest["package_versions"]["networkx"]=networkx.__version__
            except Exception: pass
        except Exception:
            pass
        (self.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def run(self) -> Dict[str, pd.DataFrame]:
        res: Dict[str, pd.DataFrame] = {}
        c = self.cfg
        cc = c.consensus
        cp = c.profiling

        # Validate clustering algorithms: auto-skip missing optional backends (e.g., kmodes).
        algos_used, algos_missing = validate_algorithms(cc.algorithms)
        self._algorithms_requested = list(cc.algorithms or [])
        self._algorithms_used = list(algos_used)
        self._algorithms_missing = list(algos_missing)
        if algos_missing:
            logger.warning('Some clustering algorithms are unavailable and will be skipped: %s', algos_missing)
        if not algos_used:
            raise ValueError(f'No valid clustering algorithms available. Requested={cc.algorithms}. Missing={algos_missing}')

        # 1) Determine global ID universe (union or intersection) without imputing missingness.
        logger.info("Loading %d layer(s)", len(c.layers))
        raw_layers = []
        id_sets = []
        for sp in c.layers:
            tmp = pd.read_csv(sp.path, usecols=[sp.id_column])
            tmp[sp.id_column] = tmp[sp.id_column].astype(str)
            ids = set(tmp[sp.id_column].unique())
            id_sets.append(ids)

        if c.input.align_mode.lower() == "intersection":
            all_ids = sorted(set.intersection(*id_sets))
        else:
            all_ids = sorted(set.union(*id_sets))
        if not all_ids:
            raise ValueError("No sample IDs found across inputs.")
        id_col = c.layers[0].id_column

        # 2) Load each layer with NA-preserving coercion + long-format support.
        layers: Dict[str, pd.DataFrame] = {}
        observed: Dict[str, pd.Series] = {}
        cov_rows = []
        qc_reports = []

        for sp in c.layers:
            lr = load_layer_csv(
                name=sp.name,
                path=sp.path,
                id_col=sp.id_column,
                all_ids=all_ids,
                feature_col=sp.feature_column if sp.format == "long" else sp.feature_column,
                value_col=sp.value_column,
            )
            # Force binary matrices only in this tool
            data = lr.data
            # If any non-binary/categorical column slipped through, drop it (and report)
            nonbin = [col for col in data.columns if not (str(data[col].dtype).startswith("Int") or pd.api.types.is_numeric_dtype(data[col]))]
            if nonbin:
                data = data.drop(columns=nonbin)
            layers[sp.name] = data
            observed[sp.name] = lr.observed
            cov_rows.append(layer_coverage(lr))

            # Feature QC (prevalence + missingness); no imputation.
            fqc = c.feature_qc
            data_qc, rep = qc_binary_features(
                data, observed=lr.observed,
                min_prev=fqc.min_prev, max_prev=fqc.max_prev, max_missing_frac=fqc.max_missing_frac
            )
            rep.insert(0, "Layer", sp.name)
            qc_reports.append(rep)

            # Drop samples/features with missingness above thresholds (complete-case by default).
            if c.input.drop_samples_with_missing:
                obs = lr.observed & data_qc.notna().any(axis=1)
                sample_miss = data_qc.isna().mean(axis=1)
                obs = obs & (sample_miss <= c.input.max_missing_sample)
                data_qc = data_qc.loc[obs]
            feat_miss = data_qc.isna().mean(axis=0)
            data_qc = data_qc.loc[:, feat_miss <= c.input.max_missing_feature]

            # Keep for clustering
            layers[sp.name] = data_qc

        cov_df = pd.DataFrame(cov_rows)
        _save(cov_df, str(self.out / "layer_coverage.csv"))
        res["coverage"] = cov_df
        qc_df = pd.concat(qc_reports, ignore_index=True) if qc_reports else pd.DataFrame()
        if not qc_df.empty:
            _save(qc_df, str(self.out / "feature_qc.csv"))
            res["feature_qc"] = qc_df

        # 3) Per-layer consensus clustering + profiling (only on observed + QC-passing data).
        clusters_all = pd.DataFrame({id_col: all_ids})
        for lname, data in layers.items():
            logger.info("=== %s ===", lname)
            ld = self.out / lname
            ld.mkdir(exist_ok=True)

            if data.shape[0] < 4 or data.shape[1] < 2:
                logger.warning("  Layer %s skipped (n=%d, p=%d after QC/missingness filters).", lname, data.shape[0], data.shape[1])
                continue

            ids_layer = data.index.astype(str).tolist()
            X = data.astype("Float64").to_numpy(dtype=float)
            # ensure no NA for clustering (complete-case already enforced)
            if np.isnan(X.astype(float)).any():
                logger.warning("  Layer %s still has NA after filtering; dropping rows with NA.", lname)
                ok = ~np.isnan(X.astype(float)).any(axis=1)
                X = X[ok]
                ids_layer = [ids_layer[i] for i, v in enumerate(ok) if v]

            kr = [k for k in cc.k_range if 2 <= k < len(ids_layer)]
            if not kr:
                logger.warning("  Layer %s skipped (k_range incompatible with n=%d).", lname, len(ids_layer))
                continue

            stab = stability_path(
                X, kr, algos_used, cc.n_consensus_runs,
                cc.n_stability_splits, cc.subsample_fraction,
                c.seed, c.n_jobs
            )
            sdf = pd.DataFrame([{
                "k": r.k, "NVI_Mean": r.mean_nvi, "NVI_Std": r.std_nvi,
                "NVI_CI_Lo": r.ci_lo, "NVI_CI_Hi": r.ci_hi, "Silhouette": r.silhouette
            } for r in stab])
            _save(sdf, str(ld / "stability_path.csv"))
            res[f"{lname}_stab"] = sdf

            best = select_k(stab)
            # Guard against degenerate single-cluster outputs
            if len(np.unique(best.labels)) <= 1 and len(stab) > 1:
                logger.warning("  Selected solution collapsed to 1 cluster; selecting best non-degenerate k.")
                nondeg = [r for r in stab if len(np.unique(r.labels)) > 1]
                if nondeg:
                    best = min(nondeg, key=lambda r: (r.mean_nvi, -r.silhouette))
            logger.info("  Optimal k=%d  NVI=%.3f  Sil=%.3f", best.k, best.mean_nvi, best.silhouette)

            labels = best.labels
            # Save layer-specific clusters
            cdf = pd.DataFrame({id_col: ids_layer, "Cluster": labels})
            _save(cdf, str(ld / "clusters.csv"))
            res[f"{lname}_clusters"] = cdf
            np.save(str(ld / "consensus_matrix.npy"), best.M)

            # Assignment confidence
            conf = _assignment_confidence(best.M, labels, ids_layer)
            _save(conf, str(ld / "assignment_confidence.csv"))
            res[f"{lname}_confidence"] = conf

            # Profiling on the original QC-filtered data for these samples
            data0 = data.loc[ids_layer]
            ez = enrichment_z(data0, labels, c.alpha, c.fdr_method)
            _save(ez, str(ld / "enrichment_z.csv"))
            res[f"{lname}_enrichment"] = ez

            cd = cliff_delta_table(data0, labels, cp.effect_size_bootstrap, seed=c.seed)
            _save(cd, str(ld / "cliff_delta.csv"))
            res[f"{lname}_cliff"] = cd

            if cp.shap_enabled:
                try:
                    sh = shap_importance(data0, labels, cp.shap_n_background, cp.shap_bootstrap, c.seed)
                    _save(sh, str(ld / "shap.csv"))
                    res[f"{lname}_shap"] = sh
                except ImportError:
                    logger.warning("  shap/lightgbm not installed — skipped")

            if cp.tda_enabled:
                try:
                    tda = persistent_homology(X, cp.tda_max_dim, cp.tda_n_subsample, c.seed)
                    _save(tda["diagram"], str(ld / "persistence_diagram.csv"))
                    ts = pd.DataFrame([{
                        "Dim": d, "Betti": tda["betti"][d],
                        "Total_Persistence": tda["total_persistence"][d],
                        "Persistence_Entropy": tda["persistence_entropy"][d]
                    } for d in sorted(tda["betti"])])
                    _save(ts, str(ld / "tda_summary.csv"))
                    res[f"{lname}_tda"] = ts
                except ImportError:
                    logger.warning("  ripser not installed — TDA skipped")

            # Optional phylogeny-aware validation (purity) — *validation only*
            if c.phylo_validation.enabled and c.phylo_validation.tree_path:
                try:
                    from Bio import Phylo
                    tree = Phylo.read(c.phylo_validation.tree_path, "newick")
                    tips = [t.name for t in tree.get_terminals()]
                    if c.phylo_validation.id_match_required:
                        missing = sorted(set(ids_layer) - set(tips))
                        if missing:
                            logger.warning("  Tree missing %d clustered IDs (showing up to 10): %s", len(missing), missing[:10])
                    keep = [i for i in ids_layer if i in set(tips)]
                    if len(keep) >= 4:
                        # order by keep list
                        D = _tree_distance_matrix(tree, keep)
                        labs_keep = np.array([labels[ids_layer.index(i)] for i in keep])
                        purity, pp = _phylo_purity(labs_keep, D, c.phylo_validation.n_perm, c.seed)
                        pdf = pd.DataFrame([{"Purity_between_over_within": purity, "Perm_P": pp, "N": len(keep)}])
                        _save(pdf, str(ld / "phylo_purity.csv"))
                        res[f"{lname}_phylo_purity"] = pdf
                except ImportError:
                    logger.warning("  Biopython not installed — phylo validation skipped")

            # Integrate into global table as NA for strains not clustered in this layer
            lab_all = pd.Series(pd.NA, index=all_ids, dtype="Int64")
            lab_all.loc[ids_layer] = labels
            clusters_all[f"{lname}_Cluster"] = lab_all.values

        _save(clusters_all, str(self.out / "integrated_clusters.csv"))
        res["integrated"] = clusters_all

        # 4) Layer concordance (ARI/NMI/VI + label matching) on shared clustered strains.
        rows = []
        match_rows = []
        names = [c for c in clusters_all.columns if c.endswith("_Cluster")]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = clusters_all[names[i]]
                b = clusters_all[names[j]]
                m = a.notna() & b.notna()
                if m.sum() < 4:
                    continue
                aa = a[m].astype(int).to_numpy()
                bb = b[m].astype(int).to_numpy()
                try:
                    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                    ari = float(adjusted_rand_score(aa, bb))
                    nmi = float(normalized_mutual_info_score(aa, bb))
                except Exception:
                    ari = np.nan; nmi = np.nan
                vi = float(variation_of_information(aa, bb))
                rows.append({"Layer_A": names[i].replace("_Cluster",""), "Layer_B": names[j].replace("_Cluster",""),
                             "N_shared": int(m.sum()), "ARI": ari, "NMI": nmi, "VI": vi})
                mt = _hungarian_match(aa, bb)
                mt.insert(0, "Layer_A", names[i].replace("_Cluster",""))
                mt.insert(1, "Layer_B", names[j].replace("_Cluster",""))
                match_rows.append(mt)

        if rows:
            conc = pd.DataFrame(rows)
            _save(conc, str(self.out / "layer_concordance.csv"))
            res["layer_concordance"] = conc
        if match_rows:
            mt = pd.concat(match_rows, ignore_index=True)
            _save(mt, str(self.out / "layer_label_matching.csv"))
            res["layer_label_matching"] = mt


        # 5) Multi-view fusion (anti-salami novelty axis for bact-trait-cluster)
        layer_consensus = {}
        layer_best_labels = {}
        layer_best_X = {}
        for lname in layers.keys():
            mpath = self.out / lname / "consensus_matrix.npy"
            cpath = self.out / lname / "clusters.csv"
            if mpath.exists() and cpath.exists():
                try:
                    M = np.load(str(mpath))
                    cdf = pd.read_csv(cpath)
                    layer_consensus[lname] = M
                    lab = cdf["Cluster"].to_numpy(dtype=int)
                    layer_best_labels[lname] = lab
                    # align X to clustered ids for downstream feature-partition metrics
                    ids_here = cdf[id_col].astype(str).tolist()
                    if lname in layers:
                        layer_best_X[lname] = layers[lname].loc[ids_here].to_numpy(dtype=float)
                except Exception as e:
                    logger.warning("  Multi-view load skipped for %s: %s", lname, e)
        if len(layer_consensus) >= 2:
            try:
                # use smallest common n across layers by matching current clustered IDs from integrated table
                common_ids = None
                id_maps = {}
                for lname in layer_consensus:
                    cdf = pd.read_csv(self.out / lname / "clusters.csv")
                    ids_l = cdf[id_col].astype(str).tolist()
                    id_maps[lname] = ids_l
                    common_ids = set(ids_l) if common_ids is None else (common_ids & set(ids_l))
                common_ids = sorted(common_ids) if common_ids else []
                if len(common_ids) >= 6:
                    kernels = {}
                    labels_candidates = {}
                    for lname, M in layer_consensus.items():
                        ids_l = id_maps[lname]
                        idx = [ids_l.index(i) for i in common_ids]
                        kernels[lname] = M[np.ix_(idx, idx)]
                    # k candidate from median of per-layer optimal k if available
                    k_guess = int(np.median([len(np.unique(pd.read_csv(self.out/l/'clusters.csv')['Cluster'])) for l in layer_consensus]))
                    k_guess = max(2, min(k_guess, len(common_ids)-1))
                    Kf, wdf, cka_df = optimize_kernel_weights(kernels, k=k_guess)
                    np.save(str(self.out / 'fused_consensus.npy'), Kf)
                    _save(wdf, str(self.out / 'layer_weights.csv')); res['layer_weights']=wdf
                    _save(cka_df.reset_index().rename(columns={'index':'Layer'}), str(self.out / 'cka_matrix.csv'))
                    fusion_summary = pd.DataFrame([{
                        'N_layers_fused': int(len(kernels)),
                        'N_common_ids': int(len(common_ids)),
                        'k_guess': int(k_guess),
                        'CKA_mean_offdiag': float(np.nanmean(cka_df.values[np.triu_indices_from(cka_df.values,1)])) if cka_df.shape[0]>1 else np.nan
                    }])
                    _save(fusion_summary, str(self.out / 'fused_mode_summary.csv')); res['fused_mode_summary']=fusion_summary
                    # candidate k by spectral clustering
                    for k in [kk for kk in sorted(set([2,3,4,5,6,k_guess])) if kk < len(common_ids)]:
                        labels_candidates[k] = fused_spectral_clusters(Kf, k, seed=c.seed)
                        Xmdl = pd.concat([layers[n].loc[common_ids] for n in layers], axis=1).to_numpy(dtype=float)
                        Xmdl = np.where(np.isnan(Xmdl), np.nanmean(Xmdl, axis=0, keepdims=True), Xmdl)
                        mdl = mdl_path_from_candidates(Xmdl, labels_candidates)
                    if not mdl.empty:
                        _save(mdl, str(self.out / 'fused_mdl_path.csv')); res['fused_mdl']=mdl
                        k_star = int(mdl.sort_values('MDL').iloc[0]['k'])
                    else:
                        k_star = k_guess
                    fused_lab = labels_candidates.get(k_star, fused_spectral_clusters(Kf, k_star, seed=c.seed))
                    fdf = pd.DataFrame({id_col: common_ids, 'Fused_Cluster': fused_lab})
                    _save(fdf, str(self.out / 'fused_clusters.csv')); res['fused_clusters']=fdf
                    # prediction strength on fused concatenated feature matrix
                    Xf = pd.concat([layers[n].loc[common_ids] for n in layers], axis=1).to_numpy(dtype=float)
                    ps = prediction_strength(Xf, labels_candidates, n_splits=15, seed=c.seed)
                    if not ps.empty:
                        _save(ps, str(self.out / 'prediction_strength.csv')); res['prediction_strength']=ps
                    # partition info per layer vs fused
                    pi_rows=[]
                    for lname in layers:
                        if set(common_ids).issubset(set(layers[lname].index)):
                            info = partition_info_summary(fused_lab, layers[lname].loc[common_ids].to_numpy(dtype=float))
                            info['Layer']=lname; pi_rows.append(info)
                            fp = feature_partition_nmi(layers[lname].loc[common_ids], fused_lab)
                            if not fp.empty:
                                fp.insert(0,'Layer',lname)
                                _save(fp, str(self.out / f'{lname}_feature_partition_nmi.csv'))
                    if pi_rows:
                        _save(pd.DataFrame(pi_rows), str(self.out / 'partition_info.csv'))
            except Exception as e:
                logger.warning("  Multi-view fusion block skipped due to error: %s", e)

        # Persist config + manifest
        c.to_yaml(str(self.out / "config_used.yaml"))
        self._write_manifest(c.layers)
        logger.info("Done — results in %s", self.out)
        return res
