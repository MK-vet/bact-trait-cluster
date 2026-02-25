"""Signal-recovery and integration validation for bact-trait-cluster.

Tests verify that algorithms recover planted structure, not just output shapes.
"""
import subprocess, sys, json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── fixtures: tight 3-cluster structure ──────────────────────────────────

@pytest.fixture
def planted_matrix():
    """150 samples × 12 features, 3 planted clusters (signal:noise = 0.85:0.08)."""
    rng = np.random.RandomState(0)
    n_per, n_feat = 50, 12
    X = np.zeros((n_per * 3, n_feat), dtype=int)
    for k in range(3):
        rows = slice(k * n_per, (k + 1) * n_per)
        sig  = slice(k * 4, k * 4 + 4)
        noise = [c for c in range(n_feat) if c not in range(k * 4, k * 4 + 4)]
        X[rows, sig] = rng.binomial(1, 0.85, (n_per, 4))
        X[rows][:, noise] = rng.binomial(1, 0.08, (n_per, len(noise)))
    return X


@pytest.fixture
def planted_df(planted_matrix):
    return pd.DataFrame(planted_matrix, columns=[f"feat_{i}" for i in range(12)])


@pytest.fixture
def true_labels():
    return np.array([0] * 50 + [1] * 50 + [2] * 50)


# ── signal recovery ───────────────────────────────────────────────────────

class TestSignalRecovery:
    def test_consensus_matrix_within_vs_between(self, planted_matrix, true_labels):
        """Co-clustering probability is substantially higher within clusters than between."""
        from bacttraitcluster.clustering.consensus import consensus_matrix
        M = consensus_matrix(planted_matrix, k=3,
                             algorithms=["agglomerative_hamming"],
                             n_runs=30, seed=0, n_jobs=1)
        within, between = [], []
        for i in range(len(true_labels)):
            for j in range(i + 1, len(true_labels)):
                (within if true_labels[i] == true_labels[j] else between).append(M[i, j])
        assert np.mean(within) > np.mean(between) + 0.3, (
            f"Within={np.mean(within):.3f} not sufficiently > between={np.mean(between):.3f}"
        )

    def test_silhouette_k3_beats_k2(self, planted_matrix):
        """Silhouette score at k=3 must exceed k=2 for 3-cluster data."""
        from bacttraitcluster.clustering.consensus import stability_path
        results = stability_path(planted_matrix, k_range=[2, 3, 4],
                                 algorithms=["agglomerative_hamming"],
                                 n_runs=30, seed=0, n_jobs=1)
        by_k = {r.k: r for r in results}
        assert by_k[3].silhouette > by_k[2].silhouette, (
            f"sil(k=3)={by_k[3].silhouette:.3f} not > sil(k=2)={by_k[2].silhouette:.3f}"
        )

    def test_cliff_delta_planted_features_large(self, planted_df, true_labels):
        """Features enriched in cluster 0 (feat_0–3) yield Cliff delta > 0.5."""
        from bacttraitcluster.profiling.importance import cliff_delta_table
        cd = cliff_delta_table(planted_df, true_labels, n_boot=100, seed=0)
        c0 = cd[(cd["Cluster"] == 0) & (cd["Feature"].isin(
            [f"feat_{i}" for i in range(4)]))]
        assert (c0["Delta"] > 0.5).all(), (
            f"Expected delta>0.5 for planted features:\n{c0[['Feature','Delta']]}"
        )

    def test_enrichment_z_planted_features_significant(self, planted_df, true_labels):
        """Planted features are FDR-significant (p_adj < 0.05) in their own cluster."""
        from bacttraitcluster.profiling.importance import enrichment_z
        # API: enrichment_z(X, labels, alpha=0.05, method='fdr_bh')
        ez = enrichment_z(planted_df, true_labels, method="fdr_bh")
        planted = ez[
            (ez["Cluster"] == 0) &
            (ez["Feature"].isin([f"feat_{i}" for i in range(4)]))
        ]
        assert (planted["P_adj"] < 0.05).all(), (
            f"Planted features not FDR-significant:\n{planted[['Feature','Z','P_adj']]}"
        )

    def test_nvi_identical_labels(self):
        """VI(a, a) == 0."""
        from bacttraitcluster.clustering.consensus import variation_of_information
        a = np.array([0, 0, 1, 1, 2, 2])
        assert variation_of_information(a, a) == pytest.approx(0.0, abs=1e-10)


# ── reproducibility ───────────────────────────────────────────────────────

class TestReproducibility:
    def test_consensus_matrix_same_seed(self, planted_matrix):
        """Same seed → byte-identical consensus matrix."""
        from bacttraitcluster.clustering.consensus import consensus_matrix
        M1 = consensus_matrix(planted_matrix, k=3, algorithms=["agglomerative_hamming"],
                              n_runs=20, seed=99, n_jobs=1)
        M2 = consensus_matrix(planted_matrix, k=3, algorithms=["agglomerative_hamming"],
                              n_runs=20, seed=99, n_jobs=1)
        assert np.array_equal(M1, M2)


# ── robustness: missing data ──────────────────────────────────────────────

class TestRobustness:
    def test_missing_values_not_coerced_to_zero(self, tmp_path):
        """Loader must treat NA as missing, not as 0."""
        from bacttraitcluster.io.loader import load_layer_csv
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Strain_ID,geneA,geneB\nS1,1,\nS2,,1\nS3,1,1\n")
        result = load_layer_csv("test", str(csv_path), id_col="Strain_ID")
        # LayerLoadResult stores the matrix in .data (not .df)
        assert result.data["geneA"].isna().sum() == 1, "NA in geneA coerced to 0"
        assert result.data["geneB"].isna().sum() == 1, "NA in geneB coerced to 0"


# ── CLI integration ───────────────────────────────────────────────────────

class TestCLI:
    def test_help_exits_zero(self):
        r = subprocess.run([sys.executable, "-m", "bacttraitcluster.cli", "--help"],
                           capture_output=True)
        assert r.returncode == 0

    def test_version_output(self):
        from bacttraitcluster import __version__
        r = subprocess.run([sys.executable, "-m", "bacttraitcluster.cli", "--version"],
                           capture_output=True, text=True)
        assert __version__ in r.stdout

    def test_self_check_passes(self):
        r = subprocess.run([sys.executable, "-m", "bacttraitcluster.cli", "--self-check"],
                           capture_output=True, text=True)
        assert r.returncode == 0
        assert json.loads(r.stdout)["status"] == "PASS"

    def test_run_on_example_data(self, planted_df, tmp_path):
        """End-to-end: write CSV → run CLI → verify mandatory output files."""
        import yaml
        data_dir = tmp_path / "data"; data_dir.mkdir()
        out_dir  = tmp_path / "results"
        planted_df["Strain_ID"] = [f"S{i:03d}" for i in range(len(planted_df))]
        planted_df.to_csv(data_dir / "AMR.csv", index=False)

        config = {
            "schema_version": "1.1",
            "layers": [{"name": "AMR",
                        "path": str(data_dir / "AMR.csv"),
                        "id_column": "Strain_ID"}],
            "output_dir": str(out_dir),
            "consensus": {"algorithms": ["agglomerative_hamming"],
                          "k_range": [2, 3],
                          "n_consensus_runs": 10,
                          "subsample_fraction": 0.8,
                          "n_stability_splits": 5},
            "profiling": {"shap_enabled": False, "tda_enabled": False,
                          "effect_size_bootstrap": 20},
            "n_jobs": 1, "seed": 0,
        }
        with open(tmp_path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        r = subprocess.run(
            [sys.executable, "-m", "bacttraitcluster.cli",
             "--config", str(tmp_path / "config.yaml")],
            capture_output=True, text=True, timeout=120
        )
        assert r.returncode == 0, f"CLI failed:\n{r.stderr}"
        # integrated_clusters.csv is the top-level merged output
        assert (out_dir / "integrated_clusters.csv").exists()
        assert (out_dir / "run_manifest.json").exists()
        # Per-layer outputs
        assert (out_dir / "AMR" / "clusters.csv").exists()
        assert (out_dir / "AMR" / "stability_path.csv").exists()
