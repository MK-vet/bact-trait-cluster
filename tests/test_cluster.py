"""Tests for bact-trait-cluster consensus clustering and profiling modules."""

import numpy as np
import pandas as pd
import pytest

# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def binary_matrix():
    """50 samples × 10 features binary matrix with 3 planted clusters."""
    rng = np.random.RandomState(42)
    n, p = 50, 10
    X = np.zeros((n, p), dtype=int)
    # Cluster 0: features 0-3 enriched
    X[:17, :4] = rng.binomial(1, 0.8, (17, 4))
    X[:17, 4:] = rng.binomial(1, 0.1, (17, 6))
    # Cluster 1: features 4-6 enriched
    X[17:34, :4] = rng.binomial(1, 0.1, (17, 4))
    X[17:34, 4:7] = rng.binomial(1, 0.8, (17, 3))
    X[17:34, 7:] = rng.binomial(1, 0.1, (17, 3))
    # Cluster 2: features 7-9 enriched
    X[34:, :7] = rng.binomial(1, 0.1, (16, 7))
    X[34:, 7:] = rng.binomial(1, 0.8, (16, 3))
    return X


@pytest.fixture
def binary_df(binary_matrix):
    cols = [f"feat_{i}" for i in range(binary_matrix.shape[1])]
    return pd.DataFrame(binary_matrix, columns=cols)


# ── consensus tests ──────────────────────────────────────────────────────


class TestConsensus:
    def test_jaccard_kernel_shape(self, binary_matrix):
        from bacttraitcluster.clustering.consensus import jaccard_kernel

        K = jaccard_kernel(binary_matrix)
        assert K.shape == (50, 50)
        # Diagonal = 1.0 for rows with any positive feature
        nonzero = binary_matrix.sum(axis=1) > 0
        assert np.allclose(np.diag(K)[nonzero], 1.0)

    def test_jaccard_kernel_symmetry(self, binary_matrix):
        from bacttraitcluster.clustering.consensus import jaccard_kernel

        K = jaccard_kernel(binary_matrix)
        assert np.allclose(K, K.T)

    def test_hamming_distance_range(self, binary_matrix):
        from bacttraitcluster.clustering.consensus import hamming_distance

        D = hamming_distance(binary_matrix)
        assert D.min() >= 0.0
        assert D.max() <= 1.0
        assert np.allclose(np.diag(D), 0.0)

    def test_consensus_matrix_shape(self, binary_matrix):
        from bacttraitcluster.clustering.consensus import consensus_matrix

        M = consensus_matrix(
            binary_matrix,
            k=3,
            algorithms=["agglomerative_hamming"],
            n_runs=10,
            seed=42,
            n_jobs=1,
        )
        assert M.shape == (50, 50)
        assert M.min() >= 0.0
        assert M.max() <= 1.0

    def test_consensus_labels_count(self, binary_matrix):
        from bacttraitcluster.clustering.consensus import (
            consensus_matrix,
            consensus_labels,
        )

        M = consensus_matrix(
            binary_matrix,
            k=3,
            algorithms=["agglomerative_hamming"],
            n_runs=10,
            seed=42,
            n_jobs=1,
        )
        labels = consensus_labels(M, k=3)
        assert len(np.unique(labels)) == 3
        assert len(labels) == 50

    def test_vi_identical(self):
        from bacttraitcluster.clustering.consensus import variation_of_information

        a = np.array([0, 0, 1, 1, 2, 2])
        assert variation_of_information(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_vi_different(self):
        from bacttraitcluster.clustering.consensus import variation_of_information

        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        assert variation_of_information(a, b) > 0.0

    def test_nvi_range(self):
        from bacttraitcluster.clustering.consensus import nvi

        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([1, 1, 0, 0, 2, 2])
        v = nvi(a, b)
        assert 0 <= v <= 2.0  # generous upper bound

    def test_register_algorithm(self, binary_matrix):
        from bacttraitcluster.clustering.consensus import (
            register_algorithm,
            consensus_matrix,
        )

        def dummy(X, k, seed):
            return np.random.RandomState(seed).randint(0, k, X.shape[0])

        register_algorithm("dummy", dummy)
        M = consensus_matrix(
            binary_matrix, k=2, algorithms=["dummy"], n_runs=5, seed=42, n_jobs=1
        )
        assert M.shape == (50, 50)

    def test_select_k_returns_min_nvi(self):
        from bacttraitcluster.clustering.consensus import KResult, select_k

        results = [
            KResult(2, 0.5, 0.1, 0.3, 0.7, 0.4, np.eye(2), np.array([0, 1])),
            KResult(3, 0.2, 0.05, 0.1, 0.3, 0.6, np.eye(2), np.array([0, 1, 2])),
            KResult(4, 0.3, 0.08, 0.2, 0.4, 0.5, np.eye(2), np.array([0, 1, 2, 3])),
        ]
        best = select_k(results)
        assert best.k == 3


# ── profiling tests ──────────────────────────────────────────────────────


class TestProfiling:
    def test_cliff_delta_range(self, binary_df):
        from bacttraitcluster.profiling.importance import cliff_delta_table

        labels = np.array([0] * 17 + [1] * 17 + [2] * 16)
        cd = cliff_delta_table(binary_df, labels, n_boot=50, seed=42)
        assert "Delta" in cd.columns
        assert cd["Delta"].between(-1, 1).all()

    def test_cliff_delta_magnitude(self, binary_df):
        from bacttraitcluster.profiling.importance import cliff_delta_table

        labels = np.array([0] * 17 + [1] * 17 + [2] * 16)
        cd = cliff_delta_table(binary_df, labels, n_boot=50, seed=42)
        assert set(cd["Magnitude"].unique()).issubset(
            {"negligible", "small", "medium", "large"}
        )

    def test_enrichment_z_shape(self, binary_df):
        from bacttraitcluster.profiling.importance import enrichment_z

        labels = np.array([0] * 17 + [1] * 17 + [2] * 16)
        ez = enrichment_z(binary_df, labels)
        assert ez.shape[0] == 10 * 3  # 10 features × 3 clusters
        assert "Z" in ez.columns
        assert "P_adj" in ez.columns

    def test_enrichment_z_detects_signal(self, binary_df):
        from bacttraitcluster.profiling.importance import enrichment_z

        labels = np.array([0] * 17 + [1] * 17 + [2] * 16)
        ez = enrichment_z(binary_df, labels)
        # Cluster 0 should be enriched for feat_0..feat_3
        c0 = ez[
            (ez["Cluster"] == 0)
            & (ez["Feature"].isin(["feat_0", "feat_1", "feat_2", "feat_3"]))
        ]
        assert (c0["Z"] > 0).all()


# ── config tests ─────────────────────────────────────────────────────────


class TestConfig:
    def test_roundtrip_yaml(self, tmp_path):
        from bacttraitcluster.config import Config, LayerSpec

        cfg = Config(layers=[LayerSpec("test", "/tmp/x.csv")])
        p = tmp_path / "cfg.yaml"
        cfg.to_yaml(p)
        cfg2 = Config.from_yaml(p)
        assert cfg2.layers[0].name == "test"
        assert cfg2.seed == 42
