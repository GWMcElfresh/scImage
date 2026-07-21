"""Tests for scimage.hvg — highly variable gene selection."""

import numpy as np
import pytest
import scipy.sparse as sp

from scimage.hvg import select_hvg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CELLS = 50
N_GENES = 30
N_HVG = 10
RNG = np.random.default_rng(0)

# Dense expression matrix: non-negative counts
X_dense = RNG.poisson(lam=2.0, size=(N_CELLS, N_GENES)).astype(np.float32)
# Sparse version of the same data
X_sparse = sp.csr_matrix(X_dense)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_returns_correct_number_of_genes():
    idx, mean, std = select_hvg(X_dense, n_genes=N_HVG)
    assert idx.shape == (N_HVG,)
    assert mean.shape == (N_HVG,)
    assert std.shape == (N_HVG,)


def test_output_dtypes():
    idx, mean, std = select_hvg(X_dense, n_genes=N_HVG)
    assert idx.dtype == np.int64
    assert mean.dtype == np.float32
    assert std.dtype == np.float32


def test_indices_in_valid_range():
    idx, _, _ = select_hvg(X_dense, n_genes=N_HVG)
    assert np.all(idx >= 0)
    assert np.all(idx < N_GENES)


def test_no_duplicate_indices():
    idx, _, _ = select_hvg(X_dense, n_genes=N_HVG)
    assert len(np.unique(idx)) == N_HVG


def test_std_positive():
    """Zero-variance genes should get std = 1 to prevent division by zero."""
    idx, _, std = select_hvg(X_dense, n_genes=N_HVG)
    assert np.all(std > 0.0)


# ---------------------------------------------------------------------------
# Sparse input
# ---------------------------------------------------------------------------

def test_sparse_input_same_result():
    idx_dense, mean_dense, std_dense = select_hvg(X_dense, n_genes=N_HVG)
    idx_sparse, mean_sparse, std_sparse = select_hvg(X_sparse, n_genes=N_HVG)
    np.testing.assert_array_equal(idx_dense, idx_sparse)
    np.testing.assert_allclose(mean_dense, mean_sparse, rtol=1e-5)
    np.testing.assert_allclose(std_dense, std_sparse, rtol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases / error handling
# ---------------------------------------------------------------------------

def test_raises_if_n_genes_too_large():
    with pytest.raises(ValueError, match="only"):
        select_hvg(X_dense, n_genes=N_GENES + 1)


def test_n_genes_equals_all_genes():
    idx, mean, std = select_hvg(X_dense, n_genes=N_GENES)
    assert idx.shape == (N_GENES,)


def test_chunked_pass_consistent():
    """Results must be identical for different chunk sizes."""
    idx1, mean1, std1 = select_hvg(X_dense, n_genes=N_HVG, chunk_size=5)
    idx2, mean2, std2 = select_hvg(X_dense, n_genes=N_HVG, chunk_size=N_CELLS)
    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_allclose(mean1, mean2, rtol=1e-5)
    np.testing.assert_allclose(std1, std2, rtol=1e-5)
