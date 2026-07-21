"""Highly variable gene selection using a Seurat-style criterion.

Each gene is ranked by its residual variance after fitting a LOESS curve of
``log1p(variance)`` against ``log1p(mean)`` expression.  The top *n_genes*
genes with the largest positive residuals are returned together with the
per-gene statistics required for downstream z-scoring.
"""

from __future__ import annotations

import numpy as np

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False


def _compute_mean_var_chunked(X, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return per-gene mean and unbiased variance in a single chunked pass.

    Supports dense NumPy arrays and SciPy sparse matrices.

    Parameters
    ----------
    X:
        Expression matrix ``(n_cells, n_genes)``.
    chunk_size:
        Number of rows processed per iteration.

    Returns
    -------
    mean, var : float64 arrays of shape ``(n_genes,)``
    """
    n_cells, n_genes = X.shape
    sum_ = np.zeros(n_genes, dtype=np.float64)
    sum_sq = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = X[start:end]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        chunk = np.asarray(chunk, dtype=np.float64)
        # log1p-transform in-place on the copy
        np.log1p(chunk, out=chunk)
        sum_ += chunk.sum(axis=0)
        sum_sq += (chunk ** 2).sum(axis=0)

    mean = sum_ / n_cells
    # Bessel-corrected variance
    var = (sum_sq / n_cells - mean ** 2) * (n_cells / max(n_cells - 1, 1))
    return mean, var


def select_hvg(
    X,
    n_genes: int = 10_816,
    chunk_size: int = 1024,
    loess_frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select highly variable genes using a Seurat-style criterion.

    Expression values are ``log1p``-transformed before computing statistics.
    A LOESS curve of ``log1p(variance)`` against ``log1p(mean)`` is fit and
    genes are ranked by their residual (observed − fitted) in descending order.

    Parameters
    ----------
    X:
        Expression matrix of shape ``(n_cells, n_all_genes)`` — either a dense
        NumPy array or a SciPy sparse matrix.
    n_genes:
        Number of highly variable genes to retain.  Defaults to
        ``104 ** 2 = 10 816`` to fill a 104 × 104 scImage grid.
    chunk_size:
        Rows processed per pass when computing mean/variance.
    loess_frac:
        Fraction of data used for each LOESS estimate.

    Returns
    -------
    hvg_indices : int64 array of shape ``(n_genes,)``
        Column indices of the selected HVGs in the original matrix, ordered by
        decreasing residual variance.
    gene_mean : float32 array of shape ``(n_genes,)``
        Per-gene mean ``log1p`` expression.
    gene_std : float32 array of shape ``(n_genes,)``
        Per-gene standard deviation of ``log1p`` expression.  Zero-variance
        genes are assigned ``std = 1`` to avoid division by zero downstream.

    Raises
    ------
    ImportError
        If *statsmodels* is not installed.
    ValueError
        If *n_genes* exceeds the number of columns in *X*.
    """
    if not _HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for LOESS fitting. "
            "Install it with:  pip install statsmodels"
        )

    n_all_genes = X.shape[1]
    if n_genes > n_all_genes:
        raise ValueError(
            f"Requested {n_genes} HVGs but the matrix has only {n_all_genes} genes."
        )

    mean, var = _compute_mean_var_chunked(X, chunk_size=chunk_size)

    # LOESS fit of log1p(variance) vs log1p(mean)
    log_mean = np.log1p(mean)
    log_var = np.log1p(var)
    fitted = _lowess(log_var, log_mean, frac=loess_frac, return_sorted=False)

    residual = log_var - fitted

    # Top-n_genes by residual (descending)
    hvg_indices = np.argsort(residual)[::-1][:n_genes].copy().astype(np.int64)

    gene_mean = mean[hvg_indices].astype(np.float32)
    gene_std = np.sqrt(np.maximum(var[hvg_indices], 0.0)).astype(np.float32)
    # Prevent division by zero in z-scoring
    gene_std = np.where(gene_std == 0.0, np.float32(1.0), gene_std)

    return hvg_indices, gene_mean, gene_std
