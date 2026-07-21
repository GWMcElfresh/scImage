"""Tests for AnnData integration, sparse matrix handling, and metadata tracking.

Covers:
* ``ScImageLayout.fit()`` accepts ``anndata.AnnData`` (dense and sparse ``.X``)
* Gene names from ``adata.var_names`` are stored in ``layout.gene_names``
* ``transform()`` aligns genes by name when applied to AnnData
* ``transform()`` works on SciPy sparse matrices
* ``transform_adata()`` stores images in ``adata.obsm``
* Save/load round-trip preserves ``gene_names``
* Cross-dataset transfer via name-based alignment (permuted gene order)
* Informative ``ValueError`` on missing genes
* Results are consistent between AnnData and plain NumPy / sparse inputs
* End-to-end block-covariance scenarios using AnnData

Parameterisation (block scenarios)
-----------------------------------
n_blocks = 3 → 3 × 3 = 9 genes,  3 × 3 grid
n_blocks = 4 → 4 × 4 = 16 genes, 4 × 4 grid
n_blocks = 5 → 5 × 5 = 25 genes, 5 × 5 grid
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

import anndata as ad

from scimage import ScImageLayout, select_hvg


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_block_expression(
    n_cells: int,
    n_blocks: int,
    genes_per_block: int,
    seed: int = 0,
    signal_scale: float = 2.0,
    noise_scale: float = 0.3,
) -> np.ndarray:
    """Simulate count-like expression with block covariance structure."""
    n_genes = n_blocks * genes_per_block
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_cells, n_blocks)).astype(np.float64)
    loadings = rng.uniform(0.5, 1.5, size=n_genes)

    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    for b in range(n_blocks):
        start = b * genes_per_block
        end = start + genes_per_block
        X[:, start:end] = signal_scale * np.outer(latent[:, b], loadings[start:end])

    X += rng.normal(0.0, noise_scale, size=(n_cells, n_genes))
    X -= X.min()
    return X.astype(np.float32)


def make_adata(
    n_cells: int,
    n_blocks: int,
    genes_per_block: int,
    seed: int = 0,
    sparse: bool = False,
    shuffle_genes: bool = False,
) -> ad.AnnData:
    """Create a minimal AnnData with block-covariance expression.

    Parameters
    ----------
    shuffle_genes:
        If True, randomly permute the gene columns so the column order
        differs from the fitting order.  Gene names remain the same.
    """
    X = make_block_expression(n_cells, n_blocks, genes_per_block, seed=seed)
    n_genes = X.shape[1]
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    cell_names = [f"cell_{i}" for i in range(n_cells)]

    if shuffle_genes:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(n_genes)
        X = X[:, perm]
        gene_names = [gene_names[i] for i in perm]

    import pandas as pd

    obs = pd.DataFrame(
        {"cell_type": [f"type_{i % 3}" for i in range(n_cells)]},
        index=cell_names,
    )
    var = pd.DataFrame(
        {"gene_id": gene_names},
        index=gene_names,
    )

    X_store = sp.csr_matrix(X) if sparse else X
    return ad.AnnData(X=X_store, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Parametrised block scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    pytest.param(3, id="3_blocks"),
    pytest.param(4, id="4_blocks"),
    pytest.param(5, id="5_blocks"),
]


@pytest.fixture(params=SCENARIOS)
def adata_scenario(request):
    """Return (adata_dense, grid_size) for each block count."""
    n_blocks: int = request.param
    grid_size = n_blocks
    adata = make_adata(n_cells=80, n_blocks=n_blocks, genes_per_block=n_blocks, seed=n_blocks * 13)
    return adata, grid_size


# ---------------------------------------------------------------------------
# Fitting from AnnData
# ---------------------------------------------------------------------------

class TestFitFromAnnData:
    """ScImageLayout.fit() correctly processes AnnData inputs."""

    def test_fit_dense_adata(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        assert isinstance(layout, ScImageLayout)

    def test_fit_sparse_adata(self, adata_scenario):
        adata, grid_size = adata_scenario
        adata_sp = make_adata(
            n_cells=adata.n_obs,
            n_blocks=grid_size,
            genes_per_block=grid_size,
            sparse=True,
        )
        layout = ScImageLayout.fit(adata_sp, grid_size=grid_size, n_iter=5)
        assert isinstance(layout, ScImageLayout)

    def test_fit_stores_gene_names(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        assert layout.gene_names is not None, "gene_names should be set after fitting from AnnData"
        assert len(layout.gene_names) == grid_size ** 2

    def test_gene_names_are_valid_var_names(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        var_set = set(adata.var_names)
        for name in layout.gene_names:
            assert name in var_set, f"Stored gene name '{name}' not in adata.var_names"

    def test_fit_plain_array_no_gene_names(self, adata_scenario):
        adata, grid_size = adata_scenario
        X = np.asarray(adata.X)
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5)
        assert layout.gene_names is None, "gene_names should be None when fitting from plain array"

    def test_fit_sparse_plain_no_gene_names(self, adata_scenario):
        adata, grid_size = adata_scenario
        X_sp = sp.csr_matrix(np.asarray(adata.X))
        layout = ScImageLayout.fit(X_sp, grid_size=grid_size, n_iter=5)
        assert layout.gene_names is None


# ---------------------------------------------------------------------------
# Transform with AnnData input
# ---------------------------------------------------------------------------

class TestTransformWithAnnData:
    """transform() and transform_adata() handle AnnData inputs correctly."""

    def _fit_from_adata(self, adata, grid_size):
        return ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5, seed=42)

    def test_transform_adata_returns_float16(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)
        images = layout.transform(adata)
        assert images.dtype == np.float16

    def test_transform_adata_batch_shape(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)
        images = layout.transform(adata)
        assert images.shape == (adata.n_obs, 1, grid_size, grid_size)

    def test_transform_adata_finite(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)
        images = layout.transform(adata)
        assert np.all(np.isfinite(images.astype(np.float32)))

    def test_transform_adata_matches_plain_array(self, adata_scenario):
        """Transforming AnnData and equivalent plain array must yield identical images."""
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)

        images_adata = layout.transform(adata)

        # Build a plain array already aligned to the HVG gene order using gene_names.
        var_names = list(adata.var_names)
        name_to_col = {n: i for i, n in enumerate(var_names)}
        col_indices = np.array([name_to_col[g] for g in layout.gene_names])
        X_aligned = np.asarray(adata.X)[:, col_indices]

        # Create an equivalent layout with sequential gene_indices so the
        # plain-array path (no gene_names) applies the same transform on the
        # pre-aligned matrix without mutating the original layout.
        layout_seq = ScImageLayout(
            projection=layout.projection,
            gene_indices=np.arange(len(layout.gene_names), dtype=np.int64),
            gene_mean=layout.gene_mean,
            gene_std=layout.gene_std,
            grid_size=layout.grid_size,
        )
        images_plain = layout_seq.transform(X_aligned)

        np.testing.assert_array_equal(images_adata, images_plain)

    def test_transform_adata_permuted_genes(self, adata_scenario):
        """Gene-name alignment must handle permuted column order transparently."""
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)

        # Permute all gene columns in a new AnnData
        n_genes = adata.n_vars
        rng = np.random.default_rng(99)
        perm = rng.permutation(n_genes)
        X_perm = np.asarray(adata.X)[:, perm]
        gene_names_perm = list(adata.var_names[perm])
        adata_perm = ad.AnnData(
            X=X_perm,
            obs=adata.obs.copy(),
        )
        adata_perm.var_names = gene_names_perm

        images_orig = layout.transform(adata)
        images_perm = layout.transform(adata_perm)
        np.testing.assert_array_equal(images_orig, images_perm)

    def test_transform_adata_sparse_x(self, adata_scenario):
        """transform() must handle AnnData whose .X is a sparse matrix."""
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)

        adata_sp = adata.copy()
        adata_sp.X = sp.csr_matrix(np.asarray(adata.X))
        images_sp = layout.transform(adata_sp)
        images_dense = layout.transform(adata)
        np.testing.assert_array_equal(images_sp, images_dense)

    def test_transform_missing_gene_raises(self, adata_scenario):
        """ValueError raised when the AnnData is missing HVG genes."""
        adata, grid_size = adata_scenario
        layout = self._fit_from_adata(adata, grid_size)

        # Drop first gene column to trigger missing-gene error
        adata_missing = adata[:, adata.var_names[1:]].copy()
        with pytest.raises(ValueError, match="not found in adata.var_names"):
            layout.transform(adata_missing)


# ---------------------------------------------------------------------------
# transform_adata() convenience method
# ---------------------------------------------------------------------------

class TestTransformAdataMethod:
    """transform_adata() stores images in obsm and returns the modified AnnData."""

    def _fit(self, adata, grid_size):
        return ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)

    def test_obsm_key_default(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)
        result = layout.transform_adata(adata)
        assert "X_scimage" in result.obsm

    def test_obsm_key_custom(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)
        result = layout.transform_adata(adata, obsm_key="my_images")
        assert "my_images" in result.obsm

    def test_obsm_shape(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)
        layout.transform_adata(adata)
        expected_cols = grid_size ** 2  # 1 * H * W = H * W (channel dim squeezed)
        assert adata.obsm["X_scimage"].shape == (adata.n_obs, expected_cols)

    def test_obsm_dtype(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)
        layout.transform_adata(adata)
        assert adata.obsm["X_scimage"].dtype == np.float16

    def test_obsm_values_finite(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)
        layout.transform_adata(adata)
        assert np.all(np.isfinite(adata.obsm["X_scimage"].astype(np.float32)))

    def test_transform_adata_returns_adata(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)
        result = layout.transform_adata(adata)
        assert result is adata, "transform_adata should return the same AnnData object"

    def test_obsm_consistent_with_transform(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = self._fit(adata, grid_size)

        images = layout.transform(adata)  # (n_cells, 1, H, W)
        layout.transform_adata(adata)     # stores flattened in obsm

        expected = images.reshape(adata.n_obs, -1)
        np.testing.assert_array_equal(adata.obsm["X_scimage"], expected)


# ---------------------------------------------------------------------------
# Sparse matrix support (plain scipy sparse, not AnnData)
# ---------------------------------------------------------------------------

class TestSparseMatrixTransform:
    """transform() handles plain SciPy sparse matrices."""

    def test_sparse_csr_transform(self, adata_scenario):
        adata, grid_size = adata_scenario
        X_dense = make_block_expression(
            n_cells=adata.n_obs,
            n_blocks=grid_size,
            genes_per_block=grid_size,
        )
        layout = ScImageLayout.fit(X_dense, grid_size=grid_size, n_iter=5)

        X_sp = sp.csr_matrix(X_dense)
        images_sp = layout.transform(X_sp)
        images_dense = layout.transform(X_dense)
        np.testing.assert_array_equal(images_sp, images_dense)

    def test_sparse_csc_transform(self, adata_scenario):
        adata, grid_size = adata_scenario
        X_dense = make_block_expression(
            n_cells=adata.n_obs,
            n_blocks=grid_size,
            genes_per_block=grid_size,
        )
        layout = ScImageLayout.fit(X_dense, grid_size=grid_size, n_iter=5)

        X_sp = sp.csc_matrix(X_dense)
        images_sp = layout.transform(X_sp)
        images_dense = layout.transform(X_dense)
        np.testing.assert_array_equal(images_sp, images_dense)

    def test_sparse_transform_shape(self, adata_scenario):
        adata, grid_size = adata_scenario
        X_dense = make_block_expression(
            n_cells=10,
            n_blocks=grid_size,
            genes_per_block=grid_size,
        )
        layout = ScImageLayout.fit(X_dense, grid_size=grid_size, n_iter=5)

        X_sp = sp.csr_matrix(X_dense)
        images = layout.transform(X_sp)
        assert images.shape == (10, 1, grid_size, grid_size)


# ---------------------------------------------------------------------------
# Save / load preserves gene_names
# ---------------------------------------------------------------------------

class TestPersistenceWithGeneNames:
    """Save/load round-trip preserves gene_names and produces identical images."""

    def test_save_load_preserves_gene_names(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layout")
            layout.save(path)
            loaded = ScImageLayout.load(path)

        assert loaded.gene_names is not None
        np.testing.assert_array_equal(layout.gene_names, loaded.gene_names)

    def test_save_load_images_unchanged_with_gene_names(self, adata_scenario):
        adata, grid_size = adata_scenario
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        images_original = layout.transform(adata)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layout")
            layout.save(path)
            loaded = ScImageLayout.load(path)

        images_loaded = loaded.transform(adata)
        np.testing.assert_array_equal(images_original, images_loaded)

    def test_save_load_no_gene_names_unchanged(self, adata_scenario):
        adata, grid_size = adata_scenario
        X = np.asarray(adata.X)
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5)
        assert layout.gene_names is None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layout")
            layout.save(path)
            loaded = ScImageLayout.load(path)

        assert loaded.gene_names is None
        np.testing.assert_array_equal(
            layout.transform(X), loaded.transform(X)
        )


# ---------------------------------------------------------------------------
# Metadata tracking: obs columns are preserved after transform_adata
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    """transform_adata() must not destroy obs/var metadata."""

    def test_obs_columns_preserved(self, adata_scenario):
        adata, grid_size = adata_scenario
        original_obs_cols = list(adata.obs.columns)
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        layout.transform_adata(adata)
        assert list(adata.obs.columns) == original_obs_cols

    def test_var_names_preserved(self, adata_scenario):
        adata, grid_size = adata_scenario
        original_var_names = list(adata.var_names)
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        layout.transform_adata(adata)
        assert list(adata.var_names) == original_var_names

    def test_n_obs_unchanged(self, adata_scenario):
        adata, grid_size = adata_scenario
        n_obs_before = adata.n_obs
        layout = ScImageLayout.fit(adata, grid_size=grid_size, n_iter=5)
        layout.transform_adata(adata)
        assert adata.n_obs == n_obs_before
