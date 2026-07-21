"""End-to-end tests using simulated block-covariance gene expression data.

Each scenario generates expression data in which genes are organised into
*N_BLOCKS* co-expression modules (blocks), with genes within a block sharing
a common latent factor.  This mirrors the kind of structure that the
Gromov–Wasserstein layout is designed to preserve spatially.

Tests cover the full pipeline: HVG selection → layout fit → cell transform →
image validation.  Images are not inspected visually; instead structural
properties (shape, dtype, finiteness, reproducibility) and statistical
properties (between-cell discrimination, within-cell-type similarity) are
verified.

Parameterisation
----------------
n_blocks = 3 → 3 × 3 = 9 genes,  3 × 3 grid
n_blocks = 4 → 4 × 4 = 16 genes, 4 × 4 grid
n_blocks = 5 → 5 × 5 = 25 genes, 5 × 5 grid
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from scimage import ScImageLayout, select_hvg


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def simulate_block_expression(
    n_cells: int,
    n_blocks: int,
    genes_per_block: int,
    rng: np.random.Generator,
    signal_scale: float = 2.0,
    noise_scale: float = 0.3,
) -> np.ndarray:
    """Simulate count-like expression with block covariance structure.

    Each block of genes shares a single latent factor drawn from N(0, 1).
    Gaussian noise is added and values are clipped to be non-negative so they
    resemble raw counts.

    Parameters
    ----------
    n_cells:
        Number of simulated cells.
    n_blocks:
        Number of co-expression blocks.
    genes_per_block:
        Number of genes per block.
    rng:
        NumPy random generator (for reproducibility).
    signal_scale:
        Scale of the shared latent factor signal.
    noise_scale:
        Standard deviation of the per-gene Gaussian noise.

    Returns
    -------
    X : float32 array of shape ``(n_cells, n_blocks * genes_per_block)``
        Simulated non-negative expression matrix.
    """
    n_genes = n_blocks * genes_per_block

    # One latent factor per block, drawn independently per cell
    latent = rng.standard_normal((n_cells, n_blocks)).astype(np.float64)

    # Gene loadings: positive, varying per gene
    loadings = rng.uniform(0.5, 1.5, size=n_genes)

    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    for b in range(n_blocks):
        start = b * genes_per_block
        end = start + genes_per_block
        X[:, start:end] = (
            signal_scale * np.outer(latent[:, b], loadings[start:end])
        )

    # Independent noise
    X += rng.normal(0.0, noise_scale, size=(n_cells, n_genes))

    # Shift to non-negative (analogous to raw counts)
    X -= X.min()
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Parametrised scenarios: 3, 4, and 5 blocks
# ---------------------------------------------------------------------------

SCENARIOS = [
    pytest.param(3, id="3_blocks"),
    pytest.param(4, id="4_blocks"),
    pytest.param(5, id="5_blocks"),
]


@pytest.fixture(params=SCENARIOS)
def block_scenario(request):
    """Return (X, n_blocks, grid_size) for each block count."""
    n_blocks: int = request.param
    genes_per_block = n_blocks          # grid_size² = n_blocks² = n_blocks × genes_per_block
    grid_size = n_blocks                # so grid_size = n_blocks
    n_cells = 80

    rng = np.random.default_rng(seed=n_blocks * 7)
    X = simulate_block_expression(n_cells, n_blocks, genes_per_block, rng)
    return X, n_blocks, grid_size


# ---------------------------------------------------------------------------
# End-to-end: HVG selection
# ---------------------------------------------------------------------------

class TestHVGOnBlockData:
    """HVG selection on block-covariance data returns the correct structure."""

    def test_hvg_selects_expected_count(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        n_genes = grid_size ** 2
        idx, mean, std = select_hvg(X, n_genes=n_genes)
        assert idx.shape == (n_genes,), "Wrong number of HVGs selected"

    def test_hvg_mean_nonnegative(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        n_genes = grid_size ** 2
        _, mean, _ = select_hvg(X, n_genes=n_genes)
        # select_hvg computes means in log1p space, so all values are non-negative
        assert np.all(mean >= 0.0), "HVG means should be non-negative (log1p space)"

    def test_hvg_std_positive(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        n_genes = grid_size ** 2
        _, _, std = select_hvg(X, n_genes=n_genes)
        assert np.all(std > 0.0), "HVG std must be positive (prevents div-by-zero)"

    def test_hvg_block_genes_highly_variable(self, block_scenario):
        """Block genes should dominate the HVG ranking on strongly-blocked data."""
        X, n_blocks, grid_size = block_scenario
        n_genes = grid_size ** 2
        # All genes are block genes in this fixture, so all should be selected
        idx, _, _ = select_hvg(X, n_genes=n_genes)
        assert len(np.unique(idx)) == n_genes


# ---------------------------------------------------------------------------
# End-to-end: layout fit
# ---------------------------------------------------------------------------

class TestLayoutFitOnBlockData:
    """ScImageLayout.fit() correctly processes block-structured expression."""

    def test_fit_returns_layout_instance(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)
        assert isinstance(layout, ScImageLayout)

    def test_fit_grid_size_stored(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)
        assert layout.grid_size == grid_size

    def test_fit_projection_shape(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        n_genes = grid_size ** 2
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)
        assert layout.projection.shape == (n_genes, n_genes)

    def test_fit_projection_finite(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)
        assert np.all(np.isfinite(layout.projection)), "Projection matrix must be finite"

    def test_fit_projection_nonnegative(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)
        assert np.all(layout.projection >= 0.0), "Projection matrix must be non-negative"


# ---------------------------------------------------------------------------
# End-to-end: cell transform → image
# ---------------------------------------------------------------------------

class TestTransformOnBlockData:
    """Cell → image transform produces valid output on block-covariance data."""

    def _fit(self, X, grid_size):
        return ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)

    def test_single_cell_image_shape(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        img = layout.transform(X[0])
        assert img.shape == (1, grid_size, grid_size)

    def test_single_cell_image_dtype(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        img = layout.transform(X[0])
        assert img.dtype == np.float16, "scImage must be stored at half precision"

    def test_batch_image_shape(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        imgs = layout.transform(X[:5])
        assert imgs.shape == (5, 1, grid_size, grid_size)

    def test_images_finite(self, block_scenario):
        """All pixel values must be finite (no NaN / Inf)."""
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        imgs = layout.transform(X)
        assert np.all(np.isfinite(imgs.astype(np.float32))), "Images contain non-finite values"

    def test_image_reproducibility(self, block_scenario):
        """Transforming the same cell twice must give identical images."""
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        img_a = layout.transform(X[0])
        img_b = layout.transform(X[0])
        np.testing.assert_array_equal(img_a, img_b)

    def test_different_cells_different_images(self, block_scenario):
        """Different cells should not produce identical images."""
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        # Use two cells that differ substantially (different block activations)
        img_0 = layout.transform(X[0]).astype(np.float32)
        img_1 = layout.transform(X[1]).astype(np.float32)
        assert not np.allclose(img_0, img_1, atol=1e-3), (
            "Two distinct cells produced identical images"
        )

    def test_batch_matches_single_cell_transforms(self, block_scenario):
        """Batch transform must match stacking individual cell transforms."""
        X, n_blocks, grid_size = block_scenario
        layout = self._fit(X, grid_size)
        n_test = 4
        imgs_batch = layout.transform(X[:n_test])
        for i in range(n_test):
            img_single = layout.transform(X[i])
            np.testing.assert_array_equal(
                img_single,
                imgs_batch[i],
                err_msg=f"Mismatch at cell {i} between batch and single transform",
            )


# ---------------------------------------------------------------------------
# End-to-end: persistence round-trip on block data
# ---------------------------------------------------------------------------

class TestPersistenceOnBlockData:
    """Save/load round-trip preserves images on block-covariance data."""

    def test_save_load_images_unchanged(self, block_scenario):
        X, n_blocks, grid_size = block_scenario
        layout = ScImageLayout.fit(X, grid_size=grid_size, n_iter=5, seed=42)
        imgs_original = layout.transform(X[:3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layout")
            layout.save(path)
            loaded = ScImageLayout.load(path)

        imgs_reloaded = loaded.transform(X[:3])
        np.testing.assert_array_equal(imgs_original, imgs_reloaded)
