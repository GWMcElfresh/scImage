"""Tests for scimage.layout — GW layout and ScImageLayout class.

All tests use a tiny 4 × 4 grid (16 genes / pixels) to keep runtimes short.
"""

import os
import tempfile

import numpy as np
import pytest

from scimage.layout import ScImageLayout, _pixel_coords, compute_gw_layout


GRID = 4          # 4 × 4
G = GRID ** 2     # 16 genes = 16 pixels
N_CELLS = 30

RNG = np.random.default_rng(1)
X_small = RNG.poisson(lam=3.0, size=(N_CELLS, G)).astype(np.float32)


# ---------------------------------------------------------------------------
# _pixel_coords
# ---------------------------------------------------------------------------

def test_pixel_coords_shape():
    coords = _pixel_coords(grid_size=GRID)
    assert coords.shape == (G, 2)
    assert coords.dtype == np.float32


def test_pixel_coords_column_major_order():
    """Pixel 0 should be at (0, 0), pixel 1 at (1, 0), pixel GRID at (0, 1)."""
    coords = _pixel_coords(grid_size=GRID)
    np.testing.assert_array_equal(coords[0], [0, 0])
    np.testing.assert_array_equal(coords[1], [1, 0])
    np.testing.assert_array_equal(coords[GRID], [0, 1])


# ---------------------------------------------------------------------------
# compute_gw_layout
# ---------------------------------------------------------------------------

def test_compute_gw_layout_output_shape():
    P = compute_gw_layout(X_small, grid_size=GRID, n_iter=5)
    assert P.shape == (G, G)
    assert P.dtype == np.float32


def test_compute_gw_layout_column_marginals():
    """Each column of P / M should sum to 1/G (gene marginal)."""
    P = compute_gw_layout(X_small, grid_size=GRID, n_iter=5)
    M = GRID ** 2
    col_sums = P.sum(axis=1)  # sum over pixels for each gene
    expected = np.ones(G)  # P = M * T*, and T* has gene marginal 1/G
    np.testing.assert_allclose(col_sums, expected, atol=1e-4)


def test_compute_gw_layout_raises_wrong_gene_count():
    with pytest.raises(ValueError, match="gene columns"):
        compute_gw_layout(X_small[:, :5], grid_size=GRID, n_iter=2)


# ---------------------------------------------------------------------------
# ScImageLayout — properties
# ---------------------------------------------------------------------------

def _make_layout():
    projection = RNG.random((G, G)).astype(np.float32)
    gene_indices = np.arange(G, dtype=np.int64)
    gene_mean = RNG.random(G).astype(np.float32)
    gene_std = (RNG.random(G) + 0.5).astype(np.float32)
    return ScImageLayout(projection, gene_indices, gene_mean, gene_std, grid_size=GRID)


def test_layout_properties():
    layout = _make_layout()
    assert layout.n_genes == G
    assert layout.n_pixels == G
    assert layout.grid_size == GRID


# ---------------------------------------------------------------------------
# ScImageLayout.fit
# ---------------------------------------------------------------------------

def test_fit_returns_layout():
    layout = ScImageLayout.fit(X_small, n_genes=G, grid_size=GRID, n_iter=5)
    assert isinstance(layout, ScImageLayout)
    assert layout.n_genes == G
    assert layout.grid_size == GRID


def test_fit_with_cell_subsampling():
    layout = ScImageLayout.fit(
        X_small,
        n_genes=G,
        grid_size=GRID,
        n_iter=5,
        n_cells_sample=20,
        seed=42,
    )
    assert layout.n_genes == G


# ---------------------------------------------------------------------------
# ScImageLayout.transform
# ---------------------------------------------------------------------------

def test_transform_single_cell_shape():
    layout = _make_layout()
    # Full gene expression vector (same as gene_indices ordering for this toy)
    expr = RNG.random(G).astype(np.float32)
    img = layout.transform(expr)
    assert img.shape == (1, GRID, GRID)
    assert img.dtype == np.float16


def test_transform_batch_shape():
    layout = _make_layout()
    expr_batch = RNG.random((5, G)).astype(np.float32)
    imgs = layout.transform(expr_batch)
    assert imgs.shape == (5, 1, GRID, GRID)
    assert imgs.dtype == np.float16


def test_transform_fortran_reshape_consistent():
    """Verify that the batch reshape matches per-cell Fortran reshapes."""
    layout = _make_layout()
    expr_batch = RNG.random((3, G)).astype(np.float32)
    imgs_batch = layout.transform(expr_batch)

    for i in range(3):
        img_single = layout.transform(expr_batch[i])
        np.testing.assert_array_equal(img_single, imgs_batch[i])


# ---------------------------------------------------------------------------
# ScImageLayout.save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    layout = _make_layout()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layout")
        layout.save(path)
        loaded = ScImageLayout.load(path)

    assert loaded.grid_size == layout.grid_size
    np.testing.assert_array_equal(loaded.gene_indices, layout.gene_indices)
    np.testing.assert_allclose(loaded.projection, layout.projection)
    np.testing.assert_allclose(loaded.gene_mean, layout.gene_mean)
    np.testing.assert_allclose(loaded.gene_std, layout.gene_std)


def test_transform_same_after_save_load():
    layout = _make_layout()
    expr = RNG.random((4, G)).astype(np.float32)
    imgs_before = layout.transform(expr)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layout")
        layout.save(path)
        loaded = ScImageLayout.load(path)

    imgs_after = loaded.transform(expr)
    np.testing.assert_array_equal(imgs_before, imgs_after)
