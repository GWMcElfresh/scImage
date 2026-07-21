# scImage

Single-cell expression images via **Gromov–Wasserstein optimal transport**.

This package implements the *scImage* construction method described in:

> *scImage construction by optimal transport* — arXiv 2607.14163v1

Each cell is rendered as a single-channel `(1, 104, 104)` image in which
**10 816 highly variable genes** are assigned to pixels via a
Gromov–Wasserstein transport plan that aligns gene co-expression geometry with
the two-dimensional pixel-lattice geometry.

---

## How it works

1. **HVG selection** — The *G* = 104² = 10 816 most highly variable genes are
   selected with a Seurat-style criterion: a LOESS curve is fit to
   log1p(variance) vs log1p(mean) expression (single chunked pass), and genes
   are ranked by their positive residual.

2. **Layout computation** — A gene–gene correlation-distance matrix
   **D**ᵍ (entries 1 − *r*ᵢₖ) and a pixel–pixel Euclidean-distance matrix
   **D**ˢ are aligned by solving the Gromov–Wasserstein problem with KL
   divergence loss and uniform marginals (no entropic regularisation,
   200 iterations).  The resulting coupling **T**★ is stored as the scaled
   projection matrix **P** = *M* · **T**★.

3. **Cell transform** — Each cell's raw expression **e** ∈ ℝᴳ is
   *z*-scored gene-wise, projected to pixel space (**y** = **P**ᵀ **ẽ**),
   and reshaped into a 104 × 104 image in column-major (Fortran) order.
   The result is stored at half precision as a `(1, 104, 104)` tensor.

---

## Installation

```bash
pip install -e ".[dev]"
```

**Dependencies:** `numpy`, `scipy`, `POT` (Python Optimal Transport),
`statsmodels`.

---

## Quick start

```python
import numpy as np
from scimage import ScImageLayout

# X: (n_cells, n_genes) expression matrix — raw counts or log-normalised
X = np.random.poisson(2, size=(500, 20_000)).astype("float32")

# Fit the layout (use n_cells_sample to subsample, as in the paper: 100 000 cells)
layout = ScImageLayout.fit(X, n_cells_sample=500, seed=42)

# Transform a single cell or a batch
image = layout.transform(X[0])            # -> float16 array (1, 104, 104)
images = layout.transform(X[:10])         # -> float16 array (10, 1, 104, 104)

# Persist for reuse
layout.save("my_layout")
layout = ScImageLayout.load("my_layout.npz")
```

---

## API reference

### `ScImageLayout`

| Method / property | Description |
|---|---|
| `ScImageLayout.fit(X, ...)` | Fit layout from expression matrix |
| `layout.transform(expression)` | Cell(s) -> scImage(s) |
| `layout.save(path)` | Persist to `.npz` |
| `ScImageLayout.load(path)` | Restore from `.npz` |
| `layout.projection` | Soft projection matrix **P** (G x M) |
| `layout.gene_indices` | HVG column indices |
| `layout.gene_mean` / `gene_std` | Z-scoring statistics |

### `select_hvg(X, n_genes, ...)`

Low-level HVG selection returning `(hvg_indices, gene_mean, gene_std)`.

### `compute_gw_layout(expression_matrix, grid_size, n_iter, ...)`

Low-level GW layout computation returning the projection matrix **P**.

---

## Running tests

```bash
pytest
```
