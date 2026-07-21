"""scImage — single-cell expression images via Gromov–Wasserstein optimal transport.

The package implements the scImage construction method described in:

    *scImage construction by optimal transport* (arXiv 2607.14163v1).

Each cell is rendered as a single-channel image ``(1, 104, 104)`` where genes
are assigned to pixels through a Gromov–Wasserstein transport plan that aligns
gene co-expression geometry with the 2-D pixel lattice geometry.

Quickstart
----------
>>> from scimage import ScImageLayout
>>> layout = ScImageLayout.fit(X)          # X: (n_cells, n_genes) expression matrix
>>> image = layout.transform(expression)   # image: float16 array (1, 104, 104)
>>> layout.save("my_layout")
>>> layout = ScImageLayout.load("my_layout.npz")
"""

from .hvg import select_hvg
from .layout import ScImageLayout, compute_gw_layout

__all__ = [
    "ScImageLayout",
    "compute_gw_layout",
    "select_hvg",
]
