"""Gromov–Wasserstein gene-to-pixel layout and the ``ScImageLayout`` class.

The layout fixes a single shared gene arrangement on a square pixel lattice,
computed once from a representative expression matrix and then applied to every
cell.  See :class:`ScImageLayout` for the high-level API and
:func:`compute_gw_layout` for the lower-level transport step.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

import ot

from .hvg import select_hvg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pixel_coords(grid_size: int = 104) -> np.ndarray:
    """Return 2-D pixel coordinates for a *grid_size* × *grid_size* grid.

    Pixels are indexed in column-major (Fortran) order so that pixel *j* maps
    to position ``(j % grid_size, j // grid_size)``.

    Parameters
    ----------
    grid_size:
        Side length of the square grid.

    Returns
    -------
    coords : float32 array of shape ``(grid_size ** 2, 2)``
    """
    M = grid_size ** 2
    rows = np.arange(M, dtype=np.float32) % grid_size
    cols = np.arange(M, dtype=np.float32) // grid_size
    return np.stack([rows, cols], axis=1)


# ---------------------------------------------------------------------------
# Core transport step
# ---------------------------------------------------------------------------

def compute_gw_layout(
    expression_matrix: np.ndarray,
    grid_size: int = 104,
    n_iter: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Compute the Gromov–Wasserstein gene-to-pixel layout.

    Given an expression matrix of the *G* most highly variable genes across
    *n_cells* cells, this function builds:

    * the **gene–gene correlation-distance matrix**
      ``D_g[i, k] = 1 − r(gene_i, gene_k)`` where *r* is the Pearson
      correlation computed across cells; and
    * the **pixel–pixel Euclidean-distance matrix**
      ``D_s[j, l] = ‖c_j − c_l‖₂`` for pixel coordinates ``c_j`` on the
      *grid_size* × *grid_size* lattice in column-major order.

    The coupling ``T*`` that minimises the KL-loss Gromov–Wasserstein objective
    (no entropic regularisation, *n_iter* iterations, uniform marginals) is
    returned as the scaled projection matrix ``P = M · T*``.

    Parameters
    ----------
    expression_matrix:
        Array of shape ``(n_cells, G)`` containing the log1p-normalised (or
        raw) expression of the *G* = *grid_size* ** 2 selected HVGs.
    grid_size:
        Side length of the square image grid (default 104).
    n_iter:
        Maximum number of conditional-gradient iterations (default 200).
    seed:
        Ignored — kept for API symmetry; the GW solver is deterministic given
        the default ``p·qᵀ`` initialisation.

    Returns
    -------
    P : float32 array of shape ``(G, M)``
        Soft projection matrix ``P = M · T*``, where ``M = grid_size ** 2``.
    """
    G = expression_matrix.shape[1]
    M = grid_size ** 2
    if G != M:
        raise ValueError(
            f"expression_matrix must have exactly {M} gene columns for a "
            f"{grid_size}×{grid_size} grid, but got {G}."
        )

    # ---- gene–gene correlation-distance matrix ----
    corr = np.corrcoef(np.asarray(expression_matrix, dtype=np.float64).T)
    D_g = np.clip(1.0 - corr, 0.0, None)  # ensure non-negative for KL loss

    # ---- pixel–pixel Euclidean-distance matrix ----
    coords = _pixel_coords(grid_size).astype(np.float64)
    D_s = cdist(coords, coords, metric="euclidean")

    # ---- uniform marginals ----
    mu = np.ones(G, dtype=np.float64) / G
    nu = np.ones(M, dtype=np.float64) / M

    # ---- Gromov–Wasserstein with KL loss, no entropic regularisation ----
    T_star = ot.gromov.gromov_wasserstein(
        D_g,
        D_s,
        mu,
        nu,
        loss_fun="kl_loss",
        max_iter=n_iter,
        verbose=False,
    )

    # Scale coupling by number of pixels → P = M · T*
    P = (M * T_star).astype(np.float32)
    return P


# ---------------------------------------------------------------------------
# High-level class
# ---------------------------------------------------------------------------

class ScImageLayout:
    """Stores the gene-to-pixel layout for converting cells to scImages.

    A layout is computed once from a representative expression matrix (via
    :meth:`fit`) and then used to transform any number of cells (via
    :meth:`transform`).  It can be persisted with :meth:`save` / :meth:`load`.

    Attributes
    ----------
    projection : float32 array of shape ``(n_genes, n_pixels)``
        Soft projection matrix ``P = M · T*``.
    gene_indices : int64 array of shape ``(n_genes,)``
        Column indices of the HVGs in the original expression matrix.
    gene_mean : float32 array of shape ``(n_genes,)``
        Per-gene mean ``log1p`` expression used for z-scoring.
    gene_std : float32 array of shape ``(n_genes,)``
        Per-gene standard deviation used for z-scoring.
    grid_size : int
        Side length of the square image grid.
    """

    #: Default grid side length (104 × 104 = 10 816 pixels / genes).
    GRID_SIZE: int = 104

    def __init__(
        self,
        projection: np.ndarray,
        gene_indices: np.ndarray,
        gene_mean: np.ndarray,
        gene_std: np.ndarray,
        grid_size: int = 104,
    ) -> None:
        self.projection = np.asarray(projection, dtype=np.float32)
        self.gene_indices = np.asarray(gene_indices, dtype=np.int64)
        self.gene_mean = np.asarray(gene_mean, dtype=np.float32)
        self.gene_std = np.asarray(gene_std, dtype=np.float32)
        self.grid_size = int(grid_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_genes(self) -> int:
        """Number of HVGs used in the layout."""
        return int(self.gene_indices.shape[0])

    @property
    def n_pixels(self) -> int:
        """Number of pixels (= ``grid_size ** 2``)."""
        return self.grid_size ** 2

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        X,
        n_genes: int | None = None,
        grid_size: int = 104,
        n_iter: int = 200,
        seed: int = 42,
        chunk_size: int = 1024,
        loess_frac: float = 0.3,
        n_cells_sample: int | None = None,
    ) -> "ScImageLayout":
        """Compute the layout from an expression matrix.

        Parameters
        ----------
        X:
            Expression matrix of shape ``(n_cells, n_all_genes)`` — a dense
            NumPy array or a SciPy sparse matrix.
        n_genes:
            Number of HVGs to select.  Defaults to ``grid_size ** 2``.
        grid_size:
            Side length of the square image grid.  Default 104.
        n_iter:
            Maximum GW iterations.  Default 200 (as in the paper).
        seed:
            Random seed used when sub-sampling cells.  Default 42.
        chunk_size:
            Rows processed per pass during HVG mean/variance computation.
        loess_frac:
            LOESS smoothing fraction for HVG selection.
        n_cells_sample:
            If given, subsample this many cells (drawn with ``seed``) before
            fitting.  The paper uses 100 000 cells stratified by tissue.

        Returns
        -------
        ScImageLayout
        """
        if n_genes is None:
            n_genes = grid_size ** 2

        # Optional cell sub-sampling
        if n_cells_sample is not None and X.shape[0] > n_cells_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=n_cells_sample, replace=False)
            if hasattr(X, "toarray"):
                X_sample = X[idx].toarray()
            else:
                X_sample = np.asarray(X)[idx]
        else:
            X_sample = X

        # Select HVGs
        hvg_indices, gene_mean, gene_std = select_hvg(
            X_sample,
            n_genes=n_genes,
            chunk_size=chunk_size,
            loess_frac=loess_frac,
        )

        # Extract HVG sub-matrix for GW computation
        if hasattr(X_sample, "toarray"):
            X_hvg = X_sample[:, hvg_indices].toarray().astype(np.float32)
        else:
            X_hvg = np.asarray(X_sample, dtype=np.float32)[:, hvg_indices]

        # Compute GW layout
        P = compute_gw_layout(X_hvg, grid_size=grid_size, n_iter=n_iter, seed=seed)

        return cls(P, hvg_indices, gene_mean, gene_std, grid_size=grid_size)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, expression: np.ndarray) -> np.ndarray:
        """Transform cell expression(s) into scImages.

        For each cell the raw expression vector is:

        1. Subset to the *n_genes* HVG columns.
        2. ``log1p``-transformed.
        3. z-scored gene-wise using the stored mean and standard deviation.
        4. Projected via ``y = Pᵀ · ẽ``.
        5. Reshaped into ``(1, H, W)`` in column-major (Fortran) order.

        The output is stored at half precision (``float16``).

        Parameters
        ----------
        expression:
            Array of shape ``(n_all_genes,)`` for a single cell or
            ``(n_cells, n_all_genes)`` for a batch.  The column ordering must
            match the one used during :meth:`fit`.

        Returns
        -------
        images : float16 array
            Shape ``(1, H, W)`` for a single cell or ``(n_cells, 1, H, W)``
            for a batch, where ``H = W = grid_size``.
        """
        expression = np.asarray(expression, dtype=np.float32)
        single_cell = expression.ndim == 1
        if single_cell:
            expression = expression[np.newaxis, :]

        n_cells = expression.shape[0]
        H = W = self.grid_size

        # 1 & 2. Select HVG columns and log1p-transform
        X_hvg = np.log1p(expression[:, self.gene_indices])

        # 3. Z-score gene-wise
        X_z = (X_hvg - self.gene_mean) / self.gene_std  # (n_cells, G)

        # 4. Project: Y = X_z @ P  →  (n_cells, M)
        Y = X_z @ self.projection

        # 5. Reshape each row in column-major (Fortran) order.
        #    y.reshape((H, W), order='F')  ≡  y.reshape((W, H)).T
        images = Y.reshape(n_cells, W, H).transpose(0, 2, 1)  # (n_cells, H, W)
        images = images[:, np.newaxis, :, :]  # (n_cells, 1, H, W)
        images = images.astype(np.float16)

        return images[0] if single_cell else images

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the layout to a compressed ``.npz`` file.

        Parameters
        ----------
        path:
            Destination file path.  The ``.npz`` extension is appended if
            absent.
        """
        np.savez_compressed(
            path,
            projection=self.projection,
            gene_indices=self.gene_indices,
            gene_mean=self.gene_mean,
            gene_std=self.gene_std,
            grid_size=np.array([self.grid_size], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "ScImageLayout":
        """Load a layout previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file.

        Returns
        -------
        ScImageLayout
        """
        data = np.load(path)
        return cls(
            projection=data["projection"],
            gene_indices=data["gene_indices"],
            gene_mean=data["gene_mean"],
            gene_std=data["gene_std"],
            grid_size=int(data["grid_size"][0]),
        )
