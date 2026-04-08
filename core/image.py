import warnings
import numpy as np
from core.utils import (
    vander,
    truncate_by_energy,
    threshold_small_coeffs,
    split_into_blocks_2d,
    compute_alpha,
)
from typing import Literal, Tuple
from lpfun import Transform
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning


def compress_image(
    image: np.ndarray,
    poly_degree: int,
    block_size: int = 20,
    lp_degree: float = 2.0,
    lasso: bool = False,
    p_inv: bool = True,
    cutoff: Literal["mag", "energy", None] = "energy",
) -> Tuple[np.ndarray, np.ndarray, callable]:
    # Scale image
    image_min, image_max = image.min(), image.max()
    image = (image - image_min) / (image_max - image_min)
    rescale = lambda x: (image_max - image_min) * x + image_min

    # Split image into blocks
    blocks, nrows, ncols = split_into_blocks_2d(image, block_size)

    # Construct design matrix
    x = np.linspace(-1, 1, block_size)
    y = np.linspace(-1, 1, block_size)
    X, Y = np.meshgrid(y, x)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    t = Transform(2, poly_degree, lp_degree, basis="chebyshev", report=False)
    design_matrix = vander(poly_degree, coords, t._A)

    # Fit image
    p_inv_design_matrix = np.linalg.pinv(design_matrix) if p_inv else None
    c = Parallel(n_jobs=-1, verbose=5)(
        delayed(_compress_image_block)(
            block,
            design_matrix,
            p_inv_design_matrix,
            lasso,
            cutoff,
        )
        for block in blocks
    )
    c = np.asarray(c).reshape(nrows, ncols, len(t))
    return c, design_matrix, rescale


def _compress_image_block(
    image_block: np.ndarray,
    design_matrix: np.ndarray,
    p_inv_design_matrix: np.ndarray,
    lasso: bool,
    cutoff: Literal["mag", "energy"],
):
    z = image_block.ravel()
    c = np.zeros_like(z)

    # Fit coefficients
    if lasso:
        alpha = compute_alpha(image_block)
        opt = Lasso(alpha, max_iter=10_000, tol=1e-12, random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            opt.fit(design_matrix, z)
        c = opt.coef_
    elif p_inv_design_matrix is None:
        c, *_ = np.linalg.lstsq(design_matrix, z)
    else:
        c = p_inv_design_matrix @ z

    # Apply cutoff
    if cutoff == "energy":
        c = truncate_by_energy(c, p=0.995)
    if cutoff == "mag":
        c = threshold_small_coeffs(c)

    return c


def reconstruct_image(
    c: np.ndarray,
    X_design: np.ndarray,
    rescale: callable,
) -> np.ndarray:
    blocks = np.einsum("kl,ijl->ijk", X_design, c)
    n_rows, n_cols, n_block_squared = blocks.shape
    n_block = int(np.sqrt(n_block_squared))

    # reshape to 4D: (n_rows, n_cols, n_block, n_block)
    blocks_4d = blocks.reshape(n_rows, n_cols, n_block, n_block)

    # Swap axes to bring blocks together
    blocks_4d = blocks_4d.transpose(
        0, 2, 1, 3
    )  # shape: (n_rows, n_block, n_cols, n_block)

    # Merge blocks
    full_image = blocks_4d.reshape(n_rows * n_block, n_cols * n_block)

    # Rescale image
    full_image = rescale(full_image)
    return full_image
