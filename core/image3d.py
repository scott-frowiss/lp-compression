import numpy as np
from typing import Literal, Tuple
from lpfun import Transform
from joblib import Parallel, delayed

from core.image import _compress_image_block
from core.utils import split_into_blocks_3d, vander


def compress_image3d(
    image3d: np.ndarray,
    poly_degree: int,
    block_size: int = 10,
    lp_degree: float = 2.0,
    lasso: bool = False,
    p_inv: bool = True,
    cutoff: Literal["mag", "energy", None] = "energy",
) -> Tuple[np.ndarray, np.ndarray, callable]:
    # Scale image3d
    image3d_min, image3d_max = image3d.min(), image3d.max()
    image3d = (image3d - image3d_min) / (image3d_max - image3d_min)
    rescale = lambda x: (image3d_max - image3d_min) * x + image3d_min

    # Split image into blocks
    blocks, nrows, ncols, ntubes = split_into_blocks_3d(image3d, block_size)

    # Construct design matrix
    k = np.arange(block_size)
    x = np.cos(np.pi * k / (block_size - 1))
    y = np.cos(np.pi * k / (block_size - 1))
    z = np.cos(np.pi * k / (block_size - 1))
    X, Y, Z = np.meshgrid(z, y, x)
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    t = Transform(3, poly_degree, lp_degree, basis="chebyshev", report=False)
    design_matrix = vander(poly_degree, coords, t._A, m=3)

    # Fit image
    p_inv_design_matrix = np.linalg.pinv(design_matrix) if p_inv else None
    c = Parallel(n_jobs=-1, verbose=5)(
        delayed(_compress_image_block)(
            block,
            design_matrix,
            p_inv_design_matrix,
            lasso,
            cutoff
        )
        for block in blocks
    )
    c = np.asarray(c).reshape(nrows, ncols, ntubes, len(t))
    return c, design_matrix, rescale


def reconstruct_image3d(
    c: np.ndarray,
    X_design: np.ndarray,
    rescale: callable,
) -> np.ndarray:
    # blocks: (n_depth, n_rows, n_cols, n_block_cubed)
    blocks = np.einsum("kl,ijml->ijmk", X_design, c)

    n_depth, n_rows, n_cols, n_block_cubed = blocks.shape
    n_block = int(round(n_block_cubed ** (1 / 3))) # n_block = block_size

    assert (
        n_block**3 == n_block_cubed
    ), f"Last dim must be a perfect cube, got {n_block_cubed}"

    # reshape to 6D: (n_depth, n_rows, n_cols, n_block, n_block, n_block)
    blocks_6d = blocks.reshape(n_depth, n_rows, n_cols, n_block, n_block, n_block)

    # transpose to interleave grid axes with block-interior axes:
    # (d, i, j, bd, bi, bj) -> (d, bd, i, bi, j, bj)
    blocks_6d = blocks_6d.transpose(0, 3, 1, 4, 2, 5)

    # merge into full volume: (n_depth*n_block, n_rows*n_block, n_cols*n_block)
    full_volume = blocks_6d.reshape(
        n_depth * n_block,
        n_rows * n_block,
        n_cols * n_block,
    )

    full_volume = rescale(full_volume)
    return full_volume
