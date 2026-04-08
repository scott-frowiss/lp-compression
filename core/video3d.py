import numpy as np
from typing import Literal, Tuple
from lpfun import Transform
from joblib import Parallel, delayed

from core.image import _compress_image_block
from core.utils import vander, vander_1d, split_into_blocks_3d_plus_t


def compress_video3d(
    video3d: np.ndarray,
    poly_degree: int,
    t_degree: int,
    block_size: int = 10,
    lp_degree: float = 2.0,
    lasso: bool = False,
    cutoff: Literal["mag", "energy", None] = "energy",
) -> Tuple[np.ndarray, np.ndarray, callable]:

    # Scale video3d
    video3d_min, video3d_max = video3d.min(), video3d.max()
    video3d = (video3d - video3d_min) / (video3d_max - video3d_min)
    rescale = lambda x: (video3d_max - video3d_min) * x + video3d_min

    # Split each frame into blocks
    blocks, F, ntubes, nrows, ncols = split_into_blocks_3d_plus_t(video3d, block_size)
    # blocks has shape (F * n_tubes * n_rows * n_cols, block_size, block_size, block_size)

    # Construct design matrix
    k = np.arange(block_size)
    x = np.cos(np.pi * k / (block_size - 1))
    y = np.cos(np.pi * k / (block_size - 1))
    z = np.cos(np.pi * k / (block_size - 1))
    X, Y, Z = np.meshgrid(z, y, x)
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    t = Transform(3, poly_degree, lp_degree, basis="chebyshev", report=False)
    design_matrix = vander(poly_degree, coords, t._A, m=3)
    p_inv_design_matrix = np.linalg.pinv(design_matrix)

    # Fit image
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
    c = np.asarray(c).reshape(F, ntubes, nrows, ncols, len(t))

    #
    # Compress coefficients as they evolve through time - c(t)
    #

    # Construct design matrix
    k_time = np.arange(c.shape[0])
    time = np.cos(np.pi * k_time / (c.shape[0] - 1))
    t_design_matrix = vander_1d(
        time, t_degree
    )  # has shape (n_frames, n_coeffs_per_frame)
    t_p_inv_design_matrix = np.linalg.pinv(t_design_matrix)
    # t_p_inv_design_matrix has shape (n_coeffs_per_frame, n_frames)

    # Reshape c
    c = c.reshape(F, -1)  # has shape (n_frames, n_coeffs_per_frame)

    # Obtain c(t)
    c_t = (
        t_p_inv_design_matrix @ c
    )  # has shape (n_coeffs_per_frame, n_coeffs_per_frame)

    c_t = c_t.reshape(t_degree + 1, ntubes, nrows, ncols, len(t))

    return c_t, design_matrix, t_design_matrix, rescale


def reconstruct_video3d(
    video3d: np.ndarray,
    block_size: int,
    c_t: np.ndarray,
    X_design: np.ndarray,
    t_design_matrix: np.ndarray,
    rescale: callable,
) -> np.ndarray:

    blocks, F, ntubes, nrows, ncols = split_into_blocks_3d_plus_t(video3d, block_size)
    # blocks has shape (F * nt * nr * nc, bs, bs, bs)

    t_basis = t_design_matrix.shape[1]
    X_basis = X_design.shape[1] #TODO: equiv to len(t)?

    c_t_rec = t_design_matrix @ c_t.reshape(t_basis, -1)
    c_t_rec = c_t_rec.reshape(F, ntubes, nrows, ncols, X_basis)
    # c_t_rec.shape should match that of c in reconstruct_image3d, just before np.einsum

    # blocks: (F, n_depth, n_rows, n_cols, n_block_cubed)
    #TODO: isn't it (n_rows, n_cols, n_tubes/n_depth)?
    blocks = np.einsum("kl,ijmnl->ijmnk", X_design, c_t_rec)
    #TODO: I need an extra dimension, correct? (compared to reconstruct_image3d)

    F, n_depth, n_rows, n_cols, n_block_cubed = blocks.shape

    assert (block_size**3 == n_block_cubed
            ), f"Last dim must be a perfect cube, got {n_block_cubed}"

    # reshape to 7D: (F, n_depth, n_rows, n_cols, block_size, block_size, block_size)
    blocks_7d = blocks.reshape(F, n_depth, n_rows, n_cols, block_size, block_size, block_size)

    # transpose to interleave grid axes with block-interior axes:
    # (F, d, i, j, bd, bi, bj) -> (F, d, bd, i, bi, j, bj)
    blocks_7d = blocks_7d.transpose(0, 1, 4, 2, 5, 3, 6)

    # merge time + into full volume:
    # (F, n_depth * block_size, n_rows * block_size, n_cols * block_size)

    full_time_and_volume = blocks_7d.reshape(
            F,
            n_depth * block_size,
            n_rows * block_size,
            n_cols * block_size
            )
    full_time_and_volume = rescale(full_time_and_volume)

    return full_time_and_volume
