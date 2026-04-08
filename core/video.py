import numpy as np
from typing import Literal, Tuple
from lpfun import Transform
from joblib import Parallel, delayed

from core.image import _compress_image_block
from core.utils import vander, vander_1d, split_frames_into_blocks


def compress_video(
    video: np.ndarray,
    poly_degree: int,
    t_degree: int,
    block_size: int = 20,
    lp_degree: float = 2.0,
    lasso: bool = False,
    cutoff: Literal["mag", "energy", None] = "energy",
) -> Tuple[np.ndarray, np.ndarray, callable]:

    # Scale video
    video_min, video_max = video.min(), video.max()
    video = (video - video_min) / (video_max - video_min)
    rescale = lambda x: (video_max - video_min) * x + video_min

    # Split each frame into blocks
    blocks, nrows, ncols = split_frames_into_blocks(video, block_size)
    # blocks has shape (F, n_rows, n_cols, block_size, block_size)

    F, *_ = blocks.shape

    # Reshaping 5D blocks tensor into a list of 2d blocks
    # shape of vec_blocks: (n_frames * n_rows * n_cols, block_size, block_size)
    vec_blocks = blocks.reshape(-1, block_size, block_size)

    # Construct design matrix
    k = np.arange(block_size)
    x = np.cos(np.pi * k / (block_size - 1))
    y = np.cos(np.pi * k / (block_size - 1))
    X, Y = np.meshgrid(y, x)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    t = Transform(2, poly_degree, lp_degree, basis="chebyshev", report=False)
    design_matrix = vander(poly_degree, coords, t._A)
    p_inv_design_matrix = np.linalg.pinv(design_matrix)

    # Fit image
    c = Parallel(n_jobs=-1, verbose=5)(
        delayed(_compress_image_block)(
            block, design_matrix, p_inv_design_matrix, lasso, cutoff
        )
        for block in vec_blocks
    )
    c = np.asarray(c).reshape(F, nrows, ncols, len(t))

    #
    # Compress each frame
    #

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

    c_t = c_t.reshape(t_degree + 1, nrows, ncols, len(t))

    return c_t, design_matrix, t_design_matrix, rescale


def reconstruct_video(
    video: np.ndarray,
    block_size: int,
    c_t: np.ndarray,
    X_design: np.ndarray,
    t_design_matrix: np.ndarray,
    rescale: callable,
) -> np.ndarray:
    #    blocks = np.einsum("kl,ijl->ijk", X_design, c)
    #    n_rows, n_cols, n_block_squared = blocks.shape
    #    n_block = int(np.sqrt(n_block_squared))
    #
    #    # reshape to 5D: (n_rows, n_cols, n_block, n_block)
    #    blocks_5d = blocks.reshape(n_rows, n_cols, n_block, n_block)
    #
    #    # Swap axes to bring blocks together
    #    blocks_5d = blocks_5d.transpose(
    #        0, 2, 1, 3
    #    )  # shape: (n_rows, n_block, n_cols, n_block)
    #
    #    # Merge blocks
    #    full_video = blocks_5d.reshape(n_rows * n_block, n_cols * n_block)
    #
    #    # Rescale video
    #    full_video = rescale(full_video)

    # blocks has shape (F, nrows, ncols, block_size, block_size)
    blocks, nrows, ncols = split_frames_into_blocks(video, block_size)
    F, *_ = blocks.shape
    t_basis = t_design_matrix.shape[1]
    X_basis = X_design.shape[1]

    c_t_rec = t_design_matrix @ c_t.reshape(t_basis, -1)
    c_t_rec = c_t_rec.reshape(F, nrows, ncols, X_basis).swapaxes(0, -1)
    video_rec = X_design @ c_t_rec.reshape(X_basis, -1)

    full_video = video_rec.reshape(block_size * block_size, nrows, ncols, F)
    full_video = full_video.swapaxes(0, -1)
    full_video = full_video.reshape(F, nrows, ncols, block_size, block_size)
    full_video = full_video.swapaxes(2, 3)

    full_video = rescale(full_video)
    full_video = full_video.reshape(F, nrows * block_size, ncols * block_size)

    return full_video
