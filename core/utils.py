import numpy as np
from numba import njit, prange
from scipy.ndimage import laplace


def round_sig(x, sig=2):
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=np.float64)

    nonzero = x != 0
    ax = np.abs(x[nonzero]).astype(np.float64)

    # scale so that rounding happens at the (sig)th significant digit
    exp = np.floor(np.log10(ax))
    scale = np.power(10.0, sig - 1 - exp)

    out[nonzero] = np.round(x[nonzero].astype(np.float16) * scale) / scale
    return out.astype(np.float16, copy=False)


@njit(parallel=True)
def vander(n: int, points: np.ndarray, A: np.ndarray, m: int = 2) -> np.ndarray:
    len_points = len(points)
    num_terms = len(A)
    mat = np.empty((len_points, num_terms), dtype=np.float64)
    for l in prange(len_points):
        x = points[l]
        basis = np.empty((m, n + 1), dtype=np.float64)
        basis[:, 0] = 1.0
        if n >= 1:
            basis[:, 1] = x
        for j in range(1, n):
            basis[:, j + 1] = 2 * x * basis[:, j] - basis[:, j - 1]
        for i in range(num_terms):
            prod = 1.0
            for j in range(m):
                prod *= basis[j, A[i, j]]
            mat[l, i] = prod

    return mat


@njit(parallel=True)
def vander_1d(x, deg):
    # x = np.asarray(x)
    # theta = np.arccos(np.clip(x, -1.0, 1.0))
    # j = np.arange(deg + 1)
    # Tx = np.cos(np.outer(theta, j))
    # return Tx
    N = x.size
    Tx = np.zeros((N, deg + 1), dtype=np.float64)

    if deg >= 0:
        Tx[:, 0] = 1.0
    if deg >= 1:
        Tx[:, 1] = x

    for n in range(1, deg):
        Tx[:, n + 1] = 2.0 * x * Tx[:, n] - Tx[:, n - 1]

    return Tx


def make_update(video: np.ndarray, im):  # im: : matplotlib.image.AxesImage
    def update(frame):
        im.set_array(video[frame])
        return [im]

    return update


def make_dual_update(movie1, movie2, im1, im2):
    def update(frame):
        im1.set_array(movie1[frame])
        im2.set_array(movie2[frame])
        return [im1, im2]

    return update


def split_frames_into_blocks(video, block_size):
    F, H, W = video.shape
    n_rows = H // block_size
    n_cols = W // block_size

    blocks = video.reshape(F, n_rows, block_size, n_cols, block_size).swapaxes(2, 3)

    return blocks, n_rows, n_cols


def split_into_blocks_2d(image: np.ndarray, block_size: int):
    H, W = image.shape

    if H % block_size != 0 or W % block_size != 0:
        raise ValueError("Image dimensions must be divisible by block_size.")

    n_rows = H // block_size
    n_cols = W // block_size

    blocks = (
        image.reshape(n_rows, block_size, n_cols, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size, block_size)
    )

    return blocks, n_rows, n_cols


def split_into_blocks_3d(image, block_size):
    D, H, W = image.shape

    if D % block_size != 0 or H % block_size != 0 or W % block_size != 0:
        raise ValueError("All dimensions must be divisible by block_size.")

    n_tubes = D // block_size
    n_rows = H // block_size
    n_cols = W // block_size

    blocks = (
        image.reshape(n_tubes, block_size, n_rows, block_size, n_cols, block_size)
        .transpose(0, 2, 4, 1, 3, 5)  # (nt, nr, nc, bs, bs, bs)
        .reshape(-1, block_size, block_size, block_size)
    )

    return blocks, n_rows, n_cols, n_tubes


def split_into_blocks_3d_plus_t(video, block_size):
    F, D, H, W = video.shape 

    if D % block_size != 0 or H % block_size != 0 or W % block_size != 0:
        raise ValueError("All dimensions must be divisible by block_size.")

    n_tubes = D // block_size
    n_rows = H // block_size
    n_cols = W // block_size

    blocks = (
        video.reshape(F, n_tubes, block_size, n_rows, block_size, n_cols, block_size)
        .transpose(0, 1, 3, 5, 2, 4, 6)  # (F, nt, nr, nc, bs, bs, bs)
        .reshape(-1, block_size, block_size, block_size)
    )
    # blocks has shape (F * n_tubes * n_rows * n_cols, block_size, block_size, block_size)

    return blocks, F, n_tubes, n_rows, n_cols



def truncate_by_energy(c: np.ndarray, p: float = 0.995, eps=1e-6) -> np.ndarray:
    energy = np.cumsum(np.sort(c**2)[::-1])
    if energy[-1] < eps:
        return np.zeros_like(c)
    energy /= energy[-1]
    idxs = np.where(energy <= p)[0]
    if len(idxs) == 0:
        return c
    idx = idxs[-1] + 1
    threshold = np.sort(np.abs(c))[::-1][idx - 1]
    c_trunc = c.copy()
    c_trunc[np.abs(c) < threshold] = 0.0

    return c_trunc


def threshold_small_coeffs(c: np.ndarray, rel: float = 0.001) -> np.ndarray:
    c_thr = c.copy()
    c_thr[np.abs(c_thr) < rel * np.max(np.abs(c_thr))] = 0.0

    return c_thr


def convert_to_gray_scale(image_rgb: np.ndarray) -> np.ndarray:
    # Normalization
    image_rgb = image_rgb / 255.0

    # Getting color channels
    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    # Convert to gray scale by applying the SMPTE 295M-1997 standard to coefficients
    # to the RGB intensities (as recommended by author of TESTIMAGES)
    return (0.2126 * r) + (0.7152 * g) + (0.0722 * b)


# NOTE: work in progress...
def compute_alpha(y: np.ndarray, min=1e-10, max=1.0, exponent=0.25) -> float:
    # normalized max Laplacian
    x = np.clip(np.max(np.abs(laplace(y))), 0, 1)
    # exponentiate to increase sensitivity for small inputs
    x = x**exponent
    # map to log10 scale
    return 10 ** (np.log10(min) + x * (np.log10(max) - np.log10(min)))
