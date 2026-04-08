import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    mean_squared_error as mse,
    structural_similarity as ssim,
)

from dataclasses import dataclass, field


@dataclass
class CompressionMetrics:
    image: np.ndarray
    image_rec: np.ndarray
    c: np.ndarray

    # Metrics (computed automatically)
    nz_percent: float = field(init=False)
    mse: float = field(init=False)
    psnr: float = field(init=False)
    compression_ratio: float = field(init=False)
    space_saved: float = field(init=False)
    ssim: float = field(init=False)

    def __post_init__(self):
        nz = np.count_nonzero(self.c)
        self.nz_percent = 100 * nz / self.c.size
        self.mse = mse(self.image, self.image_rec)
        self.psnr = psnr(self.image, self.image_rec, data_range=255)
        self.compression_ratio = 100 * nz / self.image.size
        self.space_saved = 100 - self.compression_ratio
        self.ssim = ssim(self.image, self.image_rec, data_range=255)
