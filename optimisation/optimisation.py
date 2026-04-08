#
# Functions used in parameter optimisation of our method (for a specific video)
#

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(".."))

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from core.data import CompressionMetrics
from core.video import reconstruct_video, compress_video

def loss_function(poly_degree:int,
                  block_size:int,
                  cutoff:None,
                  t_degree:int,
                  lp_degree:float, 
                  lambda_rd:float
                  ): 
    """
    poly_degree: int - degree of polynomial
    block_size: int - size of blocks to split video into
    cutoff: str or None - for now, just using None
    t_degree: int - degree of temporal polynomial
    lp_degree: lp degree float parameter
    lambda_rd: rate-distortion tradeoff parameter
    """

    video_raw = iio.imread("../sources/drosophila_1slice.y4m")
    video_raw = video_raw[:, :, :, 0]

    nx, ny = 100, 100
    full_video = video_raw[:, :nx, :ny]
    
    dtype = "float16"

    # Compress video
    c_t, X_design, t_design_matrix, rescale_vid = compress_video(full_video,
                                                                 poly_degree=poly_degree,
                                                                 lp_degree=lp_degree,
                                                                 t_degree=t_degree,
                                                                 block_size=block_size,
                                                                 lasso=False,
                                                                 cutoff=cutoff
                                                                 )
    
    file_path = "../results/video/drosophila_1slice/optimisation/coefficients/"
    np.save(file_path + "coefficients__bs=%s__cutoff=%s__lp=%s__poly_deg=%s__t_degree=%s__\
dtype=%s.npy" % (block_size, cutoff, lp_degree, poly_degree, t_degree, dtype), c_t, \
        allow_pickle=False)

    # Measure rate (file size)
    coefficients_t = np.load(file_path + "coefficients__bs=%s__cutoff=%s__lp=%s__\
poly_deg=%s__t_degree=%s__dtype=%s.npy" % 
                  (block_size, cutoff, lp_degree, poly_degree, t_degree, dtype),\
                          allow_pickle=False)
    file_size_bytes = os.path.getsize(file_path + "coefficients__bs=%s__cutoff=%s__lp=%s__\
poly_deg=%s__t_degree=%s__dtype=%s.npy" % 
                                      (block_size, cutoff, lp_degree, poly_degree, t_degree,\
                                              dtype))
    file_size_kb = file_size_bytes / 1024


    # Reconstruct video
    # video_rec has shape (n_frames, n_rows * block_size, n_cols * block_size)
    video_rec = reconstruct_video(
            full_video,
            block_size, 
            coefficients_t,
            X_design, 
            t_design_matrix,
            rescale_vid
            )

    # Distortion
    metrics_vid = CompressionMetrics(full_video, video_rec, coefficients_t)
    print(full_video)
    mse = metrics_vid.mse
    psnr = metrics_vid.psnr

    # RD objective
    loss = mse + lambda_rd * file_size_kb

    return loss, mse, psnr, file_size_kb


def optimize_lp(poly_degree:int,
                block_size:int,
                cutoff:None,
                t_degree:int,
                lambda_rd:float,
                lp_min:float,
                lp_max:float,
                n_coarse:int=15,
                n_refine:int=10,
                refine_width:float=0.2):
    """
    Returns best_lp, best_loss, best_mse, best_file_size
    """

    # Coarse grid
    lp_degrees = np.linspace(lp_min, lp_max, n_coarse)

    results = []
    for lp in lp_degrees:
        loss, mse, psnr, size = loss_function(poly_degree=poly_degree,
                                        block_size=block_size,
                                        cutoff=cutoff,
                                        t_degree=t_degree,
                                        lp_degree=lp, 
                                        lambda_rd=lambda_rd
                                        )
        results.append((lp, loss, mse, psnr, size))

    best_lp, best_loss, best_mse, best_psnr, best_size = min(results, key=lambda x: x[1])

# Experienced difficulties with the Refinement block below; ignored for now.
#    # Refinement
#    half_width = refine_width * (lp_max - lp_min)
#    lp_lo = max(lp_min, best_lp - half_width)
#    lp_hi = min(lp_max, best_lp + half_width)
#
#    lps_refined = np.linspace(lp_lo, lp_hi, n_refine)
#
#    for lp in lps_refined:
#        loss, mse, size = loss_function(
#                poly_degree=poly_degree,
#                block_size=block_size,
#                cutoff=cutoff,
#                t_degree=t_degree,
#                lp_degree=lp,
#                lambda_rd=lambda_rd
#                )
#        if loss < best_loss:
#            best_lp = lp
#            best_loss = loss
#            best_mse = mse
#            best_size = size

    return best_lp, best_loss, best_mse, best_psnr, best_size
