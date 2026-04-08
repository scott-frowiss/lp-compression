[<- Previous](7_video3d_psnr.md) | [Next ->](9_view_video_from_coeffs.md)

Now let's take a look at optimising a parameter set for a specific video.  This notebook was
written before any of the `image3d`/`video3d` notebooks or `.py` scripts, and is therefore
structured a little differently.  It still uses the `drosophila_1slice.y4m` video, but it's
imported in the `optimisation.py` script, and not in the `optimize_parameters.ipynb`
notebook, as with the other notebooks until now.

It's also worth mentioning the results we observed varried quite dramatically, and sometimes
didn't make any sense at all.  The "optimized" parameter set produced was often for a video
of poor quality, which indicates either our optimisation pipeline needs refinement and/or
troubleshooting, or the metrics used are simply not well-suited to our reconstructed data.

When manually trying to optimize the parameter set for both the garden table image and
drosophila video before the creation of this notebook, we also observed good metrics for
poor quality results, and vice-versa, as mentioned in previous sections.  Although the
optimisation pipeline also needs refinements, our main goal has been developing the project
and expanding into higher-dimensional abstract datasets, therefore has not been a priority.

Since MSE/PSNR didn't seem to be well-suited to our reconstructed data, we added SSIM, which
initial impressions seem to indicate is an improvement.  Expanding the optimisation pipeline
to include SSIM and other metrics is planned for future stages of the project.


```python
import numpy as np

from optimisation import *
from itertools import product
```
```python
# Hyperparameters
poly_degrees = np.arange(2, 5, 1)
cutoff = None
t_degrees = np.arange(15, 29, 3)
lambda_rds = [0.0]
```
```python
block_sizes = []

for i in np.arange(10, 26, 1):
    if 100 % i == 0:
        block_sizes.append(i)

# print(block_sizes)
```
This next block may take a while, depending on the ranges you chose for your parameters.
When using the parameters listed above, however, runtime is not significant.
```python
# Loop through hyperparameters, feed each one into loss_function
results = []

for poly_degree, block_size, t_degree, lambda_rd in product(
    poly_degrees,
    block_sizes,
    t_degrees,
    lambda_rds
):
    print("Testing block_size=%s, poly_degree=%s, t_degree=%s, lambda_rd=%s" % 
          (block_size, poly_degree, t_degree, lambda_rd))
    best_lp, best_loss, best_mse, best_psnr, best_size = optimize_lp(
        poly_degree=poly_degree,
        block_size=block_size,
        cutoff=cutoff,
        t_degree=t_degree,
        lambda_rd=lambda_rd,
        lp_min=1.0,
        lp_max=2.0,
        n_coarse=5,
        n_refine=5,
        refine_width=0.2
    )
    results.append({
        "poly_degree": poly_degree,
        "block_size": block_size,
        "t_degree": t_degree,
        "cutoff": cutoff,
        "lp_degree": best_lp,
        "loss": best_loss,
        "mse": best_mse,
        "psnr": best_psnr,
        "file_size": best_size
    })
```
```python
prescribed_mse = 50
results_filtered = [r for r in results if r["mse"] <= prescribed_mse]
print(min(results, key=lambda r: r["mse"]))
```
```python
best_result = min(results_filtered, key=lambda r: r["file_size"])

print("Best configuration:")
print("===============")
print("poly_degree = %s" % best_result["poly_degree"])
print("block_size = %s" % best_result["block_size"])
print("t_degree = %s" % best_result["t_degree"])
print("lp_degree = %.3f" % best_result["lp_degree"])
print("MSE = %.3f" % best_result["mse"])
print("PSNR = %.3f" % best_result["psnr"])
print("file_size = %.3f KB" % best_result["file_size"])
```
For clarity on `loss_function` and `optimize_lp`, please see `core/optimisation/optimisation.py`

---

[<- Previous](7_video3d_psnr.md) | [Next ->](9_view_video_from_coeffs.md)
