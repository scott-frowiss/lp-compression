[<- Previous](3_video_psnr.md) | [Next ->](5_image3d_psnr.md)

Now things start to change a bit.  The name `image3d` can be a bit misleading, because we're
still working with the `drosophila_1slice.y4m` video — this is the video we converted from
TIFF -> Y4M

We're still compressing a video, but we're treating it as an "image" — just in
3D.  Now, time is treated as a spatial dimension. This framework could potentially work on
abstract 3D datasets, but has not yet been tested.

```python
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(".."))
```
```python
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.data import CompressionMetrics
from core.utils import make_dual_update, round_sig
from core.image3d import reconstruct_image3d, compress_image3d
```
```python
image3d_raw = iio.imread("../sources/drosophila_1slice.y4m")
image3d_raw = image3d_raw[:, :, :, 0]
nt, nx, ny = 30, 30, 30
image3d = image3d_raw[:nt, :nx, :ny]
# plt.imshow(image3d[0], cmap="gray")
# plt.show()
```
You may be wondering why we've cropped the video to only 30x30x30.  This is because lpfun
expects squares, cubes, etc., and our video only has 31 frames.  We're also splitting up the
video into 3D blocks, so the dimensions of the video must be divisible by `block_size`.

In `video_example.ipynb`, however, we were able to use spatial dimensions of 100x100, along with
all 31 timestamps.  Here, lpfun was used to construct the spatial design matrix (`X_design`),
but not to construct the temporal design matrix (`t_design_matrix`)

This means our spatial design matrix (`X_design`) expects squares, but our temporal design
matrix (`t_design_matrix`) is not limited to the same constraint.

```python
# Hyperparameters
poly_degree = 6
lp_degree = 1.0
block_size = 10
cutoff = None
```
```python
c, X_design, rescale = compress_image3d(
    image3d, poly_degree, block_size, lp_degree, p_inv=True, cutoff=cutoff
)
```
```python
c = round_sig(c)
image3d_rec = reconstruct_image3d(c, X_design, rescale)
```
```python
# Saving coefficients
np.save("../results/image3d/drosophila_1slice/coefficients__bs=%s__cutoff=%s__lp=%s__poly_deg=%s__dtype=%s.npy" %
(block_size, cutoff, lp_degree, poly_degree, c.dtype), c, allow_pickle=False)
```
```python
fig, ax = plt.subplots(1, 2, figsize=(8,4))

im_orig = ax[0].imshow(image3d[0], cmap='gray')
ax[0].set_title("Original")

im_rec = ax[1].imshow(image3d_rec[0], cmap='gray')
ax[1].set_title("Reconstructed")

update = make_dual_update(image3d, image3d_rec, im_orig, im_rec)
ani = animation.FuncAnimation(fig, update, frames=len(image3d), interval=200, blit=True)
```
```python
from IPython.display import HTML
HTML(ani.to_jshtml())
```
![drosophila\_image3d\_example](../../reconstructions/drosophila_image3d_example.mp4)
**Remember:** This method to view the video only works in Jupyter Notebook.  If you're not using
Jupyter Notebook, Napari may be a useful tool.

```python
metrics = CompressionMetrics(image3d, image3d_rec, c) 
```
```python
import pandas as pd
metrics_dict = {
    "Metric": ["Nonzero Coefficients (%)", "MSE", "PSNR (dB)", "Compression Ratio (%)", "Space Saved (%)", "SSIM"],
    "Value": [metrics.nz_percent, metrics.mse, metrics.psnr, metrics.compression_ratio, metrics.space_saved, metrics.ssim]
}
df = pd.DataFrame(metrics_dict)
print(df)
```

---

[<- Previous](3_video_psnr.md) | [Next ->](5_image3d_psnr.md)
