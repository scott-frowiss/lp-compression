[Back to Home](../../README.md)

[<- Previous](1_image_psnr.md) | [Next ->](3_video_psnr.md)

It may seem a little strange to cover `video_example.ipynb` before `image3d_example.ipynb`,
however this is the natural next step in the project.  This will become clear later on.

The video used is one slice of a 3D video.  Originally, it was one TIFF file, but trying to
create a benchmark and compare our compressed results to those produced by H.264 required us
to convert from TIFF to Y4M.  To do this, we used the Python scripts in `format_conversion`
section of the project.  We first needed to convert each frame of the TIFF to a PNG, then
rescaled and converted to Y4M.

The video used can be found in `sources`, and the original 3D video came from the
[Cell-Tracking-Challenge](https://celltrackingchallenge.net/). They also have standard videos
(2D + t), if you'd like to use one of those.

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
from core.video import reconstruct_video, compress_video
```
```python
video_raw = iio.imread("../sources/drosophila_1slice.y4m")
video_raw = video_raw[:, :, :, 0]
nx, ny = 100, 100
video = video_raw[:, :nx, :ny]

# To view first frame
plt.imshow(video[0], cmap="gray")
plt.show()
```
```python
# Hyperparameters
lp_degree = 1.0
block_size = 10
cutoff = None
poly_degree = 5
t_degree = 20
```
```python
# Acquire coefficients, both design matrices, and the rescale (callable) function
c_t, X_design, t_design_matrix, rescale = compress_video(
    video, poly_degree, t_degree, block_size, lp_degree, lasso=False, cutoff=cutoff
)
```

In this example, we have an additional hyperparameter `t_degree`.  This corresponds to another
polynomial interpolation accross all coefficients through time, whereas previously we only
performed a polynomial interpolation over a 2D grid.

Essentially, we compress each frame (still splitting them up into 2D blocks), then compress
the coefficients once more through time, as they change from frame to frame.  This additional
interpolation step is only 1D, and therefore does not utilize lpfun (lp degree).

```python
c_t = round_sig(c_t)
# c_t = c_t.astype(np.float16)
video_rec = reconstruct_video(
    video, block_size, c_t, X_design, t_design_matrix, rescale
)
```
```python
# Saving coefficients
np.save("../results/video/drosophila_video_full_01/coefficients/coefficients__bs=%s__cutoff=%s__lp=%s__poly_deg=%s__t_degree=%s__dtype=%s.npy" %
(block_size, cutoff, lp_degree, poly_degree, t_degree, c_t.dtype), c_t, allow_pickle=False)
```
```python
# To view video: calling make_update function defined in utils.py - next cell plays the video
fig, ax = plt.subplots(1, 2, figsize=(8,4))

im_orig = ax[0].imshow(video[0], cmap='gray')
ax[0].set_title("Original")

im_rec = ax[1].imshow(video_rec[0], cmap='gray')
ax[1].set_title("Reconstructed")

update = make_dual_update(video, video_rec, im_orig, im_rec)
ani = animation.FuncAnimation(fig, update, frames=len(video), interval=200, blit=True)
```
```python
from IPython.display import HTML
HTML(ani.to_jshtml())
```
![drosophila\_1slice\_example](../../reconstructions/drosophila_1slice_example.mp4)
**Note:** The above method to view the video only works in Jupyter Notebook.  To view the video
without Jupyter Notebook, Napari may be a good option.

```python
metrics_vid = CompressionMetrics(video, video_rec, c_t) 
```
```python
import pandas as pd
idx_met = metrics_vid
metrics_vid_dict = {
    "Metric": ["Nonzero Coefficients (%)", "MSE", "PSNR (dB)", "Compression Ratio (%)", "Space Saved (%)", "SSIM"],
    "Value": [idx_met.nz_percent, idx_met.mse, idx_met.psnr, idx_met.compression_ratio, idx_met.space_saved, idx_met.ssim]
}
df = pd.DataFrame(metrics_vid_dict)
print(df)
```

---
[Back to Home](../../README.md)

[<- Previous](1_image_psnr.md) | [Next ->](3_video_psnr.md)
