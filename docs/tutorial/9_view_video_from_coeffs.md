[Back to Home](../../README.md)

[<- Previous](8_optimize_parameters.md)

This is a short notebook created just to view the reconstructed video from the coefficients
produced during compression.  The main purpose of this notebook was to get an idea of what it
would take to export/import the compressed results, and which file format would be best suited
to this project.

Currently, the project is only saving the coefficients generated during compression in a `.npy`
file, as saving the reconstructed video requires too much space.  Later, however, our goal is
to export the parameters necessary to recreate the objects required for reconstruction.

Hierarchical Data Format version 5 (HDF5; `.h5`, `.hdf5`) appears to be a good candidate for a
future file format for the project, and will be explored more at a later date.

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
from lpfun import Transform

from core.utils import vander, vander_1d, make_dual_update, split_frames_into_blocks
from core.video import reconstruct_video
```
```python
video_raw = iio.imread("../sources/drosophila_1slice.y4m")
video_raw = video_raw[:, :, :, 0]
nx, ny = 100, 100
video = video_raw[:, :nx, :ny]
```
```python
# Hyperparameters
lp_degree = 1.0
block_size = 10
cutoff = None
poly_degree = 5
t_degree = 20
dtype = "float16"
```
```python
video_min, video_max = video.min(), video.max()
video = (video - video_min) / (video_max - video_min)
rescale = lambda x: (video_max - video_min) * x + video_min
```
```python
# Recreating blocks object
blocks, nrows, ncols = split_frames_into_blocks(video, block_size)

# blocks has shape (F, n_rows, n_cols, block_size, block_size)
print(blocks.shape)

# Recreate X_design
k = np.arange(block_size)
x = np.cos(np.pi * k / (block_size - 1))
y = np.cos(np.pi * k / (block_size - 1))
X, Y = np.meshgrid(y, x)
coords = np.column_stack([X.ravel(), Y.ravel()])
t = Transform(2, poly_degree, lp_degree, basis="chebyshev", report=False)
X_design = vander(poly_degree, coords, t._A)

# Recreate t_design_matrix
k_time = np.arange(blocks.shape[0])
time = np.cos(np.pi * k_time / (blocks.shape[0] - 1))
t_design_matrix = vander_1d(time, t_degree)  # has shape (n_frames, n_coeffs_per_frame)
```
```python
# Need to update path to the directory where you're saving the coefficients `.npy` file to.
# Load naming scheme should match Save naming scheme - shouldn't need to change anything
c_t = np.load(
    "../results/video/drosophila_1slice/coefficients/coefficients__bs=%s__cutoff=%s__lp=%s__poly_deg=%s__t_degree=%s__dtype=%s.npy"
    % (block_size, cutoff, lp_degree, poly_degree, t_degree, dtype),
    allow_pickle=False,
)
```
```python
# video_rec has shape (n_frames, n_rows * block_size, n_cols * block_size)
video_rec_load = reconstruct_video(
    video, block_size, c_t, X_design, t_design_matrix, rescale
)
```
```python
# To view video: calling make_update function defined in utils.py - next cell plays the video
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

im_orig = ax[0].imshow(video[0], cmap="gray")
ax[0].set_title("Original")

im_rec = ax[1].imshow(video_rec_load[0], cmap="gray")
ax[1].set_title("Reconstructed")

update = make_dual_update(video, video_rec_load, im_orig, im_rec)
ani = animation.FuncAnimation(
    fig, update, frames=len(video), interval=200, blit=True
)
```
```python
from IPython.display import HTML
HTML(ani.to_jshtml())
```
The video is not actually displayed here, as examples have already been provided.
**Note:** Viewing this way requires Jupyter Notebook.  If you're not using Jupyter Notebook,
Napari may be a useful tool.

---
[Back to Home](../../README.md)

[<- Previous](8_optimize_parameters.md)
