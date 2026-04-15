[Back to Home](../../README.md)

[<- Previous](5_image3d_psnr.md) | [Next ->](7_video3d_psnr.md)

Now we use the full 3D video (3D + time) from [The Cell-Tracking-Challenge](https://celltrackingchallenge.net/).

Just for clarity, you can download the video directly from their website by going to Datasets ->
3D+Time Datasets, and searching for "drosophila".

We're only concerned with the original dataset, not the reference annotations, therefore we
recommend to download the test dataset.

For a deeper understanding of the naming convention of the files within the ZIP archive, please see [Naming and file content convension.pdf](https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf), which can be found on their website under Datasets -> Dataset Description.

**Note:** This is a large file — the ZIP archive alone is 5.9 GB.  Within the ZIP archive are 4
folders — `01`, `01_GT`, `02`, `02_GT`.  The files named simply `01` and `02` are the files we're
interested in, so we can ignore the files labeled with `_GT`.  From their naming convention PDF,
we can see that `01` and `02` correspond to "sequence 1" and "sequence 2", respectively.

Both `01` and `02` contain 50 TIFF files — each of these TIFFs is one timestamp.  In the
following example, we only used `01` (sequence 1).  The file size of each TIFF ranges between
68.1 and 68.8 MB.

```python
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(".."))
```
```python
import numpy as np
import tifffile as tiff
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.data import CompressionMetrics
from core.utils import make_dual_update, round_sig
from core.video3d import reconstruct_video3d, compress_video3d
```
```python
# Importing the video takes a while, so I put the import statement in its own cell.
# This way, we can make changes in the notebook without needing to import the video again.
video3d_raw = tiff.imread("../sources/Fluo-N3DL-DRO/01/*.tif")
```
The first time running this notebook may take a while, mostly due to the size of the video, and
constructing the design matrices.  Since the video is loaded in its own cell above, we can alter
the hyperparameters and/or the notebook without needing to load it again.  This may seem trivial,
however ensuring we only load the video one time significantly reduces runtime.

After running the notebook, avoid restarting the kernel, as having the cached version also
significantly reduces runtime.  Again, this may seem trivial, however runtimes may increase
exponentially, depending on the changes you make.

**Note:** In the cell below, we are only using blocks of size 100x100x100.  This is significantly
smaller than the original video, and was done to avoid longer runtimes in the initial phase of
the project.  Once the video has been loaded, and the notebook cached, you may adjust how much
of the video is included in `video3d`.

There are also other videos from [the Cell-Tracking-Challenge](https://celltrackingchallenge.net/) if you wish to experiment with this project some more.

```python
## Drodophila full 3D video ##
# True dimensions are (50, 125, 603, 1272)
nz, ny, nx = 100, 100, 100
video3d = video3d_raw[:, :100, 200:300, 400:500]
```
**Remember**: `lpfun` expects cubes, so when choosing how much of the video to include in
`video3d`, be sure the spatial volume is symmetric.

For example, `video3d = video3d_raw[:, :100, 200:300, 400:500]` corresponds to a spatial volume
of 100x100x100.

The spatial volume must also be divisible by `block_size`.
```python
# Hyperparameters
poly_degree = 6
t_degree = 35
lp_degree = 1.0
block_size = 10
cutoff = None
```
```python
c_t, X_design, t_design_matrix, rescale = compress_video3d(
    video3d,
    poly_degree=poly_degree,
    t_degree=t_degree,
    block_size=block_size,
    lp_degree=lp_degree,
    cutoff=cutoff
)
```
```python
c_t = round_sig(c_t)
video3d_rec = reconstruct_video3d(
    video3d,
    block_size,
    c_t,
    X_design,
    t_design_matrix,
    rescale
)
```
```python
# Saving coefficients
np.save("../results/video3d/drosophila_video_full_01/coefficients/coefficients__shape=%s__bs=%s__cutoff=%s__lp=%s__poly_deg=%s__t_degree=%s__dtype=%s.npy" %
(vid3d.shape, block_size, cutoff, lp_degree, poly_degree, t_degree, c_t.dtype), c_t, allow_pickle=False)
```
We can no longer view the video as we previously did in Jupyter Notebook, since we're using a 4D
dataset (3D + time).  Instead, we use Napari.  After running the cell below, a Napari window
should pop up that allows you to view the video.  Viewing the original and reconstructed videos
side-by-side is no longer an option (so far as we know), but we can switch between them on the
left side of the window.

At first, you'll notice two play bars at the bottom.  By default, the video is displayed in 2D,
however our video is 3D.  In the bottom-left there's a button to toggle 2D/3D view.  Clicking
that should display the video in 3D, removing one of the play bars and leaving us with one.
This makes viewing the videos more intuitive, but comes at a computational cost.  The larger
the video, the slower our playback capabilities become.

```python
import napari

viewer = napari.Viewer()
viewer.add_image(video3d)
viewer.add_image(video3d_rec)
napari.run()
```
<video src="https://github.com/user-attachments/assets/8b2b0c67-56c3-45cd-96d4-a7ca990c86ba" controls width="400"></video>

Here we use only the small 100x100x100 volume of the video for simplicity.
```python
metrics = CompressionMetrics(video3d, video3d_rec, c_t) 
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

**Increasing the size of the video**

After entering
```python
video3d = video3d_raw[:, :100, 200:600, 400:800]
```
and running the rest of the cells, we observed the compression and reconstruction runtimes to
be surprisingly quick.  Displaying the video using Napari was also faster than expected, but
even after the kernel finished processing the cell, we needed to wait a little extra for the
window to pop up.

Once the window appeared, playing the video was an obvious struggle.  After manually cycling
through the frames (forward and backward), and letting the video play at its own
(very slow) speed a few times, we were able to view the video (original and reconstruction)
with relative fluidity.

<video src="https://github.com/user-attachments/assets/41dc262a-a127-4541-b26b-b7164338ddec" controls width="400"></video>

We're getting closer to seeing the full 3D drosophila video, but as you can see from the
difficulties we've had simply viewing this smaller portion, it's becomming quite computationally
demanding.

As the size of the video got larger, we found that running each cell individually, and waiting
for the kernel to process a cell before moving on to the next resulted in better runtimes, with
the exception of the metrics.  After many attempts, the notebook timed out and restarted the
kernel before being able to compute them, even as we approched only about 15% of the video.

Making the project overall more efficient is one of our goals, as we hope to apply this
method to other multidimentional datasets — not just images/videos and their 3D counterparts.

---
[Back to Home](../../README.md)

[<- Previous](5_image3d_psnr.md) | [Next ->](7_video3d_psnr.md)
