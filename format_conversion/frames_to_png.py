#
# Saving each frame of mouse/drosophila TIFF stack as PNG
# (For h264 compression)
#

import numpy as np
import tifffile as tiff
import imageio.v3 as iio

video_raw = tiff.imread("mouse-brain_1slice.tif")

file_path = "/Users/scottfrowiss/Documents/deutschland/arbeit/casus_hecht/projekte/\
pseudo-inverse/mouse_video/png_from_frames/"

for i, frame in enumerate(video_raw):
    file_name = file_path + f"mouse-brain_1slice_frame_{i:04d}.png"
    iio.imwrite(file_name, frame)

