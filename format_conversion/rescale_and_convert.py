#
# Rescaling and converting to [0, 255] 8bit
#

import numpy as np
import imageio.v3 as iio
import tifffile as tiff

video_raw_16bit = tiff.imread("mouse-brain_1slice.tif")

min_val = video_raw_16bit.min()
max_val = video_raw_16bit.max()

print("Rescaling from [%s, %s] -> [0, 255]" % (min_val, max_val))

frames_float = video_raw_16bit.astype(np.float32)

frames_norm = (frames_float - min_val) / (max_val - min_val)

frames_8bit = (frames_norm * 255.0).round().astype(np.uint8)

file_path = "/Users/scottfrowiss/Documents/deutschland/arbeit/casus_hecht/projekte/\
pseudo-inverse/mouse_video/png_from_frames/8bit_rescaled/"

for i, frame in enumerate(frames_8bit):
    file_name = file_path + f"mouse-brain_8bit_frame_{i:04d}.png"
    iio.imwrite(file_name, frame)
