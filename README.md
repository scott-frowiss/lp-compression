# Lp-compression

A Python-based compression pipeline for images/videos & 3D videos that also evaluates the
reconstruction quality using PSNR and other metrics.

---

## Contents

- [License](#license)
- [Overview](#overview)
- [Features](#features)
- [Team and Support](#team-and-support)
- [Citation](#citation)
- [Installation](#installation)
- [Tutorial](#tutorial)
    - [0. image\_example.ipynb](docs/tutorial/0_image_example.md)
    - [1. image\_psnr.ipynb](docs/tutorial/1_image_psnr.md)
    - [2. video\_example.ipynb](docs/tutorial/2_video_example.md)
    - [3. video\_psnr.ipynb](docs/tutorial/3_video_psnr.md)
    - [4. image3d\_example.ipynb](docs/tutorial/4_image3d_example.md)
    - [5. image3d\_psnr.ipynb](docs/tutorial/5_image3d_psnr.md)
    - [6. video3d\_example.ipynb](docs/tutorial/6_video3d_example.md)
    - [7. video3d\_psnr.ipynb](docs/tutorial/7_video3d_psnr.md)
    - [8. optimize\_parameters.ipynb](docs/tutorial/8_optimize_parameters.md)
    - [9. viewing\_video\_from\_coeffs.ipynb](docs/tutorial/9_view_video_from_coeffs.md)

---

## License

The project is licensed under the [MIT License](LICENSE.txt).

---

## Overview

This project implements a custom compression pipeline for image, video, and 3D video data.

The method compresses data whilst preserving analytical structure. This allows one to compute
quantities such as gradients and higher-order derivatives directly from the compressed
representation.

Although the current implementation supports images, videos, and 3D videos, the pipeline is
designed to be generalizable to higher-dimensional data. In principle, it may be applicable to
difficult-to-compress scientific datasets such as protein–protein interaction data.

**Note:** This extension is not yet implemented and is planned for a future phase of the project.

---

## Features

- Compression algorithm based on mathematics
- Reconstruction of compressed images, videos, and 3D videos
- Quality metrics: MSE, PSNR, SSIM
- In-notebook visualisation for images and videos
- Visualisation of 3D video data using Napari

---

## Team and Support

We deeply acknowledge:

- [Scott Frowiss](https://github.com/scott-frowiss)
- [Phil-Alexander Hofmann](https://github.com/philippocalippo)
- [Michael Hecht](https://github.com/mikeypice)

and the support and resources provided by the Center for Advanced Systems Understanding
(Helmholtz-Zentrum Dresden-Rossendorf) where the development of this project took place.

---

## Citation

The images we used in this project are from the [TESTIMAGES](https://testimages.org/) dataset.

Images used:
- `img_2400x2400_3x16bit_C00C00_RGB_garden_table.png` — named `garden_table.png` in `sources`

** Related references **

- ASUNI N, GIACHETTI A, "TESTIMAGES: A Large Data Archive For Display and Algorithm Testing", Journal of Graphics Tools, Volume 17, Issue 4, 2015, pages 113-125, DOI:10.1080/2165347X.2015.1024298

- ASUNI N, GIACHETTI A, "TESTIMAGES: a large-scale archive for testing visual devices and basic image processing algorithms", STAG - Smart Tools & Apps for Graphics Conference, 2014

Videos used:
- `drosophila_1slice.tif` — originally available from the [Mastadon cell tracking software tutorial](https://mastodon.readthedocs.io/en/latest/docs/partA/getting_started.html) as a 3D timeseries dataset, adapted to a single slice animation by [Samuel Pantze](https://github.com/smlpt), a colleague at the Center for Advanced Systems Understanding (CASUS) — Helmholtz-Zentrum Dresden-Rossendorf (HZDR)
- `Fluo-N3DL-DRO/01` — test dataset (3D+Time), original image data 1st sequence; provided by the [Cell-Tracking-Challenge](https://celltrackingchallenge.net/) (CTC) website — can download [here](https://data.celltrackingchallenge.net/test-datasets/Fluo-N3DL-DRO.zip)

** Related CTC references **

- Maška, M., Ulman, V., Delgado-Rodriguez, P. et al. The Cell Tracking Challenge: 10 years of objective benchmarking. *Nat Methods* 20, 1010–1020 (2023). https://doi.org/10.1038/s41592-023-01879-y

- Ulman, V., Maška, M., Magnusson, K. et al. An objective comparison of cell-tracking algorithms. *Nat Methods* 14, 1141–1152 (2017). https://doi.org/10.1038/nmeth.4473

For more information on the theory:

- Michael Hecht, Phil-Alexander Hofmann, Damar Wicaksono, Uwe Hernandez Acosta, Krzysztof Gonciarz, Jannik Kissinger, Vladimir Sivkin, Ivo F Sbalzarini, Multivariate Newton interpolation in downward closed spaces reaches the optimal Bernstein–Walsh approximation rate, *IMA Journal of Numerical Analysis*, 2026;, draf137, https://doi.org/10.1093/imanum/draf137

---

## Installation

This project uses Conda for environment setup and management, and requires Jupyter Notebook.
Both `jupyter` and `ipykernel` are installed automatically when creating the environment.
The only additional step is to register the environment as a Jupyter kernel.

1. Clone the repository

        git clone https://github.com/scott-frowiss/lp-compression.git
        cd lp-compression

2. Create and activate the environment

        conda env create --file=environment.yaml
        conda activate lp-compression

3. Register the kernel

        python -m ipykernel install --user --name lp-compression --display-name "Python (lp-compression)"

   After this step, the kernel "Python (lp-compression)" will be available in Jupyter.

4. Launch Jupyter Notebook


        jupyter notebook

   Alternatively:

        jupyter lab


5. Verify the setup (optional, but recommended)

   Open any notebook and make sure the selected kernel is:
   `Python (lp-compression)`

- For help installing Conda, see the official [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

---

## Tutorial

The project consists of core Python (`.py`) scripts, and example (`.ipynb`) notebooks.  The core
scripts are necessary for proper project function, and the example notebooks are where the
actual compression and reconstruction are done.

See the full tutorial index here:
[Tutorial Overview](docs/tutorial/tutorial_index.md)
