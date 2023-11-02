import os
os.chdir('/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/toliman-phd')
# os.chdir('/Users/mc/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/toliman-phd')

import dLux as dl
import matplotlib.pyplot as plt
import dLuxToliman as dlT
import zodiax as zdx
from jax import numpy as np
import jax
from tqdm.notebook import tqdm

plt.rcParams['image.cmap'] = 'inferno'
# plt.rcParams["font.family"] = "monospace"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Enable 64bit precision (note this must be run in the first cell of the notebook)
jax.config.update("jax_enable_x64", True)

oversample = 4
det_pscale = 0.375
det_npixels = 128

jitter_params = {"A": 0.1, "phi": 0}  # these will vary
radial_orders = [1, ]

src = dlT.AlphaCen()
det = dl.LayeredDetector([
    ('Jitter', dlT.SHMJitter(**jitter_params, kernel_size = 51)),
    ('Downsample', dl.Downsample(oversample)),
])

osys = dlT.TolimanOpticalSystem(
    oversample=oversample,
    psf_pixel_scale=det_pscale,
    psf_npixels=det_npixels,
    radial_orders=radial_orders,
    )
osys = osys.divide('aperture.basis', 1e9) # Set basis units to nanometers

telescope = dl.Telescope(source=src, optics=osys, detector=det)

psf = telescope.model()
plt.imshow(psf)
plt.show()
