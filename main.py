import dLux as dl
import matplotlib.pyplot as plt
import dLuxToliman as dlT
from jax import numpy as np
from jax import Array
from detector_layers import ApplyJitter
from jax.scipy.stats import multivariate_normal
import zodiax as zdx
from tqdm.notebook import tqdm

plt.rcParams["image.origin"] = 'lower'


@zdx.filter_jit
def calc_cov(model, parameters):
    return zdx.self_covariance_matrix(model, parameters, zdx.bayes.poiss_loglike)


pscale = 0.375  # arcsec/pixel
det_npix = 128
oversample = 4
wf_npixels = 512
kernel_size = 100
jitter_params = {"r": 10, "phi": 45, "shear": 0.3}

# src = dlT.AlphaCen()
# osys = dlT.TolimanOptics(
#     wf_npixels=512,
#     psf_pixel_scale=pscale,
#     psf_oversample=oversample,
#     psf_npixels=det_npix * oversample,
# )
det = dl.LayeredDetector(
    [
        (ApplyJitter(**jitter_params, kernel_size=kernel_size), "Jitter"),
        (dl.IntegerDownsample(oversample), "Downsample"),
    ]
)

# tel = dl.Instrument(sources=src, optics=osys, detector=det)

# Marginal params
ext = np.array([-1, 1, -1, 1]) * pscale * kernel_size / 2
kernel = det.Jitter.generate_kernel(pscale)
plt.imshow(kernel / np.max(kernel), extent=ext)
plt.show()
