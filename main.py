import dLux as dl
import matplotlib.pyplot as plt
import dLuxToliman as dlT
from jax import numpy as np
from jax import Array
from detector_layers import ApplyAsymmetricJitter
from jax.scipy.stats import multivariate_normal
import zodiax as zdx
from tqdm.notebook import tqdm


@zdx.filter_jit
def calc_cov(model, parameters):
    return zdx.self_covariance_matrix(model, parameters, zdx.bayes.poiss_loglike)


pscale = 0.375  # arcsec/pixel
det_npix = 128
oversample = 4
wf_npixels = 512
kernel_size = 20
jitter_params = {"r": 1, "phi": 60, "shear": 0.7}

src = dlT.AlphaCen()
osys = dlT.TolimanOptics(
    wf_npixels=512,
    psf_pixel_scale=pscale,
    psf_oversample=oversample,
    psf_npixels=det_npix * oversample,
)
det = dl.LayeredDetector(
    [
        (ApplyAsymmetricJitter(**jitter_params), "Jitter"),
        (dl.IntegerDownsample(oversample), "Downsample"),
    ]
)

tel = dl.Instrument(sources=src, optics=osys, detector=det)

# Marginal params
marginal_params = ["detector.Jitter." + param for param in ["r", "shear", "phi"]]

mags = np.linspace(0, 2, 100)
covs = []
for mag in tqdm(mags):
    tel.set('detector.Jitter.r', mag)
    covs.append(calc_cov(tel, marginal_params))
