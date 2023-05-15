"""
Module for functions and classes for developing a model of Alpha Centauri.
"""

# importing
from jax import numpy as np
from jax import random as jr
from jax import vmap
import dLux as dl


class AlphaCenPSF:
    """
    A class to to build TOLIMAN model of Alpha Centauri A and B.

    Attributes
    ----------
        sep : float
            binary separation in arcseconds

        pa : float
            position angle in degrees

        flux : float
            average flux of the two stars in photons per frame

        contrast : float
            flux contrast between two stars

    Methods
    -------

        GetInstrument(mask, wavefront_npixels, detector_npixels, sampling_rate, pixel_scale_out)
            Generates optics and source for Alpha Cen model through TOLIMAN
    """

    # Parameters for Alpha Cen
    sep = 10  # binary separation in arcseconds
    pa = 90  # position angle in degrees
    flux = 6.229e7 / 10 / 2  # AVERAGE flux of the two stars per frame
    contrast = 3.372873  # flux contrast from V band magnitudes

    # TOLIMAN optical parameters
    bandpass_min = 545  # minimum wavelength in nm
    bandpass_max = 645  # maximum wavelength in nm
    aperture = 0.125  # TOLIMAN aperture in m  # TODO: check this

    def __init__(self, mask_dir: str, n_wavels: int = 3):
        # wavelengths in metres
        self.wavels = 1e-9 * np.linspace(self.bandpass_min, self.bandpass_max, n_wavels)
        # loading mask, converting phase to OPD and turning mask into a layer
        self.mask = dl.optics.AddOPD(dl.utils.phase_to_opd(
            np.load(mask_dir), wavelength=self.wavels.mean()
        )
        )

    def GetInstrument(self,
                      wavefront_npixels=256,  # wavefront layer size
                      detector_npixels=128,  # detector size
                      sampling_rate=5,  # pixels per fringe i.e. 5x Nyquist
                      pixel_scale_out=0.3,  # pixel scale in arcseconds
                      ):
        """
        Generate the optics and source for an Alpha Cen model through TOLIMAN.

        Parameters
        ----------

        wavefront_npixels : int, optional
            Wavefront layer size. The default is 256.

        detector_npixels : int, optional
            Detector size. The default is 128.

        sampling_rate : int, optional
            Pixels per fringe. The default is 5 (i.e. 5x Nyquist).

        pixel_scale_out : float, optional
            Output pixel scale in arcseconds. The default is 0.3.

        Returns
        -------
        optics : dLux optical system
            TOLIMAN optical system

        source : dLux source
            Model Alpha Cen source
        """

        # Grabbing the pixel scale required for given sampling rate
        pixel_scale_in = dl.utils.get_pixel_scale(
            sampling_rate,
            self.wavels.max(),
            self.aperture,
            focal_length=None,
        )

        # Make optical system
        optics = dl.utils.toliman(wavefront_npixels,
                                  detector_npixels,
                                  detector_pixel_size=dl.utils.radians_to_arcseconds(pixel_scale_in),
                                  extra_layers=[self.mask],
                                  angular=True,
                                  )

        # Resetting the pixel scale of output
        optics = optics.set(
            ['AngularMFT.pixel_scale_out'],
            [dl.utils.arcseconds_to_radians(pixel_scale_out)]
        )

        # Creating a model Alpha Cen source
        source = dl.BinarySource(separation=dl.utils.arcseconds_to_radians(self.sep),
                                 wavelengths=self.wavels,
                                 contrast=self.contrast,
                                 flux=self.flux,
                                 position_angle=np.deg2rad(self.pa),
                                 )

        return optics, source


def add_noise_to_psf(PSF, seed: int = 0, detector: bool = True, poisson: bool = True):
    """Adding poissonian and detector noise to PSF.

    Parameters
    ----------
        PSF : numpy array
            Ideal PSF of Alpha Cen

        seed : int, optional
            Seed for random number generator. The default is 0.

        detector : bool, optional
            Whether to add detector noise. The default is True.

        poisson : bool, optional
            Whether to add poissonian noise. The default is True.

    Returns
    -------
        PSF : numpy array
            Noisy PSF of Alpha Cen
    """

    key = [jr.PRNGKey(seed), jr.PRNGKey(seed + 1)]

    if poisson:
        PSF = jr.poisson(key[0], PSF)
    if detector:
        # TODO: let user define detector noise level
        det_noise = np.round(2 * jr.normal(key[1], PSF.shape), decimals=0).astype(int)
        PSF += det_noise

    return PSF


def get_jitter_func(optics, source):
    """Generates a function to add jitter to the PSF given an optics and source.

    Parameters
    ----------

    optics : dLux optical system
        dLux optics system

    source : dLux source
        dLux source

    Returns
    -------

    jitter_func : function
        Function to add jitter to PSF
    """

    # Defining a function to set the source position and propagate through the optics
    def set_and_model(optics, source, pos):
        src = source.set(['position'], [pos])
        return optics.model(src)

    vmap_prop = vmap(set_and_model, in_axes=(None, None, 0))
    pixel_scale_out = optics.AngularMFT.pixel_scale_out  # arcseconds per pixel

    def jitter_func(rad: float, angle: float = 0, centre: tuple = (0, 0), npsf: int = 10):
        """
        Returns a jittered PSF by summing a number of shifted PSFs.

        Parameters
        ----------
        rad : float
            The radius of the jitter in pixels.
        angle : float, optional
            The angle of the jitter in degrees, by default 0
        centre : tuple, optional
            The centre of the jitter in pixels, by default (0,0)
        npsf : int, optional
            The number of PSFs to sum, by default 10

        Returns
        -------
        np.ndarray
            The jittered PSF.
        """

        angle = np.deg2rad(angle)  # converting to radius

        # converting to cartesian coordinates
        x_lim = rad / 2 * np.cos(angle)
        y_lim = rad / 2 * np.sin(angle)
        xs = np.linspace(-x_lim, x_lim, npsf)  # pixels
        ys = np.linspace(-y_lim, y_lim, npsf)  # pixels
        positions = pixel_scale_out * (np.stack([xs, ys], axis=1) + np.array(centre))  # arcseconds

        psfs = vmap_prop(optics, source, positions)
        jit_psf = psfs.sum(0) / npsf  # adding and renormalising

        return jit_psf

    return jitter_func

