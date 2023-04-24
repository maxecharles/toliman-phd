"""
Module for functions and classes for developing a model of Alpha Centauri.
"""


# importing
from jax import numpy as np
from jax import random as jr
from numpy import random as rd
import dLux as dl
from PIL import Image, ImageDraw
from scipy.signal import convolve2d


class Psf:

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
        LoadMask(mask_dir)
            Loads mask from directory and converts phase to OPD

        GetInstrument(mask, wavefront_npixels, detector_npixels, sampling_rate, pixel_scale_out)
            Generates optics and source for Alpha Cen model through TOLIMAN

        AddNoise(PSF)
    """

    # Parameters for Alpha Cen
    sep = 10  # binary separation in arcseconds
    pa = 90  # position angle in degrees
    flux = 6.229e7 / 10 / 2  # AVERAGE flux of the two stars per frame
    contrast = 3.372873  # flux contrast from V band magnitudes

    # TOLIMAN optical parameters
    bandpass_min = 545  # minimum wavelength in nm
    bandpass_max = 645  # maximum wavelength in nm
    aperture = 0.125  # TOLIMAN aperture in m  #TODO: check this

    def __init__(self, seed: int = 0, n_wavels: int = 3):
        self.key = [jr.PRNGKey(seed), jr.PRNGKey(seed+1)]
        self.wavels = 1e-9 * np.linspace(self.bandpass_min, self.bandpass_max, n_wavels)  # wavelengths in m
        return

    # loading mask, converting phase to OPD and turning mask into a layer
    def LoadMask(self, mask_dir):
        mask = dl.optics.AddOPD(dl.utils.phase_to_opd(np.load(mask_dir), wavelength=self.wavels.mean()))
        return mask

    def GetInstrument(self,
                      mask,
                      wavefront_npixels=256,  # wavefront layer size
                      detector_npixels=128,  # detector size
                      sampling_rate=5,  # pixels per fringe i.e. 5x Nyquist
                      pixel_scale_out=0.3,  # pixel scale in arcseconds
                      ):
        """
        Generate the optics and source for an Alpha Cen model through TOLIMAN.

        Parameters
        ----------
        mask : numpy array
            Binary Phase Mask of TOLIMAN

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
        pixel_scale_in = dl.utils.get_pixel_scale(sampling_rate, self.wavels.max(), self.aperture, focal_length=None)

        # Make optical system
        optics = dl.utils.toliman(wavefront_npixels,
                                  detector_npixels,
                                  detector_pixel_size=dl.utils.radians_to_arcseconds(pixel_scale_in),
                                  extra_layers=[mask],
                                  angular=True,
                                  )

        # Resetting the pixel scale of output
        optics = optics.set(['AngularMFT.pixel_scale_out'], [dl.utils.arcseconds_to_radians(pixel_scale_out)])

        # Creating a model Alpha Cen source
        source = dl.BinarySource(separation=dl.utils.arcseconds_to_radians(self.sep),
                                 wavelengths=self.wavels,
                                 contrast=self.contrast,
                                 flux=self.flux,
                                 position_angle=np.deg2rad(self.pa),
                                 )

        return optics, source

    def AddNoise(self, PSF):
        """Adding poissonian and detector noise to PSF.

        Parameters
        ----------
            PSF : numpy array
                Ideal PSF of Alpha Cen

        Returns
        -------
            noisy_PSF : numpy array
                Noisy PSF of Alpha Cen
        """

        noisy_PSF = jr.poisson(self.key[0], PSF)
        det_noise = np.round(2 * jr.normal(self.key[1], noisy_PSF.shape), decimals=0).astype(int)
        noisy_PSF += det_noise

        return noisy_PSF


def linear_jitter(PSF, radius=12, theta=None, im_size=25):
    """Convolving PSF with a line to simulate linear jitter.

    Parameters
    -------
        PSF : numpy array
            PSF of Alpha Cen

        radius : int, optional
            Radius of jitter line in pixels. The default is 12.

        theta : float, optional
            Angle of jitter line in radians. If None, a random direction is generated.

        im_size : int, optional
            Size of convolution kernel in pixels. The default is 25.

    Returns
    -------
        jit_PSF : numpy array
            PSF with linear jitter applied
    """

    def sample_circle(centre: tuple, r: float, theta: float):
        """Function to sample a circle for random direction."""
        h, k = centre
        x = h + r * np.sin(theta)
        y = k + r * np.cos(theta)
        return x, y

    # enforce odd image size
    if im_size % 2 != 1:
        raise ValueError("`im_size` must be an odd integer.")

    origin = (im_size // 2, im_size // 2)  # for centre of image
    if theta is None:
        theta = rd.uniform(0, 2 * np.pi)  # generating random theta

    points = [origin, sample_circle(origin, radius, theta)]  # creating line endpoints
    kernel_img = Image.new("1", (im_size, im_size))  # creating new Image object
    img = ImageDraw.Draw(kernel_img)  # create image
    img.line(points, fill="white", width=0)  # drawing line on image
    kernel = np.asarray(kernel_img)  # convert image to numpy array

    # convolving PSF with line
    jit_PSF = convolve2d(PSF, kernel, mode='same')
    # renormalising to intensity before convolution
    PSF_sum = np.sum(PSF)
    jit_PSF = jit_PSF / np.sum(jit_PSF) * PSF_sum

    return jit_PSF
