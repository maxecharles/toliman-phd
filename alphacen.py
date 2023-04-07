"""
Creating class for the Alpha Centauri model.
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

        mask_dir : string
            Disk directory of numpy array containing mask

    Methods
    -------
        GetPSF(wavefront_npixels=256, detector_npixels=128, sampling_rate=5,):
            Generates point spread function

        AddNoise(ideal_PSF):
            Adds poissonian and detector noise to point spread function

        LinearJitter(PSF):
            Convolves PSF with a line to simulate linear jitter.

    """

    # Parameters for Alpha Cen model
    sep = 10  # binary separation in arcseconds
    pa = 90  # position angle in degrees
    flux = 1e4  # average flux of the two stars
    contrast = 3.372873  # flux contrast from V band magnitudes
    mask_dir = 'AlfCenModel/data/test_mask.npy'

    # other system parameters
    wavels = 1e-9 * np.linspace(595, 695, 3)  # wavelengths

    def __init__(self, seed: int = 0):
        self.key = [jr.PRNGKey(seed), jr.PRNGKey(seed+1)]
        return

    # loading mask, converting phase to OPD and turning mask into a layer
    def LoadMask(self, mask_dir):
        mask = dl.optics.AddOPD(dl.utils.phase_to_opd(np.load(mask_dir), wavelength=self.wavels.mean()))
        return mask

    def GetPSF(self,
               mask,
               wavefront_npixels=256,  # wavefront layer size
               detector_npixels=128,  # detector size
               sampling_rate=5,  # pixels per fringe i.e. 5x Nyquist
               ):
        """Generating PSF of Alpha Cen through TOLIMAN."""

        detector_pixel_size = dl.utils.get_pixel_scale(sampling_rate, self.wavels.max(), 0.125)

        # Make optical system
        optics = dl.utils.toliman(wavefront_npixels,
                                  detector_npixels,
                                  detector_pixel_size=dl.utils.radians_to_arcseconds(detector_pixel_size),
                                  extra_layers=[mask],
                                  angular=True)

        # Resetting the pixel scale of output
        optics = optics.set(['AngularMFT.pixel_scale_out'], [dl.utils.arcseconds_to_radians(.375)])

        # Creating a model Alpha Cen source
        source = dl.BinarySource(separation=dl.utils.arcseconds_to_radians(self.sep),
                                 wavelengths=self.wavels,
                                 contrast=self.contrast,
                                 flux=self.flux,
                                 position_angle=np.deg2rad(self.pa),
                                 )

        # Creating the instrument by combining optics and source
        tol = dl.Instrument(optics=optics, sources=[source])

        # Generating and returning the PSF
        ideal_PSF = tol.model()
        return ideal_PSF

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

    def LinearJitter(self, PSF, radius=12, theta=None, im_size=25):
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
                PSF with linear jitter
        """

        def sample_circle(centre: tuple, r: float, theta: float):
            """Function to sample a circle for random direction."""
            h, k = centre
            x = h + r * np.sin(theta)
            y = k + r * np.cos(theta)
            return x, y

        origin = (im_size // 2, im_size // 2)  # for centre of image
        if theta is None:
            theta = rd.uniform(0, 2 * np.pi)  # generating random theta

        points = [origin, sample_circle(origin, radius, theta)]  # creating line endpoints
        kernel_img = Image.new("1", (im_size, im_size))  # creating new Image object
        img = ImageDraw.Draw(kernel_img)  # create image
        img.line(points, fill="white", width=0)  # drawing line on image
        kernel = np.asarray(kernel_img)

        # convolving PSF with line
        PSF_sum = np.sum(PSF)
        jit_PSF = convolve2d(PSF, kernel, mode='same')
        jit_PSF = jit_PSF / np.sum(jit_PSF) * PSF_sum

        return jit_PSF
