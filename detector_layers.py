import dLux as dl
from jax import numpy as np

from dLux.detector_layers import DetectorLayer
from jax import Array
from jax.scipy.stats import multivariate_normal


class ApplyAsymmetricJitter(DetectorLayer):
    kernel_size: int
    cov: Array

    def __init__(self: DetectorLayer, cov: Array, kernel_size: int = 10):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        cov : Array, pixels
            The covariance matrix of the Gaussian kernel, in units of arcseconds.
        kernel_size : int = 10
            The size of the convolution kernel in pixels to use.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.cov = np.asarray(cov, dtype=float)
        if self.cov.shape != (2, 2):
            raise ValueError('Covariance matrix must be 2x2.')

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the normalised Gaussian kernel.

        Parameters
        ----------
        pixel_scale : float, arcsec/pixel
            The pixel scale of the image.

        Returns
        -------
        kernel : Array
            The normalised Gaussian kernel.
        """
        # Generate distribution
        extent = pixel_scale * self.kernel_size
        x = np.linspace(0, extent, self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = multivariate_normal.pdf(pos, mean=np.array([0., 0.]), cov=self.cov)

        return kernel / np.sum(kernel)

    def __call__(self: DetectorLayer, image: Image()) -> Image():
        """
        Applies the layer to the Image.

        Parameters
        ----------
        image : Image
            The image to operate on.

        Returns
        -------
        image : Image
            The transformed image.
        """
        kernel = self.generate_kernel(dl.utils.rad_to_arcsec(image.pixel_scale))

        return image.convolve(kernel)