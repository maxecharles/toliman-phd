import jax
from jax import numpy as np
from jax import Array
from jax.scipy.stats import multivariate_normal
from dLux.detector_layers import DetectorLayer
import dLux
import dLux.utils as dlu

Image = lambda: dLux.images.Image


class ApplyJitter(DetectorLayer):
    """
    Convolves the image with a Gaussian kernel parameterised by the standard
    deviation (sigma).

    Attributes
    ----------
    kernel_size : int
        The size in pixels of the convolution kernel to use.
    r : float, arcseconds
        The magnitude of the jitter.
    shear : float
        The shear of the jitter.
    phi : float, degrees
        The angle of the jitter.
    """

    kernel_size: int
    r: float = None
    shear: float = None
    phi: float = None
    kernel_oversample: int = None

    def __init__(
        self: DetectorLayer,
        r: float,
        shear: float = 0,
        phi: float = 0,
        kernel_size: int = 10,
        kernel_oversample: int = 1,
    ):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        r : float
            The jitter magnitude, defined as the determinant of the covariance
            matrix of the multivariate Gaussian kernel. This is the product of the
            standard deviations of the minor and major axes of the kernel, given in
            arcseconds.
        shear : float, [0, 1)
            A measure of how asymmetric the jitter is. Defined as one minus the ratio between
            the standard deviations of the minor/major axes of the multivariate
            Gaussian kernel. It must lie on the interval [0, 1). A shear of 0
            corresponds to a symmetric jitter, while as shear approaches one the
            jitter kernel becomes more linear.
        phi : float
            The angle of the jitter in degrees.
        kernel_size : int = 10
            The size of the convolution kernel in pixels to use.
        """
        super().__init__()

        # checking shear is valid
        if shear >= 1 or shear < 0:
            raise ValueError("shear must lie on the interval [0, 1)")

        self.kernel_size = int(kernel_size)
        self.r = r
        self.shear = shear
        self.phi = phi
        self.kernel_oversample = int(kernel_oversample)

    @property
    def covariance_matrix(self):
        """
        Generates the covariance matrix for the multivariate normal distribution.

        Returns
        -------
        covariance_matrix : Array
            The covariance matrix.
        """
        # Compute the rotation angle
        rot_angle = np.radians(self.phi)

        # Construct the rotation matrix
        R = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )

        # calculating the eigenvalues (lambda1 > lambda2)
        lambda1 = (self.r / (1 - self.shear))**0.25
        lambda2 = lambda1 * (1 - self.shear)

        # Construct the skew matrix
        base_matrix = np.array(
            [
                [lambda1**2, 0],
                [0, lambda2**2],
            ]
        )

        # Compute the covariance matrix
        covariance_matrix = np.dot(np.dot(R, base_matrix), R.T)

        return covariance_matrix

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the normalised multivariate Gaussian kernel.

        Parameters
        ----------
        pixel_scale : float
            The pixel scale of the image in arcseconds per pixel.

        Returns
        -------
        kernel : Array
            The normalised Gaussian kernel.
        """
        # Generate distribution
        extent = pixel_scale * self.kernel_size  # kernel size in arcseconds
        x = np.linspace(0, extent, self.kernel_oversample * self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = dlu.downsample(multivariate_normal.pdf(
            pos, mean=np.array([0.0, 0.0]), cov=self.covariance_matrix
        ), self.kernel_oversample)

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
        kernel = self.generate_kernel(dLux.utils.rad_to_arcsec(image.pixel_scale))

        return image.convolve(kernel)
