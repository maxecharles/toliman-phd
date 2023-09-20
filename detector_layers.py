import jax
from jax import numpy as np
from jax import Array
from jax.scipy.stats import multivariate_normal
from dLux.detector_layers import DetectorLayer
import dLux

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

    def __init__(
        self: DetectorLayer,
        r: float,
        shear: float = 0,
        phi: float = 0,
        kernel_size: int = 10,
    ):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        r : float, arcseconds
            The standard deviation of the jitter in arcseconds.
        shear : float
            The shear of the jitter. Must lie on the interval [0, 1).
            A radially symmetric Gaussian kernel would have a shear value of 0,
            whereas a shear value approaching 1 would approach a linear kernel.
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
        # the -pi/4 offset is such that the rotation angle is relative to the x-axis rather than the line y=x
        rot_angle = np.radians(self.phi) - np.pi / 4

        # Construct the rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )

        # Construct the skew matrix
        skew_matrix = np.array(
            [
                [1, self.shear],
                [self.shear, 1],
            ]
        )  # Ensure skew_matrix is symmetric

        # Compute the covariance matrix
        covariance_matrix = (self.r**2) * np.dot(
            np.dot(rotation_matrix, skew_matrix), rotation_matrix.T
        )

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
        x = np.linspace(0, extent, self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = multivariate_normal.pdf(
            pos, mean=np.array([0.0, 0.0]), cov=self.covariance_matrix
        )

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
