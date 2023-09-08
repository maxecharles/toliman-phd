import jax
from jax import numpy as np
from jax import Array
from jax.scipy.stats import multivariate_normal
from dLux.detector_layers import DetectorLayer
import dLux as dl

Image = lambda: dl.images.Image


class ApplyAsymmetricJitter(DetectorLayer):
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
            The magnitude of the jitter.
        shear : float
            The shear of the jitter.
        phi : float, degrees
            The angle of the jitter.
        kernel_size : int = 10
            The size of the convolution kernel in pixels to use.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.r = r
        self.shear = shear
        self.phi = phi

    @property
    def covariance_matrix(self):
        angle_rad = np.radians(self.phi)

        # Construct the rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Construct the skew matrix
        skew_matrix = np.array(
            [[1, self.shear], [self.shear, 1]]
        )  # Ensure skew_matrix is symmetric

        # Compute the covariance matrix
        covariance_matrix = self.r * np.dot(
            np.dot(rotation_matrix, skew_matrix), rotation_matrix.T
        )

        # Ensure positive semi-definiteness
        try:
            # Attempt Cholesky decomposition
            jax.scipy.linalg.cholesky(covariance_matrix)
            return covariance_matrix
        except:
            raise ValueError("Covariance matrix is not positive semi-definite.")

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
        extent = pixel_scale * self.kernel_size  # kernel size in arcseconds
        x = np.linspace(0, extent, self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = multivariate_normal.pdf(pos, mean=np.array([0.0, 0.0]), cov=self.covariance_matrix)

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
