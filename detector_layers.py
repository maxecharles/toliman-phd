from jax import numpy as np
from jax import Array
from jax.scipy.stats import multivariate_normal
from dLux.detector_layers import DetectorLayer
import dLux as dl

Image = lambda: dl.images.Image


class ApplyAsymmetricJitter(DetectorLayer):
    kernel_size: int
    cov: Array | None
    params: list | None

    def __init__(
        self: DetectorLayer,
        r: float = None,
        shear: float = None,
        phi: float = None,
        kernel_size: int = 10,
        cov: Array = None,
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
        cov : Array, pixels
            The covariance matrix of the Gaussian kernel, in units of arcseconds.
            Must be a 2x2, symmetric, positive semi-definite matrix. If specified, r, shear, and phi are ignored.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)

        if cov is not None:
            if cov.shape != (2, 2):
                raise ValueError("Covariance matrix must be 2x2.")
            if not np.allclose(cov, cov.T):
                raise ValueError("Covariance matrix must be symmetric.")
            eigvals = np.linalg.eigvals(cov)
            if np.any(eigvals <= 0):
                raise ValueError(
                    "The covariance matrix must be positive semi-definite."
                )

            self.cov = np.asarray(cov, dtype=float)
            self.params = None

        elif r is not None and shear is not None and phi is not None:
            self.cov = self._generate_covariance_matrix(r, shear, phi)
            self.params = [r, shear, phi]

        else:
            raise ValueError(
                "Must specify either covariance matrix or r, shear, and phi."
            )

    @staticmethod
    def _generate_covariance_matrix(r, shear, phi):
        angle_rad = np.radians(phi)

        # Construct the rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Construct the skew matrix
        skew_matrix = np.array(
            [[1, shear], [shear, 1]]
        )  # Ensure skew_matrix is symmetric

        # Compute the covariance matrix
        covariance_matrix = r * np.dot(
            np.dot(rotation_matrix, skew_matrix), rotation_matrix.T
        )

        # Ensure positive semi-definiteness
        eigvals = np.linalg.eigvals(covariance_matrix)
        if np.any(eigvals <= 0):
            raise ValueError(
                "The resulting covariance matrix is not positive semi-definite."
            )

        return covariance_matrix

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

        kernel = multivariate_normal.pdf(pos, mean=np.array([0.0, 0.0]), cov=self.cov)

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
