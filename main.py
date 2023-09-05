from jax.scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import jax.numpy as np

cov = np.array([[0.5, 0], [0, 0.5]])

pscale = 1/32  # arcseconds per pixel
kernel_size = 1000  # pixels
extent = pscale * kernel_size  # arcseconds

x = np.linspace(0, extent, kernel_size) - 0.5*extent
pos = np.dstack(np.meshgrid(x, x))
rv = multivariate_normal.pdf(pos, mean=np.array([0, 0]), cov=cov)

plt.figure()
plt.imshow(rv/rv.max(), extent=[x.min(), x.max(), x.min(), x.max()])
plt.colorbar()
plt.show()
