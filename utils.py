"""Some useful functions."""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def PlotPSF(PSF, title: str = None):
    """
    Plot the square root of the PSF.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(PSF, cmap='inferno')
    fig.colorbar(im, ax=ax, label='photons')
    ax.set(
            title=title,
            xticks=[0, PSF.shape[0] - 1],
            yticks=[0, PSF.shape[1] - 1],
            )
    plt.show()

    return


def PlotSqrtPSF(PSF, title: str = None):
    """
    Plot the square root of the PSF.
    """

    # Checking for negative values
    if np.any(PSF < 0):
        PSF = np.abs(PSF)
        print('Warning: Negative values in PSF. Taking absolute value.')

    fig, ax = plt.subplots()
    im = ax.imshow(PSF, cmap='inferno', norm=colors.PowerNorm(gamma=0.5))
    fig.colorbar(im, ax=ax, label='photons')
    ax.set(
            title=title,
            xticks=[0, PSF.shape[0] - 1],
            yticks=[0, PSF.shape[1] - 1],
            )
    plt.show()

    return
