def main():
    import os  # Set CPU count for numpyro multichain multi-thread
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
    from jax import device_count
    if device_count() != 4:
        raise ValueError("JAX/XLA is not using all 4 cores")

    import jax.numpy as np
    import jax.random as jr
    import numpy
    from dLuxToliman import AlphaCen, TolimanOptics, Toliman  # Source & Optics
    import numpyro as npy
    import numpyro.distributions as dist
    import chainconsumer as cc
    import matplotlib.pyplot as plt

    ##############################
    warmup = 500
    samples = 4500

    indices = np.array([  # which parameters to fit for in the MCMC
        0,  # x_position
        1,  # y_position
        2,  # separation
        3,  # position_angle
        4,  # log_flux
        5,  # contrast
    ])

    priors = np.array([  # (high, low) of uniform distribution
        (-0.1, 0.1),  # x_position
        (-0.1, 0.1),  # y_position
        (7, 8),  # separation
        (20, 30),  # position_angle
        (6.2, 6.5),  # log_flux
        (3.2, 3.5),  # contrast
    ])[indices]
    ################################

    param_list = list(numpy.array([
        'x_position',
        'y_position',
        'separation',
        'position_angle',
        'log_flux',
        'contrast',
    ])[indices])

    names = list(numpy.array([
        'x',
        'y',
        'sep',
        'pa',
        'logF',
        'cont',
    ])[indices])

    # creating list of uniform distributions
    distributions = [dist.Uniform(*p) for p in priors]

    def psf_model(data, model):
        # sampling from the distributions
        values = [npy.sample(name=names[i], fn=distributions[i]) for i in range(len(indices))]

        with npy.plate("data", len(data.flatten())):
            poisson_model = dist.Poisson(
                model.set(param_list, values).model().flatten())
            return npy.sample("psf", poisson_model, obs=data.flatten())

    # generating truth values
    truths = [npy.sample(name=names[i],
                         fn=distributions[i],
                         rng_key=jr.PRNGKey(i)
                         ) for i in range(len(indices))]

    for name, value in zip(names, truths):
        print(f"{name} = {value}")

    param_dict = {names[i]: np.tile(truths[i], device_count()) for i in range(len(indices))}

    # generating data with truth values
    optics = TolimanOptics(psf_npixels=128, psf_oversample=1.5)
    source = AlphaCen(nwavels=3).set(param_list, truths)
    telescope = Toliman(optics, source)
    psf = telescope.model()
    # adding noise
    psf_photon = jr.poisson(jr.PRNGKey(0), psf)
    bg_noise = 3 * jr.normal(jr.PRNGKey(0), psf_photon.shape)
    data = psf_photon + np.abs(bg_noise)

    # running MCMC
    sampler = npy.infer.MCMC(
        npy.infer.NUTS(psf_model),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=device_count(),
        progress_bar=True,
    )

    sampler.run(jr.PRNGKey(0), data, telescope, init_params=param_dict)

    # analysing output
    sampler.print_summary()
    values_out = sampler.get_samples()

    # corner plot
    chain = cc.ChainConsumer()
    chain.add_chain(values_out)  # , parameters=param_list, name="Recovered Parameters")
    chain.configure(serif=True, shade=True, bar_shade=True, shade_alpha=0.2, spacing=1., max_ticks=3)
    fig = chain.plotter.plot(truth=truths)
    fig.set_size_inches((12, 12))
    fig.savefig("figs/mcmc/output.png", dpi=120)


if __name__ == '__main__':
    main()
