def main():
    import os  # Set CPU count for numpyro multichain multi-thread
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
    from jax import device_count
    if device_count() != 4:
        raise ValueError("JAX/XLA is not using all 4 cores")

    import jax.numpy as np
    import jax.random as jr
    from dLuxToliman import AlphaCen, TolimanOptics, Toliman  # Source & Optics
    import numpyro as npy
    import numpyro.distributions as dist

    def psf_model(data, model):
        """
        Define the numpyro function
        """
        values = [
            # npy.sample("x",        dist.Uniform(-0.1, 0.1)),
            # npy.sample("y",        dist.Uniform(-0.1, 0.1)),
            npy.sample("sep",      dist.Uniform(6, 10)),
            npy.sample("pa",       dist.Uniform(0, 360)),
            npy.sample("log_flux", dist.Uniform(5, 8)),
            npy.sample("contrast", dist.Uniform(3, 5)),
        ]

        with npy.plate("data", len(data.flatten())):
            poisson_model = dist.Poisson(
                model.set(param_list, values).model().flatten())
            return npy.sample("psf", poisson_model, obs=data.flatten())

    param_list = [
        # 'x_position',
        # 'y_position',
        'separation',
        'position_angle',
        'log_flux',
        'contrast',
                  ]

    truths = [
        # npy.sample("x", dist.Uniform(-0.1, 0.1), rng_key=jr.PRNGKey(0)),
        # npy.sample("y",    dist.Uniform(-0.1, 0.1), rng_key=jr.PRNGKey(1)),
        npy.sample("sep",  dist.Uniform(6, 10), rng_key=jr.PRNGKey(3)),
        npy.sample("pa",   dist.Uniform(0, 360), rng_key=jr.PRNGKey(3)),
        npy.sample("logF", dist.Uniform(5, 8), rng_key=jr.PRNGKey(4)),
        npy.sample("cont", dist.Uniform(3, 4), rng_key=jr.PRNGKey(6)),
    ]

    optics = TolimanOptics(psf_npixels=128, psf_oversample=1.5)
    source = AlphaCen().set(param_list, truths)
    telescope = Toliman(optics, source)
    psf = telescope.model()

    # adding noise
    psf_photon = jr.poisson(jr.PRNGKey(0), psf)
    bg_noise = 3*jr.normal(jr.PRNGKey(0), psf_photon.shape)
    data = psf_photon + np.abs(bg_noise)

    sampler = npy.infer.MCMC(
        npy.infer.NUTS(psf_model),
        num_warmup=2000,
        num_samples=10000,
        num_chains=device_count(),
        progress_bar=True,
    )
    sampler.run(jr.PRNGKey(0), data, telescope)

    sampler.print_summary()
    values_out = sampler.get_samples()
    print(truths)

    import chainconsumer as cc

    chain = cc.ChainConsumer()
    chain.add_chain(values_out)  # , parameters=param_list, name="Recovered Parameters")
    chain.configure(serif=True, shade=True, bar_shade=True, shade_alpha=0.2, spacing=1., max_ticks=3)
    fig = chain.plotter.plot(truth=truths)
    fig.set_size_inches((6, 6))
    fig.savefig("figs/test.png", dpi=120)


if __name__ == '__main__':
    main()
