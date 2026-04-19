import numpy as np

PIXEL_SCALE = 0.2
FWHM_MEAN = 0.8
FWHM_SIGMA = 0.1
FWHM_MIN = 0.65
E_SIGMA = 0.03


def simulate_stack(
    nobj,
    fwhm,
    e1,
    e2,
    flux_min,
    flux_max,
    noise,
    background,
    stamp_size,
    rng,
):
    for i in range(nobj):
        tcat = make_cat()

        log_flux_true = rng.uniform(
            np.log10(flux_min),
            np.log10(flux_max),
        )

        tcat['flux_true'] = 10.0**log_flux_true
        tobs = make_obs(
            rng=rng,
            flux=tcat['flux_true'],
            fwhm=fwhm,
            e1=e1,
            e2=e2,
            stamp_size=stamp_size,
            noise=noise,
        )

        with tobs.writeable():
            tobs.image += background

        if i == 0:
            obstot = tobs
        else:
            with obstot.writeable():
                obstot.image += tobs.image
                obstot.weight += tobs.weight

    with obstot.writeable():
        obstot.image *= 1.0 / nobj

    return obstot


def make_obs(rng, fwhm, e1, e2, flux, stamp_size, noise):
    import ngmix

    im = make_image(
        rng=rng,
        fwhm=fwhm,
        e1=e1,
        e2=e2,
        flux=flux,
        stamp_size=stamp_size,
        noise=noise,
    )

    cen = (np.array(im.shape) - 1.0) / 2.0

    jac = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=PIXEL_SCALE,
    )
    return ngmix.Observation(
        image=im,
        weight=im * 0 + 1.0 / noise ** 2,
        jacobian=jac,
    )


def make_obj(flux, fwhm, e1, e2):
    import galsim

    return galsim.Moffat(
        fwhm=fwhm,
        flux=flux,
        beta=3.5,
    ).shear(
        e1=e1,
        e2=e2,
    )


def make_image(rng, fwhm, e1, e2, flux, stamp_size, noise):
    obj = make_obj(flux=flux, e1=e1, e2=e2, fwhm=fwhm)

    offset = rng.uniform(low=-0.5, high=0.5, size=2)
    gsim = obj.drawImage(
        nx=stamp_size,
        ny=stamp_size,
        scale=PIXEL_SCALE,
        offset=offset,
        # method='no_pixel',
    )

    im = gsim.array
    im += rng.normal(scale=noise, size=im.shape)
    return im


def make_cat(n=1):
    dtype = [
        ('flux_true', 'f4'),
        ('snr', 'f4'),
        ('T', 'f4'),
        ('T_err', 'f4'),
        ('flux', 'f4'),
        ('flux_err', 'f4'),
        ('bg', 'f4'),
        ('bg_err', 'f4'),
    ]
    return np.zeros(n, dtype=dtype)


def get_fwhm(rng):
    import ngmix

    p = ngmix.priors.LogNormal(mean=FWHM_MEAN, sigma=FWHM_SIGMA, rng=rng)

    while True:
        fwhm = p.sample()
        if fwhm > FWHM_MIN:
            break

    return fwhm


def get_e1e2(rng):
    while True:
        e1, e2 = rng.normal(scale=E_SIGMA, size=2)
        e = np.sqrt(e1 ** 2 + e2 ** 2)
        if e < 0.99:
            break

    return e1, e2
