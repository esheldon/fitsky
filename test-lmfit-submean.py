from numba import njit
import matplotlib.pyplot as mplt
import numpy as np
import ngmix
from ngmix import GMixRangeError
from ngmix.gmix.gmix_nb import (
    gmix_set_norms,
    gmix_eval_pixel_fast,
)
from ngmix.defaults import LOWVAL
import galsim
from tqdm import trange
import esutil as eu

PIXEL_SCALE = 0.2
STAMP_SIZE = 51
FWHM = 0.7
# NOISE = 1.0
NOISE = 0.00001
# NOISE = 0.01


@njit
def measure_edge_bg(image, width):
    nrows, ncols = image.shape

    bsum = 0.0
    for row in range(nrows):
        if width <= row < nrows - width:
            for col in range(ncols):
                if width <= col < ncols - width:
                    bsum += image[row, col]

    return bsum / image.size


@njit
def fill_fdiff_subtract_mean(gmix, pixels, fdiff, start):
    """
    fill fdiff array (model-data)/err

    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    fdiff: array
        Array to fill, should be same length as pixels
    """

    if gmix["norm_set"][0] == 0:
        gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]

    pix_sum = 0.0
    model_sum = 0.0
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        pix_sum += pixel['val']
        model_sum += gmix_eval_pixel_fast(gmix, pixel)

    pix_mean = pix_sum / n_pixels
    model_mean = model_sum / n_pixels

    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        pixel_val = pixel['val'] - pix_mean
        model_val = gmix_eval_pixel_fast(gmix, pixel) - model_mean

        fdiff[start + ipixel] = (model_val - pixel_val) * pixel["ierr"]


class SubMeanCoellipFitter(ngmix.fitting.CoellipFitter):
    def _make_fit_model(self, obs, guess):
        return SubMeanCoellipFitModel(
            obs=obs, ngauss=self._ngauss, guess=guess, prior=self.prior,
        )


class SubMeanCoellipFitModel(ngmix.fitting.results.CoellipFitModel):
    def calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)

        try:

            # all norms are set after fill
            self._fill_gmix_all(pars)

            start = self._fill_priors(pars=pars, fdiff=fdiff)

            for pixels, gm in zip(self._pixels_list, self._gmix_data_list):
                fill_fdiff_subtract_mean(gm, pixels, fdiff, start)

                start += pixels.size

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff


class SubMeanFitter(ngmix.fitting.Fitter):
    def _make_fit_model(self, obs, guess):
        return SubMeanFitModel(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


class SubMeanFitModel(ngmix.fitting.results.FitModel):
    def calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)

        try:

            # all norms are set after fill
            self._fill_gmix_all(pars)

            start = self._fill_priors(pars=pars, fdiff=fdiff)

            for pixels, gm in zip(self._pixels_list, self._gmix_data_list):
                fill_fdiff_subtract_mean(gm, pixels, fdiff, start)

                start += pixels.size

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff


def make_obj(flux):
    return galsim.Moffat(fwhm=FWHM, flux=flux, beta=3.5)
    # return galsim.Gaussian(fwhm=FWHM, flux=flux)


def make_image(rng, flux):
    obj = make_obj(flux)

    offset = rng.uniform(low=-0.5, high=0.5, size=2)
    gsim = obj.drawImage(
        nx=STAMP_SIZE,
        ny=STAMP_SIZE,
        scale=PIXEL_SCALE,
        offset=offset,
        # method='no_pixel',
    )

    im = gsim.array
    im += rng.normal(scale=NOISE, size=im.shape)
    return im


def make_obs(rng, flux):
    im = make_image(rng=rng, flux=flux)

    cen = (np.array(im.shape) - 1.0) / 2.0

    jac = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=PIXEL_SCALE,
    )
    return ngmix.Observation(
        image=im,
        weight=im * 0 + 1.0 / NOISE ** 2,
        jacobian=jac,
    )


def get_coellip_prior(rng, ngauss, scale, T_range=None, F_range=None):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    T_range: (float, float), optional
        The range for the prior on T
    F_range: (float, float), optional
        Fhe range for the prior on flux
    """
    if T_range is None:
        T_range = [-1.0, 1.e3]
        # T_range = [0.05, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]
        # F_range = [0.05, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.5, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0,
        cen2=0,
        sigma1=scale * 0.01,
        sigma2=scale * 0.01,
        rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(
        minval=T_range[0], maxval=T_range[1], rng=rng,
    )
    F_prior = ngmix.priors.FlatPrior(
        minval=F_range[0], maxval=F_range[1], rng=rng,
    )

    prior = ngmix.joint_prior.PriorCoellipSame(
        ngauss=ngauss,
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior


def get_prior(rng, scale, T_range=None, F_range=None):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    T_range: (float, float), optional
        The range for the prior on T
    F_range: (float, float), optional
        Fhe range for the prior on flux
    """
    if T_range is None:
        T_range = [-1.0, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(
        minval=T_range[0], maxval=T_range[1], rng=rng,
    )
    F_prior = ngmix.priors.FlatPrior(
        minval=F_range[0], maxval=F_range[1], rng=rng,
    )

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior


def get_runner(rng, submean=False):
    prior = get_prior(rng=rng, scale=PIXEL_SCALE)
    if submean:
        fitter = SubMeanFitter(model='gauss', prior=prior)
    else:
        fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    guesser = ngmix.guessers.SimplePSFGuesser(
        rng=rng, guess_from_moms=True,
    )
    return ngmix.runners.Runner(
        fitter=fitter, guesser=guesser,
        ntry=2,
    )


def get_coellip_runner(rng, ngauss, submean=False):
    # prior = None
    prior = get_coellip_prior(ngauss=ngauss, rng=rng, scale=PIXEL_SCALE)
    if submean:
        fitter = SubMeanCoellipFitter(ngauss=ngauss, prior=prior)
    else:
        fitter = ngmix.fitting.CoellipFitter(ngauss=ngauss, prior=prior)

    # guesser = ngmix.guessers.SimplePSFGuesser(
    #     rng=rng, guess_from_moms=True,
    # )
    guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=ngauss)

    return ngmix.runners.Runner(
        fitter=fitter, guesser=guesser,
        ntry=2,
    )


def get_flux(res, obs):
    # has norm 1
    gm = res.get_gmix()
    model = gm.make_image(obs.image.shape, jacobian=obs.jacobian)

    xcorr_sum = (model * obs.image).sum()
    msq_sum = (model * model).sum()
    flux = xcorr_sum / msq_sum
    return flux


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


def fit_for_background(obs, res):
    # from espy.images import compare_images

    gm = res.get_gmix()
    im = gm.make_image(obs.image.shape, jacobian=obs.jacobian)

    # compare_images(obs.image, im)

    imdiff = obs.image - im
    bg = imdiff.mean()
    bg_err = imdiff.std() / np.sqrt(im.size)

    bg_frac = bg * obs.image.size / obs.image.sum()
    return bg, bg_err, bg_frac


def do_em(obs, rng, ngauss):
    obs_with_sky, sky = ngmix.em.prep_obs(obs)

    fitter = ngmix.em.EMFitter(
        maxiter=5000,
        vary_sky=True,
    )
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=True,
    )

    for itrial in range(10):
        guess = guesser(obs_with_sky)
        res = fitter.go(obs=obs_with_sky, guess=guess, sky=sky)
        if res['flags'] == 0:
            break

    if res['flags'] != 0:
        raise RuntimeError(f'failed {res["message"]}')

    bg = res['sky'] - sky

    gm = res.get_gmix()
    model = gm.make_image(obs.image.shape, jacobian=obs.jacobian)
    # this should be the sky
    imdiff = obs.image - model
    bg_err = imdiff.std() / np.sqrt(model.size)

    # bg_pix = bg * PIXEL_SCALE ** 2
    # bg_frac = bg_pix * obs.image.size / obs.image.sum()
    bg_frac = bg * obs.image.size / obs.image.sum()
    # print(f'bg: {bg:g}')
    return bg, bg_err, bg_frac


def do_trial_avg(rng, nobj, flux_min, flux_max, background, show=False):

    for i in range(nobj):

        tcat = make_cat()

        log_flux_true = rng.uniform(
            np.log10(flux_min),
            np.log10(flux_max),
        )

        tcat['flux_true'] = 10.0 ** log_flux_true
        tobs = make_obs(rng=rng, flux=tcat['flux_true'])

        with tobs.writeable():
            tobs.image += background * PIXEL_SCALE ** 2

        if i == 0:
            obstot = tobs
        else:
            with obstot.writeable():
                obstot.image += tobs.image
                obstot.weight += tobs.weight

    with obstot.writeable():
        obstot.image *= 1.0 / nobj

    if True:
        # runner = get_runner(rng, submean=True)
        runner = get_coellip_runner(rng, submean=True, ngauss=5)
        res = runner.go(obstot)
        bg, bg_err, bg_frac = fit_for_background(obs=obstot, res=res)
        bg *= 1 / PIXEL_SCALE ** 2
        bg_err *= 1 / PIXEL_SCALE ** 2
    else:
        try:
            bg, bg_err, bg_frac = do_em(obs=obstot, rng=rng, ngauss=3)
            bg *= 1 / PIXEL_SCALE ** 2
            bg_err *= 1 / PIXEL_SCALE ** 2
        except ngmix.GMixRangeError as err:
            print(str(err))
            return None

    # bg = measure_edge_bg(obstot.image, width=2)
    return {
        'bg': bg,
        'bg_err': bg_err,
        'bg_frac': bg_frac,
    }


def plot_hist_results(results, background, plotfile):

    fig, axs = mplt.subplots(
        nrows=2, ncols=2, figsize=(10, 8), layout='tight',
    )

    ax = axs[0, 0]
    ax.set(xlabel=r'$b$')

    bmean, bstd, berr = eu.stat.sigma_clip(
        results['background'],
        get_err=True,
    )
    # bmean = results['background'].mean()
    # berr = results['background'].std() / np.sqrt(results.size)

    x = 0.6
    y = 0.9
    ax.text(
        x, y, f'b mean: {bmean:.3f} +/- {berr:.3f}',
        transform=ax.transAxes,
    )

    ax.hist(
        results['background'], bins=50, color='C1', edgecolor='black',
        alpha=0.5,
    )
    ax.axvline(background, color='red')

    ax = axs[0, 1]
    ax.set(xlabel=r'$(b - b_{\mathrm{true}}) / b_{\mathrm{true}}$')

    fdiff = results['background'] / background - 1
    fdiff_mean, fdiff_std, fdiff_err = eu.stat.sigma_clip(
        fdiff,
        get_err=True,
    )
    # fdiff_mean = fdiff.mean()
    # fdiff_err = fdiff.std() / np.sqrt(results.size)

    ax.text(
        x, y, f'mean: {fdiff_mean:.2f} +/- {fdiff_err:.2f}',
        transform=ax.transAxes,
    )

    ax.hist(fdiff, bins=50, color='C1', edgecolor='black', alpha=0.5)

    ax = axs[1, 0]
    ax.set(
        xlabel=r'$(b - b_{\mathrm{true}}) / \sigma$',
    )

    mean_err = results['background_err'].mean()
    fdiff = (results['background'] - background) / mean_err

    fdiff_mean, fdiff_std, fdiff_err = eu.stat.sigma_clip(
        fdiff,
        get_err=True,
    )

    # fdiff_mean = fdiff.mean()
    # fdiff_std = fdiff.std()
    # fdiff_err = fdiff_std / np.sqrt(results.size)

    ax.text(
        x, y, f'mean: {fdiff_mean:.2f} +/- {fdiff_err:.2f}',
        transform=ax.transAxes,
    )
    ax.text(
        x, y - 0.05, f'std: {fdiff_std:.2f}',
        transform=ax.transAxes,
    )

    ax.hist(fdiff, bins=50, color='C1', edgecolor='black', alpha=0.5)

    ax = axs[1, 1]
    ax.set(xlabel=r'$b_{\mathrm{frac}}$')

    # bfrac_mean = results['background_frac'].mean()
    # bfrac_err = results['background_frac'].std() / np.sqrt(results.size)
    bfrac_mean, _, bfrac_err = eu.stat.sigma_clip(
        results['background_frac'],
        get_err=True,
    )

    ax.text(
        x - 0.1, y,
        rf'bfrac mean: {bfrac_mean:.3f} +/- {bfrac_err:.3f}',
        transform=ax.transAxes,
    )

    ax.hist(
        results['background_frac'],
        bins=50,
        color='C1',
        edgecolor='black',
        alpha=0.5,
    )

    print('writing:', plotfile)
    fig.savefig(plotfile)
    # mplt.show()


def make_result(n=1):
    dtype = [
        ('background', 'f4'),
        ('background_err', 'f4'),
        ('background_frac', 'f4'),
    ]

    return np.zeros(n, dtype=dtype)


def main(ntrial, nper, flux_min, flux_max, background, plotfile, show):

    rng = np.random.RandomState()

    dlist = []
    for i in trange(ntrial):

        tres = make_result()
        res = do_trial_avg(
            rng=rng,
            nobj=nper,
            flux_min=flux_min,
            flux_max=flux_max,
            background=background,
            show=show,
        )
        if res is None:
            continue

        bg = res['bg']
        bg_err = res['bg_err']
        tres['background_frac'] = res['bg_frac']
        # print(res['bg_frac'])

        tres['background'] = bg
        tres['background_err'] = bg_err

        dlist.append(tres)

    results = eu.numpy_util.combine_arrlist(dlist)
    plot_hist_results(results, background, plotfile=plotfile)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotfile', required=True)
    parser.add_argument('--ntrial', type=int, default=1000)
    parser.add_argument('--nper', type=int, default=100)
    parser.add_argument('--flux-min', type=float, default=320)
    parser.add_argument('--flux-max', type=float, default=3500)
    parser.add_argument(
        '--background',
        type=float,
        default=0.025,
        help='background per arcsecond squared',
    )
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(
        plotfile=args.plotfile,
        ntrial=args.ntrial,
        nper=args.nper,
        flux_min=args.flux_min,
        flux_max=args.flux_max,
        background=args.background,
        show=args.show,
    )
