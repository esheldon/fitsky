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
from tqdm import trange
import esutil as eu

from sfitsky.coellip_fitter import get_coellip_runner_with_sky
from sfitsky.sim import simulate_stack, get_fwhm, get_e1e2

PIXEL_SCALE = 0.2
FWHM_MEAN = 0.8
FWHM_SIGMA = 0.1
FWHM_MIN = 0.65
E_SIGMA = 0.03


@njit
def fill_fdiff_with_sky(gmix, sky, pixels, fdiff, start):
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

    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        pixel_val = pixel['val']
        model_val = gmix_eval_pixel_fast(gmix, pixel) + sky

        fdiff[start + ipixel] = (model_val - pixel_val) * pixel["ierr"]


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
            obs=obs,
            ngauss=self._ngauss,
            guess=guess,
            prior=self.prior,
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
            obs=obs,
            model=self.model,
            guess=guess,
            prior=self.prior,
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
        T_range = [-1.0, 1.0e3]
        # T_range = [0.05, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.0e9]
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
        minval=T_range[0],
        maxval=T_range[1],
        rng=rng,
    )
    F_prior = ngmix.priors.FlatPrior(
        minval=F_range[0],
        maxval=F_range[1],
        rng=rng,
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
        T_range = [-1.0, 1.0e3]
    if F_range is None:
        F_range = [-100.0, 1.0e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0,
        cen2=0,
        sigma1=scale,
        sigma2=scale,
        rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(
        minval=T_range[0],
        maxval=T_range[1],
        rng=rng,
    )
    F_prior = ngmix.priors.FlatPrior(
        minval=F_range[0],
        maxval=F_range[1],
        rng=rng,
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
        rng=rng,
        guess_from_moms=True,
    )
    return ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )


def get_coellip_runner(rng, ngauss, submean=False):
    # prior = None
    prior = get_coellip_prior(ngauss=ngauss, rng=rng, scale=PIXEL_SCALE)
    if submean:
        fitter = SubMeanCoellipFitter(ngauss=ngauss, prior=prior)
    else:
        fitter = ngmix.fitting.CoellipFitter(ngauss=ngauss, prior=prior)

    guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=ngauss)

    return ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
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


def fit_for_background(obs, res):
    # from espy.images import compare_images

    gm = res.get_gmix()
    im = gm.make_image(obs.image.shape, jacobian=obs.jacobian)

    # compare_images(obs.image, im)

    imdiff = obs.image - im

    # bg, _, bg_err = eu.stat.sigma_clip(
    #     imdiff.ravel(),
    #     get_err=True,
    # )

    bg = imdiff.mean()
    # bg = np.median(imdiff)
    bg_err = imdiff.std() / np.sqrt(im.size)

    bg_frac = bg * obs.image.size / obs.image.sum()
    return bg, bg_err, bg_frac


def prep_obs(obs):
    noise = np.sqrt(1.0 / np.median(obs.weight))
    im = obs.image.copy()

    # need no zero pixels and sky value
    im_min = im.min()

    desired_minval = 0.001 * noise

    sky = desired_minval - im_min
    im += sky

    newobs = ngmix.Observation(
        im,
        jacobian=obs.jacobian,
    )

    return newobs, sky


def do_em(obs, rng, ngauss):
    # obs_with_sky, sky = ngmix.em.prep_obs(obs)
    obs_with_sky, sky = prep_obs(obs)

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


def fit_stack(
    obs, method, noise, rng, sky_prior_mean=0.0, sky_prior_sigma=None,
):
    if sky_prior_sigma is None:
        sky_prior_sigma = noise

    if method == 'fitsky':
        runner = get_coellip_runner_with_sky(
            rng,
            ngauss=5,
            cen_prior_sigma=PIXEL_SCALE * 0.01,
            sky_prior_mean=sky_prior_mean,
            sky_prior_sigma=sky_prior_sigma,
        )
        res = runner.go(obs)
        bg = res['sky']
        bg_err = res['sky_err']
        bg_frac = bg * obs.image.size / obs.image.sum()
    elif method == 'submean':
        # runner = get_runner(rng, submean=True)
        runner = get_coellip_runner(rng, submean=True, ngauss=5)
        res = runner.go(obs)
        bg, bg_err, bg_frac = fit_for_background(obs=obs, res=res)
        # bg *= 1 / PIXEL_SCALE**2
        # bg_err *= 1 / PIXEL_SCALE**2
    elif 'em' in method:
        try:
            bg, bg_err, bg_frac = do_em(obs=obs, rng=rng, ngauss=5)
            # bg *= 1 / PIXEL_SCALE**2
            # bg_err *= 1 / PIXEL_SCALE**2
        except ngmix.GMixRangeError as err:
            print(str(err))
            return None

        if method == 'emhybrid':
            runner = get_coellip_runner_with_sky(
                rng,
                ngauss=5,
                sky_prior_mean=bg,
                sky_prior_sigma=bg_err * 10,
            )
            res = runner.go(obs)
            bg = res['sky']
            bg_err = res['sky_err']
            bg_frac = bg * obs.image.size / obs.image.sum()

    else:
        raise ValueError(f'Bad method name "{method}"')

    return {
        'bg': bg,
        'bg_err': bg_err,
        'bg_frac': bg_frac,
    }


def get_nbins(data, std):
    binsize = 0.3 * std
    return int((data.max() - data.min()) / binsize)


def plot_hist_results(results, background, plotfile):
    fig, axs = mplt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        layout='tight',
    )

    ax = axs[0, 0]
    ax.set(xlabel=r'$b$')

    bmean, bstd, berr = eu.stat.sigma_clip(
        results['bg'],
        get_err=True,
    )

    x = 0.5
    y = 0.9
    ax.text(
        x,
        y,
        f'mean: {bmean:.3g} +/- {berr:.3g}',
        transform=ax.transAxes,
    )

    ax.hist(
        results['bg'],
        bins=get_nbins(results['bg'], bstd),
        color='C1',
        edgecolor='black',
        alpha=0.5,
    )
    ax.axvline(background, color='red')

    ax = axs[0, 1]
    ax.set(xlabel=r'$(b - b_{\mathrm{true}}) / b_{\mathrm{true}}$')

    fdiff = results['bg'] / background - 1
    fdiff_mean, fdiff_std, fdiff_err = eu.stat.sigma_clip(
        fdiff,
        get_err=True,
    )

    ax.text(
        x,
        y,
        f'mean: {fdiff_mean:.3g} +/- {fdiff_err:.3g}',
        transform=ax.transAxes,
    )

    ax.hist(
        fdiff,
        bins=get_nbins(fdiff, fdiff_std),
        color='C1', edgecolor='black', alpha=0.5,
    )

    ax = axs[1, 0]
    ax.set(
        xlabel=r'$(b - b_{\mathrm{true}}) / \sigma$',
    )

    mean_err = np.median(results['bg_err'])
    fdiff = (results['bg'] - background) / mean_err

    fdiff_mean, fdiff_std, fdiff_err = eu.stat.sigma_clip(
        fdiff,
        get_err=True,
    )

    ax.text(
        x,
        y,
        f'mean: {fdiff_mean:.3g} +/- {fdiff_err:.3g}',
        transform=ax.transAxes,
    )
    ax.text(
        x,
        y - 0.05,
        f'std: {fdiff_std:.3g}',
        transform=ax.transAxes,
    )

    ax.hist(
        fdiff,
        bins=get_nbins(fdiff, fdiff_std),
        color='C1',
        edgecolor='black',
        alpha=0.5,
    )

    ax = axs[1, 1]
    ax.set(xlabel=r'$b_{\mathrm{frac}}$')

    bfrac_mean, bfrac_std, bfrac_err = eu.stat.sigma_clip(
        results['bg_frac'],
        get_err=True,
    )

    ax.text(
        x,
        y,
        rf'bfrac mean: {bfrac_mean:.3g} +/- {bfrac_err:.3g}',
        transform=ax.transAxes,
    )

    ax.hist(
        results['bg_frac'],
        bins=get_nbins(results['bg_frac'], bfrac_std),
        color='C1',
        edgecolor='black',
        alpha=0.5,
    )

    print('writing:', plotfile)
    fig.savefig(plotfile)
    # mplt.show()


def make_result(n=1):
    dtype = [
        ('bg', 'f4'),
        ('bg_err', 'f4'),
        ('bg_frac', 'f4'),
    ]

    return np.zeros(n, dtype=dtype)


def main(
    method,
    ntrial,
    nper,
    flux_min,
    flux_max,
    stamp_size,
    noise,
    background,
    npass,
    plotfile,
    seed,
    show,
):
    rng = np.random.RandomState(seed)

    sky_prior_mean = 0.0
    sky_prior_sigma = noise
    assert 1 <= npass <= 2

    print('simulating stacks')
    observations = []
    for i in trange(ntrial, ascii=True, ncols=70):
        fwhm = get_fwhm(rng)
        e1, e2 = get_e1e2(rng)

        tmpobs = simulate_stack(
            nobj=nper,
            fwhm=fwhm,
            e1=e2,
            e2=e2,
            stamp_size=stamp_size,
            flux_min=flux_min,
            flux_max=flux_max,
            noise=noise,
            background=background,
            rng=rng,

        )
        observations.append(tmpobs)

    print('doing fits')
    for ipass in range(npass):
        if npass == 2:
            print(f'pass {ipass + 1}')

        dlist = []
        for i in trange(ntrial, ascii=True, ncols=70):

            obstot = observations[i]

            res = fit_stack(
                obs=obstot,
                method=method,
                noise=noise,
                rng=rng,
                sky_prior_mean=sky_prior_mean,
                sky_prior_sigma=sky_prior_sigma,
            )

            if res is None:
                continue

            tres = make_result()

            bg = res['bg']
            bg_err = res['bg_err']
            tres['bg_frac'] = res['bg_frac']

            tres['bg'] = bg
            tres['bg_err'] = bg_err

            dlist.append(tres)

        results = eu.numpy_util.combine_arrlist(dlist)
        if npass == 2:
            sky_prior_mean, sky_prior_sigma = eu.stat.sigma_clip(results['bg'])

    plot_hist_results(results, background, plotfile=plotfile)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plotfile', required=True)
    parser.add_argument('--ntrial', type=int, default=1000)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', default='fitsky')
    parser.add_argument('--nper', type=int, default=100)
    parser.add_argument('--flux-min', type=float, default=320)
    parser.add_argument('--flux-max', type=float, default=3500)
    parser.add_argument('--stamp-size', type=int, default=51)
    parser.add_argument('--noise', type=float, default=1)
    parser.add_argument(
        '--background',
        type=float,
        default=-0.3,
        help='background value',
    )
    parser.add_argument('--npass', type=int, default=1)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(
        method=args.method,
        ntrial=args.ntrial,
        seed=args.seed,
        nper=args.nper,
        flux_min=args.flux_min,
        flux_max=args.flux_max,
        stamp_size=args.stamp_size,
        noise=args.noise,
        background=args.background,
        plotfile=args.plotfile,
        npass=args.npass,
        show=args.show,
    )
