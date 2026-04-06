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
# NOISE = 0.00001
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


class PriorCoellipWithSky(ngmix.joint_prior.PriorSimpleSep):
    def __init__(
        self, ngauss, cen_prior, g_prior, T_prior, F_prior, sky_prior
    ):
        self.ngauss = ngauss
        self.npars = ngmix.gmix.get_coellip_npars(ngauss) + 1

        super().__init__(
            cen_prior, g_prior, T_prior, F_prior
        )
        self.sky_prior = sky_prior

        if self.nband != 1:
            raise ValueError("coellip only supports one band")

        self.bounds = None

    def set_bounds(self):
        """
        set possibe bounds
        """
        self.bounds = None

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err

        Parameters
        ----------
        pars: array
            Array of parameters values
        fdiff: array
            the fdiff array to fill
        """

        if len(pars) != self.npars:
            raise ValueError(
                'pars size %d expected %d' % (len(pars), self.npars)
            )

        ngauss = self.ngauss

        index = 0

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1

        for i in range(ngauss):
            fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4 + i])
            index += 1

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            fdiff[index] = F_prior.get_lnprob_scalar(pars[4 + ngauss + i])
            index += 1

        # sky
        fdiff[index] = self.sky_prior.get_lnprob_scalar(
            # ngauss for T and flux, plus one for sky
            pars[4 + 2 * ngauss]
        )
        index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = np.sqrt(chi2)

        return index

    def sample(self, nrand=None):
        """
        Get random samples

        Parameters
        ----------
        nrand: int, optional
            Number of samples, default to a single set with size [npars].  If n
            is sent the result will have shape [n, npars]
        """

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        ngauss = self.ngauss
        samples = np.zeros((nrand, self.npars))

        cen1, cen2 = self.cen_prior.sample(nrand)
        g1, g2 = self.g_prior.sample2d(nrand)
        T = self.T_prior.sample(nrand)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T

        for i in range(ngauss):
            samples[:, 4 + i] += self.T_prior.sample(nrand)

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            samples[:, 4 + ngauss + i] = F_prior.sample(nrand)

        if is_scalar:
            samples = samples[0, :]
        return samples


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


class FitSkyCoellipFitter(ngmix.fitting.CoellipFitter):
    def _make_fit_model(self, obs, guess):
        return FitSkyCoellipFitModel(
            obs=obs,
            ngauss=self._ngauss,
            guess=guess,
            prior=self.prior,
        )


class FitSkyCoellipFitModel(ngmix.fitting.results.CoellipFitModel):
    def calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)

        # c1, c2, g1, g2, T1, ..TN, F1...FN, sky
        # sky = pars[4 + 2 * self._ngauss]
        sky = pars[-1]

        try:
            # all norms are set after fill
            self._fill_gmix_all(pars)

            start = self._fill_priors(pars=pars, fdiff=fdiff)

            for pixels, gm in zip(self._pixels_list, self._gmix_data_list):
                fill_fdiff_with_sky(gm, sky, pixels, fdiff, start)

                start += pixels.size

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff

    def set_fit_result(self, result):
        super().set_fit_result(result)
        self['sky'] = result['pars'][-1]
        self['sky_err'] = np.sqrt(result["pars_cov"][-1, -1])

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the fit parameter set, which
        definition depends on the sub-class

        Parameters
        ----------
        band: int, optional
            Band index, default 0
        """
        pars = self.get_band_pars(pars=self["pars"][:-1], band=band)
        return ngmix.gmix.make_gmix_model(pars, self.model)

    def _init_gmix_all(self, pars):
        super()._init_gmix_all(pars[:-1])

    def _fill_gmix_all(self, pars):
        super()._fill_gmix_all(pars[:-1])

    def _set_npars(self):
        """
        single band, npars determined from ngauss
        """
        # add one for sky
        self.npars = 4 + 2 * self._ngauss + 1

    def _set_n_prior_pars(self):
        # center1 + center2 + shape + T + fluxes
        if self.prior is None:
            self.n_prior_pars = 0
        else:
            # add one for sky
            ngauss = self._ngauss
            self.n_prior_pars = 1 + 1 + 1 + ngauss + ngauss + 1


def get_coellip_with_sky_prior(
    rng,
    ngauss,
    scale,
    # background,
    noise,
    T_range=None,
    F_range=None,
):
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

    sky_prior = ngmix.priors.Normal(
        mean=0,
        sigma=noise,
        # mean=0,
        # sigma=2 * abs(background),
        # mean=background,
        # sigma=0.1 * background,
        rng=rng,
    )

    prior = PriorCoellipWithSky(
        ngauss=ngauss,
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
        sky_prior=sky_prior,
    )

    return prior


class FitSkyCoellipPSFGuesser(ngmix.guessers.CoellipPSFGuesser):
    def __call__(self, obs):
        """
        Get a guess for the EM algorithm

        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument

        Returns
        -------
        guess: array
            The guess array, [cen1, cen2, g1, g2, T1, T2, ..., F1, F2, ...]
        """
        sky_guess = self.rng.normal(scale=0.01)

        guess0 = super()._get_guess(obs=obs)

        guess = np.zeros(guess0.size + 1)
        guess[:guess0.size] = guess0

        guess[-1] = sky_guess

        return guess


def get_coellip_runner_with_sky(
    rng,
    ngauss,
    # background,
    noise,
):
    # prior = None
    prior = get_coellip_with_sky_prior(
        ngauss=ngauss,
        rng=rng,
        scale=PIXEL_SCALE,
        # background=background,
        noise=noise,
    )

    fitter = FitSkyCoellipFitter(ngauss=ngauss, prior=prior)

    # guesser = ngmix.guessers.SimplePSFGuesser(
    #     rng=rng, guess_from_moms=True,
    # )
    guesser = FitSkyCoellipPSFGuesser(rng=rng, ngauss=ngauss)

    return ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )


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


def make_obj(flux):
    return galsim.Moffat(fwhm=FWHM, flux=flux, beta=3.5)
    # return galsim.Gaussian(fwhm=FWHM, flux=flux)


def make_image(rng, flux, noise):
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
    im += rng.normal(scale=noise, size=im.shape)
    return im


def make_obs(rng, flux, noise):
    im = make_image(rng=rng, flux=flux, noise=noise)

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

    # guesser = ngmix.guessers.SimplePSFGuesser(
    #     rng=rng, guess_from_moms=True,
    # )
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


def do_trial_avg(
    method,
    nobj,
    flux_min,
    flux_max,
    noise,
    background,
    rng,
    show=False,
):
    for i in range(nobj):
        tcat = make_cat()

        log_flux_true = rng.uniform(
            np.log10(flux_min),
            np.log10(flux_max),
        )

        tcat['flux_true'] = 10.0**log_flux_true
        tobs = make_obs(rng=rng, flux=tcat['flux_true'], noise=noise)

        with tobs.writeable():
            # tobs.image += background * PIXEL_SCALE**2
            tobs.image += background

        if i == 0:
            obstot = tobs
        else:
            with obstot.writeable():
                obstot.image += tobs.image
                obstot.weight += tobs.weight

    with obstot.writeable():
        obstot.image *= 1.0 / nobj

    if method == 'fitsky':
        runner = get_coellip_runner_with_sky(
            rng,
            ngauss=5,
            # background=background,
            noise=noise,
        )
        res = runner.go(obstot)
        bg = res['sky']
        bg_err = res['sky_err']
        bg_frac = bg * obstot.image.size / obstot.image.sum()
    elif method == 'submean':
        # runner = get_runner(rng, submean=True)
        runner = get_coellip_runner(rng, submean=True, ngauss=5)
        res = runner.go(obstot)
        bg, bg_err, bg_frac = fit_for_background(obs=obstot, res=res)
        # bg *= 1 / PIXEL_SCALE**2
        # bg_err *= 1 / PIXEL_SCALE**2
    elif method == 'em':
        try:
            bg, bg_err, bg_frac = do_em(obs=obstot, rng=rng, ngauss=3)
            # bg *= 1 / PIXEL_SCALE**2
            # bg_err *= 1 / PIXEL_SCALE**2
        except ngmix.GMixRangeError as err:
            print(str(err))
            return None
    else:
        raise ValueError(f'Bad method name "{method}"')

    # bg = measure_edge_bg(obstot.image, width=2)
    return {
        'bg': bg,
        'bg_err': bg_err,
        'bg_frac': bg_frac,
    }


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
        results['background'],
        get_err=True,
    )
    # bmean = results['background'].mean()
    # berr = results['background'].std() / np.sqrt(results.size)

    x = 0.5
    y = 0.9
    ax.text(
        x,
        y,
        f'mean: {bmean:.3g} +/- {berr:.3g}',
        transform=ax.transAxes,
    )

    ax.hist(
        results['background'],
        bins=50,
        color='C1',
        edgecolor='black',
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
        x,
        y,
        f'mean: {fdiff_mean:.3g} +/- {fdiff_err:.3g}',
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
        x,
        y,
        rf'bfrac mean: {bfrac_mean:.3g} +/- {bfrac_err:.3g}',
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


def main(
    method,
    ntrial,
    nper,
    flux_min,
    flux_max,
    noise,
    background,
    plotfile,
    seed,
    show,
):
    rng = np.random.RandomState(seed)

    dlist = []
    for i in trange(ntrial):
        tres = make_result()
        res = do_trial_avg(
            method=method,
            nobj=nper,
            flux_min=flux_min,
            flux_max=flux_max,
            noise=noise,
            background=background,
            rng=rng,
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
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', default='fitsky')
    parser.add_argument('--nper', type=int, default=100)
    parser.add_argument('--flux-min', type=float, default=320)
    parser.add_argument('--flux-max', type=float, default=3500)
    parser.add_argument('--noise', type=float, default=1)
    parser.add_argument(
        '--background',
        type=float,
        default=-0.3,
        help='background value',
    )
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
        noise=args.noise,
        background=args.background,
        plotfile=args.plotfile,
        show=args.show,
    )
