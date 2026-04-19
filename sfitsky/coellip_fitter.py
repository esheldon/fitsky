import numpy as np
from numba import njit

import ngmix
from ngmix.defaults import LOWVAL
from ngmix import GMixRangeError
from ngmix.gmix.gmix_nb import (
    gmix_set_norms,
    gmix_eval_pixel_fast,
)


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
    cen_prior_sigma,
    sky_prior_mean,
    sky_prior_sigma,
    T_range=None,
    F_range=None,
):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    cen_prior_sigma: float
        Sigma for gaussian center prior
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
        sigma1=cen_prior_sigma,
        sigma2=cen_prior_sigma,
        # sigma1=scale * 0.01,
        # sigma2=scale * 0.01,
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
        mean=sky_prior_mean,
        sigma=sky_prior_sigma,
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
    def __init__(self, rng, ngauss, sky_prior, guess_from_moms=False):
        super().__init__(
            rng=rng,
            ngauss=ngauss,
            guess_from_moms=guess_from_moms,
        )
        self.sky_prior = sky_prior
        # self.npars = get_coellip_npars(ngauss) + 1

    def __call__(self, obs):
        """
        Get a guess for the coelliptical PSF and sky

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
        sky_guess = self.sky_prior.sample()

        guess0 = super()._get_guess(obs=obs)

        guess = np.zeros(guess0.size + 1)
        guess[:guess0.size] = guess0

        guess[-1] = sky_guess

        return guess


def get_coellip_runner_with_sky(
    rng,
    ngauss,
    cen_prior_sigma,
    sky_prior_mean,
    sky_prior_sigma,
):
    """
    Get the ngmix runner for fitting coelliptical gaussians with a sky.

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    ngauss: int
        The number of gaussians
    cen_prior_sigma: float
        Sigma for gaussian center prior
    sky_prior_mean: float
        The mean of the prior on the sky.
    sky_prior_sigma: float
        Sigma of prior on sky.
    """
    # prior = None
    prior = get_coellip_with_sky_prior(
        ngauss=ngauss,
        rng=rng,
        cen_prior_sigma=cen_prior_sigma,
        sky_prior_mean=sky_prior_mean,
        sky_prior_sigma=sky_prior_sigma,
    )

    fitter = FitSkyCoellipFitter(ngauss=ngauss, prior=prior)
    guesser = FitSkyCoellipPSFGuesser(
        rng=rng,
        ngauss=ngauss,
        sky_prior=prior.sky_prior,
    )

    return ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
