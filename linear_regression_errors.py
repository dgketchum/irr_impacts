import numpy as np
import warnings
import pickle

try:
    import pymc3 as pm
    import theano.tensor as tt
    from packaging.version import Version

    PYMC_LT_39 = Version(pm.__version__) < Version("3.9")
except ImportError:
    warnings.warn('LinearRegressionwithErrors requires PyMC3 to be installed')
    PYMC_LT_39 = True

from astroML.linear_model import LinearRegression

__all__ = ['LinearRegressionwithErrors']


class LinearRegressionwithErrors(LinearRegression):

    def __init__(self, fit_intercept=False, regularization='none', kwds=None):
        super().__init__(fit_intercept, regularization, kwds)
        self.clf_ = None
        self.trace = None

    def fit(self, X, y, y_error=1, x_error=None, *,
            sample_kwargs={'draws': 1000, 'target_accept': 0.9}, save_model=None):
        if not PYMC_LT_39:
            sample_kwargs['return_inferencedata'] = False

        sample_kwargs['progressbar'] = False
        kwds = {}
        if self.kwds is not None:
            kwds.update(self.kwds)
        kwds['fit_intercept'] = False
        model = self._choose_regressor()
        self.clf_ = model(**kwds)

        self.fit_intercept = False

        if x_error is not None:
            x_error = np.atleast_2d(x_error)

        with pm.Model():
            # slope and intercept of eta-ksi relation
            slope = pm.Flat('slope', shape=(X.shape[0],))
            inter = pm.Flat('inter')

            # intrinsic scatter of eta-ksi relation
            # int_std = pm.HalfFlat('int_std')
            int_std = 0.001
            # standard deviation of Gaussian that ksi are drawn from (assumed mean zero)
            tau = pm.HalfFlat('tau', shape=(X.shape[0],))
            # intrinsic ksi
            mu = pm.Normal('mu', mu=0, sigma=tau, shape=(X.shape[0],))

            # Some wizzarding with the dimensions all around.
            ksi = pm.Normal('ksi', mu=mu, tau=tau, shape=X.T.shape)

            # intrinsic eta-ksi linear relation + intrinsic scatter
            eta = pm.Normal('eta', mu=(tt.dot(slope.T, ksi.T) + inter),
                            sigma=int_std, shape=y.shape)

            # observed xi, yi
            if x_error is None:
                x = pm.Normal('xi', mu=ksi.T, observed=X, shape=X.shape)

            else:
                x = pm.Normal('xi', mu=ksi.T, sigma=x_error, observed=X, shape=X.shape)

            y = pm.Normal('yi', mu=eta, sigma=y_error, observed=y, shape=y.shape)

            self.trace = pm.sample(**sample_kwargs)

            # TODO: make it optional to choose a way to define best
            slopes = self.trace['slope']
            # fix burn-in and why does this give [4000, 30]?
            if slopes.shape[1] > 1:
                slopes = slopes.mean(axis=1).reshape((slopes.shape[0], 1))
            HND, edges = np.histogramdd(np.hstack((slopes, self.trace['inter'][:, None])), bins=50)

            w = np.where(HND == HND.max())

            # choose the maximum posterior slope and intercept
            slope_best = [edges[i][w[i][0]] for i in range(len(edges) - 1)]
            intercept_best = edges[-1][w[-1][0]]
            self.clf_.coef_ = np.array([intercept_best, *slope_best])

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': self.trace}, buff)

        return self


class LinearModel(LinearRegression):

    def __init__(self, fit_intercept=False, regularization='none', kwds=None):
        super().__init__(fit_intercept, regularization, kwds)
        self.clf_ = None
        self.trace = None

    def fit(self, X, y, y_error=1, x_error=None, save_model=None,
            sample_kwargs=None):
        sample_kwargs['progressbar'] = False
        kwds = {}
        if self.kwds is not None:
            kwds.update(self.kwds)
        kwds['fit_intercept'] = False
        model = self._choose_regressor()
        self.clf_ = model(**kwds)

        if sample_kwargs is None:
            sample_kwargs = {'draws': 1000, 'target_accept': 0.9}

        with pm.Model():
            # slope and intercept of eta-ksi relation
            b0 = pm.Normal("inter", mu=0, sigma=10)
            b1 = pm.Normal("slope", mu=0, sigma=10)

            y_est = b0 + b1 * X

            likelihood = pm.Normal("likelihood", mu=y_est, sigma=y_error, observed=y)
            self.trace = pm.sample(**sample_kwargs)

            slopes = np.array(self.trace['posterior']['slope']).ravel()
            intercepts = np.array(self.trace['posterior']['inter']).ravel()

            HND, edges = np.histogramdd(np.hstack((slopes, intercepts)), bins=50)

            w = np.where(HND == HND.max())

            # choose the maximum posterior slope and intercept
            slope_best = [edges[i][w[i][0]] for i in range(len(edges) - 1)]
            intercept_best = edges[-1][w[-1][0]]
            self.clf_.coef_ = np.array([intercept_best, *slope_best])

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': self.trace}, buff)

        return self

