import os
import json

import numpy as np
import pymc3 as pm
import arviz as az
from sklearn.base import BaseEstimator
from scipy.stats.stats import linregress
from matplotlib import pyplot as plt


def linear_regression(x, y, lr, prior_slope):
    with pm.Model() as model:
        slope = pm.Normal("slope", mu=lr.slope, sigma=prior_slope)
        intercept = pm.Normal("intercept", mu=lr.intercept, sigma=lr.intercept_stderr)
        sigma = pm.HalfNormal("σ", sd=1)
        y = pm.Normal("y", mu=slope * x + intercept, sigma=sigma, observed=y)

    return model


class LinearRegression(BaseEstimator):
    """Simple Linear Regression with errors in y
    This is a stripped-down version of sklearn.linear_model.LinearRegression
    which can correctly accounts for errors in the y variable
    Parameters
    ----------
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data
    regularization : string (optional)
        ['l1'|'l2'|'none'] Use L1 (Lasso) or L2 (Ridge) regression
    kwds: dict
        additional keyword arguments passed to sklearn estimators:
        LinearRegression, Lasso (L1), or Ridge (L2)
    Notes
    -----
    This implementation may be compared to that in
    sklearn.linear_model.LinearRegression.
    The difference is that here errors are
    """
    _regressors = {'none': LinearRegression,
                   'l1': Lasso,
                   'l2': Ridge}

    def __init__(self, fit_intercept=True, regularization='none', kwds=None):
        if regularization.lower() not in ['l1', 'l2', 'none']:
            raise ValueError("regularization='{}' not recognized"
                             "".format(regularization))
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.kwds = kwds

    def _transform_X(self, X):
        X = np.asarray(X)
        if self.fit_intercept:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        return X

    @staticmethod
    def _scale_by_error(X, y, y_error=1):
        """Scale regression by error on y"""
        X = np.atleast_2d(X)
        y = np.asarray(y)
        y_error = np.asarray(y_error)

        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        if y_error.ndim == 0:
            return X / y_error, y / y_error

        elif y_error.ndim == 1:
            assert y_error.shape == y.shape
            X_out, y_out = X / y_error[:, None], y / y_error

        elif y_error.ndim == 2:
            assert y_error.shape == (y.size, y.size)
            evals, evecs = np.linalg.eigh(y_error)
            X_out = np.dot(evecs * (evals ** -0.5),
                           np.dot(evecs.T, X))
            y_out = np.dot(evecs * (evals ** -0.5),
                           np.dot(evecs.T, y))
        else:
            raise ValueError("shape of y_error does not match that of y")

        return X_out, y_out

    def _choose_regressor(self):
        model = self._regressors.get(self.regularization.lower(), None)
        if model is None:
            raise ValueError("regularization='{}' unrecognized"
                             "".format(self.regularization))
        return model

    def fit(self, X, y, y_error=1):
        kwds = {}
        if self.kwds is not None:
            kwds.update(self.kwds)
        kwds['fit_intercept'] = False

        model = self._choose_regressor()
        self.clf_ = model(**kwds)

        X = self._transform_X(X)
        X, y = self._scale_by_error(X, y, y_error)

        self.clf_.fit(X, y)
        return self

    def predict(self, X):
        X = self._transform_X(X)
        return self.clf_.predict(X)

    @property
    def coef_(self):
        return self.clf_.coef_


class LinearRegressionwithErrors(LinearRegression):
    """ Credit to astroML {https://github.com/astroML/astroML/blob/main/astroML/
    linear_model/linear_regression_errors.py}"""

    def __init__(self, fit_intercept=False, regularization='none', kwds=None):
        super().__init__(fit_intercept, regularization, kwds)

    def fit(self, X, y, y_error=1, x_error=None, *,
            sample_kwargs={'draws': 1000, 'target_accept': 0.9}):

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
            int_std = pm.HalfFlat('int_std')
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
            x = pm.Normal('xi', mu=ksi.T, sigma=x_error, observed=X, shape=X.shape)  # noqa: F841
            y = pm.Normal('yi', mu=eta, sigma=y_error, observed=y, shape=y.shape)

            self.trace = pm.sample(**sample_kwargs)

            # TODO: make it optional to choose a way to define best

            HND, edges = np.histogramdd(np.hstack((self.trace['slope'],
                                                   self.trace['inter'][:, None])), bins=50)

            w = np.where(HND == HND.max())

            # choose the maximum posterior slope and intercept
            slope_best = [edges[i][w[i][0]] for i in range(len(edges) - 1)]
            intercept_best = edges[-1][w[-1][0]]
            self.clf_.coef_ = np.array([intercept_best, *slope_best])

        return self


def estimate_parameter_distributions(impacts_json):
    with open(impacts_json, 'r') as f:
        stations = json.load(f)

    for station, data in stations.items():
        if station != '06329500':
            continue
        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]
        for period in impact_keys:
            records = data[period]

            ai, q = records['ai_data'], records['q_data']
            q = np.log10(q)
            q_clim_sigma_slope = np.sqrt((max(ai) - min(ai)) / (max(q) - min(q)))
            q_clim_lr = linregress(q, ai)
            q_clim = linear_regression(ai, q, q_clim_lr, q_clim_sigma_slope)
            with q_clim:
                q_clim_fit = pm.sample(return_inferencedata=True)

            cc, q_res = records['cc_data'], records['q_resid']
            q_res, cc = list(np.array(q_res) / 1e9), list(np.array(cc) / 1e9)
            qres_cc_sigma_slope = np.sqrt((max(cc) - min(cc)) / (max(q_res) - min(q_res)))
            qres_cc_lr = linregress(q_res, cc)
            qres_cc = linear_regression(cc, q_res, qres_cc_lr, qres_cc_sigma_slope)
            with qres_cc:
                qres_cc_fit = pm.sample(return_inferencedata=True)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex='col')
            az.plot_posterior(q_clim_fit, var_names=["slope"], ref_val=q_clim_lr.slope, ax=ax[0])
            ax[0].set(title="Flow - Climate", xlabel="slope")

            az.plot_posterior(qres_cc_fit, var_names=["slope"], ref_val=qres_cc_lr.slope, ax=ax[1])
            ax[1].set(title="Residual Flow - Crop Consumption", xlabel="slope")

            desc_str = '{} {}\n' \
                       '{} months climate'.format(station, data['STANAME'], records['lag'])
            plt.suptitle(desc_str)
            plt.savefig('/home/dgketchum/Downloads/{}_slope_dist.png'.format(station))
            plt.close()
            exit(0)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'
    _json = os.path.join(root, 'station_metadata/cc_impacted.json')
    estimate_parameter_distributions(_json)
# ========================= EOF ====================================================================
