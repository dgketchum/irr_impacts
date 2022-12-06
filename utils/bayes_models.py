import pickle

import pymc3 as pm
from astroML.linear_model import LinearRegression


class LinearModel(LinearRegression):

    def __init__(self, fit_intercept=False, regularization='none', kwds=None):
        super().__init__(fit_intercept, regularization, kwds)
        self.clf_ = None
        self.trace = None

    def fit(self, x, y, y_error=1, x_error=None, save_model=None,
            sample_kwargs=None):
        with pm.Model():
            intercept = pm.Normal('inter', 0, sd=20)
            gradient = pm.Normal('slope', 0, sd=20)
            true_x = pm.Normal('true_x', mu=0, sd=20, shape=len(x))
            likelihood_x = pm.Normal('x', mu=true_x, sd=x_error, observed=x)
            true_y = pm.Deterministic('true_y', intercept + gradient * true_x)
            likelihood_y = pm.Normal('y', mu=true_y, sd=y_error, observed=y)

            self.trace = pm.sample(**sample_kwargs)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': self.trace}, buff)
                    print('saving', save_model)


class MVLinearModel():

    def __init__(self):
        pass

    def fit(self, x1, x2, y, y_error=1, x1_error=None, x2_error=None, save_model=None,
            sample_kwargs=None):
        with pm.Model():
            alpha = pm.Normal('inter', 0, sd=20)
            slope_1 = pm.Normal('slope_1', 0, sd=20)
            slope_2 = pm.Normal('slope_2', 0, sd=20)

            x1_ = pm.Normal('x_1', mu=0, sd=x1_error)
            x2_ = pm.Normal('x_2', mu=0, sd=x2_error)

            x1_hat = pm.Normal('x1_hat', mu=x1_, sd=x1_error, observed=x1)
            x2_hat = pm.Normal('x2_hat', mu=x2_, sd=x2_error, observed=x2)

            mu = alpha + slope_1 * x1_hat + slope_2 * x2_hat
            y_hat = pm.Normal('y_hat', mu=mu, sd=y_error, observed=y)

            self.trace = pm.sample(**sample_kwargs)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': self.trace}, buff)
                    print('saving', save_model)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
