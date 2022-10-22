import pickle

import pymc3 as pm


class LinearModel:

    def __init__(self, fit_intercept=False, regularization='none', kwds=None):
        super().__init__(fit_intercept, regularization, kwds)
        self.trace = None

    def fit(self, x, y, y_error=1, x_error=None, save_model=None,
            sample_kwargs=None):
        with pm.Model() as model:
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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
