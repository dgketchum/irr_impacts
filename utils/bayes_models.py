import os
import pickle
import tempfile

import pymc as pm
import pymc.sampling_jax

from figs.trace_figs import trace_only

DEFAULTS = {'draws': 1000,
            'tune': 5000,
            'chains': 4,
            'progress_bar': False}


class UniLinearModelErrY():

    def __init__(self):
        self.dirpath = tempfile.mkdtemp()
        os.environ['AESARA_FLAGS'] = "base_compiledir=${}/.aesara".format(self.dirpath)

    def fit(self, x, y, y_err, save_model=None, figure=None):
        with pm.Model():
            intercept = pm.Normal('inter', 0, sigma=20)
            gradient = pm.Normal('slope', 0, sigma=20)
            true_y = pm.Deterministic('true_y', intercept + gradient * x)
            likelihood = pm.Normal('y', mu=true_y, sigma=y_err, observed=y)
            trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': trace}, buff)
                    print('saving', save_model)

                var_names = ['slope', 'inter']
                trace_only(save_model, figure, var_names)

        os.rmdir(self.dirpath)


class BiVarLinearModel():

    def __init__(self):
        self.dirpath = tempfile.mkdtemp()
        os.environ['AESARA_FLAGS'] = "base_compiledir=${}/.aesara".format(self.dirpath)

    def fit(self, x1, x1_err, x2, x2_err, y, y_error, save_model=None, var_names=None, figure=None):

        if not var_names:
            var_names = {'x1_name': 'slope_1',
                         'x2_name': 'slope_2'}

        with pm.Model():

            intercept = pm.Normal('inter', 0, sigma=20)
            gradient_1 = pm.Normal(var_names['x1_name'], 0, sigma=20)
            gradient_2 = pm.Normal(var_names['x2_name'], 0, sigma=20)

            true_x1 = pm.Normal('true_x1', mu=0, sigma=20, shape=len(x1))
            true_x2 = pm.Normal('true_x2', mu=0, sigma=20, shape=len(x2))

            likelihood_x1 = pm.Normal('x1', mu=true_x1, sigma=x1_err, observed=x1)
            likelihood_x2 = pm.Normal('x2', mu=true_x2, sigma=x2_err, observed=x2)

            true_y = pm.Deterministic('true_y', intercept + gradient_1 * true_x1 + gradient_2 * true_x2)
            likelihood_y = pm.Normal('y', mu=true_y, sigma=y_error, observed=y)

            trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': trace}, buff)
                    print('saving', save_model)

                var_names = [v for k, v in var_names.items()] + ['inter']
                trace_only(save_model, figure, var_names)

        os.rmdir(self.dirpath)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
