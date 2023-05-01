import os
import sys
import pickle
import tempfile
import json

import aesara
import numpy as np
import pymc as pm
import pymc.sampling_jax

aesara.config.blas__ldflags = f'"-L{os.path.join(sys.prefix, "Library", "bin")}" -lmkl_core -lmkl_intel_thread -lmkl_rt'

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
            mu = intercept + gradient * x
            likelihood = pm.Normal('y', mu=mu, sigma=y_err, observed=y)
            trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': trace}, buff)
                    print('saving', save_model)

                var_names = ['slope', 'inter']
                trace_only(save_model, figure, var_names)

        os.rmdir(self.dirpath)


class TimeTrendModel():

    def __init__(self):
        self.dirpath = tempfile.mkdtemp()
        os.environ['AESARA_FLAGS'] = "base_compiledir=${}/.aesara".format(self.dirpath)

    def fit(self, x, y, y_err, save_model=None, figure=None):
        with pm.Model():
            intercept = pm.Normal('inter', 0, sigma=20)
            gradient = pm.Normal('slope', 0, sigma=20)
            mu = intercept + gradient * x
            likelihood = pm.Normal('y', mu=mu, sigma=y_err, observed=y)
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
        with pm.Model():
            intercept = pm.Normal('inter', 0, sigma=20)
            gradient_1 = pm.Normal(var_names['x1_name'], 0, sigma=20)
            gradient_2 = pm.Normal(var_names['x2_name'], 0, sigma=20)

            true_x1 = pm.Normal('true_x1', mu=0, sigma=20, shape=len(x1))
            true_x2 = pm.Normal('true_x2', mu=0, sigma=20, shape=len(x2))

            likelihood_x1 = pm.Normal('x1', mu=true_x1, sigma=x1_err, observed=x1)
            likelihood_x2 = pm.Normal('x2', mu=true_x2, sigma=x2_err, observed=x2)

            mu = intercept + gradient_1 * true_x1 + gradient_2 * true_x2
            likelihood_y = pm.Normal('y', mu=mu, sigma=y_error, observed=y)

            trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': trace}, buff)
                    print('saving', save_model)

                var_names = [v for k, v in var_names.items()] + ['inter']
                trace_only(save_model, figure, var_names)

        os.rmdir(self.dirpath)


class TriVarLinearModel():

    def __init__(self):
        self.dirpath = tempfile.mkdtemp()
        os.environ['AESARA_FLAGS'] = "base_compiledir=${}/.aesara".format(self.dirpath)

    def fit(self, x1, x1_err, x2, x2_err, x3, x3_err, y, y_error, save_model=None, var_names=None, figure=None):
        with pm.Model():
            intercept = pm.Normal('inter', 0, sigma=20)
            gradient_1 = pm.Normal(var_names['x1_name'], 0, sigma=20)
            gradient_2 = pm.Normal(var_names['x2_name'], 0, sigma=20)
            gradient_3 = pm.Normal(var_names['x3_name'], 0, sigma=20)

            true_x1 = pm.Normal('true_x1', mu=0, sigma=20, shape=len(x1))
            true_x2 = pm.Normal('true_x2', mu=0, sigma=20, shape=len(x2))
            true_x3 = pm.Normal('true_x3', mu=0, sigma=20, shape=len(x3))

            likelihood_x1 = pm.Normal('x1', mu=true_x1, sigma=x1_err, observed=x1)
            likelihood_x2 = pm.Normal('x2', mu=true_x2, sigma=x2_err, observed=x2)
            likelihood_x3 = pm.Normal('x2', mu=true_x3, sigma=x3_err, observed=x3)

            mu = intercept + gradient_1 * true_x1 + gradient_2 * true_x2 + gradient_3 * true_x3
            likelihood_y = pm.Normal('y', mu=mu, sigma=y_error, observed=y)

            trace = pm.sampling_jax.sample_numpyro_nuts(**DEFAULTS)

            if save_model:
                with open(save_model, 'wb') as buff:
                    pickle.dump({'model': self, 'trace': trace}, buff)
                    print('saving', save_model)

                var_names = [v for k, v in var_names.items()] + ['inter']
                trace_only(save_model, figure, var_names)

        os.rmdir(self.dirpath)


def mv_test():
    for m in range(4, 11):
        f = '/media/research/IrrigationGIS/impacts/mv_traces/mv_trends/time_cc/data/09371010_q_{}.data'.format(m)
        with open(f, 'r') as f_obj:
            d = json.load(f_obj)
        model_ = '/media/research/IrrigationGIS/impacts/mv_traces/mv_trends/09371010_q_{}.model'.format(m)
        trace_ = '/media/research/IrrigationGIS/impacts/mv_traces/mv_trends/09371010_q_{}_trace.png'.format(m)
        lm = BiVarLinearModel()
        variable_names = {'x1_name': 'time_coeff',
                          'x2_name': 'cwd_coeff'}
        lm.fit(d['years'], np.zeros(len(d['years'])) + 0.001,
               d['x'], d['x_err'], d['y'], d['y_err'], save_model=model_, figure=trace_,
               var_names=variable_names)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
