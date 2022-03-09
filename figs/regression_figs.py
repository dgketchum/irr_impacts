import os
import json
import pickle

import numpy as np
from scipy.stats.stats import linregress
from matplotlib import pyplot as plt
from astroML.plotting import plot_regressions, plot_regression_from_trace
from astroML.linear_model import LinearRegressionwithErrors


def plot_saved_traces(impacts_json, trc_dir, fig_dir, cc_err, qres_err):
    with open(impacts_json, 'r') as f:
        stations = json.load(f)

    for station, data in stations.items():

        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]

        for period in impact_keys:
            records = data[period]
            cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
            qres = np.array(records['q_resid'])

            cc = (cc - cc.min()) / (cc.max() - cc.min())
            qres = (qres - qres.min()) / (qres.max() - qres.min())
            qres_cc_lr = linregress(qres, cc)
            qres_err = qres_err * np.ones_like(qres)
            cc_err = cc_err * np.ones_like(cc)
            saved_model = os.path.join(trc_dir, '{}_cc{}_q{}.model'.format(station,
                                                                           period,
                                                                           records['q_window']))
            desc = [station, data['STANAME'], 'cc', period, 'q', records['q_window']]

            plot_trace(cc, qres, cc_err, qres_err, saved_model, qres_cc_lr, fig_dir, desc)


def plot_trace(cc, qres, cc_err, qres_err, model, qres_cc_lr, fig_dir,
               desc_str=''):
    plot_regressions(0.0, 0.0, cc[0], qres,
                     cc_err[0], qres_err,
                     add_regression_lines=True,
                     alpha_in=qres_cc_lr.intercept, beta_in=qres_cc_lr.slope)

    plt.scatter(cc, qres)
    plt.xlim([cc.min(), cc.max()])
    plt.ylim([qres.min(), qres.max()])

    if isinstance(model, LinearRegressionwithErrors):
        plot_regression_from_trace(model, (cc, qres, cc_err, qres_err),
                                   ax=plt.gca(), chains=50, traces=None)
    else:
        with open(model, 'rb') as buff:
            data = pickle.load(buff)
            model, trace = data['model'], data['trace']

        plot_regression_from_trace(model, (cc, qres, cc_err, qres_err),
                                   ax=plt.gca(), chains=50, traces=trace)

    plt.suptitle(' '.join(desc_str))
    fig_file = os.path.join(fig_dir, '{}_{}_cc_{}_q_{}.png'.format(desc_str[0], desc_str[1], desc_str[3], desc_str[5]))
    plt.savefig(fig_file)
    plt.close()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    for var in ['cci']:
        state = 'ccerr_0.20_qreserr_0.17'
        trace_dir = os.path.join(root, 'gages', 'bayes', 'traces', state)
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        _json = os.path.join(root, 'gages', 'station_metadata', '{}_impacted.json'.format(var))
        o_fig = os.path.join(root, 'gages', 'figures', 'slope_trace_{}'.format(var), state)
        plot_saved_traces(_json, trace_dir, o_fig, qres_err=0.17, cc_err=0.20)
# ========================= EOF ====================================================================
