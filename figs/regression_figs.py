import os
import sys
import json
import pickle

import numpy as np
from scipy.stats.stats import linregress
from matplotlib import pyplot as plt
import arviz as az

from figs.regr_fig_utils import plot_regressions, plot_regression_from_trace
import bayes_models

sys.modules['linear_regression_errors'] = bayes_models


def standardize(arr):
    arr = (arr - arr.mean()) / arr.std()
    return arr


def magnitude(arr):
    mag = np.floor(np.log10(arr))
    return mag


def plot_saved_traces(impacts_json, trc_dir, fig_dir, cc_err, qres_err, overwrite=False):
    with open(impacts_json, 'r') as f:
        stations = json.load(f)

    lp_cc, lp_qres = 1, 1

    for station, data in stations.items():

        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]

        for period in impact_keys:

            records = data[period]

            cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
            qres = np.array(records['q_resid'])
            years_ = [x for x in range(1991, 2021)]

            cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
            qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001

            qres_err = qres_err * np.ones_like(qres)
            cc_err = cc_err * np.ones_like(cc)

            years = (np.linspace(0, 1, len(qres)) + 0.001).reshape(1, -1)
            dummy_error = np.zeros_like(years)

            qres_cc_lr = linregress(qres, cc)
            time_cc_lr = linregress(years_, cc)
            time_qres_lr = linregress(years_, qres)

            tccp, tqrp = time_cc_lr.pvalue, time_qres_lr.pvalue

            # print(station, period)
            # if tccp < lp_cc:
            #     print('cc qres p: {:.3f}'.format(qres_cc_lr.pvalue))
            #     print('cc trend p: {:.3f}'.format(lp_cc))
            #     print('qres trend p: {:.3f}\n'.format(lp_qres))

            regression_combs = [(cc, qres, cc_err, qres_err, qres_cc_lr, 'cc', 'qres'),
                                (years, cc, dummy_error, cc_err, time_cc_lr, 'years', 'cc'),
                                (years, qres, dummy_error, qres_err, time_qres_lr, 'years', 'qres')]

            trc_subdirs = ['cc_qres', 'time_cc', 'time_qres']

            for subdir in trc_subdirs:
                model_dir = os.path.join(fig_dir, subdir)
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)

            for (x, y, x_err, y_err, lr, varstr_x, varstr_y), subdir in zip(regression_combs,
                                                                            trc_subdirs):
                if subdir == 'time_cc':
                    y = y[0, :]

                desc = [station, data['STANAME'], varstr_x, period, varstr_y, records['q_window']]
                fig_file = os.path.join(fig_dir, subdir,
                                        '{}_{}_{}_{}_{}_{}.png'.format(*desc))

                if os.path.exists(fig_file) and not overwrite:
                    print(fig_file, ' exists, skipping')
                    continue

                saved_model = os.path.join(trc_dir, subdir, '{}_cc_{}_q_{}.model'.format(station,
                                                                                         period,
                                                                                         records['q_window']))

                if not os.path.exists(saved_model):
                    print(saved_model, ' missing, skipping')
                    continue

                plot_trace(x, y, x_err, y_err, saved_model, lr, os.path.join(fig_dir, subdir), desc)


def plot_trace(x, y, x_err, y_err, model, ols, fig_dir, desc_str, fig_file=None):
    try:
        with open(model, 'rb') as buff:
            data = pickle.load(buff)
            model, traces = data['model'], data['trace']
            az.plot_trace(traces, var_names=['slope'], rug=True)

    except (EOFError, ValueError):
        print(model, 'error')
        return None

    if not fig_file:
        fig_file = os.path.join(fig_dir,
                                '{}_{}_{}_{}_{}_{}.png'.format(*desc_str))

    plt.savefig(fig_file.replace('.png', '_trace.png'))
    plt.close()

    plot_regressions(x[0], y, x_err[0], y_err)

    plt.scatter(x, y)
    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])

    beta_ = plot_regression_from_trace(model, (x, y, x_err, y_err),
                                       ax=plt.gca(), chains=4, traces=traces)
    plt.xlabel(desc_str[2])
    plt.ylabel(desc_str[4])
    # print('beta: {}, mean trace: {}, divergence rate: {}'.format(beta_, np.mean(betas),
    #                                                              div_))

    plt.suptitle(' '.join(desc_str))
    plt.savefig(fig_file)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    cc_err_ = '0.233'
    qres_err_ = '0.17'

    for var in ['cci']:
        state = 'ccerr_{}_qreserr_{}'.format(str(cc_err_), str(qres_err_))
        trace_dir = os.path.join(root, 'gages', 'bayes', 'traces', state)
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        _json = os.path.join(root, 'gages', 'station_metadata',
                             '{}_impacted.json'.format(var))

        o_fig = os.path.join(root, 'gages', 'figures', 'slope_trace_{}'.format(var), state)

        plot_saved_traces(_json, trace_dir, o_fig, qres_err=float(qres_err_) / 2,
                          cc_err=float(cc_err_) / 2, overwrite=True)

# ========================= EOF ====================================================================
