import os
import json
import pickle

import numpy as np
from scipy.stats.stats import linregress
from matplotlib import pyplot as plt
from astroML.plotting import plot_regressions, plot_regression_from_trace
import arviz as az
from astroML.linear_model import LinearRegressionwithErrors


def plot_saved_traces(impacts_json, trc_dir, fig_dir, cc_err, qres_err, overwrite=False):
    with open(impacts_json, 'r') as f:
        stations = json.load(f)

    for station, data in stations.items():

        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]

        for period in impact_keys:
            # if station != '06025500' or period != '10-10':
            #     continue
            # if station != '06016000' or period != '5-7':
            #     continue
            records = data[period]
            desc = [station, data['STANAME'], 'cc', period, 'q', records['q_window']]
            fig_file = os.path.join(fig_dir, '{}_{}_cc_{}_q_{}.png'.format(desc[0], desc[1],
                                                                            desc[3], desc[5]))
            if os.path.exists(fig_file) and not overwrite:
                print(fig_file, ' exists, skipping')
                continue

            cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
            qres = np.array(records['q_resid'])

            cc = (cc - cc.min()) / (cc.max() - cc.min())
            qres = (qres - qres.min()) / (qres.max() - qres.min())
            qres_cc_lr = linregress(qres, cc)
            qres_err = qres_err * np.ones_like(qres)
            cc_err = cc_err * np.ones_like(cc)
            saved_model = os.path.join(trc_dir, '{}_cc_{}_q_{}.model'.format(station,
                                                                           period,
                                                                           records['q_window']))
            if not os.path.exists(saved_model):
                continue

            plot_trace(cc, qres, cc_err, qres_err, saved_model, qres_cc_lr, fig_dir, desc)


def plot_trace(cc, qres, cc_err, qres_err, model, qres_cc_lr, fig_dir,
               desc_str='', fig_file=None):

    try:
        with open(model, 'rb') as buff:
            data = pickle.load(buff)
            model, traces = data['model'], data['trace']
            betas = [trace['slope'][0] for trace in traces]
            az.plot_trace(traces, var_names=['slope'], rug=True)
            div_ = float(np.count_nonzero(traces.diverging)) / np.count_nonzero(~traces.diverging)

    except EOFError:
        print(model, 'error')
        return None

    if div_ < 0.05:
        subdir = 'converged'
    else:
        subdir = 'not_converged'

    if not fig_file:
        fig_file = os.path.join(fig_dir, subdir,
                                '{}_{}_cc_{}_q_{}.png'.format(desc_str[0],
                                                               desc_str[1],
                                                               desc_str[3],
                                                               desc_str[5]))
    plt.savefig(fig_file.replace('.png', '_trace.png'))
    plt.close()

    plot_regressions(0.0, 0.0, cc[0], qres,
                     cc_err[0], qres_err,
                     add_regression_lines=True,
                     alpha_in=qres_cc_lr.intercept, beta_in=qres_cc_lr.slope)

    plt.scatter(cc, qres)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

    beta_ = plot_regression_from_trace(model, (cc, qres, cc_err, qres_err),
                                       ax=plt.gca(), chains=50, traces=traces)

    print('beta: {}, mean trace: {}, divergence rate: {}'.format(beta_, np.mean(betas),
                                                                 div_))

    print(fig_file, '\n')

    plt.suptitle(' '.join(desc_str))
    plt.savefig(fig_file)
    # plt.show()
    # exit()
    plt.close()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    for var in ['cci']:
        state = 'ccerr_0.40_qreserr_0.17'
        trace_dir = os.path.join(root, 'gages', 'bayes', 'traces', state, 'cc_qres')
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        _json = os.path.join(root, 'gages', 'station_metadata',
                             '{}_impacted.json'.format(var))

        o_fig = os.path.join(root, 'gages', 'figures', 'slope_trace_{}'.format(var), state)

        plot_saved_traces(_json, trace_dir, o_fig, qres_err=0.40, cc_err=0.18, overwrite=True)

# ========================= EOF ====================================================================
