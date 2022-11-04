import os
import json
import pickle

import numpy as np
from scipy.stats import linregress
import arviz as az
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_saved_traces(metadata, trc_dir, fig_dir, overwrite=False, selected=None):
    with open(metadata, 'r') as f_obj:
        metadata = json.load(f_obj)

    trc_files = [os.path.join(trc_dir, x) for x in os.listdir(trc_dir) if x.endswith('.model')]
    data_files = [trc.replace('.model', '.data') for trc in trc_files]
    stations = [os.path.basename(x).split('_')[0] for x in trc_files]

    for sid, model, data in zip(stations, trc_files, data_files):

        if selected and sid not in selected:
            continue

        with open(data, 'r') as f_obj:
            dct = json.load(f_obj)
        x, y, y_err = np.array(dct['x']), np.array(dct['y']), np.array(dct['y_err'])

        if dct['x_err']:
            x_err = np.array(dct['x_err'])
        else:
            x_err = None

        info_ = metadata[sid]

        if os.path.basename(trc_dir) in ['cc_qres', 'ccres_qres']:
            per = os.path.basename(model).split('_')[2]
            desc = [sid, info_['STANAME'], dct['xvar'], per, dct['yvar']]
        else:
            per = os.path.basename(model).split('.')[0].split('_')[-1]
            desc = [sid, info_['STANAME'], dct['xvar'], per, dct['yvar']]

        plot_trace(x, y, x_err, y_err, model, fig_dir, desc, overwrite)


def plot_trace(x, y, x_err, y_err, model, fig_dir, desc_str, overwrite=False):
    fig_file = os.path.join(fig_dir, '{}.png'.format('_'.join(desc_str)))

    if not overwrite and os.path.exists(fig_file):
        return

    try:
        with open(model, 'rb') as buff:
            data = pickle.load(buff)
            model, traces = data['model'], data['trace']
            vars_ = ['slope', 'inter']
            az.plot_trace(traces, var_names=vars_, rug=True)

    except (EOFError, ValueError):
        print(model, 'error')
        return

    plt.savefig(fig_file.replace('.png', '_trace.png'))
    plt.close()

    if isinstance(x_err, type(None)):
        plot_regressions(x, y, y_err / 2.)
    else:
        plot_regressions(x, y, y_err / 2., x_err / 2.)
    plot_regression_from_trace(model, (x, y, x_err, y_err),
                               ax=plt.gca(), chains=150)

    plt.xlim([x.min() - 0.05, x.max() + 0.05])
    plt.ylim([y.min() - 0.2, y.max() + 0.2])
    plt.xlabel(desc_str[2])
    plt.ylabel(desc_str[4])

    plt.suptitle(' '.join(desc_str))
    print('write {}'.format(fig_file))
    plt.savefig(fig_file)
    plt.close()


def plot_regressions(x, y, sigma_y, sigma_x=None):
    figure = plt.figure(figsize=(16, 10))
    ax = figure.add_subplot(111)
    ax.scatter(x, y, alpha=0.8, color='b')
    lr = linregress(x, y)
    y_pred = lr.slope * x + lr.intercept
    ax.plot(x, y_pred)
    if isinstance(sigma_x, np.ndarray):
        ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, alpha=0.3, ls='', color='b')
    else:
        ax.errorbar(x, y, yerr=sigma_y, alpha=0.3, ls='', color='b')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_regression_from_trace(fitted, observed, ax=None, chains=None):
    traces = [fitted.trace, ]
    xi, yi, sigx, sigy = observed

    for i, trace in enumerate(traces):

        trace_slope = trace['slope']
        trace_inter = trace['inter']

        if chains is not None:
            for chain in range(100, len(trace), 5):
                alpha, beta = trace_inter[chain], trace_slope[chain]
                y = alpha + beta * xi
                ax.plot(xi, y, alpha=0.1, c='red')

        # plot the best-fit line only
        H2D, bins1, bins2 = np.histogram2d(trace_slope,
                                           trace_inter, bins=50)

        w = np.where(H2D == H2D.max())

        # choose the maximum posterior slope and intercept
        slope_best = bins1[w[0][0]]
        intercept_best = bins2[w[1][0]]

        print('beta: {:.3f} alpha: {:.3f}'.format(slope_best, intercept_best))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
