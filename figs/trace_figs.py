import os
import json
import pickle

import numpy as np
from scipy.stats import linregress
import arviz as az
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("dark")
sns.set_theme()
sns.despine()


def plot_saved_traces(metadata, trc_dir, fig_dir, month, overwrite=False, selected=None, fmt='png', only_trace=False):
    with open(metadata, 'r') as f_obj:
        metadata = json.load(f_obj)

    trc_files = [os.path.join(trc_dir, x) for x in os.listdir(trc_dir) if x.endswith('.model')]
    data_files = [trc.replace('.model', '.data') for trc in trc_files]
    stations = [os.path.basename(x).split('_')[0] for x in trc_files]

    for sid, model, data in zip(stations, trc_files, data_files):

        if selected and sid not in selected:
            continue

        if only_trace:
            fig_file_ = os.path.join(fig_dir, os.path.basename(model).replace('.model', '.png'))
            trace_only(model, fig_file_)
            return

        with open(data, 'r') as f_obj:
            dct = json.load(f_obj)
        x, y, y_err = np.array(dct['x']), np.array(dct['y']), np.array(dct['y_err'])

        if dct['x_err']:
            x_err = np.array(dct['x_err'])
        else:
            x_err = None

        info_ = metadata[sid]
        base = os.path.basename(model)

        if os.path.basename(trc_dir) in ['cc_qres', 'ccres_qres']:
            splt = base.split('_')
            per, q_mo = splt[2], splt[-1].split('.')[0]
            if int(q_mo) != month:
                continue

            desc = [sid, info_['STANAME'], dct['xvar'], per, q_mo, dct['yvar']]
        else:
            per = base.split('.')[0].split('_')[-1]
            desc = [sid, info_['STANAME'], dct['xvar'], dct['yvar'], per]

        plot_trace(x, y, x_err, y_err, model, fig_dir, desc, overwrite, fmt='png')


def plot_trace(x, y, x_err, y_err, model, fig_dir, desc_str, overwrite=False, fmt='png', arviz=False):
    fig_file = os.path.join(fig_dir, '{}.{}'.format('_'.join(desc_str), fmt))

    if not overwrite and os.path.exists(fig_file):
        return

    try:
        with open(model, 'rb') as buff:
            data = pickle.load(buff)
            model, traces = data['model'], data['trace']
            vars_ = ['slope', 'inter']
            if arviz:
                az.plot_trace(traces, var_names=vars_, rug=True)
                plt.savefig(fig_file.replace('.png', '_trace.png'))
                plt.close()
                plt.clf()

    except (EOFError, ValueError):
        print(model, 'error')
        return

    figure = plt.figure(figsize=(14, 4.5), dpi=900)
    ax = figure.add_subplot(111)
    ax.scatter(x, y, color='b')

    if isinstance(x_err, np.ndarray):
        ax.errorbar(x, y, xerr=x_err / 2.0, yerr=y_err / 2.0, alpha=0.3, ls='', color='b')
    else:
        ax.errorbar(x, y, yerr=y_err / 2.0, alpha=0.3, ls='', color='b')

    traces = [model.trace, ]
    for i, trace in enumerate(traces):

        trace_slope = trace['slope']
        trace_inter = trace['inter']

        for chain in range(100, len(trace), 5):
            alpha, beta = trace_inter[chain], trace_slope[chain]
            y = alpha + beta * x
            ax.plot(x, y, alpha=0.1, c='red')

    plt.xlim([-0.2, 1.2])
    print('write {}'.format(fig_file))
    plt.savefig(fig_file, format=fmt)
    plt.close(figure)
    plt.clf()


def trace_only(model, fig_file):
    with open(model, 'rb') as buff:
        data = pickle.load(buff)
        model, traces = data['model'], data['trace']
        vars_ = ['slope_1', 'slope_2', 'inter']
        az.plot_trace(traces, var_names=vars_, rug=True)
        plt.savefig(fig_file.replace('.png', '_trace.png'))
        plt.close()
        plt.clf()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
