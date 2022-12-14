import os
import json
import pickle

import numpy as np
import arviz as az
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("dark")
sns.set_theme()
sns.despine()


def plot_saved_traces(metadata, trc_dir, month, overwrite=False, station=None, only_trace=False, selectors=None):
    with open(metadata, 'r') as f_obj:
        metadata = json.load(f_obj)

    for param in selectors:
        param_dir = os.path.join(trc_dir, param)
        model_dir, data_dir = os.path.join(param_dir, 'model'), os.path.join(param_dir, 'data')

        model_files = [os.path.join(model_dir, x) for x in os.listdir(model_dir)]
        targets = [os.path.basename(m).split('.')[0] for m in model_files]
        data_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.split('.')[0] in targets]

        stations = [os.path.basename(x).split('_')[0] for x in model_files]

        f_dir = os.path.join(param_dir, 'plots')
        if not os.path.exists(f_dir):
            os.mkdir(f_dir)

        for sid, model, data in zip(stations, model_files, data_files):

            if station and sid not in station:
                continue

            if only_trace:
                fig_file_ = os.path.join(f_dir, os.path.basename(model).replace('.model', '.png'))
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

            if os.path.basename(param_dir) in ['cc_qres', 'ccres_qres']:
                splt = base.split('_')
                per, q_mo = splt[2], splt[-1].split('.')[0]
                if int(q_mo) != month:
                    continue

                desc = [sid, info_[str(month)]['STANAME'], dct['xvar'], per, q_mo, dct['yvar']]
            else:
                per = base.split('.')[0].split('_')[-1]
                desc = [sid, info_[str(month)]['STANAME'], dct['xvar'], dct['yvar'], per]

            plot_trace(x, y, x_err, y_err, model, f_dir, desc, overwrite, fmt='png')


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

    trace_slope = traces.posterior.slope.values.flatten()
    trace_inter = traces.posterior.inter.values.flatten()

    for chain in range(100, len(trace_slope), 5):
        alpha, beta = trace_inter[chain], trace_slope[chain]
        y = alpha + beta * x
        ax.plot(x, y, alpha=0.01, c='red')

    plt.xlim([-0.2, 1.2])
    print('write {}'.format(fig_file))
    plt.savefig(fig_file, format=fmt)
    plt.close(figure)
    plt.clf()


def trace_only(model, fig_file, vars=None):
    with open(model, 'rb') as buff:
        data = pickle.load(buff)
        model, traces = data['model'], data['trace']
        if not vars:
            vars = ['slope_1', 'slope_2', 'inter']
        az.plot_trace(traces, var_names=vars, rug=True)
        plt.savefig(fig_file.replace('.png', '_trace.png'))
        plt.close()
        plt.clf()


def bayes_lines(x, x_err, y, y_err, traces, fig_file):
    figure = plt.figure(figsize=(14, 4.5), dpi=900)
    ax = figure.add_subplot(111)
    ax.scatter(x, y, color='b')

    if isinstance(x_err, np.ndarray):
        ax.errorbar(x, y, xerr=x_err / 2.0, yerr=y_err / 2.0, alpha=0.3, ls='', color='b')
    else:
        ax.errorbar(x, y, yerr=y_err / 2.0, alpha=0.3, ls='', color='b')

    trace_inter = az.extract_dataset(traces).inter.values
    trace_slope = az.extract_dataset(traces).slope.values
    idx = list(range(0, len(trace_inter), 20))
    for i, chain in enumerate(idx):
        alpha, beta = trace_inter[chain], trace_slope[chain]
        y = alpha + beta * x
        ax.plot(x, y, alpha=0.05, c='red')

    plt.savefig(fig_file)
    plt.close()
    plt.clf()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
