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
sns.set()


def plot_saved_traces(metadata, trc_dir, month, overwrite=False, station=None,
                      only_trace=False, selectors=None):
    with open(metadata, 'r') as f_obj:
        metadata = json.load(f_obj)

    for param in selectors:
        if param == 'cc_q':
            param_dir = trc_dir
        else:
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
                continue

            with open(data, 'r') as f_obj:
                dct = json.load(f_obj)

            x, y, y_err = np.array(dct['x']), np.array(dct['y']), np.array(dct['y_err'])

            if dct['x_err']:
                x_err = np.array(dct['x_err'])
            else:
                x_err = None

            info_ = metadata[sid]
            base = os.path.basename(model)

            if os.path.basename(param_dir) == 'cc':
                splt = base.split('_')
                per, q_mo = splt[2], splt[-1].split('.')[0]
                if int(q_mo) != month:
                    continue

                desc = [sid, info_[str(month)]['STANAME'], dct['xvar'], per, q_mo, dct['yvar']]
            else:
                per = base.split('.')[0].split('_')[-1]
                desc = [sid, info_[str(month)]['STANAME'], dct['xvar'], dct['yvar'], per]

            plot_trace(x, y, x_err, y_err, model, f_dir, desc, overwrite, fmt='png')


def plot_trace(x, y, x_err, y_err, model, fig_dir, desc_str, overwrite=False,
               fmt='png', arviz=False):
    fig_file = os.path.join(fig_dir, '{}.{}'.format('_'.join(desc_str), fmt))

    if not overwrite and os.path.exists(fig_file):
        return

    try:
        with open(model, 'rb') as buff:
            data = pickle.load(buff)
            model, traces = data['model'], data['trace']
            vars_ = ['slope', 'inter']
            summary = az.summary(traces)
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

    chains = np.random.choice(range(100, len(trace_slope), 5), 1000)
    array = np.zeros((len(chains), len(x)))
    for i, chain in enumerate(chains):
        alpha, beta = trace_inter[chain], trace_slope[chain]
        y = alpha + beta * x
        array[i, :] = y

    std_ = np.std(array, axis=0)
    q97, q03 = np.percentile(array, [97, 3], axis=0)

    y = summary.loc['inter', 'mean'] + summary.loc['slope', 'mean'] * x
    plt.fill_between(x, q03, q97, color='b', alpha=0.2)
    lr = linregress(x, y)
    y = lr.intercept + lr.slope * x
    ax.plot(x, y, alpha=1, c='red')
    plt.xlim([-0.2, 1.2])
    plt.suptitle('{} {}'.format(desc_str[0], desc_str[1]))
    print('write {}'.format(os.path.basename(fig_file)))
    plt.savefig(fig_file, format=fmt)
    plt.close(figure)
    plt.clf()


def trace_only(model, fig_file, vars=None):
    with open(model, 'rb') as buff:
        data = pickle.load(buff)
        model, traces = data['model'], data['trace']
        if not vars:
            vars = ['slope', 'inter']
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
    root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'impacts')
    figures = os.path.join(root, 'figures')

    uv_trends_bayes = os.path.join(root, 'analysis', 'uv_trends', 'trends_bayes_{}.json')
    mv_trends_bayes = os.path.join(root, 'analysis', 'mv_trends', 'trends_bayes_{}.json')
    cc_q_bayes = os.path.join(root, 'analysis', 'cc_q', 'cc_q_bayes_{}.json')

    uv_traces = os.path.join(root, 'uv_traces', 'uv_trends')
    mv_traces = os.path.join(root, 'mv_traces', 'mv_trends')
    cc_traces = os.path.join(root, 'mv_traces', 'cc_q')

    month = 9
    station = '13172500'
    meta = cc_q_bayes.format(month)
    plot_saved_traces(meta, cc_traces, month, station=station, selectors=['cc_q'],
                      overwrite=True, only_trace=False)

# ========================= EOF ====================================================================
