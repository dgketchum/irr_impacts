import os
import json
import pickle

from matplotlib import pyplot as plt
import numpy as np
import arviz as az

from figs.trace_fig_utils import plot_regressions, plot_regression_from_trace


def plot_saved_traces(metadata, trc_dir, fig_dir, overwrite=False):
    with open(metadata, 'r') as f_obj:
        metadata = json.load(f_obj)

    trc_files = [os.path.join(trc_dir, x) for x in os.listdir(trc_dir) if x.endswith('.model')]
    data_files = [os.path.join(trc_dir, x) for x in os.listdir(trc_dir) if x.endswith('.data')]
    stations = [os.path.basename(x).split('_')[0] for x in trc_files]

    for sid, model, data in zip(stations, trc_files, data_files):

        with open(data, 'r') as f_obj:
            dct = json.load(f_obj)
        x, y, y_err = np.array(dct['x']), np.array(dct['y']), np.array(dct['y_err'])

        if dct['x_err']:
            x_err = np.array(dct['x_err'])
        else:
            x_err = None

        info_ = metadata[sid]

        if os.path.basename(trc_dir) == 'cc_qres':
            per = model[-17:-14]
            desc = [sid, info_['STANAME'], dct['xvar'], per, dct['yvar']]
        else:
            desc = [sid, info_['STANAME'], dct['xvar'], '', dct['yvar']]

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

    plot_regressions(x, y, y_err / 2., x_err / 2.)
    plot_regression_from_trace(model, (x, y, x_err, y_err),
                               ax=plt.gca(), chains=50)

    plt.xlim([x.min() - 0.05, x.max() + 0.05])
    plt.ylim([y.min() - 0.2, y.max() + 0.2])
    plt.xlabel(desc_str[2])
    plt.ylabel(desc_str[4])

    plt.suptitle(' '.join(desc_str))
    print('write {}'.format(fig_file))
    plt.savefig(fig_file)
    # plt.show()
    plt.close()


def trace_only(trc_dir):
    l = [os.path.join(trc_dir, x) for x in os.listdir(trc_dir)]
    for model_file in l:
        try:
            with open(model_file, 'rb') as buff:
                data = pickle.load(buff)
                model, traces = data['model'], data['trace']
                az.plot_trace(traces, var_names=['slope'], rug=True)

        except (EOFError, ValueError):
            print(model, 'error')
            return None
        plt.suptitle(os.path.basename(model_file))
        plt.plot()
        plt.show()
        pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages/gridmet_analysis'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages/gridmet_analysis'

    meta_ = os.path.join(root, 'station_metadata.json')

    for m in range(7, 11):

        var_ = 'cc'
        subdirs_ = ['cc_qres']  # , 'time_qres', 'time_ai', 'time_cc']
        subdirs_.reverse()
        state = 'm_{}'.format(m)

        for r in subdirs_:
            trace_dir = os.path.join(root, 'traces', var_, state, r)

            if not os.path.exists(trace_dir):
                os.makedirs(trace_dir)

            o_fig = os.path.join(root, 'figures', 'slope_trace_{}'.format(var_), state, r)
            if not os.path.exists(o_fig):
                os.makedirs(o_fig)

            plot_saved_traces(meta_, trace_dir, o_fig, overwrite=True)

# ========================= EOF ====================================================================
