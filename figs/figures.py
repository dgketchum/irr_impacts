import os
import json

import numpy as np
import pandas as pd
from scipy.stats import linregress
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from gage_data import hydrograph

sns.set_theme(style="whitegrid")
large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }


def plot_climate_flow(climate_flow_data, fig_dir_, selected=None):
    with open(climate_flow_data, 'r') as f_obj:
        clim_q = json.load(f_obj)

    for sid, records in clim_q.items():

        if selected and sid not in selected:
            continue

        figure = plt.figure(figsize=(16, 10))
        ax = figure.add_subplot(111)
        q = np.array(records['q']) / 1e9
        month = records['q_mo']
        ai = np.array(records['ai']) / 1e9

        lr = linregress(ai, q)
        line = ai * lr.slope + lr.intercept
        sns.lineplot(ai, line, label='Least Squares', ax=ax)
        sns.scatterplot(ai, q, label='Observation', ax=ax)
        ax.set_xlabel('Aridity Index [km^3]')
        ax.set_ylabel('Stream Discharge [km^3]')
        plt.suptitle(records['STANAME'])
        fig_file = os.path.join(fig_dir_, '{}_{}.png'.format(sid, month))
        plt.savefig(fig_file)


def hydrograph_vs_crop_consumption(ee_data, sid, fig_dir):
    _file = os.path.join(ee_data, '{}.csv'.format(sid))
    df = hydrograph(_file)
    df[np.isnan(df)] = 0.0
    df = df.resample('A').agg(pd.DataFrame.sum, skipna=False)
    df = df.loc['1991-01-01': '2020-12-31']
    df /= 1e9
    plt_cols = ['cc', 'q']
    df = df[plt_cols]
    df = df.rename(columns={'cc': 'Crop Consumption', 'q': 'Flow'})
    blue = tuple(np.array([43, 131, 186]) / 256.)
    orange = tuple(np.array([253, 174, 97]) / 256.)
    y_err = [x * 0.25 for x in df['Crop Consumption']]
    y_err_zero = [x * 0. for x in df['Crop Consumption']]
    ax = df.plot(kind='bar', stacked=False, color=[orange, blue], width=0.9, yerr=[y_err, y_err_zero])
    plt.suptitle('Snake River Flow and Crop Consumption')
    plt.ylabel('Cubic Kilometers')
    plt.xlabel('Year')
    ticks = [str(x) for x in range(1991, 2021)]
    ticks = ['' if x not in ['1991', '2000', '2010', '2020'] else x for x in ticks]
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticks))
    plt.tight_layout()
    out_fig = os.path.join(fig_dir, '{}'.format(sid))
    plt.savefig(out_fig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'
# ========================= EOF ====================================================================
