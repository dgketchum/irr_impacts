import os
import json

import numpy as np
import pandas as pd
from scipy.stats import linregress
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from gage_data import hydrograph

sns.set_style("dark")
sns.set_theme()
sns.despine()


def plot_climate_flow(climate_flow_data, fig_dir_, selected=None, label=True, fmt='png'):
    with open(climate_flow_data, 'r') as f_obj:
        clim_q = json.load(f_obj)

    for sid, records in clim_q.items():

        if selected and sid not in selected:
            continue

        fig = plt.figure(figsize=(7, 1.75))
        ax = fig.add_subplot(111)
        q = np.array(records['q']) / 1e9
        month = records['q_mo']
        ai = np.array(records['ai']) / 1e9

        lr = linregress(ai, q)
        line = ai * lr.slope + lr.intercept
        sns.scatterplot(ai, q, ax=ax, color='blue', s=20)
        sns.lineplot(ai, line, ax=ax, color='black', linewidth=1.0)
        if label:
            ax.set_xlabel('Aridity Index [km^3]')
            ax.set_ylabel('Stream Discharge [km^3]')
            plt.suptitle(records['STANAME'])
        fig_file = os.path.join(fig_dir_, '{}_{}.{}'.format(sid, month, fmt))
        # ax.tick_params(axis='both', which='major', labelsize=8)
        # ax.tick_params(axis='both', which='minor', labelsize=8)
        # ax.grid(False)
        fig.tight_layout()
        plt.savefig(fig_file, format=fmt)


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
