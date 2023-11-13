import os
import json

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from gage_data import hydrograph

# sns.set_style("dark")
# sns.set_theme()
# sns.despine()


def bidirectional_histogram(csv):
    df = pd.read_csv(csv, index_col=['Unnamed: 0'])
    # df = df[df['AREA'] < 0.2]
    df.drop(columns=['AREA', 'geometry'], inplace=True)
    pos_ct = np.count_nonzero(df.values > 0, axis=0).astype(int)
    negct = np.count_nonzero(df.values < 0, axis=0).astype(int)
    color_red = '#e66e61'
    color_blue = '#63a9cf'
    index = [x + 1 for x in range(12)]
    title0 = 'Negative'
    title1 = 'Positve'
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)
    fig.tight_layout()
    axes[0].barh(index, pos_ct, align='center', color=color_red, zorder=10)
    axes[0].set_title(title0, fontsize=14, pad=15)
    axes[1].barh(index, negct, align='center', color=color_blue, zorder=10)
    axes[1].set_title(title1, fontsize=14, pad=15)
    axes[0].invert_xaxis()
    plt.suptitle('IWU - Flow Relationship Direction', y=0.97, x=0.56)
    plt.xlabel('Count', x=0)
    plt.xlim([0, 70])
    fig_ = csv.replace('.csv', '_histogram.png')
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    plt.savefig(fig_)


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


def hydrograph_vs_crop_consumption(ee_data, sid='13172500', fig_dir=None):
    matplotlib.rc('font', family='Ubuntu')
    _file = os.path.join(ee_data, '{}.csv'.format(sid))
    df = hydrograph(_file)
    df[np.isnan(df)] = 0.0
    df = df.resample('A').agg(pd.DataFrame.sum, skipna=False)
    df = df.loc['1987-01-01': '2021-12-31']
    df /= 1e9
    plt_cols = ['cc', 'q']
    df = df[plt_cols]
    cc = df.cc.values
    q = df.q.values
    cc_err = [x * 0.31 for x in cc]
    q_err = [x * 0.08 for x in q]
    q_pos = np.arange(0, len(q) * 2, 2)
    cc_pos = np.arange(1, len(q) * 2 + 1, 2)
    fig, ax = plt.subplots(1, 1, figsize=(39, 16))
    error_kw = dict(lw=5, capsize=5, capthick=3)
    plt.hlines([1, 2], -1, 70, alpha=1, color='k')
    ax.bar(q_pos, q, width=0.9, yerr=q_err, color='#2b83ba', align='center',
           capsize=2, edgecolor='k', label='Annual Streamflow', error_kw=error_kw)
    ax.bar(cc_pos, cc, width=0.9, yerr=cc_err, color='#fdae61', align='center',
           capsize=2, edgecolor='k', label='Apr - Oct Irrigation Water Use', error_kw=error_kw)
    font_size = 70
    plt.suptitle('Gallatin River Near Logan, Montana', size=font_size)
    plt.ylabel('km$^{3}$', size=font_size)
    # plt.xlabel('Year', size=font_size)
    ticks = [str(int(x)) for x in np.arange(1986, 2022)]
    ticks = ['' if x not in ['1991', '2000', '2010', '2020'] else x for x in ticks]
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticks))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim([-1, 70])
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.07)
    plt.gcf().subplots_adjust(left=0.075)
    out_fig = os.path.join(fig_dir, '{}'.format(sid))
    plt.legend(prop={'size': font_size})
    plt.savefig(out_fig)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'

    figs = os.path.join(root, 'figures', 'cc_q_ratios')
    data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')
    hydrograph_vs_crop_consumption(data_tables, sid='06052500', fig_dir=figs)

    _dir = os.path.join(root, 'figures', 'shapefiles', 'cc_q')
    csv_ = os.path.join(_dir, 'cc_q_bayes.csv')
    # bidirectional_histogram(csv_)
# ========================= EOF ====================================================================
