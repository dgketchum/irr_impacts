import os
import json
from pandas import read_csv, to_datetime, DatetimeIndex, date_range
import numpy as np
import matplotlib
from matplotlib import colors, cm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import statsmodels.api as sm
from PyEMD import EMD
from stations import hydrograph
from tables import CLMB_STATIONS


def plot_station_hydrographs(csv, fig_dir=None):
    df = read_csv(csv, index_col=0, parse_dates=True)
    df.index.freq = 'A'
    df[df == 0.0] = np.nan
    df = df.dropna(axis=1, how='any')
    linear = [x.year for x in df.index]
    means = df.mean(axis=0)
    z_totals = df.div(means, axis=1)
    z_totals.index = linear
    for i, r in df.iteritems():
        fig, ax = plt.subplots()
        r.index = linear
        cycle, trend = sm.tsa.filters.hpfilter(r.values, lamb=6.25)
        imf = EMD().emd(r.values, linear)
        ax = r.plot(ax=ax, kind='line', x=linear, y=r.values, alpha=0.6)
        ax = plt.plot(linear, trend)
        ax = plt.plot(linear, imf[-1, :])
        # plt.show()
        plt.savefig(os.path.join(fig_dir, '{}.png'.format(r.name[5:13])))
        print(r.name[5:13])


def plot_log_volumes(csv_dir, fig_dir, metadata):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]
    l.sort()
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for f in l:
        if '06192500' not in f:
            continue
        sid = os.path.basename(f).split('.')[0]
        df = read_csv(f, parse_dates=True, index_col='datetimeUTC')
        df = df.rename(columns={list(df.columns)[0]: 'q'})
        df[df <= 0.0] = 1.0
        df = np.log10(df)
        if not np.all(df['cc'].values) > 0:
            print(sid, ' unirrigated')
            continue
        df.plot(y=['cc', 'pr', 'etr', 'q', 'irr'])
        figname = os.path.join(fig_dir, '{}.png'.format(sid))
        plt.title('{}: 10^{:.2f} sq km'.format(meta[sid]['STANAME'], np.log10(meta[sid]['AREA_SQKM'])))
        plt.savefig(figname)
        print(figname)


def plot_irrigated_fraction(dir_):
    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]
    irr_idx = []
    for c in l:
        sid = os.path.basename(c).split('_')[0]
        df = read_csv(c)
        df['datetimeUTC'] = to_datetime(df['datetimeUTC'])
        df = df.set_index('datetimeUTC')
        df.rename(columns={list(df.columns)[0]: 'q'}, inplace=True)
        df['if'] = df['cc'] / df['q']
        frac = df['cc'].sum() / df['q'].sum()
        irr_idx.append(frac)
        print(sid, frac)


def plot_bf_time_series(daily_q_dir, fig_dir, metadata, start_month=None, end_month=None):
    s, e = '1984-01-01', '2020-12-31'
    idx = DatetimeIndex(date_range(s, e, freq='D'))
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        df = hydrograph(c)
        if start_month or end_month:
            idx = idx[idx.month.isin([x for x in range(start_month, end_month)])]
            df = df[df.index.month.isin([x for x in range(start_month, end_month)])]

        df.plot(y=['q', 'qb'])
        figname = os.path.join(fig_dir, '{}.png'.format(sid))
        plt.title('{} ({}): 10^{:.2f} sq km'.format(meta[sid]['STANAME'], sid, np.log10(meta[sid]['AREA_SQKM'])))
        plt.savefig(figname)
        # plt.show()
        print(figname)


def scatter_bfi_cc(metadata, csv_dir, fig):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    irr_area, basin_area, ratio = [], [], []
    bfi = []
    exclude = []
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        df = read_csv(c)
        m = metadata[sid]
        mean_area = df['irr'].mean()
        basin_sqm = m['AREA_SQKM'] * 1e6
        if mean_area == 0.0:
            irr_area.append(mean_area)
        else:
            irr_area.append(mean_area)

        try:
            bfi_max = m['bfi_']
            rat = mean_area / basin_sqm
            if bfi_max < 0:
                exclude.append(sid)
                continue

            bfi.append(bfi_max)
            basin_area.append(basin_sqm)
            ratio.append(rat)
        except KeyError:
            exclude.append(sid)
            print('no bfi {} {}'.format(sid, m['STANAME']))

    plt.scatter(bfi, ratio)
    plt.savefig(fig)
    print(exclude)


def parameter_shift(jsn):
    with open(jsn, 'r') as f:
        meta = json.load(f)

    x_e, x_l = [], []
    y_e, y_l = [], []
    delt = []
    params = ['cc', 'irr', 'qb']
    for k, v in meta.items():
        try:
            if v['irr_late'] > 0.01:
                x_e.append(v['{}_early'.format(params[0])])
                x_l.append(v['{}_late'.format(params[0])])
                y_e.append(v['{}_early'.format(params[1])])
                y_l.append(v['{}_late'.format(params[1])])
                delt.append(v['{}_late'.format(params[2])] - v['{}_early'.format(params[2])])
        except KeyError:
            pass
    lines = [[(_xe, _ye), (_xl, _yl)] for _xe, _ye, _xl, _yl in zip(x_e, y_e, x_l, y_l)]
    slope = [(x[1][1] - x[0][1]) / (x[1][0] - x[0][0]) for x in lines]
    slope_bin = [1 if x > 0. else 0 for x in slope]
    _min, _max = min(slope), max(slope)
    slope_scale = np.interp(slope, (_min, _max), (0, 1))
    cmap = cm.get_cmap('RdYlGn')
    irr_clr = [cmap(i) for i in slope_bin]
    lc = LineCollection(lines, colors=irr_clr)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_title('Current')
    plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    c = '/media/research/IrrigationGIS/gages/hydrographs/group_stations/stations_annual.csv'
    # plot_station_hydrographs(c, fig)
    data = '/media/research/IrrigationGIS/gages/merged_q_ee/full_year'
    figs = '/media/research/IrrigationGIS/gages/figures/station_log_volumes'

    jsn = '/media/research/IrrigationGIS/gages/station_metadata/metadata_flows_gridded_julOct.json'

    fig = '/media/research/IrrigationGIS/gages/figures/bfi_vs_irr.png'
    src = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'
    # plot_bf_time_series(src, fig, jsn)
    # scatter_bfi_cc(jsn, data, fig)
    parameter_shift(jsn)

# ========================= EOF ====================================================================
