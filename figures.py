import os
import json
from pandas import read_csv
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm


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
        ax = r.plot(ax=ax, kind='line', x=linear, y=r.values, alpha=0.6)
        ax = plt.plot(linear, trend)
        plt.savefig(os.path.join(fig_dir, '{}.png'.format(r.name[5:13])))
        print(r.name[5:13])


def plot_log_volumes(csv_dir, fig_dir, metadata):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]
    l.sort()
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for f in l:
        sid = os.path.basename(f).split('.')[0]
        df = read_csv(f, parse_dates=True, index_col='datetimeUTC')
        name_col = [x for x in list(df.columns) if 'USGS' in x][0]
        df = df.rename(columns={name_col: 'q'})
        df[df <= 0.0] = 1.0
        df = np.log10(df)
        if not np.all(df['cc'].values) > 0:
            print(sid, ' unirrigated')
            continue
        df.plot(y=['cc', 'ppt', 'pet', 'q', 'irr'])
        figname = os.path.join(fig_dir, '{}.png'.format(sid))
        plt.title('{}: 10^{:.2f} sq km'.format(meta[sid]['STANAME'], np.log10(meta[sid]['AREA_SQKM'])))
        plt.savefig(figname)
        print(figname)


if __name__ == '__main__':
    c = '/media/research/IrrigationGIS/gages/hydrographs/group_stations/stations_annual.csv'
    fig = '/media/research/IrrigationGIS/gages/hydrographs/figures/station_time_series'
    # plot_station_hydrographs(c,fig)
    data = '/media/research/IrrigationGIS/gages/merged_q_ee'
    figs = '/media/research/IrrigationGIS/gages/figures/station_log_volumes'
    jsn = '/media/research/IrrigationGIS/gages/station_metadata/metadata.json'
    plot_log_volumes(data, figs, jsn)
# ========================= EOF ====================================================================
