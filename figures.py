import os
from pandas import read_csv, DataFrame
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
        cycle, trend = sm.tsa.filters.hpfilter(r.values)
        ax = r.plot(ax=ax, kind='line', x=linear, y=r.values, alpha=0.6)
        ax = plt.plot(linear, trend)
        plt.savefig(os.path.join(fig_dir, '{}.png'.format(r.name[5:13])))
        print(r.name[5:13])

    # z_totals.name = 'All'
    # # z_totals.plot(ax=ax, kind='line', color='k', alpha=0.7, x=linear, y=z_totals.values)
    # # plt.title('Normalized Irrigated Area')
    # # ax.axvspan(2011.5, 2012.5, alpha=0.5, color='red')
    # plt.xlim(linear[0], linear[-1])
    # # plt.ylim(0.0, 2.0)
    # # plt.legend(loc='lower center', ncol=5, labelspacing=0.5)
    # if fig_dir:
    #     plt.savefig(fig_dir)
    #     return None
    # plt.show()


if __name__ == '__main__':
    c = '/media/research/IrrigationGIS/gages/hydrographs/group_stations/stations_annual.csv'
    fig = '/media/research/IrrigationGIS/gages/hydrographs/figures/station_time_series'
    plot_station_hydrographs(c,fig)
    pass
# ========================= EOF ====================================================================
