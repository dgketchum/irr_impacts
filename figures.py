import os
import json
from datetime import date

import fiona
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from pylab import rcParams

from hydrograph import hydrograph
from gage_list import EXCLUDE_STATIONS

SYSTEM_STATIONS = ['06109500', '06329500', '09180500', '09315000',
                   '09379500', '12396500', '13269000', '13317000']

SELECTED_SYSTEMS = ['06109500', '06329500', '09180500', '09315000',
                    '09379500', '09466500', '12389000', '12510500',
                    '13269000', '13317000', '14048000',
                    '14103000', '14211720']


def plot_clim_q_resid(q, ai, fit_clim, desc_str, years, cc, resid, fit_resid, fig_d, cci_per, flow_per):
    resid_line = fit_resid.params[1] * cc + fit_resid.params[0]
    clim_line = fit_clim.params[1] * ai + fit_clim.params[0]
    rcParams['figure.figsize'] = 16, 10
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(ai, q)
    ax1.plot(ai, clim_line)
    ax1.set(xlabel='ETr / PPT [-]')
    ax1.set(ylabel='q [m^3]')

    for i, y in enumerate(years):
        ax1.annotate(y, (ai[i], q[i]))
        plt.suptitle(desc_str)

    ax2.set(xlabel='cc [m]')
    ax2.set(ylabel='q epsilon [m^3]')
    ax2.scatter(cc, resid)
    ax2.plot(cc, resid_line)
    for i, y in enumerate(years):
        ax2.annotate(y, (cc[i], resid[i]))

    desc_split = desc_str.strip().split('\n')
    file_name = desc_split[0].replace(' ', '_')

    fig_name = os.path.join(fig_d, '{}_cc_{}-{}_q_{}-{}.png'.format(file_name, cci_per[0], cci_per[1],
                                                                    flow_per[0], flow_per[1]))

    plt.savefig(fig_name)
    plt.close('all')


def fraction_cc_water_balance_fig(metadata, ee_series, fig):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    frac = []
    suspect = []
    # TODO use json created in gage_analysis
    print(len(frac), 'irrigated basins')
    # print(min(frac), max(frac), np.mean(frac))
    lim_ = np.round(max(frac), decimals=2)
    bins = np.linspace(0, lim_, 10)
    plt.xlim(0, lim_)
    plt.hist(frac, bins=bins, histtype='barstacked', rwidth=0.95)
    plt.title('cc_ratio_q\n{} Basins'.format(len(frac)))
    plt.xlabel('Crop Consumption Fraction')
    plt.ylabel('Count')
    plt.savefig(fig)
    print(suspect)


def impact_time_series_bars(sig_stations, figures, min_area=None):
    cmap = cm.get_cmap('RdYlGn')

    with open(sig_stations, 'r') as f:
        stations = json.load(f)

    if min_area:
        stations_l = [k for k, v in stations.items() if v['AREA'] > min_area]
    else:
        stations_l = SELECTED_SYSTEMS

    sort_key = 'cci'
    sort_keys = sorted(stations_l, key=lambda x: stations[x][sort_key], reverse=False)

    all_slope = []
    min_slope, max_slope = -0.640, 0.530

    vert_increment = 1 / 7.
    v_position = 0
    fig, axes = plt.subplots(figsize=(12, 24))
    fig.subplots_adjust(left=0.4)
    ytick, ylab = [], []
    for sid in sort_keys:
        dct = stations[sid]
        impact_keys = [p for p, v in dct.items() if isinstance(v, dict)]

        periods = [(int(p.split('-')[0]), int(p.split('-')[1])) for p in impact_keys]
        periods = sorted(periods, key=lambda x: x[1] - x[0])
        period_arr = [[k for k in range(x[0], x[1] + 1)] for x in periods]

        slopes = [dct[k]['slope'] for k in dct.keys() if k in impact_keys]
        [all_slope.append(s) for s in slopes]

        single_month_resp = [x[1] - x[0] == 0 for x in periods]
        periods = [p[0] for p, s in zip(periods, single_month_resp) if s]
        slopes = [sl for sl, s in zip(slopes, single_month_resp) if s]
        if np.all(slopes == 0.0):
            continue

        for m in [x for x in range(5, 11)]:
            if m in periods:
                s = slopes[periods.index(m)]
                slope_scale = (s - min_slope) / (max_slope - min_slope)
                color = cmap(slope_scale)
                axes.barh(v_position, left=m, width=1, height=vert_increment, color=color,
                          edgecolor=color, align='edge', alpha=0.5)
            else:
                axes.barh(v_position, left=m, width=1, height=vert_increment, color='none',
                          edgecolor='k', align='edge', alpha=0.3)

        tick_pos = v_position + (vert_increment * 0.5)
        ytick.append(tick_pos), ylab.append(dct['STANAME'])
        v_position += vert_increment

    plt.yticks(ytick, ylab)
    x_tick = list(np.linspace(5.5, 11.5, 6))
    x_tick_lab = [x for x in range(5, 11)]
    plt.yticks(ytick, ylab)
    plt.xticks(x_tick, x_tick_lab)
    fig_file = os.path.join(figures, 'slope_bar_ALL_9DEC2021.png')
    plt.xlim([5, 11])
    plt.ylim([0, v_position])
    plt.suptitle('Irrigation Impact on Gages')
    # plt.tight_layout()
    plt.show()
    # plt.savefig(fig_file)
    # plt.close()

    print('\nall slopes min {:.3f}, max {:.3f}'.format(min(all_slope), max(all_slope)))
    print('colorbar slopes used min {:.3f}, max {:.3f}'.format(min_slope, max_slope))


def response_time_to_area(climate_resp, fig_dir):
    with open(climate_resp, 'r') as fp:
        c_dct = json.load(fp)

    irr_ids = []
    c_lags = []
    i_lags = []
    i_areas = []
    c_areas = []
    for k, v in c_dct.items():
        impact_keys = [v for p, v in v.items() if isinstance(v, dict)]
        irr_pct = impact_keys[0]['irr_pct']
        if irr_pct > 0.01:
            irr_ids.append(k)
            i_lags.append(np.median([d['lag'] for d in impact_keys]))
            i_areas.append(v['AREA'])
        else:
            c_lags.append(np.median([d['lag'] for d in impact_keys]))
            c_areas.append(v['AREA'])

    plt.hist(c_lags, bins=20, color='r', label='Unirrigated Basins')
    plt.hist(i_lags, bins=20, color='b', label='Irrigated Basins')
    plt.legend()
    plt.suptitle('Basin Median Response Time')
    plt.ylabel('Response Time [months]')
    plt.xlabel('Area [sq km]')
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'slope_irr.png'))
    plt.close()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    figs = '/media/research/IrrigationGIS/gages/figures'

    cc_frac_json = '/media/research/IrrigationGIS/gages/basin_cc_ratios.json'
    heat_figs = os.path.join(figs, 'heat_bars_largeSystems_singlemonth')
    impact_time_series_bars(cc_frac_json, heat_figs, min_area=20000.)

    # i_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_all.json'
    # c_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_climate_response_irr.json'
    # scatter_figs = os.path.join(figs, 'scatter_area_v_climate_response')
    # response_time_to_area(c_json, fig_dir=figs)
# ========================= EOF ====================================================================
