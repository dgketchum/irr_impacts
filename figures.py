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

SYSTEM_STATIONS = ['06109500', '06130500', '06329500', '09180500', '09315000', '09333500',
                   '09379500', '09419000', '09466500', '12396500', '12510500', '13269000',
                   '13317000', '13333000', '14048000', '14103000', '14211720']


def plot_clim_q_resid(q, ai, fit_clim, desc_str, years, cc, resid, fit_resid, fig_d, cci_per):
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

    fig_name = os.path.join(fig_d, '{}_{}-{}.png'.format(file_name, cci_per[0], cci_per[1]))

    plt.savefig(fig_name)
    plt.close('all')


def fraction_cc_water_balance(metadata, ee_series, fig, watersheds=None, metadata_out=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    frac = []
    suspect = []
    for sid, v in metadata.items():
        # if v['irr_mean'] < 0.005:
        #     continue
        if sid in EXCLUDE_STATIONS:
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        cdf = hydrograph(_file)
        years = [x for x in range(1991, 2021)]
        cc_dates = [(date(y, 5, 1), date(y, 10, 31)) for y in years]
        clim_dates = [(date(y, 1, 1), date(y, 12, 31)) for y in years]
        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in clim_dates])
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
        f = cc.sum() / ppt.sum()
        if f > 0.15:
            suspect.append(sid)
            print('\nsuspect {:.3f}'.format(f), v['STANAME'], sid, '\n')
        else:
            print('{:.3f}'.format(f), v['STANAME'], sid)
        frac.append((sid, f))
    frac_dict = {k: v for k, v in frac}
    stations_ = [f[0] for f in frac]
    frac = [f[1] for f in frac]
    print(len(frac), 'irrigated basins')
    # print(min(frac), max(frac), np.mean(frac))
    lim_ = np.round(max(frac), decimals=2)
    bins = np.linspace(0, lim_, 10)
    plt.xlim(0, lim_)
    plt.hist(frac, bins=bins, histtype='barstacked', rwidth=0.95)
    plt.title('Crop Consumption of Total Available Water\n{} Basins'.format(len(frac)))
    plt.xlabel('Crop Consumption Fraction')
    plt.ylabel('Count')
    plt.savefig(fig)
    print(suspect)
    if metadata_out:
        with open(metadata_out, 'w') as fp:
            json.dump(frac_dict, fp, indent=4, sort_keys=False)
    if watersheds:
        with fiona.open(watersheds, 'r') as src:
            features = [f for f in src]
            meta = src.meta
        meta['schema']['properties']['cc_f'] = 'float:19.11'
        out_shp = os.path.join(os.path.dirname(watersheds), os.path.basename(fig).replace('png', 'shp'))
        with fiona.open(out_shp, 'w', **meta) as dst:
            for f in features:
                sid = f['properties']['STAID']
                if sid in stations_:
                    f['properties']['cc_f'] = frac_dict[sid]
                    dst.write(f)


def impact_time_series_heat(sig_stations, sorting_data):

    cmap = cm.get_cmap('RdYlGn')

    with open(sig_stations, 'r') as f:
        stations = json.load(f)
    with open(sorting_data, 'r') as f:
        sorting_data = json.load(f)

    sort_key = 'cc_frac'

    [stations[k].update({sort_key: sorting_data[k]}) for k, v in stations.items() if k in sorting_data.keys()]

    all_slope = []
    min_slope, max_slope = -0.584, 0.582

    for i, sid in enumerate(SYSTEM_STATIONS):
        if sid not in ['06130500']:
            continue
        fig, axes = plt.subplots(figsize=(12, 18))
        dct = stations[sid]
        impact_keys = [p for p, v in dct.items() if isinstance(v, dict)]

        periods = [(int(p.split('-')[0]), int(p.split('-')[1])) for p in impact_keys]
        periods = sorted(periods, key=lambda x: x[1] - x[0])
        period_arr = [[k for k in range(x[0], x[1] + 1)] for x in periods]
        durations = [len(x) for x in period_arr]

        slopes = [dct[k]['slope'] for k in dct.keys() if k in impact_keys]
        [all_slope.append(s) for s in slopes]

        vert_increment = 1 / 7.
        v_position = 0
        print('')
        for j, (s, p, d) in enumerate(zip(slopes, periods, durations)):
            slope_scale = (s - min_slope) / (max_slope - min_slope)
            color = cmap(slope_scale)
            if d == 1:
                pass
            else:
                v_position += vert_increment
            axes.barh(v_position, left=p[0], width=d, height=vert_increment, color=color,
                      edgecolor=color, align='edge', alpha=0.5)
            print('{} {:.2f} {}'.format(sid, s, d))

        plt.xlim([4, 12])
        plt.suptitle('{}\n{}'.format(sid, dct['STANAME']))
        # plt.show()
        plt.savefig('/home/dgketchum/Downloads/impacts/impacted_gages_{}_{}_6DEC2021.png'.format(sid, sort_key))
        plt.close()

    print('\nall slopes min {:.3f}, max {:.3f}'.format(min(all_slope), max(all_slope)))
    print('colorbar slopes used min {:.3f}, max {:.3f}'.format(min_slope, max_slope))


def scatter_from_json(json_, fig_dir):
    with open(json_, 'r') as fp:
        dct = json.load(fp)
    slope = [v['slope'] for k, v in dct.items()]
    irr = [v['irr_pct'] for k, v in dct.items()]
    ccfr = [v['cc_frac'] for k, v in dct.items()]

    plt.scatter(slope, irr)
    plt.xlabel('slope')
    plt.ylabel('irr')
    plt.savefig(os.path.join(fig_dir, 'slope_irr.png'))
    plt.close()

    plt.scatter(slope, ccfr)
    plt.xlabel('slope')
    plt.ylabel('ccfr')
    plt.savefig(os.path.join(fig_dir, 'slope_ccfr.png'))
    plt.close()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    figs = '/media/research/IrrigationGIS/gages/figures'

    watersheds_shp = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds.shp'
    frac_fig = os.path.join(figs, 'water_balance_frac_cc.png')
    _json = '/media/research/IrrigationGIS/gages/station_metadata/basin_climate_response_gt8000.json'
    ee_data = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021'
    cc_frac_json = '/media/research/IrrigationGIS/gages/basin_cc_fraction_water_bal.json'
    # fraction_cc_water_balance(_json, ee_data, frac_fig, watersheds=None, metadata_out=cc_frac_json)

    o_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_gt8000.json'

    coords = '/media/research/IrrigationGIS/gages/basin_areas.json'
    impact_time_series_heat(o_json, cc_frac_json)
    # scatter_from_json(s_json, fig_dir=figs)
# ========================= EOF ====================================================================
