import os
import json
from datetime import date

import fiona
from pandas import read_csv, DataFrame
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pylab import rcParams
import statsmodels.api as sm
from hydrograph import hydrograph
from gage_list import CLMB_STATIONS, UMRB_STATIONS

STATIONS = CLMB_STATIONS + UMRB_STATIONS

MAJOR_IMPORTS = ['06088500', '06253000', '12472600', '12513000', '12324680', '12329500',
                 '12467000', '13108150', '13153500', '13152500',
                 '13153500', '09372000', '09371492']


def get_station_coordinates(shp, coords):
    dct = {}
    with fiona.open(shp, 'r') as src:
        for f in src:
            coord = f['geometry']['coordinates']
            dct[f['properties']['STAID']] = {'lat': coord[1], 'lon': coord[0]}
    if coords:
        with open(coords, 'w') as f:
            json.dump(dct, f, indent=4, sort_keys=False)


def fraction_cc_water_balance(metadata, ee_series, fig, watersheds=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    frac = []
    suspect = []
    for sid, v in metadata.items():
        if v['irr_mean'] < 0.01:
            continue
        if sid in MAJOR_IMPORTS:
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
        if f > 0.1:
            suspect.append(sid)
            print('\n{:.3f}'.format(f), v['STANAME'], sid, '\n')
            continue
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


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    watersheds_shp = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds.shp'

    figs = '/media/research/IrrigationGIS/gages/figures'
    o_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_metadata_17NOV2021.json'

    figs = '/media/research/IrrigationGIS/gages/figures'

    frac_fig = os.path.join(figs, 'water_balance_frac_cc.png')
    # fraction_cc_water_balance(_json, ee_data, frac_fig, watersheds=None)
    coords = '/media/research/IrrigationGIS/gages/station_coordinates.json'
    stations = '/media/research/IrrigationGIS/gages/gage_loc_usgs/selected_gages.shp'
    get_station_coordinates(stations, coords)
# ========================= EOF ====================================================================
