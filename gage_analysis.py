import os
import json
from datetime import date

import fiona
import numpy as np
import warnings

from hydrograph import hydrograph

warnings.filterwarnings('ignore')

from station_lists import EXCLUDE_STATIONS


def water_balance_ratios(metadata, ee_series, stations=None, metadata_out=None, out_shp=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    dct = {}
    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS:
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        if not os.path.exists(_file):
            continue
        cdf = hydrograph(_file)
        cdf['cci'] = cdf['cc'] / cdf['irr']
        years = [x for x in range(1991, 2021)]
        cc_dates = [(date(y, 4, 1), date(y, 10, 31)) for y in years]
        clim_dates = [(date(y, 1, 1), date(y, 12, 31)) for y in years]
        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in clim_dates])
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
        irr = np.array([cdf['irr'][d[0]: d[1]].mean() for d in cc_dates])
        cci = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])
        # if not np.all(irr > 0.0):
        #     continue
        print('cci: {:.3f}, {}'.format(np.mean(cci), v['STANAME']))

        dct[sid] = v
        irr_area = (np.mean(irr)).item() / 1e6
        irr_frac = irr_area / v['AREA']
        if irr_area < 0.001:
            irr_area = 0.0

        cc_q_ratio = (cc.sum() / q.sum()).item()
        if cc_q_ratio < 0.0:
            cc_q_ratio = 0.0

        dct[sid].update({'IAREA': irr_area})
        dct[sid].update({'irr_frac': irr_frac})
        dct[sid].update({'cc_q': cc_q_ratio})
        dct[sid].update({'cci': (np.mean(cci)).item()})
        dct[sid].update({'q_ppt': (q.sum() / ppt.sum()).item()})
        dct[sid].update({'ai': (etr.sum() / ppt.sum()).item()})

    if metadata_out:
        with open(metadata_out, 'w') as fp:
            json.dump(dct, fp, indent=4, sort_keys=False)
    if stations:
        with fiona.open(stations, 'r') as src:
            features = [f for f in src]
            meta = src.meta

        del meta['schema']['properties']['LAT']
        del meta['schema']['properties']['LON']

        [meta['schema']['properties'].update({k: 'float:19.11'}) for k, v in dct[sid].items()
         if k not in meta['schema']['properties'].keys() and isinstance(v, float)]

        areas = {f['properties']['STAID']: f['properties']['AREA'] for f in features}
        area_arr = np.array([areas[_id] for _id in dct.keys()])
        areas = {k: (a - min(area_arr)) / (max(area_arr) - min(area_arr)) for k, a in areas.items()}

        with fiona.open(out_shp, 'w', **meta) as dst:
            for f in features:
                sid = f['properties']['STAID']
                if sid in dct.keys():
                    d = {k: v for k, v in dct[sid].items() if isinstance(v, str)}
                    d.update({k: v for k, v in dct[sid].items() if isinstance(v, float)})
                    d['STAID'] = sid
                    d['AREA'] = areas[sid]
                    f['properties'] = d
                    dst.write(f)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    ee_data = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021_unfiltered_q')
    i_json = os.path.join(root, 'gridmet_analysis', 'station_metadata.json')
    station_pts = os.path.join(root, 'gage_loc_usgs', 'selected_gages.shp')
    shp = os.path.join(root, 'gridmet_analysis', 'fig_shapes', 'basin_cc_ratios_annual_q.shp')

    water_balance_ratios(i_json, ee_data, stations=station_pts, out_shp=shp)
# ========================= EOF ====================================================================
