import os
import json
from collections import OrderedDict

import fiona
from pandas import read_csv, Series
import numpy as np
import statsmodels.api as sm

SHORT_RECORDS = ['06017000', '06024580', '06040000', '06074000', '06077500',
                 '06079000', '06080900', '06082200', '06090500', '06102050',
                 '06130000', '06167500', '06307830', '06327500', '09015000',
                 '09021000', '09072550', '09073005', '09076000', '09113500',
                 '09127000', '09129600', '09222400', '09246500', '09357000',
                 '09359500', '09361000', '09383400', '09385700', '09396100',
                 '09397000', '09398300', '09399400', '09408195', '09417500',
                 '09419700', '09419800', '09424447', '09486055', '09502800',
                 '09508500', '12392300', '12422000', '12426000', '12445900',
                 '12464800', '12506000', '13014500', '13025500', '13057000',
                 '13057500', '13063000', '13090500', '13119000', '13135500',
                 '13137000', '13137500', '13265500', '13296000', '13307000',
                 '13337500', '13348000', '14101500', '14113200', '14144900',
                 '14150800', '14152500', '14159110', '14159200', '14163000',
                 '14165500', '14171000', '14184100', '14185800', '14188800',
                 '14200000', '14200300', '14201500', '14216000', '14219000'
                 ]

DISCONTINUITIES = []


def add_metadata_to_shapefile(basin_gages, series, out):
    features = []
    with fiona.open(basin_gages, 'r') as src:
        meta = src.meta
        for f in src:
            sid = f['properties']['STAID']
            s_path = os.path.join(series, '{}_annual.csv'.format(sid))
            s = read_csv(s_path, parse_dates=True, index_col=0)
            try:
                c, t = sm.tsa.filters.hpfilter(s.values, lamb=6.25)
            except:
                continue
            t[t < 0] = 0
            dt = (t[-1] - t[0]) / np.mean(t)
            f['properties']['change'] = dt
            trend = Series(t, index=s.index)
            features.append(f)

    meta['schema'] = {'properties': OrderedDict([('STAID', 'str:40'),
                                                 ('STANAME', 'str:254'),
                                                 ('SQMI', 'float:19.11'),
                                                 ('change', 'float:19.11'),
                                                 ('start', 'str:254'),
                                                 ('end', 'str:254')]),
                      'geometry': 'Polygon'}

    ct = 0
    with fiona.open(out, 'w', **meta) as dst:
        for f in features:
            feature = {'geometry': {'coordinates': f['geometry']['coordinates'],
                                    'type': 'Polygon'},
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                  ('STANAME', f['properties']['STANAME']),
                                                  ('SQMI', f['properties']['SQMI']),
                                                  ('change', f['properties']['change']),
                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            dst.write(feature)
            ct += 1


def write_json_to_shapefile(in_shp, out_shp, meta):
    with open(meta, 'r') as f:
        meta = json.load(f)

    features = []
    out_gages = []
    gages = list(meta.keys())
    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        for f in src:
            sid = f['properties']['STAID']
            if sid in gages:
                out_gages.append(sid)
                features.append(f)

    shape_meta['schema']['properties'].update({'STANAME': 'str:40',
                                               'cc_frac': 'float:19.11',
                                               'irr_pct': 'float:19.11',
                                               'slope': 'float:19.11'})

    ct = 0
    with fiona.open(out_shp, 'w', **shape_meta) as dst:
        for f in features:
            ct += 1
            sid = f['properties']['STAID']
            feature = {'geometry': f['geometry'],
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                  ('STANAME', f['properties']['STANAME']),
                                                  ('SQMI', f['properties']['SQMI']),

                                                  ('cc_frac', meta[sid]['cc_frac']),
                                                  ('irr_pct', meta[sid]['irr_pct']),
                                                  ('slope', meta[sid]['slope']),

                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                print(f['properties']['STAID'], f['properties']['STANAME'])
            except TypeError:
                pass


if __name__ == '__main__':
    w = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds.shp'
    o = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds_meta.shp'
    a_series = '/media/research/IrrigationGIS/gages/hydrographs/annual_stations'
    add_metadata_to_shapefile(w, a_series, o)
# ========================= EOF ====================================================================
