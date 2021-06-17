import os
import fiona
import hydrofunctions as hf
import pandas as pd
from pandas import concat, read_csv, date_range, DatetimeIndex
from copy import deepcopy
from collections import OrderedDict
from datetime import datetime as dt

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


def get_station_ids(shp):
    features = []
    with fiona.open(shp, 'r') as src:
        meta = src.meta
        meta['schema']['properties']['start'] = 'str:254'
        meta['schema']['properties']['end'] = 'str:254'
        for f in src:
            try:
                station = f['properties']['SOURCE_FEA']
                nwis = hf.NWIS(station, 'dv', start_date='1900-01-01', end_date='2020-12-31')
                df = nwis.df('discharge')
                s, e = df.index[0], df.index[-1]
                rec_start = '{}-{}-{}'.format(s.year, str(s.month).rjust(2, '0'), str(s.day).rjust(2, '0'))
                rec_end = '{}-{}-{}'.format(e.year, str(e.month).rjust(2, '0'), str(e.day).rjust(2, '0'))
                f['properties']['start'] = rec_start
                f['properties']['end'] = rec_end
                features.append(f)
            except Exception as e:
                print(f['properties']['SOURCE_FEA'], e)

    with fiona.open(shp.replace('.shp', '_dates.shp'), 'w', **meta) as dst:
        for f in features:
            dst.write(f)


def select_stations(basin_gages, out):
    features = []
    with fiona.open(basin_gages, 'r') as src:
        meta = src.meta
        for f in src:
            f['properties']['STAID'] = f['properties']['SOURCE_FEA']
            features.append(f)

    meta['schema'] = {'properties': OrderedDict([('STAID', 'str:40'),
                                                 ('STANAME', 'str:254'),
                                                 ('start', 'str:254'),
                                                 ('end', 'str:254')]),
                      'geometry': 'Point'}

    ct = 0
    with fiona.open(out, 'w', **meta) as dst:
        for f in features:
            nwis = hf.NWIS(f['properties']['STAID']).meta
            key = list(nwis.keys())[0]
            name = nwis[key]['siteName']
            feature = {'geometry': {'coordinates': f['geometry']['coordinates'],
                                    'type': 'Point'},
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                  ('STANAME', name),
                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            dst.write(feature)


def get_station_watersheds(stations_source, watersheds_source, out_stations, out_watersheds):
    with fiona.open(stations_source, 'r') as src:
        point_meta = src.meta
        all_stations = [f for f in src]
        station_ids = [f['properties']['STAID'] for f in src]
        station_meta = [(f['properties']['STANAME'], f['properties']['start'], f['properties']['end']) for f in src]

    selected_shapes = []
    selected_points = []
    with fiona.open(watersheds_source, 'r') as sheds:
        meta = sheds.meta
        for f in sheds:
            p = f['properties']
            if p['SITE_NO'] in station_ids:
                idx = station_ids.index(f['properties']['SITE_NO'])
                attrs = station_meta[idx]
                station_point = all_stations[idx]
                f['properties']['STANAME'] = attrs[0]
                f['properties']['start'] = attrs[1]
                f['properties']['end'] = attrs[2]
                selected_shapes.append(f)
                selected_points.append(station_point)

    meta['schema'] = {'properties': OrderedDict([('STAID', 'str:40'),
                                                 ('SQMI', 'float:19.11'),
                                                 ('STANAME', 'str:254'),
                                                 ('start', 'str:254'),
                                                 ('end', 'str:254')]),
                      'geometry': 'Polygon'}
    ct = 0
    with fiona.open(out_watersheds, 'w', **meta) as dst:
        for f in selected_shapes:

            if f['geometry']['type'] == 'MultiPolygon':
                lengths = [len(x[0]) for x in f['geometry']['coordinates']]
                coords = f['geometry']['coordinates'][lengths.index(max(lengths))]
                # print(f['properties']['SITE_NO'], lengths)
            else:
                coords = f['geometry']['coordinates']

            feature = {'geometry': {'coordinates': coords,
                                    'type': 'Polygon'},
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['SITE_NO']),
                                                  ('STANAME', f['properties']['STANAME'].upper()),
                                                  ('SQMI', f['properties']['SQMI']),
                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
            except TypeError:
                pass

    with fiona.open(out_stations, 'w', **point_meta) as dst:
        for f in selected_points:
            dst.write(f)


def get_station_daily_q(stations_shapefile, out_dir):
    with fiona.open(stations_shapefile, 'r') as src:
        station_ids = [f['properties']['STAID'] for f in src]

    for sid in station_ids:
        nwis = hf.NWIS(sid, 'dv', start_date=s, end_date=e)
        df = nwis.df('discharge')
        out_file = os.path.join(out_dir, 'daily_stations', '{}_daily.csv'.format(sid))
        df.to_csv(out_file)


def get_station_annual_q(start, end, annual_q_dir, daily_q_dir):

    range = (dt.strptime(end, '%Y-%m-%d') - dt.strptime(start, '%Y-%m-%d')).days + 1
    idx = DatetimeIndex(date_range(start, end, freq='D'))
    out_records, short_records = [], []
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    for c in l:
        sid = os.path.basename(c).split('_')[0]
        df = read_csv(c)
        df['datetimeUTC'] = pd.to_datetime(df['datetimeUTC'])
        df = df.set_index('datetimeUTC')

        if range - 1000 <= df.shape[0] < range:
            df = df.reindex(idx)
        elif df.shape[0] < int(range * 0.7):
            short_records.append(sid)
            print(sid, ' skipped')
            continue

        df = df * 2446.58
        df_annual = df.resample('A').sum()
        out_file = os.path.join(annual_q_dir, '{}_annual.csv'.format(sid))
        df_annual.to_csv(out_file)
        out_records.append(sid)
        print(sid)

    print('{} processed, {} short'.format(len(out_records), len(short_records)))


if __name__ == '__main__':
    s, e = '1984-01-01', '2020-12-31'
    annual = '/media/research/IrrigationGIS/gages/hydrographs/annual_q'
    daily = '/media/research/IrrigationGIS/gages/hydrographs/daily_q'
    get_station_annual_q(s, e, annual, daily)
# ========================= EOF ====================================================================
