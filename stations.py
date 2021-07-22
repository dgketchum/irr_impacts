import os
import json
from pprint import pprint
import numpy as np
import fiona
import hydrofunctions as hf
from pandas import date_range, DatetimeIndex, Series
from collections import OrderedDict
from gage_analysis import hydrograph, EXCLUDE_STATIONS


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


def get_station_daily_data(param, start, end, stations_shapefile, out_dir):
    if param not in ['00010', 'discharge']:
        raise ValueError('Use param = 00010 or discharge.')
    with fiona.open(stations_shapefile, 'r') as src:
        station_ids = [f['properties']['STAID'] for f in src]

    for sid in station_ids:
        if sid in EXCLUDE_STATIONS:
            continue
        nwis = hf.NWIS(sid, 'dv', start_date=start, end_date=end)
        try:
            df = nwis.df(param)
            out_file = os.path.join(out_dir, '{}.csv'.format(sid))
            df.to_csv(out_file)
            print(out_file)
        except ValueError:
            pass


def get_station_daterange_data(year_start, aggregate_q_dir, daily_q_dir, start_month=None, end_month=None,
                               temp_dir=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]

    if temp_dir:
        temp_files = [os.path.join(temp_dir, x) for x in os.listdir(temp_dir)]
        temp_stations = [os.path.basename(x).split('.')[0] for x in temp_files]
        q_files = [x for x in q_files if os.path.basename(x).split('.')[0] in temp_stations]

    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    idx = DatetimeIndex(date_range(s, e, freq='D'))

    out_records, short_records = [], []
    full_ct, partial_ct = 0, 0
    full_years_by_station = {yr: 0 for yr in range(1991, 2021)}
    for c in q_files:
        sid = os.path.basename(c).split('.')[0]
        df = hydrograph(c)

        if temp_files:
            t_file = os.path.join(temp_dir, '{}.csv'.format(sid))
            dft = hydrograph(t_file)
            try:
                col = [x for x in list(dft.columns) if '00002' in x][0]
            except IndexError:
                continue
            dft = Series(dft[col]).asfreq('d')
            counts = dft.value_counts()
            count_pct = counts / df.shape[0]
            if np.count_nonzero(count_pct > 0.1) > 0:
                continue
            dft = dft.reindex_like(df)
            count_nan = np.count_nonzero(np.isnan(dft.values))
            if count_nan > 0:
                partial_ct += 1
            else:
                full_ct +=1
            print('{}: {} nan of {}'.format(sid, count_nan, dft.shape[0]))
            year_counts = []
            for yr in range(1991, 2021):
                try:
                    dfy = dft.loc['{}-01-01'.format(yr): '{}-12-31'.format(yr)]
                    non_nan = np.count_nonzero(~np.isnan(dfy))
                    year_counts.append((yr, non_nan))
                    if non_nan == dfy.shape[0]:
                        full_years_by_station[yr] += 1
                except IndexError:
                    pass
            [print(x[0], x[1]) for x in year_counts]

        continue
    print('{} partial, {} full'.format(partial_ct, full_ct))
    pprint(full_years_by_station)
    #     if start_month or end_month:
    #         idx = idx[idx.month.isin([x for x in range(start_month, end_month)])]
    #         df = df[df.index.month.isin([x for x in range(start_month, end_month)])]
    #
    #     if idx.shape[0] == df.shape[0]:
    #         pass
    #     elif df.shape[0] < int(idx.shape[0] * 0.7):
    #         short_records.append(sid)
    #         print(sid, ' q skipped')
    #         continue
    #     else:
    #         print('df {} idx {}'.format(df.shape[0], idx.shape[0]))
    #
    #     # cfs to m ^3 d ^-1
    #     df = df * 2446.58
    #
    #     # TODO: interpolate missing values
    #     df_annual = df.resample('A').sum()
    #     out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
    #     df_annual.to_csv(out_file)
    #     out_records.append(sid)
    #     print(sid)
    #
    # print('{} processed'.format(len(out_records)))


def write_impacted_stations(in_shp, out_shp, json_):
    with open(json_, 'r') as f:
        station_data = json.load(f)
    keys = station_data.keys()
    features = []
    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for f in src:
            sid = str(f['properties']['SITE_NO']).rjust(8, '0')
            if sid in keys:
                f['properties']['STANAME'] = station_data[sid]['STANAME']
                f['properties']['SIG'] = station_data[sid]['SIG']
                f['properties']['IRR_AREA'] = station_data[sid]['IRR_AREA']
                f['properties']['SLOPE'] = station_data[sid]['SLOPE']
                features.append(f)

    meta['schema']['properties']['STANAME'] = 'str:254'
    meta['schema']['properties']['SIG'] = 'float:19.11'
    meta['schema']['properties']['IRR_AREA'] = 'float:19.11'
    meta['schema']['properties']['SLOPE'] = 'float:19.11'

    with fiona.open(out_shp, 'w', **meta) as dst:
        for f in features:
            dst.write(f)


if __name__ == '__main__':
    s, e = '1984-01-01', '2020-12-31'
    src = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'
    temp_src = '/media/research/IrrigationGIS/gages/hydrographs/daily_temp'
    dst = '/media/research/IrrigationGIS/gages/hydrographs/temp_q_bf_JAS'
    # shp = '/media/research/IrrigationGIS/gages/gage_loc_usgs/selected_gages.shp'
    # ishp = '/media/research/IrrigationGIS/gages/watersheds/combined_station_watersheds.shp'
    # oshp = '/media/research/IrrigationGIS/gages/watersheds/impacted_watersheds.shp'
    # jsn = '/media/research/IrrigationGIS/gages/station_metadata/impacted_julOct_bf.json'

    # get_station_daily_q(s, e, shp, src)
    get_station_daterange_data(1991, dst, src, 7, 9, temp_src)
    # write_impacted_stations(ishp, oshp, jsn)
# ========================= EOF ====================================================================
