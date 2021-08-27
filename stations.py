import os
import fiona
import hydrofunctions as hf
from pandas import date_range, DatetimeIndex, DataFrame
from collections import OrderedDict
from gage_analysis import EXCLUDE_STATIONS
from hydrograph import hydrograph


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
        # TODO: investigate why 09196500 returns data 1991 - ~1996 while https://nwis.waterdata.usgs.gov/wy/nwis/uv/?cb_00010=on&format=gif_default&site_no=09196500&period=&begin_date=1991-01-01&end_date=2020-12-31 does not
        # if sid != '09196500':
        #     continue
        if sid in EXCLUDE_STATIONS:
            continue
        nwis = hf.NWIS(sid, 'dv', start_date=start, end_date=end)
        try:
            df = nwis.df(param)
            out_file = os.path.join(out_dir, '{}.csv'.format(sid))
            df.to_csv(out_file)
            print(out_file)
        except ValueError as e:
            print(e)


def get_station_daterange_data(year_start, daily_q_dir, aggregate_q_dir, start_month=None, end_month=None,
                               resample_freq='A'):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]

    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    daterange = date_range(s, e, freq='D')
    idx = DatetimeIndex(daterange, tz=None)

    out_records, short_records = [], []
    full_ct, partial_ct, suspect_ct, err_ct = 0, 0, 0, 0
    for c in q_files:
        sid = os.path.basename(c).split('.')[0]
        df = hydrograph(c)

        if start_month or end_month:
            idx_window = idx[idx.month.isin([x for x in range(start_month, end_month + 1)])]
            df = df[df.index.month.isin([x for x in range(start_month, end_month + 1)])]
            df = df[df.index.year.isin([x for x in range(year_start, 2021)])]
            idx = idx_window

        dflen, idxlen = df.shape[0], idx.shape[0]
        if dflen < idxlen:
            short_records.append(sid)
            if float(dflen) / idxlen < 0.8:
                print(sid, 'df: {}, idx: {}, q skipped'.format(df.shape[0], int(idx.shape[0])))
                continue
            df = df.reindex(idx)

        # cfs to m ^3 d ^-1
        df = df * 2446.58
        df = df.resample(resample_freq).agg(DataFrame.sum, skipna=False)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

    print('{} processed'.format(len(out_records)))


if __name__ == '__main__':

    src = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'
    dst = '/media/research/IrrigationGIS/gages/hydrographs/q_bf_monthly'
    get_station_daterange_data(1986, src, dst, resample_freq='M')
# ========================= EOF ====================================================================
