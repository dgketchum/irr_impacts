import os
import json
import fiona
import hydrofunctions as hf
from pandas import date_range, DatetimeIndex, DataFrame
from collections import OrderedDict
from hydrograph import hydrograph


def station_basin_designations(in_shp, out_meta):
    dct = {}
    with fiona.open(in_shp, 'r') as src:
        for f in src:
            if f['properties']['BASIN'] not in dct.keys():
                dct[f['properties']['BASIN']] = [f['properties']['STAID']]
                continue
            dct[f['properties']['BASIN']].append(f['properties']['STAID'])
    with open(out_meta, 'w') as fp:
        json.dump(dct, fp, indent=4)


def get_station_watersheds(stations_source, watersheds_source, out_stations, out_watersheds, station_json=None):
    sta_dct = {}
    with fiona.open(stations_source, 'r') as src:
        point_meta = src.meta
        all_stations = [f for f in src]
        station_ids = [f['properties']['STAID'] for f in src]
        station_meta = [(f['properties']['STANAME'].replace(',', ''), f['properties']['start'], f['properties']['end'])
                        for f in src]

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
                f['properties']['STANAME'] = attrs[0].upper().replace(',', '')
                f['properties']['start'] = attrs[1]
                f['properties']['end'] = attrs[2]
                f['properties']['AREA'] = f['properties']['SQMI'] / 0.386102
                selected_shapes.append(f)
                station_point['properties']['AREA'] = f['properties']['AREA']
                selected_points.append(station_point)

    meta['schema'] = {'properties': OrderedDict([('STAID', 'str:40'),
                                                 ('AREA', 'float:19.11'),
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
            else:
                coords = f['geometry']['coordinates']

            feature = {'geometry': {'coordinates': coords,
                                    'type': 'Polygon'},
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['SITE_NO']),
                                                  ('STANAME', f['properties']['STANAME']),
                                                  ('AREA', f['properties']['AREA']),
                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                sta_dct[f['properties']['SITE_NO']] = {k: v for k, v in f['properties'].items() if k != 'SITE_NO'}
            except TypeError:
                pass

    point_meta['schema']['properties']['AREA'] = 'float:19.11'
    with fiona.open(out_stations, 'w', **point_meta) as dst:
        for f in selected_points:
            dst.write(f)

    if station_json:
        with open(station_json, 'w') as fp:
            json.dump(sta_dct, fp, indent=4)

    print(len(selected_points))


def get_station_daily_data(param, start, end, stations_shapefile, out_dir, freq='dv'):
    if param not in ['00010', 'discharge']:
        raise ValueError('Use param = 00010 or discharge.')
    with fiona.open(stations_shapefile, 'r') as src:
        station_ids = [f['properties']['STAID'] for f in src]

    for sid in station_ids:
        try:
            nwis = hf.NWIS(sid, freq, start_date=start, end_date=end)
            df = nwis.df(param)
            if freq == 'iv':
                out_file = os.path.join(out_dir, '{}_{}.csv'.format(sid, start[:4]))
            else:
                out_file = os.path.join(out_dir, '{}.csv'.format(sid))
            df.to_csv(out_file)
            print(out_file)
        except ValueError as e:
            print(e)
        except hf.exceptions.HydroNoDataError:
            print('no data for {} to {}'.format(start, end))
            pass


def get_station_daterange_data(year_start, daily_q_dir, aggregate_q_dir, start_month=None, end_month=None,
                               resample_freq='A', convert_to_mcube=True):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    daterange = date_range(s, e, freq='D')
    idx = DatetimeIndex(daterange, tz=None)

    out_records, short_records = [], []
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
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(DataFrame.sum, skipna=False)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

    print('{} processed'.format(len(out_records)))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    src = os.path.join(root, 'hydrographs/daily_q')
    dst = os.path.join(root, 'hydrographs/q_monthly')
    get_station_daterange_data(1986, src, dst, resample_freq='M')
# ========================= EOF ====================================================================
