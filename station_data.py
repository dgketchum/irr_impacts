import os
import json
from copy import deepcopy

import fiona
import hydrofunctions as hf
import numpy as np
from pandas import date_range, DatetimeIndex, DataFrame
from collections import OrderedDict
from hydrograph import hydrograph
import matplotlib.pyplot as plt


class AmbiguousColumns(Exception):
    pass


class MissingValues(Exception):
    pass


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


def get_station_daily_data(param, start, end, out_dir, freq='dv', stations_shapefile=None,
                           plot_dir=None, targets=None, overwrite=False):
    if param not in ['00010', 'discharge']:
        raise ValueError('Use param = 00010 or discharge.')
    ct, successes = 0, []
    if stations_shapefile:
        with fiona.open(stations_shapefile, 'r') as src:
            station_ids = [f['properties']['STAID'] for f in src]
    elif targets:
        station_ids = targets
    else:
        raise NotImplementedError

    daily_values = len(date_range(start, end, freq='D'))
    for sid in station_ids:
        if sid != '13266000':
            continue
        if freq == 'iv':
            out_file = os.path.join(out_dir, '{}_{}.csv'.format(sid, start[:4]))
        else:
            out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            continue

        try:
            nwis = hf.NWIS(sid, freq, start_date=start, end_date=end)
            df = nwis.df(param)

            potential_q_cols = [x for x in list(df.columns) if '00060' in x]

            if len(potential_q_cols) == 0:
                raise AmbiguousColumns

            q_col = potential_q_cols[0]
            df.loc[:, 'Date'] = deepcopy(df.index)
            df = df.rename(columns={q_col: 'q'})
            df = df[['q', 'Date']]
            nan_count = np.count_nonzero(np.isnan(df['q']))
            if nan_count > 0:
                s = df['q']
                s = s.interpolate(limit=7, method='linear')
                if np.count_nonzero(np.isnan(s.values)) > 0:
                    raise MissingValues
                else:
                    df['q'] = deepcopy(s.values)

            if df.shape[0] < daily_values:
                raise MissingValues

            df.to_csv(out_file)
            ct += 1
            print(ct, out_file)

        except ValueError as e:
            print(e)
            continue
        except hf.exceptions.HydroNoDataError:
            print('no data for {} to {}'.format(start, end))
            continue
        except AmbiguousColumns:
            print(sid, 'ambiguous columns')
            continue
        except MissingValues:
            print(sid, 'missing values')
            continue

        if plot_dir:
            df.plot('Date', 'q')
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()


def get_station_daterange_data(year_start, daily_q_dir, aggregate_q_dir, start_month=None, end_month=None,
                               resample_freq='A', convert_to_mcube=True, plot_dir=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    daterange = date_range(s, e, freq='D')
    idx = DatetimeIndex(daterange, tz=None)

    out_records, short_records = [], []
    for sid, c in zip(sids, q_files):

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
        df = df['q']
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(DataFrame.sum, skipna=False)
        dates = deepcopy(df.index)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

        if plot_dir:
            pdf = DataFrame(data={'Date': dates, 'q': df.values})
            pdf.plot('Date', 'q')
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()

    print('{} processed'.format(len(out_records)))
    print(out_records)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages/'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages/'
    src = os.path.join(root, 'hydrographs/daily_q')
    stations_ = '/media/research/IrrigationGIS/gages/gage_loc_usgs/selected_gages.shp'
    ofig = os.path.join(root, 'figures/complete_daily_hydrographs')
    # get_station_daily_data('discharge', '1986-01-01', '2020-12-31', src, freq='dv', targets=TARGET_GAGES,
    #                        plot_dir=ofig, overwrite=False)
    dst = os.path.join(root, 'hydrographs/q_monthly')
    ofig = os.path.join(root, 'figures/complete_monthly_hydrographs')
    get_station_daterange_data(1986, src, dst, resample_freq='M', plot_dir=ofig)
# ========================= EOF ====================================================================
