import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from io import StringIO

from gage_data import month_days_count


def read_gp_res70(url):
    skip = list(range(10)) + [11]
    txt = requests.get(url).content.decode('utf-8')
    lines = [l for i, l in enumerate(txt.splitlines()) if i not in skip]
    table = []
    for l in lines:
        if l.startswith(' Year'):
            table.append(l.split()[:13])
        try:
            _ = int(l.strip()[:4])
            table.append(l.split()[:13])
        except ValueError:
            continue
    try:
        df = pd.DataFrame(columns=table[0], data=table[1:])
        df = df.melt(id_vars=['Year'], var_name='Month', value_name='storage')
        df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
        df = df.set_index('date').drop(['Year', 'Month'], axis=1)
        df = df.sort_index()
        df['storage'] = pd.to_numeric(df['storage'], errors='coerce')
    except IndexError:
        return False

    return df


def read_pnw(url):
    df = pd.read_csv(url, infer_datetime_format=True, parse_dates=True, index_col='DateTime')
    df.columns = ['storage']
    df['storage'] = df['storage'].values.astype(float)
    if np.all(np.isnan(df['storage'].values)):
        return False
    return df


def get_reservoir_data(csv, out_shp, out_dir, resops_shp, start, end):
    resops = gpd.read_file(resops_shp)
    counts = month_days_count(start, end)

    ucrb_url = 'https://www.usbr.gov/uc/water/hydrodata/reservoir_data/{}/csv/{}.csv'
    ucrb_keys = {'inflow': '29', 'outflow': '42', 'storage': '17',
                 'columns': {'datetime': 'date', 'storage': 'storage'}}

    gp_url = 'https://www.usbr.gov/gp-bin/res070.pl?station={}&parameters={}&byear=1995&eyear=2015'
    gp_keys = {'units': 'af', 'storage': 'AF.EOM', 'inflow': 'AF.IN', 'outflow': 'AF.QD',
               'columns': {'date': 'date', 'storage': 'storage'}}

    pnw_url = 'https://www.usbr.gov/pn-bin/daily.pl?station={}&' \
              'format=csv&year=1987&month=1&day=1&year=2021&month=12&day=31&pcode={}'
    pnw_keys = {'inflow': '29', 'outflow': '42', 'storage': 'af',
                'columns': {'datetime': 'date', 'storage': 'storage'}}

    regions = {'UCRB': (ucrb_url, ucrb_keys), 'GP': (gp_url, gp_keys),
               'CPN': (pnw_url, pnw_keys)}

    stations = pd.read_csv(csv).sample(frac=1)
    shp_cols = list(stations.columns) + ['s_rng', 's_95', 's_05']
    shp = pd.DataFrame(columns=shp_cols)
    ct = 0
    for i, r in stations.iterrows():
        sid, _name, region = r['STAID'], r['STANAME'], r['REGION']

        map = regions[region]
        url, key = map[0], map[1]['storage']
        url = url.format(sid, key)

        if region == 'GP':
            df = read_gp_res70(url)
            if df is False:
                print('{}: {} failed'.format(sid, _name))
                continue
        elif region == 'UCRB':
            df = pd.read_csv(url)
        else:
            df = read_pnw(url)
            if df is False:
                print('{}: {} failed'.format(sid, _name))
                continue

        df['storage'] = df['storage'] * 1233.48
        df = df.loc['1982-01-01': '2021-12-31']
        ofile = os.path.join(out_dir, '{}.csv'.format(sid))
        df.to_csv(ofile)

        try:
            q95, q05 = np.nanpercentile(df['storage'], [95, 5])
            s_range = q95 - q05
            dct = r.to_dict()
            dct['s_rng'] = s_range
            dct['s_95'] = q95
            dct['s_05'] = q05
            shp = shp.append(dct, ignore_index=True)
            ct += 1
            print(sid, _name, '{:.1f}'.format(s_range / 1e6))
        except Exception as e:
            print(e, sid, _name)
            continue

    shp['geometry'] = shp.apply(lambda x: Point(x['LONGITUDE'], x['LATITUDE']), axis=1)
    shp = gpd.GeoDataFrame(shp, crs='EPSG:4326')
    shp.to_file(out_shp)


def read_mean_join_csvs(reservoirs, in_q, storage, out_q, shp_path):
    gdf = pd.read_csv(reservoirs, index_col='DAM_ID', engine='python')
    gdf['geometry'] = gdf[['LONG', 'LAT']].apply(lambda x: Point(x['LONG'], x['LAT']), axis=1)
    gdf = gpd.GeoDataFrame(gdf, crs='EPSG:4326')

    gdf.drop(columns=['INCONSISTENCIES_NOTED'], inplace=True)
    in_q = pd.read_csv(in_q, index_col='date', infer_datetime_format=True, parse_dates=True)
    storage = pd.read_csv(storage, index_col='date', infer_datetime_format=True, parse_dates=True)
    out_q = pd.read_csv(out_q, index_col='date', infer_datetime_format=True, parse_dates=True)

    # incoming data in [m3, m3, million m3]
    items = zip(['inflow', 'outflow', 'storage'], [in_q, out_q, storage])

    for t, df in items:
        df = df.loc['1984-01-01':]
        for c in df.columns:
            s = df[c].values
            s = s.astype(float)
            s[s < -10] = np.nan
            s[s < 0] = 0
            _recs = np.count_nonzero(~np.isnan(s))
            if _recs / len(s) < 0.4:
                gdf.loc[int(c), t] = 'False'
                continue
            else:
                gdf.loc[int(c), t] = 'True'
            _mean = np.nanmean(s)
            gdf.loc[int(c), '{}_r'.format(t)] = _recs
            gdf.loc[int(c), '{}_m'.format(t)] = _mean
            if t == 's':
                d = gdf.loc[int(c)]
                if d['out_q_recs'] > d['in_q_recs']:
                    gdf.loc[int(c), 'pref'] = 'out_q'
                elif d['in_q_recs'] > 1000 and d['s_recs'] > 1000:
                    gdf.loc[int(c), 'pref'] = 'in_q'
                else:
                    gdf.loc[int(c), 'pref'] = 'none'
                    continue

                q95, q05 = np.nanpercentile(s, [95, 5])
                s_range = q95 - q05

                ann_q = np.nanmean(s) * 60 * 60 * 24 * 365 / 1e6

                gdf.loc[int(c), 'q_an'] = ann_q
                gdf.loc[int(c), 's_rng'] = s_range
                gdf.loc[int(c), 's_95'] = q95
                gdf.loc[int(c), 's_05'] = q05
                gdf.loc[int(c), 's_rat'] = s_range / ann_q

    gdf.to_file(shp_path)
    df = pd.DataFrame(gdf).drop(columns=['geometry'])
    csv_path = shp_path.replace('.shp', '.csv')
    df.to_csv(csv_path, float_format='%.3f')


def process_resops_hydrographs(reservoirs, time_series, out_dir, start, end):
    adf = pd.read_csv(reservoirs)
    counts = month_days_count(start, end)
    for i, r in adf.iterrows():
        d = r.to_dict()
        sid = d['DAM_ID']

        ts_file = os.path.join(time_series, 'ResOpsUS_{}.csv'.format(sid))
        df = pd.read_csv(ts_file, index_col='date', infer_datetime_format=True, parse_dates=True)
        df = df.loc['1982-01-01':]

        odf = pd.DataFrame()

        for c in ['storage']:

            if d[c] == 'False':
                continue

            try:
                nan_count = np.count_nonzero(np.isnan(df[c]))
                df, match, missing_mo = complete_records(df, c, counts, start, end)
                odf[c] = df.loc[match, c]

                if nan_count > 0:

                    print(sid, c, 'missing {} months'.format(missing_mo), d['DAM_NAME'], d['STATE'])
                else:
                    odf[c] = df[c]

                if c == 'storage':
                    odf[c] *= 1e6
                elif c in ['inflow', 'outflow']:
                    odf[c] = df[c] * 86400

            except ValueError as e:
                print(sid, e)
                continue

        try:
            odf = odf.resample('M').agg({'outflow': 'sum',
                                         'inflow': 'sum',
                                         'storage': 'mean'})
        except Exception as e:
            print(sid, e)
            continue

        ofile = os.path.join(out_dir, '{}.csv'.format(sid))
        odf.to_csv(ofile, float_format='%.3f')
        print(sid, d['DAM_NAME'], d['STATE'])


def join_reservoirs_to_basins(basins, reservoirs, out_json):
    basins = gpd.read_file(basins)
    reservoirs = gpd.read_file(reservoirs)
    res_geo = [r['geometry'] for i, r in reservoirs.iterrows()]
    res_id = [r['DAM_ID'] for i, r in reservoirs.iterrows()]
    dct = {r['STAID']: [] for i, r in basins.iterrows()}
    for i, r in basins.iterrows():
        g = r['geometry']
        for j, res_g in enumerate(res_geo):
            if res_g.within(g):
                dct[r['STAID']].append(res_id[j])

    with open(out_json, 'w') as f:
        json.dump(dct, f, indent=4)


def complete_records(df, column, counts, start, end):
    df[column] = df[column].interpolate(limit=7, method='linear')
    df[column] = df[column].dropna(axis=0)
    record_ct = df[column].groupby([df.index.year, df.index.month]).agg('count')
    records = [r for i, r in record_ct.items()]
    mask = [0] + [int(a == b) for a, b in zip(records, counts)]
    missing = len(counts) - sum(mask)

    if df.index[0] > pd.to_datetime(start):
        resamp_start = df.index[0] - pd.DateOffset(months=1)
    else:
        resamp_start = pd.to_datetime(start) - pd.DateOffset(months=1)

    mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(resamp_start, end, freq='M')),
                     data=mask).resample('D').bfill()
    mask = mask[1:]
    match = [i for i in mask.index if i in df.index]
    return df, match, missing


if __name__ == '__main__':
    root = '/home/dgketchum/IrrigationGIS/expansion'

    start_yr, end_yr = 1982, 2020
    s, e = '{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr)
    csv_ = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/attributes/reservoir_flow_summary.csv'
    res_gages = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/time_series_all'
    processed = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/time_series_processed'
    process_resops_hydrographs(csv_, res_gages, processed, s, e)

    sites = '/media/research/IrrigationGIS/impacts/reservoirs/usbr/candidate_sites.csv'
    oshp = '/media/research/IrrigationGIS/impacts/reservoirs/usbr/reservoir_sites.shp'
    hyd = '/media/research/IrrigationGIS/impacts/reservoirs/usbr/hydrographs'
    # get_reservoir_data(sites, resops_shp=processed, out_shp=oshp, out_dir=hyd, start=s, end=e)

# ========================= EOF ====================================================================
