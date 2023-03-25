import os
import json
from calendar import monthrange

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests

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


def process_resops_hydrographs(reservoirs, time_series, out_dir, start, end):
    adf = pd.read_csv(reservoirs)
    eom = pd.date_range(start, end, freq='M')
    for i, r in adf.iterrows():
        d = r.to_dict()
        sid = d['DAM_ID']

        ts_file = os.path.join(time_series, 'ResOpsUS_{}.csv'.format(sid))
        df = pd.read_csv(ts_file, index_col='date', infer_datetime_format=True, parse_dates=True)
        df = df.loc[start: end]
        series = df['storage']
        series = series.reindex(eom)
        series.dropna(inplace=True)
        ofile = os.path.join(out_dir, '{}.csv'.format(sid))
        series.to_csv(ofile, float_format='%.3f')
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


def complete_records(df, counts, start, end):
    df = df.interpolate(limit=7, method='linear')
    df = df.dropna(axis=0)
    record_ct = df.groupby([df.index.year, df.index.month]).agg('count')
    records = [r for i, r in record_ct.items()]
    mask = [int(a == b) for a, b in zip(records, counts)]
    missing = len(counts) - sum(mask)

    if df.index[0] > pd.to_datetime(start):
        resamp_start = df.index[0] - pd.DateOffset(months=1)
    else:
        resamp_start = pd.to_datetime(start) - pd.DateOffset(months=1)

    resamp_ind = pd.DatetimeIndex(pd.date_range(resamp_start, end, freq='M'))
    mask = mask + [0 for i in range(len(resamp_ind) - len(mask))]
    mask = pd.Series(index=resamp_ind, data=mask).resample('D').bfill()
    df = df.loc[[i for i, r in mask.items() if r and i in df.index]]
    return df, missing


if __name__ == '__main__':
    root = '/home/dgketchum/IrrigationGIS/expansion'

    # s, e = '1982-01-01', '2020-12-31'
    # csv_ = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/attributes/reservoir_flow_summary.csv'
    # res_gages = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/time_series_all'
    # processed = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/time_series_processed'
    # process_resops_hydrographs(csv_, res_gages, processed, s, e)

    s, e = '1982-01-01', '2021-12-31'
    resops_ = '/media/research/IrrigationGIS/impacts/reservoirs/resopsus/attributes/reservoir_flow_summary.shp'
    sites = '/media/research/IrrigationGIS/impacts/reservoirs/usbr/candidate_sites.csv'
    oshp = '/media/research/IrrigationGIS/impacts/reservoirs/usbr/reservoir_sites.shp'
    hyd = '/media/research/IrrigationGIS/impacts/reservoirs/usbr/hydrographs'
    get_reservoir_data(sites, resops_shp=processed, out_shp=oshp, out_dir=hyd, start=s, end=e)

# ========================= EOF ====================================================================
