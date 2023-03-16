import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from gage_data import get_station_daily_data, get_station_daterange_data


def read_mean_join_csvs(reservoirs, in_q, storage, out_q, shp_path):
    gdf = pd.read_csv(reservoirs, index_col='DAM_ID', engine='python')
    gdf['geometry'] = gdf[['LONG', 'LAT']].apply(lambda x: Point(x['LONG'], x['LAT']), axis=1)
    gdf = gpd.GeoDataFrame(gdf, crs='EPSG:4326')

    gdf.drop(columns=['INCONSISTENCIES_NOTED'], inplace=True)
    in_q = pd.read_csv(in_q, index_col='date', infer_datetime_format=True, parse_dates=True)
    storage = pd.read_csv(storage, index_col='date', infer_datetime_format=True, parse_dates=True)
    out_q = pd.read_csv(out_q, index_col='date', infer_datetime_format=True, parse_dates=True)

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


def process_reservoir_hydrographs(reservoirs, time_series, out_dir, start, end):
    adf = pd.read_csv(reservoirs)

    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]

    for i, r in adf.iterrows():
        d = r.to_dict()
        sid = d['DAM_ID']
        ts_file = os.path.join(time_series, 'ResOpsUS_{}.csv'.format(sid))
        df = pd.read_csv(ts_file, index_col='date', infer_datetime_format=True, parse_dates=True)
        df = df.loc['1984-01-01':]

        odf = pd.DataFrame()

        for c in ['storage', 'inflow', 'outflow']:

            if d[c] == 'False':
                continue

            try:
                nan_count = np.count_nonzero(np.isnan(df[c]))

                if nan_count > 0:
                    df[c] = df[c].interpolate(limit=7, method='linear')
                    df[c] = df[c].dropna(axis=0)
                    record_ct = df[c].groupby([df.index.year, df.index.month]).agg('count')
                    records = [r for i, r in record_ct.items()]
                    mask = [0] + [int(a == b) for a, b in zip(records, counts)]
                    missing_mo = len(counts) - sum(mask)

                    if df.index[0] > pd.to_datetime(start):
                        resamp_start = df.index[0] - pd.DateOffset(months=1)
                    else:
                        resamp_start = pd.to_datetime(start) - pd.DateOffset(months=1)

                    mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(resamp_start, end, freq='M')),
                                     data=mask).resample('D').bfill()
                    mask = mask[1:]
                    match = [i for i in mask.index if i in df.index]
                    odf[c] = df.loc[match, c]
                    print(sid, c, 'missing {} months'.format(missing_mo), d['DAM_NAME'], d['STATE'])
                else:
                    odf[c] = df[c]

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

        ofile = os.path.join(processed, '{}.csv'.format(sid))
        odf.to_csv(ofile, float_format='%.3f')
        print(sid, d['DAM_NAME'], d['STATE'])


if __name__ == '__main__':
    root = '/home/dgketchum/IrrigationGIS/expansion'

    tables = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/time_series_single_variable_table'
    res_attr = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/attributes'
    res_attr_csv = os.path.join(res_attr, 'reservoir_attributes.csv')
    a = os.path.join(tables, 'DAILY_AV_INFLOW_CUMECS.csv')
    b = os.path.join(tables, 'DAILY_AV_STORAGE_MCM.csv')
    c = os.path.join(tables, 'DAILY_AV_OUTFLOW_CUMECS.csv')
    shp_ = os.path.join(res_attr, 'reservoir_flow_summary.shp')
    # read_mean_join_csvs(res_attr_csv, a, b, c, shp_)

    start_yr, end_yr = 1987, 2020
    csv_ = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/attributes/reservoir_flow_summary.csv'
    res_gages = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/time_series_all'
    processed = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/time_series_processed'
    process_reservoir_hydrographs(csv_, res_gages, processed, '{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr))

    monthly_q = os.path.join(root, 'tables', 'hydrographs', 'monthly_q')
    months = list(range(1, 13))
    select = '13090500'
    gages_metadata = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
    daily_q = os.path.join(root, 'tables', 'hydrographs', 'daily_q')
    get_station_daily_data('{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr), gages_metadata,
                           daily_q, plot_dir=None, overwrite=False)
    get_station_daterange_data(daily_q, monthly_q, convert_to_mcube=True, resample_freq='M')
# ========================= EOF ====================================================================
