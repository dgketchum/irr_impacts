import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def read_mean_join_csvs(reservoirs, in_q, storage, out_q, shp_path):
    gdf = pd.read_csv(reservoirs, index_col='DAM_ID', engine='python')
    gdf['geometry'] = gdf[['LONG', 'LAT']].apply(lambda x: Point(x['LONG'], x['LAT']), axis=1)
    gdf = gpd.GeoDataFrame(gdf, crs='EPSG:4326')

    gdf.drop(columns=['INCONSISTENCIES_NOTED'], inplace=True)
    in_q = pd.read_csv(in_q, index_col='date', infer_datetime_format=True, parse_dates=True)
    storage = pd.read_csv(storage, index_col='date', infer_datetime_format=True, parse_dates=True)
    out_q = pd.read_csv(out_q, index_col='date', infer_datetime_format=True, parse_dates=True)

    items = zip(['in_q', 'out_q', 's'], [in_q, out_q, storage])

    for t, df in items:
        df = df.loc['1984-01-01':]
        for c in df.columns:
            s = df[c].values
            s = s.astype(float)
            s[s < -10] = np.nan
            s[s < 0] = 0
            _recs = np.count_nonzero(~np.isnan(s))
            _mean = np.nanmean(s)
            gdf.loc[int(c), '{}_recs'.format(t)] = _recs
            gdf.loc[int(c), '{}_mean'.format(t)] = _mean
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


if __name__ == '__main__':
    tables = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/time_series_single_variable_table'
    res_attr = '/home/dgketchum/Downloads/ResOpsUS/ResOpsUS/attributes'
    res_attr_csv = os.path.join(res_attr, 'reservoir_attributes.csv')
    a = os.path.join(tables, 'DAILY_AV_INFLOW_CUMECS.csv')
    b = os.path.join(tables, 'DAILY_AV_STORAGE_MCM.csv')
    c = os.path.join(tables, 'DAILY_AV_OUTFLOW_CUMECS.csv')
    shp_ = os.path.join(res_attr, 'reservoir_flow_summary.shp')
    read_mean_join_csvs(res_attr_csv, a, b, c, shp_)
# ========================= EOF ====================================================================
