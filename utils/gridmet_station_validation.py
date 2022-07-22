import os

import numpy as np
import fiona
from shapely.geometry import shape
from pandas import concat, read_csv, to_datetime, date_range

from utils.thredds import GridMet


def compare_gridmet_ghcn(stations, station_data, out_dir):
    sta_names, ct, bad_ct, all_sta = [], 0, 0, []
    with fiona.open(stations, 'r') as src:
        for f in src:
            props = f['properties']
            geo = shape(f['geometry'])
            sta_names.append((props['STAID'], geo.y, geo.x))

    pct_error = []
    for i, sta in enumerate(sta_names):

        lat, lon, staid = sta[1], sta[2], sta[0]

        _file = os.path.join(station_data, '{}.csv'.format(staid))
        sta_df = read_csv(_file, parse_dates=True, infer_datetime_format=True,
                          index_col='DATE')

        if sta_df.index[-1] < to_datetime('2021-01-01'):
            bad_ct += 1
            continue

        if sta_df.index[0] < to_datetime('1991-01-01'):
            continue

        sta_df = sta_df.loc['1991-01-01': '2020-12-31']
        full_yr_idx, years_covered = [], []
        for year in range(1991, 2021):
            s, e = '{}-01-01'.format(year), '{}-12-31'.format(year)
            y_idx = date_range(s, e)
            _chunk = sta_df.loc[s: e]
            if _chunk.shape[0] < len(y_idx):
                continue
            else:
                full_yr_idx += y_idx
                years_covered.append(year)

        if not full_yr_idx:
            print(staid, 'no full years')
            bad_ct += 1
            continue
        else:
            sta_df = sta_df.loc[full_yr_idx]

        try:
            sta_df = sta_df['PRCP']
            sta_df /= 10.

            if sta_df.empty or sta_df.shape[0] < 365:
                print(staid, 'insuf records in date range')
                continue

        except KeyError as e:
            print(staid, 'missing', e)
            continue

        grd = GridMet(variable='pr', start='1991-01-01', end='2020-12-31',
                      lat=lat, lon=lon)
        grd_df = grd.get_point_timeseries()
        grd_df = grd_df.loc[full_yr_idx]
        df = concat([sta_df, grd_df], axis=1)
        df.dropna(how='any', axis=0, inplace=True)
        for y in years_covered:
            y_data = df.loc[[x for x in df.index if x.year == y]].sum(axis=0)
            ghcn, grdmet = y_data['PRCP'], y_data['pr']
            if ghcn == 0.0:
                continue
            error = abs((grdmet - ghcn) / ghcn)
            pct_error.append(error)
            mean_error = np.array(pct_error).mean()
            print('mean {:.3f} at {} station-years'.format(mean_error, len(pct_error)))

    print('count', ct)
    print('bad count', bad_ct)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/'

    stations_ = os.path.join(d, 'stations', 'ghcn_study_basins.shp')
    station_data_ = os.path.join(d, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    out_dir_ = os.path.join(d, 'ghcn', 'ghcn_gridmet_comp')
    compare_gridmet_ghcn(stations_, station_data_, out_dir_)
# ========================= EOF ====================================================================
