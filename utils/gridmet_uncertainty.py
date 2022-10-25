import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils.gridmet_data import GridMet


def gridmet_ppt_error(stations, station_data, out_csv):
    sta_names, ct, bad_ct, all_sta = [], 0, 0, []
    pct_error = []
    adf = pd.DataFrame(columns=['name', 'basin', 'year', 'count', 'st_ppt', 'gm_ppt'])
    stations = pd.read_csv(stations)
    for i, r in stations.iterrows():

        lat, lon, staid, basin = r['LAT'], r['LON'], r['STAID'], r['basin']

        _file = os.path.join(station_data, '{}.csv'.format(staid))
        sta_df = pd.read_csv(_file, parse_dates=True, infer_datetime_format=True,
                             index_col='DATE')

        sta_df = sta_df.loc['1982-01-01': '2020-12-31']

        try:
            sta_df = sta_df['PRCP']
            sta_df /= 10.

        except KeyError as e:
            print(staid, 'missing', e)
            continue

        try:
            grd = GridMet(variable='pr', start='1982-01-01', end='2020-12-31',
                          lat=lat, lon=lon)
            grd_df = grd.get_point_timeseries()
            grd_df = grd_df.loc[sta_df.index]
            df = pd.concat([sta_df, grd_df], axis=1)
            df.dropna(how='any', axis=0, inplace=True)
        except Exception as e:
            print(e, staid, 'error')
            continue

        years = list(set([i.year for i in df.index]))
        for y in years:
            yr_idx = [x for x in df.index if x.year == y]
            if len(yr_idx) < 250:
                continue
            y_data = df.loc[yr_idx].sum(axis=0)
            ghcn, grdmet = y_data['PRCP'], y_data['pr']
            if ghcn == 0.0:
                continue
            adf = adf.append({'name': staid, 'basin': basin, 'count': len(yr_idx),
                              'year': y, 'st_ppt': ghcn, 'gm_ppt': grdmet},
                             ignore_index=True)

            error = abs((grdmet - ghcn) / ghcn)
            pct_error.append(error)
            mean_error = np.array(pct_error).mean()
            print('mean {:.3f} at {} station-years'.format(mean_error, len(pct_error)))
        adf.to_csv(out_csv.replace('.csv', '_.csv'))

    adf.to_csv(out_csv)

    print('count', ct)
    print('bad count', bad_ct)


def gridmet_etr_error(stations, station_data, out_csv):
    sta_names, ct, bad_ct, all_sta = [], 0, 0, []
    stations = pd.read_csv(stations)
    stations = stations.loc[(stations['Filename'] != 'removed') & (stations['nonAg'] == 1)]

    adf = pd.DataFrame(columns=['name', 'basin', 'year', 'count', 'st_etr', 'gm_etr'])
    ct = 0
    for i, r in stations.iterrows():
        name, basin = r['cleaned_st'], r['basin']
        lat, lon = r['Latitude'], r['Longitude']
        _file = os.path.join(station_data, r['Filename'])
        sta_df = pd.read_excel(_file)
        sta_df.index = sta_df.apply(lambda row: datetime(row['year'], row['month'], row['day']), axis=1)

        sta_df = sta_df[['ETr (mm)']]
        sta_df = sta_df.rename(columns={'ETr (mm)': 'st_etr'})
        sta_df = sta_df.dropna(how='any', axis=0)

        print('\nstation {}, {} records'.format(name, sta_df.shape[0]))

        try:
            etr = GridMet(variable='etr', start='1982-01-01', end='2020-12-31',
                          lat=lat, lon=lon)
            etr = etr.get_point_timeseries()
            etr = etr.loc[sta_df.index]
        except Exception as e:
            continue

        df = pd.concat([sta_df, etr], axis=1)
        df.dropna(how='any', axis=0, inplace=True)

        years = list(set([i.year for i in df.index]))
        for y in years:
            yr_idx = [x for x in df.index if x.year == y]
            y_data = df.loc[yr_idx].sum(axis=0)
            sta_etr, grdmet_etr = y_data['st_etr'], y_data['etr']
            etr_error = abs((grdmet_etr - sta_etr) / sta_etr)

            adf = adf.append({'name': name, 'basin': basin, 'year': y, 'count': len(yr_idx),
                              'st_etr': sta_etr, 'gm_etr': grdmet_etr}, ignore_index=True)
            print('{} mean annual etr err: {:.3f}%,'
                  '  at {} station-days'.format(y, etr_error, len(yr_idx)))

    adf.to_csv(out_csv)
    print('count', ct)
    print('bad count', bad_ct)


def get_rmse(csv, vars_=['st_ppt', 'gm_ppt'], basins=False):
    def _rmse(df):
        rmse = np.sqrt(((df[vars_[0]] - df[vars_[1]]) ** 2).mean())
        rmsep = np.sqrt((((df[vars_[0]] - df[vars_[1]]) / df[vars_[0]]) ** 2).mean())
        print('rmse {:.3f} mm, {:.3f}'.format(rmse, rmsep))

    df = pd.read_csv(csv)

    if basins:
        basins_ = list(set(df['basin']))
        for b in basins_:
            print(b)
            _rmse(df[df['basin'] == b])
    else:
        _rmse(df)


if __name__ == '__main__':
    root = os.path.join('/media', 'research', 'IrrigationGIS', 'climate')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'climate')

    stations_ = os.path.join(root, 'stations', 'study_basisns_ghcn_stations.csv')
    station_data_ = os.path.join(root, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    out_csv_ = os.path.join(root, 'ghcn', 'ghcn_gridmet_comp.csv')
    # gridmet_ppt_error(stations_, station_data_, out_csv_)
    out_csv_ = os.path.join(root, 'ghcn', 'ghcn_gridmet_com.csv')
    # get_rmse(out_csv_, vars_=['st_ppt', 'gm_ppt'])

    station_path_ = os.path.join(root, 'gridwxcomp', 'gridwxcomp_basins_all.csv')
    etr_station_data_ = os.path.join(root, 'gridwxcomp', 'station_data')
    etr_comp_csv = os.path.join(root, 'gridwxcomp', 'etr_comp.csv')
    # gridmet_etr_error(station_path_, etr_station_data_, etr_comp_csv)
    get_rmse(etr_comp_csv, vars_=['st_etr', 'gm_etr'], basins=False)
# ========================= EOF ====================================================================
