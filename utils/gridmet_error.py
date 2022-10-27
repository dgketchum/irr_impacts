import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils.gridmet_data import GridMet


def gridmet_ppt_error(stations, station_data, out_csv, basin=None):
    adf = pd.DataFrame(columns=['name', 'basin', 'year', 'lat', 'lon', 'elev', 'count', 'st_ppt', 'gm_ppt', 'diffpct'])
    stations = pd.read_csv(stations)

    if basin:
        stations = stations[stations['basin'] == basin]

    len_ = stations.shape[0]
    for e, (i, r) in enumerate(stations.iterrows(), start=1):

        lat, lon, staid, basin, elev = r['LAT'], r['LON'], r['STAID'], r['basin'], r['ELEV']

        print('\n{}: {} of {} stations'.format(staid, e, len_))

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

            y_data = df.loc[yr_idx]

            # filter out potential non-detecting stations
            detections = np.count_nonzero(y_data['PRCP'] > 0.) / np.count_nonzero(y_data['pr'] > 0.)
            if detections < 0.9:
                continue

            y_data = y_data.sum(axis=0)
            ghcn, grdmet = y_data['PRCP'], y_data['pr']
            diffpct = (y_data['PRCP'] / y_data['pr']) / y_data['PRCP']
            if ghcn == 0.0:
                continue
            adf = adf.append({'name': staid, 'basin': basin, 'lat': lat, 'lon': lon,
                              'elev': elev, 'count': len(yr_idx),
                              'year': y, 'st_ppt': ghcn, 'gm_ppt': grdmet, 'diffpct': diffpct},
                             ignore_index=True)
        adf.to_csv(out_csv.replace('.csv', '_.csv'))
        get_rmse(adf)

    adf.to_csv(out_csv)


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


def get_rmse(csv, vars_=('st_ppt', 'gm_ppt'), basins=False):
    def _rmse(df_):
        rmse = np.sqrt(((df_[vars_[0]] - df_[vars_[1]]) ** 2).mean())
        rmsep = np.sqrt((((df_[vars_[0]] - df_[vars_[1]]) / df_[vars_[0]]) ** 2).mean())
        print('rmse {:.3f} mm, {:.3f}'.format(rmse, rmsep))

    if not isinstance(csv, pd.DataFrame):
        df = pd.read_csv(csv)
    else:
        df = csv

    df['diffpct'] = (df[vars_[0]] - df[vars_[1]]) / df[vars_[0]]
    df = df[df['diffpct'] < 1.]

    if basins:
        basins_ = list(set(df['basin']))
        for b in basins_:
            print(b)
            bdf = df[(df['basin'] == b) & (df['year'] > 1989)]
            _rmse(bdf)
    else:
        _rmse(df)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
