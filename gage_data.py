import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataretrieval.nwis as nwis

from utils.gage_lists import TARGET_IRRIGATED_GAGES


class MissingValues(Exception):
    pass


def hydrograph(c):
    df = pd.read_csv(c)

    if 'Unnamed: 0' in list(df.columns):
        df = df.rename(columns={'Unnamed: 0': 'dt'})
    try:
        df['dt'] = pd.to_datetime(df['dt'])
    except KeyError:
        df['dt'] = pd.to_datetime(df['datetimeUTC'])
    df = df.set_index('dt')
    try:
        df.drop(columns='datetimeUTC', inplace=True)
    except KeyError:
        pass
    try:
        df = df.tz_convert(None)
    except:
        pass

    if 'USGS' in list(df.columns)[0]:
        df = df.rename(columns={list(df.columns)[0]: 'q'})

    return df


def get_station_daily_data(start, end, stations, out_dir, plot_dir=None, overwrite=False):
    ct = 0

    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]

    with open(stations, 'r') as f_obj:
        stations = json.load(f_obj)

    for sid, data in stations.items():

        out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            continue

        df = nwis.get_record(sites=sid, service='dv', start=start, end=end, parameterCd='00060')
        if df.empty:
            print(sid, ' is empty')
            continue

        q_col = '00060_Mean'
        df = df.rename(columns={q_col: 'q'})
        df = df.reindex(pd.DatetimeIndex(dt_range, tz='UTC'), axis=0)

        df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])
        nan_count = np.count_nonzero(np.isnan(df['q']))

        # exclude months without complete data
        if nan_count > 0:
            df['q'] = df['q'].interpolate(limit=7, method='linear')
            df['q'] = df['q'].dropna(axis=0)
            record_ct = df['q'].groupby([df.index.year, df.index.month]).agg('count')
            records = [r for i, r in record_ct.items()]
            mask = [int(a == b) for a, b in zip(records, counts)]
            print('missing {} months'.format(len(counts) - sum(mask)))
            mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(start, end, freq='M'), tz='UTC'),
                             data=mask).resample('D').bfill()
            df = df.loc[mask[mask == 1].index, 'q']
        else:
            df = df['q']

        df.to_csv(out_file)
        ct += 1
        print(ct, 'write {:.1f}'.format(data['AREA']),
              data['STANAME'])

        if plot_dir:
            plt.plot(df.index, df)
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()



def get_station_daterange_data(daily_q_dir, aggregate_q_dir, resample_freq='A', convert_to_mcube=True, plot_dir=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    out_records, short_records = [], []
    for sid, c in zip(sids, q_files):

        df = pd.read_csv(c, index_col=0, infer_datetime_format=True, parse_dates=True)

        # cfs to m ^3 d ^-1
        df = df['q']
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(pd.DataFrame.sum, skipna=False)
        dates = deepcopy(df.index)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

        if plot_dir:
            pdf = pd.DataFrame(data={'Date': dates, 'q': df.values})
            pdf.plot('Date', 'q')
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()

    print('{} processed'.format(len(out_records)))
    print(out_records)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
