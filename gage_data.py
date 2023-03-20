import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataretrieval.nwis as nwis

from utils.gage_lists import EXCLUDE_STATIONS, TARGET_STATIONS


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
    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]

    with open(stations, 'r') as f_obj:
        stations = json.load(f_obj)

    for sid, data in stations.items():

        if sid in EXCLUDE_STATIONS:
            continue

        out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            continue

        df = nwis.get_record(sites=sid, service='dv', start=start, end=end, parameterCd='00060')
        df = df.tz_convert(None)

        if df.empty:
            print(sid, ' is empty')
            continue

        q_col = '00060_Mean'
        df = df.rename(columns={q_col: 'q'})
        df = df.reindex(pd.DatetimeIndex(dt_range), axis=0)

        df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])
        nan_count = np.count_nonzero(np.isnan(df['q']))

        # exclude months without complete data
        if nan_count > 0:
            # df['q'] = df['q'].interpolate(limit=7, method='linear')
            df['q'] = df['q'].dropna(axis=0)
            record_ct = df['q'].groupby([df.index.year, df.index.month]).agg('count')
            records = [r for i, r in record_ct.items()]
            # add a zero to the front of the list to hack the resample, then remove
            mask = [0] + [int(a == b) for a, b in zip(records, counts)]
            missing_mo = len(counts) - sum(mask)
            resamp_start = pd.to_datetime(start) - pd.DateOffset(months=1)
            mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(resamp_start, end, freq='M')),
                             data=mask).resample('D').bfill()
            mask = mask[1:]
            df = df.loc[mask[mask == 1].index, 'q']
            print('write {:.1f}'.format(data['AREA']), sid, 'missing {} months'.format(missing_mo), data['STANAME'])
        else:
            df = df['q']
            print('write {:.1f}'.format(data['AREA']), sid, data['STANAME'])

        df.to_csv(out_file)

        if plot_dir:
            plt.plot(df.index, df)
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()


def get_station_daterange_data(daily_q_dir, aggregate_q_dir, resample_freq='A', convert_to_mcube=True,
                               plot_dir=None, reservoirs=None, res_js=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    out_records, short_records = [], []
    annual_q_test, diversion_test = [], []


    if reservoirs:
        with open(res_js, 'r') as f_obj:
            dct = json.load(f_obj)

        rfiles = [os.path.join(reservoirs, x) for x in os.listdir(reservoirs) if x.endswith('.csv')]
        rkeys = [int(x.strip('.csv')) for x in os.listdir(reservoirs) if x.endswith('.csv')]

    for sid, c in zip(sids, q_files):

        if sid not in TARGET_STATIONS:
            continue

        if sid != '13090000':
            continue

        _name = dct[sid]['STANAME']

        df = pd.read_csv(c, index_col=0, infer_datetime_format=True, parse_dates=True)

        # cfs to m ^3 d ^-1
        df = df['q']
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(pd.DataFrame.sum, skipna=False)
        dates = deepcopy(df.index)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))

        if reservoirs:
            rdf = pd.Series(index=df.index, data=[0 for _ in range(df.shape[0])])
            rlist = dct[sid]['res']

            for r in rlist:
                try:
                    rfile = rfiles[rkeys.index(r)]
                except ValueError:
                    continue

                c = pd.read_csv(rfile, index_col='date', infer_datetime_format=True, parse_dates=True)

                if 'inflow' in c.columns and 'outflow' in c.columns and not 'storage' in c.columns:
                    c['storage'] = c['outflow'] - c['inflow']

                if resample_freq == 'M':
                    c['delta_s'] = c['storage'].diff() * 1e6
                    print('{:.1f} {}'.format(np.nanmean([abs(s) for s in c['delta_s'].values]), r))
                elif resample_freq == 'A':
                    # doesn't need aggregation, just difference year-to-year
                    idx = [i for i in c.index if i.month == 12]
                    c = c.loc[idx].diff()
                    c['delta_s'] = c['storage'] * 1e6
                else:
                    raise NotImplementedError

                match = [i for i in c.index if i in rdf.index]
                c = c.loc[match]
                c = c.reindex(rdf.index)
                rdf += c['delta_s']

            rtest = pd.DataFrame(data=np.array([df.values, rdf.values]).T, columns=['flow', 'delta_s'], index=df.index)
            rtest['diff'] = rtest['delta_s'] / rtest['flow']

            if resample_freq == 'M':
                sept = [i for i in rtest.index if i.month == 9]
                sept_mean = rtest.loc[sept, 'diff'].mean()
                print('{:.1f}'.format(sept_mean), len(rlist), _name, sid)

            elif resample_freq == 'A':
                an_stor_delta = np.nanmean([abs(s) for s in rtest['diff'].values])
                if not np.isfinite(an_stor_delta):
                    diversion_test.append(sid)
                elif an_stor_delta > 0.03:
                    diversion_test.append(sid)
                elif an_stor_delta < 0.03:
                    annual_q_test.append(sid)
                else:
                    print(sid, 'did not find a place')
                    raise NotImplementedError

                print('{:.3f}'.format(an_stor_delta), len(rlist), _name, sid)

            df += rtest['delta_s']

        df.to_csv(out_file)
        out_records.append(sid)

        if plot_dir:
            pdf = pd.DataFrame(data={'Date': dates, 'q': df.values})
            pdf.plot('Date', 'q')
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()

    print('{} processed'.format(len(out_records)))
    print(out_records)

    print(annual_q_test)
    print(diversion_test)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
