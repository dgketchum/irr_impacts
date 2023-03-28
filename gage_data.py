import json
import os

import dataretrieval.nwis as nwis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.gage_lists import EXCLUDE_RESERVOIRS


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


def month_days_count(start, end):
    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]
    return counts, dt_range


def get_station_daily_data(start, end, stations, out_dir, plot_dir=None, overwrite=False):
    counts, dt_range = month_days_count(start, end)
    with open(stations, 'r') as f_obj:
        stations = json.load(f_obj)

    for sid, data in stations.items():

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


def get_station_monthly_data(daily_q_dir, aggregate_q_dir, metadata, reservoirs, interbasin,
                             convert_to_mcube=True, plot_dir=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    out_records, short_records = [], []

    with open(metadata, 'r') as f_obj:
        dct = json.load(f_obj)

    rfiles = [os.path.join(reservoirs, x) for x in os.listdir(reservoirs) if x.endswith('.csv')]
    rkeys = [int(x.strip('.csv')) for x in os.listdir(reservoirs) if x.endswith('.csv')]

    ibt_files = [os.path.join(interbasin, x) for x in os.listdir(interbasin) if x.endswith('.csv')]
    ibt_keys = [int(x.strip('.csv')) for x in os.listdir(interbasin) if x.endswith('.csv')]

    for sid, c in zip(sids, q_files):

        _name = dct[sid]['STANAME']
        print('\n', _name)

        df = pd.read_csv(c, index_col=0, infer_datetime_format=True, parse_dates=True)

        # cfs to m ^3 d ^-1
        df = df['q']
        if convert_to_mcube:
            df = df * 2446.58

        df = df.resample('M').agg(pd.DataFrame.sum, skipna=False)
        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        cols = ['delta_s']
        rdf = pd.DataFrame(index=df.index, columns=cols)
        rlist = [int(k) for k in dct[sid]['res']]

        for r in rlist:

            if str(r) in EXCLUDE_RESERVOIRS:
                continue

            try:
                rfile = rfiles[rkeys.index(r)]
            except ValueError:
                continue

            c = pd.read_csv(rfile, index_col=0, infer_datetime_format=True, parse_dates=True)

            c['delta_s'] = c['storage'].diff()
            q95, q05 = np.nanpercentile(c['storage'], [75, 25])
            s_range = q95 - q05
            s_range_af = s_range / 1233.48
            match = [i for i, r in c.iterrows() if i in rdf.index and not np.isnan(r['storage'])]
            c = c.loc[match]
            c = c.reindex(rdf.index)
            c = c['delta_s']

            if len(match) < df.shape[0] and s_range_af < 20000:
                print('{} of {} res {} active: {:1f} af, backfilling'.format(len(match), len(rdf.index), r, s_range_af))
                monthly_means = c.copy().groupby(c.index.month).mean()
                for idx in c.index:
                    if pd.isnull(c[idx]):
                        c[idx] = monthly_means[idx.month]

            elif len(match) < df.shape[0] and s_range_af > 100000:
                if len(rlist) > 1:
                    print('missing {} of {} at {}, ignoring'.format(df.shape[0] - len(match), df.shape[0], r))

                else:
                    print('missing {} of {} at {}, not backfilling'.format(df.shape[0] - len(match), df.shape[0], r))

            else:
                print('{} of {} res {} mean active: {:1f} af'.format(len(match), len(rdf.index), r, s_range_af))

            stack = np.stack([rdf['delta_s'].values, c], axis=1)
            rdf.loc[df.index, 'delta_s'] = np.nansum(stack, axis=1)

        df = pd.DataFrame(df)
        df['delta_s'] = rdf['delta_s']

        ibt_list = [int(k) for k in dct[sid]['ibt']]
        for ibt in ibt_list:
            try:
                ibt_file = ibt_files[ibt_keys.index(ibt)]
            except ValueError:
                continue
            c = pd.read_csv(ibt_file, index_col=0, infer_datetime_format=True, parse_dates=True)
            match = [i for i, r in c.iterrows() if i in df.index and not np.isnan(r['q'])]
            c = c['q']

            if len(match) < df.shape[0]:
                print('{} of {} res {}, backfilling'.format(len(match), len(rdf.index), ibt))
                monthly_means = c.copy().groupby(c.index.month).mean()
                for idx in c.index:
                    if pd.isnull(c[idx]):
                        c[idx] = monthly_means[idx.month]

            else:
                print('{} of {} on canal {}'.format(len(match), len(df.index), ibt))

            df.loc[match, 'q'] -= c

        df.to_csv(out_file)
        out_records.append(sid)

        if plot_dir:
            df = df.loc['2000-01-01':'2004-12-31']
            plt.plot(df.index, df['q'], label='observed')
            df['adjusted'] = df['q'] + df['delta_s']
            plt.plot(df.index, df['adjusted'], label='adjusted')
            plt.legend()
            plt.suptitle(_name)
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
