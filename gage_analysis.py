import os
import json
from copy import copy
from collections import OrderedDict, Counter
from pprint import pprint

import numpy as np
from scipy.stats.stats import linregress
from pandas import DataFrame, read_csv, concat
from datetime import date
from dateutil.relativedelta import relativedelta as rdlt
import fiona
import statsmodels.api as sm

from hydrograph import hydrograph

os.environ['R_HOME'] = '/home/dgketchum/miniconda3/envs/renv/lib/R'
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.embedded import RRuntimeError

EXCLUDE_STATIONS = ['05015500', '06154400', '06311000', '06329590', '06329610', '06329620',
                    '09125800', '09131495', '09147022', '09213700', '09362800', '09398300',
                    '09469000', '09509501', '09509502', '12371550', '12415500', '12452000',
                    '13039000', '13106500', '13115000', '13119000', '13126000', '13142000',
                    '13148200', '13171500', '13174000', '13201500', '13238500', '13340950',
                    '14149000', '14150900', '14153000', '14155000', '14162100', '14168000',
                    '14180500', '14186100', '14186600', '14207740', '14207770', '14234800',
                    '12472600']

from gage_list import CLMB_STATIONS, UMRB_STATIONS

STATIONS = CLMB_STATIONS + UMRB_STATIONS


def daily_median_flow(daily_q_dir, metadata_in):
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    l.sort()
    d = {}
    with open(metadata_in, 'r') as f:
        meta = json.load(f)
    for c in l:
        try:
            sid = os.path.basename(c).split('.')[0]
            d[sid] = {}
            if sid not in meta.keys():
                continue
            df = hydrograph(c)
            s, e = df.index[0], df.index[-1]
            annual_df = None
            first = True
            for y in range(s.year, e.year + 1):
                ydf = copy(df['q'].loc['{}-01-01'.format(y): '{}-12-31'.format(y)])
                if ydf.shape[0] < 365:
                    continue
                ydf.name = 'q_{}'.format(y)
                months = [i.month for i, r in ydf.iteritems()]
                days = [i.day for i, r in ydf.iteritems()]
                if first:
                    base_idx = ['{}-{}'.format(m, d) for m, d in zip(months, days)]
                    ydf.index = base_idx
                    annual_df = ydf
                    first = False
                else:
                    idx = ['{}-{}'.format(m, d) for m, d in zip(months, days)]
                    ydf.index = idx
                    ydf = ydf.reindex(base_idx)
                    if np.count_nonzero(np.isnan(ydf.values)) > 0:
                        continue
                    else:
                        annual_df = concat([annual_df, ydf], axis=1)
        except Exception as e:
            print('error', sid, e)

        try:
            medians = annual_df.median(axis=1)
            d[sid]['max_mo'] = months[medians.values.argmax()]
            d[sid]['min_mo'] = months[medians.values.argmin()]
            d[sid]['max_median_q'] = medians.max()
            d[sid]['min_median_q'] = medians.min()
        except Exception as e:
            pass

    return d


def low_flow_months(monthly_q_dir, n_months=1):
    l = [os.path.join(monthly_q_dir, x) for x in os.listdir(monthly_q_dir)]
    d = {}
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        d[sid] = {}
        full_yr_rec = []
        df = hydrograph(c)
        if 'qb' not in list(df.columns):
            continue
        s, e = df.index[0], df.index[-1]
        lf_months = []

        for y in range(s.year, e.year + 1):
            ydf = copy(df.loc['{}-01-01'.format(y): '{}-12-31'.format(y)])
            if np.count_nonzero(np.isnan(ydf['qb'].values)) > 0:
                continue
            else:
                full_yr_rec.append(y)
            series = ydf['qb'].values
            idx = np.argpartition(series, n_months) + 1
            lf_months += list(idx[:m])
        hist = Counter(lf_months).most_common(n_months)
        target_months = sorted([int(x[0]) for x in hist])
        d[sid]['lf_m'] = target_months
        d[sid]['qb_years'] = full_yr_rec

    return d


def irrigation_metadata(csv):
    d = {}
    df = read_csv(csv)
    df['frac'] = [99 for x in range(df.shape[0])]
    df['sid'] = [str(c).rjust(8, '0') for c in df['STAID'].values]
    df.index = df['sid']
    for s, r in df.iterrows():
        d[s] = {}
        i = np.array([r[x] for x in r.index if 'irr' in x])
        if i.mean() > 1e-4:
            d[s]['mean_'], d[s]['std'], d[s]['norm_std'] = i.mean(), i.std(), i.std() / i.mean()
        else:
            d[s]['mean_'], d[s]['std'], d[s]['norm_std'] = 0, 0, 0
    return d


def get_basin_metadata(base_metadata, monthly_q_dir, daily_q_dir, irr_csv, metadata_out, months=3):
    with open(base_metadata, 'r') as f:
        metadata = json.load(f)
    dq = daily_median_flow(daily_q_dir, base_metadata)
    lf = low_flow_months(monthly_q_dir, n_months=months)
    irr = irrigation_metadata(irr_csv)
    new_metadata = {}
    for sid, v in metadata.items():
        try:
            mean_ = irr[sid]['mean_'] / (metadata[sid]['AREA_SQKM'] * 1e6)
            std = irr[sid]['std'] / (metadata[sid]['AREA_SQKM'] * 1e6)
            norm_std = irr[sid]['norm_std'] / (metadata[sid]['AREA_SQKM'] * 1e6)
            lowflo = lf[sid]['lf_m']
            qb_yrs = lf[sid]['qb_years']
            max_mo = dq[sid]['max_mo']
            min_mo = dq[sid]['min_mo']
            max_median_q = dq[sid]['max_median_q']
            min_median_q = dq[sid]['min_median_q']

            new_metadata[sid] = metadata[sid]

            new_metadata[sid]['max_mo'] = max_mo
            new_metadata[sid]['min_mo'] = min_mo
            new_metadata[sid]['max_median_q'] = max_median_q
            new_metadata[sid]['min_median_q'] = min_median_q
            new_metadata[sid]['irr_mean'] = mean_
            new_metadata[sid]['irr_std'] = std
            new_metadata[sid]['irr_normstd'] = norm_std
            new_metadata[sid]['lf_m'] = lowflo
            new_metadata[sid]['qb_years'] = qb_yrs

        except KeyError as e:
            print(sid, e)
            pass
    with open(metadata_out, 'w') as f:
        json.dump(new_metadata, f)


def get_baseflow(dir_, metadata_in, out_dir, metadata_out):
    pandas2ri.activate()
    r['source']('BaseflowSeparationFunctions.R')
    rec_const_r = robjects.globalenv['baseflow_RecessionConstant']
    bfi_max_r = robjects.globalenv['baseflow_BFImax']
    bf_eckhardt_r = robjects.globalenv['baseflow_Eckhardt']

    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]

    with open(metadata_in, 'r') as f:
        meta = json.load(f)

    for c in l:

        sid = os.path.basename(c).split('.')[0]
        if sid in EXCLUDE_STATIONS:
            print('exclude {}'.format(sid))
            continue

        if sid not in meta.keys():
            print(sid, 'not found')
            continue
        print(sid, meta[sid]['STANAME'])

        df = hydrograph(c)
        df.rename(columns={list(df.columns)[0]: 'q'}, inplace=True)

        keys = ('bfi_pr', 'k_pr')
        slices = [('1986-01-01', '2020-12-31')]

        for bk, s in zip(keys, slices):

            with localconverter(ro.default_converter + pandas2ri.converter):
                dfs = df['q'].loc[s[0]: s[1]]
                dfr = ro.conversion.py2rpy(dfs)
                dfs = DataFrame(dfs)

            try:
                k = rec_const_r(dfr)[0]
                bfi_max = bfi_max_r(dfr, k)[0]
                dfs['qb'] = bf_eckhardt_r(dfr, bfi_max, k)
                meta[sid].update({bk[0]: bfi_max, bk[1]: k})

            except RRuntimeError:
                print('error ', sid, meta[sid]['STANAME'])

            dfs.to_csv(os.path.join(out_dir, '{}.csv'.format(sid)))

    with open(metadata_out, 'w') as f:
        json.dump(meta, f)


def write_json_to_shapefile(in_shp, out_shp, meta):
    with open(meta, 'r') as f:
        meta = json.load(f)

    features = []
    gages = list(meta.keys())
    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        for f in src:
            sid = f['properties']['STAID']
            if sid in gages:
                features.append(f)

    shape_meta['schema']['properties']['lag'] = 'float:19.11'
    shape_meta['schema']['properties']['IRR_AREA'] = 'float:19.11'
    shape_meta['schema']['properties']['SLOPE'] = 'float:19.11'

    ct = 0
    with fiona.open(out_shp, 'w', **shape_meta) as dst:
        for f in features:
            ct += 1
            sid = f['properties']['STAID']
            feature = {'geometry': f['geometry'],
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                  ('STANAME', f['properties']['STANAME']),
                                                  ('SQMI', f['properties']['SQMI']),

                                                  ('lag', meta[sid]['lag']),
                                                  ('IRR_AREA', meta[sid]['IRR_AREA']),
                                                  ('SLOPE', meta[sid]['SLOPE']),

                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                print(f['properties']['STAID'], f['properties']['STANAME'])
            except TypeError:
                pass


def baseflow_correlation_search(climate_dir, in_json, out_json):
    """Find linear relationship between climate and flow in an expanding time window"""
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    offsets = [x for x in range(1, 36)]
    windows = {}
    lags = []
    month_series = [x for x in range(1, 13)] + [x for x in range(1, 13)]
    date_sample = None
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        try:
            sid = os.path.basename(csv).strip('.csv')
            if sid not in STATIONS:
                continue
            s_meta = metadata[sid]
            print('\n{}'.format(sid))
            max_q_mo, min_q_mo = s_meta['max_mo'], s_meta['min_mo']
            if min_q_mo < max_q_mo:
                recession_months = month_series[max_q_mo: min_q_mo + 12]
            else:
                recession_months = month_series[max_q_mo: min_q_mo + 1]
            print(max_q_mo, min_q_mo, recession_months)
            irr = s_meta['irr_mean']
            if irr == 0.0:
                continue
            df = hydrograph(csv)
            df['ai'] = (df['etr'] - df['ppt']) / (df['etr'] + df['ppt'])
            ai = df['ai']

            years = [x for x in range(1991, 2021)]
            flow_periods = [x for x in range(7, 11)]
            corr = (0, 0.0)
            found_sig = False
            for q_win in flow_periods:
                q_dates = [(date(y, q_win, 1), date(y, 11, 1)) for y in years]
                q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])
                for lag in offsets:
                    dates = [(date(y, 11, 1), date(y, 11, 1) + rdlt(months=-lag)) for y in years]
                    ind = [ai[d[1]: d[0]].sum() for d in dates]
                    lr = linregress(ind, q)
                    r, p = lr.rvalue, lr.pvalue
                    if abs(r) > corr[1] and p < 0.05:
                        corr = (lag, abs(r))
                        d = {'STANAME': s_meta['STANAME'], 'qb_month': m, 'q_start': q_win,
                             'lag': lag, 'r': r, 'pval': p, 'recession_months': recession_months}
                        date_sample = dates[0]
                        # print('month {}'.format(d['qb_month']), 'lag {}'.format(d['lag']), 'r {:.2f}'.format(d['r']),
                        #       '\np {:.2f}'.format(d['pval']), d['STANAME'], '{:.3f}'.format(irr), date_sample)
                        found_sig = True

            if not found_sig:
                print('no significant r at {}: {} ({} nan)'.format(sid, s_meta['STANAME'],
                                                                   np.count_nonzero(np.isnan(ai.values))))
                continue

            print('month {}'.format(d['qb_month']), 'lag {}'.format(d['lag']),
                  'q start {}'.format(d['q_start']), 'r {:.2f}'.format(d['r']),
                  '\np {:.2f}'.format(d['pval']), d['STANAME'], '{:.3f}'.format(irr), date_sample)

            windows[sid] = {**d, **s_meta}
            if irr > 0.01:
                lags.append(d['lag'])
        except Exception as e:
            print(csv, e)

    hist = sorted(Counter(lags).items())
    pprint(hist)
    print(sum([x[1] for x in hist]), ' stations')

    with open(out_json, 'w') as f:
        json.dump(windows, f)


if __name__ == '__main__':
    dq = '/media/research/IrrigationGIS/gages/hydrographs/daily_q'
    # i_json = '/media/research/IrrigationGIS/gages/station_metadata/metadata.json'
    # o_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_irr_lf.json'
    mq = '/media/research/IrrigationGIS/gages/hydrographs/q_bf_monthly'
    m = 1
    irr_data = '/media/research/IrrigationGIS/gages/ee_exports/series/extracts_comp_25AUG2021.csv'
    # get_basin_metadata(i_json, mq, dq, irr_data, o_json, m)

    clim_dir = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q'
    i_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_irr_lf.json'
    o_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_lag_recession_11OCT2021.json'
    baseflow_correlation_search(clim_dir, i_json, o_json)

    # i_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_metadata_31AUG2021.json'
    # i_shp = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds_meta.shp'
    # o_shp = '/media/research/IrrigationGIS/gages/watersheds/response_31AUG2021.shp'
    # write_json_to_shapefile(i_shp, o_shp, i_json)

# ========================= EOF ====================================================================
