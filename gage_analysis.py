import os
import json
from copy import copy
import numpy as np
from pandas import to_datetime, read_csv, date_range, DatetimeIndex
from datetime import datetime as dt, datetime
from dateutil.rrule import rrule, DAILY
from matplotlib import pyplot as plt

os.environ['R_HOME'] = '/home/dgketchum/miniconda3/envs/renv/lib/R'
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.embedded import RRuntimeError

def hydrograph(c):
    df = read_csv(c)
    df['datetimeUTC'] = to_datetime(df['datetimeUTC'])
    df = df.set_index('datetimeUTC')
    df.rename(columns={list(df.columns)[0]: 'q'}, inplace=True)
    return df


def peak_q_doy(daily_q_dir):
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    for c in l:
        df = hydrograph(c)
        s, e = df['datetimeUTC'].iloc[0], df['datetimeUTC'].iloc[-1]
        df['doy'] = [int(datetime.strftime(x, '%j')) for x in rrule(DAILY, dtstart=s, until=e)]
        peak_doy = []
        for y in range(s.year, e.year + 1):
            ydf = copy(df.loc['{}-01-01'.format(y): '{}-12-31'.format(y)])
            if ydf.shape[0] < 364:
                continue
            peak_doy.append(ydf['doy'].loc[ydf['q'].idxmax()])

        print(peak_doy)


def irrigation_fraction(dir_, metadata):
    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]
    irr_idx = []
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        m = meta[sid]
        df = hydrograph(c)
        if np.all(df['irr'].values < 10e4):
            continue
            # print('no irrigation', m['STANAME'])
        df['if'] = df['cc'] / df['q']
        frac = df['cc'].sum() / df['q'].sum()
        if df['cc'].sum() < 0:
            print(sid, frac, m['STANAME'])
        irr_idx.append(frac)
        # print(sid, frac, meta[sid]['STANAME'])

    # [print(x) for x in irr_idx]


def write_flow_parameters(dir_, metadata_in, out_dir, metadata_out):

    pandas2ri.activate()
    r['source']('BaseflowSeparationFunctions.R')
    rec_const_r = robjects.globalenv['baseflow_RecessionConstant']
    bfi_max_r = robjects.globalenv['baseflow_BFImax']
    bf_eckhardt_r = robjects.globalenv['baseflow_Eckhardt']

    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]

    with open(metadata_in, 'r') as f:
        meta = json.load(f)

    for c in l:

        sid = os.path.basename(c).split('_')[0]

        if sid not in meta.keys():
            print(sid, 'not found')
            continue
        print(sid, meta[sid]['STANAME'])

        df = hydrograph(c)

        with localconverter(ro.default_converter + pandas2ri.converter):
            dfr = ro.conversion.py2rpy(df['q'])

        try:
            k = rec_const_r(dfr)[0]
            bfi_max = bfi_max_r(dfr, k)[0]
            df['qb'] = bf_eckhardt_r(dfr, bfi_max, k)
        except RRuntimeError:
            print('error ', sid, meta[sid]['STANAME'])

        meta[sid].update({'bfi_max': bfi_max, 'k': k})
        df.to_csv(os.path.join(out_dir, '{}.csv'.format(sid)))
        # TODO: write to csv, save k, bfi to metadata

    with open(metadata_out, 'w') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    data = '/media/research/IrrigationGIS/gages/merged_q_ee'

    daily = '/media/research/IrrigationGIS/gages/hydrographs/daily_q'
    daily_bf = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'

    jsn = '/media/research/IrrigationGIS/gages/station_metadata/metadata.json'
    jsn_out = '/media/research/IrrigationGIS/gages/station_metadata/metadata_flows.json'
    # peak_q_dates(s, e, daily, peak_dir)
    # irrigation_fraction(data, jsn)
    write_flow_parameters(daily, jsn, daily_bf, jsn_out)
# ========================= EOF ====================================================================
