import os
from datetime import datetime, date
from dateutil.rrule import rrule, DAILY

import numpy as np
from pandas import read_csv, to_datetime, concat, DataFrame


def hydrograph(c):
    df = read_csv(c)

    if 'Unnamed: 0' in list(df.columns):
        df = df.rename(columns={'Unnamed: 0': 'dt'})
    try:
        df['dt'] = to_datetime(df['dt'])
    except KeyError:
        df['dt'] = to_datetime(df['datetimeUTC'])
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


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages/'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages/'
    pass
# ========================= EOF ====================================================================
