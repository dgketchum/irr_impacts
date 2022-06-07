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

    hydro = os.path.join(root, 'hydrographs', 'daily_q')
    stations_ = ['13317000', '13307000', '13309220']
    first = True
    for s in stations_:
        if first:
            df = hydrograph('{}.csv'.format(os.path.join(hydro, s)))
            df.rename(columns={'q': s}, inplace=True)
            first = False
        else:
            c = hydrograph('{}.csv'.format(os.path.join(hydro, s)))
            c.rename(columns={'q': s}, inplace=True)
            df = concat([df, c], axis=1)
    s = df[['13307000', '13309220']].sum(axis=1)
    frac = df[['13307000', '13309220']].sum(axis=1) / df['13317000']
    max_tuple = []
    for y in range(2003, 2021):
        d = s.loc['{}-01-01'.format(y): '{}-12-31'.format(y)]
        max_tuple.append((d.iloc[d.argmax()], int(datetime.strftime(d.index[d.argmax()], '%j'))))

    a = date(2021, 1, 1)
    b = date(2021, 12, 31)
    years = [x for x in range(2003, 2021)]

    dct = {'median': [], 'max': [], 'min': []}
    dtime = []
    fracs = []
    for dt in rrule(DAILY, dtstart=a, until=b):
        dates = [date(year, dt.month, dt.day).strftime('%Y-%m-%d') for year in years]
        vals = s.loc[dates].values
        frac_ = frac.loc[dates].values

        fracs.append(np.median(frac_))
        median_ = np.median(vals)

        dct['median'].append(median_)
        dct['max'].append(vals.max())
        dct['min'].append(vals.min())
        dtime.append('{}-{}'.format(dt.month, dt.day))

        med = DataFrame(data=dct, index=dtime)
    pass
# ========================= EOF ====================================================================
