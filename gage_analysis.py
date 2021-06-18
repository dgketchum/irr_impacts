import os
from copy import copy
from pandas import to_datetime, read_csv, date_range, DatetimeIndex
from datetime import datetime as dt, datetime
from dateutil.rrule import rrule, DAILY


def peak_q_dates(start, end, daily_q_dir, peak_out):
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    for c in l:
        if '12352500' not in c:
            continue
        sid = os.path.basename(c).split('_')[0]
        df = read_csv(c)
        df['datetimeUTC'] = to_datetime(df['datetimeUTC'])
        s, e = df['datetimeUTC'].iloc[0], df['datetimeUTC'].iloc[-1]
        df = df.set_index('datetimeUTC')
        df.columns = ['q']
        df['doy'] = [int(datetime.strftime(x, '%j')) for x in rrule(DAILY, dtstart=s, until=e)]
        peak_doy = []
        for y in range(s.year, e.year + 1):
            ydf = copy(df.loc['{}-01-01'.format(y): '{}-12-31'.format(y)])
            if ydf.shape[0] < 364:
                continue
            peak_doy.append(ydf['doy'].loc[ydf['q'].idxmax()])

        print(peak_doy)



if __name__ == '__main__':
    s, e = '1984-01-01', '2020-12-31'
    peak_dir = '/media/research/IrrigationGIS/gages/hydrographs/peak_q'
    daily = '/media/research/IrrigationGIS/gages/hydrographs/daily_q'
    peak_q_dates(s, e, daily, peak_dir)
# ========================= EOF ====================================================================
