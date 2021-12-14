import os
from pprint import pprint
import numpy as np
from pandas import read_csv, concat, errors, DataFrame, DatetimeIndex, date_range

from hydrograph import hydrograph


def merge_ssebop_tc_q(extracts, flow_dir, out_dir, glob='glob'):
    missing, missing_ct, processed_ct = [], 0, 0

    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[2]), int(splt[3].split('.')[0])
        try:
            if first:
                df = read_csv(csv, index_col='STAID')
                df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
                first = False
            else:
                c = read_csv(csv, index_col='STAID')
                if y < 1991:
                    c['irr'] = [np.nan for _ in range(c.shape[0])]
                    c['et'] = [np.nan for _ in range(c.shape[0])]
                    c['cc'] = [np.nan for _ in range(c.shape[0])]
                cols = list(c.columns)
                c.columns = ['{}_{}_{}'.format(col, y, m) for col in cols]
                df = concat([df, c], axis=1)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    year_start = int(sorted([x.split('_')[1] for x in list(df.columns)])[0])
    df['STAID_STR'] = [str(x).rjust(8, '0') for x in list(df.index.values)]
    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    idx = DatetimeIndex(date_range(s, e, freq='M'))
    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]
    for d in dfd:
        try:
            sta = d['STAID_STR']
            # if sta not in ['06177000']:
            #     continue

            # handle pre-1991 and off-growing season KeyError
            irr, cc, et = [], [], []
            for y, m in months:
                try:
                    irr.append(d['irr_{}_{}'.format(y, m)])
                    cc.append(d['cc_{}_{}'.format(y, m)])
                    et.append(d['et_{}_{}'.format(y, m)])
                except KeyError:
                    irr.append(np.nan)
                    cc.append(np.nan)
                    et.append(np.nan)

            irr = irr, 'irr'
            cc = cc, 'cc'
            et = et, 'et'

            if not np.any(irr[0]):
                print(sta, 'no irrigation')
                continue

            ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
            etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'
            sm = [d['swb_aet_{}_{}'.format(y, m)] for y, m in months], 'sm'
            recs = DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, sm]]), index=idx)
            q_file = os.path.join(flow_dir, '{}.csv'.format(sta))
            qdf = hydrograph(q_file)
            h = concat([qdf, recs], axis=1)
            file_name = os.path.join(out_dir, '{}.csv'.format(sta))
            h.to_csv(file_name)
            processed_ct += 1
            print(file_name)

        except FileNotFoundError:
            missing_ct += 1
            print(d['STAID_STR'], 'not found')
            missing.append(d['STAID_STR'])

    print(processed_ct, 'processed')
    print(missing_ct, 'missing')
    pprint(missing)


if __name__ == '__main__':
    gage_src = '/media/research/IrrigationGIS/gages/hydrographs/q_monthly'
    extract_ = '/media/research/IrrigationGIS/gages/ee_exports/monthly/IrrMapperComp'
    out_dir = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q_Comp_14DEC2021'
    g = 'Comp_13DEC2021'
    merge_ssebop_tc_q(extract_, gage_src, out_dir, glob=g)
# ========================= EOF ====================================================================
