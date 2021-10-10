import os
import numpy as np
from pandas import read_csv, concat, errors, DataFrame, DatetimeIndex, date_range

from hydrograph import hydrograph


def merge_ssebop_tc_q(extracts, flow_dir, out_dir, glob='glob'):
    missing_ct = 0
    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[3]), int(splt[4].split('.')[0])
        try:
            if first:
                df = read_csv(csv, index_col='STAID')
                df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
                first = False
            else:
                c = read_csv(csv, index_col='STAID')
                c.columns = ['{}_{}_{}'.format(col, y, m) for col in list(c.columns)]
                df = concat([df, c], axis=1)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    year_start = [x.split('_')[1] for x in list(df.columns)][0]
    df['STAID_STR'] = [str(x).rjust(8, '0') for x in list(df.index.values)]
    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    idx = DatetimeIndex(date_range(s, e, freq='M'))
    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]
    for d in dfd:
        try:
            sta = d['STAID_STR']
            irr = [d['irr_{}_{}'.format(y, m)] for y, m in months], 'irr'
            if not np.any(irr[0]):
                print(sta, 'no irrigation')
                continue
            et = [d['et_{}_{}'.format(y, m)] for y, m in months], 'et'
            cc = [d['cc_{}_{}'.format(y, m)] for y, m in months], 'cc'
            ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
            etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'
            sm = [d['swb_aet_{}_{}'.format(y, m)] for y, m in months], 'sm'
            recs = DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, sm]]), index=idx)
            q_file = os.path.join(flow_dir, '{}.csv'.format(sta))
            qdf = hydrograph(q_file)
            h = concat([qdf, recs], axis=1)[s: e]
            file_name = os.path.join(out_dir, '{}.csv'.format(sta))
            h.to_csv(file_name)
            print(file_name)
        except FileNotFoundError:
            missing_ct += 1
            print(d['STAID_STR'], 'not found')
    print(missing_ct, 'missing')


if __name__ == '__main__':
    r = '/media/research/IrrigationGIS/gages/ee_exports/annual'
    extracts = '/media/research/IrrigationGIS/gages/ee_exports/series/extracts_comp_25AUG2021.csv'
    g = 'basins_Comp_5OCT2021'
    # concatenate_irrmapper_ssebop_extracts(r, extracts, g)

    gage_src = '/media/research/IrrigationGIS/gages/hydrographs/q_bf_monthly'
    extract_ = '/media/research/IrrigationGIS/gages/ee_exports/monthly'
    out_dir = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q'
    g = 'basins_Comp_5OCT2021'
    merge_ssebop_tc_q(extract_, gage_src, out_dir, glob=g)
# ========================= EOF ====================================================================
