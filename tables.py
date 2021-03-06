import os
from pprint import pprint
import numpy as np
from pandas import read_csv, concat, errors, DataFrame, DatetimeIndex, date_range

from hydrograph import hydrograph


def merge_ssebop_tc_gridmet_q(extracts, out_dir, flow_dir, gridmet=None, glob='glob', division='basins',
                              join_key='STAID'):
    missing, missing_ct, processed_ct = [], 0, 0

    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])

        try:
            if first:
                df = read_csv(csv, index_col=join_key)
                df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
                first = False
            else:
                c = read_csv(csv, index_col=join_key)
                if y < 1991:
                    if division == 'basins':
                        c['irr'] = [np.nan for _ in range(c.shape[0])]
                    c['et'] = [np.nan for _ in range(c.shape[0])]
                    c['cc'] = [np.nan for _ in range(c.shape[0])]
                cols = list(c.columns)
                c.columns = ['{}_{}_{}'.format(col, y, m) for col in cols]
                df = concat([df, c], axis=1)
            if gridmet:
                gm_file = os.path.join(gridmet, 'gridmet_{}_{}.csv'.format(y, str(m).rjust(2, '0')))
                gm_csv = read_csv(gm_file, index_col=join_key)
                cols = list(gm_csv.columns)
                gm_csv.columns = ['gm_{}_{}_{}'.format(col, y, m) for col in cols]
                gm_csv *= 0.001
                df = concat([df, gm_csv], axis=1)

        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    year_start = int(sorted([x.split('_')[1] for x in list(df.columns)])[0])

    if division == 'basins':
        df['STAID_STR'] = [str(x).rjust(8, '0') for x in list(df.index.values)]
    else:
        df['GEOID_STR'] = [str(x).rjust(5, '0') for x in list(df.index.values)]
        drop_cols = [x for x in list(df.columns) if 'STUDY' in x]
        df['STUDY_INT'] = df['STUDYINT_2020_9']
        df.drop(columns=drop_cols, inplace=True)

    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    idx = DatetimeIndex(date_range(s, e, freq='M'))
    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]
    for d in dfd:
        try:
            if division == 'basins':
                sta = d['STAID_STR']
            else:
                sta = d['GEOID_STR']

            if sta == '05015500':
                a = 1
            # handle pre-1991 and off-growing season KeyError
            irr, cc, et = [], [], []
            for y, m in months:
                try:
                    cc_, et_ = d['cc_{}_{}'.format(y, m)], d['et_{}_{}'.format(y, m)]
                    cc.append(cc_)
                    et.append(et_)
                except KeyError:
                    cc.append(np.nan)
                    et.append(np.nan)

                try:
                    irr_ = d['irr_{}_{}'.format(y, m)]
                    irr.append(irr_)
                except KeyError:
                    irr.append(np.nan)

            irr = irr, 'irr'
            cc = cc, 'cc'
            et = et, 'et'

            if not np.any(irr[0]):
                print(sta, 'no irrigation')
                continue

            ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
            gm_ppt = [d['gm_ppt_{}_{}'.format(y, m)] for y, m in months], 'gm_ppt'
            etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'
            gm_etr = [d['gm_etr_{}_{}'.format(y, m)] for y, m in months], 'gm_etr'
            sm = [d['swb_aet_{}_{}'.format(y, m)] for y, m in months], 'swb'
            recs = DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, sm, gm_ppt,
                                                          gm_etr]]), index=idx)

            if division == 'basins':
                q_file = os.path.join(flow_dir, '{}.csv'.format(sta))
                qdf = hydrograph(q_file)
                h = concat([qdf, recs], axis=1)
            else:
                h = recs

            file_name = os.path.join(out_dir, '{}.csv'.format(sta))
            h.to_csv(file_name)
            processed_ct += 1

            print(file_name)

        except FileNotFoundError:
            missing_ct += 1
            print(sta, 'not found')
            missing.append(sta)

    print(processed_ct, 'processed')
    print(missing_ct, 'missing')
    pprint(missing)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    gage_src = os.path.join(root, 'gages/hydrographs/q_monthly')
    gridmet_ = os.path.join(root, 'gages/ee_exports/gridmet/monthly')
    extract_ = os.path.join(root, 'gages/ee_exports/monthly/IrrMapperComp_21DEC2021')
    out_dir = os.path.join(root, 'gages/merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')
    g = 'Comp_21DEC2021'
    merge_ssebop_tc_gridmet_q(extracts=extract_, out_dir=out_dir, flow_dir=gage_src, gridmet=gridmet_, glob=g,
                              division='basins')
# ========================= EOF ====================================================================
