import os
from pandas import read_csv, concat, errors, DataFrame
from copy import copy

DROP = ['system:index', '.geo']
ATTRS = ['SQMI', 'STANAME', 'start', 'end']


def concatenate_series(root, out_csv, glob='None'):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
    l.sort()
    first = True
    for csv in l:
        try:
            if first:
                df = read_csv(csv, index_col='STAID').drop(columns=DROP)
                print(df.shape, csv)
                first = False
            else:
                c = read_csv(csv, index_col='STAID').drop(columns=DROP + ATTRS)
                df = concat([df, c], axis=1)
                print(c.shape, csv)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df.to_csv(out_csv)


def merge_hydrograph_gridded(csv, hydrograph_src, out_dir):
    df = read_csv(csv)
    df['STAID_STR'] = [str(x).rjust(8, '0') for x in df['STAID'].values]
    dfd = df.to_dict(orient='records')
    for d in dfd:
        years = [x for x in range(1991, 2021)]
        cc = [d['cc_{}'.format(y)] for y in years], 'cc'
        irr = [d['irr_{}'.format(y)] for y in years], 'irr'
        ppt = [d['ppt_{}'.format(y)] for y in years], 'ppt'
        pet = [d['pet_{}'.format(y)] for y in years], 'pet'
        hydrograph = os.path.join(hydrograph_src, '{}_annual.csv'.format(d['STAID_STR']))
        h = read_csv(hydrograph, index_col='datetimeUTC')
        # select range, convert to cubic meters
        h = h.loc['1991-01-01':] * 1233.48
        try:
            recs = DataFrame(dict([(x[1], x[0]) for x in [cc, irr, ppt, pet]]), index=h.index)
            print(d['STAID_STR'])
        except ValueError as e:
            print(d['STAID_STR'], e)
            continue
        h = concat([h, recs], axis=1)
        h.to_csv(os.path.join(out_dir, '{}.csv'.format(d['STAID_STR'])))
        pass


if __name__ == '__main__':
    r = '/media/research/IrrigationGIS/gages/ee_exports/annual'
    extracts = '/media/research/IrrigationGIS/gages/ee_exports/series/extracts_17JUN2021.csv'
    g = 'basins'
    # concatenate_series(r, extracts, g)
    gage_src = '/media/research/IrrigationGIS/gages/hydrographs/annual_q'
    dst = '/media/research/IrrigationGIS/gages/merged_hydro_ee'
    merge_hydrograph_gridded(extracts, gage_src, dst)
# ========================= EOF ====================================================================
