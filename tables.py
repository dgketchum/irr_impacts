import os
from pandas import read_csv, concat, errors

DROP = ['system:index', '.geo']
ATTRS = ['SQMI', 'STANAME', 'start', 'end']


def concatenate_series(root, out_dir, glob='None'):
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


if __name__ == '__main__':
    r = '/media/research/IrrigationGIS/gages/ee_exports/annual'
    o = '/media/research/IrrigationGIS/gages/ee_exports/series'
    g = 'basins'
    concatenate_series(r, o, g)
# ========================= EOF ====================================================================
