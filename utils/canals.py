import os
from calendar import monthrange
import json

import requests
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

replace = '!@#$%^&*()[]{};:,./<>?\|`~-=_+'


def find_canals_ucrb(file_path, out_path):
    df = gpd.read_file(file_path)
    df.set_index('field_1', inplace=True)
    manmade_stream_names = ['CANAL', 'DITCH', 'DIVERSION', 'TUNEL', 'OUTLET', 'INLET']
    df['is_manmade_stream'] = df['site_name'].str.contains('|'.join(manmade_stream_names), case=False, regex=True)
    df = df[df['is_manmade_stream'] & np.isnan(df['usgs_id'])]
    df.drop(columns=['is_manmade_stream'], inplace=True)
    df.to_file(out_path)


def find_canals_pn(file_path, out_path):
    to_add = ['bcp']
    df = gpd.read_file(file_path)
    df.set_index('siteid', inplace=True)
    manmade_stream_names = ['CANAL', 'DITCH', 'DIVERSION', 'SLOUGH']
    df['descriptio'] = df['descriptio'].apply(lambda x: x.upper())
    df['is_ditch'] = df['descriptio'].str.contains('|'.join(manmade_stream_names), case=False, regex=True)
    df = df[df['is_ditch']]
    df['is_river'] = df['descriptio'].str.contains('|'.join(['RIVER']), case=False, regex=True)
    df = df[~df['is_river']]
    df.to_file(out_path)


def find_canals_gp(file_path, out_path):
    df = gpd.read_file(file_path)
    manmade_stream_names = ['CANAL', 'DITCH', 'DIVERSION', 'SLOUGH', 'FEEDER']
    df['is_ditch'] = df['Name'].str.contains('|'.join(manmade_stream_names), case=False, regex=True)
    df = df[df['is_ditch']]
    df['is_river'] = df['Name'].str.contains('|'.join(['RIVER']), case=False, regex=True)
    df = df[~df['is_river']]
    df.drop(columns=['is_ditch', 'is_river'], inplace=True)
    df['Longitude'] = df['Longitude'].values.astype(float)
    df['Latitude'] = df['Latitude'].values.astype(float)
    df['geometry'] = df.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis=1)
    df.drop(columns=['field_1'], inplace=True)
    df.to_file(out_path, crs='EPSG:4326')


def get_data_gp(shp, out_dir):
    sdf = gpd.read_file(shp)
    # incoming data are in acre-foot per month
    dtr = pd.date_range('1982-01-01', '2021-12-31', freq='M')
    for i, r in sdf.iterrows():
        converted = False

        df = pd.DataFrame(columns=['q'], index=dtr)
        sid, _name = r['ID'], r['Name'].replace('/', '_')

        print('\n\n{} {}'.format(_name, sid))
        res070 = 'https://www.usbr.gov/gp-bin/res070.pl?station={}&parameters=AF.QJ&byear=1982&eyear=2021'.format(sid)
        txt = requests.get(res070).content.decode('utf-8')
        txt = txt.replace('-----', '0')
        splt = [t.split() for t in txt.splitlines()]
        try:
            cols = splt[10][:13]
        except IndexError:
            print(splt[2])
            continue
        data = [x[:13] for x in splt[14:]]
        for i, x in enumerate(data):
            try:
                _ = int(x[0])
            except ValueError:
                break
        data = data[:i]
        c = pd.DataFrame(data=data, columns=cols)
        c.index = c['Year']
        c.drop(columns=['Year'], inplace=True)
        c.columns = list(range(10, 13)) + list(range(1, 10))

        for i, r in c.iterrows():
            for m, v in r.items():
                y = int(i)
                if m > 9:
                    y -= 1
                end_mo = monthrange(int(y), m)[1]
                # convert to meter cube
                df.loc['{}-{}-{}'.format(i, m, end_mo), 'q'] = float(v) * 1233.48
                converted = True

        july = df.loc[[i for i in df.index if i.month == 7], 'q']
        july = july[july > 0]
        print(
            'july mean: {:.1f} {} at {} {}, {} years'.format(july.mean() / 1e6, 'm^3 * 1e6', sid, _name, july.shape[0]))

        df['q'] = df['q'].values.astype(float)
        print('{} of {} at {}'.format(np.count_nonzero(~np.isnan(df['q'].values)), df.shape[0], _name))
        assert converted
        ofile = os.path.join(out_dir, '{}_{}.csv'.format(sid, _name.replace(replace, '_')))
        df.to_csv(ofile)


def get_data_ucrb(shp, out_dir, unit_strict=True):
    sdf = gpd.read_file(shp)
    sdf = sdf.sort_values('site_id')
    dtr = pd.date_range('1982-01-01', '2021-12-31', freq='M')
    converted = False
    for i, r in sdf.iterrows():

        sid, _name, dtype = r['site_id'], r['site_name'], r['datatype_i']
        units = r['unit_commo']

        ofile = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(ofile):
            continue

        df = pd.DataFrame(columns=['q'], index=dtr)
        print('\n\n{} {}'.format(_name, sid))
        url = 'https://www.usbr.gov/uc/water/hydrodata/gage_data/{}/json/{}.json'.format(sid, dtype)
        txt = requests.get(url).content.decode('utf-8')

        try:
            js = json.loads(txt)
            c = pd.DataFrame(columns=js['columns'], data=js['data'])
        except Exception as e:
            print(e, sid, _name, units)
            continue

        c.index = [pd.to_datetime(x) for x in c['datetime']]
        col = js['columns'][1]
        c.drop(columns=['datetime'], inplace=True)

        if units == 'acre-feet':
            c = c.resample('M').agg(pd.DataFrame.sum, skipna=False)
            # acre-foot to cubic meters
            c *= 1233.48
            converted = True
        elif units == 'cfs':
            # cfs to m ^3 d ^-1
            c = c * 2446.58
            c = c.resample('M').agg(pd.DataFrame.sum, skipna=False)
            converted = True
        else:
            raise NotImplementedError('Unknown units and resampling technique')

        aaf = (c.loc['2000-01-01': '2022-12-31'].resample('A').sum() / 1233.48).mean().values[0]
        print('2000-2022 annual mean: {:,.1f} {} at {} {}, {} years'.format(aaf, 'af', sid, _name, c.shape[0] / 12))

        ind_match = [i for i in c.index if i in df.index]
        df.loc[ind_match, 'q'] = c.loc[ind_match, col]
        assert converted
        df.to_csv(ofile, float_format='%.3f')


def get_data_pnw(shp, out_dir):
    sdf = gpd.read_file(shp)
    # incoming data are in acre-foot per month
    dtr = pd.date_range('1982-01-01', '2021-12-31', freq='M')
    for i, r in sdf.iterrows():
        converted = False
        sid, _name, _type = r['siteid'], r['descriptio'], 'qj'  # avg canal q: cfs

        if sid not in ['bcp']:
            continue

        df = pd.DataFrame(columns=['q'], index=dtr)

        print('\n\n{} {}'.format(_name, sid))
        url = 'https://www.usbr.gov/pn-bin/daily.pl?station={}&format=csv' \
              '&year=1982&month=1&day=1&year=2022&month=12&day=31&pcode={}'.format(sid, _type)
        c = pd.read_csv(url)
        c.index = [pd.to_datetime(x) for x in c['DateTime']]
        c.columns = ['date', 'q']
        c.drop(columns=['date'], inplace=True)
        c['q'] = c.values.astype(float)
        c = c * 2446.58
        c = c.resample('M').agg(pd.DataFrame.sum, skipna=False)
        converted = True

        july = c.loc[[i for i in c.index if i.month == 7], 'q']
        july = july[july > 0]
        mean_jul = july.mean() / 1e6
        print('july mean: {:.1f} {} at {} {}, {} years'.format(mean_jul, 'm^3 * 1e6',
                                                               sid, _name, july.shape[0]))

        ind_match = [i for i in c.index if i in df.index]
        df.loc[ind_match, 'q'] = c.loc[ind_match, 'q']
        assert converted
        ofile = os.path.join(out_dir, '{}_{}.csv'.format(sid, _name.replace(replace, '_')))
        df.to_csv(ofile, float_format='%.3f')


def plot(hydr_dir, plot_dir):
    l = [os.path.join(hydr_dir, x) for x in os.listdir(hydr_dir) if x.endswith('.csv')]
    for f in l:
        sid = f.split('_')[0]
        bname = os.path.basename(f)
        ofile = os.path.join(plot_dir, '{}.png'.format(bname))
        if os.path.exists(ofile):
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True, infer_datetime_format=True)
        plt.plot(df.index, df['q'])
        plt.suptitle(bname)
        plt.savefig(ofile)
        plt.close()
        print(ofile)


if __name__ == '__main__':
    ucrb_ = '/media/research/IrrigationGIS/impacts/canals/shapefiles/hydromet_ucrb.shp'
    ucrb_out = '/media/research/IrrigationGIS/impacts/canals/shapefiles/ucrb_diversions.shp'
    # find_canals_ucrb(ucrb_, ucrb_out)

    cpn_ = '/media/research/IrrigationGIS/usbr/pnw/Hydromet_Stations_(CPN).shp'
    cpn_out = '/media/research/IrrigationGIS/impacts/canals/shapefiles/pn_diversions.shp'
    # find_canals_pn(cpn_, cpn_out)

    gp_ = '/media/research/IrrigationGIS/impacts/canals/hydromet_station_list.csv'
    gp_out = '/media/research/IrrigationGIS/impacts/canals/shapefiles/gp_diversions.shp'
    # find_canals_gp(gp_, gp_out)

    cpn_study = '/media/research/IrrigationGIS/impacts/canals/shapefiles/pn_diversions_study.shp'
    ucrb_study = '/media/research/IrrigationGIS/impacts/canals/shapefiles/ucrb_diversions_study.shp'
    gp_study = '/media/research/IrrigationGIS/impacts/canals/shapefiles/gp_diversions_study.shp'

    od = '/media/research/IrrigationGIS/impacts/canals/hydrographs'
    # get_data_gp(gp_study, od)
    get_data_ucrb(ucrb_study, od, unit_strict=True)
    # get_data_pnw(cpn_study, od)

    plt_dir = '/media/research/IrrigationGIS/impacts/figures/hydrographs/canal_hydrographs'
    # plot(od, plt_dir)
# ========================= EOF ====================================================================
