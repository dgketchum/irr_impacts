import os
from calendar import monthrange
import json

import requests
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

replace = '!@#$%^&*()[]{};:,./<>?\|`~-=_+'


def find_canals_ucrb(file_path, out_path):
    df = gpd.read_file(file_path)
    df.set_index('site_id', inplace=True)
    manmade_stream_names = ['CANAL', 'DITCH', 'DIVERSION']
    df['is_manmade_stream'] = df['site_met_1'].str.contains('|'.join(manmade_stream_names), case=False, regex=True)
    df = df[df['is_manmade_stream'] & np.isnan(df['site_met16'])]
    df.to_file(out_path)


def find_canals_pn(file_path, out_path):
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
    df.set_index('StationID', inplace=True)
    manmade_stream_names = ['CANAL', 'DITCH', 'DIVERSION', 'SLOUGH']
    df['is_ditch'] = df['StationName'].str.contains('|'.join(manmade_stream_names), case=False, regex=True)
    df = df[df['is_ditch']]
    df['is_river'] = df['StationName'].str.contains('|'.join(['RIVER']), case=False, regex=True)
    df = df[~df['is_river']]
    df['StationLongitude'] = df['StationLongitude'].values.astype(float)
    df['StationLatitude'] = df['StationLatitude'].values.astype(float)
    df['geometry'] = df.apply(lambda x: Point(x['StationLongitude'], x['StationLatitude']), axis=1)
    df.to_file(out_path, crs='EPSG:4326')


def get_data_gp(shp, out_dir):
    sdf = gpd.read_file(shp)
    # incoming data are in acre-foot per month
    dtr = pd.date_range('1987-01-01', '2021-12-31', freq='M')
    for i, r in sdf.iterrows():
        converted = False

        if r['StationSta'] not in ['WY', 'MT']:
            continue

        df = pd.DataFrame(columns=['q'], index=dtr)
        sid, _name = r['StationID'], r['StationNam'].replace('/', '_')

        print('\n\n{} {}'.format(_name, sid))
        res070 = 'https://www.usbr.gov/gp-bin/res070.pl?station={}&parameters=AF.QJ&byear=1987&eyear=2021'.format(sid)
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
    sdf = sdf.sort_values('SID')
    dtr = pd.date_range('1987-01-01', '2021-12-31', freq='M')
    converted = False
    for i, r in sdf.iterrows():

        sid, _name, dtype = r['SID'], r['site_name'], r['DTYPE']
        units = r['unit_commo']

        if units != 'cfs' and unit_strict:
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
        elif units == 'cfs':
            # cfs to m ^3 d ^-1
            c = c * 2446.58
            c = c.resample('M').agg(pd.DataFrame.sum, skipna=False)
            converted = True
        else:
            raise NotImplementedError('Unknown units and resampling technique')

        july = c.loc[[i for i in c.index if i.month == 7], col]
        july = july[july > 0]
        print('july mean: {:.1f} {} at {} {}, {} years'.format(july.mean(), 'm^3', sid, _name, july.shape[0]))

        ind_match = [i for i in c.index if i in df.index]
        df.loc[ind_match, 'q'] = c.loc[ind_match, col]
        assert converted
        ofile = os.path.join(out_dir, '{}_{}.csv'.format(sid, _name.replace(replace, '_')))
        df.to_csv(ofile, float_format='%.3f')


if __name__ == '__main__':
    ucrb_ = '/media/research/IrrigationGIS/impacts/canals/shapefiles/hydromet_ucrb.shp'
    ucrb_out = '/media/research/IrrigationGIS/impacts/canals/shapefiles/ucrb_diversions.shp'
    # find_canals_ucrb(ucrb_, ucrb_out)

    cpn_ = '/media/research/IrrigationGIS/usbr/pnw/Hydromet_Stations_(CPN).shp'
    cpn_out = '/media/research/IrrigationGIS/impacts/canals/shapefiles/pn_diversions.shp'
    # find_canals_pn(cpn_, cpn_out)

    gp_ = '/home/dgketchum/Downloads/hydromet_station_list.csv'
    gp_out = '/media/research/IrrigationGIS/impacts/canals/shapefiles/gp_diversions.shp'
    # find_canals_gp(gp_, gp_out)

    od = '/media/research/IrrigationGIS/impacts/canals/hydrographs'
    get_data_gp(gp_out, od)
    get_data_ucrb(ucrb_out, od, unit_strict=True)
# ========================= EOF ====================================================================
