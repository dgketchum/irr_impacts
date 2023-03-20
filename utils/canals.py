import os
from pprint import pprint

import requests
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


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


def get_data_ucrb():
    dtype = {'19': 'cfs', '20': 'af', '1191': 'cfs'}
    ex = 'https://www.usbr.gov/uc/water/hydrodata/gage_data/451/json/19.json'


def get_data_gp(shp, out_dir):
    sdf = gpd.read_file(shp)
    dtr = pd.date_range('1987-01-01', '2021-12-31', freq='M')
    for i, r in sdf.iterrows():

        if r['StationSta'] not in ['WY', 'MT']:
            continue

        df = pd.DataFrame(columns=['q'], index=dtr)
        sid = r['StationID']
        # if sid != 'SFKDIVCO':
        #     continue
        print('\n\n{} {}'.format(r['StationNam'], sid))
        txt = requests.get('https://www.usbr.gov/gp-bin/inventory.pl?site={}'.format(sid)).content.decode('utf-8')

        try:
            splt = [t.split() for t in txt.splitlines() if 'QC' in t][0]
        except IndexError:
            splt = [t.split() for t in txt.splitlines() if 'Q' in t][0]
            pprint([t.split() for t in txt.splitlines() if 'Q' in t])
            continue

        units = splt[-2]
        yr_splt = splt[-1].split('-')
        syr, eyr = int(yr_splt[0]), int(yr_splt[1])
        pprint([t.split() for t in txt.splitlines() if 'Q' in t])
        print('==== Using QC: {} - {} in {} ====='.format(syr, eyr, units))

        for year in range(syr, eyr + 1):

            try:
                url = 'https://www.usbr.gov/gp-bin/webdaycsv.pl?parameter={}%20Q&syer={}' \
                      '&smnth=1&sdy=1&eyer={}&emnth=12&edy=31&format=2'.format(sid, year, year)

                c = pd.read_csv(url, skiprows=20, infer_datetime_format=True, parse_dates=True)

                c.columns = ['date', 'q']
                end_ = [i for i, x in enumerate(c['date']) if 'END' in x][0]
                c = c.iloc[:end_]

                if c.empty:
                    print('{} empty, skipping'.format(year))
                    continue

                c.index = [pd.to_datetime(x) for x in c['date']]
                c = c[['q']]
                c[c['q'] == 998877] = np.nan
                c = c.resample('D').agg(pd.DataFrame.mean, skipna=False)

                if len(c.loc['{}-05-01'.format(year): '{}-9-30'.format(year)]) < 153:
                    print('{} incomplete, skipping'.format(year))
                    continue

                # convert to cubic meter per day
                c = c * 60 * 60 * 24 * 0.02831
                c = c.resample('M').agg(pd.DataFrame.sum, skipna=False)
                df.loc[c.index, 'q'] = c['q'].values

            except Exception as e:
                print(e, year, sid)

        ofile = os.path.join(out_dir, '{}.csv'.format(sid))
        df.to_csv(ofile)


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

# ========================= EOF ====================================================================
