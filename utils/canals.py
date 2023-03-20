import os

import numpy as np
import geopandas as gpd
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


if __name__ == '__main__':
    ucrb_ = '/media/research/IrrigationGIS/impacts/canals/hydromet_ucrb.shp'
    ucrb_out = '/media/research/IrrigationGIS/impacts/canals/ucrb_diversions.shp'
    # find_canals_ucrb(ucrb_, ucrb_out)

    cpn_ = '/media/research/IrrigationGIS/usbr/pnw/Hydromet_Stations_(CPN).shp'
    cpn_out = '/media/research/IrrigationGIS/impacts/canals/pn_diversions.shp'
    # find_canals_pn(cpn_, cpn_out)

    gp_ = '/home/dgketchum/Downloads/hydromet_station_list.csv'
    gp_out = '/media/research/IrrigationGIS/impacts/canals/gp_diversions.shp'
    find_canals_gp(gp_, gp_out)

# ========================= EOF ====================================================================
