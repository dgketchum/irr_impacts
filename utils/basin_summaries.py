import os
import json
from datetime import date
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from gage_data import hydrograph

warnings.filterwarnings('ignore')

from gage_lists import EXCLUDE_STATIONS, WATER_BALANCE_EXCLUDE


def water_balance_ratios(metadata, ee_series, stations=None, out_shp=None,
                         monthly_summary=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    cols = ['STAID', 'AREA', 'IAREA', 'irr_frac', 'cc_q_r', 'cc_q_pct', 'cci',
            'q_ppt', 'ai', 'cc_q_ratio_5', 'cc_q_pct_5']

    if monthly_summary:
        m_cols = ['cc_q_m{}'.format(m) for m in range(5, 11)]

    df = pd.DataFrame(columns=cols + m_cols)
    first = True

    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS + WATER_BALANCE_EXCLUDE:
            print('exclude', sid, v['STANAME'])
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        if not os.path.exists(_file):
            continue
        dct = {}
        cdf = hydrograph(_file)
        cdf['cci'] = cdf['cc'] / cdf['irr']
        years = np.arange(1987, 2022)
        cc_dates = [(date(y, 4, 1), date(y, 10, 31)) for y in years]

        clim_dates = [(date(y, 1, 1), date(y, 12, 31)) for y in years]
        full_q = [cdf['q'][d[0]: d[1]].count() for d in clim_dates]
        clim_dates = [d for i, d in enumerate(clim_dates) if full_q[i] == 12]

        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in clim_dates])
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
        irr = np.array([cdf['irr'][d[0]: d[1]].mean() for d in cc_dates])
        cci = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])

        dct[sid] = v
        irr_area = (np.mean(irr)).item() / 1e6
        irr_frac = irr_area / v['AREA']
        if irr_area < 0.001:
            irr_area = 0.0

        cc_q_ratio_5 = (cc[:5].sum() / q[:5].sum()).item()
        cc_q_pct_5 = (cc[:5].sum() / (cc[:5].sum() + q[:5].sum())).item()
        if cc_q_ratio_5 < 0.0:
            cc_q_ratio_5 = 0.0
        if cc_q_pct_5 < 0.0:
            cc_q_pct_5 = 0.0

        cc_q_ratio = (cc.sum() / q.sum()).item()
        cc_q_pct = (cc.sum() / (cc.sum() + q.sum())).item()
        if cc_q_ratio < 0.0:
            cc_q_ratio = 0.0
        if cc_q_pct < 0.0:
            cc_q_pct = 0.0

        if monthly_summary and v['basin'] == 'missouri':
            for m in range(5, 11):
                indx = [d for d in cdf.index if d.month == m]
                mdf = cdf.loc[indx]
                mdf[mdf['cc'] < 0.0] = 0.0
                mcc, mq = mdf.loc['1987-01-01': '2021-12-31', 'cc'], mdf.loc['1987-01-01': '2021-12-31', 'q']
                m_key = 'cc_q_m{}'.format(m)
                dct[m_key] = np.nanmean((mcc / mq))

        dct['IAREA'] = irr_area
        dct['irr_frac'] = irr_frac
        dct['cc_q_r'] = cc_q_ratio
        dct['cc_q_pct'] = cc_q_pct
        dct['cc_q_ratio_5'] = cc_q_ratio_5
        dct['cc_q_pct_5'] = cc_q_pct_5

        print('{}: {:.3f}, {}'.format(sid, np.mean(cc_q_pct), v['STANAME']))
        dct['cci'] = (np.mean(cci)).item()
        dct['q_ppt'] = (q.sum() / ppt.sum()).item()
        dct['ai'] = (etr.sum() / ppt.sum()).item()
        dct['AREA'] = v['AREA']
        dct['STANAME'] = v['STANAME']
        dct['STAID'] = sid
        df.loc[sid] = dct

    df['AREA'] = (df['AREA'] - df['AREA'].min()) / (df['AREA'].max() - df['AREA'].min())
    gdf = gpd.read_file(stations)
    gdf.index = gdf['STAID']
    gdf = gdf.loc[df.index]
    gdf[m_cols] = df[m_cols]
    gdf[cols] = df[cols]
    gdf.to_file(out_shp, crs='epsg:4326')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'

    ee_data = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')
    i_json = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
    station_pts = os.path.join(root, 'gages', 'selected_gages.shp')
    shp = os.path.join(root, 'figures', 'shapefiles', 'water_balance', 'basin_cc_ratios_monthly_q.shp')
    monthly_ = os.path.join('gages', 'monthly_summaries.csv')
    water_balance_ratios(i_json, ee_data, station_pts, shp, monthly_summary=monthly_)
# ========================= EOF ====================================================================
