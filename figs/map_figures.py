import os
import json

import numpy as np
import pandas as pd
from pandas import DataFrame, concat, isna
import fiona
import geopandas as gpd
from shapely.geometry import shape

from utils.gage_lists import EXCLUDE_STATIONS


def monthly_trends(regressions_dir, in_shape, glob=None, out_shape=None, selectors=None):
    with fiona.open(in_shape, 'r') as src:
        feats = [f for f in src]

    geo_ = {f['properties']['STAID']: shape(f['geometry']) for f in feats}
    areas = {f['properties']['STAID']: f['properties']['AREA'] for f in feats}
    names = {f['properties']['STAID']: f['properties']['STANAME'] for f in feats}

    trends_dct = {}

    l = [os.path.join(regressions_dir, x) for x in os.listdir(regressions_dir) if x.startswith(glob)]

    for f in l:
        m = int(os.path.basename(f).split('.')[0].split('_')[-1])
        with open(f, 'r') as _file:
            dct = json.load(_file)

        trends_dct.update({m: dct})

    base_ = os.path.basename(regressions_dir)
    if base_ == 'ols_trends':
        trc_subdirs = {'time_aim': 'lr', 'time_q': 'mk', 'time_etr': 'lr', 'time_ppt': 'lr', 'time_irr': 'bayes',
                       'time_pptm': 'lr', 'time_etrm': 'lr'}
    elif base_ == 'mv_trends':
        trc_subdirs = {'time_q': 'bayes', 'time_cc': 'bayes'}
    elif base_ == 'uv_trends':
        trc_subdirs = {'time_q': 'bayes', 'time_cc': 'bayes', 'time_irr': 'bayes',
                       'time_ai': 'bayes', 'time_aim': 'bayes'}
    else:
        raise KeyError

    trends_stations = []
    [[trends_stations.append(sid) for sid in v.keys()] for k, v in trends_dct.items()]
    trends_stations = list(set(trends_stations))
    area_arr = np.array([areas[_id] for _id in trends_stations])
    areas = [(a - min(area_arr)) / (max(area_arr) - min(area_arr)) for a in area_arr]

    range_ = np.arange(1, 13)
    for var, test in trc_subdirs.items():

        if selectors and var not in selectors:
            continue

        df = pd.DataFrame(index=trends_stations, data=areas, columns=['AREA'])

        for m in range_:

            if m not in trends_dct.keys():
                continue

            d = trends_dct[m]

            if var in ['time_cc', 'time_ccres'] and m in [1, 2, 3, 11, 12]:
                data = {k: 0.0 for k, v in d.items() if k not in EXCLUDE_STATIONS}

            else:
                if test == 'bayes':
                    data = {}
                    for k, v in d.items():
                        if var not in v.keys():
                            data[k] = 0
                        elif not v[var]:
                            data[k] = 0
                        elif np.sign(v[var]['hdi_2.5%']) == np.sign(v[var]['hdi_97.5%']):
                            data[k] = v[var]['mean']
                        else:
                            data[k] = 0
                elif test == 'mk':
                    data = {k: v[str(m)][var]['b'] if v[str(m)][var]['p'] < 0.05 else 0.0 for k, v in d.items() if
                            var in v[str(m)]}
                elif test == 'lr':
                    data = {k: v[str(m)][var]['b_norm'] if v[str(m)][var]['p'] < 0.05 else 0.0 for k, v in d.items() if
                            var in v[str(m)]}
                else:
                    data = {k: v[var]['b_norm'] if v[var]['p'] < 0.05 else 0.0 for k, v in d.items() if
                            var in v}

            index, data = [i[0] for i in data.items()], [i[1] for i in data.items()]
            c = DataFrame(data=data, index=index, columns=[m]).astype(float)
            df = concat([df, c], axis=1)

        vals = df.values
        vals[np.isnan(vals)] = 0.0
        gdf = gpd.GeoDataFrame(df)
        gdf['STANAME'] = [names[_id] for _id in gdf.index]
        geo = [geo_[_id] for _id in gdf.index]

        gdf.drop(columns=['STANAME'], inplace=True)

        shp_file = os.path.join(out_shape, '{}.shp'.format(var))

        gdf[gdf[[i for i in range(1, 13)]] == 0.0] = np.nan
        gdf['median'] = gdf[[i for i in range(1, 13)]].median(axis=1)
        gdf['min'] = gdf[[i for i in range(1, 13)]].min(axis=1)
        gdf['max'] = gdf[[i for i in range(1, 13)]].max(axis=1)
        gdf['median'][isna(gdf['median'])] = 0.0
        gdf['min'][isna(gdf['min'])] = 0.0
        gdf['max'][isna(gdf['max'])] = 0.0

        gdf['sum_med'] = gdf[[i for i in range(5, 11)]].median(axis=1)
        gdf['sum_med'][isna(gdf['sum_med'])] = 0.0

        gdf['win_med'] = gdf[[i for i in [11, 12, 1, 2, 3]]].median(axis=1)
        gdf['win_med'][isna(gdf['win_med'])] = 0.0

        gdf.loc[:, :] = np.where(np.isfinite(gdf.values), gdf.values, np.zeros_like(gdf.values))
        gdf['geometry'] = geo
        gdf['STANAME'] = [names[_id] for _id in gdf.index]
        cols = [str(c) for c in gdf.columns]
        gdf.columns = cols
        gdf.to_file(shp_file, crs='epsg:4326')
        print(shp_file)


def monthly_cc_qres(regressions_dir, in_shape, glob=None, out_shape=None, bayes=False, var='cc_q'):
    feats = gpd.read_file(in_shape)
    geo = {f['STAID']: shape(f['geometry']) for i, f in feats.iterrows()}
    areas = {f['STAID']: f['AREA'] for i, f in feats.iterrows()}
    names = {f['STAID']: f['STANAME'] for i, f in feats.iterrows()}

    trends_dct = {}

    l = [os.path.join(regressions_dir, x) for x in os.listdir(regressions_dir) if glob in x]

    for f in l:
        m = int(os.path.basename(f).split('.')[0].split('_')[-1])
        with open(f, 'r') as _file:
            dct = json.load(_file)

        trends_dct.update({m: dct})

    bayes_sig = 0
    p_sig = 0

    multi_impact = {}
    first = True
    for m in range(1, 13):

        d = trends_dct[m]
        data = {}

        for k, v in d.items():
            has_sig = False
            slopes_, cc_periods = [], []
            for kk, vv in v.items():
                if not isinstance(vv, dict):
                    continue
                else:
                    if not bayes and vv['p'] < 0.05:
                        slopes_.append(vv['b'])
                        has_sig = True
                    elif bayes and var in vv.keys():
                        if np.sign(vv[var]['hdi_2.5%']) == np.sign(vv[var]['hdi_97.5%']):
                            slopes_.append(vv['cc_q']['mean'])
                            cc_periods.append(kk)
                            has_sig = True
                            bayes_sig += 1
                    if bayes and vv['p'] < 0.05:
                        p_sig += 1
            if k not in multi_impact.keys():
                multi_impact[k] = [(p, s) for p, s in zip(cc_periods, slopes_)]
            else:
                [multi_impact[k].append((p, s)) for p, s in zip(cc_periods, slopes_)]

            if has_sig:
                data[k] = np.nanmedian(slopes_).item()
            else:
                data[k] = 0.0

        index, data = [i[0] for i in data.items()], [i[1] for i in data.items()]

        if first:
            df = DataFrame(data=data, index=index, columns=[m])
            first = False
        else:
            c = DataFrame(data=data, index=index, columns=[m])
            df = concat([df, c], axis=1)

        df[np.isnan(df)] = 0.0

    area_sort = sorted([x for x in areas.items() if x[0] in multi_impact.keys()], key=lambda x: x[1], reverse=True)
    multi_impact = {k: (multi_impact[k], names[k]) for k, v in area_sort}

    gdf = gpd.GeoDataFrame(df)
    gdf.geometry = [geo[_id] for _id in gdf.index]

    area_arr = np.array([areas[_id] for _id in gdf.index])
    areas = {k: (a - min(area_arr)) / (max(area_arr) - min(area_arr)) for k, a in areas.items()}
    gdf['AREA'] = [areas[_id] for _id in gdf.index]

    mixed = []
    m_arr = gdf[[x for x in range(1, 13)]].copy().values
    for i in range(m_arr.shape[0]):
        r = m_arr[i, :]
        if np.all(r == 0):
            mixed.append('None')
            continue
        r = r[m_arr[i, :] != 0.]
        if np.all(r > 0.0):
            mixed.append('pos')
        elif np.all(r < 0.0):
            mixed.append('neg')
        else:
            mixed.append('mixed')

    geo = gdf['geometry']
    gdf.drop(columns=['AREA', 'geometry'], inplace=True)
    gdf[gdf[[i for i in range(1, 13)]] == 0.0] = np.nan
    gdf['median'] = gdf[[i for i in range(1, 13)]].median(axis=1)
    gdf['sum_med'] = gdf[[i for i in range(7, 11)]].median(axis=1)
    gdf['win_med'] = gdf[[1, 2, 3, 11, 12]].median(axis=1)
    gdf['median'][isna(gdf['median'])] = 0.0
    gdf['sum_med'][isna(gdf['sum_med'])] = 0.0
    gdf['win_med'][isna(gdf['win_med'])] = 0.0
    gdf.loc[:, :] = np.where(np.isfinite(gdf.values), gdf.values, np.zeros_like(gdf.values))
    gdf['signs'] = mixed
    gdf['geometry'] = geo
    gdf['AREA'] = [areas[_id] for _id in gdf.index]
    gdf['STANAME'] = [names[_id] for _id in gdf.index]
    cols = [str(c) for c in gdf.columns]
    gdf.columns = cols
    shp_file = os.path.join(out_shape, '{}.shp'.format(glob))
    gdf.to_file(shp_file, crs='epsg:4326')
    df = DataFrame(gdf)
    df.drop(columns=['AREA', 'geometry'], inplace=True)
    df.to_csv(shp_file.replace('.shp', '.csv'))
    print('write {}'.format(shp_file))


def sustainability_trends(q_data, cc_data):
    qdf = gpd.read_file(q_data)
    cdf = gpd.read_file(cc_data)
    geo = [x for x in cdf['geometry']]
    sdf = gpd.GeoDataFrame(data=[0 for _ in cdf['median']], geometry=geo)
    sdf.columns = ['sign', 'geometry']
    sdf['sign'][(qdf['median'] < 0) & (cdf['median'] > 0)] = 'neg q, pos cc'
    sdf['sign'][(qdf['median'] > 0) & (cdf['median'] < 0)] = 'pos q, neg cc'
    sdf['sign'][(qdf['median'] > 0) & (cdf['median'] > 0)] = 'pos q, pos cc'
    sdf['sign'][(qdf['median'] < 0) & (cdf['median'] < 0)] = 'neg q, neg cc'
    sdf['AREA'] = cdf['AREA']

    _file = os.path.join(os.path.dirname(q_data), 'sustainability.shp')
    sdf.to_file(_file, crs='EPSG:4326')


def flow_change_climate_coincidence(q_data, climate_data):
    qdf = gpd.read_file(q_data)
    cdf = gpd.read_file(climate_data)
    geo = [x for x in cdf['geometry']]
    sdf = gpd.GeoDataFrame(data=[0 for _ in cdf['median']], geometry=geo)
    sdf.columns = ['sign', 'geometry']
    sdf['sign'][(qdf['median'] < 0) & (cdf['median'] > 0)] = 'neg q, pos ai'
    sdf['sign'][(qdf['median'] > 0) & (cdf['median'] < 0)] = 'pos q, neg ai'
    sdf['sign'][(qdf['median'] > 0) & (cdf['median'] > 0)] = 'pos q, pos ai'
    sdf['sign'][(qdf['median'] < 0) & (cdf['median'] < 0)] = 'neg q, neg ai'

    sdf['sign'][(qdf['median'] < 0) & (cdf['median'] == 0)] = 'neg q, zero ai'
    sdf['sign'][(qdf['median'] > 0) & (cdf['median'] == 0)] = 'pos q, zero ai'
    sdf['sign'][(qdf['median'] == 0) & (cdf['median'] > 0)] = 'zero q, pos ai'
    sdf['sign'][(qdf['median'] == 0) & (cdf['median'] < 0)] = 'zero q, neg ai'
    sdf['AREA'] = cdf['AREA']
    sdf['STANAME'] = cdf['STANAME']
    sdf['STAID'] = cdf['index']

    _file = os.path.join(os.path.dirname(q_data), 'flow_ai_trends.shp')
    sdf.to_file(_file, crs='EPSG:4326')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'

    inshp = os.path.join(root, 'gages', 'selected_gages.shp')
    # lr_ = os.path.join(root, 'analysis', 'trends')

    # lr_ = os.path.join(root, 'analysis', 'uv_trends')
    lr_ = os.path.join(root, 'analysis', 'mv_trends')
    fig_shp = os.path.join(root, 'figures', 'shapefiles', 'mv_trends')
    glb = 'mv_traces'
    monthly_trends(lr_, inshp, glob=glb, out_shape=fig_shp)

    v_ = 'cc_q'
    glb = '{}_bayes'.format(v_)
    cc_qres = os.path.join(root, 'analysis', '{}'.format(v_))
    out_shp = os.path.join(root, 'figures', 'shapefiles', '{}'.format(v_))
    # monthly_cc_qres(cc_qres, inshp, glob=glb, out_shape=out_shp, bayes=True)

    q_trend = os.path.join(root, 'figures', 'shapefiles', 'trends', 'time_q.shp')
    cc_trend = os.path.join(root, 'figures', 'shapefiles', 'trends', 'time_cc.shp')
    # sustainability_trends(q_trend, cc_trend)

    ai_trend = os.path.join(root, 'figures', 'shapefiles', 'trends', 'time_ai.shp')
    # flow_change_climate_coincidence(q_trend, ai_trend)

# ========================= EOF ====================================================================
