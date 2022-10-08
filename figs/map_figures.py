import os
import json
from itertools import product

import numpy as np
from pandas import DataFrame, concat, isna

import fiona
import geopandas as gpd
import cartopy.feature as cf
import cartopy.crs as ccrs
from shapely.geometry import shape

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

from figs.scalebar import scale_bar
from gage_analysis import EXCLUDE_STATIONS


def map_fig_one(basins_shp, irrmap, gages_shp, states, all_gages, png, n_div=5):
    adf = gpd.read_file(all_gages)
    # bdf = gpd.read_file(basins_shp)

    extent = [-138, -100, 30, 50]

    # with rasterio.open(irrmap, 'r') as rsrc:
    #     left, bottom, right, top = rsrc.bounds
    #     irr = rsrc.read()[0, :, :].astype(np.dtype('uint8'))

    # irr = majority(irr, disk(2))
    # irr = np.ma.masked_where(irr == 0,
    #                          irr,
    #                          copy=True)
    # irr[irr >= 1] = 1
    # irr = binary_dilation(irr).astype(irr.dtype)

    gdf = gpd.read_file(gages_shp)
    gdf['import'] = (gdf['cc_q'].values > 0.2) & (gdf['AREA'].values < 7000.)
    gdf = gdf[gdf['import'] == 0]
    [print('{} less than 0.0'.format(r['STAID'])) for i, r in gdf.iterrows() if r['cc_q'] < 0.0]
    gdf = gdf[gdf['cc_q'] >= 0.0]
    [print('{} more than 1.0'.format(r['STAID'])) for i, r in gdf.iterrows() if r['cc_q'] > 1.0]
    gdf = gdf[gdf['cc_q'] <= 1.0]
    gdf['rank'] = gdf['cc_q'].rank(ascending=True)
    rank_vals = [int(x) for x in np.linspace(0, len(gdf['rank'].values) - 1, n_div)]
    sort = sorted(gdf['cc_q'].values)
    cc_lab_vals = [0.0] + [sort[r] for r in rank_vals]
    cc_lab = ['{:.2} to {:.2}'.format(cc_lab_vals[x], cc_lab_vals[x + 1]) for x in range(len(cc_lab_vals) - 1)]

    proj = ccrs.LambertConformal(central_latitude=40,
                                 central_longitude=-110)

    plt.figure(figsize=(16, 15))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    basins = ShapelyFeature(Reader(basins_shp).geometries(),
                            ccrs.PlateCarree(), edgecolor='red')
    ax.add_feature(basins, facecolor='none')

    ax.add_feature(cf.BORDERS)
    ax.add_feature(cf.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'))

    # ax.imshow(irr, transform=ccrs.PlateCarree(), cmap='Greens',
    #           extent=(left, right, bottom, top))

    adf.plot(color='k', linewidth=1, edgecolor='black',
             facecolor='none', ax=ax, transform=ccrs.PlateCarree(), alpha=0.3)

    gdf.plot(column='rank', linewidth=1., cmap='coolwarm', scheme='quantiles',
             legend=False, ax=ax, transform=ccrs.PlateCarree())

    scale_bar(ax, proj, length=300, bars=1)

    cmap = matplotlib.cm.get_cmap('coolwarm')
    patches, labels = [], []
    for c, lab in zip(np.linspace(0, 1, n_div), cc_lab):
        color_ = cmap(c)
        patches.append(plt.Line2D([], [], color=color_, marker='o', label=lab, linewidth=0))

    legend = ax.legend(handles=patches, loc=(0.4, 0.36), title='Crop Consumption to\n    '
                                                               'Summer Flow'
                                                               '\n      (1991 - 2020) ')
    legend.get_frame().set_facecolor('white')
    plt.box(False)
    plt.savefig(png)
    # plt.show()
    plt.close()


def map_fig_two_triple_panel(basins_shp, irrmap, gages_shp, states, all_gages, png, n_div=5):
    adf = gpd.read_file(all_gages)
    # bdf = gpd.read_file(basins_shp)

    # with rasterio.open(irrmap, 'r') as rsrc:
    #     left, bottom, right, top = rsrc.bounds
    #     irr = rsrc.read()[0, :, :].astype(np.dtype('uint8'))

    # irr = majority(irr, disk(2))
    # irr = np.ma.masked_where(irr == 0,
    #                          irr,
    #                          copy=True)
    # irr[irr >= 1] = 1
    # irr = binary_dilation(irr).astype(irr.dtype)

    gdf = gpd.read_file(gages_shp)
    gdf['import'] = (gdf['cc_q'].values > 0.2) & (gdf['AREA'].values < 7000.)
    gdf = gdf[gdf['import'] == 0]
    [print('{} less than 0.0'.format(r['STAID'])) for i, r in gdf.iterrows() if r['cc_q'] < 0.0]
    gdf = gdf[gdf['cc_q'] >= 0.0]
    [print('{} more than 1.0'.format(r['STAID'])) for i, r in gdf.iterrows() if r['cc_q'] > 1.0]
    gdf = gdf[gdf['cc_q'] <= 1.0]
    gdf['rank'] = gdf['cc_q'].rank(ascending=True)
    rank_vals = [int(x) for x in np.linspace(0, len(gdf['rank'].values) - 1, n_div)]
    sort = sorted(gdf['cc_q'].values)
    cc_lab_vals = [0.0] + [sort[r] for r in rank_vals]
    cc_lab = ['{:.2} to {:.2}'.format(cc_lab_vals[x], cc_lab_vals[x + 1]) for x in range(len(cc_lab_vals) - 1)]

    proj = ccrs.LambertConformal(central_latitude=40,
                                 central_longitude=-110)

    fig, axs = plt.subplots(1, 3, figsize=(24, 16),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    for i in range(3):
        ax = axs[i]

        geos = [[g] for g in Reader(basins_shp).geometries()]
        basin = ShapelyFeature(geos[i], ccrs.PlateCarree(), edgecolor='red')
        ax.add_feature(basin, facecolor='none')

        bounds = geos[i].bounds
        extent = bounds[0], bounds[2], bounds[1], bounds[3]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_feature(cf.BORDERS)
        ax.add_feature(cf.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'))

        # ax.imshow(irr, transform=ccrs.PlateCarree(), cmap='Greens',
        #           extent=(left, right, bottom, top))

        adf.plot(color='k', linewidth=1, edgecolor='black',
                 facecolor='none', ax=ax, transform=ccrs.PlateCarree(), alpha=0.3)

        gdf.plot(column='rank', linewidth=1., cmap='coolwarm', scheme='quantiles',
                 legend=False, ax=ax, transform=ccrs.PlateCarree())

        scale_bar(ax, proj, length=300, bars=1)

        cmap = matplotlib.cm.get_cmap('coolwarm')
        patches, labels = [], []
        for c, lab in zip(np.linspace(0, 1, n_div), cc_lab):
            color_ = cmap(c)
            patches.append(plt.Line2D([], [], color=color_, marker='o', label=lab, linewidth=0))

        legend = ax.legend(handles=patches, loc=(0.4, 0.36), title='Crop Consumption to\n    '
                                                                   'Summer Flow'
                                                                   '\n      (1991 - 2020) ')
        legend.get_frame().set_facecolor('white')
    plt.box(False)
    plt.savefig(png)
    # plt.show()
    plt.close()


def monthly_trends(regressions_dir, in_shape, glob=None, out_shape=None, bayes=True):

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

    if bayes:
        trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_q']
    else:
        trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_aim', 'time_q', 'time_etr', 'time_ppt', 'time_irr']

    range_ = [x for x in range(1, 13)]
    for var in trc_subdirs:
        # if var != 'time_qres':
        #     continue
        first = True
        for m in range_:

            d = trends_dct[m]

            if var == 'time_cc' and m in [1, 2, 3, 11, 12]:
                data = {k: 0.0 for k, v in d.items() if k not in EXCLUDE_STATIONS}
            else:
                if bayes:
                    data = {}
                    for k, v in d.items():
                        if not v[var]:
                            data[k] = 0
                        elif np.sign(v[var]['hdi_2.5%']) == np.sign(v[var]['hdi_97.5%']):
                            data[k] = v[var]['mean']
                        else:
                            data[k] = 0
                else:
                    data = {k: v[var]['b'] if v[var]['p'] < 0.05 else 0.0 for k, v in d.items() if
                            k not in EXCLUDE_STATIONS}

            index, data = [i[0] for i in data.items()], [i[1] for i in data.items()]
            if first:
                df = DataFrame(data=data, index=index, columns=[m])
                first = False
            else:
                c = DataFrame(data=data, index=index, columns=[m])
                df = concat([df, c], axis=1)

        vals = df.values
        vals[np.isnan(vals)] = 0.0
        gdf = gpd.GeoDataFrame(df)
        area_arr = np.array([areas[_id] for _id in gdf.index])
        areas = {k: (a - min(area_arr)) / (max(area_arr) - min(area_arr)) for k, a in areas.items()}
        gdf['STANAME'] = [names[_id] for _id in gdf.index]
        geo = [geo_[_id] for _id in gdf.index]

        gdf.drop(columns=['STANAME'], inplace=True)

        if bayes:
            shp_file = os.path.join(out_shape, '{}_median_bayes.shp'.format(var))
        else:
            shp_file = os.path.join(out_shape, '{}_median_mk.shp'.format(var))

        gdf[gdf[[i for i in range(1, 13)]] == 0.0] = np.nan
        gdf['median'] = gdf[[i for i in range(1, 13)]].median(axis=1)
        gdf['median'][isna(gdf['median'])] = 0.0
        gdf.loc[:, :] = np.where(np.isfinite(gdf.values), gdf.values, np.zeros_like(gdf.values))
        gdf['geometry'] = geo
        gdf['AREA'] = [areas[_id] for _id in gdf.index]
        gdf['STANAME'] = [names[_id] for _id in gdf.index]
        cols = [str(c) for c in gdf.columns]
        gdf.columns = cols
        gdf.to_file(shp_file, crs='epsg:4326')
        print(shp_file)


def monthly_cc_qres(regressions_dir, in_shape, glob=None, out_shape=None, bayes=False):

    with fiona.open(in_shape, 'r') as src:
        feats = [f for f in src]

    geo = {f['properties']['STAID']: shape(f['geometry']) for f in feats}
    areas = {f['properties']['STAID']: f['properties']['AREA'] for f in feats}
    names = {f['properties']['STAID']: f['properties']['STANAME'] for f in feats}

    trends_dct = {}

    l = [os.path.join(regressions_dir, x) for x in os.listdir(regressions_dir) if glob in x]

    for f in l:
        m = int(os.path.basename(f).split('_acc.')[0].split('_')[-1])
        with open(f, 'r') as _file:
            dct = json.load(_file)

        trends_dct.update({m: dct})

    bayes_sig = 0
    p_sig = 0

    first = True
    for m in range(1, 13):

        d = trends_dct[m]
        data = {}

        for k, v in d.items():
            if k in EXCLUDE_STATIONS:
                continue
            has_sig = False
            slopes_ = []
            for kk, vv in v.items():
                if not isinstance(vv, dict):
                    continue
                else:

                    if not bayes and vv['p'] < 0.05:
                        slopes_.append(vv['b'])
                        has_sig = True
                    elif bayes and 'cc_qres' in vv.keys():
                        if np.sign(vv['cc_qres']['hdi_2.5%']) == np.sign(vv['cc_qres']['hdi_97.5%']):
                            slopes_.append(vv['cc_qres']['mean'])
                            has_sig = True
                            bayes_sig += 1
                    if bayes and vv['p'] < 0.05:
                        p_sig += 1

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

    gdf = gpd.GeoDataFrame(df)
    gdf.geometry = [geo[_id] for _id in gdf.index]

    area_arr = np.array([areas[_id] for _id in gdf.index])
    areas = {k: (a - min(area_arr)) / (max(area_arr) - min(area_arr)) for k, a in areas.items()}
    gdf['AREA'] = [areas[_id] for _id in gdf.index]

    shp_file = os.path.join(out_shape, '{}_median_acc.shp'.format(glob))

    geo = gdf['geometry']
    gdf.drop(columns=['AREA', 'geometry'], inplace=True)
    gdf[gdf[[i for i in range(1, 13)]] == 0.0] = np.nan
    gdf['median'] = gdf[[i for i in range(1, 13)]].median(axis=1)
    gdf['median'][isna(gdf['median'])] = 0.0
    gdf.loc[:, :] = np.where(np.isfinite(gdf.values), gdf.values, np.zeros_like(gdf.values))
    gdf['geometry'] = geo
    gdf['AREA'] = [areas[_id] for _id in gdf.index]
    gdf['STANAME'] = [names[_id] for _id in gdf.index]
    cols = [str(c) for c in gdf.columns]
    gdf.columns = cols
    gdf.to_file(shp_file, crs='epsg:4326')


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
    _file = os.path.join(os.path.dirname(q_data), 'sustainability_qres.shp')
    sdf.to_file(_file, crs='EPSG:4326')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'
    fig_data = os.path.join(root, 'figures', 'map_one')
    gages = os.path.join(fig_data, 'basin_cc_ratios_summer_7FEB2022.shp')
    basins = os.path.join(fig_data, 'study_basins.shp')
    states_ = os.path.join(fig_data, 'western_states_11_wgs.shp')
    all_gages_ = os.path.join(fig_data, 'study_gages_all.shp')
    irrmapper = os.path.join(fig_data, 'irr_freq_merge_360m.tif')
    fig = os.path.join(fig_data, 'map_fig_one_summer_q.png')
    # map_fig_one(basins, irrmapper, gages, states_, all_gages_, fig)

    study_area_ = os.path.join(root, 'figures', 'fig_shapes', 'study_basins.shp')
    inshp = os.path.join(root, 'gage_loc_usgs', 'selected_gages.shp')
    lr_ = os.path.join(root, 'gridmet_analysis', 'analysis')

    out_shp = os.path.join(root, 'gridmet_analysis', 'fig_shapes')
    figs = os.path.join(root, 'gridmet_analysis', 'figures', 'trends_maps')
    # glb = 'mk_trends_'
    # glb = 'trends_'
    glb = 'bayes_trend_'
    # monthly_trends(lr_, inshp, glob=glb, out_shape=out_shp, bayes=True)

    glb = 'bayes_cc_qres'
    figs = os.path.join(root, 'gridmet_analysis', 'figures', 'cc_qres_maps')
    monthly_cc_qres(lr_, inshp, glob=glb, out_shape=out_shp, bayes=True)

    q_trend = os.path.join(root, 'gridmet_analysis', 'fig_shapes', 'time_qres_median_bayes.shp')
    cc_trend = os.path.join(root, 'gridmet_analysis', 'fig_shapes', 'time_cc_median_bayes.shp')
    # sustainability_trends(q_trend, cc_trend)


# ========================= EOF ====================================================================
