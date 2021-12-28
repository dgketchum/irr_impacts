import os

import geopandas as gpd
import numpy as np
import rasterio
from skimage.filters.rank import majority
from skimage.morphology import disk
from rasterio.plot import show
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def map_fig_one(basins_shp, irrmap, gages_shp, states, all_gages, png):
    adf = gpd.read_file(all_gages)
    sdf = gpd.read_file(states)
    bdf = gpd.read_file(basins_shp)

    src = rasterio.open(irrmap)
    left, bottom, right, top = src.bounds
    irr = src.read()[0, :, :].astype(np.dtype('uint8'))
    src.close()
    irr = majority(irr, disk(2))
    irr = np.ma.masked_where(irr < 5,
                             irr,
                             copy=True)

    gdf = gpd.read_file(gages_shp)
    gdf['import'] = (gdf['cc_q'].values > 0.2) & (gdf['AREA'].values < 7000.)
    gdf = gdf[gdf['import'] == 0]
    gdf = gdf[gdf['cc_q'] >= 0.0]
    gdf = gdf[gdf['cc_q'] <= 0.5]
    gdf['rank'] = gdf['cc_q'].rank(ascending=True)

    fig, ax = plt.subplots(1, figsize=(20, 16))
    show(irr.data, cmap='RdBu', alpha=0.5, ax=ax, extent=(left, right, bottom, top),
         norm=colors.Normalize(vmin=-1.0, vmax=1.0),)
    sdf.geometry.boundary.plot(color=None, edgecolor='k', linewidth=1, alpha=0.8, ax=ax)
    bdf.geometry.boundary.plot(color=None, edgecolor='purple', linewidth=2, ax=ax)
    adf.plot(color='k', linewidth=0, alpha=0.1, ax=ax)
    gdf.plot(column='rank', linewidth=1., cmap=plt.cm.coolwarm, legend=True, scheme='quantiles', ax=ax)
    # locs, labels = plt.xticks()
    # plt.setp(labels, rotation=90)
    # plt.axis('equal')
    # plt.savefig(png)
    plt.xlim(-125, -102)
    plt.ylim(30, 50)
    plt.show()
    # plt.close()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'
    fig_data = os.path.join(root, 'figures')
    fig_shp = os.path.join(root, 'figures', 'fig_shapes')
    fig_tif = os.path.join(root, 'figures', 'fig_tifs')
    gages = os.path.join(fig_shp, 'basin_cc_ratios.shp')
    basins = os.path.join(fig_shp, 'study_basins.shp')
    states_ = os.path.join(fig_shp, 'western_11_states.shp')
    all_gages_ = os.path.join(fig_shp, 'study_gages_all.shp')
    irrmapper = os.path.join(fig_tif, 'irr_freq_merge_360m.tif')
    fig = os.path.join(fig_data, 'map_fig_one.png')
    map_fig_one(basins, irrmapper, gages, states_, all_gages_, fig)
# ========================= EOF ====================================================================
