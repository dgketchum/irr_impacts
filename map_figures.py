import os

import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import majority

import geopandas as gpd
import cartopy.io.shapereader as shpreader
import cartopy.feature as cf
import cartopy.crs as ccrs
import rasterio
from rasterio.plot import show
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scalebar import scale_bar


def map_fig_one(basins_shp, irrmap, gages_shp, states, all_gages, png):
    adf = gpd.read_file(all_gages)
    sdf = gpd.read_file(states)
    bdf = gpd.read_file(basins_shp)

    with rasterio.open(irrmap, 'r') as rsrc:
        left, bottom, right, top = rsrc.bounds
        irr = rsrc.read()[0, :, :].astype(np.dtype('uint8'))

    # irr = majority(irr, disk(2))
    irr = np.ma.masked_where(irr == 0,
                             irr,
                             copy=True)

    gdf = gpd.read_file(gages_shp)
    gdf['import'] = (gdf['cc_q'].values > 0.2) & (gdf['AREA'].values < 7000.)
    gdf = gdf[gdf['import'] == 0]
    gdf = gdf[gdf['cc_q'] >= 0.0]
    gdf = gdf[gdf['cc_q'] <= 0.5]
    gdf['rank'] = gdf['cc_q'].rank(ascending=True)

    proj = ccrs.LambertConformal(central_latitude=40,
                                 central_longitude=-110)
    fig = plt.figure(figsize=(40, 30))
    ax = plt.axes(projection=proj)
    ax.set_extent([-127, -102, 30, 52], crs=ccrs.PlateCarree())

    shape_feature = ShapelyFeature(Reader(basins_shp).geometries(),
                                   ccrs.PlateCarree(), edgecolor='red')
    ax.add_feature(shape_feature, facecolor='none')

    ax.add_feature(cf.BORDERS)
    ax.add_feature(cf.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'))

    ax.imshow(irr, transform=ccrs.PlateCarree(), cmap='jet_r',
              extent=(left, right, bottom, top))

    sdf.geometry.boundary.plot(color=None, edgecolor='k', linewidth=1, alpha=0.8, ax=ax)
    bdf.geometry.boundary.plot(color=None, edgecolor='red', linewidth=0.75, ax=ax)
    adf.plot(color='k', linewidth=0, edgecolor='black',
             facecolor='none', ax=ax, transform=ccrs.PlateCarree())
    gdf.plot(column='rank', linewidth=1., cmap='jet', scheme='quantiles',
             legend=True, ax=ax, transform=ccrs.PlateCarree())
    # scale_bar(ax, proj, length=300, bars=1)

    plt.box(False)
    plt.savefig(png)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'
    fig_data = os.path.join(root, 'figures', 'map_one')
    gages = os.path.join(fig_data, 'basin_cc_ratios.shp')
    basins = os.path.join(fig_data, 'study_basins.shp')
    states_ = os.path.join(fig_data, 'western_states_11_wgs.shp')
    all_gages_ = os.path.join(fig_data, 'study_gages_all.shp')
    irrmapper = os.path.join(fig_data, 'irr_freq_merge_360m.tif')
    fig = os.path.join(fig_data, 'map_fig_one.png')
    map_fig_one(basins, irrmapper, gages, states_, all_gages_, fig)
# ========================= EOF ====================================================================
