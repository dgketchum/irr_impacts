import os

import numpy as np
import rasterio
from pandas import isna
import geopandas as gpd
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from skimage.filters.rank import majority
import matplotlib.patches as mpatches


from figs.scalebar import scale_bar


def map_fig_one(basins_shp, irrmap, gages_shp, states, all_gages, png):
    adf = gpd.read_file(all_gages)
    # bdf = gpd.read_file(basins_shp)

    extent = [-135, -100, 30, 50]

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
    gdf = gdf[gdf['cc_q'] >= 0.0]
    gdf = gdf[gdf['cc_q'] <= 0.5]
    gdf['rank'] = gdf['cc_q'].rank(ascending=True)
    rank_vals = [int(x) for x in np.linspace(0, len(gdf['rank'].values) - 1, 5)]
    sort = sorted(gdf['cc_q'].values)
    cc_lab_vals = [0.0] + [sort[r] for r in rank_vals]

    proj = ccrs.LambertConformal(central_latitude=40,
                                 central_longitude=-110)

    plt.figure(figsize=(24, 16))
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
    for c in np.linspace(0, 1, 5):
        color_ = cmap(c)
        patches.append(plt.Line2D([], [], color=color_, marker='o', label='lab', linewidth=0))

    ax.legend(handles=patches, loc=(0.8, 0.2))
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
    fig = os.path.join(fig_data, 'map_fig_one_debug.png')
    map_fig_one(basins, irrmapper, gages, states_, all_gages_, fig)
# ========================= EOF ====================================================================
