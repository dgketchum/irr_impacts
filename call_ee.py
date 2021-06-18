import os
import sys

import ee
import fiona
from shapely import geometry

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(2000)

from assets import is_authorized

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper_RF'
BASINS = 'users/dgketchum/gages/gage_basins'
ET_ASSET = 'users/dgketchum/ssebop/columbia'


def reduce_classification(tables, years=None, description=None, min_years=0):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    fc = ee.FeatureCollection(tables)
    # fc = ee.FeatureCollection(ee.FeatureCollection(tables).filter(ee.Filter.eq('STAID', '12352500')))
    irr_coll = ee.ImageCollection(RF_ASSET)
    sum = irr_coll.filterDate('1991-01-01', '2020-12-31').select('classification').sum()
    sum_mask = sum.gt(min_years)

    for yr in years:
        irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
        mask = irr.eq(1)
        mask = mask.updateMask(sum_mask)
        irr = irr.mask(mask)

        et_coll = ee.ImageCollection(ET_ASSET).filter(ee.Filter.date('{}-01-01'.format(yr), '{}-12-31'.format(yr)))
        et = et_coll.mosaic().mask(mask).multiply(0.001)
        et = et.reproject(crs='EPSG:5070', scale=30)

        s, e = '{}-01-01'.format(yr), '{}-12-31'.format(yr)
        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(s, e).select('pr', 'etr')
        gridmet_red = ee.Reducer.sum()
        gridmet_sums = gridmet.select('pr', 'etr'). \
            reduce(gridmet_red).multiply(0.001).rename(['pr', 'etr']).reproject(crs='EPSG:5070', scale=30)

        area = ee.Image.pixelArea()
        ppt = gridmet_sums.select('pr').multiply(area)
        pet = gridmet_sums.select('etr').multiply(area)
        crop_cons = et.subtract(gridmet_sums.select('pr')).rename('cc').multiply(area)

        irr = irr.reproject(crs='EPSG:5070', scale=30)
        irr = irr.multiply(area).rename('irr_{}'.format(yr))
        bands = irr.addBands([ppt, pet, crop_cons])

        data = bands.reduceRegions(collection=fc,
                                   reducer=ee.Reducer.sum(),
                                   scale=30)

        # fields = data.first().propertyNames().remove('.geo')
        out_desc = '{}_{}'.format(description, yr)
        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=out_desc,
            bucket='wudr',
            fileNamePrefix=out_desc,
            fileFormat='CSV')
        task.start()
        print(out_desc)


def area(shp):
    area = 0
    with fiona.open(shp, 'r') as src:
        for f in src:
            g = geometry.shape(f['geometry'])
            area += g.area
    print(area)


if __name__ == '__main__':
    is_authorized()
    reduce_classification(BASINS, years=[i for i in range(1991, 2021)], description='basins', min_years=3)
    # area('/home/dgketchum/Downloads/bitterroot_fields.shp')
# ========================= EOF ================================================================================
