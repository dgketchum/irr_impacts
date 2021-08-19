import os
import sys

import ee
import fiona
from shapely import geometry

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(2000)

from assets import is_authorized

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
BASINS = 'users/dgketchum/gages/gage_basins'

ET_ASSET = 'users/dgketchum/ssebop/columbia'


def extract_terraclimate_monthly(tables, years, description):
    fc = ee.FeatureCollection(tables)
    for yr in years:
        for m in range(1, 13):
            m_str, m_str_next = str(m).rjust(2, '0'), str(m + 1).rjust(2, '0')
            if m == 12:
                dataset = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('{}-{}-01'.format(yr, m_str),
                                                                                     '{}-{}-31'.format(yr, m_str))
            else:
                dataset = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('{}-{}-01'.format(yr, m_str),
                                                                                     '{}-{}-01'.format(yr, m_str_next))
            area = ee.Image.pixelArea()
            pet = dataset.select('pet').first().multiply(0.1).multiply(area).rename('etr')
            soil = dataset.select('soil').first().multiply(0.1).multiply(area).rename('sm')
            ppt = dataset.select('pr').first().multiply(area).rename('ppt')

            bands = pet.addBands([soil, ppt])
            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum())

            out_desc = '{}_{}_{}'.format(description, yr, m_str)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket='wudr',
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=['STAID', 'etr', 'sm', 'ppt'])

            task.start()
            print(out_desc)


def extract_gridmet(tables, years=None, description=None, min_years=0):
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
    # fc = ee.FeatureCollection(ee.FeatureCollection(tables).filter(ee.Filter.eq('STAID', '12484500')))

    irr_coll = ee.ImageCollection(RF_ASSET)
    sum = ee.ImageCollection(irr_coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])).sum()
    sum_mask = sum.lt(min_years)

    for yr in years:
        irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).mosaic().select('classification').remap(
            [0, 1, 2, 3], [1, 0, 0, 0])
        irr = irr.mask(sum_mask)

        # et_coll = ee.ImageCollection(ET_ASSET).filter(ee.Filter.date('{}-01-01'.format(yr), '{}-12-31'.format(yr)))
        # et = et_coll.mosaic().mask(irr).multiply(0.001)
        # et = et.reproject(crs='EPSG:5070', scale=30)
        #
        # s, e = '{}-10-01'.format(yr - 1), '{}-09-30'.format(yr)
        # gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(s, e).select('pr', 'etr')
        # gridmet_red = ee.Reducer.sum()
        # gridmet_sums = gridmet.select('pr', 'etr'). \
        #     reduce(gridmet_red).multiply(0.001).rename(['pr', 'etr']).reproject(crs='EPSG:5070', scale=30)

        # area = ee.Image.pixelArea()
        # ppt = gridmet_sums.select('pr').multiply(area).rename('ppt_{}'.format(yr))
        # pet = gridmet_sums.select('etr').multiply(area).rename('etr_{}'.format(yr))
        # crop_cons = et.subtract(gridmet_sums.select('pr')).rename('cc_{}'.format(yr)).multiply(area)

        # late season
        # s, e = '{}-07-01'.format(yr), '{}-10-31'.format(yr)
        # gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(s, e).select('pr', 'etr')
        # gridmet_red = ee.Reducer.sum()
        # gridmet_sums = gridmet.select('pr', 'etr'). \
        #     reduce(gridmet_red).multiply(0.001).rename(['pr', 'etr']).reproject(crs='EPSG:5070', scale=30)

        # ppt_lt = gridmet_sums.select('pr').multiply(area).rename('ppt_lt_{}'.format(yr))
        # pet_lt = gridmet_sums.select('etr').multiply(area).rename('etr_lt_{}'.format(yr))

        area = ee.Image.pixelArea()
        irr = irr.reproject(crs='EPSG:5070', scale=30)
        irr = irr.multiply(area).rename('irr_{}'.format(yr))
        # bands = irr.addBands([ppt, pet, crop_cons])

        data = irr.reduceRegions(collection=fc,
                                 reducer=ee.Reducer.sum(),
                                 scale=30)

        # fields = data.first().propertyNames().remove('.geo')
        out_desc = '{}_Comp_18AUG2021_{}'.format(description, yr)
        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=out_desc,
            bucket='wudr',
            fileNamePrefix=out_desc,
            fileFormat='CSV',
            selectors=['STAID',
                       'irr_{}'.format(yr)])
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
    extract_gridmet(BASINS, years=[i for i in range(1986, 2021)], description='basins', min_years=3)
    # extract_terraclimate_monthly(BASINS, [i for i in range(1984, 2021)], description='terraclim')
    # area('/home/dgketchum/Downloads/bitterroot_fields.shp')
# ========================= EOF ================================================================================
