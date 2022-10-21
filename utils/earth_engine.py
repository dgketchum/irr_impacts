import os
import sys
from calendar import monthrange

import ee

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
BASINS = 'users/dgketchum/gages/gage_basins'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
BUCKET = 'wudr'

ET_ASSET = ee.ImageCollection('users/dgketchum/ssebop/cmbrb')

BOZEMAN = ee.Geometry.Polygon([[-111.19206055457778, 45.587493372544984],
                               [-110.91946228797622, 45.587493372544984],
                               [-110.91946228797622, 45.754947053477565],
                               [-111.19206055457778, 45.754947053477565],
                               [-111.19206055457778, 45.587493372544984]])

NAVAJO = ee.Geometry.Polygon([[-108.50192867920967, 36.38701227276218],
                              [-107.92995297120186, 36.38701227276218],
                              [-107.92995297120186, 36.78068624960868],
                              [-108.50192867920967, 36.78068624960868],
                              [-108.50192867920967, 36.38701227276218]])


def export_gridded_data(tables, years=None, description=None,
                        min_years=0, basins=True):
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
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    # fc = ee.FeatureCollection(ee.FeatureCollection(tables).filter(ee.Filter.eq('STAID', '12484500')))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1991-01-01', '2020-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gt(min_years)
    # sum = remap.sum().mask(irr_mask)

    for yr in years:
        for month in range(4, 11):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_cmb = et_coll.sum().multiply(0.00001).clip(cmb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_corb = et_coll.sum().multiply(0.00001).clip(corb_clip.geometry())

            annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
            et_coll = annual_coll_.filter(ee.Filter.date(s, e))
            et_umrb = et_coll.sum().multiply(0.00001).clip(umrb_clip.geometry())

            et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()
            et = et_sum.mask(irr_mask)

            tclime = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(s, e).select('pr', 'pet', 'aet')
            tclime_red = ee.Reducer.sum()
            tclime_sums = tclime.select('pr', 'pet', 'aet').reduce(tclime_red)
            ppt = tclime_sums.select('pr_sum').multiply(0.001)
            etr = tclime_sums.select('pet_sum').multiply(0.0001)
            swb_aet = tclime_sums.select('aet_sum').mask(irr_mask).multiply(0.0001)

            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            swb_aet = swb_aet.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(swb_aet)

            area = ee.Image.pixelArea()
            irr = irr_mask.multiply(area).rename('irr')
            et = et.multiply(area).rename('et')
            cc = cc.multiply(area).rename('cc')
            ppt = ppt.multiply(area).rename('ppt')
            etr = etr.multiply(area).rename('etr')
            swb_aet = swb_aet.multiply(area).rename('swb_aet')

            if basins:
                selector = ['STAID']
            else:
                selector = ['GEOID', 'STUDYINT']

            if yr > 1990 and month in [x for x in range(4, 11)]:
                bands = irr.addBands([et, cc, ppt, etr, swb_aet])
                select_ = selector + ['irr', 'et', 'cc', 'ppt', 'etr', 'swb_aet']
            else:
                bands = irr.addBands([ppt, etr, swb_aet])
                select_ = selector + ['ppt', 'etr', 'swb_aet', 'irr']

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum(),
                                       scale=30)

            # fields = data.first().propertyNames().remove('.geo')
            # p = data.first().getInfo()['properties']

            out_desc = '{}_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=BUCKET,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)
            task.start()
            print(out_desc)


def export_et_images(polygon, tables, years=None, description=None,
                     min_years=0):
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
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    # fc = ee.FeatureCollection(ee.FeatureCollection(tables).filter(ee.Filter.eq('STAID', '12484500')))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1991-01-01', '2020-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gt(min_years)
    # sum = remap.sum().mask(irr_mask)

    for yr in years:
        for month in range(4, 5):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_cmb = et_coll.sum().multiply(0.00001).clip(cmb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_corb = et_coll.sum().multiply(0.00001).clip(corb_clip.geometry())

            annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
            et_coll = annual_coll_.filter(ee.Filter.date(s, e))
            et_umrb = et_coll.sum().multiply(0.00001).clip(umrb_clip.geometry())

            et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()
            et = et_sum.mask(irr_mask)

            tclime = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(s, e).select('pr', 'pet', 'aet')
            tclime_red = ee.Reducer.sum()
            tclime_sums = tclime.select('pr', 'pet', 'aet').reduce(tclime_red)
            ppt = tclime_sums.select('pr_sum').multiply(0.001)
            etr = tclime_sums.select('pet_sum').multiply(0.0001)
            swb_aet = tclime_sums.select('aet_sum').mask(irr_mask).multiply(0.0001)

            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            swb_aet = swb_aet.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(swb_aet)

            area = ee.Image.pixelArea()
            irr = irr_mask.multiply(area).rename('irr')
            et = et.multiply(area).rename('et')
            cc = cc.multiply(area).rename('cc')
            ppt = ppt.multiply(area).rename('ppt')
            etr = etr.multiply(area).rename('etr')
            swb_aet = swb_aet.multiply(area).rename('swb_aet')

            v = [irr]  # , cc, ppt, etr, swb_aet]
            s = ['irr']  # , 'cc', 'ppt', 'etr', 'swb_aet']

            for var_, st in zip(v, s):
                out_desc = '{}_{}_{}_{}'.format(description, st, yr, month)
                task = ee.batch.Export.image.toCloudStorage(
                    var_,
                    description=out_desc,
                    bucket=BUCKET,
                    fileNamePrefix=out_desc,
                    region=polygon,
                    scale=30,
                    maxPixels=1e13,
                    crs='EPSG:5071')
                task.start()
                print(out_desc)


def export_naip(region):
    dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2017-01-01', '2018-12-31')).mosaic()
    task = ee.batch.Export.image.toCloudStorage(
        dataset,
        description='NAIP_Navajo',
        bucket=BUCKET,
        fileNamePrefix='NAIP_Navajo',
        region=region,
        scale=30,
        maxPixels=1e13,
        crs='EPSG:5071')
    task.start()


def is_authorized():
    try:
        ee.Initialize()  # investigate (use_cloud_api=True)
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


if __name__ == '__main__':
    is_authorized()

# ========================= EOF ================================================================================
