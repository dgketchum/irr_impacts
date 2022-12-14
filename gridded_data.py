import os
import sys
from calendar import monthrange

import numpy as np
import ee

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'


def get_geomteries():
    bozeman = ee.Geometry.Polygon([[-111.19206055457778, 45.587493372544984],
                                   [-110.91946228797622, 45.587493372544984],
                                   [-110.91946228797622, 45.754947053477565],
                                   [-111.19206055457778, 45.754947053477565],
                                   [-111.19206055457778, 45.587493372544984]])

    navajo = ee.Geometry.Polygon([[-108.50192867920967, 36.38701227276218],
                                  [-107.92995297120186, 36.38701227276218],
                                  [-107.92995297120186, 36.78068624960868],
                                  [-108.50192867920967, 36.78068624960868],
                                  [-108.50192867920967, 36.38701227276218]])

    test_point = ee.Geometry.Point(-110.644980, 45.970375)

    western_us = ee.Geometry.Polygon([[-127.00073221292574, 30.011505140554807],
                                      [-100.63354471292574, 30.011505140554807],
                                      [-100.63354471292574, 49.908396143431744],
                                      [-127.00073221292574, 49.908396143431744],
                                      [-127.00073221292574, 30.011505140554807]])

    return bozeman, navajo, test_point, western_us


def export_gridded_data(tables, bucket, years, description, features=None, min_years=0, debug=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param features:
    :param bucket:
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    initialize()
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_years)
    irr_mask = irr_min_yr_mask

    for yr in years:
        for month in range(1, 13):
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

            gm_ppt, gm_etr = extract_gridmet_monthly(yr, month)

            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            swb_aet = swb_aet.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            gm_ppt = gm_ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            gm_etr = gm_etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(swb_aet)

            area = ee.Image.pixelArea()
            irr = irr_mask.multiply(area).rename('irr')
            et = et.multiply(area).rename('et')
            cc = cc.multiply(area).rename('cc')
            ppt = ppt.multiply(area).rename('ppt')
            etr = etr.multiply(area).rename('etr')
            gm_ppt = gm_ppt.multiply(area).rename('gm_ppt')
            gm_etr = gm_etr.multiply(area).rename('gm_etr')
            swb_aet = swb_aet.multiply(area).rename('swb_aet')

            if yr > 1986 and month in np.arange(4, 11):
                bands = irr.addBands([et, cc, ppt, etr, swb_aet, gm_ppt, gm_etr])
                select_ = ['STAID', 'irr', 'et', 'cc', 'ppt', 'etr', 'swb_aet', 'gm_ppt', 'gm_etr']
            else:
                bands = ppt.addBands([etr, gm_ppt, gm_etr])
                select_ = ['STAID', 'ppt', 'etr', 'gm_ppt', 'gm_etr']

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum(),
                                       scale=30)

            if debug:
                pt = bands.sample(region=get_geomteries()[2],
                                  numPixels=1,
                                  scale=30)
                p = pt.first().getInfo()['properties']
                print('propeteries {}'.format(p))

            out_desc = '{}_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)
            task.start()
            print(out_desc)


def extract_ndvi_change(tables, bucket, features=None):
    initialize()
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))

    roi = get_geomteries()[-1]
    early = landsat_masked(1987, 1991, 182, 243, roi)
    late = landsat_masked(2017, 2021, 182, 243, roi)
    early_mean = ee.Image(early.map(lambda x: x.normalizedDifference(['B5', 'B4'])).median())
    late_mean = ee.Image(late.map(lambda x: x.normalizedDifference(['B5', 'B4'])).median())
    ndvi_diff = late_mean.subtract(early_mean).rename('nd_diff')
    # ndvi_diff = ndvi_diff.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

    dataset = ee.ImageCollection('USDA/NASS/CDL').filter(ee.Filter.date('2013-01-01', '2017-12-31'))
    cultivated = dataset.select('cultivated').mode()

    ndvi_cult = ndvi_diff.mask(cultivated.eq(2))
    increase = ndvi_cult.gt(0.2).rename('gain')
    decrease = ndvi_cult.lt(-0.2).rename('loss')
    change = increase.addBands([decrease])
    change = change.mask(cultivated)
    change = change.multiply(ee.Image.pixelArea())

    fc = fc.filterMetadata('STAID', 'equals', '13269000')

    change = change.reduceRegions(collection=fc,
                                  reducer=ee.Reducer.sum(),
                                  scale=30)
    p = change.first().getInfo()
    out_desc = 'ndvi_change'
    selectors = ['STAID', 'loss', 'gain']
    task = ee.batch.Export.table.toCloudStorage(
        change,
        description=out_desc,
        bucket=bucket,
        fileNamePrefix=out_desc,
        fileFormat='CSV',
        selectors=selectors)
    task.start()
    print(out_desc)


def extract_gridmet_monthly(year, month):
    m_str, m_str_next = str(month).rjust(2, '0'), str(month + 1).rjust(2, '0')
    if month == 12:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-31'.format(year, m_str))
    else:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-01'.format(year, m_str_next))
    pet = dataset.select('etr').sum().multiply(0.001).rename('gm_etr')
    ppt = dataset.select('pr').sum().multiply(0.001).rename('gm_ppt')
    return ppt, pet


def export_et_images(polygon, tables, bucket, years=None, description=None,
                     min_years=0):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    """
    fc = ee.FeatureCollection(tables)
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    # fc = ee.FeatureCollection(ee.FeatureCollection(tables).filter(ee.Filter.eq('STAID', '12484500')))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gt(min_years)
    first = True
    img = None

    for yr in years:
        for month in range(4, 11):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            # irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            # irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

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
            # et = et_sum.mask(irr_mask)
            et = et_sum.mask(irr_min_yr_mask)

            tclime = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(s, e).select('pr', 'pet', 'aet')
            tclime_red = ee.Reducer.sum()
            tclime_sums = tclime.select('pr', 'pet', 'aet').reduce(tclime_red)
            # swb_aet = tclime_sums.select('aet_sum').mask(irr_mask).multiply(0.0001)
            swb_aet = tclime_sums.select('aet_sum').mask(irr_min_yr_mask).multiply(0.0001)

            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            swb_aet = swb_aet.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(swb_aet).rename('cc_{}_{}'.format(month, yr))

            if first:
                img = cc
                first = False
            else:
                img = img.add(cc)

    roi = ee.FeatureCollection(polygon).first().geometry()
    img = img.clip(roi)

    # pt = img.sample(region=get_geomteries()[2],
    #                 numPixels=1,
    #                 scale=30)
    # p = pt.first().getInfo()['properties']
    # print('propeteries {}'.format(p))

    out_desc = '{}_int'.format(description)
    task = ee.batch.Export.image.toCloudStorage(
        img,
        description=out_desc,
        bucket=bucket,
        fileNamePrefix=out_desc,
        region=roi,
        scale=30,
        maxPixels=1e13,
        crs='EPSG:5071')
    task.start()
    print(out_desc)


def export_naip(region, bucket):
    dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2017-01-01', '2018-12-31')).mosaic()
    task = ee.batch.Export.image.toCloudStorage(
        dataset,
        description='NAIP_Navajo',
        bucket=bucket,
        fileNamePrefix='NAIP_Navajo',
        region=region,
        scale=30,
        maxPixels=1e13,
        crs='EPSG:5071')
    task.start()


def landsat_c2_sr(input_img):
    # credit: cgmorton; https://github.com/Open-ET/openet-core-beta/blob/master/openet/core/common.py

    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                    'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))

    prep_image = input_img \
        .select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                   0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _cloud_mask(i):
        qa_img = i.select(['QA_PIXEL'])
        cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))
        sat_mask = i.select(['QA_RADSAT']).gt(0)
        cloud_mask = cloud_mask.Or(sat_mask)

        cloud_mask = cloud_mask.Not().rename(['cloud_mask'])

        return cloud_mask

    mask = _cloud_mask(input_img)

    image = prep_image.updateMask(mask).copyProperties(input_img, ['system:time_start'])

    return image


def landsat_masked(start_year, end_year, doy_start, doy_end, roi):
    start = '{}-01-01'.format(start_year)
    end_date = '{}-01-01'.format(end_year)

    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).filter(ee.Filter.calendarRange(
        doy_start, doy_end, 'day_of_year')).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).filter(ee.Filter.calendarRange(
        doy_start, doy_end, 'day_of_year')).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).filter(ee.Filter.calendarRange(
        doy_start, doy_end, 'day_of_year')).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))
    return lsSR_masked


def initialize():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))


if __name__ == '__main__':
    initialize()
    basins = 'users/dgketchum/gages/gage_basins'
    bucket = 'wudr'
    basin = 'users/dgketchum/boundaries/shields'
    export_et_images(basin, basins, bucket, years=list(range(1987, 2021)),
                     description='shields', min_years=33)

    # extract_ndvi_change(basins, bucket)
# ========================= EOF ================================================================================
