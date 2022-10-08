import os
import sys
from calendar import monthrange

import fiona
from pandas import concat, to_datetime
import ee
from utils.earth_engine_assets import is_authorized
import numpy as np
from pandas import read_csv

is_authorized()

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
# RF_ASSET = 'users/dgketchum/IrrMapper/IrrMapper_sw'
BASINS = 'users/dgketchum/gages/gage_basins'
COUNTIES = 'users/dgketchum/boundaries/western_11_co_study'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'

FLUX_SHP = '/media/research/IrrigationGIS/ameriflux/select_flux_sites/select_flux_sites_impacts_ECcorrrected.shp'
FLUX_DIR = '/media/research/IrrigationGIS/ameriflux/ec_data'

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
                                       reducer=ee.Reducer.sum(),
                                       scale=1000)

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


def extract_gridmet_monthly(tables, years, description):
    fc = ee.FeatureCollection(tables)
    for yr in years:
        for m in range(1, 13):
            m_str, m_str_next = str(m).rjust(2, '0'), str(m + 1).rjust(2, '0')
            if m == 12:
                dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(yr, m_str),
                                                                                '{}-{}-31'.format(yr, m_str))
            else:
                dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(yr, m_str),
                                                                                '{}-{}-01'.format(yr, m_str_next))
            area = ee.Image.pixelArea()
            pet = dataset.select('etr').sum().multiply(area).rename('etr')
            ppt = dataset.select('pr').sum().multiply(area).rename('ppt')

            bands = pet.addBands([ppt])
            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum(),
                                       scale=1000)

            out_desc = '{}_{}_{}'.format(description, yr, m_str)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket='wudr',
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=['STAID', 'etr', 'ppt'])

            task.start()
            print(out_desc)


def extract_gridded_data(tables, years=None, description=None,
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
                bucket='wudr',
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)
            task.start()
            print(out_desc)


def export_et_images(polygon, tables, years=None, description=None,
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
                    bucket='wudr',
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
        bucket='wudr',
        fileNamePrefix='NAIP_Navajo',
        region=region,
        scale=30,
        maxPixels=1e13,
        crs='EPSG:5071')
    task.start()


def extract_flux_stations(flux_dir, shp, pixels=1):
    with fiona.open(shp, 'r') as src:
        dct = {}
        for feat in src:
            p = feat['properties']
            if p['basin'] == 'umrb':
                dct[p['site_id']] = p
                dct[p['site_id']]['clip_feat'] = UMRB_CLIP
                dct[p['site_id']]['geo'] = feat['geometry']['coordinates']
            elif p['basin'] == 'cmbrb':
                dct[p['site_id']] = p
                dct[p['site_id']]['clip_feat'] = CMBRB_CLIP
                dct[p['site_id']]['geo'] = feat['geometry']['coordinates']
            elif p['basin'] == 'corb':
                dct[p['site_id']] = p
                dct[p['site_id']]['clip_feat'] = CORB_CLIP
                dct[p['site_id']]['geo'] = feat['geometry']['coordinates']
            else:
                raise ValueError('invalid collection')

    et_comp_all, adf = [], None
    first = True
    for site, props in dct.items():

        if props['basin'] == 'cmbrb':
            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
        elif props['basin'] == 'umrb':
            annual_coll = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
        elif props['basin'] == 'corb':
            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
        else:
            raise ValueError('invalid collection')

        _file = '{}_monthly_data.csv'.format(props['site_id'])
        csv = os.path.join(flux_dir, 'monthly', _file)
        df = read_csv(csv, infer_datetime_format=True, parse_dates=True)
        df = df[df['ET_corr'].notna()]
        et_ssebop = []
        dates = [('{}-01'.format(x[:7]), x) for x in df.date.values]
        df['date'] = to_datetime(df['date'])
        df['site'] = [site for x in range(len(dates))]
        et_corr = [x for x in df['ET_corr'].values]

        geo = ee.Geometry.Point(props['geo'][0], props['geo'][1]).buffer(pixels * 30.0)

        for et_ec, (s, e) in zip(et_corr, dates):
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et = et_coll.sum().multiply(0.01)

            data = et.reduceRegion(geometry=geo,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)
            try:
                ee_obj = data.getInfo()
                et_extract = ee_obj['et']
                if et_extract is None:
                    et_extract = np.nan
            except (ee.EEException, KeyError):
                et_extract = np.nan

            et_ssebop.append(et_extract)
            if not np.any(np.isnan([et_ec, et_extract])):
                et_comp_all.append((et_ec, et_extract))

            # print('{}: {:.2f} et, {:.2f} ssebop, {}'.format(site, et_ec, et_extract, s))

        df['et_ssebop'] = et_ssebop
        if first:
            adf = df
            first = False
        else:
            adf = concat([df, adf], axis=0, ignore_index=True)
        _file = '{}_monthly_data_ee.csv'.format(props['site_id'])
        csv = os.path.join(flux_dir, 'monthly', _file)
        df.to_csv(csv)

    adf.to_csv(os.path.join(flux_dir, 'ec_ssebop_comp.csv'))


if __name__ == '__main__':
    # export_et_images(NAVAJO, BASINS, years=[i for i in range(1986, 2022)],
    #                  description='Navajo', min_years=5,
    #                  basins=True)

    export_naip(NAVAJO)

    # export_et_images(BASINS, years=[i for i in range(1991, 2021)],
    #                  description='DNRC_Basins_30SSEPT2022', min_years=5,
    #                  basins=True)

    # extract_flux_stations(FLUX_DIR, FLUX_SHP, pixels=10)
# ========================= EOF ================================================================================
