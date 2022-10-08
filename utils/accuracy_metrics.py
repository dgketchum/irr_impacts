import os
import json
import datetime

from pandas import DataFrame
import numpy as np
import ee

from station_lists import STATION_BASINS

ee.Initialize()
BOUNDARIES = 'users/dgketchum/boundaries'

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
E_STATES = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

BASIN = ['users/dgketchum/boundaries/umrb_ylstn_clip',
         'users/dgketchum/boundaries/CMB_RB_CLIP',
         'users/dgketchum/boundaries/CO_RB']

BASIN_MAP = {'missouri': 'users/dgketchum/boundaries/umrb_ylstn_clip',
             'columbia': 'users/dgketchum/boundaries/CMB_RB_CLIP',
             'colorado': 'users/dgketchum/boundaries/CO_RB'}

# Years with at least 1000 reference fields
BASIN_YEARS = {'missouri': [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015],
               'colorado': [1998, 2003, 2006, 2009, 2013, 2016],
               'columbia': [1988, 1996, 1997, 1998, 2001, 2006, 2008, 2009, 2010,
                            2011, 2012, 2013]}

BASIN_F1 = {'missouri': 0.8985,
            'colorado': 0.8345,
            'columbia': 0.8649}

BASIN_CC_ERR = {'missouri': {'rmse': 0.19, 'bias': 0.06},
                'colorado': {'rmse': 0.28, 'bias': -0.14},
                'columbia': {'rmse': 0.23, 'bias': -0.02}}


def confusion(irr_labels, unirr_labels, irr_image, unirr_image, sid, yr, clip=False):
    if sid in ['columbia', 'colorado', 'missouri']:
        fc = ee.FeatureCollection(BASIN_MAP[sid])
        domain = fc.toList(fc.size()).get(0)
        domain = ee.Feature(domain)
    else:
        domain = ee.FeatureCollection('users/dgketchum/gages/gage_basins')
        domain = domain.filterMetadata('STAID', 'equals', sid)
        domain = ee.FeatureCollection(domain)
        domain = domain.toList(domain.size()).get(0)
        domain = ee.Feature(domain)

    true_positive = irr_image.eq(irr_labels)
    false_positive = irr_image.eq(unirr_labels)
    true_negative = unirr_image.eq(unirr_labels)
    false_negative = unirr_image.eq(irr_labels)

    if clip:
        true_positive = true_positive.clip(domain)
        false_positive = false_positive.clip(domain)
        true_negative = true_negative.clip(domain)
        false_negative = false_negative.clip(domain)

    TP = true_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FP = false_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FN = false_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    TN = true_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )

    out = {'TP': TP.getInfo(), 'FP': FP.getInfo(), 'FN': FN.getInfo(), 'TN': TN.getInfo()}
    return out


def create_rf_labels(year):
    rf = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp').filterDate('{}-01-01'.format(year),
                                                                                               '{}-12-31'.format(year))
    rf = rf.select('classification').mosaic()
    irrMask = rf.lt(1)
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not())
    irrImage = ee.Image(1).byte().updateMask(irrMask)
    return irrImage, unirrImage


def create_irrigated_labels(all_data, year):
    if all_data:
        non_irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/dryland')
        fallow = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/fallow')
        irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))
    else:
        root = 'users/dgketchum/validation/'
        non_irrigated = ee.FeatureCollection(root + 'uncultivated')
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'dryland'))
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'wetlands'))

        fallow = ee.FeatureCollection(root + 'fallow')
        irrigated = ee.FeatureCollection(root + 'irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow).map(lambda x: x.buffer(-50))
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))

    irr_labels = ee.Image(1).byte().paint(irrigated, 0)
    irr_labels = irr_labels.updateMask(irr_labels.Not())
    unirr_labels = ee.Image(1).byte().paint(non_irrigated, 0)
    unirr_labels = unirr_labels.updateMask(unirr_labels.Not())

    return irr_labels, unirr_labels


def metrics(arr):
    precision = arr[0, 0] / (arr[0, 1] + arr[0, 0])
    recall = arr[0, 0] / (arr[1, 0] + arr[0, 0])

    return precision, recall


def accuracy_by_watershed(stations, out_dir, super_basin=False):
    csv_out = os.path.join(out_dir, 'basin_accuracy.csv')
    globe_conf = np.zeros((2, 2))

    if super_basin:
        stations = BASIN_YEARS
        stations = {k: [0.0, v] for k, v in stations.items()}
        csv_out = os.path.join(out_dir, 'superbasin_accuracy.csv')
    else:
        with open(stations, 'r') as f:
            stations = json.load(f)

        stations = {k: v for k, v in sorted(stations.items(), key=lambda item: item[1], reverse=True)}

    cols = ['STAID', 'YEAR', 'P', 'R', 'TP', 'FP', 'FN', 'TN']
    ind_len = 0
    for k, v in stations.items():
        ind_len += len(v[1])

    df = DataFrame(columns=cols, index=[x for x in range(ind_len)],
                   data=np.zeros((ind_len, len(cols))))

    ind = 0
    _all = len(stations.keys())
    for num, (sid, (area, years)) in enumerate(stations.items()):
        out_json = os.path.join(out_dir, 'basin_json', '{}.json'.format(sid))
        if os.path.exists(out_json):
            continue
        if not years:
            continue
        basin_conf = np.zeros((2, 2))
        for year in years:
            row = {c: None for c in cols}
            try:
                row['YEAR'] = year
                row['STAID'] = sid

                irr_labels, unirr_labels = create_irrigated_labels(False, int(year))

                if not irr_labels:
                    continue

                irr_image, unirr_image = create_rf_labels(year)
                cmt = confusion(irr_labels, unirr_labels, irr_image, unirr_image, sid, year, clip=True)
                if not cmt:
                    continue
                if cmt['TP']['constant'] + cmt['FN']['constant'] == 0:
                    print('no positive reference data')
                    continue

                print('{} {} {}'.format(sid, year, cmt))
                for pos, ct in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['TP', 'FP', 'FN', 'TN']):
                    globe_conf[pos] += cmt[ct]['constant']
                    basin_conf[pos] += cmt[ct]['constant']
                    row[ct] = cmt[ct]['constant']

                p, r = metrics(basin_conf)
                row['P'], row['R'] = np.round(p, decimals=3), np.round(r, decimals=3)

                if np.isnan(p) or np.isnan(r):
                    continue

                df.loc[ind] = row
                ind += 1

            except Exception as e:
                print(e, row['STAID'], row['YEAR'])
                pass

        p, r = metrics(basin_conf)
        print('prec {:.3f}, rec {:.3f}'.format(p, r))
        print(print('{} {} of {}\n{}\n\n\n'.format(sid, num, _all, basin_conf)))
        jsn = {sid: {k: v for k, v in zip(['TP', 'FP', 'FN', 'TN'], basin_conf.flatten())}}
        with open(out_json, 'w') as f:
            json.dump(jsn, f, indent=4)

        if num % 1 == 0:
            df.to_csv(csv_out)

    print(globe_conf)
    p, r = metrics(globe_conf)
    df.to_csv(csv_out)
    print('prec {}, rec {}'.format(p, r))
    print(datetime.datetime.now())


def basin_accuracy(_dir, all_stations, out_json):
    basin_acc = os.path.join(_dir, 'basin_json')
    js = [os.path.join(basin_acc, x) for x in os.listdir(basin_acc) if x.endswith('.json')]
    stations = [os.path.basename(j).split('.')[0] for j in js]

    out_acc = {}

    with open(all_stations, 'r') as fp:
        dct = json.load(fp)

    sort_ = sorted([(k, v['AREA']) for k, v in dct.items()], key=lambda x: x[1], reverse=True)
    dct = {k: dct[k] for k, v in sort_}

    for sid, meta in dct.items():

        if sid not in STATION_BASINS.keys():
            continue

        if sid in stations:
            j = js[stations.index(sid)]
            with open(j, 'r') as f:
                cmt = json.load(f)[sid]
            conf = np.zeros((2, 2))
            for pos, ct in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['TP', 'FP', 'FN', 'TN']):
                conf[pos] += cmt[ct]

            if conf.sum() == 0:
                pass
            else:
                prec, rec = metrics(conf)
                conf_l = list(conf.flatten())
                ratio = (conf_l[0] + conf_l[2]) / (conf_l[1] + conf_l[3])
                out_acc[sid] = {'f1': 2 * ((prec * rec) / (prec + rec)),
                                'area': meta['AREA'],
                                'STANAME': meta['STANAME'],
                                'conf': conf_l,
                                'ratio': ratio}

        if sid not in out_acc.keys():
            out_acc[sid] = {'f1': BASIN_F1[STATION_BASINS[sid]],
                            'area': meta['AREA'],
                            'STANAME': meta['STANAME'],
                            'conf': None}

    with open(out_json, 'w') as fp:
        json.dump(out_acc, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    # j = os.path.join(root, 'gages', 'watershed_irr_accuracy', 'watershed_irr_training_years.json')
    j_out = os.path.join(root, 'gages', 'gridmet_analysis', 'watershed_accuracy.json')
    j_meta = os.path.join(root, 'gages', 'gridmet_analysis', 'station_metadata.json')
    # j = os.path.join(root, 'gages', 'watershed_irr_accuracy', 'superbasin_irr_training_years.json')
    acc = os.path.join(root, 'gages', 'watershed_irr_accuracy')
    # accuracy_by_watershed(j, out_acc, super_basin=True)

    basin_accuracy(acc, j_meta, j_out)
# ========================= EOF ====================================================================
