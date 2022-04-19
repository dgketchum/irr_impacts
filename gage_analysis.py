import os
import json
from datetime import date

import fiona
import numpy as np
import warnings

from hydrograph import hydrograph
from . import EXCLUDE_STATIONS

warnings.filterwarnings('ignore')


def water_balance_ratios(metadata, ee_series, stations=None, metadata_out=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    dct = {}
    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS:
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        cdf = hydrograph(_file)
        cdf['cci'] = cdf['cc'] / cdf['irr']
        years = [x for x in range(1991, 2021)]
        cc_dates = [(date(y, 5, 1), date(y, 10, 31)) for y in years]
        clim_dates = [(date(y, 5, 1), date(y, 10, 31)) for y in years]
        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in clim_dates])
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
        irr = np.array([cdf['irr'][d[0]: d[1]].sum() for d in cc_dates])
        cci = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])
        if not np.all(irr > 0.0):
            continue
        print('cci: {:.3f}, {}'.format(np.mean(cci), v['STANAME']))

        dct[sid] = v
        dct[sid].update({'IAREA': (np.mean(irr)).item()})
        dct[sid].update({'cc_q': (cc.sum() / q.sum()).item()})
        dct[sid].update({'cci': (np.mean(cci)).item()})
        dct[sid].update({'q_ppt': (q.sum() / ppt.sum()).item()})
        dct[sid].update({'ai': (etr.sum() / ppt.sum()).item()})

    if metadata_out:
        with open(metadata_out, 'w') as fp:
            json.dump(dct, fp, indent=4, sort_keys=False)
    if stations:
        with fiona.open(stations, 'r') as src:
            features = [f for f in src]
            meta = src.meta

        [meta['schema']['properties'].update({k: 'float:19.11'}) for k, v in dct[sid].items()
         if k not in meta['schema']['properties'].keys() and isinstance(v, float)]

        out_shp = os.path.join(os.path.dirname(stations),
                               os.path.basename(metadata_out).replace('json', 'shp'))
        with fiona.open(out_shp, 'w', **meta) as dst:
            for f in features:
                sid = f['properties']['STAID']
                if sid in dct.keys():
                    d = {k: v for k, v in dct[sid].items() if isinstance(v, str)}
                    d.update({k: v for k, v in dct[sid].items() if isinstance(v, float)})
                    d['STAID'] = sid
                    f['properties'] = d
                    dst.write(f)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
