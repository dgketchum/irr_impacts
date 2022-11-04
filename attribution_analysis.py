import os
import json
from pprint import pprint

import numpy as np
from pandas import DataFrame
from dominance_analysis import Dominance


def dominance_analysis(_dir, target, predictors):
    _files = [os.path.join(_dir, 'trends_initial_{}.json'.format(m)) for m in range(4, 11)]

    target_d = {}

    for m, f in enumerate(_files, start=1):
        with open(f, 'r') as f_obj:
            dct = json.load(f_obj)
        for k, v in dct.items():
            print('month', m, k)
            df = DataFrame({kk: vv for kk, vv in v.items() if kk in predictors + [target]})
            corr = df.corr()
            d_regression = Dominance(data=corr, target=target, data_format=1)
            incremental_r_sq = d_regression.incremental_rsquare()
            if m not in target_d.keys():
                target_d[m] = {p: 0 for p in predictors}
            else:
                keys, vals = [k for k in incremental_r_sq.keys()], [v for k, v in incremental_r_sq.items()]
                idx = np.argmax(np.array(vals))
                target_d[m][predictors[idx]] += 1

    pprint(target_d)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'

    analysis_d = os.path.join(root, 'analysis', 'trends')
    dominance_analysis(analysis_d, target='cc_month', predictors=['etr', 'irr'])
# ========================= EOF ====================================================================
