import os
import json
import numpy as np

# CONFUSION = np.array([[4616365., 296629.],
#                       [690902., 85123826.]])

CONFUSION_ = np.array([[4616365., 690902.],
                       [296629., 85123826.]])

CONFUSION_ = np.array([[940., 8.],
                       [60., 992.]])
from hydrograph import hydrograph
from station_lists import VERIFIED_IRRIGATED_HYDROGRAPHS


def consumer_precision(arr):
    """
    If attention is focused on the accuracy of the map as a predictive device, concern is with errors of commission.
    In this situation what is generally termed user’s accuracy may be derived, which is based on the
    ratio of correctly allocated cases of a class relative to the total number of testing cases allocated to
    that class
    :param arr:
    :return:
    """
    c = [(arr[x, x] / sum(arr[x, :])) for x in range(0, arr.shape[1])]
    return c


def producer_recall(arr):
    c = [(arr[x, x] / sum(arr[:, x])) for x in range(0, arr.shape[0])]
    return c


def olofsson_error(trends, metadata, conf_mat):
    """
    p_qplus: map data
    p_plusq : reference data
    :param trends:
    :param metadata:
    :param conf_mat:
    :return:
    """
    with open(trends, 'r') as f:
        trends = json.load(f)

    with open(metadata, 'r') as f:
        metadata = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in trends.items()]
    diter = [i for ll in diter for i in ll]

    for station, period, records in diter:
        if station not in VERIFIED_IRRIGATED_HYDROGRAPHS:
            continue

        w_i = np.array(records['irr_frac'])
        p = consumer_precision(conf_mat)
        w_n = 1 - w_i
        n_i, n_n = sum(conf_mat[:, 0]), sum(conf_mat[:, 1])
        phat_i, phat_n = w_i * p[0], w_n * p[1]
        i_term = ((w_i * phat_i) - (phat_i ** 2)) / (n_i - 1)
        n_term = ((w_n * phat_n) - (phat_n ** 2)) / (n_n - 1)
        s = np.sqrt(i_term + n_term)
        ci = 1.96 * s
        pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    ee_data = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')

    clim_dir = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')
    i_json = os.path.join(root, 'station_metadata/station_metadata.json')
    fig_dir_ = os.path.join(root, 'figures/clim_q_correlations')

    analysis_d = os.path.join(root, 'gridmet_analysis', 'analysis')

    trends_json = os.path.join(analysis_d, 'climate_q_7.json')
    i_json = os.path.join(root, 'station_metadata/station_metadata.json')
    olofsson_error(trends_json, i_json, CONFUSION_)
# ========================= EOF ====================================================================
