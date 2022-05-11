import os
import json

import numpy as np

from gage_analysis import EXCLUDE_STATIONS


def count_impacted_gages(clim_q, bayes, out):
    with open(clim_q, 'r') as clm:
        clim_dct = json.load(clm)

    with open(bayes, 'r') as qcc:
        cc_dct = json.load(qcc)

    cc_gages = list(cc_dct)

    sig_clim_q_gages, sig_clim_q_ct, lag = [], 0, []
    area = []
    for k, v in clim_dct.items():
        if k not in cc_gages:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    if vv['pval'] < 0.05:
                        sig_clim_q_ct += 1
                        sig_clim_q_gages.append(k)
                        lag.append(vv['lag'])
                        area.append(v['SQMI'])
    q_clim_set = set(sig_clim_q_gages)
    print('{} gages'.format(len(q_clim_set)))

    sig_p_q_cc_gages, sig_p_q_cc_ct = [], 0
    sig_q_cc_gages, sig_q_cc_ct, sig_q_cc_b = [], 0, []
    sig_t_cc_gages, sig_t_cc_ct, sig_t_cc_b = [], 0, []
    sig_t_q_gages, sig_t_q_ct, sig_t_q_b = [], 0, []
    q_months, cc_months = [], []
    imapact_assessed = 0
    impact_dct, coincident_impacts = {}, {}

    sig_q_cc_b_val = None
    sig_t_cc_b_val = None
    sig_t_q_b_val = None

    for k, v in cc_dct.items():

        if k in EXCLUDE_STATIONS:
            continue

        impacted = False
        if isinstance(v, dict):
            for kk, vv in v.items():

                if isinstance(vv, dict):
                    imapact_assessed += 1
                    bool_t_cc, bool_t_q, bool_q_cc = False, False, False
                    if vv['res_sig'] < 0.05:
                        sig_p_q_cc_ct += 1
                        sig_p_q_cc_gages.append(k)

                    try:

                        bayes_est = vv['cc_qres']
                        if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):

                            bool_q_cc = True
                            if not impacted:
                                impacted = True
                                impact_dct[k] = {}
                                impact_dct[k][kk] = bayes_est['mean']
                            else:
                                impact_dct[k][kk] = bayes_est['mean']

                            sig_q_cc_gages.append(k)
                            sig_q_cc_ct += 1
                            sig_q_cc_b_val = bayes_est['mean']
                            sig_q_cc_b.append(sig_q_cc_b_val)

                            s, e = vv['q_window'].split('-')
                            [q_months.append(x) for x in range(int(s), int(e) + 1)]

                            s, e = kk.split('-')
                            [cc_months.append(x) for x in range(int(s), int(e) + 1)]

                            # indent to check coincident trends and impacts
                            bayes_est = vv['time_cc']
                            if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                                sig_t_cc_gages.append(k)
                                sig_t_cc_ct += 1
                                sig_t_cc_b_val = bayes_est['mean']
                                sig_t_cc_b.append(sig_t_cc_b_val)
                                bool_t_cc = True

                            bayes_est = vv['time_qres']
                            if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                                sig_t_q_gages.append(k)
                                sig_t_q_ct += 1
                                sig_t_q_b_val = bayes_est['mean']
                                sig_t_q_b.append(sig_t_q_b_val)
                                bool_t_q = True

                        if all([bool_q_cc, bool_t_cc, bool_t_q]):
                            if k not in coincident_impacts.keys():
                                coincident_impacts[k] = {}
                            coincident_impacts[k][kk] = {'cc_qres': sig_q_cc_b_val,
                                                         'time_cc': sig_t_cc_b_val,
                                                         'time_qres': sig_t_q_b_val,
                                                         'q_window': vv['q_window']}

                    except KeyError as e:
                        print(k, kk, e)
                        continue

        if not impacted:
            impact_dct[k] = 'None'

    with open(out, 'w') as fp:
        json.dump(coincident_impacts, fp, indent=4)

    clim_q_set = set(sig_clim_q_gages)
    sig_q_cc_set = set(sig_q_cc_gages)
    sig_t_cc_set = set(sig_t_cc_gages)
    sig_t_q_set = set(sig_t_q_gages)
    q_ct_mo = {i: np.count_nonzero(q_months == i) for i in np.unique(q_months)}
    cc_ct_mo = {i: np.count_nonzero(cc_months == i) for i in np.unique(cc_months)}

    # sum([1 if x > 0.0 else 0 for x in sig_t_cc_b]) / float(len(sig_t_cc_b))

    print()


def count_coincident_trends_impacts(metadata):
    with open(metadata, 'r') as clm:
        dct = json.load(clm)

    counts = {'ccqneg_ccpos_qneg': 0,
              'ccqpos_ccpos_qneg': 0,
              'ccqneg_ccneg_qpos': 0,
              'ccqpos_ccneg_qpos': 0,
              'mix': 0,
              'ccqneg_ccpos_qneg_qmo': [],
              'ccqpos_ccpos_qneg_qmo': [],
              'ccqneg_ccneg_qpos_qmo': [],
              'ccqpos_ccneg_qpos_qmo': [],
              'ccqneg_ccpos_qneg_ccmo': [],
              'ccqpos_ccpos_qneg_ccmo': [],
              'ccqneg_ccneg_qpos_ccmo': [],
              'ccqpos_ccneg_qpos_ccmo': [],
              }

    bidir_gages, gages_unsus, gages_sus = [], [], []

    for k, v in dct.items():
        bidirectional = [False, False]

        for kk, vv in v.items():

            s, e = vv['q_window'].split('-')
            q_mos = [x for x in range(int(s), int(e) + 1)]

            s, e = kk.split('-')
            cc_mos = [x for x in range(int(s), int(e) + 1)]

            ccqres, tcc, tq = vv['cc_qres'], vv['time_cc'], vv['time_qres']

            if ccqres < 0 < tcc and tq < 0:

                counts['ccqneg_ccpos_qneg'] += 1
                gages_sus.append(k)
                bidirectional[0] = True
                [counts['ccqneg_ccpos_qneg_qmo'].append(x) for x in q_mos]
                [counts['ccqneg_ccpos_qneg_ccmo'].append(x) for x in cc_mos]

            elif ccqres < 0 < tq and tcc < 0:
                counts['ccqneg_ccneg_qpos'] += 1
                gages_unsus.append(k)
                bidirectional[1] = True
                [counts['ccqneg_ccneg_qpos_qmo'].append(x) for x in q_mos]
                [counts['ccqneg_ccneg_qpos_ccmo'].append(x) for x in cc_mos]

            elif ccqres > 0 > tq and 0 < tcc:
                counts['ccqpos_ccneg_qpos'] += 1
                gages_unsus.append(k)
                bidirectional[1] = True
                [counts['ccqpos_ccpos_qneg_qmo'].append(x) for x in q_mos]
                [counts['ccqpos_ccpos_qneg_ccmo'].append(x) for x in cc_mos]

            elif ccqres > 0 > tcc and 0 < tq:
                counts['ccqpos_ccneg_qpos'] += 1
                gages_unsus.append(k)
                bidirectional[1] = True
                [counts['ccqpos_ccneg_qpos_qmo'].append(x) for x in q_mos]
                [counts['ccqpos_ccneg_qpos_ccmo'].append(x) for x in cc_mos]

            elif ccqres > 0:
                counts['mix'] += 1
                # print(k, kk, vv)

        if all(bidirectional):
            bidir_gages.append(k)

    gages_sus_set, gages_unsus_set = sorted(list(set(gages_sus))), sorted(list(set(gages_unsus)))

    def mode(arr):
        vals, counts = np.unique(arr, return_counts=True)
        idx = np.argmax(counts)
        return vals[idx]

    ccqneg_ccpos_qneg_qmo = mode(counts['ccqneg_ccpos_qneg_qmo'])
    ccqneg_ccpos_qneg_cmo = mode(counts['ccqneg_ccpos_qneg_ccmo'])
    ccqneg_ccneg_qpos_qmo = mode(counts['ccqneg_ccneg_qpos_qmo'])
    ccqneg_ccneg_qpos_cmo = mode(counts['ccqneg_ccneg_qpos_ccmo'])

    ccqpos_ccpos_qneg_qmo = mode(counts['ccqpos_ccpos_qneg_qmo'])
    ccqpos_ccpos_qneg_cmo = mode(counts['ccqpos_ccpos_qneg_ccmo'])
    ccqpos_ccneg_qpos_qmo = mode(counts['ccqpos_ccneg_qpos_qmo'])
    ccqpos_ccneg_qpos_cmo = mode(counts['ccqpos_ccneg_qpos_ccmo'])

    print(counts)
    # TODO: rewrite json to include q_window values


if __name__ == '__main__':
    rt = '/media/research/IrrigationGIS/gages/station_metadata'
    climr = os.path.join(rt, 'basin_climate_response_all.json')
    bayes_ = os.path.join(rt, 'cci_impacted_bayes_ccerr_0.233_qreserr_0.17.json')
    out_ = os.path.join(rt, 'cci_impacted_bayes_ccerr_0.233_qreserr_0.17_forShape.json')
    count_impacted_gages(climr, bayes_, out_)

    count_coincident_trends_impacts(out_)
# ========================= EOF ====================================================================
