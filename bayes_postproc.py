import os
import json

import numpy as np
import matplotlib.pyplot as plt

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
    impact_dct, coincident_impacts, mixed_sign = {}, {}, {}

    sig_q_cc_b_val = None
    sig_t_cc_b_val = None
    sig_t_q_b_val = None
    ct_mixed, ct_single_sign = 0, 0

    for k, v in cc_dct.items():

        ccq_vals = []
        multi_impact = False

        if len([k for k in v.keys() if isinstance(v, dict)]) > 1:
            multi_impact = True

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
                        bayes_est = vv['time_qres']
                        if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                            sig_t_q_gages.append(k)
                            sig_t_q_ct += 1
                            sig_t_q_b_val = bayes_est['mean']
                            sig_t_q_b.append(sig_t_q_b_val)
                            bool_t_q = True

                        bayes_est = vv['cc_qres']
                        if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):

                            bool_q_cc = True
                            if not impacted:
                                impacted = True
                                impact_dct[k] = {}
                                impact_dct[k][kk] = (bayes_est['mean'], vv['q_window'])
                            else:
                                impact_dct[k][kk] = (bayes_est['mean'], vv['q_window'])

                            sig_q_cc_gages.append(k)
                            sig_q_cc_ct += 1
                            sig_q_cc_b_val = bayes_est['mean']
                            sig_q_cc_b.append(sig_q_cc_b_val)

                            if multi_impact:
                                ccq_vals.append(bayes_est['mean'])

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

        if multi_impact:
            if all(item >= 0 for item in ccq_vals) or all(item < 0 for item in ccq_vals):
                ct_single_sign += 1
                impact_dct.pop(k, None)
            else:
                ct_mixed += 1

        # if not impacted:
        #     impact_dct[k] = 'None'

    with open(out, 'w') as fp:
        json.dump(impact_dct, fp, indent=4)

    clim_q_set = set(sig_clim_q_gages)
    sig_q_cc_set = set(sig_q_cc_gages)
    sig_t_cc_set = set(sig_t_cc_gages)
    sig_t_q_set = set(sig_t_q_gages)
    q_ct_mo = {i: np.count_nonzero(q_months == i) for i in np.unique(q_months)}
    cc_ct_mo = {i: np.count_nonzero(cc_months == i) for i in np.unique(cc_months)}

    # sum([1 if x > 0.0 else 0 for x in sig_t_cc_b]) / float(len(sig_t_cc_b))
    plt.hist(sig_t_q_b)
    plt.savefig(os.path.join('/home/dgketchum/Downloads', os.path.basename(clim_q).replace('.json', '.png')))
    plt.close()
    print()


def mixed_impacts(_meta):
    with open(_meta, 'r') as qcc:
        impact_dct = json.load(qcc)

    pos_q, pos_cc = [], []
    neg_q, neg_cc = [], []

    for k, v in impact_dct.items():
        for kk, vv in v.items():
            s, e = kk.split('-')
            b = vv[0]
            ccmo = [x for x in range(int(s), int(e) + 1)]
            s, e = vv[1].split('-')
            qmo = [x for x in range(int(s), int(e) + 1)]

            if b > 0:
                [pos_q.append(x) for x in qmo]
                [pos_cc.append(x) for x in ccmo]

            if b < 0:
                [neg_q.append(x) for x in qmo]
                [neg_cc.append(x) for x in ccmo]

    pass


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
    ct_mixed, ct_single_sign = 0, 0
    for k, v in dct.items():
        multi_impact = False

        if len(v.keys()) > 1:
            ccq_vals = [v['cc_qres'] for k, v in v.items()]
            if all(item >= 0 for item in ccq_vals) or all(item < 0 for item in ccq_vals):
                ct_single_sign += 1
            else:
                ct_mixed += 1

        for kk, vv in v.items():

            s, e = vv['q_window'].split('-')
            q_mos = [x for x in range(int(s), int(e) + 1)]

            s, e = kk.split('-')
            cc_mos = [x for x in range(int(s), int(e) + 1)]

            ccqres, tcc, tq = vv['cc_qres'], vv['time_cc'], vv['time_qres']

            if ccqres < 0 < tcc and tq < 0:

                counts['ccqneg_ccpos_qneg'] += 1
                gages_sus.append(k)
                [counts['ccqneg_ccpos_qneg_qmo'].append(x) for x in q_mos]
                [counts['ccqneg_ccpos_qneg_ccmo'].append(x) for x in cc_mos]

            elif ccqres < 0 < tq and tcc < 0:
                counts['ccqneg_ccneg_qpos'] += 1
                gages_sus.append(k)
                [counts['ccqneg_ccneg_qpos_qmo'].append(x) for x in q_mos]
                [counts['ccqneg_ccneg_qpos_ccmo'].append(x) for x in cc_mos]

            elif ccqres > 0 > tq and 0 < tcc:
                counts['ccqpos_ccneg_qpos'] += 1
                gages_unsus.append(k)
                [counts['ccqpos_ccpos_qneg_qmo'].append(x) for x in q_mos]
                [counts['ccqpos_ccpos_qneg_ccmo'].append(x) for x in cc_mos]

            elif ccqres > 0 > tcc and 0 < tq:
                counts['ccqpos_ccneg_qpos'] += 1
                gages_unsus.append(k)
                [counts['ccqpos_ccneg_qpos_qmo'].append(x) for x in q_mos]
                [counts['ccqpos_ccneg_qpos_ccmo'].append(x) for x in cc_mos]

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
    rt = '/media/research/IrrigationGIS/gages/station_metadata/summerflow'
    # climr = os.path.join(rt, 'basin_climate_response_summerflow_7_10_20MAY2022.json')
    for m in range(7, 11):

        climr = os.path.join(rt, 'basin_climate_response_{}_7JUN2022.json'.format(m))
        bayes_ = os.path.join(rt, 'bayes_impacts_summerflow_{}_qnorm_{}_qreserr_0.17.json'.format(m, m))
        out_ = os.path.join(rt, 'impacts_summerflow_{}_out.json'.format(m))
        count_impacted_gages(climr, bayes_, out_)

    # count_coincident_trends_impacts(out_)

    # mixed_impacts(out_)
# ========================= EOF ====================================================================
