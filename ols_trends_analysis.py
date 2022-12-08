import json
import os
from pprint import pprint

import numpy as np
import pymannkendall as mk
from matplotlib import pyplot as plt
from scipy.stats import linregress, anderson


def initial_trends_test(in_json, out_json, plot_dir=None, selectors=None):
    with open(in_json, 'r') as f:
        stations = json.load(f)

    regressions, counts, ct = {}, None, 0
    all_counts, sig_counts = None, None

    for enu, (station, records) in enumerate(stations.items(), start=1):

        try:
            q = np.array(records['q'])
        except KeyError:
            continue

        month = records['q_mo']
        qres = np.array(records['qres'])
        ai = np.array(records['ai'])

        cc = np.array(records['cc_month'])
        try:
            ccres = np.array(records['ccres_month'])
        except KeyError:
            ccres = cc * np.zeros_like(cc)
        aim = np.array(records['ai_month'])
        etr_m = np.array(records['etr_month'])
        etr = np.array(records['etr'])
        ppt_m = np.array(records['ppt_month'])
        ppt = np.array(records['ppt'])
        irr = np.array(records['irr'])
        cci = cc / irr

        years = np.array(records['years'])

        regression_combs = [(years, cc, 'time_cc'),
                            (years, ccres, 'time_ccres'),
                            (years, qres, 'time_qres'),
                            (years, ai, 'time_ai'),
                            (years, aim, 'time_aim'),
                            (years, q, 'time_q'),
                            (years, etr, 'time_etr'),
                            (years, ppt, 'time_ppt'),
                            (years, etr_m, 'time_etrm'),
                            (years, ppt_m, 'time_pptm'),
                            (years, irr, 'time_irr'),
                            (years, cci, 'time_cci')]

        if not sig_counts:
            sig_counts = {k[2]: [0, 0] for k in regression_combs}
            sig_counts.update({'q_cwd': [0, 0]})
            all_counts = {k[2]: 0 for k in regression_combs}
            all_counts.update({'q_cwd': 0})

        all_counts['q_cwd'] += 1
        if records['p'] < 0.05:
            if records['b'] < 0.0:
                sig_counts['q_cwd'][0] += 1
            else:
                sig_counts['q_cwd'][1] += 1

        regressions[station] = records

        for x, y, subdir in regression_combs:

            if selectors and subdir not in selectors:
                continue

            if month not in range(4, 11) and subdir in ['time_irr', 'time_cc', 'time_ccres', 'time_cci']:
                continue

            all_counts[subdir] += 1

            if subdir == 'time_q':
                mk_test = mk.hamed_rao_modification_test(y)
                y_pred = x * mk_test.slope + mk_test.intercept
                mk_slope_std = mk_test.slope * np.std(x) / np.std(y)
                p = mk_test.p
                if p < 0.05:
                    if mk_slope_std > 0:
                        sig_counts[subdir][1] += 1
                    else:
                        sig_counts[subdir][0] += 1
                    regressions[station][subdir] = {'test': 'mk',
                                                    'b': mk_slope_std,
                                                    'p': p}

            else:
                lr = linregress(x, y)
                b, inter, r, p = lr.slope, lr.intercept.item(), lr.rvalue, lr.pvalue
                y_pred = x * b + inter
                b_norm = b * np.std(x) / np.std(y)
                resid = y - (b * x + inter)
                ad_test = anderson(resid, 'norm').statistic.item()
                if ad_test < 0.05:
                    print('{} month {} failed normality test'.format(station, month))
                if p < 0.05:
                    if b_norm > 0:
                        sig_counts[subdir][1] += 1
                    else:
                        sig_counts[subdir][0] += 1
                    regressions[station][subdir] = {'test': 'ols',
                                                    'b': b,
                                                    'inter': inter,
                                                    'p': p,
                                                    'rsq': r,
                                                    'b_norm': b_norm,
                                                    'anderson': ad_test}

            if plot_dir:
                d = os.path.join(plot_dir, str(month), subdir)

                if not os.path.exists(d):
                    os.makedirs(d)

                yvar = subdir.split('_')[1]
                plt.scatter(x, y)
                if yvar == 'q':
                    lr = linregress(x, y)
                    b, inter, r, p = lr.slope, lr.intercept.item(), lr.rvalue, lr.pvalue
                    y_pred = x * b + inter

                plt.plot(x, y_pred)
                plt.xlabel('time')
                plt.ylabel(yvar)
                desc = '{} {} {} {} yrs {}'.format(month, station, yvar, len(x), records['STANAME'])
                plt.suptitle(desc)
                f = os.path.join(d, '{}.png'.format(desc))
                plt.savefig(f)
                plt.close()

    print('\n {}'.format(in_json))
    pprint(sig_counts)
    print(sum([np.array(v).sum() for k, v in sig_counts.items()]))

    with open(out_json, 'w') as f:
        json.dump(regressions, f, indent=4, sort_keys=False)

    return sig_counts


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
