import os
from calendar import monthrange
from datetime import date

from matplotlib import rcParams, pyplot as plt
from pandas import read_csv
import numpy as np

from gage_data import hydrograph
from utils.geo_codes import state_fips_code, state_county_code, included_counties


def plot_clim_q_resid(q, ai, clim_line, desc_str, years, cc, resid, resid_line, fig_d, cci_per, flow_per):
    rcParams['figure.figsize'] = 16, 10
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(ai, q)
    ax1.plot(ai, clim_line)
    ax1.set(xlabel='ETr / PPT [-]')
    ax1.set(ylabel='q [m^3]')

    for i, y in enumerate(years):
        ax1.annotate(y, (ai[i], q[i]))
        plt.suptitle(desc_str)

    ax2.set(xlabel='cc [m]')
    ax2.set(ylabel='q epsilon [m^3]')
    ax2.scatter(cc, resid)
    ax2.plot(cc, resid_line)
    for i, y in enumerate(years):
        ax2.annotate(y, (cc[i], resid[i]))

    desc_split = desc_str.strip().split('\n')
    file_name = desc_split[0].replace(' ', '_')

    fig_name = os.path.join(fig_d, '{}_cc_{}-{}_q_{}-{}.png'.format(file_name, cci_per[0], cci_per[1],
                                                                    flow_per[0], flow_per[1]))

    plt.savefig(fig_name)
    plt.close('all')


def plot_water_balance_trends(data, data_line, data_str, years, desc_str, fig_d):
    rcParams['figure.figsize'] = 16, 10
    fig, ax1 = plt.subplots(1, 1)

    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.scatter(years, data, color=color)
    ax1.plot(years, data_line, color=color)
    ax1.set_ylabel(data_str, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    desc_split = desc_str.strip().split('\n')
    file_name = desc_split[0].replace(' ', '_')

    fig_name = os.path.join(fig_d, '{}_{}.png'.format(file_name, data_str))

    plt.savefig(fig_name)
    plt.close('all')


def nass_irrmapper_climate(irr_dir, nass_c, fig_dir, countywise=True):

    study_counties = included_counties()

    ndf = read_csv(nass_c)
    ndf.dropna(how='any', axis=0, subset=['FIPS'], inplace=True)
    ndf.dropna(how='any', axis=0, subset=['ST_CNTY_STR'], inplace=True)
    ndf.fillna(0.0, inplace=True)
    ndf['GEOID'] = [str(int(x)).rjust(5, '0') for x in ndf['FIPS']]
    ndf.index = ndf['GEOID']

    m_start, m_end = 10, 9
    years = np.arange(1997, 2017)
    clim_dates = [(date(y, 4, 1),  date(y, 9, monthrange(y, m_end)[1])) for y in years]
    cc_dates = [(date(y, 5, 1), date(y, 10, 31)) for y in years]
    irr_dates = [(date(y, 7, 1), date(y, 7, 31)) for y in years]

    l = [os.path.join(irr_dir, x) for x in os.listdir(irr_dir)]
    if countywise:
        for c in l:
            co = os.path.basename(c).split('.')[0]
            idf = hydrograph(c)
            # idf['cci'] = idf['cc'] / idf['irr']

            try:
                co_desc = ndf.loc[co]['ST_CNTY_STR'].split('_')
            except (KeyError, AttributeError):
                print('\n{} not found\n'.format(co))

            co_str, st_str = co_desc[1].title(), co_desc[0]
            ppt = np.array([idf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
            etr = np.array([idf['etr'][d[0]: d[1]].sum() for d in clim_dates])
            ai = etr / ppt
            cc = np.array([idf['cc'][d[0]: d[1]].sum() for d in cc_dates])
            cc[cc == 0.0] = np.nan
            irr = np.array([idf['irr'][d[0]: d[1]].sum() for d in irr_dates]) / 4046.86
            if np.any(irr[5:] < 1000):
                continue
            nrow = [(k[-4:], v) for k, v in ndf.loc[co].items() if 'VALUE' in k]
            n_v, n_y = [x[1] for x in nrow], [int(x[0]) for x in nrow]
            fig, ax = plt.subplots(4, 1)
            fig.set_figheight(8)
            fig.set_figwidth(12)
            fig.tight_layout()

            ax[0].plot(years, irr, color='purple', label='IrrMapper')
            ax[0].plot(n_y, n_v, color='pink', label='NASS')
            ax[0].legend()
            ax[0].set(ylabel='Acres Irrigated')
            ax[0].set_xlim(years[0], years[-1])
            ax[1].plot(years, ppt, color='blue')
            ax[1].set(ylabel='AMJJA Precipitation [m^3]')
            ax[1].set_xlim(years[0], years[-1])
            ax[2].plot(years, cc, color='black')
            ax[2].set(ylabel='Crop Consumption [m^3]')
            ax[2].set_xlim(years[0], years[-1])
            ax[3].plot(years, ai, color='red')
            ax[3].set(ylabel='AMJJA Aridity Index [-]')
            ax[3].set_xlim(years[0], years[-1])

            plt.suptitle('{} Co. {}'.format(co_str, st_str))
            plt.xlim(1985, 2021)
            plt.gcf().subplots_adjust(left=0.1)
            plt.tight_layout()

            fig_file = '{}_{}.png'.format(st_str, co_str)

            if co in study_counties:
                sub_dir = 'impacts_study'
            else:
                sub_dir = 'non_study'

            plt.savefig(os.path.join(fig_dir, sub_dir, fig_file))
            plt.close()
            print(fig_file, sub_dir)
    else:
        western_11 = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
        inlcude_all = ['ALL'] + western_11
        fips = state_fips_code()
        western_fips = [v for k, v in fips.items() if k in western_11]

        for s in inlcude_all:

            if s == 'ALL':
                nass_rows = [i for i, r in ndf.iterrows() if r['ST_CNTY_STR'][:2] in western_11]
            else:
                nass_rows = [i for i, r in ndf.iterrows() if r['ST_CNTY_STR'].startswith(s)]

            sndf = ndf.loc[nass_rows]
            sndf = sndf[[x for x in sndf.columns if 'VALUE' in x]]
            n_v = sndf.sum(axis=0).values
            n_y = [int(x[-4:]) for x in sndf.columns]

            if s == 'ALL':
                csv_l = [x for x in l if os.path.basename(x).split('.')[0][:2] in western_fips]
            else:
                csv_l = [x for x in l if os.path.basename(x).split('.')[0].startswith(state_fips_code()[s])]

            if not s == 'ALL':
                if not len(state_county_code()[s].keys()) == len(csv_l):
                    csv_l = [x for x in csv_l if os.path.basename(x).split('.')[0] in nass_rows]
                    print('{} is short records from EE'.format(s))
                if not len(state_county_code()[s].keys()) == sndf.shape[0]:
                    csv_l = [x for x in csv_l if os.path.basename(x).split('.')[0] in nass_rows]
                    print('{} is short records from NASS'.format(s))

            first = True
            for c in csv_l:
                if first:
                    idf = hydrograph(c)
                    first = False
                    continue
                idf += hydrograph(c)

            ppt = np.array([idf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
            etr = np.array([idf['etr'][d[0]: d[1]].sum() for d in clim_dates])
            ai = etr / ppt
            cc = np.array([idf['cc'][d[0]: d[1]].sum() for d in cc_dates])
            cc[cc == 0.0] = np.nan
            irr = np.array([idf['irr'][d[0]: d[1]].sum() for d in irr_dates]) / 4046.86

            fig, ax = plt.subplots(4, 1)
            fig.set_figheight(8)
            fig.set_figwidth(12)
            fig.tight_layout()

            ax[0].plot(years, irr, color='purple', label='IrrMapper')
            ax[0].plot(n_y, n_v, color='pink', label='NASS')
            ax[0].legend()
            ax[0].set(ylabel='Acres Irrigated')
            ax[0].set_xlim(years[0], years[-1])
            ax[1].plot(years, ppt, color='blue')
            ax[1].set(ylabel='AMJJA Precipitation [m^3]')
            ax[1].set_xlim(years[0], years[-1])
            ax[2].plot(years, cc, color='black')
            ax[2].set(ylabel='Crop Consumption [m^3]')
            ax[2].set_xlim(years[0], years[-1])
            ax[3].plot(years, ai, color='red')
            ax[3].set(ylabel='AMJJA Aridity Index [-]')
            ax[3].set_xlim(years[0], years[-1])

            plt.suptitle('{}'.format(s))
            plt.xlim(years[0], years[-1])
            plt.gcf().subplots_adjust(left=0.1)
            plt.tight_layout()

            fig_file = '{}.png'.format(s)

            sub_dir = 'statewise'

            plt.savefig(os.path.join(fig_dir, sub_dir, fig_file))
            plt.close()
            print(fig_file, sub_dir)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    nass = os.path.join(root, 'nass_data', 'nass_merged.csv')
    co_irr = os.path.join(root, 'time_series/counties_IrrMapperComp_21DEC2021/county_monthly')
    figs = os.path.join(root, 'time_series/counties_IrrMapperComp_21DEC2021/figures')
    nass_irrmapper_climate(co_irr, nass, figs, countywise=False)
# ========================= EOF ====================================================================
