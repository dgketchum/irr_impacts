import os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geo_codes import state_fips_code, included_counties

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
E_STATES = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']
STATES = TARGET_STATES + E_STATES

FLOAT_COLS = ['VALUE_1987', 'VALUE_1992', 'VALUE_1997', 'VALUE_2002', 'VALUE_2007', 'VALUE_2012', 'VALUE_2017']

DROP = ['SOURCE_DESC', 'SECTOR_DESC', 'GROUP_DESC',
        'COMMODITY_DESC', 'CLASS_DESC', 'PRODN_PRACTICE_DESC',
        'UTIL_PRACTICE_DESC', 'STATISTICCAT_DESC', 'UNIT_DESC',
        'SHORT_DESC', 'DOMAIN_DESC', 'DOMAINCAT_DESC', 'STATE_FIPS_CODE',
        'ASD_CODE', 'ASD_DESC', 'COUNTY_ANSI',
        'REGION_DESC', 'ZIP_5', 'WATERSHED_CODE',
        'WATERSHED_DESC', 'CONGR_DISTRICT_CODE', 'COUNTRY_CODE',
        'COUNTRY_NAME', 'LOCATION_DESC', 'YEAR', 'FREQ_DESC',
        'BEGIN_CODE', 'END_CODE', 'REFERENCE_PERIOD_DESC',
        'WEEK_ENDING', 'LOAD_TIME', 'AGG_LEVEL_DESC',
        'CV_%', 'STATE_ALPHA', 'STATE_NAME', 'COUNTY_NAME']


def nass_statewide_summary(csv):
    df = pd.read_csv(csv)
    df = df.groupby(['STATE_ANSI'])[FLOAT_COLS].sum()
    df.index = [str(int(x)).rjust(2, '0') for x in df.index]
    state_codes = state_fips_code()
    state_inv = {v: k for k, v in state_codes.items()}
    states = [state_inv[i] for i in df.index]
    years = [int(x[-4:]) for x in FLOAT_COLS]
    fig, ax = plt.subplots()
    df = df.apply(lambda x: x.div(x.mean(), x.values), axis=1)
    diff = {state_inv[k]: v['VALUE_2017'] - v['VALUE_1987'] for k, v in df.iterrows() if state_inv[k] in TARGET_STATES}
    for i, r in df.iterrows():
        if state_inv[i] in TARGET_STATES:
            r.index = years
            r.name = state_inv[i]
            ax = r.plot(ax=ax, kind='line', x=years, y=r.values, alpha=0.6)

    plt.xlim(1984, 2020)
    plt.ylim(0.4, 1.6)
    plt.legend(loc='lower center', ncol=5, labelspacing=0.5)
    plt.show()

    pass


def nass_included_county_summary(csv):
    df = pd.read_csv(csv)
    df = df.groupby(['STATE_ANSI'])[FLOAT_COLS].sum()
    df.index = [str(int(x)).rjust(2, '0') for x in df.index]
    state_codes = state_fips_code()
    state_inv = {v: k for k, v in state_codes.items()}
    states = [state_inv[i] for i in df.index]
    years = [int(x[-4:]) for x in FLOAT_COLS]
    fig, ax = plt.subplots()
    df = df.apply(lambda x: x.div(x.mean(), x.values), axis=1)
    diff = {state_inv[k]: v['VALUE_2017'] - v['VALUE_2012'] for k, v in df.iterrows() if state_inv[k] in TARGET_STATES}
    for i, r in df.iterrows():
        if state_inv[i] in TARGET_STATES:
            r.index = years
            r.name = state_inv[i]
            ax = r.plot(ax=ax, kind='line', x=years, y=r.values, alpha=0.6)

    plt.xlim(1984, 2020)
    plt.ylim(0.4, 1.6)
    plt.legend(loc='lower center', ncol=5, labelspacing=0.5)
    plt.show()

    pass


def nass_included_county_summary_economic(csv):
    df = pd.read_csv(csv, index_col=0)
    idx = [str(i).rjust(5, '0') for i in df.index]
    df.index = idx
    df['VALUE'] = df['VALUE'].apply(lambda x: 0.0 if 'D' in x else int(x.replace(',', '')))
    df = df.loc[[i for i in idx if i in included_counties()], :]
    print(df['VALUE'].sum())
    pass


def get_nass_economic(csv, out_file):
    df = pd.read_table(csv, sep='\t')
    df.dropna(axis=0, subset=['COUNTY_CODE'], inplace=True, how='any')
    cty_str = df['COUNTY_CODE'].map(lambda x: str(int(x)).zfill(3))
    idx_str = df['STATE_FIPS_CODE'].map(lambda x: str(int(x))) + cty_str
    idx = idx_str.map(int)
    df.index = idx
    df['ST_CNTY_STR'] = df['STATE_ALPHA'] + '_' + df['COUNTY_NAME']

    prod_ag = deepcopy(df)
    prod_ag = prod_ag[(df['SOURCE_DESC'] == 'CENSUS') &
                      (df['SECTOR_DESC'] == 'ECONOMICS') &
                      (df['GROUP_DESC'] == 'INCOME') &
                      (df['COMMODITY_DESC'] == 'INCOME, FARM-RELATED') &
                      (df['CLASS_DESC'] == 'ALL CLASSES') &
                      (df['PRODN_PRACTICE_DESC'] == 'ALL PRODUCTION PRACTICES') &
                      (df['UTIL_PRACTICE_DESC'] == 'ALL UTILIZATION PRACTICES') &
                      (df['STATISTICCAT_DESC'] == 'RECEIPTS') &
                      (df['UNIT_DESC'] == '$') &
                      (df['DOMAIN_DESC'] == 'TOTAL')]
    prod_ag = prod_ag['VALUE'].apply(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))

    prod_irr = deepcopy(df)
    prod_irr = prod_irr[(df['SOURCE_DESC'] == 'CENSUS') &
                        (df['SECTOR_DESC'] == 'ECONOMICS') &
                        (df['GROUP_DESC'] == 'INCOME') &
                        (df['COMMODITY_DESC'] == 'INCOME, FARM-RELATED') &
                        (df['CLASS_DESC'] == 'ALL CLASSES') &
                        (df['PRODN_PRACTICE_DESC'] == 'IRRIGATED') &
                        (df['UTIL_PRACTICE_DESC'] == 'ALL UTILIZATION PRACTICES') &
                        (df['STATISTICCAT_DESC'] == 'RECEIPTS') &
                        (df['UNIT_DESC'] == '$') &
                        (df['DOMAIN_DESC'] == 'TOTAL')]
    prod_irr = prod_irr['VALUE'].apply(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))

    irr_area = deepcopy(df)
    irr_area = irr_area[(irr_area['SOURCE_DESC'] == 'CENSUS') &
                        (irr_area['SECTOR_DESC'] == 'ECONOMICS') &
                        (irr_area['GROUP_DESC'] == 'FARMS & LAND & ASSETS') &
                        (irr_area['COMMODITY_DESC'] == 'AG LAND') &
                        (irr_area['CLASS_DESC'] == 'AG LAND') &
                        (irr_area['PRODN_PRACTICE_DESC'] == 'IRRIGATED') &
                        (irr_area['UTIL_PRACTICE_DESC'] == 'ALL UTILIZATION PRACTICES') &
                        (irr_area['STATISTICCAT_DESC'] == 'AREA') &
                        (irr_area['UNIT_DESC'] == 'ACRES') &
                        (irr_area['SHORT_DESC'] == 'AG LAND, IRRIGATED - ACRES') &
                        (irr_area['DOMAIN_DESC'] == 'TOTAL')]

    irr_area = irr_area['VALUE'].apply(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))
    irr_area.drop(columns=DROP + ['VALUE'], inplace=True)

    ag_area = deepcopy(df)
    ag_area = ag_area[(ag_area['SOURCE_DESC'] == 'CENSUS') &
                      (ag_area['COMMODITY_DESC'] == 'AG LAND') &
                      (ag_area['STATISTICCAT_DESC'] == 'AREA') &
                      (ag_area['UNIT_DESC'] == 'ACRES') &
                      (df['SHORT_DESC'] == 'AG LAND, CROPLAND - ACRES') &
                      (ag_area['DOMAIN_DESC'] == 'TOTAL')]
    ag_area = ag_area['VALUE'].apply(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))
    ag_area.drop(columns=DROP + ['VALUE'], inplace=True)

    df.drop(columns=DROP, inplace=True)
    df.to_csv(out_file)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nass_csv = os.path.join(root, 'nass_data', 'qs.census2017.txt')
    prod_ = os.path.join(root, 'nass_data', 'economic_prod_2017.csv')
    get_nass_economic(nass_csv, out_file=prod_)
    # nass_included_county_summary_economic(prod_)
# ========================= EOF ====================================================================
