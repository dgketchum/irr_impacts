import os
import json
from pprint import pprint

from pandas import read_csv, concat, errors, DataFrame

from gage_analysis import EXCLUDE_STATIONS
from hydrograph import hydrograph

DROP = ['system:index', '.geo']
ATTRS = ['SQMI', 'STANAME', 'start', 'end']

CLMB_STATIONS = ['12302055',
                 '12323600',
                 '12323770',
                 '12324200',
                 '12324590',
                 '12324680',
                 '12325500',
                 '12329500',
                 '12330000',
                 '12331500',
                 '12332000',
                 '12334510',
                 '12334550',
                 '12335500',
                 '12340000',
                 '12340500',
                 '12342500',
                 '12344000',
                 '12350250',
                 '12352500',
                 '12353000',
                 '12354000',
                 '12354500',
                 '12358500',
                 '12359800',
                 '12362500',
                 '12363000',
                 '12366000',
                 '12370000',
                 '12371550',
                 '12372000',
                 '12374250',
                 '12375900',
                 '12377150',
                 '12388700',
                 '12389000',
                 '12389500',
                 '12390700',
                 '12392155',
                 '12392300',
                 '12395000',
                 '12395500',
                 '12396500',
                 '12398600',
                 '12409000',
                 '12411000',
                 '12413000',
                 '12413210',
                 '12413470',
                 '12413500',
                 '12414500',
                 '12414900',
                 '12415500',
                 '12419000',
                 '12422000',
                 '12422500',
                 '12424000',
                 '12426000',
                 '12433000',
                 '12433200',
                 '12445900',
                 '12447390',
                 '12448500',
                 '12448998',
                 '12449500',
                 '12449950',
                 '12451000',
                 '12452000',
                 '12452500',
                 '12452800',
                 '12456500',
                 '12457000',
                 '12458000',
                 '12459000',
                 '12462500',
                 '12464800',
                 '12465000',
                 '12465400',
                 '12467000',
                 '12472600',
                 '12484500',
                 '12488500',
                 '12500450',
                 '12502500',
                 '12506000',
                 '12508990',
                 '12510500',
                 '12513000',
                 '13010065',
                 '13011000',
                 '13011500',
                 '13011900',
                 '13014500',
                 '13015000',
                 '13018300',
                 '13018350',
                 '13018750',
                 '13022500',
                 '13023000',
                 '13025500',
                 '13027500',
                 '13032500',
                 '13037500',
                 '13038500',
                 '13039000',
                 '13039500',
                 '13042500',
                 '13046000',
                 '13047500',
                 '13049500',
                 '13050500',
                 '13052200',
                 '13055000',
                 '13055340',
                 '13056500',
                 '13057000',
                 '13057500',
                 '13057940',
                 '13058000',
                 '13062500',
                 '13063000',
                 '13066000',
                 '13069500',
                 '13073000',
                 '13075000',
                 '13075500',
                 '13075910',
                 '13077000',
                 '13078000',
                 '13081500',
                 '13082500',
                 '13083000',
                 '13090000',
                 '13090500',
                 '13094000',
                 '13105000',
                 '13106500',
                 '13108150',
                 '13112000',
                 '13115000',
                 '13116500',
                 '13119000',
                 '13120000',
                 '13120500',
                 '13126000',
                 '13127000',
                 '13132500',
                 '13135000',
                 '13135500',
                 '13137000',
                 '13137500',
                 '13141500',
                 '13142000',
                 '13142500',
                 '13147900',
                 '13148200',
                 '13148500',
                 '13152500',
                 '13153500',
                 '13159800',
                 '13161500',
                 '13168500',
                 '13171500',
                 '13171620',
                 '13172500',
                 '13174000',
                 '13174500',
                 '13181000',
                 '13183000',
                 '13185000',
                 '13186000',
                 '13190500',
                 '13200000',
                 '13201500',
                 '13206000',
                 '13213000',
                 '13213100',
                 '13233300',
                 '13235000',
                 '13236500',
                 '13238500',
                 '13239000',
                 '13240000',
                 '13245000',
                 '13246000',
                 '13247500',
                 '13249500',
                 '13250000',
                 '13251000',
                 '13258500',
                 '13265500',
                 '13266000',
                 '13269000',
                 '13295000',
                 '13296000',
                 '13296500',
                 '13297330',
                 '13297350',
                 '13297355',
                 '13302500',
                 '13305000',
                 '13307000',
                 '13309220',
                 '13310700',
                 '13311000',
                 '13313000',
                 '13316500',
                 '13317000',
                 '13331500',
                 '13333000',
                 '13336500',
                 '13337000',
                 '13337500',
                 '13338500',
                 '13339500',
                 '13340000',
                 '13340600',
                 '13340950',
                 '13341050',
                 '13342450',
                 '13342500',
                 '13344500',
                 '13345000',
                 '13346800',
                 '13348000',
                 '13351000',
                 '14013000',
                 '14015000',
                 '14018500',
                 '14020000',
                 '14020300',
                 '14033500',
                 '14034470',
                 '14034480',
                 '14034500',
                 '14038530',
                 '14044000',
                 '14046000',
                 '14046500',
                 '14048000',
                 '14076500',
                 '14087400',
                 '14091500',
                 '14092500',
                 '14092750',
                 '14093000',
                 '14096850',
                 '14097100',
                 '14101500',
                 '14103000',
                 '14107000',
                 '14113000',
                 '14113200',
                 '14120000',
                 '14123500',
                 '14137000',
                 '14138800',
                 '14138850',
                 '14138870',
                 '14138900',
                 '14139800',
                 '14140000',
                 '14141500',
                 '14142500',
                 '14144800',
                 '14144900',
                 '14145500',
                 '14147500',
                 '14148000',
                 '14149000',
                 '14150000',
                 '14150800',
                 '14150900',
                 '14151000',
                 '14152000',
                 '14152500',
                 '14153000',
                 '14153500',
                 '14154500',
                 '14155000',
                 '14155500',
                 '14157500',
                 '14158500',
                 '14158790',
                 '14158850',
                 '14159110',
                 '14159200',
                 '14159500',
                 '14161100',
                 '14161500',
                 '14162100',
                 '14162200',
                 '14162500',
                 '14163000',
                 '14163150',
                 '14163900',
                 '14165000',
                 '14165500',
                 '14166000',
                 '14166500',
                 '14168000',
                 '14169000',
                 '14170000',
                 '14171000',
                 '14174000',
                 '14178000',
                 '14179000',
                 '14180500',
                 '14181500',
                 '14182500',
                 '14183000',
                 '14184100',
                 '14185000',
                 '14185800',
                 '14185900',
                 '14186100',
                 '14186600',
                 '14187000',
                 '14187200',
                 '14187500',
                 '14188800',
                 '14189000',
                 '14190500',
                 '14191000',
                 '14200000',
                 '14200300',
                 '14201500',
                 '14202000',
                 '14202980',
                 '14203500',
                 '14206900',
                 '14207500',
                 '14207740',
                 '14207770',
                 '14208700',
                 '14209000',
                 '14209500',
                 '14210000',
                 '14211500',
                 '14211550',
                 '14211720',
                 '14216000',
                 '14216500',
                 '14219000',
                 '14219800',
                 '14220500',
                 '14222500',
                 '14226500',
                 '14231000',
                 '14233500',
                 '14234800',
                 '14236200',
                 '14238000',
                 '14242580',
                 '14243000']


def compile_terraclime(in_dir, out_dir):
    for m in range(1, 13):
        m_str = str(m).rjust(2, '0')
        l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if '{}.csv'.format(m_str) in x]
        first = True
        for csv in l:
            y = int(os.path.basename(csv).split('_')[1])
            try:
                if first:
                    df = read_csv(csv, index_col='STAID')
                    df.columns = ['{}_{}'.format(col, y) for col in list(df.columns)]
                    first = False
                else:
                    c = read_csv(csv, index_col='STAID')
                    c.columns = ['{}_{}'.format(col, y) for col in list(c.columns)]
                    df = concat([df, c], axis=1)
                    print(c.shape, csv)
            except errors.EmptyDataError:
                print('{} is empty'.format(csv))
                pass

        out_file = os.path.join(out_dir, '{}.csv'.format(m))
        df.to_csv(out_file)


def concatenate_extracts(root, out_csv, glob='None'):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
    l.sort()
    first = True
    for csv in l:
        try:
            if first:
                df = read_csv(csv, index_col='STAID').drop(columns=DROP)
                print(df.shape, csv)
                first = False
            else:
                c = read_csv(csv, index_col='STAID').drop(columns=DROP + ATTRS)
                df = concat([df, c], axis=1)
                print(c.shape, csv)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df.to_csv(out_csv)


def write_station_metadata(csv, out_json):
    df = read_csv(csv)
    df['STAID_STR'] = [str(x).rjust(8, '0') for x in df['STAID'].values]
    dfd = df.to_dict(orient='records')
    meta_d = {}
    for d in dfd:
        meta_d[d['STAID_STR']] = {}
        for attr in ['STANAME', 'start', 'end']:
            meta_d[d['STAID_STR']][attr] = d[attr]
        meta_d[d['STAID_STR']]['AREA_SQKM'] = d['SQMI'] * 2.59
    with open(out_json, 'w') as f:
        json.dump(meta_d, f)


def compare_gridded_versions(csv_dir_1, csv_dir_2):
    # TODO: why is irrigation so much higher from Comp?
    l_1 = [os.path.join(csv_dir_1, x) for x in os.listdir(csv_dir_1)]
    l_2 = [os.path.join(csv_dir_2, x) for x in os.listdir(csv_dir_2)]
    d = {}
    first = True
    for v2, comp in zip(l_1, l_2):
        try:
            sid = os.path.basename(v2).split('.')[0]
            df1 = hydrograph(v2)
            cols = list(df1.columns)
            new_cols = ['{}_1'.format(x) for x in cols]
            df1.columns = new_cols

            if first:
                for c in new_cols:
                    d[c] = df1[c].sum()
            else:
                for c in new_cols:
                    d[c] += df1[c].sum()

            df2 = hydrograph(comp)
            cols = list(df2.columns)
            new_cols = ['{}_2'.format(x) for x in cols]
            df2.columns = new_cols

            if first:
                for c in new_cols:
                    d[c] = df2[c].sum()
            else:
                for c in new_cols:
                    d[c] += df2[c].sum()

            first = False
            df = concat([df1, df2])

        except Exception as e:
            print(sid, e)

    pprint(d)


def merge_hydrograph_gridded(csv, hydrograph_src, out_dir, metadata, per_area=False):
    df = read_csv(csv)
    df['STAID_STR'] = [str(x).rjust(8, '0') for x in df['STAID'].values]
    dfd = df.to_dict(orient='records')
    ct, skip = 0, 0

    with open(metadata, 'r') as f:
        meta = json.load(f)

    for d in dfd:
        if d['STAID_STR'] not in CLMB_STATIONS:
            # continue to exclude non-clmb stations
            pass
        if d['STAID_STR'] in EXCLUDE_STATIONS:
            continue
        # if d['STAID_STR'] != '12484500':
        #     continue

        q_file = os.path.join(hydrograph_src, '{}.csv'.format(d['STAID_STR']))

        try:
            h = hydrograph(q_file)
            if h.shape[0] < 30:
                continue
        except FileNotFoundError:
            skip += 1
            continue

        years = [x for x in range(1991, 2021)]
        try:
            irr = [d['irr_{}'.format(y)] for y in years], 'irr'
            cc = [d['cc_{}'.format(y)] for y in years], 'cc'
            ppt = [d['ppt_{}'.format(y)] for y in years], 'ppt'
            ppt_lt = [d['ppt_lt_{}'.format(y)] for y in years], 'ppt_lt'
            pet = [d['etr_{}'.format(y)] for y in years], 'etr'
            pet_lt = [d['etr_lt_{}'.format(y)] for y in years], 'etr_lt'
            recs = DataFrame(dict([(x[1], x[0]) for x in [cc, irr, ppt, pet, ppt_lt, pet_lt]]), index=h.index)

        except KeyError:
            ranges = [x for x in range(30)]
            cc = ['cc.{}'.format(y) for y in ranges]
            cc[0] = 'cc'
            cc = [d[c] for c in cc], 'cc'

            irr = [d['irr_{}'.format(y)] for y in years], 'irr'

            pr = ['pr.{}'.format(y) for y in ranges]
            pr[0] = 'pr'
            ppt = [d[c] for c in pr], 'pr'

            etr = ['etr.{}'.format(y) for y in ranges]
            etr[0] = 'etr'
            pet = [d[c] for c in etr], 'etr'
            recs = DataFrame(dict([(x[1], x[0]) for x in [cc, irr, ppt, pet]]), index=h.index)

        try:
            if per_area:
                # basin-averaged mm, irr fraction
                irr = recs['irr'] / (meta[d['STAID_STR']]['AREA_SQKM'] * 1e6)
                recs = recs / (meta[d['STAID_STR']]['AREA_SQKM'] * 1e3)
                recs['irr'] = irr
                h['q'] = h['q'] / (meta[d['STAID_STR']]['AREA_SQKM'] * 1e3)
                h['qb'] = h['qb'] / (meta[d['STAID_STR']]['AREA_SQKM'] * 1e3)

        except ValueError as e:
            print(d['STAID_STR'], e)
            skip += 1
            continue
        except KeyError as e:
            print(d['STAID_STR'], e)
            skip += 1
            continue

        h = concat([h, recs], axis=1)

        try:
            h['intensity'] = h['cc'] / h['irr']
            h['aridity'] = h['etr'] / h['ppt']
        except KeyError:
            h['intensity'] = h['cc'] / h['irr']
            h['aridity'] = h['etr'] / h['pr']

        print('{:.3f} {}'.format(h['irr'].mean(), d['STANAME']))
        out_file = os.path.join(out_dir, '{}.csv'.format(d['STAID_STR']))
        h.to_csv(out_file, encoding='utf-8')
        ct += 1
    print('wrote {} csv, skipped {}'.format(ct, skip))


def write_gridded_data_jsn(csv_dir, jsn, jsn_out):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]

    with open(jsn, 'r') as f:
        meta = json.load(f)

    meta_grid = {}
    for c in l:
        df = hydrograph(c)
        try:
            df['aridity'] = df['etr'] / df['ppt']
        except KeyError:
            df['aridity'] = df['etr'] / df['pr']
        sid = os.path.basename(c).split('.')[0]
        print(sid)
        meta_grid[sid] = meta[sid]
        cols = list(df.columns)
        e, l = df.loc[:'2005-12-31', :].mean(axis=0), df.loc['2005-12-31':, :].mean(axis=0)
        for s, v in zip([e, l], ['early', 'late']):
            d = s.to_dict()
            for param in cols:
                meta_grid[sid]['{}_{}'.format(param, v)] = d[param]
    with open(jsn_out, 'w') as f:
        json.dump(meta_grid, f)


if __name__ == '__main__':
    # TODO find why irrigated fractions are so high in e.g. 6099000
    r = '/media/research/IrrigationGIS/gages/ee_exports/water_year'
    extracts = '/media/research/IrrigationGIS/gages/ee_exports/series/extracts_wy_Comp_4AUG2021.csv'
    g = 'basins_wy_Comp_4AUG2021'
    # concatenate_extracts(r, extracts, g)
    gage_src = '/media/research/IrrigationGIS/gages/hydrographs/q_bf_JAS'
    dst = '/media/research/IrrigationGIS/gages/merged_q_ee/JAS_Comp_4AUG2021'
    jsn_i = '/media/research/IrrigationGIS/gages/station_metadata/metadata_flows.json'
    # jsn_o = '/media/research/IrrigationGIS/gages/station_metadata/metadata_flows_gridded_JAS_4AUG2021.json'
    # write_station_metadata(extracts, jsn_i)
    # write_gridded_data_jsn(dst, jsn_i, jsn_o)
    # merge_hydrograph_gridded(extracts, gage_src, dst, jsn_i, per_area=True)
    tc = '/media/research/IrrigationGIS/gages/ee_exports/terraclim/raw_export'
    tc_concat = '/media/research/IrrigationGIS/gages/ee_exports/terraclim/monthly'
    compile_terraclime(tc, tc_concat)
# ========================= EOF ====================================================================
