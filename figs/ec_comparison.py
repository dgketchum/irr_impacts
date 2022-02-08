import os

import numpy as np
from pandas import read_csv, isna, to_datetime
from matplotlib import pyplot as plt
from scipy.stats import linregress


def comparison(csv):
    df = read_csv(csv)
    df['date'] = to_datetime(df['date'])
    et_comp = [(r['ET'], r['et_ssebop']) for i, r in df.iterrows() if not isna(r['ET']) and not isna(r['et_ssebop'])]
    print(len(et_comp), ' records')
    scatter = '/home/dgketchum/Downloads/et_comp.png'
    et_ec = np.array([x[0] for x in et_comp])
    et_model = np.array([x[1] for x in et_comp])
    rmse = np.sqrt(np.mean((et_model - et_ec) ** 2))
    lin = np.arange(0, 300)
    plt.plot(lin, lin, '--', color='k', linewidth=0.2)
    # m, b = np.polyfit(et_ec, et_model, 1)
    m, b, r, p, stderr = linregress(et_ec, et_model)
    print('{:.2f}x + {:.2f}\n rmse: {:.2f} mm/month\n r2: {:.2f}\n\n\n'.format(m, b, rmse, r ** 2))
    plt.annotate('{:.2f}x + {:.2f}\n rmse: {:.2f} mm/month\n r2: {:.2f}'.format(m, b, rmse, r ** 2),
                 xy=(0.05, 0.75), xycoords='axes fraction')
    plt.plot(lin, m * lin + b)
    plt.scatter(et_ec, et_model)
    plt.ylabel('SSEBop ET')
    plt.xlabel('Eddy Covariance')
    plt.savefig(scatter)


if __name__ == '__main__':
    fdir = '/media/research/IrrigationGIS/ameriflux/ec_data'
    _csv = os.path.join(fdir, 'ec_ssebop_comp.csv')
    comparison(_csv)
# ========================= EOF ====================================================================
