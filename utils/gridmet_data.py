import warnings
from tempfile import mkdtemp
from datetime import datetime
from urllib.parse import urlunparse
import pandas as pd
from xarray import open_dataset
import numpy as np

warnings.simplefilter(action='ignore', category=DeprecationWarning)


class GridMet:
    """ U of I Gridmet

    Return as numpy array per met variable in daily stack unless modified.

    """

    def __init__(self, variable=None, date=None, start=None, end=None, bbox=None,
                 target_profile=None, clip_feature=None, lat=None, lon=None):

        self.date = date
        self.start = start
        self.end = end

        if isinstance(start, str):
            self.start = datetime.strptime(start, '%Y-%m-%d')
        if isinstance(end, str):
            self.end = datetime.strptime(end, '%Y-%m-%d')
        if isinstance(date, str):
            self.date = datetime.strptime(date, '%Y-%m-%d')

        if self.start and self.end is None:
            raise AttributeError('Must set both start and end date')

        self.bbox = bbox
        self.target_profile = target_profile
        self.clip_feature = clip_feature
        self.lat = lat
        self.lon = lon

        self.service = 'thredds.northwestknowledge.net:8080'
        self.scheme = 'http'

        self.temp_dir = mkdtemp()
        self.subset = None

        self.variable = variable
        self.available = ['elev', 'pr', 'rmax', 'rmin', 'sph', 'srad',
                          'th', 'tmmn', 'tmmx', 'pet', 'vs', 'erc', 'bi',
                          'fm100', 'pdsi']

        if self.variable not in self.available:
            Warning('Variable {} is not available'.
                    format(self.variable))

        mapping = {'etr': 'daily_mean_reference_evapotranspiration_alfalfa',
                   'pr': 'precipitation_amount'}

        self.long_name = mapping[self.variable]

        if self.date:
            self.start = self.date
            self.end = self.date

    def get_point_timeseries(self):

        url = self._build_url()
        url = url + '#fillmismatch'
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method='nearest')
        subset = subset.loc[dict(day=slice(self.start, self.end))]
        subset = subset.rename({'day': 'time'})
        date_ind = self._date_index()
        subset['time'] = date_ind
        time = subset['time'].values
        series = subset[self.long_name].values
        df = pd.DataFrame(data=series, index=time)
        df.columns = [self.variable]
        return df

    def _build_url(self):

        if self.variable == 'elev':
            url = urlunparse([self.scheme, self.service,
                              '/thredds/dodsC/MET/{0}/metdata_elevationdata.nc'.format(self.variable),
                              '', '', ''])
        else:
            url = urlunparse([self.scheme, self.service,
                              '/thredds/dodsC/agg_met_{}_1979_CurrentYear_CONUS.nc'.format(self.variable),
                              '', '', ''])

        return url

    def _date_index(self):
        date_ind = pd.date_range(self.start, self.end, freq='d')

        return date_ind
# ========================= EOF ====================================================================
