"""
Custom dataset processing/generation functions should be added to this file
"""

import pandas as pd
from src.paths import interim_data_path
import numpy as np
from sklearn import preprocessing
from ..logging import logger

__all__ = ['load_csv', 'process_csv', 'scale_distribution']


def load_datetime_csv(filename):
    with open(filename, 'r') as datafile:
        logger.debug(f"load_datetime_csv()-->loading datetime csv file={filename} ...")
        df = pd.read_csv(datafile, index_col='Date', parse_dates=True, date_parser=lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))
        return df


def load_csv(filename):
    """Read csv file

    filename: csv file to be read.
    """
    with open(filename, 'r') as datafile:
        logger.debug(f"load_csv()-->loading csv file={filename} ...")
        df = pd.read_csv(datafile, na_values=['?'],
                                  names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
        return df

def normalize(df):
    """normalize data in dataframe e.g. remove NaN rows.
    df: panda dataframe.
    """
    df.dropna(inplace=True)
    features = df[['age', 'shape', 'margin', 'density']].values
    target = df['severity'].values
    return features, target


def scale_distribution(features):
    """normalize standard distribution.
    features: list
        the list of features
    """
    scaler = preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


def process_csv(datasetname='mammographic', target_filename='mammographic_masses.data', metadata=None):
    """ process csv file
    datasetname: string (default: mammographic)
        dataset folder name
    target_filename: string (default: mammographic_masses.data)
        the name of the file to be processed
    metadata: dict
        Dict of metadata key/value pairs
    """

    unpack_dir = interim_data_path / datasetname
    df = load_csv(unpack_dir / target_filename)
    data, target = normalize(df)
    data_scaled = scale_distribution(data)

    dset_opts = {
        'dataset_name': datasetname,
        'data': data_scaled,
        'target': target,
        'metadata': metadata
    }
    return dset_opts
