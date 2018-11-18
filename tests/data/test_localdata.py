import numpy as np
from unittest.mock import patch, mock_open, ANY
import pytest
import pandas as pd
from src.paths import project_dir, interim_data_path

from src.data.localdata import load_csv, process_csv, scale_distribution, normalize


def test_load_csv_file_success(mock_df):
    test_filename = project_dir / 'tests/data/test_data.csv'
    actual = load_csv(test_filename)
    assert actual.columns[actual.isna().any()].tolist() == ['margin', 'density']
    pd.testing.assert_frame_equal(actual, mock_df)


def test_load_csv_file_fail():
    test_filename = 'date_not_found.csv'
    with pytest.raises(FileNotFoundError, message="Expecting FileNotFoundError"):
        load_csv(test_filename)


def test_normalize_data_success(mock_df):
    expected = np.array([28, 1, 1, 3])
    actual_features, actual_target = normalize(mock_df)
    assert actual_features.shape == (1, 4)
    assert actual_target.shape == (1,)
    assert np.all(actual_features == expected)


@patch('src.data.localdata.preprocessing.StandardScaler.fit_transform')
@patch('src.data.localdata.pd.read_csv')
@patch('src.data.localdata.open', create=True)
def test_process_csv_with_default_value_success(open_mock, mock_read_csv, mock_fit_transform, mock_df):
    test_filename = interim_data_path / 'mammographic/mammographic_masses.data'
    mock_read_csv.return_value = mock_df
    expected_scaled_features = np.array([2, 1, 2, 1])
    mock_fit_transform.return_value = expected_scaled_features

    actual = process_csv()

    open_mock.assert_called_once_with(test_filename, 'r')
    mock_fit_transform.assert_called()
    mock_read_csv.assert_called_once_with(ANY, na_values=['?'],
                                          names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
    assert actual.get('dataset_name') == 'mammographic'
    assert np.array_equal(actual.get('data'), expected_scaled_features)
    assert actual.get('metadata') is None
    assert actual.get('target') == np.array([0])


@patch('src.data.localdata.preprocessing.StandardScaler.fit_transform')
@patch('src.data.localdata.pd.read_csv')
@patch('src.data.localdata.open', create=True)
def test_process_csv_with_parameter_values_success(open_mock, mock_read_csv, mock_fit_transform, mock_df):
    expected_datasetname = 'test_dataset'
    expected_filename = 'test_file.csv'
    test_filename = interim_data_path / expected_datasetname / expected_filename
    expected_metadata = {'DESCR': 'descr', 'TEST': 'testing'}
    expected_parameters = {'datasetname': expected_datasetname, 'target_filename': test_filename,
                           'metadata': expected_metadata}
    mock_read_csv.return_value = mock_df
    expected_scaled_features = np.array([2, 1, 2, 1])

    mock_fit_transform.return_value = expected_scaled_features
    actual = process_csv(**expected_parameters)

    open_mock.assert_called_once_with(test_filename, 'r')
    mock_fit_transform.assert_called()
    mock_read_csv.assert_called_once_with(ANY, na_values=['?'],
                                          names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
    assert actual.get('dataset_name') == expected_datasetname
    assert np.array_equal(actual.get('data'), expected_scaled_features)
    assert actual.get('metadata') == expected_metadata
    assert actual.get('target') == np.array([0])


@pytest.fixture()
def mock_df():
    d = {'BI-RADS': [4, 5, 4], 'age': [28, 74, 65], 'shape': [1, 1, 1], 'margin': [1.0, 5.0, np.nan],
         'density': [3.0, np.nan, 3.0], 'severity': [0, 1, 0]}
    return pd.DataFrame(data=d)


def test_scale():
    expected_data = np.random.rand(5, 3)
    print(scale_distribution(expected_data))
