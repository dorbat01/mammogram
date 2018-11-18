import pytest
import pandas as pd
from src.paths import project_dir, interim_data_path

from src.data.localdata import load_datetime_csv

@pytest.mark.skip(reason="save it for demo")
def test_load_csv_file_success():
    test_filename = project_dir / 'tests/data/test_datetime_data.csv'
    actual = load_datetime_csv(test_filename)
    assert isinstance(actual.index, pd.DatetimeIndex)

