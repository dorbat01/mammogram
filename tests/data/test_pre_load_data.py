import unittest
import numpy as np
from src.paths import project_dir, interim_data_path

from src.data.localdata import load_csv

@unittest.skip(reason='skip for demo')
class PreloadDataTestCase(unittest.TestCase):

    def test_load_csv_file_success(self):
        test_filename = project_dir / 'tests/data/test_data_outofrange.csv'
        df = load_csv(test_filename)
        self.assertTrue(np.all(df['age'] >= 18) and np.all(df['age'] <= 100), "invalid age")


if __name__ == '__main__':
    unittest.main()

