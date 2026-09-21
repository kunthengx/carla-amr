import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.mypath import MyPath


class DatasetPathTests(unittest.TestCase):
    def test_bare_filename_uses_dataset_root(self):
        with patch.object(MyPath, 'db_root_dir', return_value='/datasets/amr'):
            self.assertEqual(MyPath.resolve_dataset_file('amr', 'data.xlsx'),
                             '/datasets/amr/data.xlsx')

    def test_explicit_relative_path_is_not_prefixed(self):
        relative = 'anomaly-injection-v2-complete/sample_output/Data_AMR_Anomaly_v2.xlsx'
        with patch.object(MyPath, 'db_root_dir') as root:
            self.assertEqual(MyPath.resolve_dataset_file('amr', relative),
                             os.path.abspath(relative))
            root.assert_not_called()

    def test_absolute_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.xlsx'
            self.assertEqual(MyPath.resolve_dataset_file('amr', path), str(path))

    def test_missing_explicit_path_does_not_fall_back_to_other_data(self):
        path = './missing/data.xlsx'
        with patch.object(MyPath, 'db_root_dir') as root:
            self.assertEqual(MyPath.resolve_dataset_file('amr', path),
                             os.path.abspath(path))
            root.assert_not_called()


if __name__ == '__main__':
    unittest.main()
