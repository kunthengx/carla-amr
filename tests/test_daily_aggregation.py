import unittest
import numpy as np
from utils.amr_dataset import aggregate_daily, AMR3PhaseDataset

class DailyTests(unittest.TestCase):
    def test_stats_count_labels_and_missing_day(self):
        x,y,dates,names=aggregate_daily(
            np.array([[2.],[4.],[9.],[5.],[7.]]), np.array([0,1,2,0,0]),
            ['2025-01-31 08:00','2025-01-31 10:00','2025-01-31 12:00',
             '2025-02-01 08:00','2025-02-03 08:00'], ['voltage'])
        np.testing.assert_allclose(x,[[5,2,9,3],[5,5,5,1],[7,7,7,1]])
        self.assertEqual(y.tolist(),[2,0,0])
        self.assertEqual(names,['voltage_mean','voltage_min','voltage_max','reading_count'])
        ds=AMR3PhaseDataset(None,x,y,np.array(['001']*3),window_size=2,dates=dates)
        self.assertEqual(len(ds),1)

    def test_invalid_measurements_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_daily([[np.nan]],[0],['2025-01-01'],['voltage'])
