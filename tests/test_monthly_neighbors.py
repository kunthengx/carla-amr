import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.amr_dataset import read_amr_files, AMR3PhaseDataset
from utils.repository import TSRepository

class MonthlyTests(unittest.TestCase):
    def test_month_boundary_gap_customer_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'LOCATION_CODE':['001','001','002'],
                          'READ_DATE':['2025-01-30','2025-01-31','2025-01-31'],
                          'value':[1,2,9]}).to_csv(root/'jan.csv', index=False)
            pd.DataFrame({'LOCATION_CODE':['001','001','001'],
                          'READ_DATE':['2025-01-31','2025-02-01','2025-02-03'],
                          'value':[2,3,4]}).to_excel(root/'feb.xlsx', index=False)
            frame = read_amr_files(root)
            self.assertEqual(len(frame),5)
            self.assertIn('001',frame.LOCATION_CODE.values)
            ds=AMR3PhaseDataset(None,frame[['value']].values,
                np.zeros(len(frame),dtype=int),frame.LOCATION_CODE.values,
                window_size=3,dates=frame.READ_DATE.values)
            self.assertEqual(ds.windows.shape,(1,3,1))
            self.assertEqual(ds.window_locations.tolist(),['001'])
            pd.DataFrame({'LOCATION_CODE':['001'],'READ_DATE':['2025-01-31'],
                          'value':[99]}).to_csv(root/'conflict.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                read_amr_files(root)

class NeighborTests(unittest.TestCase):
    def test_neighbors_per_sample_and_unused_capacity(self):
        repo=TSRepository(8,1,3,0.5)
        repo.update(torch.tensor([[0.],[1.],[4.],[10.]]),torch.zeros(4,dtype=torch.long))
        far,near=repo.furthest_nearest_neighbors(1)
        np.testing.assert_array_equal(near[:,0],[1,0,1,2])
        np.testing.assert_array_equal(far[:,0],[3,3,3,0])
        self.assertEqual(repo.mine_nearest_neighbors(1)[2],1.0)

    def test_ties_and_large_k_exclude_self(self):
        repo=TSRepository(3,2,3,0.5)
        repo.update(torch.zeros(3,2),torch.zeros(3,dtype=torch.long))
        far,near=repo.furthest_nearest_neighbors(10)
        for indices in (far,near):
            self.assertEqual(indices.shape,(3,2))
            for i,row in enumerate(indices):
                self.assertEqual(set(row),set(range(3))-{i})

if __name__=='__main__':
    unittest.main()
