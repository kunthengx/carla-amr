import unittest

import numpy as np
import torch

from losses.losses import PretextLoss
from utils.common_config import (
    get_train_dataset, get_val_dataset, get_val_dataloader,
)


class AMRPipelineTests(unittest.TestCase):
    def setUp(self):
        self.p = {'train_db_name': 'amr', 'window_size': 2,
                  'batch_size': 4, 'num_workers': 0}
        self.data = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.labels = np.zeros(10, dtype=np.int64)
        self.locations = np.repeat([10, 20], 5)

    def train_dataset(self, augmented=False):
        return get_train_dataset(
            self.p, None, None, to_augmented_dataset=augmented,
            data=self.data, label=self.labels, location_ids=self.locations,
        )

    def test_windows_preserve_customers_and_training_mode(self):
        ds = self.train_dataset()
        self.assertTrue(ds.is_train)
        self.assertEqual(ds.windows.shape, (8, 2, 3))
        np.testing.assert_array_equal(ds.window_locations, [10]*4 + [20]*4)

    def test_augmentation_does_not_normalize_twice(self):
        ds = self.train_dataset(augmented=True)
        torch.testing.assert_close(ds[0]['ts_org'].cpu(), ds.dataset[0]['ts_org'])

    def test_validation_uses_training_statistics_and_keeps_last_batch(self):
        train = self.train_dataset()
        val = get_val_dataset(
            self.p, mean_data=train.mean, std_data=train.std,
            data=self.data[:8] + 100, label=self.labels[:8],
            location_ids=self.locations[:8],
        )
        np.testing.assert_allclose(val.normalized_data,
                                   (self.data[:8] + 100 - train.mean) / train.std)
        batches = list(get_val_dataloader(self.p, val))
        self.assertEqual([len(b['target']) for b in batches], [4, 2])

    def test_pretext_loss_supports_batch_smaller_than_configured(self):
        features = torch.randn(9, 8, requires_grad=True)
        loss = PretextLoss(128, temperature=0.5)(features)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())


if __name__ == '__main__':
    unittest.main()
