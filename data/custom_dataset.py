
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import euclidean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" 
    AugmentedDataset
    Returns a ts together with an augmentation.
"""


class AugmentedDataset(Dataset):
    """
    Dataset wrapper untuk CARLA.

    Menghasilkan:
        ts_org
        ts_w_augment
        ts_ss_augment
        target
    """

    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()

        self.current_epoch = 0

        self.dataset = dataset

        # =========================================================
        # Simpan statistik normalisasi dari dataset asli
        # =========================================================
        self.mean = getattr(dataset, 'mean', 0)
        self.std = getattr(dataset, 'std', 1)

        # =========================================================
        # Ambil transform
        # =========================================================
        transform = dataset.transform
        sanomaly = dataset.sanomaly

        # Transform dilakukan di wrapper ini
        dataset.transform = None

        if isinstance(transform, dict):
            self.ts_transform = transform.get('standard', None)
            self.augmentation_transform = transform.get('augment', None)
        else:
            self.ts_transform = transform
            self.augmentation_transform = transform

        # =========================================================
        # Sub-sequence anomaly augmentation
        # =========================================================
        if callable(sanomaly):
            self.subseq_anomaly = sanomaly
        else:
            self.subseq_anomaly = self._default_subseq_anomaly

        # =========================================================
        # PENTING:
        # Buat self.samples SEBELUM create_pairs()
        # =========================================================
        self.samples = [{} for _ in range(len(dataset))]

        # =========================================================
        # Buat pasangan augmented samples
        # =========================================================
        self.create_pairs()

    def _default_subseq_anomaly(self, ts):
        """
        Augmentasi fallback untuk data AMR.
        Menambahkan Gaussian noise kecil.
        """

        augmented = ts.clone()

        if augmented.numel() == 0:
            return augmented

        noise = torch.randn_like(augmented) * 0.01

        return augmented + noise

    def create_pairs(self):

        # =========================================================
        # Buat augmented samples
        # =========================================================
        for index in range(len(self.dataset)):

            item = self.dataset[index]

            ts_org = item['ts_org'].clone().detach().to(device)
            ts_trg = item['target'].clone().detach().to(device)

            # -----------------------------------------------------
            # Augmentasi temporal/window
            # -----------------------------------------------------
            if index > 10:

                rand_nei = np.random.randint(
                    index - 10,
                    index
                )

                sample_nei = self.dataset[rand_nei]

                ts_w_augment = (
                    sample_nei['ts_org']
                    .clone()
                    .detach()
                    .to(device)
                )

            else:

                if callable(self.augmentation_transform):
                    ts_w_augment = self.augmentation_transform(ts_org)
                else:
                    ts_w_augment = ts_org.clone()

            # -----------------------------------------------------
            # Subsequence anomaly augmentation
            # -----------------------------------------------------
            ts_ss_augment = self.subseq_anomaly(ts_org)

            # -----------------------------------------------------
            # Simpan sample
            # -----------------------------------------------------
            self.samples[index] = {

                'ts_org':
                    ts_org,

                'ts_w_augment':
                    ts_w_augment,

                'ts_ss_augment':
                    ts_ss_augment,

                'target':
                    ts_trg
            }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.samples[index]

    def concat_ds(self, new_ds):

        self.dataset.data = np.concatenate(
            (
                self.dataset.data,
                new_ds.dataset.data
            ),
            axis=0
        )

        self.dataset.targets = np.concatenate(
            (
                self.dataset.targets,
                new_ds.dataset.targets
            ),
            axis=0
        )

""" 
    NeighborsDataset
    Returns a ts with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, transform, N_indices, F_indices, p, sanomaly=None):
        super(NeighborsDataset, self).__init__()
        
        if isinstance(transform, dict):
            self.ts_transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.ts_transform = transform
            self.augmentation_transform = transform

        self.subseq_anomaly = sanomaly
        # self.create_pairs()
       
        dataset.transform = None
        if hasattr(dataset, 'windows'):
            from data.ra_dataset import SaveAugmentedDataset
            dataset = SaveAugmentedDataset(
                torch.as_tensor(dataset.windows, dtype=torch.float32),
                torch.as_tensor(dataset.window_labels, dtype=torch.long)
            )
        all_data = dataset.data.to(device)
        self.dataset = dataset

        NN_indices = N_indices.copy() # Nearest neighbor indices (np.array  [len(dataset) x k])
        FN_indices = F_indices.copy()  # Nearest neighbor indices (np.array  [len(dataset) x k])

        num_neighbors = p.get('num_neighbors', 10)

        if p['num_neighbors'] is not None:
            self.NN_indices = NN_indices[:, :p['num_neighbors']]
            self.FN_indices = FN_indices[:, -p['num_neighbors']:]

        else:
            self.NN_indices = NN_indices
            self.FN_indices = FN_indices

        self.dataset.data = dataset.data.to(device)
        self.dataset.targets = dataset.targets.to(device)
        num_samples = self.dataset.data.shape[0]
        NN_index = np.array([np.random.choice(self.NN_indices[i], 1)[0] for i in range(num_samples)])
        FN_index = np.array([np.random.choice(self.FN_indices[i], 1)[0] for i in range(num_samples)])
        self.NNeighbor = all_data[NN_index]
        self.FNeighbor = all_data[FN_index]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        #NN_index = np.random.choice(self.N_indices[index], 1)[0]
        NNeighbor = self.NNeighbor.__getitem__(index)
        #FN_index = np.random.choice(self.F_indices[index], 1)[0]
        FNeighbor = self.FNeighbor.__getitem__(index)

        #anchor['ts_org'] = self.anchor_transform(anchor['ts_org'])
        #NNeighbor['ts_org'] = self.neighbor_transform(NNeighbor['ts_org'])
        #FNeighbor['ts_org'] = self.neighbor_transform(FNeighbor['ts_org'])

        output['anchor'] = anchor['ts_org']
        output['NNeighbor'] = NNeighbor
        output['FNeighbor'] = FNeighbor
        output['possible_nneighbors'] = torch.from_numpy(self.NN_indices[index])
        output['possible_fneighbors'] = torch.from_numpy(self.FN_indices[index])
        output['target'] = anchor['target']
        
        return output

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate((self.dataset.data, new_ds.dataset.data), axis=0)
        self.dataset.targets = np.concatenate((self.dataset.targets, new_ds.dataset.targets), axis=0)
