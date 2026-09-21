import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, List, Optional

class AMR3PhaseDataset(Dataset):
    """
    Dataset untuk data AMR 3 Fasa dengan multi-pelanggan
    """
    def __init__(
        self,
        dataframe,
        data: np.ndarray,
        labels: np.ndarray,
        location_ids: np.ndarray,
        window_size: int = 50,
        train=True,
        sanomaly=None,
        transform: Optional[callable] = None,
        is_train: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        anomaly_ratio=0.20,      # 20% pelanggan akan diinjeksi
        defect_ratio=0.40,       # 40% anomaly = defect
        theft_ratio=0.60,        # 60% anomaly = theft
        random_seed=42
    ):
        """
        Args:
            data: Array dengan shape (n_samples, n_features) 
                  Fitur: [V_L1, V_L2, V_L3, I_L1, I_L2, I_L3, I_N, PF_L1, PF_L2, PF_L3, PF_total,
                         P_L1, P_L2, P_L3, P_total, Q_L1, Q_L2, Q_L3, Q_total, S_L1, S_L2, S_L3, S_total, freq]
            labels: Array label (0=normal, 1=defect, 2=theft)
            location_ids: ID pelanggan per sampel
            window_size: Ukuran window untuk sliding window
            transform: Transformasi/augmentasi
            is_train: Flag training
            mean, std: Untuk normalisasi
            anomaly_ratio, defect_ratio, theft_ratio: Rasio anomali untuk masing-masing kelas
            random_seed: Seed untuk reproduksibilitas
        """
        self.data = data
        self.labels = labels
        self.location_ids = location_ids

        self.window_size = window_size

        self.transform = transform
        self.is_train = is_train
        self.sanomaly = sanomaly
        
        # Normalisasi data
        if is_train:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
        else:
            self.mean = mean if mean is not None else np.mean(data, axis=0)
            self.std = std if std is not None else np.std(data, axis=0) + 1e-8
            
        self.normalized_data = (data - self.mean) / self.std
        
        # ============================================================
        # SLIDING WINDOW PER PELANGGAN
        # ============================================================

        self.windows = []
        self.window_labels = []
        self.window_locations = []

        ws = self.window_size

        # Jangan membuat window lintas pelanggan
        unique_locations = np.unique(self.location_ids)

        for loc in unique_locations:

            # Ambil index seluruh data milik pelanggan ini
            loc_idx = np.where(self.location_ids == loc)[0]

            loc_data = self.normalized_data[loc_idx]
            loc_labels = self.labels[loc_idx]

            n_loc = len(loc_data)

            # Pelanggan harus mempunyai minimal ws data
            if n_loc < ws:
                continue

            # Sliding window hanya dalam pelanggan yang sama
            for start in range(0, n_loc - ws + 1):

                end = start + ws

                window = loc_data[start:end]

                # Label window:
                # jika ada anomaly di dalam window,
                # gunakan label anomaly tertinggi
                #
                # 0 = normal
                # 1 = defect
                # 2 = theft
                window_label = int(
                    np.max(loc_labels[start:end])
                )

                self.windows.append(window)
                self.window_labels.append(window_label)
                self.window_locations.append(loc)

        # Convert ke numpy
        self.windows = np.asarray(
            self.windows,
            dtype=np.float32
        )

        self.window_labels = np.asarray(
            self.window_labels,
            dtype=np.int64
        )

        self.window_locations = np.asarray(
            self.window_locations
        )

        print(
            f"Window size = {ws}, "
            f"total windows = {len(self.windows)}, "
            f"customers = {len(unique_locations)}"
        )
        if len(self.windows) == 0:
            print("No customer has enough data for the requested window size.")
        
    def __len__(self):
        return len(self.windows)

    def get_info(self):
        return self.mean, self.std
    
    def __getitem__(self, idx):
        window = self.windows[idx]

        if not isinstance(window, torch.Tensor):
            window = torch.from_numpy(window).float()
        else:
            window = window.float()

        target = torch.tensor(self.window_labels[idx], dtype=torch.long)

        return {
            'ts_org': window,
            'target': target
    }

    def concat_ds(self, other_ds):
        """Menggabungkan dengan dataset lain"""
        self.windows = np.concatenate([self.windows, other_ds.windows])
        self.window_labels = np.concatenate([self.window_labels, other_ds.window_labels])
        self.window_locations = np.concatenate([self.window_locations, other_ds.window_locations])
        # Update mean dan std
        combined_data = np.concatenate([
            self.data, 
            other_ds.data
        ], axis=0)
        self.mean = np.mean(combined_data, axis=0)
        self.std = np.std(combined_data, axis=0) + 1e-8


def load_amr_data(
    file_path: str = '/home/kunthengx/Documents/CARLA/anomaly-injection-v2-complete/sample_output/Data_AMR_Anomaly_v2.xlsx',
    window_size: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> dict:
    """
    Memuat dan memproses data AMR dari Excel
    
    Returns:
        dict dengan keys: 'train_data', 'train_labels', 'val_data', 'val_labels',
                          'test_data', 'test_labels', 'features', 'locations'
    """
    # Baca semua sheet
    excel_file = pd.ExcelFile(file_path)
    all_data = []
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        all_data.append(df)
    
    # Gabungkan semua data
    df = pd.concat(all_data, ignore_index=True)
    
    print(f"Total data: {len(df)} samples dari {df['LOCATION_CODE'].nunique()} pelanggan")
    
    # === FEATURE ENGINEERING ===
    
    # Pilih fitur numerik
    feature_columns = [
        'VOLTAGE_L1', 'VOLTAGE_L2', 'VOLTAGE_L3',
        'CURRENT_L1', 'CURRENT_L2', 'CURRENT_L3', 'CURRENT_N',
        'POWER_FACTOR_L1', 'POWER_FACTOR_L2', 'POWER_FACTOR_L3', 'POWER_FACTOR_TOTAL',
        'ACTIVE_POWER_L1', 'ACTIVE_POWER_L2', 'ACTIVE_POWER_L3', 'ACTIVE_POWER_TOTAL',
        'REACTIVE_POWER_L1', 'REACTIVE_POWER_L2', 'REACTIVE_POWER_L3', 'REACTIVE_POWER_TOTAL',
        'APPARENT_POWER_L1', 'APPARENT_POWER_L2', 'APPARENT_POWER_L3', 'APPARENT_POWER_TOTAL',
        'FREQUENCY'
    ]
    
    # === FEATURE ENGINEERING TAMBAHAN (Domain Knowledge) ===
    
    # 1. Ketidakseimbangan Tegangan (Voltage Unbalance)
    voltages = df[['VOLTAGE_L1', 'VOLTAGE_L2', 'VOLTAGE_L3']].values
    df['VOLTAGE_UNBALANCE'] = np.std(voltages, axis=1) / np.mean(voltages, axis=1)
    
    # 2. Ketidakseimbangan Arus (Current Unbalance)
    currents = df[['CURRENT_L1', 'CURRENT_L2', 'CURRENT_L3']].values
    df['CURRENT_UNBALANCE'] = np.std(currents, axis=1) / (np.mean(currents, axis=1) + 1e-8)
    
    # 3. Rasio Arus Netral terhadap Arus Rata-rata
    avg_current = np.mean(currents, axis=1)
    df['NEUTRAL_RATIO'] = df['CURRENT_N'] / (avg_current + 1e-8)
    
    # 4. Efisiensi Daya (Total Active / Total Apparent)
    df['POWER_EFFICIENCY'] = df['ACTIVE_POWER_TOTAL'] / (df['APPARENT_POWER_TOTAL'] + 1e-8)
    
    # 5. Faktor Daya Rata-rata
    df['AVG_PF'] = df[['POWER_FACTOR_L1', 'POWER_FACTOR_L2', 'POWER_FACTOR_L3']].mean(axis=1)
    
    # 6. Daya Reaktif vs Daya Aktif (indikasi beban induktif/kapasitif)
    df['REACTIVE_ACTIVE_RATIO'] = df['REACTIVE_POWER_TOTAL'] / (df['ACTIVE_POWER_TOTAL'] + 1e-8)
    
    # 7. Power Factor Standard Deviation (indikasi ketidakstabilan)
    df['PF_STD'] = df[['POWER_FACTOR_L1', 'POWER_FACTOR_L2', 'POWER_FACTOR_L3']].std(axis=1)
    
    # Update feature columns
    feature_columns += [
        'VOLTAGE_UNBALANCE', 'CURRENT_UNBALANCE', 'NEUTRAL_RATIO',
        'POWER_EFFICIENCY', 'AVG_PF', 'REACTIVE_ACTIVE_RATIO', 'PF_STD'
    ]

    # ============================================================
    # SPLIT DATA BERDASARKAN PELANGGAN (GROUP SPLIT)
    # ============================================================
    #
    # Setiap LOCATION_CODE hanya boleh berada pada satu split.
    # Tujuan:
    #   Train      ≈ 70% pelanggan
    #   Validation ≈ 15% pelanggan
    #   Test       ≈ 15% pelanggan
    #
    # Ini menghindari:
    #   - Train/Val kosong akibat pelanggan hanya punya 1 record
    #   - Data pelanggan yang sama masuk ke train dan test
    # ============================================================

    all_data = []
    all_labels = []
    all_locations = []

    locations = df['LOCATION_CODE'].dropna().unique()

    for loc in locations:

        loc_data = df[df['LOCATION_CODE'] == loc].copy()

        # Urutkan berdasarkan waktu
        loc_data = loc_data.sort_values('READ_DATE')

        # Extract features
        X = loc_data[feature_columns].values.astype(np.float32)

        # ========================================================
        # LABELING
        # ========================================================

        # --------------------------------------------------------
        # PRIORITAS 1:
        # Gunakan label yang sudah tersedia dari anomaly-injection-v2
        # 0 = Normal
        # 1 = Defect
        # 2 = Theft
        # --------------------------------------------------------
        if 'label' in loc_data.columns:

            labels = (
                pd.to_numeric(loc_data['label'], errors='coerce')
                .fillna(0)
                .astype(np.int64)
                .values
            )

            # Validasi label
            valid_labels = np.isin(labels, [0, 1, 2])

            if not np.all(valid_labels):
                invalid_values = np.unique(labels[~valid_labels])

                raise ValueError(
                    f"Label tidak valid ditemukan: {invalid_values}. "
                    "Label yang diperbolehkan hanya 0, 1, dan 2."
                )

        # --------------------------------------------------------
        # PRIORITAS 2:
        # Jika dataset tidak memiliki kolom label,
        # gunakan heuristic/rule lama.
        # --------------------------------------------------------
        else:

            labels = np.zeros(len(X), dtype=np.int64)

            # ==========================
            # DEFECT
            # ==========================

            loc_currents = loc_data[
                ['CURRENT_L1', 'CURRENT_L2', 'CURRENT_L3']
            ].values

            loc_avg_current = np.mean(
                loc_currents,
                axis=1
            )

            neutral_high = (
                (loc_data['CURRENT_N'] > 0.5)
                &
                (
                    loc_data['CURRENT_N']
                    /
                    (loc_avg_current + 1e-8)
                    > 0.3
                )
            )

            voltage_low = (
                (loc_data['VOLTAGE_L1'] < 180)
                |
                (loc_data['VOLTAGE_L2'] < 180)
                |
                (loc_data['VOLTAGE_L3'] < 180)
            )

            labels[
                neutral_high | voltage_low
            ] = 1

            # ==========================
            # THEFT
            # ==========================

            pf_low = (
                loc_data['POWER_FACTOR_TOTAL'] < 0.5
            )

            power_negative = (
                loc_data['ACTIVE_POWER_TOTAL'] < 0
            )

            labels[
                pf_low | power_negative
            ] = 2

        # ========================================================
        # SIMPAN SEMUA DATA TERLEBIH DAHULU
        # ========================================================

        all_data.append(X)
        all_labels.append(labels)
        all_locations.extend([loc] * len(X))


    # ============================================================
    # GABUNGKAN SELURUH DATA
    # ============================================================

    X_all = np.concatenate(all_data, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    groups = np.asarray(all_locations)


    print("\n=== GROUP SPLIT BY CUSTOMER ===")
    print(f"Total samples  : {len(X_all)}")
    print(f"Total customers: {len(np.unique(groups))}")


    # ============================================================
    # SPLIT 70% TRAIN - 30% TEMP
    # ============================================================

    gss_train = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42
    )

    train_idx, temp_idx = next(
        gss_train.split(
            X_all,
            y_all,
            groups=groups
        )
    )


    # ============================================================
    # SPLIT TEMP 50% VALIDATION - 50% TEST
    #
    # 30% temp × 50% = 15% validation
    # 30% temp × 50% = 15% test
    # ============================================================

    gss_val_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_rel_idx, test_rel_idx = next(
        gss_val_test.split(
            X_all[temp_idx],
            y_all[temp_idx],
            groups=groups[temp_idx]
        )
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]


    # ============================================================
    # AMBIL DATA SESUAI SPLIT
    # ============================================================

    train_data = X_all[train_idx]
    train_labels = y_all[train_idx]
    train_locations = groups[train_idx]

    val_data = X_all[val_idx]
    val_labels = y_all[val_idx]
    val_locations = groups[val_idx]

    test_data = X_all[test_idx]
    test_labels = y_all[test_idx]
    test_locations = groups[test_idx]


    # ============================================================
    # INFORMASI CUSTOMER
    # ============================================================

    train_customers = np.unique(groups[train_idx])
    val_customers = np.unique(groups[val_idx])
    test_customers = np.unique(groups[test_idx])


    # ============================================================
    # VALIDASI TIDAK ADA CUSTOMER YANG OVERLAP
    # ============================================================

    assert len(
        set(train_customers) & set(val_customers)
    ) == 0, "Customer overlap antara TRAIN dan VAL!"

    assert len(
        set(train_customers) & set(test_customers)
    ) == 0, "Customer overlap antara TRAIN dan TEST!"

    assert len(
        set(val_customers) & set(test_customers)
    ) == 0, "Customer overlap antara VAL dan TEST!"


    # ============================================================
    # PRINT HASIL SPLIT
    # ============================================================

    print("\n=== HASIL SPLIT ===")

    print(
        f"Train: {len(train_data)} samples "
        f"dari {len(train_customers)} pelanggan"
    )

    print(
        f"Val  : {len(val_data)} samples "
        f"dari {len(val_customers)} pelanggan"
    )

    print(
        f"Test : {len(test_data)} samples "
        f"dari {len(test_customers)} pelanggan"
    )

    print(
        f"\nClass distribution train: "
        f"{np.bincount(train_labels, minlength=3)}"
    )

    print(
        f"Class distribution val  : "
        f"{np.bincount(val_labels, minlength=3)}"
    )

    print(
        f"Class distribution test : "
        f"{np.bincount(test_labels, minlength=3)}"
    )

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'train_locations': train_locations,

        'val_data': val_data,
        'val_labels': val_labels,
        'val_locations': val_locations,

        'test_data': test_data,
        'test_labels': test_labels,
        'test_locations': test_locations,

        'features': feature_columns,
        'locations': all_locations
    }
