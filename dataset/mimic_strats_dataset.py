import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import json
import gc


class MIMICStratsDataset(Dataset):
    def __init__(self, pkl_path, json_path, tsid_list, total_hours=24, max_seq_len=880,
                 truncation_seed=None):
        """Prepare standardized iSTraTS event triples and demographics for each stay."""
        self.max_seq_len = max_seq_len
        self.total_hours = total_hours
        self.truncation_seed = truncation_seed

        with open(json_path, 'r') as f:
            self.var_stats = json.load(f)

        self.demo_names = ['Age', 'Gender']
        all_vars = sorted(list(self.var_stats.keys()))
        self.feature_names = [v for v in all_vars if v not in self.demo_names]

        self.feature_map = {name: i for i, name in enumerate(self.feature_names)}
        self.num_features = len(self.feature_names)
        self.num_demo = len(self.demo_names)

        with open(pkl_path, 'rb') as f:
            all_data, labels_df, _, _, _ = pickle.load(f)

        self.ids = tsid_list

        mask = all_data['ts_id'].isin(self.ids)
        _ts_id = all_data.loc[mask, 'ts_id'].values
        _variable = all_data.loc[mask, 'variable'].values
        _minute = all_data.loc[mask, 'minute'].values
        _value = all_data.loc[mask, 'value'].values

        del all_data
        del mask
        gc.collect()

        patient_data = pd.DataFrame({
            'ts_id': _ts_id.astype(int),
            'variable': _variable.astype(str),
            'minute': _minute.astype(float),
            'value': _value.astype(float)
        })

        self.data = patient_data[patient_data['variable'].isin(self.feature_names)]
        self.data = self.data[(self.data['minute'] <= (total_hours * 60)) & (self.data['minute'] >= 0)]
        self.data = self.data.dropna(subset=['value'])
        self.grouped_data = dict(tuple(self.data.groupby('ts_id')))

        demo_subset = patient_data[patient_data['variable'].isin(self.demo_names)].copy()

        # Replace age values shifted for de-identification.
        mask_age = (demo_subset['variable'] == 'Age') & (demo_subset['value'] > 200)
        demo_subset.loc[mask_age, 'value'] = 91.4
        # ====================================================

        self.grouped_demo = demo_subset.groupby(['ts_id', 'variable'])['value'].mean().unstack()

        label_subset = labels_df[labels_df['ts_id'].isin(self.ids)]
        self.labels = dict(zip(label_subset['ts_id'], label_subset['in_hospital_mortality']))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ts_id = self.ids[idx]

        demo_vec = np.zeros(self.num_demo, dtype=np.float32)
        if ts_id in self.grouped_demo.index:
            for i, name in enumerate(self.demo_names):
                if name in self.grouped_demo.columns:
                    val = self.grouped_demo.loc[ts_id, name]
                    if not np.isnan(val):
                        mean = self.var_stats[name]['mean']
                        std = self.var_stats[name]['std']
                        if std == 0: std = 1e-6
                        demo_vec[i] = (val - mean) / std
        tensor_demo = torch.from_numpy(demo_vec)

        values = []
        times = []
        varis = []

        if ts_id in self.grouped_data:
            patient_df = self.grouped_data[ts_id]

            if len(patient_df) > self.max_seq_len:
                # Sample events when a stay exceeds the sequence length limit.
                random_state = None if self.truncation_seed is None else (
                    int(self.truncation_seed) + int(ts_id)
                ) % (2**32)
                patient_df = patient_df.sample(
                    n=self.max_seq_len, replace=False, random_state=random_state
                )
            # ==============================================================

            max_minute = self.total_hours * 60.0

            for _, row in patient_df.iterrows():
                var_name = row['variable']
                if var_name not in self.feature_map: continue

                val = row['value']
                minute = row['minute']

                mean = self.var_stats[var_name]['mean']
                std = self.var_stats[var_name]['std']
                if std == 0: std = 1e-6
                norm_val = (val - mean) / std

                # Map the observation time to [-1, 1].
                norm_time = (minute / max_minute) * 2 - 1.0
                # ===========================================

                values.append(norm_val)
                times.append(norm_time)
                varis.append(self.feature_map[var_name])

        seq_len = len(values)
        obs_mask = [1] * seq_len

        pad_len = self.max_seq_len - seq_len
        values.extend([0.0] * pad_len)
        times.extend([0.0] * pad_len)
        varis.extend([0] * pad_len)
        obs_mask.extend([0] * pad_len)

        tensor_values = torch.tensor(values, dtype=torch.float32)
        tensor_times = torch.tensor(times, dtype=torch.float32)
        tensor_varis = torch.tensor(varis, dtype=torch.long)
        tensor_obs_mask = torch.tensor(obs_mask, dtype=torch.float32)

        label = self.labels.get(ts_id, 0)
        tensor_label = torch.tensor(label, dtype=torch.float32)

        return tensor_values, tensor_times, tensor_varis, tensor_obs_mask, tensor_demo, tensor_label


if __name__ == "__main__":
    pkl_file = "data/mimic3/mimic_iii.pkl"
    json_file = "data/mimic3/var_mean_std.json"

    dataset = MIMICStratsDataset(pkl_file, json_file, tsid_list=[1, 2, 3], max_seq_len=880)
    print(f"Num Features (V): {dataset.num_features}")

    values, times, varis, obs_mask, demo, y = dataset[0]
    print(f"Values Shape: {values.shape}")
    print(f"Times Preview: {times[:10]}")
    print(f"Varis Preview: {varis[:10]}")
