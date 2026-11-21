import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import json


class MIMIC3Dataset(Dataset):
    def __init__(self, pkl_path, json_path, tsid_list, split='train', slot_duration_mins=60, total_hours=24):
        """
        Args:
            pkl_path (str): Path to mimic_iii.pkl.
            json_path (str): Path to var_mean_std.json.
            split (str): 'train', 'val', or 'test'.
            slot_duration_mins (int): Duration of one time slot in minutes.
            total_hours (int): Total duration of data to use.
        """
        self.slot_duration_mins = slot_duration_mins
        self.total_slots = int(total_hours * 60 / slot_duration_mins)

        # 1. Load Statistics
        with open(json_path, 'r') as f:
            self.var_stats = json.load(f)

        # 2. Separate Demographic vars from Time-Series vars
        # Explicitly define static variables to exclude from the Graph
        self.demo_names = ['Age', 'Gender']

        # Get all available variables from stats, then exclude demo vars
        all_vars = sorted(list(self.var_stats.keys()))
        self.feature_names = [v for v in all_vars if v not in self.demo_names]

        # Mappings for Time-Series features
        self.feature_map = {name: i for i, name in enumerate(self.feature_names)}
        self.num_features = len(self.feature_names)  # Should be 129

        # Mappings for Demo features
        self.demo_map = {name: i for i, name in enumerate(self.demo_names)}
        self.num_demo = len(self.demo_names)  # Should be 2

        # 3. Load Data and Labels
        with open(pkl_path, 'rb') as f:
            all_data, labels_df, train_list, val_list, test_list = pickle.load(f)
        all_data['ts_id'] = all_data['ts_id'].astype(int)
        # Select split IDs
        self.ids = tsid_list

        # 4. Data Filtering & Optimization
        # Filter raw data for selected patients
        patient_data = all_data[all_data['ts_id'].isin(self.ids)]

        # A. Prepare Time-Series Data
        # Filter only time-series variables
        self.data = patient_data[patient_data['variable'].isin(self.feature_names)].copy()
        self.data = self.data[self.data['minute'] <= (total_hours * 60)]
        self.data = self.data[self.data['minute'] >= 0]

        # Pre-calculate slot indices
        self.data['slot_idx'] = (self.data['minute'] // self.slot_duration_mins).astype(int)

        # Group time-series data by ts_id for fast access
        self.grouped_data = dict(tuple(self.data.groupby('ts_id')))

        # B. Prepare Demographic Data
        # Filter only demo variables
        demo_subset = patient_data[patient_data['variable'].isin(self.demo_names)].copy()

        # Since Age/Gender are static, we just need one value per patient per variable.
        # Group by ts_id and variable, then take the mean (handles duplicates if any)
        self.grouped_demo = demo_subset.groupby(['ts_id', 'variable'])['value'].mean().unstack()
        # self.grouped_demo is now a DataFrame with index=ts_id, columns=['Age', 'Gender']

        # 5. Prepare Labels
        label_subset = labels_df[labels_df['ts_id'].isin(self.ids)]
        self.labels = dict(zip(label_subset['ts_id'], label_subset['in_hospital_mortality']))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ts_id = self.ids[idx]

        # --- Part 1: Time-Series Matrix Construction ---
        # Shape: [Num_Features (129), Seq_Len (24)]
        matrix = np.full((self.num_features, self.total_slots), np.nan, dtype=np.float32)

        if ts_id in self.grouped_data:
            patient_df = self.grouped_data[ts_id]

            # Pivot: Rows=Variable, Cols=Slot_Index
            pivot_df = patient_df.pivot_table(
                index='variable',
                columns='slot_idx',
                values='value',
                aggfunc='mean',
                observed=False
            )

            for var_name, row_data in pivot_df.iterrows():
                if var_name in self.feature_map:
                    feat_idx = self.feature_map[var_name]
                    valid_slots = row_data.index.astype(int)
                    valid_slots = valid_slots[valid_slots < self.total_slots]
                    matrix[feat_idx, valid_slots] = row_data[valid_slots].values

        # Normalization & Imputation for Time Series
        for i, feat_name in enumerate(self.feature_names):
            mean = self.var_stats[feat_name]['mean']
            std = self.var_stats[feat_name]['std']
            if std == 0: std = 1e-6

            mask = ~np.isnan(matrix[i, :])
            matrix[i, mask] = (matrix[i, mask] - mean) / std

        tensor_x = torch.from_numpy(matrix)

        # Forward Fill and Zero Fill for Time Series
        for i in range(self.num_features):
            feat_name = self.feature_names[i]
            last_val = 0.0
            for t in range(self.total_slots):
                if torch.isnan(tensor_x[i, t]):
                    tensor_x[i, t] = last_val
                else:
                    last_val = tensor_x[i, t]

        tensor_x = torch.nan_to_num(tensor_x, nan=0.0)

        # --- Part 2: Demographic Vector Construction ---
        # Shape: [Num_Demo (2)]
        demo_vec = np.zeros(self.num_demo, dtype=np.float32)

        # Extract values if they exist for this patient
        if ts_id in self.grouped_demo.index:
            for i, name in enumerate(self.demo_names):
                if name in self.grouped_demo.columns:
                    val = self.grouped_demo.loc[ts_id, name]
                    # Check if value is not NaN
                    if not np.isnan(val):
                        # Normalize Demo Data
                        mean = self.var_stats[name]['mean']
                        std = self.var_stats[name]['std']
                        if std == 0: std = 1e-6
                        demo_vec[i] = (val - mean) / std
                    else:
                        # If NaN, set to 0 (mean after normalization)
                        demo_vec[i] = 0.0
                else:
                    demo_vec[i] = 0.0

        tensor_demo = torch.from_numpy(demo_vec)

        # --- Part 3: Label ---
        label = self.labels.get(ts_id, 0)

        # Return: TS_Input, Demo_Input, Label
        return tensor_x, tensor_demo, torch.tensor(label, dtype=torch.long)


# Verification Block
if __name__ == "__main__":
    pkl_file = "../data/mimic3/mimic_iii.pkl"
    json_file = "../data/mimic3/var_mean_std.json"

    try:
        dataset = MIMIC3Dataset(pkl_file, json_file)

        # Verify feature counts
        print(f"TS Features: {dataset.num_features} (Expected 129)")
        print(f"Demo Features: {dataset.num_demo} (Expected 2)")

        x, demo, y = dataset[0]
        print(f"TS Shape: {x.shape}")  # [129, 24]
        print(f"Demo Shape: {demo.shape}")  # [2]
        print(f"Demo Values: {demo}")

    except FileNotFoundError:
        print("Files not found.")