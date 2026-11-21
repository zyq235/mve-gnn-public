"""StraTS input adapter for the 24-hour remaining-ICU-LOS task."""

from __future__ import annotations

import gc
import json
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StratsLOSDataset(Dataset):
    """Create fixed-length iStraTS event sequences with LOS targets.

    A base instance owns the filtered event table. ``subset`` creates cheap
    train/validation/test views, avoiding another multi-gigabyte pickle load
    for every fold.
    """

    def __init__(
        self,
        pkl_path,
        json_path,
        tsid_list,
        target_frame: pd.DataFrame,
        total_hours=24,
        max_seq_len=880,
        events=None,
        training=False,
    ):
        self.max_seq_len = int(max_seq_len)
        self.total_hours = float(total_hours)
        self.training = bool(training)
        self.ids = list(dict.fromkeys(int(value) for value in tsid_list))
        if self.max_seq_len <= 0 or self.total_hours <= 0:
            raise ValueError("max_seq_len and total_hours must be positive")

        with open(json_path, "r", encoding="utf-8") as handle:
            self.var_stats = json.load(handle)
        self.demo_names = ["Age", "Gender"]
        missing_demo = [name for name in self.demo_names if name not in self.var_stats]
        if missing_demo:
            raise KeyError(
                f"Demographic variables are missing from {json_path}: {missing_demo}"
            )
        self.feature_names = sorted(
            variable for variable in self.var_stats if variable not in self.demo_names
        )
        self.feature_map = {
            variable: index for index, variable in enumerate(self.feature_names)
        }
        self.num_features = len(self.feature_names)
        self.num_demo = len(self.demo_names)

        if events is None:
            with open(pkl_path, "rb") as handle:
                events, _, _, _, _ = pickle.load(handle)
        required_event_columns = {"ts_id", "variable", "minute", "value"}
        missing_columns = required_event_columns - set(events.columns)
        if missing_columns:
            raise ValueError(f"Event table is missing columns: {sorted(missing_columns)}")

        selected = events.loc[
            events["ts_id"].isin(self.ids),
            ["ts_id", "variable", "minute", "value"],
        ].copy()
        selected["ts_id"] = pd.to_numeric(selected["ts_id"], errors="coerce")
        selected["minute"] = pd.to_numeric(selected["minute"], errors="coerce")
        selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
        selected["variable"] = selected["variable"].astype("string")
        selected = selected.dropna(subset=["ts_id", "variable", "minute", "value"])
        selected["ts_id"] = selected["ts_id"].astype(np.int64)
        selected["minute"] = selected["minute"].astype(np.float64)
        selected["value"] = selected["value"].astype(np.float64)

        # The LOS landmark is [0, 24h): pre-ICU and boundary/post-landmark
        # observations are excluded to prevent target leakage.
        landmark_minutes = self.total_hours * 60.0
        temporal_mask = (
            selected["variable"].isin(self.feature_map)
            & selected["minute"].ge(0.0)
            & selected["minute"].lt(landmark_minutes)
        )
        self.data = selected.loc[
            temporal_mask, ["ts_id", "variable", "minute", "value"]
        ].copy()
        self.grouped_data = self.data.groupby("ts_id", sort=False)

        demo_subset = selected.loc[
            selected["variable"].isin(self.demo_names),
            ["ts_id", "variable", "value"],
        ].copy()
        age_mask = demo_subset["variable"].eq("Age") & demo_subset["value"].gt(200)
        demo_subset.loc[age_mask, "value"] = np.float32(91.4)
        self.grouped_demo = demo_subset.groupby(
            ["ts_id", "variable"], observed=True
        )["value"].mean().unstack()
        del selected, demo_subset
        gc.collect()

        required_target_columns = {
            "ts_id",
            "los_class",
            "remaining_los_days",
            "icu_death",
            "in_hospital_mortality",
        }
        missing_target_columns = required_target_columns - set(target_frame.columns)
        if missing_target_columns:
            raise ValueError(
                f"LOS target frame is missing columns: {sorted(missing_target_columns)}"
            )
        targets = target_frame.drop_duplicates("ts_id").set_index("ts_id")
        self.los_class = {
            int(stay_id): int(value)
            for stay_id, value in targets["los_class"].items()
        }
        self.remaining_los_days = {
            int(stay_id): float(value)
            for stay_id, value in targets["remaining_los_days"].items()
        }
        self.icu_death = {
            int(stay_id): int(value)
            for stay_id, value in targets["icu_death"].items()
        }
        self.in_hospital_mortality = {
            int(stay_id): int(value)
            for stay_id, value in targets["in_hospital_mortality"].items()
        }
        missing_targets = sorted(set(self.ids) - set(self.los_class))
        if missing_targets:
            raise KeyError(
                f"LOS labels are missing for {len(missing_targets)} stays; "
                f"first IDs: {missing_targets[:10]}"
            )

    def subset(self, tsid_list, training=False):
        subset = object.__new__(type(self))
        subset.__dict__ = self.__dict__.copy()
        subset.ids = list(dict.fromkeys(int(value) for value in tsid_list))
        subset.training = bool(training)
        missing = sorted(set(subset.ids) - set(self.los_class))
        if missing:
            raise KeyError(f"Unknown ICU stays requested: {missing[:10]}")
        return subset

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _safe_scale(value, statistics):
        mean = float(statistics["mean"])
        std = float(statistics["std"])
        if not np.isfinite(std) or std == 0:
            std = 1.0
        return (value - mean) / std

    def __getitem__(self, index):
        stay_id = int(self.ids[index])
        demo = np.zeros(self.num_demo, dtype=np.float32)
        if stay_id in self.grouped_demo.index:
            for demo_index, variable in enumerate(self.demo_names):
                if variable not in self.grouped_demo.columns:
                    continue
                value = self.grouped_demo.loc[stay_id, variable]
                if pd.notna(value):
                    demo[demo_index] = self._safe_scale(
                        float(value), self.var_stats[variable]
                    )

        if stay_id in self.grouped_data.indices:
            patient = self.grouped_data.get_group(stay_id)
            if len(patient) > self.max_seq_len:
                if self.training:
                    patient = patient.sample(n=self.max_seq_len, replace=False)
                else:
                    patient = patient.nsmallest(self.max_seq_len, "minute")
        else:
            patient = self.data.iloc[0:0]

        sequence_length = min(len(patient), self.max_seq_len)
        values = np.zeros(self.max_seq_len, dtype=np.float32)
        times = np.zeros(self.max_seq_len, dtype=np.float32)
        variables = np.zeros(self.max_seq_len, dtype=np.int64)
        observation_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        if sequence_length:
            variable_names = patient["variable"].astype(str).to_numpy()[:sequence_length]
            raw_values = patient["value"].to_numpy(dtype=np.float32)[:sequence_length]
            minutes = patient["minute"].to_numpy(dtype=np.float32)[:sequence_length]
            values[:sequence_length] = np.asarray(
                [
                    self._safe_scale(value, self.var_stats[variable])
                    for variable, value in zip(variable_names, raw_values)
                ],
                dtype=np.float32,
            )
            times[:sequence_length] = (
                minutes / (self.total_hours * 60.0)
            ) * 2.0 - 1.0
            variables[:sequence_length] = np.asarray(
                [self.feature_map[variable] for variable in variable_names],
                dtype=np.int64,
            )
            observation_mask[:sequence_length] = 1.0

        return (
            torch.from_numpy(values),
            torch.from_numpy(times),
            torch.from_numpy(variables),
            torch.from_numpy(observation_mask),
            torch.from_numpy(demo),
            torch.tensor(self.los_class[stay_id], dtype=torch.long),
            torch.tensor(self.remaining_los_days[stay_id], dtype=torch.float32),
            torch.tensor(stay_id, dtype=torch.long),
            torch.tensor(self.icu_death[stay_id], dtype=torch.long),
            torch.tensor(self.in_hospital_mortality[stay_id], dtype=torch.long),
        )
