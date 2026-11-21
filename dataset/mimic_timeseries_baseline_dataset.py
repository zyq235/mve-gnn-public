"""MIMIC adapters for the GRU, GRU-D, SAnD, and TCN baselines.

The pickle is expensive to load, so :class:`MIMICTimeseriesDataStore` compacts
it once.  Fold-specific datasets are lightweight views over that shared store.
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_DENSE_MODELS = {"gru", "sand", "tcn"}
_SUPPORTED_MODELS = _DENSE_MODELS | {"grud"}


class MIMICTimeseriesDataStore:
    """Load and compact one processed MIMIC pickle for all folds.

    Time-varying values are standardized with ``var_mean_std.json``.  Missing
    normalized values therefore remain zero (mean imputation), matching the
    preprocessing used by the reference baseline code without materializing a
    dense cohort-sized tensor.
    """

    demo_names = ("Age", "Gender")

    def __init__(self, pkl_path, json_path, total_hours: float = 24):
        self.pkl_path = str(pkl_path)
        self.json_path = str(json_path)
        self.total_hours = float(total_hours)
        self.max_minute = self.total_hours * 60.0

        with Path(json_path).open("r", encoding="utf-8") as handle:
            self.var_stats = json.load(handle)
        self.feature_names = sorted(
            name for name in self.var_stats if name not in self.demo_names
        )
        self.feature_map = {
            feature_name: index
            for index, feature_name in enumerate(self.feature_names)
        }
        self.num_features = len(self.feature_names)
        self.num_demo = len(self.demo_names)
        if self.num_features == 0:
            raise ValueError("No time-varying features were found in the statistics file")

        with Path(pkl_path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, (tuple, list)) or len(payload) != 5:
            raise ValueError(
                "Expected pickle payload (events, outcomes, train_ids, val_ids, test_ids)"
            )
        all_data, outcomes, train_ids, val_ids, test_ids = payload
        del payload

        required_event_columns = {"ts_id", "variable", "minute", "value"}
        missing_columns = required_event_columns.difference(all_data.columns)
        if missing_columns:
            raise ValueError(f"Event table is missing columns: {sorted(missing_columns)}")
        if "ts_id" not in outcomes.columns or "in_hospital_mortality" not in outcomes.columns:
            raise ValueError("Outcome table must contain ts_id and in_hospital_mortality")

        self.train_ids = np.asarray(train_ids, dtype=np.int64)
        self.val_ids = np.asarray(val_ids, dtype=np.int64)
        self.test_ids = np.asarray(test_ids, dtype=np.int64)
        cohort_ids = np.concatenate((self.train_ids, self.val_ids, self.test_ids))
        cohort_ids = np.unique(cohort_ids)

        outcome_subset = outcomes.loc[outcomes["ts_id"].isin(cohort_ids)].copy()
        duplicate_labels = outcome_subset.groupby("ts_id")[
            "in_hospital_mortality"
        ].nunique()
        if duplicate_labels.gt(1).any():
            raise ValueError("At least one ICU stay has conflicting mortality labels")
        self.outcomes = outcome_subset.drop_duplicates(subset=["ts_id"])
        self.labels = {
            int(stay_id): float(label)
            for stay_id, label in self.outcomes[
                ["ts_id", "in_hospital_mortality"]
            ].itertuples(index=False, name=None)
        }
        missing_labels = [int(stay_id) for stay_id in cohort_ids if int(stay_id) not in self.labels]
        if missing_labels:
            raise ValueError(
                f"Missing mortality labels for {len(missing_labels)} stays; "
                f"first={missing_labels[:5]}"
            )

        cohort_mask = all_data["ts_id"].isin(cohort_ids)
        variable_series = all_data["variable"]

        demo_mask = cohort_mask & variable_series.isin(self.demo_names)
        demo_frame = all_data.loc[
            demo_mask, ["ts_id", "variable", "value"]
        ].copy()
        demo_frame["value"] = pd.to_numeric(demo_frame["value"], errors="coerce")
        age_mask = (demo_frame["variable"] == "Age") & (demo_frame["value"] > 200)
        demo_frame.loc[age_mask, "value"] = 91.4
        demo_table = demo_frame.groupby(["ts_id", "variable"])["value"].mean().unstack()
        self.demo = {}
        for stay_id in cohort_ids:
            vector = np.zeros(self.num_demo, dtype=np.float32)
            if stay_id in demo_table.index:
                for index, name in enumerate(self.demo_names):
                    if name not in demo_table.columns:
                        continue
                    value = demo_table.at[stay_id, name]
                    if pd.isna(value):
                        continue
                    mean, std = self._mean_std(name)
                    vector[index] = (float(value) - mean) / std
            self.demo[int(stay_id)] = vector
        del demo_frame, demo_table

        minute_series = pd.to_numeric(all_data["minute"], errors="coerce")
        value_series = pd.to_numeric(all_data["value"], errors="coerce")
        event_mask = (
            cohort_mask
            & variable_series.isin(self.feature_names)
            & minute_series.between(0, self.max_minute, inclusive="both")
            & value_series.notna()
        )
        event_frame = all_data.loc[
            event_mask, ["ts_id", "variable", "minute", "value"]
        ].copy()
        event_frame["var_idx"] = event_frame["variable"].map(self.feature_map)
        event_frame = event_frame.dropna(subset=["var_idx", "minute", "value"])

        ts_ids = event_frame["ts_id"].to_numpy(dtype=np.int64, copy=True)
        minutes = event_frame["minute"].to_numpy(dtype=np.float32, copy=True)
        var_indices = event_frame["var_idx"].to_numpy(dtype=np.int32, copy=True)
        raw_values = event_frame["value"].to_numpy(dtype=np.float32, copy=True)

        feature_means = np.asarray(
            [self._mean_std(name)[0] for name in self.feature_names], dtype=np.float32
        )
        feature_stds = np.asarray(
            [self._mean_std(name)[1] for name in self.feature_names], dtype=np.float32
        )
        values = (raw_values - feature_means[var_indices]) / feature_stds[var_indices]
        values = values.astype(np.float32, copy=False)

        # A stable sort preserves source row order for repeated measurements;
        # assignment in __getitem__ then has the same last-observation behavior
        # as the original STraTS preprocessing loop.
        order = np.argsort(ts_ids, kind="stable")
        self._event_ids = ts_ids[order]
        self._minutes = minutes[order]
        self._var_indices = var_indices[order]
        self._values = values[order]
        unique_ids, starts, counts = np.unique(
            self._event_ids, return_index=True, return_counts=True
        )
        self._spans = {
            int(stay_id): (int(start), int(start + count))
            for stay_id, start, count in zip(unique_ids, starts, counts)
        }

        del all_data, event_frame, outcomes
        del cohort_mask, variable_series, minute_series, value_series, event_mask
        del ts_ids, minutes, var_indices, raw_values, values, order
        gc.collect()

    def _mean_std(self, name):
        stats = self.var_stats[name]
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std == 0:
            std = 1.0
        return mean, std

    def events_for(self, ts_id):
        span = self._spans.get(int(ts_id))
        if span is None:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        start, stop = span
        return (
            self._minutes[start:stop],
            self._var_indices[start:stop],
            self._values[start:stop],
        )


class MIMICTimeseriesDataset(Dataset):
    """A fold view producing inputs for one of the four baseline models."""

    def __init__(
        self,
        store: MIMICTimeseriesDataStore,
        tsid_list: Iterable[int],
        model_type: str,
        time_step_minutes: float = 60,
        max_timesteps: int = 880,
        truncation_seed: Optional[int] = None,
    ):
        model_type = model_type.lower()
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {sorted(_SUPPORTED_MODELS)}, got {model_type!r}"
            )
        if time_step_minutes <= 0:
            raise ValueError("time_step_minutes must be positive")
        if max_timesteps < 1:
            raise ValueError("max_timesteps must be positive")
        self.store = store
        self.ids = np.asarray(list(tsid_list), dtype=np.int64)
        self.model_type = model_type
        self.time_step_minutes = float(time_step_minutes)
        self.max_timesteps = int(max_timesteps)
        self.truncation_seed = truncation_seed
        self.num_features = store.num_features
        self.num_demo = store.num_demo
        self.num_steps = int(np.ceil(store.max_minute / self.time_step_minutes))
        if self.num_steps < 1:
            raise ValueError("The fixed time grid contains no steps")
        missing = [int(stay_id) for stay_id in self.ids if int(stay_id) not in store.labels]
        if missing:
            raise KeyError(f"Unknown ICU stays; first={missing[:5]}")

    def __len__(self):
        return len(self.ids)

    def labels_array(self):
        return np.asarray(
            [self.store.labels[int(stay_id)] for stay_id in self.ids], dtype=np.int64
        )

    def __getitem__(self, index):
        ts_id = int(self.ids[index])
        minutes, var_indices, values = self.store.events_for(ts_id)
        demo = torch.from_numpy(self.store.demo[ts_id].copy())
        label = torch.tensor(self.store.labels[ts_id], dtype=torch.float32)
        if self.model_type in _DENSE_MODELS:
            return self._dense_item(minutes, var_indices, values, demo, label)
        return self._grud_item(ts_id, minutes, var_indices, values, demo, label)

    def _dense_item(self, minutes, var_indices, values, demo, label):
        value_grid = np.zeros(
            (self.num_steps, self.num_features), dtype=np.float32
        )
        mask_grid = np.zeros_like(value_grid)
        if len(minutes):
            bins = np.maximum(1, np.ceil(minutes / self.time_step_minutes)).astype(
                np.int64
            ) - 1
            valid = (bins >= 0) & (bins < self.num_steps)
            value_grid[bins[valid], var_indices[valid]] = values[valid]
            mask_grid[bins[valid], var_indices[valid]] = 1.0

        delta_grid = np.zeros_like(value_grid)
        delta_grid[0] = mask_grid[0]
        for timestep in range(1, self.num_steps):
            delta_grid[timestep] = (1 - mask_grid[timestep]) * (
                1 + delta_grid[timestep - 1]
            )
        delta_grid /= float(self.num_steps)
        ts = np.concatenate((value_grid, mask_grid, delta_grid), axis=-1)
        return {"ts": torch.from_numpy(ts), "demo": demo, "labels": label}

    def _grud_item(self, ts_id, minutes, var_indices, values, demo, label):
        unique_times = np.unique(minutes)
        if len(unique_times) > self.max_timesteps:
            if self.truncation_seed is None:
                unique_times = np.sort(
                    np.random.choice(
                        unique_times, size=self.max_timesteps, replace=False
                    )
                )
            else:
                seed = (int(self.truncation_seed) + int(ts_id)) % (2**32)
                rng = np.random.default_rng(seed)
                unique_times = np.sort(
                    rng.choice(unique_times, size=self.max_timesteps, replace=False)
                )

        if len(unique_times) == 0:
            unique_times = np.asarray([0.0], dtype=np.float32)
            selected_minutes = np.empty(0, dtype=np.float32)
            selected_varis = np.empty(0, dtype=np.int32)
            selected_values = np.empty(0, dtype=np.float32)
        else:
            selected = np.isin(minutes, unique_times)
            selected_minutes = minutes[selected]
            selected_varis = var_indices[selected]
            selected_values = values[selected]

        length = len(unique_times)
        x_t = np.zeros((length, self.num_features), dtype=np.float32)
        m_t = np.zeros_like(x_t)
        if len(selected_minutes):
            time_indices = np.searchsorted(unique_times, selected_minutes)
            x_t[time_indices, selected_varis] = selected_values
            m_t[time_indices, selected_varis] = 1.0

        delta_t = np.zeros_like(x_t)
        for timestep in range(1, length):
            elapsed = float(unique_times[timestep] - unique_times[timestep - 1])
            delta_t[timestep] = elapsed + (1 - m_t[timestep - 1]) * delta_t[
                timestep - 1
            ]
        # Preserve the reference implementation's minute-to-day scaling.
        delta_t /= self.store.total_hours * 60.0 * 60.0
        return {
            "x_t": torch.from_numpy(x_t),
            "m_t": torch.from_numpy(m_t),
            "delta_t": torch.from_numpy(delta_t),
            "seq_len": torch.tensor(length, dtype=torch.long),
            "demo": demo,
            "labels": label,
        }


def collate_grud(batch):
    """Pad irregular GRU-D samples to the longest sequence in the batch."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(int(item["seq_len"]) for item in batch)
    num_features = batch[0]["x_t"].size(1)
    batch_size = len(batch)
    x_t = torch.zeros((batch_size, max_length, num_features), dtype=torch.float32)
    m_t = torch.zeros_like(x_t)
    delta_t = torch.zeros_like(x_t)
    for index, item in enumerate(batch):
        length = int(item["seq_len"])
        x_t[index, :length] = item["x_t"]
        m_t[index, :length] = item["m_t"]
        delta_t[index, :length] = item["delta_t"]
    return {
        "x_t": x_t,
        "m_t": m_t,
        "delta_t": delta_t,
        "seq_len": torch.stack([item["seq_len"] for item in batch]),
        "demo": torch.stack([item["demo"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
