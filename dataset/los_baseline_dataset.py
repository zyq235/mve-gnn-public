"""First-24-hour inputs for the main-table LOS baselines.

The mortality datasets retain their original targets.  LOS targets are joined
by ICU-stay ID from the precomputed label CSV when a sample is requested.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.mimic_timeseries_baseline_dataset import (
    MIMICTimeseriesDataStore,
    MIMICTimeseriesDataset,
    collate_grud,
)


class LOSDataStore(MIMICTimeseriesDataStore):
    def events_for(self, ts_id):
        minutes, variables, values = super().events_for(ts_id)
        first_day = minutes < self.max_minute
        return minutes[first_day], variables[first_day], values[first_day]


class LOSBaselineDataset(Dataset):
    def __init__(
        self, store, stay_ids, labels, model_type, *, slot_minutes=60,
        scales=(60, 120, 240), max_timesteps=880, truncation_seed=None,
    ):
        self.store = store
        self.ids = tuple(map(int, stay_ids))
        self.labels = labels
        self.model_type = model_type
        self.slot_minutes = int(slot_minutes)
        self.scales = tuple(map(int, scales))
        self.num_features = store.num_features
        self.num_demo = store.num_demo
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Each LOS dataset split needs unique ICU-stay IDs")
        missing = set(self.ids) - set(labels)
        if missing:
            raise ValueError(f"LOS labels missing for {len(missing)} ICU stays")
        if model_type in {"gru", "grud", "sand", "tcn"}:
            self.sequence = MIMICTimeseriesDataset(
                store, self.ids, model_type,
                time_step_minutes=slot_minutes,
                max_timesteps=max_timesteps,
                truncation_seed=truncation_seed,
            )
        else:
            self.sequence = None

    def __len__(self):
        return len(self.ids)

    def _matrix(self, stay_id, slot_minutes, *, forward_fill):
        slots = int(self.store.max_minute // slot_minutes)
        values_out = np.zeros((self.num_features, slots), dtype=np.float32)
        mask = np.zeros((self.num_features, slots), dtype=np.float32)
        minutes, variables, values = self.store.events_for(stay_id)
        if len(minutes):
            bins = np.floor(minutes / slot_minutes).astype(np.int64)
            valid = (bins >= 0) & (bins < slots)
            flat_index = variables[valid].astype(np.int64) * slots + bins[valid]
            totals = np.zeros(self.num_features * slots, dtype=np.float64)
            counts = np.zeros(self.num_features * slots, dtype=np.int64)
            np.add.at(totals, flat_index, values[valid])
            np.add.at(counts, flat_index, 1)
            observed = counts > 0
            values_out.flat[observed] = (totals[observed] / counts[observed]).astype(np.float32)
            mask.flat[observed] = 1.0
        if forward_fill:
            last = np.maximum.accumulate(
                np.where(mask > 0, np.arange(slots)[None, :], 0), axis=1
            )
            values_out = np.take_along_axis(values_out, last, axis=1)
        return torch.from_numpy(values_out), torch.from_numpy(mask)

    def __getitem__(self, index):
        stay_id = self.ids[index]
        label = torch.tensor(int(self.labels[stay_id]), dtype=torch.long)
        if self.sequence is not None:
            sample = self.sequence[index]
            return {**sample, "stay_id": stay_id, "labels": label}
        demo = torch.from_numpy(self.store.demo[stay_id].copy())
        if self.model_type == "mstdgnn_multiscale":
            return {
                "stay_id": stay_id, "matrices": [
                    self._matrix(stay_id, scale, forward_fill=True)[0]
                    for scale in self.scales
                ],
                "demo": demo, "labels": label,
            }
        if self.model_type == "raindrop":
            values, mask = self._matrix(stay_id, self.slot_minutes, forward_fill=False)
            slots = values.size(1)
            return {
                "stay_id": stay_id, "matrix": values, "mask": mask,
                "times": torch.arange(slots, dtype=torch.float32) / slots,
                "demo": demo, "labels": label,
            }
        matrix, _ = self._matrix(stay_id, self.slot_minutes, forward_fill=True)
        return {"stay_id": stay_id, "matrix": matrix, "demo": demo, "labels": label}


def collate_los_grud(samples):
    batch = collate_grud(samples)
    batch["stay_id"] = torch.tensor(
        [sample["stay_id"] for sample in samples], dtype=torch.long
    )
    return batch
