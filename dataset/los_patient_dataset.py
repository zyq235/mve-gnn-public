"""LOS labels and a task-specific adapter for cached ICU graphs.

Existing graph files contain mortality labels because they were produced for
the original task.  ``LOSLargeGraphDataset`` deliberately replaces ``Data.y``
at read time, allowing the graph cache and all mortality entry points to remain
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from dataset.mimic3_patient_dataset import ICULargeGraphDataset


LOS_CLASS_UPPER_BOUNDS_DAYS = np.asarray(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 14.0], dtype=np.float64
)
LOS_CLASS_NAMES = (
    "<1d",
    "1-2d",
    "2-3d",
    "3-4d",
    "4-5d",
    "5-6d",
    "6-7d",
    "7-8d",
    "8-14d",
    ">=14d",
)


class LOSData(Data):
    """PyG data object with a stable semantic-type batching offset.

    PyG otherwise increments keys containing ``batch`` by the observed maximum
    in each graph.  A stay that happens not to contain the final semantic type
    would then shift the next stay by too little and mix patients' type groups.
    This task adapter fixes that behaviour without changing existing caches or
    the mortality dataset.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "batch_type":
            return int(self.num_type_classes)
        return super().__inc__(key, value, *args, **kwargs)


def remaining_days_to_class(values) -> np.ndarray:
    """Map non-negative remaining ICU LOS in days to the benchmark bins."""

    remaining = np.asarray(values, dtype=np.float64)
    if not np.isfinite(remaining).all():
        raise ValueError("Remaining LOS contains non-finite values")
    if (remaining < 0).any():
        raise ValueError("Remaining LOS must be non-negative at the landmark")
    return np.searchsorted(
        LOS_CLASS_UPPER_BOUNDS_DAYS, remaining, side="right"
    ).astype(np.int64)


def _normalise_cohort_ids(cohort_ids: Iterable[int]) -> np.ndarray:
    ids = np.asarray(list(cohort_ids), dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("cohort_ids must be a non-empty one-dimensional sequence")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("cohort_ids contains duplicate ICU stays")
    return ids


def _finalise_targets(frame: pd.DataFrame, cohort_ids: np.ndarray) -> pd.DataFrame:
    required = {
        "ts_id",
        "patient_id",
        "remaining_los_days",
        "icu_death",
        "in_hospital_mortality",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"LOS target frame is missing columns: {sorted(missing_columns)}")

    frame = frame.copy()
    frame["ts_id"] = pd.to_numeric(frame["ts_id"], errors="raise").astype(np.int64)
    frame = frame.drop_duplicates("ts_id", keep="first").set_index("ts_id")

    missing_ids = sorted(set(map(int, cohort_ids)) - set(map(int, frame.index)))
    if missing_ids:
        raise ValueError(
            f"Raw ICU tables are missing {len(missing_ids)} cohort stays; "
            f"first IDs: {missing_ids[:5]}"
        )

    frame = frame.loc[cohort_ids].reset_index()
    remaining = pd.to_numeric(frame["remaining_los_days"], errors="coerce")
    if remaining.isna().any():
        raise ValueError("Remaining LOS could not be computed for every cohort stay")
    # Small negative values can only arise from timestamp precision.  Cohort
    # membership itself must guarantee a complete 24-hour landmark window.
    if remaining.lt(-1e-8).any():
        bad = frame.loc[remaining.lt(-1e-8), "ts_id"].head().tolist()
        raise ValueError(f"Cohort contains stays shorter than the landmark: {bad}")
    frame["remaining_los_days"] = remaining.clip(lower=0.0).astype(np.float64)
    frame["los_class"] = remaining_days_to_class(frame["remaining_los_days"])
    frame["icu_death"] = frame["icu_death"].astype(np.int8)
    frame["in_hospital_mortality"] = frame["in_hospital_mortality"].astype(np.int8)
    return frame[
        [
            "ts_id",
            "patient_id",
            "remaining_los_days",
            "los_class",
            "icu_death",
            "in_hospital_mortality",
        ]
    ]


def _load_mimic3_targets(
    raw_data_path: Path, cohort_ids: np.ndarray, landmark_hours: int
) -> pd.DataFrame:
    icu = pd.read_csv(
        raw_data_path / "ICUSTAYS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
    ).rename(columns={"SUBJECT_ID": "patient_id", "ICUSTAY_ID": "ts_id"})
    admissions = pd.read_csv(
        raw_data_path / "ADMISSIONS.csv",
        usecols=["HADM_ID", "DEATHTIME", "HOSPITAL_EXPIRE_FLAG"],
    )
    frame = icu.loc[icu["ts_id"].isin(cohort_ids)].merge(
        admissions, on="HADM_ID", how="left", validate="many_to_one"
    )
    frame["INTIME"] = pd.to_datetime(frame["INTIME"], errors="coerce")
    frame["OUTTIME"] = pd.to_datetime(frame["OUTTIME"], errors="coerce")
    frame["DEATHTIME"] = pd.to_datetime(frame["DEATHTIME"], errors="coerce")
    frame["remaining_los_days"] = (
        (frame["OUTTIME"] - frame["INTIME"]).dt.total_seconds() / 86400.0
        - landmark_hours / 24.0
    )
    frame["icu_death"] = (
        frame["DEATHTIME"].notna()
        & frame["DEATHTIME"].ge(frame["INTIME"])
        & frame["DEATHTIME"].le(frame["OUTTIME"])
    ).astype(np.int8)
    frame["in_hospital_mortality"] = frame["HOSPITAL_EXPIRE_FLAG"]
    return _finalise_targets(frame, cohort_ids)


def _load_mimic4_targets(
    raw_data_path: Path, cohort_ids: np.ndarray, landmark_hours: int
) -> pd.DataFrame:
    icu = pd.read_csv(
        raw_data_path / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    ).rename(columns={"subject_id": "patient_id", "stay_id": "ts_id"})
    admissions = pd.read_csv(
        raw_data_path / "hosp" / "admissions.csv.gz",
        usecols=["hadm_id", "deathtime", "hospital_expire_flag"],
    )
    frame = icu.loc[icu["ts_id"].isin(cohort_ids)].merge(
        admissions, on="hadm_id", how="left", validate="many_to_one"
    )
    frame["intime"] = pd.to_datetime(frame["intime"], errors="coerce")
    frame["outtime"] = pd.to_datetime(frame["outtime"], errors="coerce")
    frame["deathtime"] = pd.to_datetime(frame["deathtime"], errors="coerce")
    frame["remaining_los_days"] = (
        (frame["outtime"] - frame["intime"]).dt.total_seconds() / 86400.0
        - landmark_hours / 24.0
    )
    frame["icu_death"] = (
        frame["deathtime"].notna()
        & frame["deathtime"].ge(frame["intime"])
        & frame["deathtime"].le(frame["outtime"])
    ).astype(np.int8)
    frame["in_hospital_mortality"] = frame["hospital_expire_flag"]
    return _finalise_targets(frame, cohort_ids)


def _load_eicu_targets(
    raw_data_path: Path, cohort_ids: np.ndarray, landmark_hours: int
) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_data_path / "patient.csv.gz",
        usecols=[
            "patientunitstayid",
            "uniquepid",
            "unitdischargeoffset",
            "unitdischargestatus",
            "hospitaldischargestatus",
        ],
        low_memory=False,
    ).rename(
        columns={"patientunitstayid": "ts_id", "uniquepid": "patient_id"}
    )
    frame = frame.loc[frame["ts_id"].isin(cohort_ids)].copy()
    frame["remaining_los_days"] = (
        pd.to_numeric(frame["unitdischargeoffset"], errors="coerce")
        - landmark_hours * 60.0
    ) / 1440.0
    unit_status = frame["unitdischargestatus"].astype("string").str.strip().str.lower()
    hospital_status = (
        frame["hospitaldischargestatus"].astype("string").str.strip().str.lower()
    )
    # eICU contains a small number of missing discharge statuses.  Preserve
    # them as -1 in exported metadata instead of silently calling them alive.
    frame["icu_death"] = np.where(
        unit_status.isna(),
        -1,
        unit_status.eq("expired").fillna(False).astype(np.int8),
    )
    frame["in_hospital_mortality"] = np.where(
        hospital_status.isna(),
        -1,
        hospital_status.eq("expired").fillna(False).astype(np.int8),
    )
    return _finalise_targets(frame, cohort_ids)


def load_los_target_frame(
    dataset_name: str,
    raw_data_path,
    cohort_ids: Iterable[int],
    landmark_hours: int = 24,
) -> pd.DataFrame:
    """Build exact and binned remaining-LOS labels from the raw ICU tables."""

    if landmark_hours <= 0:
        raise ValueError("landmark_hours must be positive")
    dataset_name = dataset_name.lower().replace("-", "")
    aliases = {"mimiciii": "mimic3", "mimiciv": "mimic4"}
    dataset_name = aliases.get(dataset_name, dataset_name)
    ids = _normalise_cohort_ids(cohort_ids)
    raw_path = Path(raw_data_path)

    loaders = {
        "mimic3": _load_mimic3_targets,
        "mimic4": _load_mimic4_targets,
        "eicu": _load_eicu_targets,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    return loaders[dataset_name](raw_path, ids, landmark_hours)


def load_los_label_frame(
    label_path,
    cohort_ids: Iterable[int],
    dataset_name: str | None = None,
    landmark_hours: int = 24,
) -> pd.DataFrame:
    """Load and validate labels produced by ``extract_los_labels.py``.

    Training and testing entry points use this function so target construction
    happens only once during preprocessing.  The stored class is checked
    against the exact remaining LOS instead of being trusted silently.
    """

    if landmark_hours <= 0:
        raise ValueError("landmark_hours must be positive")
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(
            f"LOS label file not found: {path}. Run "
            "preprocess/extract_los_labels.py first."
        )
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "ts_id",
        "patient_id",
        "remaining_los_days",
        "los_class",
        "icu_death",
        "in_hospital_mortality",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LOS label file is missing columns: {sorted(missing)}")
    if frame["ts_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["ts_id"].duplicated(), "ts_id"].head().tolist()
        raise ValueError(f"LOS label file contains duplicate ICU stays: {duplicate_ids}")

    if dataset_name is not None and "dataset" in frame.columns:
        expected_dataset = dataset_name.lower().replace("-", "").replace("_", "")
        aliases = {"mimiciii": "mimic3", "mimiciv": "mimic4"}
        expected_dataset = aliases.get(expected_dataset, expected_dataset)
        stored_datasets = {
            aliases.get(value, value)
            for value in frame["dataset"]
            .astype(str)
            .str.lower()
            .str.replace("-", "", regex=False)
            .str.replace("_", "", regex=False)
            .unique()
        }
        if stored_datasets != {expected_dataset}:
            raise ValueError(
                f"LOS label dataset mismatch: expected {expected_dataset}, "
                f"found {sorted(stored_datasets)}"
            )

    if "landmark_hours" in frame.columns:
        stored_landmarks = pd.to_numeric(
            frame["landmark_hours"], errors="coerce"
        )
        if stored_landmarks.isna().any() or not stored_landmarks.eq(
            landmark_hours
        ).all():
            found = sorted(stored_landmarks.dropna().unique().tolist())
            raise ValueError(
                f"LOS label landmark mismatch: expected {landmark_hours}h, "
                f"found {found}"
            )

    ids = _normalise_cohort_ids(cohort_ids)
    stored_class = (
        frame.assign(
            ts_id=pd.to_numeric(frame["ts_id"], errors="raise").astype(np.int64),
            los_class=pd.to_numeric(frame["los_class"], errors="raise").astype(np.int64),
        )
        .set_index("ts_id")["los_class"]
    )
    targets = _finalise_targets(frame, ids)
    expected_class = targets.set_index("ts_id")["los_class"]
    supplied_class = stored_class.loc[expected_class.index]
    mismatched = supplied_class.ne(expected_class)
    if mismatched.any():
        bad_ids = expected_class.index[mismatched].tolist()[:5]
        raise ValueError(
            f"Stored LOS classes disagree with remaining_los_days; first IDs: {bad_ids}"
        )
    return targets


class LOSLargeGraphDataset(ICULargeGraphDataset):
    """Attach LOS targets to the unchanged cached mortality-task graphs."""

    def __init__(self, *args, target_frame: pd.DataFrame, **kwargs):
        required = {
            "ts_id",
            "los_class",
            "remaining_los_days",
            "icu_death",
            "in_hospital_mortality",
        }
        missing = required - set(target_frame.columns)
        if missing:
            raise ValueError(f"target_frame is missing columns: {sorted(missing)}")
        if target_frame["ts_id"].duplicated().any():
            raise ValueError("target_frame contains duplicate ICU stays")

        indexed = target_frame.set_index("ts_id")
        self.los_class_by_tsid: Mapping[int, int] = {
            int(stay_id): int(value)
            for stay_id, value in indexed["los_class"].items()
        }
        self.remaining_days_by_tsid: Mapping[int, float] = {
            int(stay_id): float(value)
            for stay_id, value in indexed["remaining_los_days"].items()
        }
        self.icu_death_by_tsid: Mapping[int, int] = {
            int(stay_id): int(value)
            for stay_id, value in indexed["icu_death"].items()
        }
        self.hospital_mortality_by_tsid: Mapping[int, int] = {
            int(stay_id): int(value)
            for stay_id, value in indexed["in_hospital_mortality"].items()
        }
        super().__init__(*args, **kwargs)

        missing_ids = sorted(self.tsid_set - set(self.los_class_by_tsid))
        if missing_ids:
            raise KeyError(
                f"LOS targets are missing for {len(missing_ids)} requested stays; "
                f"first IDs: {missing_ids[:5]}"
            )

    def get_labels(self):
        if self.time_limit is not None:
            stay_ids = self.active_stay_ids
        else:
            stay_ids = [
                self._parse_graph_filename(name)[0] for name in self.active_files
            ]
        return [self.los_class_by_tsid[int(stay_id)] for stay_id in stay_ids]

    def get(self, idx):
        data = super().get(idx)
        stay_id = int(data.ts_id)

        # MIMIC pickles can contain pre-ICU events.  The LOS task is explicitly
        # based on ICU time [0, landmark), so remove them without changing the
        # original mortality dataset implementation.
        nonnegative = data.x_time.view(-1) >= 0
        if not bool(nonnegative.all()):
            for key in ("x_var", "x_val", "x_time", "x_raw", "batch_time", "batch_type"):
                value = getattr(data, key)
                setattr(data, key, value[nonnegative])
            data.num_nodes = int(nonnegative.sum().item())

        data.y = torch.tensor(
            [self.los_class_by_tsid[stay_id]], dtype=torch.long
        )
        data.remaining_los_days = torch.tensor(
            [self.remaining_days_by_tsid[stay_id]], dtype=torch.float32
        )
        data.icu_death = torch.tensor(
            [self.icu_death_by_tsid[stay_id]], dtype=torch.long
        )
        data.in_hospital_mortality = torch.tensor(
            [self.hospital_mortality_by_tsid[stay_id]], dtype=torch.long
        )
        payload = data.to_dict()
        los_data = LOSData(**payload)
        los_data.num_nodes = int(data.num_nodes)
        los_data.num_type_classes = len(self.class2ind)
        return los_data
