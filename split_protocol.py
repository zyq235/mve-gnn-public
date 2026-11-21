"""Leakage-safe split helpers shared by repeated-subsampling experiments."""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np


def get_sliding_window_splits(
    ids: Sequence[int],
    id_to_patient: Mapping[int, Hashable],
    n_splits: int = 6,
    window_size: int = 3,
    seed: int = 443,
):
    """Create cyclic train/validation splits without separating patient stays.

    Patients are shuffled once with a fixed seed, divided into ``n_splits``
    chunks, and then expanded back to ICU-stay IDs.  For run ``i``, the next
    ``window_size`` chunks are training data and the following chunk is the
    validation data.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 1 <= window_size < n_splits:
        raise ValueError("window_size must be in [1, n_splits - 1]")

    stay_ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(stay_ids)) != len(stay_ids):
        raise ValueError("Duplicate ICU-stay IDs were supplied")
    missing = [int(stay_id) for stay_id in stay_ids if int(stay_id) not in id_to_patient]
    if missing:
        raise KeyError(f"Missing patient IDs for {len(missing)} ICU stays; first={missing[:5]}")

    stay_patients = np.asarray(
        [id_to_patient[int(stay_id)] for stay_id in stay_ids], dtype=object
    )
    # dict preserves first-observed order and supports string or numeric IDs.
    patients = np.asarray(list(dict.fromkeys(stay_patients.tolist())), dtype=object)
    if len(patients) < n_splits:
        raise ValueError("Fewer patients than requested chunks")
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    patient_chunks = np.array_split(patients, n_splits)
    stay_chunks = [stay_ids[np.isin(stay_patients, chunk)] for chunk in patient_chunks]

    splits = []
    for index in range(n_splits):
        train_chunk_indices = [
            (index + offset) % n_splits for offset in range(window_size)
        ]
        val_chunk_index = (index + window_size) % n_splits
        train_ids = np.concatenate([stay_chunks[i] for i in train_chunk_indices])
        val_ids = stay_chunks[val_chunk_index]
        assert_split_integrity(train_ids, val_ids, (), id_to_patient)
        splits.append((train_ids.tolist(), val_ids.tolist()))
    return splits


def assert_split_integrity(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    id_to_patient: Mapping[int, Hashable],
) -> None:
    """Raise if ICU stays or patients overlap within one experiment."""

    stay_sets = [set(map(int, values)) for values in (train_ids, val_ids, test_ids)]
    if stay_sets[0] & stay_sets[1]:
        raise AssertionError("ICU-stay overlap between train and validation")
    if stay_sets[0] & stay_sets[2]:
        raise AssertionError("ICU-stay overlap between train and test")
    if stay_sets[1] & stay_sets[2]:
        raise AssertionError("ICU-stay overlap between validation and test")

    patient_sets = []
    for stay_set in stay_sets:
        missing = [stay_id for stay_id in stay_set if stay_id not in id_to_patient]
        if missing:
            raise KeyError(
                f"Missing patient IDs during overlap audit; first={missing[:5]}"
            )
        patient_sets.append({id_to_patient[stay_id] for stay_id in stay_set})
    if patient_sets[0] & patient_sets[1]:
        raise AssertionError("Patient overlap between train and validation")
    if patient_sets[0] & patient_sets[2]:
        raise AssertionError("Patient overlap between train and test")
    if patient_sets[1] & patient_sets[2]:
        raise AssertionError("Patient overlap between validation and test")


def patient_mapping_from_outcomes(outcomes) -> dict:
    """Extract ``ts_id -> patient`` from a MIMIC/eICU outcome DataFrame."""

    if not hasattr(outcomes, "columns"):
        raise TypeError(
            "Patient-level splitting requires an outcome DataFrame with a patient ID column"
        )
    patient_column = next(
        (
            name
            for name in ("SUBJECT_ID", "subject_id", "patient_id", "uniquepid")
            if name in outcomes.columns
        ),
        None,
    )
    if patient_column is None or "ts_id" not in outcomes.columns:
        raise ValueError(
            "Outcomes must contain ts_id and one of SUBJECT_ID/subject_id/patient_id/uniquepid"
        )
    pairs = outcomes[["ts_id", patient_column]].dropna().drop_duplicates()
    conflicting = pairs.groupby("ts_id")[patient_column].nunique()
    if conflicting.gt(1).any():
        raise ValueError("At least one ICU stay maps to multiple patients")
    return {
        int(stay_id): patient_id
        for stay_id, patient_id in pairs.itertuples(index=False, name=None)
    }


def patient_mapping_from_csv(
    csv_path: str,
    stay_column: str = "stay_id",
    patient_column: str = "subject_id",
) -> dict:
    """Load a stay-to-patient map without loading the full clinical dataset."""

    import pandas as pd

    pairs = pd.read_csv(csv_path, usecols=[stay_column, patient_column])
    # read_csv keeps source-file order rather than the order of ``usecols``.
    pairs = pairs[[stay_column, patient_column]].dropna().drop_duplicates()
    conflicting = pairs.groupby(stay_column)[patient_column].nunique()
    if conflicting.gt(1).any():
        raise ValueError("At least one ICU stay maps to multiple patients")
    return {
        int(stay_id): patient_id
        for stay_id, patient_id in pairs.itertuples(index=False, name=None)
    }
