"""Add SUBJECT_ID to a legacy MIMIC-IV pickle without rerunning event preprocessing.

The source pickle is never overwritten by default.  ``--replace`` keeps the
original as ``mimic_iv.before_subject_id.pkl`` after validating the new file.
The raw icustays table is needed only for this one-time migration.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "mimic-iv" / "mimic_iv.pkl"
DEFAULT_ICUSTAYS = (
    PROJECT_ROOT / "data" / "mimic-iv" / "mimic-iv-3.1" / "icu" / "icustays.csv.gz"
)


def _patient_lookup(icustays_path: Path) -> pd.Series:
    stays = pd.read_csv(icustays_path, usecols=["stay_id", "subject_id"])
    if stays.empty or stays.isna().any().any():
        raise ValueError("icustays.csv.gz has empty or missing stay/patient IDs")
    stays = stays.drop_duplicates()
    if stays["stay_id"].duplicated().any():
        raise ValueError("One MIMIC-IV stay_id maps to multiple subject_id values")
    return stays.set_index("stay_id")["subject_id"]


def _verify_pickle(
    path: Path,
    expected_ids: tuple[tuple[int, ...], ...],
    expected_patients: dict[int, int],
) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 5:
        raise ValueError(f"Written pickle has an invalid five-object payload: {path}")
    outcomes = payload[1]
    if not {"ts_id", "SUBJECT_ID", "in_hospital_mortality"}.issubset(outcomes.columns):
        raise ValueError(f"Written pickle has no complete outcomes table: {path}")
    actual_ids = tuple(tuple(map(int, ids)) for ids in payload[2:])
    if actual_ids != expected_ids:
        raise ValueError(f"Written pickle changed the train/val/test stay IDs: {path}")
    actual_patients = dict(zip(map(int, outcomes["ts_id"]), map(int, outcomes["SUBJECT_ID"])))
    if actual_patients != expected_patients:
        raise ValueError(f"Written pickle changed the stay-to-patient mapping: {path}")


def add_subject_id(
    input_path: Path,
    icustays_path: Path,
    *,
    output_path: Path | None = None,
    replace: bool = False,
) -> Path:
    input_path = input_path.resolve()
    icustays_path = icustays_path.resolve()
    if not input_path.is_file() or not icustays_path.is_file():
        raise FileNotFoundError("Both the source pickle and icustays.csv.gz must exist")
    if replace and output_path is not None:
        raise ValueError("Use either --replace or --output, not both")

    destination = (
        input_path if replace else
        (output_path or input_path.with_name("mimic_iv_with_subject_id.pkl")).resolve()
    )
    backup_path = input_path.with_name(
        f"{input_path.stem}.before_subject_id{input_path.suffix}"
    )
    if destination.exists() and destination != input_path:
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination == input_path and not replace:
        raise ValueError("Use --replace to update the source pickle with a backup")
    destination.parent.mkdir(parents=True, exist_ok=True)

    lookup = _patient_lookup(icustays_path)
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 5:
        raise ValueError("Expected [events, outcomes, train_ids, val_ids, test_ids]")
    events, outcomes, train_ids, val_ids, test_ids = payload
    if not isinstance(outcomes, pd.DataFrame):
        raise TypeError("MIMIC-IV outcomes must be a pandas DataFrame")
    required = {"ts_id", "in_hospital_mortality"}
    if not required.issubset(outcomes.columns):
        raise ValueError(f"MIMIC-IV outcomes lack columns: {sorted(required - set(outcomes.columns))}")
    if outcomes["ts_id"].isna().any() or outcomes["ts_id"].duplicated().any():
        raise ValueError("MIMIC-IV outcomes have missing or duplicate ts_id values")

    patient_ids = outcomes["ts_id"].map(lookup)
    if patient_ids.isna().any():
        missing = outcomes.loc[patient_ids.isna(), "ts_id"].head(5).tolist()
        raise ValueError(f"Missing subject_id for outcomes stays, first={missing}")
    already_complete = (
        "SUBJECT_ID" in outcomes.columns and outcomes["SUBJECT_ID"].notna().all()
    )
    if "SUBJECT_ID" in outcomes.columns:
        existing = outcomes["SUBJECT_ID"]
        known = existing.notna()
        if known.any() and not (
            existing.loc[known].astype("int64").to_numpy()
            == patient_ids.loc[known].astype("int64").to_numpy()
        ).all():
            raise ValueError("Existing SUBJECT_ID values disagree with icustays.csv.gz")
    outcomes = outcomes.copy()
    outcomes["SUBJECT_ID"] = patient_ids.astype("int64")

    split_ids = tuple(tuple(map(int, ids)) for ids in (train_ids, val_ids, test_ids))
    all_ids = set().union(*(set(ids) for ids in split_ids))
    missing_outcomes = all_ids - set(map(int, outcomes["ts_id"]))
    if missing_outcomes:
        raise ValueError(f"Split stays missing from outcomes, first={sorted(missing_outcomes)[:5]}")
    by_stay = dict(zip(map(int, outcomes["ts_id"]), map(int, outcomes["SUBJECT_ID"])))
    split_patient_sets = []
    for ids in split_ids:
        if len(ids) != len(set(ids)):
            raise ValueError("A train/val/test split contains duplicate stay IDs")
        split_patient_sets.append({by_stay[stay_id] for stay_id in ids})
    if any(
        set(split_ids[left]) & set(split_ids[right])
        or split_patient_sets[left] & split_patient_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Patient or ICU-stay overlap between train/val/test splits")
    if replace and already_complete:
        print(f"Already contains verified SUBJECT_ID values: {input_path}")
        return input_path
    if replace and backup_path.exists():
        raise FileExistsError(
            f"Backup already exists but input still needs migration: {backup_path}"
        )
    if shutil.disk_usage(destination.parent).free < input_path.stat().st_size:
        raise OSError("Not enough free disk space for a second copy of the pickle")
    updated = [events, outcomes, train_ids, val_ids, test_ids]
    if isinstance(payload, tuple):
        updated = tuple(updated)
    del payload, events, outcomes, lookup, patient_ids
    gc.collect()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".mimic_iv_subject_id_", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(updated, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        del updated
        gc.collect()
        _verify_pickle(temporary_path, split_ids, by_stay)

        if replace:
            os.replace(input_path, backup_path)
            try:
                os.replace(temporary_path, input_path)
            except BaseException:
                os.replace(backup_path, input_path)
                raise
            destination = input_path
        else:
            os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--icustays", type=Path, default=DEFAULT_ICUSTAYS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = add_subject_id(
        args.input, args.icustays, output_path=args.output, replace=args.replace
    )
    print(f"Verified MIMIC-IV pickle with SUBJECT_ID: {result}")
    if args.replace:
        input_path = args.input.resolve()
        print(
            "Original pickle backup: "
            f"{input_path.with_name(f'{input_path.stem}.before_subject_id{input_path.suffix}')}"
        )


if __name__ == "__main__":
    main()
