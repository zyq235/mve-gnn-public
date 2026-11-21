"""Extract 24-hour remaining-ICU-LOS labels for the selected cohorts.

The script uses the existing train/validation/test ICU-stay lists rather than
the full source database.  It writes one harmonised CSV into each dataset's
directory, for example ``data/eicu/los_24h_labels.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.los_patient_dataset import LOS_CLASS_NAMES, load_los_target_frame


DATASET_CONFIGS = {
    "mimic3": {
        "data_dir": PROJECT_ROOT / "data" / "mimic3",
        "pickle_name": "mimic_iii.pkl",
        "raw_data_path": Path(
            r"E:\zyq\mimic\mimiciii\mimic-iii-clinical-database-1.4"
        ),
    },
    "mimic4": {
        "data_dir": PROJECT_ROOT / "data" / "mimic-iv",
        "pickle_name": "mimic_iv.pkl",
        "raw_data_path": PROJECT_ROOT / "data" / "mimic-iv" / "mimic-iv-3.1",
    },
    "eicu": {
        "data_dir": PROJECT_ROOT / "data" / "eicu",
        "pickle_name": "eicu.pkl",
        "raw_data_path": Path(
            r"E:\zyq\mimic\eicu-collaborative-research-database-2.0"
        ),
    },
}


def _validate_split_ids(train_ids, val_ids, test_ids):
    arrays = tuple(
        np.asarray(values, dtype=np.int64).reshape(-1)
        for values in (train_ids, val_ids, test_ids)
    )
    names = ("train", "val", "test")
    for name, values in zip(names, arrays):
        if len(values) == 0:
            raise ValueError(f"The {name} split is empty")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"The {name} split contains duplicate ICU stays")
    split_sets = [set(values.tolist()) for values in arrays]
    if split_sets[0] & split_sets[1]:
        raise ValueError("ICU-stay overlap between train and validation")
    if split_sets[0] & split_sets[2]:
        raise ValueError("ICU-stay overlap between train and test")
    if split_sets[1] & split_sets[2]:
        raise ValueError("ICU-stay overlap between validation and test")
    return arrays


def load_cohort_splits(dataset_name: str, data_dir: Path, pickle_path: Path):
    """Load existing cohort membership, preferring lightweight sidecars."""

    if dataset_name == "mimic3":
        sidecar = data_dir / "mimic_iii_split_ids.pkl"
        if sidecar.exists():
            with sidecar.open("rb") as handle:
                payload = pickle.load(handle)
            return _validate_split_ids(
                payload["train_list"], payload["val_list"], payload["test_list"]
            )

    if dataset_name == "eicu":
        sidecar = data_dir / "patient_split.csv"
        if sidecar.exists():
            split_frame = pd.read_csv(sidecar, usecols=["ts_id", "split"])
            split_names = split_frame["split"].astype(str).str.strip().str.lower()
            return _validate_split_ids(
                split_frame.loc[split_names.eq("train"), "ts_id"],
                split_frame.loc[
                    split_names.isin(("val", "valid", "validation")), "ts_id"
                ],
                split_frame.loc[split_names.eq("test"), "ts_id"],
            )

    if not pickle_path.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {pickle_path}")
    # MIMIC-IV currently has no split sidecar.  Only the three ID arrays are
    # retained after this one-time load; the large event/outcome objects are
    # released when the function returns.
    with pickle_path.open("rb") as handle:
        _, _, train_ids, val_ids, test_ids = pickle.load(handle)
    return _validate_split_ids(train_ids, val_ids, test_ids)


def build_los_label_frame(
    dataset_name: str,
    raw_data_path: Path,
    train_ids,
    val_ids,
    test_ids,
    landmark_hours: int = 24,
) -> pd.DataFrame:
    """Return harmonised labels in original train/val/test order."""

    train_ids, val_ids, test_ids = _validate_split_ids(
        train_ids, val_ids, test_ids
    )
    cohort_ids = np.concatenate([train_ids, val_ids, test_ids])
    targets = load_los_target_frame(
        dataset_name,
        raw_data_path,
        cohort_ids,
        landmark_hours=landmark_hours,
    )
    split_by_id = {
        **{int(stay_id): "train" for stay_id in train_ids},
        **{int(stay_id): "val" for stay_id in val_ids},
        **{int(stay_id): "test" for stay_id in test_ids},
    }
    targets.insert(0, "dataset", dataset_name)
    targets.insert(1, "split", targets["ts_id"].map(split_by_id))
    targets.insert(4, "landmark_hours", int(landmark_hours))
    targets.insert(
        5,
        "icu_los_days",
        targets["remaining_los_days"] + landmark_hours / 24.0,
    )
    targets.insert(
        7,
        "remaining_los_minutes",
        targets["remaining_los_days"] * 1440.0,
    )
    targets.insert(
        9,
        "los_class_name",
        targets["los_class"].map(dict(enumerate(LOS_CLASS_NAMES))),
    )
    if targets["split"].isna().any():
        raise RuntimeError("At least one LOS label has no split assignment")
    return targets


def _atomic_to_csv(frame: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_summary(frame: pd.DataFrame, output_path: Path, source_paths: dict):
    class_counts = (
        frame.groupby(["split", "los_class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(len(LOS_CLASS_NAMES)), fill_value=0)
    )
    payload = {
        "dataset": str(frame["dataset"].iloc[0]),
        "label_file": str(output_path),
        "landmark_hours": int(frame["landmark_hours"].iloc[0]),
        "number_of_stays": int(len(frame)),
        "split_counts": {
            str(key): int(value) for key, value in frame["split"].value_counts().items()
        },
        "class_names": list(LOS_CLASS_NAMES),
        "class_counts_by_split": {
            split: {
                str(class_index): int(count)
                for class_index, count in counts.items()
            }
            for split, counts in class_counts.to_dict(orient="index").items()
        },
        "source_paths": {key: str(value) for key, value in source_paths.items()},
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_path, summary_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def extract_one_dataset(dataset_name: str, args) -> Path:
    config = DATASET_CONFIGS[dataset_name]
    data_dir = Path(
        getattr(args, f"{dataset_name}_data_dir") or config["data_dir"]
    )
    raw_data_path = Path(
        getattr(args, f"{dataset_name}_raw_data_path")
        or config["raw_data_path"]
    )
    pickle_path = data_dir / config["pickle_name"]
    output_path = data_dir / f"los_{args.landmark_hours}h_labels.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw {dataset_name} directory not found: {raw_data_path}")

    train_ids, val_ids, test_ids = load_cohort_splits(
        dataset_name, data_dir, pickle_path
    )
    labels = build_los_label_frame(
        dataset_name,
        raw_data_path,
        train_ids,
        val_ids,
        test_ids,
        landmark_hours=args.landmark_hours,
    )
    _atomic_to_csv(labels, output_path)
    _write_summary(
        labels,
        output_path,
        {
            "data_dir": data_dir,
            "raw_data_path": raw_data_path,
            "split_pickle": pickle_path,
        },
    )
    counts = labels["split"].value_counts()
    print(
        f"[{dataset_name}] wrote {len(labels):,} labels to {output_path} "
        f"(train={counts.get('train', 0):,}, val={counts.get('val', 0):,}, "
        f"test={counts.get('test', 0):,})"
    )
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract harmonised remaining-ICU-LOS labels for the selected "
            "train+validation+test cohorts"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASET_CONFIGS.keys()),
        default="all",
        help="Generate one dataset or all three datasets.",
    )
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    for dataset_name in DATASET_CONFIGS:
        parser.add_argument(f"--{dataset_name}-data-dir")
        parser.add_argument(f"--{dataset_name}-raw-data-path")
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)
    if parsed.landmark_hours <= 0:
        raise ValueError("landmark_hours must be positive")
    selected = (
        DATASET_CONFIGS.keys()
        if parsed.dataset == "all"
        else (parsed.dataset,)
    )
    return [extract_one_dataset(dataset_name, parsed) for dataset_name in selected]


if __name__ == "__main__":
    main()
