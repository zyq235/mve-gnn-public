"""Train the SAPS-II logistic-regression baseline on MIMIC-III/IV or eICU.

The three SAPS-II CSVs share the component-score contract.  MIMIC-IV may
name its ICU identifier ``stay_id`` while MIMIC-III and the eICU-compatible
extract use ``icustay_id``; the loader resolves either name automatically.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

from evaluator import trapezoidal_pr_auc

from split_protocol import (  # noqa: E402
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_csv,
    patient_mapping_from_outcomes,
)


SCORE_COLUMNS = [
    "age_score",
    "hr_score",
    "sysbp_score",
    "temp_score",
    "pao2fio2_score",
    "uo_score",
    "bun_score",
    "wbc_score",
    "potassium_score",
    "sodium_score",
    "bicarbonate_score",
    "bilirubin_score",
    "gcs_score",
    "comorbidity_score",
    "admissiontype_score",
]

DATASET_CONFIGS = {
    "mimic3": {
        "csv_candidates": [Path("data/mimic3/mm3_sapsii.csv")],
        "pkl_path": Path("data/mimic3/mimic_iii.pkl"),
        "id_candidates": ["icustay_id", "stay_id"],
    },
    "mimic4": {
        "csv_candidates": [
            Path("data/mimic-iv/mm4_sapsii.csv"),
            Path("data/mimic-iv/mimic4_sapsii.csv"),
            Path("data/mimic-iv/sapsii.csv"),
        ],
        "pkl_path": Path("data/mimic-iv/mimic_iv.pkl"),
        "id_candidates": ["stay_id", "icustay_id"],
        "patient_map_path": Path(
            "data/mimic-iv/mimic-iv-3.1/icu/icustays.csv.gz"
        ),
        "patient_map_stay_column": "stay_id",
        "patient_map_patient_column": "subject_id",
    },
    "eicu": {
        "csv_candidates": [Path("data/eicu/eicu_sapsii.csv")],
        "pkl_path": Path("data/eicu/eicu.pkl"),
        "id_candidates": ["icustay_id", "patientunitstayid", "stay_id"],
        "patient_map_path": Path("data/eicu/patient_split.csv"),
        "patient_map_stay_column": "ts_id",
        "patient_map_patient_column": "patient_id",
    },
}

DATASET_ALIASES = {
    "mimic3": "mimic3",
    "mimic-iii": "mimic3",
    "mimic_iii": "mimic3",
    "mimic4": "mimic4",
    "mimic-iv": "mimic4",
    "mimic_iv": "mimic4",
    "eicu": "eicu",
}


def get_precision_at_target_recall(
    y_true: np.ndarray, y_probs: np.ndarray, target_recall: float = 0.8
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    valid_indices = np.where(recall >= target_recall)[0]
    if len(valid_indices) == 0:
        return 0.0, 0.0
    index = valid_indices[-1]
    threshold = thresholds[index] if index < len(thresholds) else 0.0
    return float(precision[index]), float(threshold)


def calculate_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, y_preds: np.ndarray
) -> dict[str, float]:
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0
    accuracy = accuracy_score(y_true, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_preds, average="binary", zero_division=0
    )
    prauc = trapezoidal_pr_auc(y_true, y_probs)
    pr_values, recall_values, _ = precision_recall_curve(y_true, y_probs)
    minrp = np.minimum(pr_values, recall_values).max()
    precision_at_08, _ = get_precision_at_target_recall(y_true, y_probs, 0.8)
    precision_at_09, _ = get_precision_at_target_recall(y_true, y_probs, 0.9)
    return {
        "auc": float(auc),
        "acc": float(accuracy),
        "f1": float(f1),
        "prauc": float(prauc),
        "minrp": float(minrp),
        "recall": float(recall),
        "precision": float(precision),
        "pr08": precision_at_08,
        "pr09": precision_at_09,
    }


def _normalise_stay_id(value: object) -> object:
    if pd.isna(value):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _resolve_existing_path(
    explicit_path: Optional[Path], candidates: Iterable[Path], description: str
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"{description} does not exist: {path}")
        return path
    candidates = [Path(path) for path in candidates]
    for path in candidates:
        if path.exists():
            return path
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}; checked: {rendered}")


def _resolve_id_column(
    columns: Iterable[str], candidates: Iterable[str]
) -> str:
    by_lower = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        "SAPS-II CSV is missing an ICU-stay identifier. Expected one of: "
        + ", ".join(candidates)
    )


def load_feature_table(
    csv_path: Path, id_candidates: Iterable[str]
) -> tuple[dict[object, np.ndarray], str]:
    frame = pd.read_csv(csv_path)
    id_column = _resolve_id_column(frame.columns, id_candidates)
    missing = [column for column in SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"SAPS-II CSV {csv_path} is missing component columns: {missing}"
        )

    frame[id_column] = frame[id_column].map(_normalise_stay_id)
    if frame[id_column].isna().any():
        raise ValueError(f"SAPS-II CSV contains missing values in {id_column}")
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].head().tolist()
        raise ValueError(f"SAPS-II CSV contains duplicate stay IDs: {duplicates}")

    features = frame[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    by_stay = {
        stay_id: row
        for stay_id, row in zip(
            frame[id_column], features.to_numpy(dtype=np.float64), strict=True
        )
    }
    return by_stay, id_column


def load_data(
    dataset: str,
    csv_path: Optional[Path] = None,
    pkl_path: Optional[Path] = None,
    patient_map_path: Optional[Path] = None,
    patient_map_stay_column: Optional[str] = None,
    patient_map_patient_column: Optional[str] = None,
) -> tuple[
    dict[object, np.ndarray],
    dict[object, int],
    dict[object, object],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path,
    Path,
]:
    canonical = DATASET_ALIASES[dataset]
    config = DATASET_CONFIGS[canonical]
    csv_path = _resolve_existing_path(
        csv_path, config["csv_candidates"], f"{canonical} SAPS-II CSV"
    )
    pkl_path = _resolve_existing_path(
        pkl_path, [config["pkl_path"]], f"{canonical} processed PKL"
    )

    feature_by_stay, id_column = load_feature_table(
        csv_path, config["id_candidates"]
    )
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) < 5:
        raise ValueError(
            f"Expected {pkl_path} to contain events, labels and three split lists"
        )
    _, labels, train_ids, validation_ids, test_ids = payload[:5]
    if not isinstance(labels, pd.DataFrame):
        raise TypeError(f"The labels object in {pkl_path} is not a DataFrame")
    required_label_columns = {"ts_id", "in_hospital_mortality"}
    if not required_label_columns.issubset(labels.columns):
        raise ValueError(
            f"Labels in {pkl_path} must contain {sorted(required_label_columns)}"
        )

    label_by_stay = {
        _normalise_stay_id(stay_id): int(label)
        for stay_id, label in zip(
            labels["ts_id"], labels["in_hospital_mortality"], strict=True
        )
    }
    try:
        patient_by_stay = patient_mapping_from_outcomes(labels)
    except (KeyError, ValueError):
        configured_path = config.get("patient_map_path")
        patient_map_path = patient_map_path or configured_path
        if patient_map_path is None:
            raise ValueError(
                f"{canonical} outcomes lack patient identifiers; pass --patient-map-path"
            )
        patient_map_path = _resolve_existing_path(
            Path(patient_map_path), [], f"{canonical} patient mapping CSV"
        )
        stay_column = (
            patient_map_stay_column
            or config.get("patient_map_stay_column")
            or "ts_id"
        )
        patient_column = (
            patient_map_patient_column
            or config.get("patient_map_patient_column")
            or "patient_id"
        )
        patient_by_stay = patient_mapping_from_csv(
            patient_map_path,
            stay_column=stay_column,
            patient_column=patient_column,
        )

    patient_by_stay = {
        _normalise_stay_id(stay_id): patient_id
        for stay_id, patient_id in patient_by_stay.items()
    }
    train_ids = np.asarray([_normalise_stay_id(value) for value in train_ids])
    validation_ids = np.asarray(
        [_normalise_stay_id(value) for value in validation_ids]
    )
    test_ids = np.asarray([_normalise_stay_id(value) for value in test_ids])

    print(
        f"Loaded {canonical}: {len(feature_by_stay):,} SAPS-II rows "
        f"({id_column}) from {csv_path}"
    )
    return (
        feature_by_stay,
        label_by_stay,
        patient_by_stay,
        train_ids,
        validation_ids,
        test_ids,
        csv_path,
        pkl_path,
    )


def extract_xy(
    stay_ids: Iterable[object],
    feature_by_stay: dict[object, np.ndarray],
    label_by_stay: dict[object, int],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested_ids = [_normalise_stay_id(value) for value in stay_ids]
    matched_ids = [
        stay_id
        for stay_id in requested_ids
        if stay_id in feature_by_stay and stay_id in label_by_stay
    ]
    if not matched_ids:
        feature_sample = list(feature_by_stay)[:5]
        label_sample = list(label_by_stay)[:5]
        raise ValueError(
            f"No {split_name} IDs match both SAPS-II features and labels. "
            f"Feature ID sample={feature_sample}; label ID sample={label_sample}; "
            f"split ID sample={requested_ids[:5]}"
        )

    missing_count = len(requested_ids) - len(matched_ids)
    if missing_count:
        print(
            f"{split_name}: matched {len(matched_ids):,}/{len(requested_ids):,} stays; "
            f"skipped {missing_count:,} without both features and labels"
        )
    x = np.vstack([feature_by_stay[stay_id] for stay_id in matched_ids])
    y = np.asarray([label_by_stay[stay_id] for stay_id in matched_ids], dtype=int)
    return x, y, np.asarray(matched_ids)


def write_results_csv(
    fold_results: list[dict[str, float]],
    dataset: str,
    output_path: Path,
    *,
    seed: int,
    num_folds: int,
    window_size: int,
    regularization_c: float,
    train_sizes: list[int],
    test_size: int,
    feature_csv_path: Path,
    pkl_path: Path,
) -> pd.DataFrame:
    """Write per-fold and aggregate metrics to one machine-readable CSV."""

    if not fold_results:
        raise ValueError("Cannot write an empty SAPS-II result set")
    if len(train_sizes) != len(fold_results):
        raise ValueError("train_sizes must contain one value for every fold")

    metric_columns = [
        "auc",
        "prauc",
        "pr09",
        "pr08",
        "f1",
        "acc",
        "minrp",
        "recall",
        "precision",
    ]
    common = {
        "dataset": dataset,
        "model": "logistic_regression_sapsii",
        "seed": seed,
        "num_folds": num_folds,
        "window_size": window_size,
        "regularization_c": regularization_c,
        "feature_csv": str(Path(feature_csv_path)),
        "processed_pkl": str(Path(pkl_path)),
    }
    rows: list[dict[str, object]] = []
    for fold_index, (metrics, train_size) in enumerate(
        zip(fold_results, train_sizes, strict=True), start=1
    ):
        rows.append(
            {
                **common,
                "result_type": "fold",
                "fold": fold_index,
                "train_samples": train_size,
                "test_samples": test_size,
                **{name: metrics[name] for name in metric_columns},
            }
        )

    for result_type, reducer in (("mean", np.mean), ("std", np.std)):
        rows.append(
            {
                **common,
                "result_type": result_type,
                "fold": pd.NA,
                "train_samples": pd.NA,
                "test_samples": test_size,
                **{
                    name: float(reducer([result[name] for result in fold_results]))
                    for name in metric_columns
                },
            }
        )

    columns = [
        "dataset",
        "model",
        "result_type",
        "fold",
        "seed",
        "num_folds",
        "window_size",
        "regularization_c",
        "train_samples",
        "test_samples",
        "feature_csv",
        "processed_pkl",
        *metric_columns,
    ]
    result_frame = pd.DataFrame(rows, columns=columns)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved fold and summary metrics to {output_path.resolve()}")
    return result_frame


def run_experiment(args: argparse.Namespace) -> list[dict[str, float]]:
    canonical = DATASET_ALIASES[args.dataset]
    print(
        f"\n{'#' * 40}\n"
        f"SAPS-II Logistic Regression: {canonical}\n"
        f"{'#' * 40}"
    )
    (
        feature_by_stay,
        label_by_stay,
        patient_by_stay,
        train_ids,
        validation_ids,
        test_ids,
        feature_csv_path,
        processed_pkl_path,
    ) = load_data(
        dataset=args.dataset,
        csv_path=args.csv_path,
        pkl_path=args.pkl_path,
        patient_map_path=args.patient_map_path,
        patient_map_stay_column=args.patient_map_stay_column,
        patient_map_patient_column=args.patient_map_patient_column,
    )

    x_test, y_test, matched_test_ids = extract_xy(
        test_ids, feature_by_stay, label_by_stay, "test"
    )
    combined_ids = np.concatenate([train_ids, validation_ids])
    folds = get_sliding_window_splits(
        combined_ids,
        patient_by_stay,
        n_splits=args.num_folds,
        window_size=args.window_size,
        seed=args.seed,
    )

    fold_results: list[dict[str, float]] = []
    train_sizes: list[int] = []
    for fold_index, (fold_train_ids, fold_validation_ids) in enumerate(folds):
        assert_split_integrity(
            fold_train_ids, fold_validation_ids, test_ids, patient_by_stay
        )
        print(f"\n--- Fold {fold_index + 1} / {args.num_folds} ---")
        x_train, y_train, _ = extract_xy(
            fold_train_ids, feature_by_stay, label_by_stay, "train"
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        print(
            f"Train shape: {x_train_scaled.shape}; "
            f"test shape: {x_test_scaled.shape} ({len(matched_test_ids):,} stays)"
        )

        classifier = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            C=args.regularization_c,
            random_state=args.seed,
        )
        classifier.fit(x_train_scaled, y_train)
        probabilities = classifier.predict_proba(x_test_scaled)[:, 1]
        predictions = classifier.predict(x_test_scaled)
        metrics = calculate_metrics(y_test, probabilities, predictions)
        fold_results.append(metrics)
        train_sizes.append(len(y_train))
        print(
            f"  AUROC: {metrics['auc']:.4f} | AUPRC: {metrics['prauc']:.4f} | "
            f"PR@0.9: {metrics['pr09']:.4f} | F1: {metrics['f1']:.4f}"
        )

    print(f"\n{'=' * 20} Final LR Results ({len(fold_results)} Folds) {'=' * 20}")
    display_names = {
        "auc": "AUROC",
        "prauc": "AUPRC",
        "pr09": "PR@0.9",
        "pr08": "PR@0.8",
        "f1": "F1",
        "acc": "ACC",
    }
    for key, display_name in display_names.items():
        values = [result[key] for result in fold_results]
        print(f"{display_name.ljust(10)}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    output_path = args.output_path or (
        DEFAULT_RESULTS_DIR / canonical / "sapsii_logistic_regression.csv"
    )
    write_results_csv(
        fold_results,
        canonical,
        output_path,
        seed=args.seed,
        num_folds=args.num_folds,
        window_size=args.window_size,
        regularization_c=args.regularization_c,
        train_sizes=train_sizes,
        test_size=len(y_test),
        feature_csv_path=feature_csv_path,
        pkl_path=processed_pkl_path,
    )
    return fold_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SAPS-II baseline on MIMIC-III, MIMIC-IV or eICU."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_ALIASES),
        default="mimic3",
        help="Dataset preset (default: mimic3).",
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--pkl-path", type=Path, default=None)
    parser.add_argument("--patient-map-path", type=Path, default=None)
    parser.add_argument("--patient-map-stay-column", default=None)
    parser.add_argument("--patient-map-patient-column", default=None)
    parser.add_argument("--num-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Result CSV path. Defaults to "
            "results/<dataset>/sapsii_logistic_regression.csv."
        ),
    )
    args = parser.parse_args()
    if args.num_folds < 2:
        parser.error("--num-folds must be at least 2")
    if args.window_size < 1 or args.window_size >= args.num_folds:
        parser.error("--window-size must be in [1, num-folds - 1]")
    if args.regularization_c <= 0:
        parser.error("--regularization-c must be positive")
    return args


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
