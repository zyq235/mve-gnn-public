"""Shared access to cached ICU-stay graphs and the project split protocol.

The graph cache is produced by :mod:`dataset.mimic3_patient_dataset` for all
three supported cohorts.  Reading it here avoids loading the multi-gigabyte
event DataFrame once per fold while preserving the exact normalized values,
demographics, labels and 24-hour cohort used by the graph models.
"""

from __future__ import annotations

import gc
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_csv,
    patient_mapping_from_outcomes,
)


SUPPORTED_DATASETS = ("mimic3", "mimic4", "eicu")
_DATASET_LAYOUT = {
    "mimic3": ("mimic3", "mimic_iii.pkl", None, None, None),
    "mimic4": ("mimic-iv", "mimic_iv.pkl", None, None, None),
    "eicu": ("eicu", "eicu.pkl", "patient_split.csv", "ts_id", "patient_id"),
}
_GRAPH_FILE = re.compile(r"^graph_(?P<stay_id>\d+)_(?P<slice_id>\d+)\.pt$")


@dataclass(frozen=True)
class EHRDatasetPaths:
    name: str
    data_dir: Path
    pickle_path: Path
    var2ind_path: Path
    metadata_path: Path
    processed_dir: Path
    index_path: Path
    patient_csv_path: Path | None
    patient_stay_column: str | None
    patient_id_column: str | None


@dataclass(frozen=True)
class ProtocolSplits:
    """The held-out test set and cyclic development-set folds."""

    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    label_by_stay: Mapping[int, int]
    patient_by_stay: Mapping[int, Hashable]

    @property
    def development_ids(self) -> tuple[int, ...]:
        return self.train_ids + self.validation_ids


def resolve_ehr_dataset_paths(
    dataset_name: str, data_root: str | Path, *, require_graph_cache: bool = True,
    require_metadata: bool = True,
) -> EHRDatasetPaths:
    """Resolve the common MIMIC-III/MIMIC-IV/eICU on-disk layout."""

    if dataset_name not in _DATASET_LAYOUT:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
        )
    (
        directory,
        pickle_name,
        patient_csv_name,
        patient_stay_column,
        patient_id_column,
    ) = _DATASET_LAYOUT[dataset_name]
    data_dir = Path(data_root) / directory
    processed_dir = data_dir / "processed"
    paths = EHRDatasetPaths(
        name=dataset_name,
        data_dir=data_dir,
        pickle_path=data_dir / pickle_name,
        var2ind_path=data_dir / "var2ind.json",
        metadata_path=data_dir / "var_mean_std.json",
        processed_dir=processed_dir,
        index_path=processed_dir / "processed_files.json",
        patient_csv_path=(
            data_dir / patient_csv_name if patient_csv_name is not None else None
        ),
        patient_stay_column=patient_stay_column,
        patient_id_column=patient_id_column,
    )
    required = [paths.pickle_path]
    if require_metadata:
        required.append(paths.metadata_path)
    if require_graph_cache:
        required.extend([paths.var2ind_path, paths.index_path])
    if paths.patient_csv_path is not None:
        required.append(paths.patient_csv_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required dataset files are missing:\n" + "\n".join(missing))
    return paths


def _label_mapping(outcomes: Any) -> dict[int, int]:
    if isinstance(outcomes, Mapping):
        return {int(stay_id): int(label) for stay_id, label in outcomes.items()}
    required = {"ts_id", "in_hospital_mortality"}
    if not hasattr(outcomes, "columns") or not required.issubset(outcomes.columns):
        raise ValueError(
            "Outcomes must be a mapping or contain ts_id and "
            "in_hospital_mortality columns"
        )
    return {
        int(stay_id): int(label)
        for stay_id, label in outcomes[
            ["ts_id", "in_hospital_mortality"]
        ].itertuples(index=False, name=None)
    }


def load_protocol_splits(
    paths: EHRDatasetPaths,
    *,
    n_splits: int = 6,
    window_size: int = 3,
    seed: int = 443,
) -> ProtocolSplits:
    """Load the original hold-out split and reproduce ``finetune_kfold.py``.

    The original train and validation IDs are combined.  Patients are shuffled
    into six chunks; three consecutive chunks train a model and the following
    chunk validates it.  The original test IDs remain untouched in every run.
    """

    with paths.pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 5:
        raise ValueError(
            f"Expected five objects in {paths.pickle_path}, got "
            f"{type(payload).__name__}"
        )
    events, outcomes, train_ids, validation_ids, test_ids = payload
    del events, payload
    gc.collect()

    train_ids = tuple(map(int, train_ids))
    validation_ids = tuple(map(int, validation_ids))
    test_ids = tuple(map(int, test_ids))
    development_ids = np.asarray(train_ids + validation_ids, dtype=np.int64)

    if paths.patient_csv_path is not None:
        assert paths.patient_stay_column is not None
        assert paths.patient_id_column is not None
        patient_by_stay = patient_mapping_from_csv(
            str(paths.patient_csv_path),
            stay_column=paths.patient_stay_column,
            patient_column=paths.patient_id_column,
        )
    else:
        try:
            patient_by_stay = patient_mapping_from_outcomes(outcomes)
        except (TypeError, ValueError) as exc:
            if paths.name == "mimic4":
                raise ValueError(
                    "MIMIC-IV mimic_iv.pkl outcomes must include ts_id and "
                    "SUBJECT_ID for patient-level bootstrap. The train/val/test "
                    "stay-ID lists alone cannot identify repeat stays from one "
                    "patient; regenerate this pickle with the updated "
                    "preprocess_mimic_iv.py."
                ) from exc
            raise
    label_by_stay = _label_mapping(outcomes)
    del outcomes
    gc.collect()

    assert_split_integrity(development_ids, (), test_ids, patient_by_stay)
    folds = get_sliding_window_splits(
        development_ids,
        id_to_patient=patient_by_stay,
        n_splits=n_splits,
        window_size=window_size,
        seed=seed,
    )
    normalized_folds = tuple(
        (tuple(map(int, fold_train)), tuple(map(int, fold_validation)))
        for fold_train, fold_validation in folds
    )
    for fold_train, fold_validation in normalized_folds:
        assert_split_integrity(
            fold_train, fold_validation, test_ids, patient_by_stay
        )

    all_ids = set(development_ids.tolist()) | set(test_ids)
    missing_labels = sorted(all_ids - set(label_by_stay))
    if missing_labels:
        raise KeyError(
            f"Missing mortality labels for {len(missing_labels)} stays; "
            f"first={missing_labels[:5]}"
        )
    return ProtocolSplits(
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        folds=normalized_folds,
        label_by_stay=label_by_stay,
        patient_by_stay=patient_by_stay,
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise KeyError(f"Cached graph is missing field {name!r}")
        return record[name]
    if not hasattr(record, name):
        raise AttributeError(f"Cached graph is missing field {name!r}")
    return getattr(record, name)


class CachedICUStayDataset(Dataset):
    """Base class that joins every cached slice into one ICU-stay sample."""

    def __init__(
        self,
        processed_dir: str | Path,
        stay_ids: Sequence[int],
        *,
        index_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.stay_ids = tuple(map(int, stay_ids))
        if not self.stay_ids:
            raise ValueError("stay_ids must not be empty")
        if len(self.stay_ids) != len(set(self.stay_ids)):
            raise ValueError("stay_ids contains duplicate ICU stays")

        resolved_index = (
            Path(index_path)
            if index_path is not None
            else self.processed_dir / "processed_files.json"
        )
        payload = read_json(resolved_index)
        if isinstance(payload, Mapping):
            payload = payload.get("files", [])
        if not isinstance(payload, list):
            raise ValueError(f"Invalid processed graph index: {resolved_index}")

        requested = set(self.stay_ids)
        files_by_stay: dict[int, list[tuple[int, Path]]] = {}
        for file_name in payload:
            match = _GRAPH_FILE.fullmatch(Path(str(file_name)).name)
            if match is None:
                continue
            stay_id = int(match.group("stay_id"))
            if stay_id not in requested:
                continue
            path = self.processed_dir / str(file_name)
            if not path.is_file():
                raise FileNotFoundError(f"Indexed graph does not exist: {path}")
            files_by_stay.setdefault(stay_id, []).append(
                (int(match.group("slice_id")), path)
            )
        missing = sorted(requested - set(files_by_stay))
        if missing:
            raise FileNotFoundError(
                f"No cached graph for {len(missing)} requested ICU stays; "
                f"first={missing[:10]}"
            )
        for stay_id in files_by_stay:
            files_by_stay[stay_id].sort(key=lambda item: item[0])
        self.files_by_stay = files_by_stay

        first = self._load_graph(files_by_stay[self.stay_ids[0]][0][1])
        self.demo_dim = int(torch.as_tensor(record_field(first, "demo")).numel())
        self._labels: list[float] | None = None

    def __len__(self) -> int:
        return len(self.stay_ids)

    @staticmethod
    def _load_graph(path: Path) -> Any:
        # The trusted local cache contains PyG Data objects.  PyTorch 2.6
        # changed the default to weights_only=True, so object loading must be
        # explicit.  Never point this loader at untrusted files.
        return torch.load(path, map_location="cpu", weights_only=False)

    def _load_stay(self, stay_id: int) -> dict[str, Tensor]:
        variable_parts: list[Tensor] = []
        value_parts: list[Tensor] = []
        time_parts: list[Tensor] = []
        label: Tensor | None = None
        demo: Tensor | None = None
        for _, path in self.files_by_stay[stay_id]:
            record = self._load_graph(path)
            variables = torch.as_tensor(record_field(record, "x_var")).long().view(-1)
            values = torch.as_tensor(record_field(record, "x_val")).float().view(-1)
            times = torch.as_tensor(record_field(record, "x_time")).float().view(-1)
            if not (variables.numel() == values.numel() == times.numel()):
                raise ValueError(f"Inconsistent event lengths in {path}")
            variable_parts.append(variables)
            value_parts.append(values)
            time_parts.append(times)

            current_label = torch.as_tensor(
                record_field(record, "y"), dtype=torch.float32
            ).view(-1)
            current_demo = torch.as_tensor(
                record_field(record, "demo"), dtype=torch.float32
            ).view(-1)
            if current_label.numel() != 1:
                raise ValueError(f"Non-scalar label in {path}")
            if label is not None and not torch.equal(label, current_label):
                raise ValueError(f"Conflicting labels across slices for stay {stay_id}")
            if demo is not None and not torch.equal(demo, current_demo):
                raise ValueError(
                    f"Conflicting demographics across slices for stay {stay_id}"
                )
            label, demo = current_label, current_demo

        assert label is not None and demo is not None
        return {
            "variables": torch.cat(variable_parts),
            "values": torch.cat(value_parts),
            "times": torch.cat(time_parts),
            "demo": demo,
            "label": label[0],
        }

    @property
    def labels(self) -> list[float]:
        if self._labels is None:
            self._labels = [
                float(self._load_stay(stay_id)["label"].item())
                for stay_id in self.stay_ids
            ]
        return self._labels
