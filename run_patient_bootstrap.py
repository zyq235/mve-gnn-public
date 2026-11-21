"""Train on the original pickle train split, select on validation, bootstrap test.

Built-in adapters: main-table baselines, GCT, and GRAM.  MVE-GNN uses the
dedicated repeat-5 and distillation stages in
``train_mve_bootstrap_repeat5.py`` and
``train_mve_bootstrap_distillation.py``.
Other models can write the same predictions.csv schema and use --mode bootstrap.
``--model mve_combined`` evaluates selected distilled students by
soft-averaging their aligned test probabilities; it never trains.
No development-set folds are generated for training by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.ehr_cached_graph_dataset import (
    SUPPORTED_DATASETS,
    load_protocol_splits,
    resolve_ehr_dataset_paths,
)
from ehr_baseline_utils import cpu_state_dict, resolve_device, save_json, set_seed
from main_table_classical import TABULAR_MODELS, fit_tabular_baseline
from patient_bootstrap import bootstrap_predictions, metric_values, validate_predictions
from split_protocol import assert_split_integrity


ROOT = Path(__file__).resolve().parent
NEURAL_TABLE_MODELS = {
    "gru", "grud", "sand", "tcn", "istrats", "mtgnn", "raindrop",
    "mstdgnn", "mstdgnn_multiscale",
}
COMBINED_VIEWS = ("type", "diff", "time")
TRAIN_MODELS = {"gct", "gram", "mve_combined"} | NEURAL_TABLE_MODELS | TABULAR_MODELS


def split_digest(train_ids: Sequence[int], validation_ids: Sequence[int], test_ids: Sequence[int]) -> str:
    payload = {
        "train": list(map(int, train_ids)),
        "validation": list(map(int, validation_ids)),
        "test": list(map(int, test_ids)),
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def load_original_split(dataset: str, data_root: Path, *, require_graph_cache: bool = True,
                        require_metadata: bool = True):
    paths = resolve_ehr_dataset_paths(
        dataset, data_root, require_graph_cache=require_graph_cache,
        require_metadata=require_metadata,
    )
    # The loader reads the five objects already saved in the processed pickle.
    # Its generated folds are ignored: no re-partitioning enters this experiment.
    protocol = load_protocol_splits(paths)
    for label, ids in (("train", protocol.train_ids),
                       ("validation", protocol.validation_ids),
                       ("test", protocol.test_ids)):
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Original pickle {label} IDs are empty or duplicated")
    assert_split_integrity(
        protocol.train_ids, protocol.validation_ids, protocol.test_ids,
        protocol.patient_by_stay,
    )
    for label, ids in (("validation", protocol.validation_ids), ("test", protocol.test_ids)):
        if {protocol.label_by_stay[int(stay)] for stay in ids} != {0, 1}:
            raise ValueError(f"The fixed {label} set needs both outcome classes")
    return paths, protocol, split_digest(
        protocol.train_ids, protocol.validation_ids, protocol.test_ids
    )


def _config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--model-config must contain a JSON object")
    return payload


def _override(defaults: dict[str, Any], supplied: Mapping[str, Any], what: str) -> dict[str, Any]:
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {what} settings: {sorted(unknown)}")
    return {**defaults, **supplied}


class GCTAdapter:
    name = "gct"
    default_epochs = 20
    default_batch_size = 32
    default_eval_batch_size = 64
    default_learning_rate = 1e-3

    def __init__(self, paths, config: dict[str, Any]):
        from dataset.gct_dataset import build_variable_type_index

        if set(config) - {"dataset", "model", "regularization_coefficient"}:
            raise ValueError("GCT config keys must be dataset, model, regularization_coefficient")
        self.variable_types, _ = build_variable_type_index(paths.var2ind_path, paths.metadata_path)
        self.paths = paths
        self.dataset_kwargs = _override(
            {"time_limit_hours": 24.0, "max_codes": 256}, config.get("dataset", {}), "GCT dataset"
        )
        self.model_settings = _override(
            {"hidden_dim": 128, "num_heads": 1, "num_stacks": 3,
             "num_feedforward": 2, "dropout": 0.08,
             "attention_normalizer": "softmax", "head_aggregation": "concat",
             "use_prior": True, "use_guide": True},
            config.get("model", {}), "GCT model",
        )
        self.regularization_coefficient = float(config.get("regularization_coefficient", 0.1))
        self.collate_fn = None

    def dataset(self, stay_ids):
        from dataset.gct_dataset import GCTICUDataset, gct_collate_fn

        self.collate_fn = gct_collate_fn
        return GCTICUDataset(
            self.paths.processed_dir, stay_ids, index_path=self.paths.index_path,
            variable_type_index=self.variable_types, **self.dataset_kwargs,
        )

    def build_model(self, device, demo_dim):
        from gct.common import build_model

        return build_model({"num_variables": int(self.variable_types.numel()),
                            "demo_dim": demo_dim, "output_attentions": False,
                            **self.model_settings}, device)

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        from gct.common import train_one_epoch

        return train_one_epoch(model, loader, optimizer, device,
                               regularization_coefficient=self.regularization_coefficient,
                               gradient_clip=gradient_clip)

    def predict_batch(self, model, batch, device):
        from ehr_baseline_utils import move_batch

        batch = move_batch(batch, device)
        logits = model(batch)["logits"]
        return batch["stay_id"], batch["labels"], torch.sigmoid(logits)


class GRAMAdapter:
    name = "gram"
    default_epochs = 20
    default_batch_size = 64
    default_eval_batch_size = 128
    default_learning_rate = 1e-3

    def __init__(self, paths, config: dict[str, Any]):
        from dataset.gram_dataset import build_gram_hierarchy

        if set(config) - {"dataset", "model"}:
            raise ValueError("GRAM config keys must be dataset and model")
        self.hierarchy = build_gram_hierarchy(paths.var2ind_path, paths.metadata_path)
        self.paths = paths
        self.dataset_kwargs = _override(
            {"time_limit_hours": 24.0, "bin_minutes": 60},
            config.get("dataset", {}), "GRAM dataset",
        )
        self.model_settings = _override(
            {"demo_hidden_dim": 16, "embedding_dim": 128, "attention_dim": 128,
             "hidden_dim": 128, "dropout": 0.5, "use_values": True},
            config.get("model", {}), "GRAM model",
        )
        self.collate_fn = None

    def dataset(self, stay_ids):
        from dataset.gram_dataset import GRAMICUDataset

        return GRAMICUDataset(
            self.paths.processed_dir, stay_ids, index_path=self.paths.index_path,
            num_variables=self.hierarchy.num_leaves, **self.dataset_kwargs,
        )

    def build_model(self, device, demo_dim):
        from gram.common import build_model

        return build_model({"num_nodes": self.hierarchy.num_nodes,
                            "ancestor_index": self.hierarchy.ancestor_index,
                            "ancestor_mask": self.hierarchy.ancestor_mask,
                            "demo_dim": demo_dim, **self.model_settings}, device)

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        from gram.common import train_one_epoch

        return train_one_epoch(model, loader, optimizer, device, gradient_clip=gradient_clip)

    def predict_batch(self, model, batch, device):
        from ehr_baseline_utils import move_batch

        batch = move_batch(batch, device)
        logits = model(batch["presence"], batch["values"], batch["lengths"], batch["demo"])
        return batch["stay_id"], batch["labels"], torch.sigmoid(logits)


class MVEAdapter:
    """Reuse the original pretraining checkpoint and MVE-GNN fine-tuning model."""

    name = "mve"
    default_epochs = 20
    default_batch_size = 8
    default_eval_batch_size = 8
    default_learning_rate = None  # read the original fine-tuner's view-specific LR
    default_weight_decay = 0.0
    default_gradient_clip = 0.5

    def __init__(self, paths, config: dict[str, Any], args, output_dir: Path, device, *,
                 initialize_pretrained=True):
        if config:
            raise ValueError("MVE uses the project's ft_param_*.json; --model-config is for GCT/GRAM")
        import finetune_kfold as ft

        if device.type == "cpu" and torch.cuda.is_available():
            raise ValueError("The original MVE builder selects CUDA when available; choose cuda:N")
        ft.logger = SimpleNamespace(log=lambda message: print(message, flush=True))
        self.ft = ft
        self.paths = paths
        self.device = device
        self.initialize_pretrained = initialize_pretrained
        self.collate_fn = None
        self.margs = SimpleNamespace(
            dataset=args.dataset, data_root=args.data_root,
            settings_root=args.settings_root, checkpoint_root=args.checkpoint_root,
            output_root=output_dir, pool_mode=args.pool_mode,
            pretrained_checkpoint=(
                args.pretrained_checkpoint.resolve() if args.pretrained_checkpoint is not None else None
            ),
            device=int(device.index or 0),
            task_name=f"bootstrap_{args.dataset}_{args.pool_mode}",
        )
        self.dataset_config = ft.resolve_dataset_config(self.margs)
        if initialize_pretrained and not self.dataset_config.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Configured MVE checkpoint is missing: {self.dataset_config.checkpoint_path}. "
                "Supply a compatible encoder checkpoint with --pretrained-checkpoint."
            )
        if not self.dataset_config.pretrain_config_path.is_file():
            raise FileNotFoundError(
                f"MVE pretraining settings are missing: {self.dataset_config.pretrain_config_path}"
            )
        ft.validate_required_paths(self.dataset_config, require_pretrained=initialize_pretrained)
        self.pretrain_config = ft.load_pretrain_args(
            self.dataset_config.pretrain_config_path, self.dataset_config.pretrain_profile
        )
        if initialize_pretrained:
            pretrained = torch.load(
                self.dataset_config.checkpoint_path, map_location="cpu", weights_only=False
            )
            if isinstance(pretrained, dict) and "model_state_dict" in pretrained:
                pretrained = pretrained["model_state_dict"]
            if not isinstance(pretrained, Mapping):
                raise ValueError("MVE pretrained checkpoint must contain a state dictionary")
            signature_key = "encoder.layers_stage1.0.conv.lin.weight"
            if signature_key in pretrained:
                checkpoint_hidden = int(pretrained[signature_key].shape[0])
                configured_hidden = int(self.pretrain_config["num_hidden"])
                if checkpoint_hidden != configured_hidden:
                    raise ValueError(
                        f"MVE checkpoint has hidden size {checkpoint_hidden}, but the "
                        f"pretraining settings specify {configured_hidden}. Select a "
                        "matching checkpoint or a complete, experimentally justified "
                        "--settings-root with matching pretraining and fine-tuning dimensions."
                    )
            del pretrained
        self.finetune_config = ft.load_finetune_args(self.dataset_config.finetune_config_path)
        allowed = {
            "aggr_type", "num_clusters_k", "ts_heads", "num_layers", "num_hidden",
            "pool_mode", "max_epoch", "lr", "dropout", "weight_decay",
            "gradient_clip", "batch_size", "eval_batch_size",
        }
        unknown = set(self.finetune_config) - allowed
        if unknown:
            raise ValueError(f"Unhandled MVE fine-tuning settings: {sorted(unknown)}")
        expected_pool_mode = "diff_pool" if args.pool_mode in {"diff", "diff_pool"} else args.pool_mode
        if self.finetune_config.get("pool_mode") != expected_pool_mode:
            raise ValueError(
                f"MVE fine-tuning settings at {self.dataset_config.finetune_config_path} "
                f"must specify pool_mode={expected_pool_mode!r}"
            )
        self.default_epochs = int(self.finetune_config.get("max_epoch", 20))
        self.default_batch_size = int(self.finetune_config.get("batch_size", 8))
        self.default_eval_batch_size = int(self.finetune_config.get("eval_batch_size", 8))
        self.default_weight_decay = float(self.finetune_config.get("weight_decay", 0.0))
        self.default_gradient_clip = float(self.finetune_config.get("gradient_clip", 0.5))
        if min(self.default_epochs, self.default_batch_size, self.default_eval_batch_size) <= 0:
            raise ValueError("MVE epochs and batch sizes from settings must be positive")
        if self.default_weight_decay < 0 or self.default_gradient_clip <= 0:
            raise ValueError("Invalid MVE weight decay or gradient clip in settings")
        self.model_args = None

    def dataset(self, stay_ids):
        return self.ft.make_dataset(self.dataset_config, stay_ids)

    def build_model(self, device, demo_dim):
        if demo_dim != 2:
            raise ValueError("The original MVE model expects two demographic features")
        model, self.model_args = self.ft.build_finetune_models(
            self.pretrain_config,
            self.finetune_config,
            self.margs,
            self.dataset_config,
            initialize_pretrained=self.initialize_pretrained,
        )
        actual = next(model.parameters()).device
        if actual.type != device.type or (device.index is not None and actual.index != device.index):
            raise RuntimeError("MVE constructor did not honor the requested CUDA device")
        return model

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        from torch import nn

        model.train()
        criterion = nn.BCEWithLogitsLoss()
        mode = self.model_args.pool_mode
        confidence_weight = float(getattr(self.model_args, "lambda_confidence", 0.0))
        total_loss = 0.0
        examples = 0
        for batch in loader:
            batch = self.ft.normalize_collated_group_ids(batch.to(device), self.model_args)
            optimizer.zero_grad(set_to_none=True)
            if mode == "diff_pool" and confidence_weight > 0:
                logits, confidence = self.ft.forward_diffpool_with_confidence(model, batch, mode)
                loss = criterion(logits, batch.y.view(-1, 1)) + confidence_weight * confidence
            else:
                logits = model(batch, mode=mode)
                loss = criterion(logits, batch.y.view(-1, 1))
            if not torch.isfinite(loss):
                raise FloatingPointError("MVE training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            count = int(batch.y.numel())
            total_loss += float(loss.detach()) * count
            examples += count
        return total_loss / max(examples, 1)

    def predict_batch(self, model, batch, device):
        batch = self.ft.normalize_collated_group_ids(batch.to(device), self.model_args)
        logits = model(batch, mode=self.model_args.pool_mode)
        return batch.ts_id, batch.y, torch.sigmoid(logits)


def make_loader(adapter, dataset, batch_size: int, num_workers: int, device, *, shuffle: bool):
    if adapter.name == "mve":
        from torch_geometric.loader import DataLoader as GraphDataLoader

        loader_type = GraphDataLoader
    else:
        loader_type = DataLoader
    return loader_type(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=adapter.collate_fn, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )


def _flat_tensor(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).numpy()
    return np.asarray(value).reshape(-1)


def predict_model(adapter, model, loader, device, expected_ids: Sequence[int]) -> pd.DataFrame:
    model.eval()
    rows = []
    # GRU-D's TorchScript cell is compatible with no_grad but not inference_mode.
    with torch.no_grad():
        for batch in loader:
            stay_ids, labels, probabilities = adapter.predict_batch(model, batch, device)
            ids = _flat_tensor(stay_ids)
            y = _flat_tensor(labels)
            scores = _flat_tensor(probabilities)
            if not (len(ids) == len(y) == len(scores)):
                raise RuntimeError("Model returned misaligned predictions, labels or stay IDs")
            rows.extend((int(i), int(label), float(score)) for i, label, score in zip(ids, y, scores))
    frame = pd.DataFrame(rows, columns=["stay_id", "label", "probability"])
    if frame["stay_id"].duplicated().any() or set(frame["stay_id"]) != set(map(int, expected_ids)):
        raise RuntimeError("Prediction IDs do not match the requested split exactly")
    return frame.sort_values("stay_id").reset_index(drop=True)


def _check_labels(frame: pd.DataFrame, label_by_stay: Mapping[int, int]):
    mismatch = frame.loc[
        frame.apply(lambda row: int(row.label) != int(label_by_stay[int(row.stay_id)]), axis=1)
    ]
    if not mismatch.empty:
        raise ValueError(f"Cached-graph labels disagree with pickle outcomes for {len(mismatch)} stays")


def _check_prediction_file(frame: pd.DataFrame, args, protocol, digest):
    frame = validate_predictions(frame)
    if str(frame["dataset"].iloc[0]) != args.dataset:
        raise ValueError("Prediction file dataset does not match --dataset")
    if str(frame["split_digest"].iloc[0]) != digest:
        raise ValueError("Prediction file was not produced using these exact pickle split IDs")
    if set(frame["stay_id"]) != set(map(int, protocol.test_ids)):
        raise ValueError("Prediction file must cover every original test stay, and no other stays")
    for row in frame.itertuples(index=False):
        if int(row.label) != int(protocol.label_by_stay[row.stay_id]):
            raise ValueError(f"Wrong outcome label for stay {row.stay_id}")
        if str(row.patient_id) != str(protocol.patient_by_stay[row.stay_id]):
            raise ValueError(f"Wrong patient ID for stay {row.stay_id}")
    return frame


def _load_prediction_csv(path: Path):
    return validate_predictions(pd.read_csv(path, dtype={
        "patient_id": str, "dataset": str, "model": str, "split_digest": str,
    }))


def _read_prediction(path: Path, args, protocol, digest):
    return _check_prediction_file(_load_prediction_csv(path), args, protocol, digest)


def build_neural_adapter(model_name, paths, model_config, args, protocol,
                         output_dir, device):
    """Construct the same adapter for training or checkpoint-only evaluation."""
    if model_name == "gct":
        return GCTAdapter(paths, model_config)
    if model_name == "gram":
        return GRAMAdapter(paths, model_config)
    if model_name == "mve":
        return MVEAdapter(paths, model_config, args, output_dir, device)

    from main_table_adapters import (
        MSTDMultiscaleAdapter, MatrixClassifierAdapter,
        RaindropAdapter, StratsAdapter, TimeseriesBaselineAdapter,
    )

    if model_name in TimeseriesBaselineAdapter.supported_models:
        return TimeseriesBaselineAdapter(
            model_name, paths, model_config, args.seed,
            debug_numerics=getattr(args, "debug_numerics", False),
        )
    if model_name == "istrats":
        return StratsAdapter(paths, model_config, args, protocol)
    if model_name == "raindrop":
        return RaindropAdapter(paths, model_config)
    if model_name == "mstdgnn_multiscale":
        return MSTDMultiscaleAdapter(paths, model_config)
    return MatrixClassifierAdapter(model_name, paths, model_config)


def _combined_view_paths(args: argparse.Namespace) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for specification in args.combined_view_csv:
        view, separator, filename = specification.partition("=")
        if separator != "=" or view not in args.combined_views or not filename:
            raise ValueError(
                "--combined-view-csv must be VIEW=PATH with VIEW in "
                f"{args.combined_views}"
            )
        if view in overrides:
            raise ValueError(f"Duplicate --combined-view-csv for {view}")
        overrides[view] = Path(filename).expanduser()
    weight = f"{args.distillation_weight:g}".replace(".", "p")
    base = args.combined_results_root / args.dataset
    return {
        view: overrides.get(
            view,
            base / f"mve_{view}_distilled_w{weight}" /
            f"seed_{args.seed}" / "predictions.csv",
        )
        for view in args.combined_views
    }


def _combined_predictions(args, protocol, digest: str, model_name: str):
    if not 0.0 <= args.distillation_weight <= 1.0:
        raise ValueError("--distillation-weight must lie in [0, 1]")
    source_paths = _combined_view_paths(args)
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, Any]] = {}
    for view, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {view} distilled predictions: {path}")
        frame = _read_prediction(path, args, protocol, digest)
        metadata_path = path.parent / "run_metadata.json"
        source = {"predictions_csv": str(path.resolve()),
                  "model": str(frame["model"].iloc[0])}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "stage": "mve_repeat5_mean_logit_distillation",
                "dataset": args.dataset,
                "pool_mode": "diff_pool" if view == "diff" else view,
                "split_digest": digest,
                "distillation_weight": args.distillation_weight,
            }
            mismatched = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    f"{view} distilled metadata disagrees with this run: {mismatched}"
                )
            source["run_metadata_json"] = str(metadata_path.resolve())
            source["student_checkpoint"] = metadata.get("student_checkpoint")
        frames[view] = frame
        sources[view] = source

    reference = frames[args.combined_views[0]]
    columns = ("dataset", "split_digest", "stay_id", "patient_id", "label")
    for view in args.combined_views[1:]:
        for column in columns:
            if not np.array_equal(
                reference[column].to_numpy(), frames[view][column].to_numpy()
            ):
                raise ValueError(
                    f"{view} distilled predictions disagree on {column}; "
                    "the selected views cannot be combined"
                )
    combined = reference.copy()
    combined["model"] = model_name
    combined["probability"] = np.stack(
        [frames[view]["probability"].to_numpy(dtype=np.float64)
         for view in args.combined_views], axis=0
    ).mean(axis=0)
    return _check_prediction_file(combined, args, protocol, digest), sources


def run(args: argparse.Namespace):
    if args.bootstrap_reps < 1 or args.num_workers < 0:
        raise ValueError("bootstrap-reps must be positive and num-workers nonnegative")
    is_combined = args.model == "mve_combined"
    if is_combined and args.predictions_csv is not None:
        raise ValueError(
            "mve_combined reads the selected distilled views; do not pass "
            "--predictions-csv. To bootstrap an existing combined CSV, omit --model."
        )
    if is_combined:
        default_name = ("mve_combined" if args.combined_views == COMBINED_VIEWS
                        else "mve_combined_" + "_".join(args.combined_views))
        model_name = args.run_name or default_name
        own = None
    elif args.mode == "bootstrap":
        if args.predictions_csv is None:
            raise ValueError("--mode bootstrap requires --predictions-csv")
        own = _load_prediction_csv(args.predictions_csv)
        model_name = str(own["model"].iloc[0])
    else:
        if args.model not in TRAIN_MODELS:
            raise ValueError(f"--mode train requires --model in {sorted(TRAIN_MODELS)}")
        model_name = args.run_name or args.model
        own = None
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.mode == "train" or is_combined:
        output_dir = ROOT / "results" / "patient_bootstrap" / args.dataset / model_name / f"seed_{args.seed}"
    else:
        output_dir = (ROOT / "results" / "patient_bootstrap" / args.dataset /
                      "comparisons" / model_name / f"bootstrap_seed_{args.bootstrap_seed}")
    existing_result = output_dir / (
        "predictions.csv" if args.mode == "train" or is_combined
        else "bootstrap_summary.json"
    )
    if existing_result.exists() and not args.overwrite:
        raise FileExistsError(f"Existing result would be overwritten: {existing_result}; use --overwrite")
    paths, protocol, digest = load_original_split(
        args.dataset, args.data_root,
        require_graph_cache=args.mode == "train" and args.model in {"gct", "gram", "mve"},
        require_metadata=args.mode == "train" and args.model not in {"sapsii", "mve_combined"},
    )
    if args.mode == "bootstrap" and not is_combined:
        own = _check_prediction_file(own, args, protocol, digest)
    if is_combined:
        own, sources = _combined_predictions(args, protocol, digest, model_name)
        if (output_dir / "predictions.csv").resolve() in {
            Path(source["predictions_csv"]).resolve() for source in sources.values()
        }:
            raise ValueError("Combined output must not overwrite a distilled view input")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_combined:
        own.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
        save_json(output_dir / "run_metadata.json", {
            "dataset": args.dataset, "model": model_name,
            "model_adapter": "mve_combined", "split_digest": digest,
            "training": f"none; read {len(args.combined_views)} already-distilled test prediction files",
            "views": list(args.combined_views),
            "aggregation": "equal-weight mean of test probabilities",
            "distillation_weight": args.distillation_weight,
            "training_seed": args.seed,
            "test_stays": len(protocol.test_ids),
            "sources": sources,
            "bootstrap_repetitions": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
        })
        print(f"Combined distilled views; saved {output_dir / 'predictions.csv'}")

    if args.mode == "train" and args.model in TABULAR_MODELS:
        set_seed(args.seed)
        device = resolve_device(args.device)
        if args.model == "time2graphplus" and device.type == "cuda" and device.index not in (None, 0):
            raise ValueError("The existing Time2Graph+ code selects cuda:0; choose cuda:0 or cpu")
        if any(value is not None for value in (
            args.epochs, args.learning_rate, args.weight_decay,
            args.gradient_clip, args.eval_batch_size,
        )):
            raise ValueError("For tabular models, set estimator parameters in --model-config")
        model_config = _config(args.model_config)
        if args.batch_size is not None:
            model_config["batch_size"] = args.batch_size
        own, training_meta = fit_tabular_baseline(
            args.model, paths, protocol, args, model_config, output_dir,
        )
        _check_labels(own, protocol.label_by_stay)
        own.insert(0, "patient_id", own["stay_id"].map(
            lambda stay: str(protocol.patient_by_stay[int(stay)])
        ))
        own.insert(0, "split_digest", digest)
        own.insert(0, "model", model_name)
        own.insert(0, "dataset", args.dataset)
        own = _check_prediction_file(own, args, protocol, digest)
        own.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
        save_json(output_dir / "run_metadata.json", {
            "dataset": args.dataset, "model": model_name, "model_adapter": args.model,
            "pickle": str(paths.pickle_path), "split_digest": digest,
            "train_stays": len(protocol.train_ids),
            "validation_stays": len(protocol.validation_ids),
            "test_stays": len(protocol.test_ids),
            "train_patients": len({protocol.patient_by_stay[i] for i in protocol.train_ids}),
            "validation_patients": len({protocol.patient_by_stay[i] for i in protocol.validation_ids}),
            "test_patients": len({protocol.patient_by_stay[i] for i in protocol.test_ids}),
            "train_sampling": "all original train stays; no resampling",
            **training_meta,
        })
        print(f"Selected {args.model}; saved {output_dir / 'predictions.csv'}")

    if args.mode == "train" and args.model not in TABULAR_MODELS and not is_combined:
        set_seed(args.seed)
        device = resolve_device(args.device)
        model_config = _config(args.model_config)
        adapter = build_neural_adapter(
            args.model, paths, model_config, args, protocol, output_dir, device
        )

        train_set = adapter.dataset(protocol.train_ids)
        valid_set = adapter.dataset(protocol.validation_ids)
        if getattr(train_set, "demo_dim", 2) != getattr(valid_set, "demo_dim", 2):
            raise ValueError("Demographic feature dimensions differ across splits")
        train_loader = make_loader(
            adapter, train_set, args.batch_size or adapter.default_batch_size,
            args.num_workers, device, shuffle=True,
        )
        valid_loader = make_loader(
            adapter, valid_set, args.eval_batch_size or adapter.default_eval_batch_size,
            args.num_workers, device, shuffle=False,
        )
        model = adapter.build_model(device, getattr(train_set, "demo_dim", 2))
        epochs = args.epochs or adapter.default_epochs
        learning_rate = args.learning_rate or adapter.default_learning_rate
        if learning_rate is None:
            learning_rate = float(adapter.model_args.lr)
        weight_decay = (
            args.weight_decay if args.weight_decay is not None
            else getattr(adapter, "default_weight_decay", 0.0)
        )
        gradient_clip = (
            args.gradient_clip if args.gradient_clip is not None
            else getattr(adapter, "default_gradient_clip", 0.5)
        )
        optimizer_kind = getattr(adapter, "optimizer_kind", "adamw")
        optimizer_class = torch.optim.Adam if optimizer_kind == "adam" else torch.optim.AdamW
        optimizer = optimizer_class(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler_kind = getattr(adapter, "scheduler_kind", "cosine")
        if scheduler_kind == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_kind == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=2,
            )
        else:
            scheduler = None
        checkpoint = output_dir / "best_validation.pt"
        best_score = -float("inf")
        history = []
        for epoch in range(1, epochs + 1):
            if args.model in {"gru", "grud", "sand", "tcn"}:
                adapter.current_epoch = epoch
            loss = adapter.train_epoch(model, train_loader, optimizer, device, gradient_clip)
            validation = predict_model(adapter, model, valid_loader, device, protocol.validation_ids)
            _check_labels(validation, protocol.label_by_stay)
            metrics = metric_values(validation["label"].to_numpy(), validation["probability"].to_numpy())
            score = metrics["auc"] + metrics["prauc"]
            history.append({"epoch": epoch, "train_loss": loss, **metrics, "selection_score": score})
            if score > best_score:
                best_score = score
                torch.save({"model_state": cpu_state_dict(model), "epoch": epoch,
                            "validation": metrics, "dataset": args.dataset,
                            "model": args.model, "run_name": model_name,
                            "split_digest": digest}, checkpoint)
            if scheduler_kind == "cosine":
                scheduler.step()
            elif scheduler_kind == "plateau":
                probabilities = np.clip(validation["probability"].to_numpy(), 1e-7, 1 - 1e-7)
                labels = validation["label"].to_numpy()
                validation_log_loss = -np.mean(
                    labels * np.log(probabilities) +
                    (1 - labels) * np.log(1 - probabilities)
                )
                scheduler.step(float(validation_log_loss))
            print(f"epoch={epoch} train_loss={loss:.4f} val_auc={metrics['auc']:.4f} "
                  f"val_prauc={metrics['prauc']:.4f} best={best_score:.4f}", flush=True)
        save_json(output_dir / "history.json", history)
        selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model_state"])
        model.to(device)
        # Test features and predictions are first accessed after validation selection.
        test_set = adapter.dataset(protocol.test_ids)
        if getattr(test_set, "demo_dim", 2) != getattr(train_set, "demo_dim", 2):
            raise ValueError("Demographic feature dimensions differ in the test split")
        test_loader = make_loader(
            adapter, test_set, args.eval_batch_size or adapter.default_eval_batch_size,
            args.num_workers, device, shuffle=False,
        )
        own = predict_model(adapter, model, test_loader, device, protocol.test_ids)
        _check_labels(own, protocol.label_by_stay)
        own.insert(0, "patient_id", own["stay_id"].map(
            lambda stay: str(protocol.patient_by_stay[int(stay)])
        ))
        own.insert(0, "split_digest", digest)
        own.insert(0, "model", model_name)
        own.insert(0, "dataset", args.dataset)
        own = _check_prediction_file(own, args, protocol, digest)
        own.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
        save_json(output_dir / "run_metadata.json", {
            "dataset": args.dataset, "model": model_name,
            "model_adapter": args.model,
            "pickle": str(paths.pickle_path), "split_digest": digest,
            "train_stays": len(protocol.train_ids),
            "validation_stays": len(protocol.validation_ids),
            "test_stays": len(protocol.test_ids),
            "train_patients": len({protocol.patient_by_stay[i] for i in protocol.train_ids}),
            "validation_patients": len({protocol.patient_by_stay[i] for i in protocol.validation_ids}),
            "test_patients": len({protocol.patient_by_stay[i] for i in protocol.test_ids}),
            "selected_epoch": selected["epoch"],
            "selection_rule": "maximize validation AUROC + AUPRC",
            "validation_metrics": selected["validation"],
            "pretrained_checkpoint": str(adapter.dataset_config.checkpoint_path)
            if args.model == "mve" else None,
            "pretrain_settings_file": str(adapter.dataset_config.pretrain_config_path)
            if args.model == "mve" else None,
            "finetune_settings_file": str(adapter.dataset_config.finetune_config_path)
            if args.model == "mve" else None,
            "pretrain_config": adapter.pretrain_config if args.model == "mve" else None,
            "finetune_config": adapter.finetune_config if args.model == "mve" else None,
            "effective_mve_model_args": {
                key: getattr(adapter.model_args, key)
                for key in (
                    "num_hidden", "num_layers", "num_clusters_k", "ts_heads",
                    "aggr_type", "pool_mode", "dropout", "lambda_confidence",
                )
            } if args.model == "mve" else None,
            "effective_config": getattr(adapter, "effective_config", None),
            "strats_pretrained_checkpoint": str(adapter.pretrained_checkpoint)
            if args.model == "istrats" else None,
            "strats_pretrained_tensors_loaded": adapter.pretrained_tensors_loaded
            if args.model == "istrats" else None,
            "timeseries_initialized_tensors": adapter.initialized_tensors
            if args.model in {"gru", "grud", "sand", "tcn"} else None,
            "model_config": model_config, "learning_rate": learning_rate,
            "optimizer": optimizer_kind, "scheduler": scheduler_kind,
            "weight_decay": weight_decay, "gradient_clip": gradient_clip,
            "batch_size": train_loader.batch_size,
            "eval_batch_size": valid_loader.batch_size,
            "epochs": epochs, "train_sampling": "all original train stays once per epoch; shuffle",
        })
        print(f"Selected epoch {selected['epoch']}; saved {output_dir / 'predictions.csv'}")

    frames = {model_name: own}
    for path in args.compare_to:
        other = _read_prediction(path, args, protocol, digest)
        name = str(other["model"].iloc[0])
        if name in frames:
            raise ValueError(f"Duplicate model name {name!r} among prediction files")
        frames[name] = other
    summary, repetitions = bootstrap_predictions(
        frames, reference_model=model_name,
        repetitions=args.bootstrap_reps, seed=args.bootstrap_seed,
    )
    save_json(output_dir / "bootstrap_summary.json", summary)
    repetitions.to_csv(output_dir / "bootstrap_repetitions.csv", index=False)
    print(json.dumps(summary["models"][model_name], indent=2))
    print(f"Saved bootstrap results in {output_dir}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=("train", "bootstrap"), default="train")
    parser.add_argument("--model", choices=sorted(TRAIN_MODELS),
                        help="Train adapter; mve_combined only combines existing distilled predictions")
    parser.add_argument("--run-name", help="Unique comparison label, e.g. mve_type or mve_diff")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--predictions-csv", type=Path, help="Reference predictions for --mode bootstrap")
    parser.add_argument("--compare-to", type=Path, action="append", default=[],
                        help="Additional aligned predictions.csv; may be supplied repeatedly")
    parser.add_argument("--model-config", type=Path,
                        help="JSON hyperparameter overrides; MVE reads --settings-root instead")
    parser.add_argument("--sapsii-csv", type=Path,
                        help="SAPS-II component-score CSV, required if not at a default dataset path")
    parser.add_argument("--strats-pretrained-checkpoint", type=Path,
                        help="Override the dataset-specific iStraTS pretrained checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:N")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Optimizer weight decay; MVE defaults to settings value when present")
    parser.add_argument("--gradient-clip", type=float, default=None,
                        help="Gradient norm limit; MVE defaults to settings value when present")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time2graph-workers", type=int, default=1,
                        help="Time2Graph+ shapelet workers; one avoids concurrent GPU jobs")
    parser.add_argument("--debug-numerics", action="store_true",
                        help="Check baseline inputs, gradients and parameters each batch")
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--combined-results-root", type=Path,
                        default=ROOT / "results" / "patient_bootstrap",
                        help="Root containing per-dataset distilled MVE view results")
    parser.add_argument("--combined-views", nargs="+", choices=COMBINED_VIEWS,
                        help="Distilled views to combine (at least two); default: type diff time")
    parser.add_argument("--combined-view-csv", action="append", default=[],
                        metavar="VIEW=PATH",
                        help="Override one selected view's distilled predictions.csv path")
    parser.add_argument("--distillation-weight", type=float, default=0.75,
                        help="Select the mve_*_distilled_w* input directories for mve_combined")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pool-mode", choices=("type", "time", "node_time", "diff", "diff_pool"),
                        default="type", help="MVE only")
    parser.add_argument("--pretrained-checkpoint", type=Path,
                        help="Override the dataset's default MVE pretrained encoder checkpoint")
    parser.add_argument("--settings-root", type=Path, default=ROOT / "settings",
                        help="MVE pretrain_model_params.json and ft_param_*.json directory")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints",
                        help="MVE pretraining checkpoint directory")
    args = parser.parse_args(argv)
    if args.combined_views is None:
        args.combined_views = COMBINED_VIEWS
    else:
        if args.model != "mve_combined":
            parser.error("--combined-views requires --model mve_combined")
        if len(args.combined_views) < 2 or len(set(args.combined_views)) != len(args.combined_views):
            parser.error("--combined-views requires at least two distinct views")
        args.combined_views = tuple(view for view in COMBINED_VIEWS
                                    if view in args.combined_views)
    for name in ("epochs", "batch_size", "eval_batch_size", "time2graph_workers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.gradient_clip is not None and args.gradient_clip <= 0:
        parser.error("--gradient-clip must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("--run-name may contain only letters, numbers, underscore, dot and dash")
    if args.mode == "bootstrap" and args.run_name is not None:
        parser.error("--run-name applies only to --mode train; bootstrap uses the prediction-file label")
    return args


if __name__ == "__main__":
    run(parse_args())
