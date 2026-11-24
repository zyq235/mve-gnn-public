"""Fine-tune MVE-GNN for 24-hour, ten-class remaining ICU LOS prediction.

This entry point is intentionally separate from ``finetune_kfold.py``.  It
reuses the same pretrained encoder and graph caches but has its own labels,
ten-logit prediction head, loss, checkpoints, and result files.  The original
mortality and pretraining code paths therefore remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm
from unittest.mock import patch

from dataset.los_patient_dataset import (
    LOSLargeGraphDataset,
    LOS_CLASS_NAMES,
    load_los_label_frame,
)
from ML.los_metrics import (
    NUM_LOS_CLASSES,
    class_representative_days,
    compute_los_metrics,
)
from models.graphmae.finetune_los_model import HierarchicalEHRLOSModel
import models.graphmae.finetune_graph_model as finetune_graph_module
from models.graphmae.mimic_largegraph_premodel import EHRLargeGraphPreModel
from models.graphmae.utils import PRETRAINED_CHECKPOINT_FILENAMES, set_random_seed
from split_protocol import assert_split_integrity, get_sliding_window_splits


LOGGER = logging.getLogger("finetune_los")
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DEFAULTS = {
    "mimic3": {
        "data_root": "data/mimic3",
        "pkl_name": "mimic_iii.pkl",
        "pretrain_profile": "mimic3",
        "source_vocabulary_directory": "mimic3",
        "model_vocabulary_directory": "mimic3",
    },
    "mimic4": {
        "data_root": "data/mimic-iv",
        "pkl_name": "mimic_iv.pkl",
        "pretrain_profile": "mimic4",
        "source_vocabulary_directory": "mimic-iv",
        # The historical MIMIC-IV backbone allocates 129 embedding rows from
        # the MIMIC-III vocabulary although its graph IDs number only 124.
        "model_vocabulary_directory": "mimic3",
    },
    "eicu": {
        "data_root": "data/eicu",
        "pkl_name": "eicu.pkl",
        "pretrain_profile": "mimic4",
        "source_vocabulary_directory": "mimic-iv",
        "model_vocabulary_directory": "mimic3",
    },
}


def _normalise_dataset_name(value: str) -> str:
    compact = value.lower().replace("-", "").replace("_", "")
    aliases = {
        "mimic3": "mimic3",
        "mimiciii": "mimic3",
        "mimic4": "mimic4",
        "mimiciv": "mimic4",
        "eicu": "eicu",
    }
    if compact not in aliases:
        raise argparse.ArgumentTypeError(f"Unsupported dataset: {value}")
    return aliases[compact]


def _resolve_path(override: str | None, default: str) -> Path:
    path = Path(override if override is not None else default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_los_paths(args) -> dict[str, Path]:
    """Resolve data, settings, and checkpoint paths for LOS fine-tuning."""

    defaults = DATASET_DEFAULTS[args.dataset]
    data_root = _resolve_path(args.data_root, defaults["data_root"])
    sibling_root = data_root.parent
    settings_root = _resolve_path(args.settings_root, "settings")
    config_name = "diff" if args.pool_mode == "diff_pool" else args.pool_mode
    return {
        "data_root": data_root,
        "pkl_path": _resolve_path(args.pkl_path, str(data_root / defaults["pkl_name"])),
        "pretrained_checkpoint": _resolve_path(
            args.pretrained_checkpoint,
            f"checkpoints/{PRETRAINED_CHECKPOINT_FILENAMES[args.dataset]}",
        ),
        "pretrain_config": settings_root / "pretrain_model_params.json",
        "finetune_config": settings_root / f"ft_param_{config_name}.json",
        "var2ind": data_root / "var2ind.json",
        "var_mean_std": data_root / "var_mean_std.json",
        "model_var2ind": _resolve_path(
            args.model_var2ind_path,
            str(sibling_root / defaults["model_vocabulary_directory"] / "var2ind.json"),
        ),
        "pretrained_var2ind": _resolve_path(
            args.pretrained_var2ind_path,
            str(sibling_root / defaults["source_vocabulary_directory"] / "var2ind.json"),
        ),
        "los_labels": _resolve_path(
            args.los_labels_path,
            str(data_root / f"los_{args.landmark_hours}h_labels.csv"),
        ),
    }


def validate_los_paths(paths: dict[str, Path], no_pretrained: bool = False) -> None:
    required = (
        "pkl_path", "los_labels", "var2ind", "var_mean_std",
        "model_var2ind", "pretrain_config", "finetune_config",
    )
    if not no_pretrained:
        required += ("pretrained_checkpoint", "pretrained_var2ind")
    missing = [f"{name}: {paths[name]}" for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("Missing LOS inputs:\n" + "\n".join(missing))


def _load_fold_metadata(dataset: str, data_root: Path, pkl_path: Path):
    """Read existing cohort splits, preferring lightweight sidecars."""

    if dataset == "mimic3":
        sidecar = data_root / "mimic_iii_split_ids.pkl"
        if sidecar.exists():
            with sidecar.open("rb") as handle:
                payload = pickle.load(handle)
            return tuple(
                np.asarray(payload[name], dtype=np.int64)
                for name in ("train_list", "val_list", "test_list")
            )
    if dataset == "eicu":
        sidecar = data_root / "patient_split.csv"
        if sidecar.exists():
            split_frame = pd.read_csv(sidecar, usecols=["ts_id", "split"])
            normalised_split = split_frame["split"].astype(str).str.lower()
            return tuple(
                split_frame.loc[normalised_split.isin(names), "ts_id"]
                .to_numpy(dtype=np.int64)
                for names in (("train",), ("val", "valid", "validation"), ("test",))
            )

    with pkl_path.open("rb") as handle:
        _, _, train_ids, val_ids, test_ids = pickle.load(handle)
    return tuple(
        np.asarray(values, dtype=np.int64)
        for values in (train_ids, val_ids, test_ids)
    )


def load_los_configs(paths: dict[str, Path], dataset: str, pool_mode: str) -> tuple[dict, dict]:
    """Read architecture settings from the pretraining and fine-tuning JSON files."""

    profile = DATASET_DEFAULTS[dataset]["pretrain_profile"]
    with paths["pretrain_config"].open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    if profile not in profiles:
        raise KeyError(f"Pretraining profile {profile!r} is missing from {paths['pretrain_config']}")
    pretrain_config = profiles[profile]
    with paths["finetune_config"].open("r", encoding="utf-8") as handle:
        finetune_config = json.load(handle)
    for key in ("num_hidden", "num_layers", "mask_rate"):
        if key not in pretrain_config:
            raise KeyError(f"Pretraining setting {key!r} is missing from {paths['pretrain_config']}")
    for key in ("num_hidden", "num_layers", "num_clusters_k", "ts_heads", "aggr_type", "pool_mode"):
        if key not in finetune_config:
            raise KeyError(f"Fine-tuning setting {key!r} is missing from {paths['finetune_config']}")
    if finetune_config["pool_mode"] != pool_mode:
        raise ValueError(f"Fine-tuning pool_mode does not match {pool_mode!r}: {paths['finetune_config']}")
    for key in ("num_hidden", "num_layers"):
        if int(pretrain_config[key]) != int(finetune_config[key]):
            raise ValueError(f"Pretraining and fine-tuning {key} differ in settings/")
    return pretrain_config, finetune_config


def build_variable_class_lookup(var2ind_path: Path, var_mean_std_path: Path):
    """Build the dataset-specific semantic taxonomy used by type pooling."""

    with var2ind_path.open("r", encoding="utf-8") as handle:
        var2ind = json.load(handle)
    with var_mean_std_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    classes = sorted({entry["class"] for entry in metadata.values()})
    if not classes:
        raise ValueError(f"No variable classes found in {var_mean_std_path}")
    class2ind = {name: index for index, name in enumerate(classes)}
    lookup = torch.full((len(var2ind),), -1, dtype=torch.long)
    for variable, index in var2ind.items():
        if variable not in metadata:
            raise KeyError(f"{variable!r} is missing from {var_mean_std_path}")
        lookup[int(index)] = class2ind[metadata[variable]["class"]]
    if bool((lookup < 0).any()):
        raise ValueError("Variable IDs must be contiguous and present in metadata")
    return lookup, len(classes)


def _torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_los_pre_model(args):
    """Allocate the historical backbone capacity for the selected dataset."""

    return EHRLargeGraphPreModel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        nhead=args.num_heads,
        activation=args.activation,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        mask_rate=args.mask_rate,
        drop_edge_rate=args.drop_edge_rate,
        alpha_l=args.alpha_l,
        var2ind_path=args.model_var2ind_path,
        loss_lmb=args.loss_lmb,
        device=args.device,
        norm=args.norm,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        k=args.knn,
        negative_slope=args.negative_slope,
        cat_var=args.cat_var,
        mask_t1=-100,
        mask_t2=-100,
    )


def load_pretrained_backbone(model, paths: dict[str, Path], device) -> None:
    """Load an encoder and align source/target variable embeddings by name."""

    checkpoint = paths["pretrained_checkpoint"]
    state = _torch_load(checkpoint, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = dict(state)
    expected = model.pre_model.state_dict()
    embedding_key = "variable_emb.weight"
    if embedding_key not in state or embedding_key not in expected:
        raise KeyError(f"Missing {embedding_key!r} in pretrained model or checkpoint")
    with paths["pretrained_var2ind"].open("r", encoding="utf-8") as handle:
        source_vocab = json.load(handle)
    with paths["var2ind"].open("r", encoding="utf-8") as handle:
        target_vocab = json.load(handle)
    source_embedding = state[embedding_key]
    target_embedding = expected[embedding_key].clone()
    if source_embedding.shape[1:] != target_embedding.shape[1:]:
        raise RuntimeError("Pretrained variable embedding dimensions do not match")
    if max(map(int, source_vocab.values()), default=-1) >= source_embedding.size(0):
        raise RuntimeError("Pretrained source vocabulary exceeds checkpoint capacity")
    if max(map(int, target_vocab.values()), default=-1) >= target_embedding.size(0):
        raise RuntimeError("Dataset vocabulary exceeds model embedding capacity")
    if source_vocab != target_vocab or source_embedding.shape != target_embedding.shape:
        shared = set(source_vocab) & set(target_vocab)
        for variable in shared:
            target_embedding[int(target_vocab[variable])] = source_embedding[
                int(source_vocab[variable])
            ]
        state[embedding_key] = target_embedding
        LOGGER.info(
            "Aligned %d/%d target variable embeddings from %s",
            len(shared), len(target_vocab), checkpoint,
        )
    model.pre_model.load_state_dict(state, strict=True)


def build_los_model(args, paths: dict[str, Path], pretrain_config: dict, finetune_config: dict):
    """Build the task-specific model without touching the mortality model."""

    variable_to_class, num_classes_type = build_variable_class_lookup(
        paths["var2ind"], paths["var_mean_std"]
    )
    model_args = SimpleNamespace(
        num_heads=args.gat_heads,
        attn_drop=args.attn_drop,
        negative_slope=0.2,
        in_drop=args.dropout,
        knn=args.knn,
        norm="layernorm",
        drop_edge_rate=0.0,
        num_features=64,
        num_hidden=int(pretrain_config["num_hidden"]),
        num_layers=int(pretrain_config["num_layers"]),
        activation="prelu",
        mask_rate=float(pretrain_config["mask_rate"]),
        loss_lmb=0.0,
        alpha_l=2.0,
        encoder_type="gat",
        decoder_type="gat",
        model_var2ind_path=str(paths["model_var2ind"]),
        variable_to_class=variable_to_class,
        cat_var=False,
        device=args.device_object,
    )
    # HierarchicalEHRModel calls a module-level builder.  Patch it only for
    # this construction so the mortality/pretraining entry points are intact.
    with patch.object(finetune_graph_module, "build_model", build_los_pre_model):
        model = HierarchicalEHRLOSModel(
            input_dim=1,
            hidden_dim=model_args.num_hidden,
            args=model_args,
            num_classes_type=num_classes_type,
            num_clusters_k=int(finetune_config["num_clusters_k"]),
            num_demo=2,
            num_heads=int(finetune_config["ts_heads"]),
            num_layers=model_args.num_layers,
            dropout=args.dropout,
            aggr_type=finetune_config["aggr_type"],
            num_los_classes=NUM_LOS_CLASSES,
        ).to(args.device_object)

    if not args.no_pretrained:
        load_pretrained_backbone(model, paths, args.device_object)
    if args.freeze_encoder:
        for parameter in model.pre_model.parameters():
            parameter.requires_grad = False
    return model


def _ordinal_rps_loss(probabilities: torch.Tensor, labels: torch.Tensor):
    observed = F.one_hot(labels, num_classes=NUM_LOS_CLASSES).to(probabilities.dtype)
    cumulative_error = torch.cumsum(probabilities, dim=1)[:, :-1] - torch.cumsum(
        observed, dim=1
    )[:, :-1]
    return cumulative_error.square().sum(dim=1).div(NUM_LOS_CLASSES - 1).mean()


def _loss(logits, labels, ordinal_loss_weight: float):
    cross_entropy = F.cross_entropy(logits, labels)
    if ordinal_loss_weight == 0:
        return cross_entropy
    return cross_entropy + ordinal_loss_weight * _ordinal_rps_loss(
        torch.softmax(logits, dim=1), labels
    )


def forward_diffpool_with_confidence(model, batch, mode):
    """Mirror the mortality DiffPool assignment-entropy regularizer."""

    if mode != "diff_pool":
        raise ValueError("Confidence loss is only defined for diff_pool")
    captured_logits = []

    def capture_assignment_logits(module, inputs, output):
        captured_logits.append(output)

    handle = model.diffpool_proj1.register_forward_hook(capture_assignment_logits)
    try:
        logits = model(batch, mode=mode)
    finally:
        handle.remove()
    if len(captured_logits) != 1:
        raise RuntimeError(
            "Expected exactly one DiffPool assignment tensor, "
            f"captured {len(captured_logits)}"
        )
    assignments = F.softmax(captured_logits[0], dim=-1)
    if assignments.ndim != 2 or assignments.size(-1) < 2:
        raise RuntimeError(
            "DiffPool confidence loss requires assignment logits [N, K], K >= 2"
        )
    confidence_loss = -(
        assignments * torch.log(assignments.clamp_min(1e-9))
    ).sum(dim=-1).mean() / math.log(assignments.size(-1))
    return logits, confidence_loss


def _evaluate(model, loader, device, pool_mode, representative_days, ordinal_weight):
    model.eval()
    losses = []
    probabilities = []
    labels = []
    exact_days = []
    stay_ids = []
    icu_deaths = []
    hospital_deaths = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            logits = model(batch, mode=pool_mode)
            batch_labels = batch.y.view(-1).long()
            losses.append(float(_loss(logits, batch_labels, ordinal_weight).item()))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            exact_days.append(batch.remaining_los_days.view(-1).cpu().numpy())
            stay_ids.append(batch.ts_id.view(-1).cpu().numpy())
            icu_deaths.append(batch.icu_death.view(-1).cpu().numpy())
            hospital_deaths.append(batch.in_hospital_mortality.view(-1).cpu().numpy())

    arrays = {
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels).astype(np.int64),
        "remaining_los_days": np.concatenate(exact_days),
        "ts_id": np.concatenate(stay_ids).astype(np.int64),
        "icu_death": np.concatenate(icu_deaths).astype(np.int64),
        "in_hospital_mortality": np.concatenate(hospital_deaths).astype(np.int64),
    }
    metrics = compute_los_metrics(
        arrays["labels"],
        arrays["probabilities"],
        arrays["remaining_los_days"],
        representative_days,
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics, arrays


def _is_better(metrics: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if metrics["linear_kappa"] != best["linear_kappa"]:
        return metrics["linear_kappa"] > best["linear_kappa"]
    return metrics["rps"] < best["rps"]


def _tensor_diagnostics(value: torch.Tensor) -> dict:
    """Summarize a tensor without printing clinical measurements or full arrays."""

    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite].float()
    summary = {
        "shape": list(detached.shape),
        "finite": int(finite.sum().item()),
        "total": detached.numel(),
    }
    if finite_values.numel():
        summary["min"] = float(finite_values.min().item())
        summary["max"] = float(finite_values.max().item())
    return summary


def _diagnose_time_groups(batch) -> dict:
    """Expose gaps that can invalidate time-view dense batching."""

    if not hasattr(batch, "batch_time") or not hasattr(batch, "batch"):
        return {"available": False}
    group_index = batch.batch_time.detach().view(-1).long()
    node_batch = batch.batch.detach().view(-1).long()
    if group_index.numel() != node_batch.numel() or not group_index.numel():
        return {"nodes": int(node_batch.numel()), "time_indices": int(group_index.numel())}
    if bool((group_index < 0).any()):
        return {"negative_time_indices": int((group_index < 0).sum().item())}
    num_slots = int(group_index.max().item()) + 1
    if num_slots > 100_000:
        return {"slots": num_slots, "error": "time index is unexpectedly large"}
    counts = torch.bincount(group_index, minlength=num_slots)
    slot_patient = scatter(
        node_batch, group_index, dim=0, dim_size=num_slots, reduce="min"
    )
    return {
        "slots": num_slots,
        "empty_slots": int((counts == 0).sum().item()),
        "slot_patient_monotonic": bool(
            (slot_patient[1:] >= slot_patient[:-1]).all().item()
        ),
        "slot_patient_prefix": slot_patient[:80].cpu().tolist(),
    }


def _replay_nonfinite_forward(model, batch, mode: str) -> None:
    """Locate the first non-finite leaf-module output on a diagnostic replay."""

    first_bad = []
    handles = []

    def inspect_output(name):
        def hook(_module, _inputs, output):
            if first_bad:
                return
            values = output if isinstance(output, (tuple, list)) else (output,)
            for value in values:
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
                    first_bad.append((name, _tensor_diagnostics(value)))
                    break
        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(inspect_output(name)))
    try:
        with torch.no_grad():
            replay_logits = model(batch, mode=mode)
        LOGGER.error(
            "LOS_NAN_REPLAY first_nonfinite_module=%s replay_logits=%s",
            first_bad[0] if first_bad else None,
            _tensor_diagnostics(replay_logits),
        )
        if not first_bad and bool(torch.isfinite(replay_logits).all().item()):
            LOGGER.error(
                "LOS_NAN_REPLAY did not reproduce the failure; training-mode "
                "dropout or a transient numerical operation may be involved"
            )
    except Exception:
        LOGGER.exception("LOS_NAN_REPLAY raised while inspecting the failed batch")
    finally:
        for handle in handles:
            handle.remove()


def _log_nonfinite_training_state(
    model, batch, logits, loss, checkpoint_path: Path, epoch: int,
    batch_number: int, mode: str, *, replay_forward: bool,
    grad_norm: torch.Tensor | None = None,
) -> None:
    stay_ids = (
        batch.ts_id.detach().view(-1).cpu().tolist()
        if hasattr(batch, "ts_id") else None
    )
    labels = (
        batch.y.detach().view(-1).cpu().tolist()
        if hasattr(batch, "y") else None
    )
    LOGGER.error(
        "LOS_NAN_DIAGNOSTIC checkpoint=%s epoch=%d batch=%d mode=%s "
        "ts_id=%s labels=%s loss=%s logits=%s grad_norm=%s",
        checkpoint_path, epoch, batch_number, mode, stay_ids, labels,
        _tensor_diagnostics(loss), _tensor_diagnostics(logits),
        _tensor_diagnostics(grad_norm) if grad_norm is not None else None,
    )
    for name in ("x_val", "x_time", "x_var", "demo", "batch_time"):
        value = getattr(batch, name, None)
        if isinstance(value, torch.Tensor):
            LOGGER.error("LOS_NAN_INPUT %s=%s", name, _tensor_diagnostics(value))
    if mode == "time":
        LOGGER.error("LOS_NAN_TIME_GROUPS %s", _diagnose_time_groups(batch))
    for kind, values in (
        ("parameter", ((name, param.data) for name, param in model.named_parameters())),
        ("gradient", ((name, param.grad) for name, param in model.named_parameters())),
    ):
        nonfinite = [
            (name, _tensor_diagnostics(value))
            for name, value in values
            if value is not None and not bool(torch.isfinite(value).all().item())
        ]
        LOGGER.error("LOS_NAN_%s nonfinite=%s", kind.upper(), nonfinite[:20])
    if replay_forward:
        _replay_nonfinite_forward(model, batch, mode)


def _train_one_repeat(
    model,
    train_loader,
    val_loader,
    args,
    representative_days,
    checkpoint_path: Path,
):
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch
    )
    best_metrics = None
    use_confidence = args.pool_mode == "diff_pool" and args.lambda_confidence > 0
    for epoch in range(args.max_epoch):
        model.train()
        confidence_sum = 0.0
        example_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epoch}", leave=False)
        for batch_number, batch in enumerate(progress, start=1):
            batch = batch.to(args.device_object)
            labels = batch.y.view(-1).long()
            optimizer.zero_grad(set_to_none=True)
            if use_confidence:
                logits, confidence_loss = forward_diffpool_with_confidence(
                    model, batch, args.pool_mode
                )
            else:
                logits = model(batch, mode=args.pool_mode)
                confidence_loss = logits.new_zeros(())
            loss = _loss(logits, labels, args.ordinal_loss_weight)
            if use_confidence:
                loss = loss + args.lambda_confidence * confidence_loss
            if not torch.isfinite(loss):
                try:
                    _log_nonfinite_training_state(
                        model, batch, logits, loss, checkpoint_path, epoch + 1,
                        batch_number, args.pool_mode, replay_forward=True,
                    )
                except Exception:
                    LOGGER.exception("LOS_NAN_DIAGNOSTIC failed while inspecting loss")
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if not torch.isfinite(grad_norm):
                try:
                    _log_nonfinite_training_state(
                        model, batch, logits, loss, checkpoint_path, epoch + 1,
                        batch_number, args.pool_mode, replay_forward=False,
                        grad_norm=grad_norm,
                    )
                except Exception:
                    LOGGER.exception("LOS_NAN_DIAGNOSTIC failed while inspecting gradients")
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch + 1}, "
                    f"batch {batch_number}; optimizer step was skipped"
                )
            optimizer.step()
            confidence_sum += float(confidence_loss.detach().item()) * len(labels)
            example_count += len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_metrics, _ = _evaluate(
            model,
            val_loader,
            args.device_object,
            args.pool_mode,
            representative_days,
            args.ordinal_loss_weight,
        )
        LOGGER.info(
            "epoch=%d train_confidence_loss=%.4f val_kappa=%.4f "
            "val_rps=%.4f val_mae_days=%.4f",
            epoch + 1,
            confidence_sum / max(example_count, 1),
            val_metrics["linear_kappa"],
            val_metrics["rps"],
            val_metrics["mae_days"],
        )
        if _is_better(val_metrics, best_metrics):
            best_metrics = val_metrics.copy()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": epoch + 1,
                    "representative_days": torch.as_tensor(
                        representative_days, dtype=torch.float32
                    ),
                    "validation_metrics": best_metrics,
                    "los_class_names": LOS_CLASS_NAMES,
                    "lambda_confidence": (
                        args.lambda_confidence if use_confidence else 0.0
                    ),
                },
                checkpoint_path,
            )
    if best_metrics is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    return best_metrics


def _make_dataset(ids, target_frame, paths, landmark_hours):
    return LOSLargeGraphDataset(
        root=os.fspath(paths["data_root"]),
        ts_id=ids,
        var2ind_path=os.fspath(paths["var2ind"]),
        var_mean_std_path=os.fspath(paths["var_mean_std"]),
        max_nodes=1000,
        time_limit=landmark_hours,
        pkl_path=os.fspath(paths["pkl_path"]),
        target_frame=target_frame,
        filter_before_windowing=True,
        window_start_minute=0,
    )


def _prediction_frame(arrays, representative_days, dataset, pool_mode, fold, repeat):
    probabilities = arrays["probabilities"]
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "pool_mode": pool_mode,
            "fold": fold,
            "repeat": repeat,
            "ts_id": arrays["ts_id"],
            "true_class": arrays["labels"],
            "pred_class": predictions,
            "remaining_los_days": arrays["remaining_los_days"],
            "predicted_los_days": representative_days[predictions],
            "icu_death": arrays["icu_death"],
            "in_hospital_mortality": arrays["in_hospital_mortality"],
        }
    )
    for class_index in range(NUM_LOS_CLASSES):
        frame[f"prob_{class_index}"] = probabilities[:, class_index]
    return frame


def _write_results(
    result_dir: Path,
    prefix: str,
    fold_rows,
    repeat_rows,
    predictions,
    labels,
):
    result_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(result_dir / f"{prefix}_fold_metrics.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(
        result_dir / f"{prefix}_repeat_metrics.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        result_dir / f"{prefix}_test_predictions.csv", index=False
    )
    labels.to_csv(result_dir / "los24_labels.csv", index=False)

    metric_columns = [
        "test_linear_kappa",
        "test_rps",
        "test_mae_days",
        "test_macro_f1",
        "test_accuracy",
        "test_within_1_bin_accuracy",
    ]
    summary = pd.DataFrame(
        [
            {
                "dataset": fold_frame["dataset"].iloc[0],
                "pool_mode": fold_frame["pool_mode"].iloc[0],
                "metric": metric.removeprefix("test_"),
                "mean": fold_frame[metric].mean(),
                "std": fold_frame[metric].std(ddof=0),
                "n_folds": len(fold_frame),
            }
            for metric in metric_columns
        ]
    )
    summary.to_csv(result_dir / f"{prefix}_summary.csv", index=False)


def main(args):
    paths = resolve_los_paths(args)
    validate_los_paths(paths, no_pretrained=args.no_pretrained)

    set_random_seed(args.seed)
    train_ids, val_ids, test_ids = _load_fold_metadata(
        args.dataset, paths["data_root"], paths["pkl_path"]
    )
    development_ids = np.concatenate([train_ids, val_ids])
    cohort_ids = np.concatenate([development_ids, test_ids])
    target_frame = load_los_label_frame(
        paths["los_labels"],
        cohort_ids,
        dataset_name=args.dataset,
        landmark_hours=args.landmark_hours,
    )
    id_to_patient = dict(
        zip(target_frame["ts_id"].astype(int), target_frame["patient_id"])
    )
    splits = get_sliding_window_splits(
        development_ids,
        id_to_patient=id_to_patient,
        n_splits=args.n_folds,
        window_size=args.window_size,
        seed=args.seed,
    )
    if args.max_folds is not None:
        splits = splits[: args.max_folds]

    pretrain_config, finetune_config = load_los_configs(paths, args.dataset, args.pool_mode)
    result_dir = _resolve_path(args.results_root, "results") / args.dataset / "los"
    checkpoint_root = (
        _resolve_path(args.checkpoint_root, "checkpoints/los")
        / args.dataset / args.pool_mode
    )
    test_dataset = _make_dataset(
        test_ids, target_frame, paths, args.landmark_hours
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device_object.type == "cuda",
    }
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    fold_rows = []
    repeat_rows = []
    prediction_frames = []
    target_by_id = target_frame.set_index("ts_id")
    for fold_index, (fold_train_ids, fold_val_ids) in enumerate(splits, start=1):
        assert_split_integrity(
            fold_train_ids, fold_val_ids, test_ids, id_to_patient
        )
        train_targets = target_by_id.loc[list(map(int, fold_train_ids))]
        representatives = class_representative_days(
            train_targets["los_class"], train_targets["remaining_los_days"]
        )
        train_dataset = _make_dataset(
            fold_train_ids, target_frame, paths, args.landmark_hours
        )
        val_dataset = _make_dataset(
            fold_val_ids, target_frame, paths, args.landmark_hours
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

        best_repeat = None
        best_epoch = None
        best_val_metrics = None
        best_checkpoint = None
        selected_test_metrics = None
        selected_test_arrays = None
        for repeat_index in range(1, args.repeat_time + 1):
            repeat_seed = args.seed + fold_index * 1000 + repeat_index
            set_random_seed(repeat_seed)
            model = build_los_model(args, paths, pretrain_config, finetune_config)
            checkpoint_path = (
                checkpoint_root
                / f"fold_{fold_index}"
                / f"repeat_{repeat_index}"
                / "best.pt"
            )
            val_metrics = _train_one_repeat(
                model,
                train_loader,
                val_loader,
                args,
                representatives,
                checkpoint_path,
            )
            selected_by_validation = _is_better(val_metrics, best_val_metrics)
            if selected_by_validation:
                best_repeat = repeat_index
                best_val_metrics = val_metrics
                best_checkpoint = checkpoint_path

            # Test exactly once after this repeat has completed all epochs.  The
            # checkpoint contains the epoch selected solely on validation
            # metrics; test metrics below never participate in epoch/repeat
            # selection.
            repeat_checkpoint = _torch_load(checkpoint_path, args.device_object)
            repeat_best_epoch = int(repeat_checkpoint["best_epoch"])
            model.load_state_dict(repeat_checkpoint["model_state_dict"])
            test_metrics, test_arrays = _evaluate(
                model,
                test_loader,
                args.device_object,
                args.pool_mode,
                representatives,
                args.ordinal_loss_weight,
            )
            repeat_rows.append(
                {
                    "dataset": args.dataset,
                    "pool_mode": args.pool_mode,
                    "landmark_hours": args.landmark_hours,
                    "lambda_confidence": (
                        args.lambda_confidence if args.pool_mode == "diff_pool" else 0.0
                    ),
                    "fold": fold_index,
                    "repeat": repeat_index,
                    "best_epoch": repeat_best_epoch,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )
            LOGGER.info(
                "fold=%d repeat=%d best_epoch=%d "
                "test_loss=%.4f test_kappa=%.4f test_rps=%.4f "
                "test_mae_days=%.4f test_macro_f1=%.4f "
                "test_accuracy=%.4f test_within_1_bin_accuracy=%.4f",
                fold_index,
                repeat_index,
                repeat_best_epoch,
                test_metrics["loss"],
                test_metrics["linear_kappa"],
                test_metrics["rps"],
                test_metrics["mae_days"],
                test_metrics["macro_f1"],
                test_metrics["accuracy"],
                test_metrics["within_1_bin_accuracy"],
            )
            if selected_by_validation:
                best_epoch = repeat_best_epoch
                selected_test_metrics = test_metrics
                selected_test_arrays = test_arrays

        if (
            best_checkpoint is None
            or best_val_metrics is None
            or selected_test_metrics is None
            or selected_test_arrays is None
        ):
            raise RuntimeError(f"No checkpoint selected for fold {fold_index}")
        test_metrics = selected_test_metrics
        arrays = selected_test_arrays
        row = {
            "dataset": args.dataset,
            "pool_mode": args.pool_mode,
            "landmark_hours": args.landmark_hours,
            "lambda_confidence": (
                args.lambda_confidence if args.pool_mode == "diff_pool" else 0.0
            ),
            "fold": fold_index,
            "selected_repeat": best_repeat,
            "selected_epoch": best_epoch,
            **{f"val_{key}": value for key, value in best_val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        fold_rows.append(row)
        prediction_frames.append(
            _prediction_frame(
                arrays,
                representatives,
                args.dataset,
                args.pool_mode,
                fold_index,
                best_repeat,
            )
        )
        LOGGER.info(
            "fold=%d repeat=%d test_kappa=%.4f test_rps=%.4f test_mae_days=%.4f",
            fold_index,
            best_repeat,
            test_metrics["linear_kappa"],
            test_metrics["rps"],
            test_metrics["mae_days"],
        )

    prefix = f"{args.pool_mode}_{args.landmark_hours}h"
    _write_results(
        result_dir,
        prefix,
        fold_rows,
        repeat_rows,
        prediction_frames,
        target_frame,
    )
    LOGGER.info("LOS results written to %s", result_dir.resolve())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a ten-class remaining ICU LOS prediction head"
    )
    parser.add_argument("--dataset", type=_normalise_dataset_name, default="mimic3")
    parser.add_argument(
        "--pool-mode",
        dest="pool_mode",
        choices=("type", "time", "node_time", "diff_pool"),
        default="type",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--repeat-time", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--lambda-confidence", type=float, default=0.0044)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn-drop", type=float, default=0.2)
    parser.add_argument("--gat-heads", type=int, default=8)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--pkl-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--settings-root", "--settings_root", default="settings")
    parser.add_argument("--model-var2ind-path")
    parser.add_argument("--pretrained-var2ind-path")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--checkpoint-root", default="checkpoints/los")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parsed_args = build_parser().parse_args()
    if parsed_args.max_epoch <= 0 or parsed_args.repeat_time <= 0:
        raise ValueError("max_epoch and repeat_time must be positive")
    if parsed_args.ordinal_loss_weight < 0:
        raise ValueError("ordinal_loss_weight cannot be negative")
    if parsed_args.lambda_confidence < 0:
        raise ValueError("lambda_confidence cannot be negative")
    parsed_args.device_object = torch.device(
        f"cuda:{parsed_args.device}" if torch.cuda.is_available() else "cpu"
    )
    main(parsed_args)
