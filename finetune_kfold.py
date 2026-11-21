import argparse
import gc
import json
import logging
import math
import os
import pickle
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import models.graphmae.finetune_graph_model as finetune_graph_module
from dataset.mimic3_patient_dataset import ICULargeGraphDataset
from logger import Logger
from models.graphmae.finetune_graph_model import HierarchicalEHRModel
from models.graphmae.mimic_largegraph_premodel import EHRLargeGraphPreModel
from models.graphmae.type_group_indices import normalize_batched_type_indices
from models.graphmae.utils import (
    PRETRAINED_CHECKPOINT_FILENAMES,
    load_pretrain_profile,
    set_random_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "mimic3": {
        "directory": "mimic3",
        "pickle": "mimic_iii.pkl",
        "pretrain_profile": "mimic3",
        "output_prefix": "mm3",
        "source_vocabulary_directory": "mimic3",
        "model_vocabulary_directory": "mimic3",
    },
    "mimic4": {
        "directory": "mimic-iv",
        "pickle": "mimic_iv.pkl",
        "pretrain_profile": "mimic4",
        "output_prefix": "mm4",
        "source_vocabulary_directory": "mimic-iv",
        # Historical MIMIC-IV pretraining/fine-tuning encoded graph nodes with
        # the 124-entry MIMIC-IV vocabulary, but allocated variable_emb from
        # the 129-entry MIMIC-III vocabulary.  Only the latter file's length
        # is used by the model; its name -> ID semantics are not used.
        "model_vocabulary_directory": "mimic3",
    },
    "eicu": {
        "directory": "eicu",
        "pickle": "eicu.pkl",
        # eICU fine-tuning transfers the existing MIMIC-IV pretrained encoder.
        "pretrain_profile": "mimic4",
        "output_prefix": "eicu",
        "source_vocabulary_directory": "mimic-iv",
        # eICU uses the MIMIC-IV variable IDs and the 129-row embedding layout.
        "model_vocabulary_directory": "mimic3",
    },
}

DEFAULT_PRETRAIN_PROFILES = {
    "mimic3": {"num_hidden": 256, "num_layers": 2, "mask_rate": 0.5},
    "mimic4": {"num_hidden": 256, "num_layers": 2, "mask_rate": 0.5},
}

# Final structure-view (DiffPool) configuration selected from the repeated
# ablation experiments.  These values are applied only when the loaded
# fine-tuning configuration uses ``pool_mode == "diff_pool"``.  The command
# line alias ``--pool_mode diff`` still selects that JSON configuration.
DIFFPOOL_LR = 9e-5
DIFFPOOL_DROPOUT = 0.1
DIFFPOOL_NUM_CLUSTERS = 15
DIFFPOOL_LAMBDA_CONFIDENCE = 0.0044


def relative_to_project(path_value):
    """Resolve a CLI path relative to this script instead of the current shell."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset_config(args):
    values = dict(DATASET_CONFIGS[args.dataset])
    data_root_base = relative_to_project(args.data_root)
    settings_root = relative_to_project(args.settings_root)
    checkpoint_root = relative_to_project(args.checkpoint_root)
    output_root = relative_to_project(args.output_root)
    data_root = data_root_base / values["directory"]

    pretrained_checkpoint = getattr(args, "pretrained_checkpoint", None)
    checkpoint_path = (
        relative_to_project(pretrained_checkpoint)
        if pretrained_checkpoint is not None
        else checkpoint_root / PRETRAINED_CHECKPOINT_FILENAMES[args.dataset]
    )
    settings_pool_mode = "diff" if args.pool_mode == "diff_pool" else args.pool_mode
    source_vocabulary_root = (
        data_root_base / values["source_vocabulary_directory"]
    )
    model_vocabulary_root = data_root_base / values["model_vocabulary_directory"]
    values.update(
        name=args.dataset,
        data_root=data_root,
        pkl_path=data_root / values["pickle"],
        var2ind_path=data_root / "var2ind.json",
        var_mean_std_path=data_root / "var_mean_std.json",
        source_var2ind_path=source_vocabulary_root / "var2ind.json",
        model_var2ind_path=model_vocabulary_root / "var2ind.json",
        pretrain_config_path=settings_root / "pretrain_model_params.json",
        finetune_config_path=settings_root / f"ft_param_{settings_pool_mode}.json",
        checkpoint_path=checkpoint_path,
        output_root=output_root,
    )
    return SimpleNamespace(**values)


def validate_required_paths(config, *, require_pretrained=True):
    required = {
        "dataset pickle": config.pkl_path,
        "variable index": config.var2ind_path,
        "normalization metadata": config.var_mean_std_path,
        "model embedding-size vocabulary": config.model_var2ind_path,
        "fine-tuning settings": config.finetune_config_path,
    }
    if require_pretrained:
        required.update({"source variable index": config.source_var2ind_path,
                         "pretrained checkpoint": config.checkpoint_path})
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The selected dataset is missing required files:\n" + "\n".join(missing)
        )


def is_structure_view(mode):
    return mode in ("diff", "diff_pool")


def forward_diffpool_with_confidence(model, batch, mode):
    """Return logits and normalized assignment entropy for DiffPool only.

    A temporary forward hook captures the assignment logits produced by the
    existing DiffPool layer.  This keeps the model implementation and every
    non-structure forward path unchanged.
    """
    if mode != "diff_pool":
        raise ValueError(
            "Confidence regularization is only defined for pool_mode='diff_pool'."
        )

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
            f"but captured {len(captured_logits)}."
        )

    assignments = F.softmax(captured_logits[0], dim=-1)
    if assignments.dim() != 2:
        raise RuntimeError(
            "DiffPool confidence loss expects node-wise assignment logits "
            f"with shape [N, K], but received {tuple(assignments.shape)}."
        )
    confidence_loss = -(
        assignments * torch.log(assignments.clamp_min(1e-9))
    ).sum(dim=-1).mean() / math.log(assignments.size(-1))
    return logits, confidence_loss


def normalize_collated_group_ids(batch, args):
    """Canonicalize the grouping index used by the selected pooling mode."""
    pool_mode = getattr(args, "pool_mode", "type")
    if pool_mode == "type":
        return normalize_batched_type_indices(
            batch,
            int(args.num_classes_type),
            variable_to_class=getattr(args, "variable_to_class", None),
        )

    if pool_mode == "time":
        # PyG already gives different patients disjoint time-index ranges.  A
        # final pairwise compaction also removes any internal empty index left
        # by filtering an entire time slot, without changing slot order.
        patient_time_pairs = torch.stack(
            [batch.batch.long(), batch.batch_time.long().view(-1)], dim=1
        )
        _, compact_time_ids = torch.unique(
            patient_time_pairs, dim=0, sorted=True, return_inverse=True
        )
        batch.batch_time = compact_time_ids
    return batch


def load_pretrain_args(config_path, profile):
    if not Path(config_path).is_file():
        if profile not in DEFAULT_PRETRAIN_PROFILES:
            raise KeyError(
                f"No built-in pretraining profile {profile!r}, and "
                f"configuration file is missing: {config_path}"
            )
        pre_param = dict(DEFAULT_PRETRAIN_PROFILES[profile])
        logger.log(
            f"Using built-in pretraining profile {profile!r}; optional "
            f"configuration file not found: {config_path}"
        )
        logger.log(pre_param)
        return pre_param

    pre_param = load_pretrain_profile(config_path, profile)
    logger.log(pre_param)
    return pre_param


def load_finetune_args(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        finetune_param = json.load(f)

    logger.log(finetune_param)
    return finetune_param


def validate(model, step, val_dataloader, device, writer, args):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), desc="Validating", leave=False):
            batch = batch.to(device)
            batch = normalize_collated_group_ids(batch, args)
            logits = model(batch, mode=args.pool_mode if hasattr(args, 'pool_mode') else 'type')
            loss = criterion(logits, batch.y.unsqueeze(1))
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    avg_loss = total_loss / len(val_dataloader)

    try:
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
        pr_auc = average_precision_score(all_labels.numpy(), all_probs.numpy())
        precision = precision_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
        recall = recall_score(all_labels.numpy(), all_preds.numpy(), zero_division=0)
        min_rp = min(precision, recall)
    except ValueError:
        auc = 0.5
        pr_auc = 0
        precision = 0
        recall = 0
        min_rp = 0

    f1 = f1_score(all_labels.numpy(), all_preds.numpy())

    if writer is not None:
        writer.add_scalar("valid/loss", avg_loss, global_step=step)
        writer.add_scalar("valid/auc", auc, global_step=step)
        writer.add_scalar("valid/f1", f1, global_step=step)

    logger.log(
        f"Step {step} | Val Loss: {avg_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | MIN(RE,PR): {min_rp:.4f}")

    model.train()
    return avg_loss, auc, pr_auc, min_rp


class CudaTrainingFailure(RuntimeError):
    """CUDA failure enriched with a persisted batch diagnostic."""


def is_cuda_runtime_error(error):
    if isinstance(error, CudaTrainingFailure):
        return True
    message = f"{type(error).__name__}: {error}".lower()
    markers = (
        "cuda",
        "cudnn",
        "cublas",
        "device-side assert",
        "device side assert",
        "driver shutting down",
        "unspecified launch failure",
    )
    return any(marker in message for marker in markers)


def describe_training_batch(batch):
    """Collect CPU-side batch metadata before any CUDA operation can fail."""

    metadata = {
        "num_graphs": int(getattr(batch, "num_graphs", 0)),
        "total_nodes": int(getattr(batch, "num_nodes", 0)),
    }
    stay_ids = getattr(batch, "ts_id", None)
    if isinstance(stay_ids, torch.Tensor):
        metadata["stay_ids"] = [
            int(value) for value in stay_ids.detach().cpu().view(-1).tolist()
        ]
    elif stay_ids is not None:
        try:
            metadata["stay_ids"] = [int(value) for value in stay_ids]
        except TypeError:
            metadata["stay_ids"] = [int(stay_ids)]

    graph_index = getattr(batch, "batch", None)
    if isinstance(graph_index, torch.Tensor) and graph_index.numel():
        graph_index_cpu = graph_index.detach().cpu().long().view(-1)
        counts = torch.bincount(
            graph_index_cpu,
            minlength=int(getattr(batch, "num_graphs", 0)),
        )
        metadata["nodes_per_graph"] = [int(value) for value in counts.tolist()]
        metadata["max_nodes_per_graph"] = int(counts.max().item())
        metadata["attention_elements_proxy"] = int(
            counts.numel() * int(counts.max().item()) ** 2
        )

    node_times = getattr(batch, "x_time", None)
    if isinstance(node_times, torch.Tensor) and node_times.numel():
        node_times_cpu = node_times.detach().cpu().float()
        metadata["node_time_min"] = float(node_times_cpu.min().item())
        metadata["node_time_max"] = float(node_times_cpu.max().item())
    return metadata


def cuda_memory_snapshot(device):
    snapshot = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "device": str(device),
    }
    if device.type != "cuda":
        return snapshot
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "free_mb": round(free_bytes / 1024**2, 2),
                "total_mb": round(total_bytes / 1024**2, 2),
                "allocated_mb": round(
                    torch.cuda.memory_allocated(device) / 1024**2, 2
                ),
                "reserved_mb": round(
                    torch.cuda.memory_reserved(device) / 1024**2, 2
                ),
                "max_allocated_mb": round(
                    torch.cuda.max_memory_allocated(device) / 1024**2, 2
                ),
                "max_reserved_mb": round(
                    torch.cuda.max_memory_reserved(device) / 1024**2, 2
                ),
            }
        )
    except Exception as error:
        snapshot["snapshot_error"] = f"{type(error).__name__}: {error}"
    return snapshot


def run_diagnostic_command(command, timeout_seconds=5):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": stdout_lines[-200:],
            "stderr_tail": stderr_lines[-200:],
        }
    except Exception as error:
        return {
            "command": command,
            "command_error": f"{type(error).__name__}: {error}",
        }


def linux_nvidia_diagnostics():
    """Best-effort Linux driver/Xid evidence without requiring root."""

    diagnostics = {
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "os_name": os.name,
    }
    if platform.system() != "Linux":
        return diagnostics

    version_path = Path("/proc/driver/nvidia/version")
    try:
        diagnostics["proc_driver_nvidia_version"] = version_path.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError as error:
        diagnostics["proc_driver_nvidia_version_error"] = (
            f"{type(error).__name__}: {error}"
        )

    gpu_information = {}
    for information_path in Path("/proc/driver/nvidia/gpus").glob(
        "*/information"
    ):
        try:
            gpu_information[str(information_path)] = information_path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as error:
            gpu_information[str(information_path)] = (
                f"{type(error).__name__}: {error}"
            )
    diagnostics["proc_driver_nvidia_gpus"] = gpu_information
    diagnostics["commands"] = [
        run_diagnostic_command(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,driver_version,pci.bus_id,"
                "temperature.gpu,power.draw,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            timeout_seconds=5,
        ),
        run_diagnostic_command(
            ["dmesg", "--ctime", "--level=err,warn"],
            timeout_seconds=3,
        ),
        run_diagnostic_command(
            ["journalctl", "-k", "--no-pager", "-n", "200"],
            timeout_seconds=3,
        ),
    ]
    return diagnostics


def atomic_write_json(path, payload, *, durable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(temporary, path)


def cuda_debug_directory(args):
    return (
        Path(getattr(args, "output_root", PROJECT_ROOT))
        / "logs"
        / str(getattr(args, "task_name", "unknown_cuda_task"))
    )


def persist_cuda_failure(args, payload):
    debug_dir = cuda_debug_directory(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = debug_dir / f"cuda_failure_{timestamp}.json"
    try:
        atomic_write_json(path, payload)
        return path, None
    except Exception as error:
        return path, f"{type(error).__name__}: {error}"


def train_classifier(model, dataloaders, optimizer, max_epoch, device, scheduler, args, writer=None, fold_idx=0,
                     turn=0):
    train_loader, eval_loader = dataloaders
    criterion = nn.BCEWithLogitsLoss()
    pool_mode = args.pool_mode if hasattr(args, 'pool_mode') else 'type'
    lambda_confidence = float(getattr(args, "lambda_confidence", 0.0))
    use_confidence_loss = pool_mode == "diff_pool" and lambda_confidence > 0.0

    global_step = 0
    best_auc = 0
    best_prauc = 0
    best_minrp = 0

    for epoch in range(max_epoch):
        model.train()
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"F{fold_idx + 1}-T{turn + 1} Ep {epoch}", leave=False)
        for i, batch in train_iter:
            global_step = epoch * len(train_loader) + i
            batch_metadata = describe_training_batch(batch)
            phase = "before_batch_to_device"
            cuda_snapshots = []
            if bool(getattr(args, "cuda_debug", False)) or bool(
                getattr(args, "cuda_sync", False)
            ):
                cuda_snapshots.append(
                    {"phase": phase, **cuda_memory_snapshot(device)}
                )
            if bool(getattr(args, "cuda_debug", False)):
                try:
                    atomic_write_json(
                        cuda_debug_directory(args) / "last_cuda_batch.json",
                        {
                            "fold": fold_idx + 1,
                            "repeat": turn + 1,
                            "epoch": epoch + 1,
                            "batch_index": i,
                            "global_step": global_step,
                            "phase": phase,
                            "batch": batch_metadata,
                            "cuda": (
                                cuda_snapshots[-1] if cuda_snapshots else None
                            ),
                        },
                        durable=False,
                    )
                except OSError as error:
                    logging.warning("Unable to persist CUDA batch debug file: %s", error)

            token_limit = int(
                getattr(args, "max_node_time_tokens", 0) or 0
            )
            observed_tokens = int(
                batch_metadata.get("max_nodes_per_graph", 0)
            )
            if (
                pool_mode == "node_time"
                and token_limit > 0
                and observed_tokens > token_limit
            ):
                abort_path = (
                    cuda_debug_directory(args)
                    / "controlled_large_batch_abort.json"
                )
                abort_payload = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "reason": "max_node_time_tokens_exceeded",
                    "limit": token_limit,
                    "observed": observed_tokens,
                    "fold": fold_idx + 1,
                    "repeat": turn + 1,
                    "epoch": epoch + 1,
                    "batch_index": i,
                    "batch": batch_metadata,
                    "cuda": cuda_snapshots[-1] if cuda_snapshots else None,
                }
                atomic_write_json(abort_path, abort_payload)
                raise RuntimeError(
                    "Controlled node_time safety abort before model forward: "
                    f"max_nodes_per_graph={observed_tokens} exceeds "
                    f"--max-node-time-tokens={token_limit}; "
                    f"diagnostic={abort_path}"
                )

            try:
                phase = "batch_to_device"
                batch = batch.to(device)
                batch = normalize_collated_group_ids(batch, args)

                phase = "forward"
                if use_confidence_loss:
                    logits, confidence_loss = forward_diffpool_with_confidence(
                        model, batch, pool_mode
                    )
                    bce_loss = criterion(logits, batch.y.unsqueeze(1))
                    loss = bce_loss + lambda_confidence * confidence_loss
                else:
                    logits = model(batch, mode=pool_mode)
                    loss = criterion(logits, batch.y.unsqueeze(1))
                if bool(getattr(args, "cuda_sync", False)) and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    cuda_snapshots.append(
                        {"phase": "after_forward", **cuda_memory_snapshot(device)}
                    )

                phase = "loss_finite_check"
                if not bool(torch.isfinite(loss).item()):
                    logging.warning(
                        "Found non-finite loss at epoch %d batch %d; skipping.",
                        epoch,
                        i,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                phase = "backward"
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if bool(getattr(args, "cuda_sync", False)) and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    cuda_snapshots.append(
                        {"phase": "after_backward", **cuda_memory_snapshot(device)}
                    )

                phase = "gradient_clip"
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                phase = "optimizer_step"
                optimizer.step()
                if bool(getattr(args, "cuda_sync", False)) and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    cuda_snapshots.append(
                        {
                            "phase": "after_optimizer_step",
                            **cuda_memory_snapshot(device),
                        }
                    )
            except RuntimeError as error:
                if not is_cuda_runtime_error(error):
                    raise
                diagnostic = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "phase": phase,
                    "fold": fold_idx + 1,
                    "repeat": turn + 1,
                    "epoch": epoch + 1,
                    "batch_index": i,
                    "global_step": global_step,
                    "pool_mode": pool_mode,
                    "batch": batch_metadata,
                    "cuda_snapshots_before_failure": cuda_snapshots,
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
                    "pytorch_cuda_alloc_conf": os.environ.get(
                        "PYTORCH_CUDA_ALLOC_CONF"
                    ),
                    "linux_nvidia_diagnostics": linux_nvidia_diagnostics(),
                    "traceback": traceback.format_exc(),
                }
                diagnostic_path, write_error = persist_cuda_failure(args, diagnostic)
                message = (
                    f"CUDA failure during {phase}; fold={fold_idx + 1}, "
                    f"repeat={turn + 1}, epoch={epoch + 1}, batch={i}, "
                    f"diagnostic={diagnostic_path}"
                )
                if write_error is not None:
                    message += f"; diagnostic_write_error={write_error}"
                logger.log(message)
                raise CudaTrainingFailure(message) from error

            if i % 100 == 0 and writer is not None:
                writer.add_scalar("ftrain/loss", loss.item(), global_step=global_step)
                if use_confidence_loss:
                    writer.add_scalar(
                        "ftrain/confidence_loss",
                        confidence_loss.item(),
                        global_step=global_step,
                    )
                train_iter.set_description(f"F{fold_idx + 1}-T{turn + 1} Ep {epoch} | Loss: {loss.item():.4f}")

        avg_loss, auc, pr_auc, min_rp = validate(model, global_step, eval_loader, device, writer, args)

        if (auc + pr_auc) > (best_auc + best_prauc):
            best_auc = auc
            best_prauc = pr_auc
            best_minrp = min_rp
            if args.save_model:
                save_dir = Path(args.output_root) / "checkpoints" / args.task_name
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           save_dir / f"checkpoint_fold{fold_idx + 1}_turn{turn + 1}_best.pt")

        if scheduler is not None:
            scheduler.step()

    return best_auc, best_prauc, best_minrp


def build_variable_class_lookup(var2ind_path, var_mean_std_path):
    with open(var2ind_path, "r", encoding="utf-8") as handle:
        var2ind = json.load(handle)
    with open(var_mean_std_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    classes = sorted({item["class"] for item in metadata.values()})
    class2ind = {class_name: index for index, class_name in enumerate(classes)}
    lookup = torch.full((len(var2ind),), -1, dtype=torch.long)
    for variable, variable_index in var2ind.items():
        if variable not in metadata:
            raise KeyError(f"{variable!r} is missing from {var_mean_std_path}")
        lookup[int(variable_index)] = class2ind[metadata[variable]["class"]]
    if bool((lookup < 0).any()):
        missing_indices = torch.nonzero(lookup < 0, as_tuple=False).view(-1).tolist()
        raise ValueError(f"Unmapped variable indices: {missing_indices[:10]}")
    return lookup, len(classes)


def build_dataset_pre_model(args):
    """Build the backbone with the vocabulary size used by pretraining.

    ``args.var2ind_path`` describes the graph-node IDs.  The separate model
    path exists because the historical MIMIC-IV experiment used its own
    124 node IDs inside a 129-row embedding table allocated from MIMIC-III.
    """
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
        loss_lmb=getattr(args, "loss_lmb", 0.2),
        device=args.device,
        norm=args.norm,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        k=args.knn,
        negative_slope=args.negative_slope,
        cat_var=args.cat_var,
        mask_t1=args.mask_t1,
        mask_t2=args.mask_t2,
    )


def load_pretrained_backbone(
    model, checkpoint_path, device, source_var2ind_path, target_var2ind_path
):
    """Load a backbone and align variable embeddings by variable name."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    state_dict = dict(state_dict)
    expected = model.pre_model.state_dict()

    embedding_key = "variable_emb.weight"
    if embedding_key not in state_dict or embedding_key not in expected:
        raise KeyError(f"Checkpoint has no {embedding_key!r}: {checkpoint_path}")

    with open(source_var2ind_path, "r", encoding="utf-8") as handle:
        source_var2ind = json.load(handle)
    with open(target_var2ind_path, "r", encoding="utf-8") as handle:
        target_var2ind = json.load(handle)

    source_embedding = state_dict[embedding_key]
    target_embedding = expected[embedding_key].clone()
    if source_embedding.shape[1:] != target_embedding.shape[1:]:
        raise RuntimeError(
            "Variable embedding dimensions differ: "
            f"checkpoint={tuple(source_embedding.shape)}, "
            f"model={tuple(target_embedding.shape)}"
        )
    source_max_index = max(map(int, source_var2ind.values()), default=-1)
    target_max_index = max(map(int, target_var2ind.values()), default=-1)
    if source_max_index >= source_embedding.shape[0]:
        raise RuntimeError(
            f"Checkpoint has {source_embedding.shape[0]} embedding rows, but "
            f"the source vocabulary uses index {source_max_index}."
        )
    if target_max_index >= target_embedding.shape[0]:
        raise RuntimeError(
            f"Model has {target_embedding.shape[0]} embedding rows, but the "
            f"dataset vocabulary uses index {target_max_index}."
        )
    if source_embedding.shape[0] > len(source_var2ind):
        logger.log(
            "Preserving legacy checkpoint capacity: "
            f"{source_embedding.shape[0]} embedding rows for "
            f"{len(source_var2ind)} source variable IDs; surplus rows are not "
            "addressed by the dataset."
        )

    shared_variables = sorted(set(source_var2ind) & set(target_var2ind))
    needs_embedding_remap = (
        source_var2ind != target_var2ind
        or source_embedding.shape != target_embedding.shape
    )
    if needs_embedding_remap:
        for variable in shared_variables:
            target_embedding[int(target_var2ind[variable])] = source_embedding[
                int(source_var2ind[variable])
            ]
        state_dict[embedding_key] = target_embedding
        logger.log(
            f"Transferred variable embeddings by name: {len(shared_variables)}/"
            f"{len(target_var2ind)} target variables matched."
        )

    shape_mismatches = {
        key: (tuple(value.shape), tuple(expected[key].shape))
        for key, value in state_dict.items()
        if key in expected and value.shape != expected[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"Pretrained checkpoint shape mismatch: {shape_mismatches}")

    result = model.pre_model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def build_finetune_models(pre_param, finetune_param, margs, dataset_config, *,
                          initialize_pretrained=True):
    knn = 10
    num_hidden = pre_param["num_hidden"]
    num_layers = pre_param.get("num_layers", 2)
    norm_type = "layernorm"

    gat_num_heads = 8
    gat_attn_drop = 0.2
    gat_negative_slope = 0.2

    feat_drop = 0.2
    mask_rate = pre_param["mask_rate"]
    sce_lmb, mse_lmb = 1, 0
    cat_var = False
    load_pre_emb = False
    pool_mode = finetune_param["pool_mode"]
    aggr_type = finetune_param["aggr_type"]
    structure_view = is_structure_view(pool_mode)
    if structure_view:
        num_clusters_k = DIFFPOOL_NUM_CLUSTERS
        finetune_lr = DIFFPOOL_LR
        model_dropout = DIFFPOOL_DROPOUT
    else:
        num_clusters_k = 10
        finetune_lr = 0.00005
        model_dropout = 0.2

    ts_heads = finetune_param["ts_heads"]

    variable_to_class, num_classes_type = build_variable_class_lookup(
        dataset_config.var2ind_path, dataset_config.var_mean_std_path
    )

    all_args = SimpleNamespace(
        num_heads=gat_num_heads,
        attn_drop=gat_attn_drop,
        negative_slope=gat_negative_slope,
        in_drop=feat_drop,
        knn=knn,
        norm=norm_type,
        drop_edge_rate=0.0,
        num_features=64,
        save_model=True,
        pool_mode=pool_mode,
        max_epoch=getattr(margs, "max_epoch", 15),
        num_hidden=num_hidden,
        num_layers=num_layers,
        activation="prelu",
        mask_rate=mask_rate,
        loss_lmb=sce_lmb,
        alpha_l=2.0,
        encoder_type="gat",
        decoder_type="gat",
        lr=finetune_lr,
        mse_lmb=mse_lmb,
        sce_lmb=sce_lmb,
        scheduler=True,
        weight_decay=0,
        var2ind_path=str(dataset_config.var2ind_path),
        model_var2ind_path=str(dataset_config.model_var2ind_path),
        device=f"cuda:{margs.device}" if torch.cuda.is_available() else "cpu",
        task_name=margs.task_name,
        lock_encoder=False,
        cat_var=cat_var,
        load_pre_embedding=load_pre_emb,
        ts_heads=ts_heads,
        num_clusters_k=num_clusters_k,
        load_pre_model=initialize_pretrained,
        mask_t1=0,
        mask_t2=0,
        aggr_type=aggr_type,
        dropout=model_dropout,
        lambda_confidence=(
            DIFFPOOL_LAMBDA_CONFIDENCE if pool_mode == "diff_pool" else 0.0
        ),
        num_classes_type=num_classes_type,
        variable_to_class=variable_to_class,
        normalize_group_ids=True,
        type_index_normalization="fixed_stride_v2",
        output_root=dataset_config.output_root,
        cuda_debug=bool(getattr(margs, "cuda_debug", False)),
        cuda_sync=bool(getattr(margs, "cuda_sync", False)),
        max_node_time_tokens=int(
            getattr(margs, "max_node_time_tokens", 0) or 0
        ),
    )
    all_args.variable_to_class = all_args.variable_to_class.to(all_args.device)
    # Keep the dataset vocabulary and the model allocation vocabulary
    # explicit.  For MIMIC-IV/eICU this intentionally reproduces the legacy
    # 124-input-ID / 129-embedding-row setup used by the MIMIC-IV backbone.
    finetune_graph_module.build_model = build_dataset_pre_model
    model = HierarchicalEHRModel(
        input_dim=1,
        hidden_dim=all_args.num_hidden,
        args=all_args,
        num_classes_type=all_args.num_classes_type,
        num_clusters_k=all_args.num_clusters_k,
        num_demo=2,
        num_heads=all_args.num_heads,
        num_layers=all_args.num_layers,
        dropout=all_args.dropout,
        aggr_type=all_args.aggr_type
    )
    model.to(all_args.device)

    if all_args.load_pre_model:
        logger.log(f"Loading pretrained backbone: {dataset_config.checkpoint_path}")
        load_pretrained_backbone(
            model,
            dataset_config.checkpoint_path,
            all_args.device,
            dataset_config.source_var2ind_path,
            dataset_config.var2ind_path,
        )
        if all_args.lock_encoder:
            for param in model.pre_model.parameters():
                param.requires_grad = False
    return model, all_args


def get_sliding_window_splits(ids, n_splits=6, window_size=3):
    """Build the same stay-ID sliding-window folds as the reference script."""
    chunks = np.array_split(ids, n_splits)
    splits = []
    for i in range(n_splits):
        train_chunk_indices = [
            (i + offset) % n_splits for offset in range(window_size)
        ]
        val_chunk_index = (i + window_size) % n_splits
        train_ids = np.concatenate([chunks[idx] for idx in train_chunk_indices])
        val_ids = chunks[val_chunk_index]
        splits.append((train_ids.tolist(), val_ids.tolist()))
    return splits


def load_protocol_metadata(config, n_folds=6, window_size=3):
    """Load the processed train/valid/test IDs and build reference-style folds."""
    with open(config.pkl_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 5:
        raise ValueError(
            f"Expected five objects in {config.pkl_path}, got {type(payload).__name__}"
        )
    events, outcomes, train_ids, val_ids, test_ids = payload
    del events, outcomes, payload
    gc.collect()

    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)
    test_ids = np.asarray(test_ids, dtype=np.int64)
    development_ids = np.concatenate([train_ids, val_ids])
    folds = get_sliding_window_splits(
        development_ids,
        n_splits=n_folds,
        window_size=window_size,
    )
    return development_ids, test_ids, folds


def make_dataset(config, stay_ids):
    return ICULargeGraphDataset(
        root=str(config.data_root),
        ts_id=stay_ids,
        var2ind_path=str(config.var2ind_path),
        var_mean_std_path=str(config.var_mean_std_path),
        max_nodes=1000,
        time_limit=24,
        pkl_path=str(config.pkl_path),
        filter_before_windowing=True,
    )


def fine_tune_task_name(output_prefix, pool_mode, fold_number, turn_number):
    """Return the checkpoint directory shared by training and distillation."""
    fold_number = int(fold_number)
    turn_number = int(turn_number)
    if fold_number < 1 or turn_number < 1:
        raise ValueError("fold_number and turn_number must be positive")
    return (
        f"{output_prefix}_gat_ft_pool_{pool_mode}_"
        f"fold{fold_number}_turn{turn_number}_auc"
    )


def fine_tune_checkpoint_path(config, pool_mode, fold_number, turn_number):
    task_name = fine_tune_task_name(
        config.output_prefix,
        pool_mode,
        fold_number,
        turn_number,
    )
    return (
        config.output_root
        / "checkpoints"
        / task_name
        / f"checkpoint_fold{fold_number}_turn{turn_number}_best.pt"
    )


def load_fine_tune_state(path):
    """Load a state dict on CPU so resuming does not spike CUDA memory."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def release_cuda_cache():
    gc.collect()
    # Do not touch CUDA again while unwinding an exception: after an NVIDIA
    # Xid/reset even empty_cache() can block or obscure the original failure.
    if sys.exc_info()[0] is None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as error:
            # Preserve the original training exception when the CUDA context
            # itself has already been lost after a driver reset.
            logging.warning("CUDA cache cleanup failed: %s", error)


def main(args):
    config = resolve_dataset_config(args)
    validate_required_paths(config)
    config.output_root.mkdir(parents=True, exist_ok=True)

    pargs = load_pretrain_args(
        config.pretrain_config_path, config.pretrain_profile
    )
    fargs = load_finetune_args(config.finetune_config_path)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    set_random_seed(args.seed)

    logger.log(
        f"Dataset={args.dataset}; data_root={config.data_root}; "
        f"pretrained_checkpoint={config.checkpoint_path}; "
        f"data_vocabulary={config.var2ind_path}; "
        f"model_capacity_vocabulary={config.model_var2ind_path}; "
        f"normalization_metadata={config.var_mean_std_path}"
    )
    logger.log("Loading the original train/valid/test stay-ID splits...")
    development_ids, test_ids, folds_data = load_protocol_metadata(
        config,
        n_folds=args.n_folds,
        window_size=args.window_size,
    )
    if args.max_folds is not None:
        folds_data = folds_data[: args.max_folds]

    start_fold = int(getattr(args, "start_fold", 1))
    start_repeat = int(getattr(args, "start_repeat", 1))
    if not 1 <= start_fold <= len(folds_data):
        raise ValueError(
            f"start_fold must lie in [1, {len(folds_data)}], got {start_fold}"
        )
    if not 1 <= start_repeat <= args.repeat_time:
        raise ValueError(
            f"start_repeat must lie in [1, {args.repeat_time}], got {start_repeat}"
        )
    logger.log(
        f"Training boundary: fold={start_fold}, repeat={start_repeat}; "
        "the boundary repeat restarts from epoch 1."
    )

    logger.log(
        f"Total Combined (Train+Val): {len(development_ids)}, "
        f"Test (Hold-out): {len(test_ids)}"
    )
    test_dataset = make_dataset(config, test_ids)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    cv_results = {
        "val_auc": [],
        "val_prauc": [],
        "test_auc": [],
        "test_prauc": [],
    }

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        fold_number = fold_idx + 1
        if fold_number < start_fold:
            message = (
                f"Skipping completed fold {fold_number}/{len(folds_data)} "
                f"before resume boundary {start_fold}/{start_repeat}."
            )
            print(message)
            logger.log(message)
            continue

        print(
            f"\n{'=' * 20} Start Fold {fold_number} / "
            f"{len(folds_data)} {'=' * 20}"
        )
        train_dataset = make_dataset(config, fold_train_ids)
        val_dataset = make_dataset(config, fold_val_ids)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

        fold_best_val_auc = 0.0
        fold_best_val_prauc = 0.0
        fold_best_checkpoint = None
        first_repeat = start_repeat if fold_number == start_fold else 1

        # Re-evaluate whichever earlier boundary-fold checkpoints still exist.
        # Missing historical checkpoints are allowed: the best-repeat selection
        # then covers the available historical repeats plus newly trained ones.
        if first_repeat > 1:
            historical = [
                (
                    repeat_number,
                    fine_tune_checkpoint_path(
                        config,
                        args.pool_mode,
                        fold_number,
                        repeat_number,
                    ),
                )
                for repeat_number in range(1, first_repeat)
            ]
            available_historical = [
                item for item in historical if item[1].is_file()
            ]
            missing_historical = [
                str(path) for _, path in historical if not path.is_file()
            ]
            if missing_historical:
                logger.log(
                    f"Fold {fold_number}: {len(missing_historical)} historical "
                    "repeat checkpoint(s) are missing and will be skipped: "
                    + "; ".join(missing_historical)
                )

            if available_historical:
                args.task_name = fine_tune_task_name(
                    config.output_prefix,
                    args.pool_mode,
                    fold_number,
                    available_historical[0][0],
                )
                resume_model = None
                resume_args = None
                try:
                    resume_model, resume_args = build_finetune_models(
                        pargs,
                        fargs,
                        args,
                        config,
                        initialize_pretrained=False,
                    )
                    for repeat_number, checkpoint in available_historical:
                        state = load_fine_tune_state(checkpoint)
                        resume_model.load_state_dict(state, strict=True)
                        del state
                        _, prior_auc, prior_prauc, _ = validate(
                            resume_model,
                            0,
                            val_loader,
                            device,
                            None,
                            resume_args,
                        )
                        logger.log(
                            f"Retained Fold {fold_number} Repeat {repeat_number}: "
                            f"Val AUC={prior_auc:.4f}, "
                            f"Val PR-AUC={prior_prauc:.4f}"
                        )
                        if (prior_auc + prior_prauc) > (
                            fold_best_val_auc + fold_best_val_prauc
                        ):
                            fold_best_val_auc = prior_auc
                            fold_best_val_prauc = prior_prauc
                            fold_best_checkpoint = checkpoint
                finally:
                    del resume_model, resume_args
                    release_cuda_cache()

        for repeat_number in range(first_repeat, args.repeat_time + 1):
            turn = repeat_number - 1
            print(
                f"  --- Repeat Turn: {repeat_number} / {args.repeat_time} ---"
            )
            args.task_name = fine_tune_task_name(
                config.output_prefix,
                args.pool_mode,
                fold_number,
                repeat_number,
            )
            model = None
            all_args = None
            optimizer = None
            scheduler = None
            fold_writer = None
            try:
                model, all_args = build_finetune_models(pargs, fargs, args, config)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=all_args.lr,
                    weight_decay=all_args.weight_decay,
                )
                scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=all_args.max_epoch
                    )
                    if all_args.scheduler
                    else None
                )
                fold_writer = SummaryWriter(
                    log_dir=str(config.output_root / "logs" / all_args.task_name)
                )
                best_val_auc, best_val_prauc, _ = train_classifier(
                    model,
                    (train_loader, val_loader),
                    optimizer,
                    all_args.max_epoch,
                    device,
                    scheduler,
                    all_args,
                    writer=fold_writer,
                    fold_idx=fold_idx,
                    turn=turn,
                )
                checkpoint = fine_tune_checkpoint_path(
                    config,
                    args.pool_mode,
                    fold_number,
                    repeat_number,
                )
                if not checkpoint.is_file():
                    raise RuntimeError(
                        f"Fold {fold_number} repeat {repeat_number} produced no "
                        f"checkpoint: {checkpoint}"
                    )
                if (best_val_auc + best_val_prauc) > (
                    fold_best_val_auc + fold_best_val_prauc
                ):
                    fold_best_val_auc = best_val_auc
                    fold_best_val_prauc = best_val_prauc
                    fold_best_checkpoint = checkpoint
            finally:
                if fold_writer is not None:
                    fold_writer.close()
                del model, all_args, optimizer, scheduler, fold_writer
                release_cuda_cache()

        if fold_best_checkpoint is None:
            raise RuntimeError(
                f"No validation-selected checkpoint for fold {fold_number}"
            )

        # Load the selected checkpoint into a fresh model.  This avoids keeping
        # an earlier repeat's model resident on the GPU throughout later repeats.
        selected_model = None
        selected_args = None
        try:
            selected_model, selected_args = build_finetune_models(
                pargs,
                fargs,
                args,
                config,
                initialize_pretrained=False,
            )
            state = load_fine_tune_state(fold_best_checkpoint)
            selected_model.load_state_dict(state, strict=True)
            del state
            _, test_auc, test_prauc, _ = validate(
                selected_model,
                0,
                test_loader,
                device,
                None,
                selected_args,
            )
        finally:
            del selected_model, selected_args
            release_cuda_cache()

        logger.log(
            f"Fold {fold_number} Best Result: "
            f"Val AUC={fold_best_val_auc:.4f}, "
            f"Val PR-AUC={fold_best_val_prauc:.4f}, "
            f"Test AUC={test_auc:.4f}, Test PR-AUC={test_prauc:.4f}"
        )
        cv_results["val_auc"].append(fold_best_val_auc)
        cv_results["val_prauc"].append(fold_best_val_prauc)
        cv_results["test_auc"].append(test_auc)
        cv_results["test_prauc"].append(test_prauc)

    logger.log("\n" + "=" * 30)
    logger.log(
        "Cyclic repeated-subsampling results for folds completed in this process:"
    )
    logger.log(
        f"Mean Val AUC:     {np.mean(cv_results['val_auc']):.4f} +- "
        f"{np.std(cv_results['val_auc']):.4f}"
    )
    logger.log(
        f"Mean Val PR-AUC:  {np.mean(cv_results['val_prauc']):.4f} +- "
        f"{np.std(cv_results['val_prauc']):.4f}"
    )
    logger.log(
        f"Mean Test AUC:    {np.mean(cv_results['test_auc']):.4f} +- "
        f"{np.std(cv_results['test_auc']):.4f}"
    )
    logger.log(
        f"Mean Test PR-AUC: {np.mean(cv_results['test_prauc']):.4f} +- "
        f"{np.std(cv_results['test_prauc']):.4f}"
    )
    logger.log("=" * 30 + "\n")


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_CONFIGS), default="mimic3"
    )
    parser.add_argument(
        "--pool_mode",
        type=str,
        default="type",
        choices=["type", "time", "node_time", "diff", "diff_pool"],
    )
    parser.add_argument(
        "--pretrained-checkpoint", "--pretrained_checkpoint", type=Path,
        help="Override the dataset's default pretrained encoder checkpoint.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeat_time", type=int, default=5)
    parser.add_argument("--max_epoch", type=int, default=15, help="Epochs per teacher repeat.")
    parser.add_argument("--unfreeze_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n_folds", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--max_folds", type=int)
    parser.add_argument(
        "--start_fold",
        "--start-fold",
        dest="start_fold",
        type=int,
        default=1,
        help="1-based fold at which training restarts.",
    )
    parser.add_argument(
        "--start_repeat",
        "--start-repeat",
        dest="start_repeat",
        type=int,
        default=1,
        help="1-based repeat at which the start fold restarts.",
    )
    parser.add_argument(
        "--cuda_debug",
        "--cuda-debug",
        dest="cuda_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist the current batch and a JSON report when CUDA fails.",
    )
    parser.add_argument(
        "--cuda_sync",
        "--cuda-sync",
        dest="cuda_sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronize after forward/backward/step to localize CUDA errors.",
    )
    parser.add_argument(
        "--max_node_time_tokens",
        "--max-node-time-tokens",
        dest="max_node_time_tokens",
        type=int,
        default=0,
        help=(
            "Abort on CPU before node_time forward when any graph exceeds this "
            "token count; 0 disables the safety guard."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Base directory containing mimic3, mimic-iv, and eicu.",
    )
    parser.add_argument(
        "--settings_root", "--settings-root",
        type=Path,
        default=Path("settings"),
        help="Directory containing ft_param_*.json and optional pretrain_model_params.json.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing the dataset pretrained checkpoint files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("."),
        help="Base directory for fine-tuned checkpoints, TensorBoard logs, and logs.",
    )
    return parser


if __name__ == "__main__":
    margs = build_parser().parse_args()
    if min(margs.repeat_time, margs.max_epoch) < 1 or margs.n_folds < 2:
        raise ValueError("repeat_time and max_epoch must be positive; n_folds must be at least 2")
    resolved_output_root = relative_to_project(margs.output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    logger = Logger(
        str(resolved_output_root / f"{margs.dataset}_{margs.pool_mode}.log")
    )
    print(margs)
    main(margs)
