"""Adapters for the main-table neural baselines using the fixed pickle splits."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    """Attach the original stay ID to legacy datasets that return tensors only."""

    def __init__(self, source, stay_ids):
        self.source = source
        self.ids = tuple(map(int, stay_ids))
        if len(source) != len(self.ids):
            raise ValueError("Legacy dataset length differs from the requested split")
        self.num_features = source.num_features
        self.demo_dim = source.num_demo

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return (self.ids[index], *self.source[index])


class IndexedTimeseriesDataset(Dataset):
    """Keep the original stay ID alongside a dictionary-valued baseline sample."""

    def __init__(self, source, stay_ids):
        self.source = source
        self.ids = tuple(map(int, stay_ids))
        if len(source) != len(self.ids):
            raise ValueError("Time-series dataset length differs from the requested split")
        self.demo_dim = source.num_demo

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {"stay_id": self.ids[index], **self.source[index]}


def collate_indexed_grud(samples):
    from dataset.mimic_timeseries_baseline_dataset import collate_grud

    stay_ids = torch.tensor([sample["stay_id"] for sample in samples], dtype=torch.long)
    batch = collate_grud(samples)
    batch["stay_id"] = stay_ids
    return batch


def _numeric_batch_details(batch, model_batch, model):
    """Build diagnostics only when a numerical check has failed."""
    input_nonfinite = {}
    input_max_abs = {}
    for key, value in model_batch.items():
        if not value.is_floating_point():
            continue
        finite = torch.isfinite(value)
        input_nonfinite[key] = int((~finite).sum().item())
        input_max_abs[key] = (
            float(value[finite].abs().max().item()) if bool(finite.any()) else None
        )
    bad_parameters = [
        name for name, parameter in model.named_parameters()
        if not bool(torch.isfinite(parameter).all())
    ]
    bad_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    weight_v_norms = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".weight_v") and parameter.ndim >= 2:
            norms = torch.linalg.vector_norm(
                parameter.detach().float(),
                dim=tuple(range(1, parameter.ndim)),
            )
            weight_v_norms[name] = float(norms.min().item())
    return (
        f"stay_ids={batch['stay_id'].tolist()}; "
        f"input_nonfinite={input_nonfinite}; input_max_abs={input_max_abs}; "
        f"nonfinite_parameters={bad_parameters}; nonfinite_gradients={bad_gradients}; "
        f"weight_v_min_filter_norm={weight_v_norms}"
    )


class TimeseriesBaselineAdapter:
    """Fixed-split adapter for the repository's GRU, GRU-D, SAnD and TCN code."""

    optimizer_kind = "adam"
    scheduler_kind = "none"
    supported_models = frozenset({"gru", "grud", "sand", "tcn"})

    def __init__(self, name, paths, config, seed, debug_numerics=False):
        if name not in self.supported_models:
            raise ValueError(f"Unsupported time-series baseline: {name}")
        from dataset.mimic_timeseries_baseline_dataset import MIMICTimeseriesDataStore

        self.name = name
        self.seed = int(seed)
        self.debug_numerics = bool(debug_numerics)
        self.effective_config = _settings(config, {
            "total_hours": 24, "time_step_minutes": 60,
            "max_timesteps": 880, "epochs": 40, "batch_size": 16,
            "eval_batch_size": 32, "learning_rate": 5e-4,
            "weight_decay": 1e-4, "gradient_clip": 0.3,
            "hid_dim": 64, "num_layers": 2, "num_heads": 16,
            "dropout": 0.2, "attention_dropout": 0.2,
            "kernel_size": 4, "r": 24, "M": 12,
            "pos_class_weight": 1.0, "init_checkpoint": None,
        }, name)
        cfg = self.effective_config
        self.default_epochs = int(cfg["epochs"])
        self.default_batch_size = int(cfg["batch_size"])
        self.default_eval_batch_size = int(cfg["eval_batch_size"])
        self.default_learning_rate = float(cfg["learning_rate"])
        self.default_weight_decay = float(cfg["weight_decay"])
        self.default_gradient_clip = (None if cfg["gradient_clip"] is None
                                      else float(cfg["gradient_clip"]))
        self.collate_fn = collate_indexed_grud if name == "grud" else None
        self.store = MIMICTimeseriesDataStore(
            paths.pickle_path, paths.metadata_path,
            total_hours=float(cfg["total_hours"]),
        )
        self.initialized_tensors = 0

    def dataset(self, stay_ids):
        from dataset.mimic_timeseries_baseline_dataset import MIMICTimeseriesDataset

        source = MIMICTimeseriesDataset(
            self.store, stay_ids, model_type=self.name,
            time_step_minutes=float(self.effective_config["time_step_minutes"]),
            max_timesteps=int(self.effective_config["max_timesteps"]),
            truncation_seed=None if tuple(map(int, stay_ids)) == tuple(map(int, self.store.train_ids))
            else self.seed,
        )
        return IndexedTimeseriesDataset(source, stay_ids)

    def build_model(self, device, demo_dim):
        from timeseries_baselines.common import build_model, load_matching_weights

        if demo_dim != self.store.num_demo:
            raise ValueError("Time-series demographic dimensions differ from data store")
        cfg = self.effective_config
        model_config = SimpleNamespace(
            time_step_minutes=float(cfg["time_step_minutes"]),
            hid_dim=int(cfg["hid_dim"]), num_layers=int(cfg["num_layers"]),
            num_heads=int(cfg["num_heads"]), dropout=float(cfg["dropout"]),
            attention_dropout=float(cfg["attention_dropout"]),
            kernel_size=int(cfg["kernel_size"]), r=int(cfg["r"]),
            M=int(cfg["M"]), pos_class_weight=float(cfg["pos_class_weight"]),
        )
        model = build_model(self.name, self.store, model_config).to(device)
        if cfg["init_checkpoint"] is not None:
            checkpoint = Path(cfg["init_checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Initialization checkpoint is missing: {checkpoint}")
            self.initialized_tensors, _ = load_matching_weights(model, checkpoint, device)
            if not self.initialized_tensors:
                raise ValueError(f"No compatible model tensors in {checkpoint}")
        return model

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        model.train()
        total, examples = 0.0, 0
        for batch_number, batch in enumerate(loader, start=1):
            model_batch = {key: value.to(device) for key, value in batch.items()
                           if key != "stay_id"}
            optimizer.zero_grad(set_to_none=True)
            context = f"epoch {getattr(self, 'current_epoch', '?')} batch {batch_number}"
            if getattr(self, "debug_numerics", False):
                bad_inputs = [
                    key for key, value in model_batch.items()
                    if value.is_floating_point() and not bool(torch.isfinite(value).all())
                ]
                bad_parameters = [
                    name for name, parameter in model.named_parameters()
                    if not bool(torch.isfinite(parameter).all())
                ]
                if bad_inputs or bad_parameters:
                    raise FloatingPointError(
                        f"{self.name} non-finite input/parameter before forward at {context}; "
                        + _numeric_batch_details(batch, model_batch, model)
                    )
            loss = model(**model_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{self.name} training produced non-finite loss at {context}; "
                    + _numeric_batch_details(batch, model_batch, model)
                )
            loss.backward()
            if getattr(self, "debug_numerics", False):
                bad_gradients = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if bad_gradients:
                    raise FloatingPointError(
                        f"{self.name} non-finite gradients after backward at {context}; "
                        + _numeric_batch_details(batch, model_batch, model)
                    )
            if gradient_clip is not None:
                if getattr(self, "debug_numerics", False):
                    try:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip, error_if_nonfinite=True
                        )
                    except RuntimeError as exc:
                        raise FloatingPointError(
                            f"{self.name} non-finite gradient norm at {context}; "
                            + _numeric_batch_details(batch, model_batch, model)
                        ) from exc
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            if getattr(self, "debug_numerics", False):
                bad_parameters = [
                    name for name, parameter in model.named_parameters()
                    if not bool(torch.isfinite(parameter).all())
                ]
                if bad_parameters:
                    raise FloatingPointError(
                        f"{self.name} non-finite parameters after optimizer step at {context}; "
                        + _numeric_batch_details(batch, model_batch, model)
                    )
            count = len(batch["labels"])
            total += float(loss.detach()) * count
            examples += count
        return total / max(examples, 1)

    def predict_batch(self, model, batch, device):
        inputs = {key: value.to(device) for key, value in batch.items()
                  if key not in {"stay_id", "labels"}}
        return batch["stay_id"], batch["labels"], model(**inputs)


class MultiScaleIndexedDataset(Dataset):
    def __init__(self, sources, stay_ids):
        self.sources = sources
        self.ids = tuple(map(int, stay_ids))
        if any(len(source) != len(self.ids) for source in sources):
            raise ValueError("Multi-scale source length differs from requested split")
        self.num_features = sources[0].num_features
        self.demo_dim = sources[0].num_demo
        if any(source.num_features != self.num_features or source.num_demo != self.demo_dim
               for source in sources):
            raise ValueError("Multi-scale source vocabularies differ")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        samples = [source[index] for source in self.sources]
        if any(not torch.equal(sample[1], samples[0][1]) or
               not torch.equal(sample[2], samples[0][2]) for sample in samples[1:]):
            raise ValueError("Multi-scale demographics or labels disagree")
        return self.ids[index], [sample[0] for sample in samples], samples[0][1], samples[0][2]


def _settings(supplied, defaults, model_name):
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {model_name} settings: {sorted(unknown)}")
    return {**defaults, **supplied}


class MatrixClassifierAdapter:
    """Reuse the original MTGNN or MSTD-GNN network and 24-hour matrix loader."""

    def __init__(self, name, paths, config):
        self.name = name
        self.paths = paths
        if name == "mtgnn":
            self.optimizer_kind = "adam"
            self.scheduler_kind = "cosine"
            defaults = {
                "slot_duration_mins": 120, "epochs": 10, "batch_size": 32,
                "eval_batch_size": 32, "learning_rate": 1e-3,
                "weight_decay": 1e-4, "gradient_clip": 5.0, "dropout": 0.3,
                "layers": 3, "conv_channels": 32, "residual_channels": 32,
                "skip_channels": 64, "end_channels": 128, "out_dim": 256,
            }
        elif name == "mstdgnn":
            self.optimizer_kind = "adamw"
            self.scheduler_kind = "cosine"
            defaults = {
                "slot_duration_mins": 60, "epochs": 10, "batch_size": 16,
                "eval_batch_size": 32, "learning_rate": 1.5e-4,
                "weight_decay": 0.0, "gradient_clip": None, "dropout": 0.3,
                "gnn_model_type": "dyGIN2d", "num_layers": 3, "groups": 4,
                "pool_ratio": 0.2, "kern_size": [3, 3, 3], "in_dim": 64,
                "hidden_dim": 128, "out_dim": 256,
            }
        else:
            raise ValueError(f"Unsupported matrix classifier: {name}")
        self.effective_config = _settings(config, defaults, name)
        if 1440 % int(self.effective_config["slot_duration_mins"]):
            raise ValueError("slot_duration_mins must divide 1440")
        self.default_epochs = int(self.effective_config["epochs"])
        self.default_batch_size = int(self.effective_config["batch_size"])
        self.default_eval_batch_size = int(self.effective_config["eval_batch_size"])
        self.default_learning_rate = float(self.effective_config["learning_rate"])
        self.default_weight_decay = float(self.effective_config["weight_decay"])
        clip = self.effective_config["gradient_clip"]
        self.default_gradient_clip = None if clip is None else float(clip)
        self.collate_fn = None
        self.input_features = None

    def dataset(self, stay_ids):
        from dataset.mimic3_MSTD_dataset import MIMIC3Dataset

        source = MIMIC3Dataset(
            str(self.paths.pickle_path), str(self.paths.metadata_path),
            tsid_list=list(stay_ids),
            slot_duration_mins=int(self.effective_config["slot_duration_mins"]),
        )
        if self.input_features is None:
            self.input_features = source.num_features
        elif self.input_features != source.num_features:
            raise ValueError("Feature vocabulary changed between splits")
        return IndexedDataset(source, stay_ids)

    def build_model(self, device, demo_dim):
        config = self.effective_config
        nodes = self.input_features + demo_dim
        slots = 1440 // int(config["slot_duration_mins"])
        if self.name == "mtgnn":
            from models.mtgnn.net import GTNetClassifier

            model = GTNetClassifier(
                num_nodes=nodes, seq_len=slots, in_dim=1,
                out_dim=int(config["out_dim"]), num_classes=2,
                dropout=float(config["dropout"]), device=str(device),
                layers=int(config["layers"]),
                conv_channels=int(config["conv_channels"]),
                residual_channels=int(config["residual_channels"]),
                skip_channels=int(config["skip_channels"]),
                end_channels=int(config["end_channels"]),
            )
        else:
            from models.mstdgnn.STDGNN import GNNTest

            model = GNNTest(
                gnn_model_type=config["gnn_model_type"],
                num_layers=int(config["num_layers"]),
                groups=int(config["groups"]),
                pool_ratio=float(config["pool_ratio"]),
                kern_size=config["kern_size"],
                in_dim=int(config["in_dim"]),
                hidden_dim=int(config["hidden_dim"]),
                out_dim=int(config["out_dim"]), seq_len=slots,
                num_nodes=nodes, num_classes=2, dropout=float(config["dropout"]),
            )
        return model.to(device)

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        model.train()
        criterion = nn.CrossEntropyLoss()
        total, examples = 0.0, 0
        for _, x, demo, labels in loader:
            x, demo, labels = x.to(device), demo.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, demo), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{self.name} training produced non-finite loss")
            loss.backward()
            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            examples += len(labels)
        return total / max(examples, 1)

    def predict_batch(self, model, batch, device):
        stay, x, demo, labels = batch
        probability = torch.softmax(model(x.to(device), demo.to(device)), dim=1)[:, 1]
        return stay, labels, probability


class MSTDMultiscaleModel(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def branch_logits(self, matrices, demo):
        return [branch(matrix, demo) for branch, matrix in zip(self.branches, matrices, strict=True)]

    def forward(self, matrices, demo):
        return torch.stack(self.branch_logits(matrices, demo)).mean(dim=0)


class MSTDMultiscaleAdapter:
    """Three independently trained MSTD branches with mean-logit prediction."""

    name = "mstdgnn_multiscale"
    optimizer_kind = "adamw"
    scheduler_kind = "none"

    def __init__(self, paths, config):
        self.paths = paths
        self.effective_config = _settings(config, {
            "scales": [60, 120, 240], "epochs": 10,
            "batch_size": 16, "eval_batch_size": 32,
            "learning_rate": 1e-4, "weight_decay": 1e-4,
            "gradient_clip": None, "dropout": 0.3,
            "gnn_model_type": "dyGIN2d", "num_layers": 3, "groups": 4,
            "pool_ratio": 0.2, "kern_size": [3, 3, 3],
            "in_dim": 64, "hidden_dim": 128, "out_dim": 256,
        }, self.name)
        scales = [int(scale) for scale in self.effective_config["scales"]]
        if not scales or len(scales) != len(set(scales)) or any(
            scale <= 0 or 1440 % scale for scale in scales
        ):
            raise ValueError("MSTD scales must be unique positive divisors of 1440")
        self.scales = scales
        self.default_epochs = int(self.effective_config["epochs"])
        self.default_batch_size = int(self.effective_config["batch_size"])
        self.default_eval_batch_size = int(self.effective_config["eval_batch_size"])
        self.default_learning_rate = float(self.effective_config["learning_rate"])
        self.default_weight_decay = float(self.effective_config["weight_decay"])
        clip = self.effective_config["gradient_clip"]
        self.default_gradient_clip = None if clip is None else float(clip)
        self.collate_fn = None
        self.input_features = None

    def dataset(self, stay_ids):
        from dataset.mimic3_MSTD_dataset import MIMIC3Dataset

        sources = [MIMIC3Dataset(
            str(self.paths.pickle_path), str(self.paths.metadata_path),
            tsid_list=list(stay_ids), slot_duration_mins=scale,
        ) for scale in self.scales]
        if self.input_features is None:
            self.input_features = sources[0].num_features
        elif self.input_features != sources[0].num_features:
            raise ValueError("MSTD vocabulary changed between splits")
        return MultiScaleIndexedDataset(sources, stay_ids)

    def build_model(self, device, demo_dim):
        from models.mstdgnn.STDGNN import GNNTest

        config = self.effective_config
        branches = [GNNTest(
            gnn_model_type=config["gnn_model_type"],
            num_layers=int(config["num_layers"]),
            groups=int(config["groups"]),
            pool_ratio=float(config["pool_ratio"]),
            kern_size=config["kern_size"],
            in_dim=int(config["in_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            out_dim=int(config["out_dim"]), seq_len=1440 // scale,
            num_nodes=self.input_features + demo_dim, num_classes=2,
            dropout=float(config["dropout"]),
        ) for scale in self.scales]
        return MSTDMultiscaleModel(branches).to(device)

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        model.train()
        criterion = nn.CrossEntropyLoss()
        total, examples = 0.0, 0
        for _, matrices, demo, labels in loader:
            matrices = [matrix.to(device) for matrix in matrices]
            demo, labels = demo.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Separate branch losses reproduce independent scale training.
            losses = [criterion(logits, labels)
                      for logits in model.branch_logits(matrices, demo)]
            loss = sum(losses)
            if not torch.isfinite(loss):
                raise FloatingPointError("Multi-scale MSTD training produced non-finite loss")
            loss.backward()
            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(labels) / len(losses)
            examples += len(labels)
        return total / max(examples, 1)

    def predict_batch(self, model, batch, device):
        stay, matrices, demo, labels = batch
        logits = model([matrix.to(device) for matrix in matrices], demo.to(device))
        return stay, labels, torch.softmax(logits, dim=1)[:, 1]


class StratsAdapter:
    name = "istrats"
    optimizer_kind = "adam"
    scheduler_kind = "none"

    def __init__(self, paths, config, args, protocol, *, initialize_pretrained=True):
        self.paths = paths
        self.initialize_pretrained = initialize_pretrained
        self.training_ids = frozenset(map(int, protocol.train_ids))
        self.effective_config = _settings(config, {
            "total_hours": 24, "max_seq_len": 880, "epochs": 40,
            "batch_size": 16, "eval_batch_size": 32,
            "learning_rate": 5e-4, "weight_decay": 1e-4,
            "gradient_clip": 0.3, "hid_dim": 64, "num_layers": 2,
            "num_heads": 16, "attention_dropout": 0.2, "dropout": 0.2,
            "pos_class_weight": 1.0, "M": 12, "kernel_size": 4, "r": 24,
        }, self.name)
        self.default_epochs = int(self.effective_config["epochs"])
        self.default_batch_size = int(self.effective_config["batch_size"])
        self.default_eval_batch_size = int(self.effective_config["eval_batch_size"])
        self.default_learning_rate = float(self.effective_config["learning_rate"])
        self.default_weight_decay = float(self.effective_config["weight_decay"])
        self.default_gradient_clip = float(self.effective_config["gradient_clip"])
        self.collate_fn = None
        self.num_features = None
        task = {"mimic3": "finetune_strats_mm3_extend",
                "mimic4": "finetune_strats_mm4",
                "eicu": "finetune_strats_mm4"}[args.dataset]
        self.pretrained_checkpoint = (
            args.strats_pretrained_checkpoint or args.checkpoint_root / task / "checkpoint_best.bin"
        ).resolve()
        if initialize_pretrained and not self.pretrained_checkpoint.is_file():
            raise FileNotFoundError(
                f"iStraTS pretrained checkpoint is missing: {self.pretrained_checkpoint}; "
                "provide --strats-pretrained-checkpoint"
            )
        self.pretrained_tensors_loaded = 0

    def dataset(self, stay_ids):
        from dataset.mimic_strats_dataset import MIMICStratsDataset

        evaluation = frozenset(map(int, stay_ids)) != self.training_ids
        source = MIMICStratsDataset(
            str(self.paths.pickle_path), str(self.paths.metadata_path),
            tsid_list=list(stay_ids),
            total_hours=int(self.effective_config["total_hours"]),
            max_seq_len=int(self.effective_config["max_seq_len"]),
            truncation_seed=0 if evaluation else None,
        )
        if self.num_features is None:
            self.num_features = source.num_features
        elif self.num_features != source.num_features:
            raise ValueError("iStraTS vocabulary changed between splits")
        return IndexedDataset(source, stay_ids)

    def build_model(self, device, demo_dim):
        from models.StraTS.modeling_strats import Strats

        config = self.effective_config
        model_args = SimpleNamespace(
            model_type="istrats", V=self.num_features, D=demo_dim,
            hid_dim=int(config["hid_dim"]), num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            attention_dropout=float(config["attention_dropout"]),
            dropout=float(config["dropout"]), pretrain=0,
            load_ckpt_path=str(self.pretrained_checkpoint),
            pos_class_weight=float(config["pos_class_weight"]),
            M=int(config["M"]), kernel_size=int(config["kernel_size"]),
            r=int(config["r"]),
        )
        model = Strats(model_args)
        if not self.initialize_pretrained:
            return model.to(device)
        source = torch.load(self.pretrained_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(source, dict) and "state_dict" in source:
            source = source["state_dict"]
        if not isinstance(source, dict):
            raise ValueError("iStraTS checkpoint must contain a state dictionary")
        target = model.state_dict()
        matched = {key: value for key, value in source.items()
                   if key in target and tuple(value.shape) == tuple(target[key].shape)}
        if not matched:
            raise ValueError(f"No compatible iStraTS pretrained tensors in {self.pretrained_checkpoint}")
        target.update(matched)
        model.load_state_dict(target)
        self.pretrained_tensors_loaded = len(matched)
        return model.to(device)

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        model.train()
        total, examples = 0.0, 0
        for _, values, times, varis, mask, demo, labels in loader:
            values, times, varis = values.to(device), times.to(device), varis.to(device)
            mask, demo, labels = mask.to(device), demo.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(values, times, varis, mask, demo, labels=labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("iStraTS training produced non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            examples += len(labels)
        return total / max(examples, 1)

    def predict_batch(self, model, batch, device):
        stay, values, times, varis, mask, demo, labels = batch
        probability = model(
            values.to(device), times.to(device), varis.to(device),
            mask.to(device), demo.to(device), labels=None,
        )
        return stay, labels, probability


class RaindropAdapter:
    name = "raindrop"
    optimizer_kind = "adam"
    scheduler_kind = "plateau"

    def __init__(self, paths, config):
        self.paths = paths
        self.effective_config = _settings(config, {
            "slot_duration_mins": 60, "epochs": 20, "batch_size": 32,
            "eval_batch_size": 32, "learning_rate": 1e-4,
            "weight_decay": 0.0, "gradient_clip": None,
            "dropout": 0.2, "d_ob": 4, "nhead": 2, "nhid": 256,
            "nlayers": 2, "statistics_file": None,
        }, self.name)
        legacy_stats = paths.data_dir / "var_mean_std_old.json"
        self.statistics_file = (
            Path(self.effective_config["statistics_file"]).resolve()
            if self.effective_config["statistics_file"] else
            (legacy_stats if paths.name == "mimic3" and legacy_stats.is_file()
             else paths.metadata_path)
        )
        if not self.statistics_file.is_file():
            raise FileNotFoundError(f"Raindrop statistics file is missing: {self.statistics_file}")
        self.effective_config["statistics_file"] = str(self.statistics_file)
        self.default_epochs = int(self.effective_config["epochs"])
        self.default_batch_size = int(self.effective_config["batch_size"])
        self.default_eval_batch_size = int(self.effective_config["eval_batch_size"])
        self.default_learning_rate = float(self.effective_config["learning_rate"])
        self.default_weight_decay = float(self.effective_config["weight_decay"])
        clip = self.effective_config["gradient_clip"]
        self.default_gradient_clip = None if clip is None else float(clip)
        self.collate_fn = None
        self.num_features = None

    def dataset(self, stay_ids):
        from dataset.mimic_raindrop_dataset import MIMIC3Dataset

        source = MIMIC3Dataset(
            str(self.paths.pickle_path), str(self.statistics_file),
            tsid_list=list(stay_ids),
            slot_duration_mins=int(self.effective_config["slot_duration_mins"]),
        )
        if self.num_features is None:
            self.num_features = source.num_features
        elif self.num_features != source.num_features:
            raise ValueError("Raindrop vocabulary changed between splits")
        return IndexedDataset(source, stay_ids)

    def build_model(self, device, demo_dim):
        from models.Raindrop.models_rd import Raindrop_v2

        config = self.effective_config
        slots = 1440 // int(config["slot_duration_mins"])
        return Raindrop_v2(
            d_inp=self.num_features, d_model=self.num_features * int(config["d_ob"]),
            nhead=int(config["nhead"]), nhid=int(config["nhid"]),
            nlayers=int(config["nlayers"]), dropout=float(config["dropout"]),
            max_len=slots, d_static=demo_dim, n_classes=2,
            global_structure=torch.ones(self.num_features, self.num_features, device=device),
        ).to(device)

    @staticmethod
    def _forward(model, x, mask, times, demo):
        src = torch.cat([x.permute(2, 0, 1), mask.permute(2, 0, 1)], dim=-1)
        lengths = (mask.sum(dim=1) > 0).sum(dim=1).long()
        return model(src, demo, times.permute(1, 0), lengths)[0]

    def train_epoch(self, model, loader, optimizer, device, gradient_clip):
        model.train()
        criterion = nn.CrossEntropyLoss()
        total, examples = 0.0, 0
        for _, x, mask, times, demo, labels in loader:
            x, mask, times = x.to(device), mask.to(device), times.to(device)
            demo, labels = demo.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self._forward(model, x, mask, times, demo), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Raindrop training produced non-finite loss")
            loss.backward()
            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            examples += len(labels)
        return total / max(examples, 1)

    def predict_batch(self, model, batch, device):
        stay, x, mask, times, demo, labels = batch
        logits = self._forward(
            model, x.to(device), mask.to(device), times.to(device), demo.to(device)
        )
        return stay, labels, torch.softmax(logits, dim=1)[:, 1]
