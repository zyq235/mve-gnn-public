"""Train the main-table baselines on ten-class remaining ICU LOS.

Each run uses the existing first-24h cohort, six patient-disjoint development
folds and the held-out test stays.  Mortality training entry points are not
imported for model fitting or modified by this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.los_baseline_dataset import (
    LOSBaselineDataset, LOSDataStore, collate_los_grud,
)
from dataset.los_patient_dataset import load_los_label_frame
from ML.los_metrics import (
    NUM_LOS_CLASSES, class_representative_days, compute_los_metrics,
)
from ML.los_smoke import subset_per_class
from split_protocol import assert_split_integrity, get_sliding_window_splits


LOG = logging.getLogger("los_main_table")
MODEL_NAMES = (
    "lr", "svm", "xgboost", "gru", "grud", "tcn", "sand", "mtgnn",
    "time2graphplus", "raindrop", "mstdgnn", "mstdgnn_multiscale",
)
DATASET_DIRS = {"mimic3": "mimic3", "mimic4": "mimic-iv", "eicu": "eicu"}
PICKLE_NAMES = {"mimic3": "mimic_iii.pkl", "mimic4": "mimic_iv.pkl", "eicu": "eicu.pkl"}
SEQUENCE_MODELS = {"gru", "grud", "tcn", "sand"}
CLASSICAL_MODELS = {"lr", "svm", "xgboost", "time2graphplus"}

DEFAULTS = {
    "lr": {"slot_minutes": 120, "C": 0.1, "max_iter": 1000},
    "svm": {"slot_minutes": 120, "C": 1.0},
    "xgboost": {"slot_minutes": 120, "max_depth": 5, "learning_rate": 0.05,
                 "n_estimators": 500, "early_stopping_rounds": 20, "n_jobs": 4},
    "time2graphplus": {"slot_minutes": 120, "K": 30, "C": 300,
                       "num_segment": 4, "seg_length": 3, "warp": 2,
                       "niter": 30, "core_batch_size": 200,
                       "max_depth": 5, "learning_rate": 0.05,
                       "n_estimators": 500, "early_stopping_rounds": 20,
                       "n_jobs": 4},
    "gru": {"slot_minutes": 60, "epochs": 40, "batch_size": 16,
            "learning_rate": 5e-4, "weight_decay": 1e-4, "hidden_dim": 64,
            "dropout": 0.2, "num_layers": 2, "num_heads": 16,
            "attention_dropout": 0.2, "kernel_size": 4, "r": 24, "M": 12},
    "grud": {"slot_minutes": 60, "epochs": 40, "batch_size": 16,
             "learning_rate": 5e-4, "weight_decay": 1e-4, "hidden_dim": 64,
             "dropout": 0.2, "num_layers": 2, "num_heads": 16,
             "attention_dropout": 0.2, "kernel_size": 4, "r": 24, "M": 12,
             "max_timesteps": 880},
    "sand": {"slot_minutes": 60, "epochs": 40, "batch_size": 16,
             "learning_rate": 5e-4, "weight_decay": 1e-4, "hidden_dim": 64,
             "dropout": 0.2, "num_layers": 2, "num_heads": 16,
             "attention_dropout": 0.2, "kernel_size": 4, "r": 24, "M": 12},
    "tcn": {"slot_minutes": 60, "epochs": 40, "batch_size": 16,
            "learning_rate": 5e-4, "weight_decay": 1e-4, "hidden_dim": 64,
            "dropout": 0.2, "num_layers": 2, "num_heads": 16,
            "attention_dropout": 0.2, "kernel_size": 4, "r": 24, "M": 12},
    "mtgnn": {"slot_minutes": 120, "epochs": 10, "batch_size": 32,
              "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.3,
              "layers": 3, "conv_channels": 32, "residual_channels": 32,
              "skip_channels": 64, "end_channels": 128, "out_dim": 256},
    "mstdgnn": {"slot_minutes": 60, "epochs": 10, "batch_size": 16,
                "learning_rate": 1.5e-4, "weight_decay": 0.0,
                "dropout": 0.3, "gnn_model_type": "dyGIN2d", "num_layers": 3,
                "groups": 4, "pool_ratio": 0.2, "kern_size": [3, 3, 3],
                "in_dim": 64, "hidden_dim": 128, "out_dim": 256},
    "mstdgnn_multiscale": {"scales": [60, 120, 240], "epochs": 10,
                           "batch_size": 16, "learning_rate": 1e-4,
                           "weight_decay": 1e-4, "dropout": 0.3,
                           "gnn_model_type": "dyGIN2d", "num_layers": 3,
                           "groups": 4, "pool_ratio": 0.2,
                           "kern_size": [3, 3, 3], "in_dim": 64,
                           "hidden_dim": 128, "out_dim": 256},
    "raindrop": {"slot_minutes": 60, "epochs": 20, "batch_size": 32,
                 "learning_rate": 1e-4, "weight_decay": 0.0,
                 "dropout": 0.2, "d_ob": 4, "nhead": 2,
                 "nhid": 256, "nlayers": 2, "statistics_file": None},
}


def load_config(name, config_path):
    config = dict(DEFAULTS[name])
    if config_path:
        supplied = json.loads(Path(config_path).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        config.update(supplied)
    return config


def load_split_ids(dataset, data_dir, pickle_path):
    if dataset == "mimic3":
        sidecar = data_dir / "mimic_iii_split_ids.pkl"
        if sidecar.exists():
            with sidecar.open("rb") as stream:
                split = pickle.load(stream)
            return tuple(np.asarray(split[key], dtype=np.int64)
                         for key in ("train_list", "val_list", "test_list"))
    if dataset == "eicu":
        sidecar = data_dir / "patient_split.csv"
        if sidecar.exists():
            frame = pd.read_csv(sidecar, usecols=["ts_id", "split"])
            split = frame["split"].str.lower()
            return tuple(frame.loc[split.isin(names), "ts_id"].to_numpy(dtype=np.int64)
                         for names in (("train",), ("val", "valid", "validation"), ("test",)))
    with pickle_path.open("rb") as stream:
        _, _, train, val, test = pickle.load(stream)
    return tuple(np.asarray(ids, dtype=np.int64) for ids in (train, val, test))


def build_model(name, config, store, device):
    if name in SEQUENCE_MODELS:
        from models.los_timeseries import LOS_TIMESERIES_MODELS

        args = SimpleNamespace(
            model_type=name, V=store.num_features, D=store.num_demo,
            T=1440 // int(config["slot_minutes"]),
            hid_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
            attention_dropout=float(config["attention_dropout"]),
            kernel_size=int(config["kernel_size"]),
            r=int(config["r"]), M=int(config["M"]),
            pretrain=0, load_ckpt_path=None, pos_class_weight=1.0,
        )
        model = LOS_TIMESERIES_MODELS[name](args)
    elif name == "mtgnn":
        from models.mtgnn.net import GTNetClassifier

        model = GTNetClassifier(
            num_nodes=store.num_features + store.num_demo,
            seq_len=1440 // int(config["slot_minutes"]), in_dim=1,
            out_dim=int(config["out_dim"]), num_classes=10,
            dropout=float(config["dropout"]), device=str(device),
            layers=int(config["layers"]),
            conv_channels=int(config["conv_channels"]),
            residual_channels=int(config["residual_channels"]),
            skip_channels=int(config["skip_channels"]),
            end_channels=int(config["end_channels"]),
        )
    elif name in {"mstdgnn", "mstdgnn_multiscale"}:
        from models.mstdgnn.STDGNN import GNNTest

        scales = (config["scales"] if name == "mstdgnn_multiscale"
                  else [config["slot_minutes"]])
        branches = [GNNTest(
            gnn_model_type=config["gnn_model_type"],
            num_layers=int(config["num_layers"]), groups=int(config["groups"]),
            pool_ratio=float(config["pool_ratio"]), kern_size=config["kern_size"],
            in_dim=int(config["in_dim"]), hidden_dim=int(config["hidden_dim"]),
            out_dim=int(config["out_dim"]), seq_len=1440 // int(scale),
            num_nodes=store.num_features + store.num_demo,
            num_classes=10, dropout=float(config["dropout"]),
        ) for scale in scales]
        if len(branches) == 1:
            model = branches[0]
        else:
            from main_table_adapters import MSTDMultiscaleModel

            model = MSTDMultiscaleModel(branches)
    elif name == "raindrop":
        from models.Raindrop.models_rd import Raindrop_v2

        model = Raindrop_v2(
            d_inp=store.num_features,
            d_model=store.num_features * int(config["d_ob"]),
            nhead=int(config["nhead"]), nhid=int(config["nhid"]),
            nlayers=int(config["nlayers"]), dropout=float(config["dropout"]),
            max_len=1440 // int(config["slot_minutes"]),
            d_static=store.num_demo, n_classes=10,
            global_structure=torch.ones(store.num_features, store.num_features,
                                        device=device),
        )
    else:
        raise ValueError(f"Unknown neural model: {name}")
    return model.to(device)


def forward_logits(name, model, batch, device, *, branch_logits=False):
    demo = batch["demo"].to(device)
    if name in {"gru", "sand", "tcn"}:
        return model(batch["ts"].to(device), demo)
    if name == "grud":
        return model(*(batch[key].to(device) for key in
                       ("x_t", "m_t", "delta_t", "seq_len")), demo)
    if name == "mstdgnn_multiscale":
        matrices = [matrix.to(device) for matrix in batch["matrices"]]
        return (model.branch_logits(matrices, demo) if branch_logits
                else model(matrices, demo))
    if name == "raindrop":
        matrix = batch["matrix"].to(device)
        mask = batch["mask"].to(device)
        src = torch.cat([matrix.permute(2, 0, 1), mask.permute(2, 0, 1)], dim=-1)
        lengths = (mask.sum(dim=1) > 0).sum(dim=1).long()
        return model(src, demo, batch["times"].to(device).permute(1, 0), lengths)[0]
    return model(batch["matrix"].to(device), demo)


def los_loss(logits, labels, ordinal_weight):
    value = F.cross_entropy(logits, labels)
    if ordinal_weight:
        target = F.one_hot(labels, NUM_LOS_CLASSES).to(logits.dtype)
        delta = torch.cumsum(torch.softmax(logits, dim=1), dim=1)[:, :-1]
        delta -= torch.cumsum(target, dim=1)[:, :-1]
        value += ordinal_weight * delta.square().sum(dim=1).div(9).mean()
    return value


def evaluate(name, model, loader, device, targets, representatives):
    model.eval()
    ids, probabilities, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = forward_logits(name, model, batch, device)
            ids.append(batch["stay_id"].numpy())
            labels.append(batch["labels"].numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    stay_ids = np.concatenate(ids).astype(np.int64)
    true_classes = np.concatenate(labels).astype(np.int64)
    probs = np.concatenate(probabilities)
    exact_days = targets.loc[stay_ids, "remaining_los_days"].to_numpy(dtype=float)
    if not np.array_equal(true_classes, targets.loc[stay_ids, "los_class"].to_numpy(dtype=int)):
        raise AssertionError("LOS labels and model inputs are misaligned")
    metrics = compute_los_metrics(true_classes, probs, exact_days, representatives)
    return metrics, stay_ids, probs


def better(candidate, current):
    return (current is None or
            candidate["linear_kappa"] > current["linear_kappa"] or
            (candidate["linear_kappa"] == current["linear_kappa"] and
             candidate["rps"] < current["rps"]))


def prediction_frame(args, fold, ids, probs, targets, representatives):
    predictions = probs.argmax(axis=1)
    frame = targets.loc[ids, ["patient_id", "remaining_los_days", "los_class",
                              "icu_death", "in_hospital_mortality"]].reset_index()
    frame.insert(0, "fold", fold)
    frame.insert(0, "model", args.model)
    frame.insert(0, "dataset", args.dataset)
    frame["pred_class"] = predictions
    frame["predicted_los_days"] = representatives[predictions]
    for cls in range(NUM_LOS_CLASSES):
        frame[f"prob_{cls}"] = probs[:, cls]
    return frame


def make_dataset(store, ids, label_by_stay, name, config, *, seed):
    return LOSBaselineDataset(
        store, ids, label_by_stay, name,
        slot_minutes=config.get("slot_minutes", 60),
        scales=config.get("scales", (60, 120, 240)),
        max_timesteps=config.get("max_timesteps", 880),
        truncation_seed=seed,
    )


def make_loader(dataset, name, batch_size, workers, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_los_grud if name == "grud" else None,
    )


def matrix_arrays(store, ids, labels, name, config, *, batch_size=256):
    source = make_dataset(store, ids, labels, name, config, seed=0)
    matrices, demographics = [], []
    for batch in make_loader(source, name, batch_size, 0, False):
        matrices.append(batch["matrix"].numpy())
        demographics.append(batch["demo"].numpy())
    matrix = np.concatenate(matrices)
    demo = np.concatenate(demographics)
    if name == "time2graphplus":
        return matrix.transpose(0, 2, 1), demo
    return np.concatenate([matrix.reshape(len(ids), -1), demo], axis=1)


def xgb_kappa_loss(y_true, y_pred):
    predictions = np.asarray(y_pred).reshape(-1, NUM_LOS_CLASSES).argmax(axis=1)
    return -float(cohen_kappa_score(
        np.asarray(y_true, dtype=int), predictions,
        labels=np.arange(NUM_LOS_CLASSES), weights="linear",
    ))


def build_xgb(config, seed):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob", num_class=NUM_LOS_CLASSES,
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        eval_metric=xgb_kappa_loss,
        n_jobs=int(config["n_jobs"]), random_state=seed,
    )


def balanced_weights(labels):
    counts = np.bincount(labels, minlength=NUM_LOS_CLASSES)
    return len(labels) / (NUM_LOS_CLASSES * counts[labels])


def fit_los_time2graphplus(args, config, store, train_ids, val_ids, test_ids,
                           label_by_stay, targets, representatives, fold,
                           checkpoint_dir, device, label_path=None):
    import faulthandler

    from main_table_classical import _Time2GraphCache
    from models.time2graphplus.los_adapter import LOSTime2GraphGAT

    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)
    if int(config["num_segment"]) * int(config["seg_length"]) != 1440 // int(config["slot_minutes"]):
        raise ValueError("Time2Graph+ segments must span exactly 24 hours")
    shapelet_workers = int(getattr(args, "time2graph_workers", 1))
    if shapelet_workers < 1:
        raise ValueError("--time2graph-workers must be positive")
    LOG.info("Time2Graph+ LOS fold=%d shapelet_workers=%d xgboost_threads=%d",
             fold, shapelet_workers, int(config["n_jobs"]))
    stage_files = {
        "core_fit": ("los_core.pt",),
        "core": ("los_core.pt", "train_features.npy", "train_labels.npy"),
        "validation": ("validation_features.npy", "validation_labels.npy"),
        "head": ("xgboost_head.json",),
        "test": ("test_features.npy", "test_labels.npy"),
        "result": ("test_probabilities.npy",),
    }
    paths = SimpleNamespace(
        pickle_path=getattr(store, "pkl_path", None),
        metadata_path=getattr(store, "json_path", None),
        los_labels_path=label_path,
    )
    protocol = SimpleNamespace(train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids)
    cache_args = SimpleNamespace(dataset=args.dataset, seed=args.seed + fold, device=str(device))
    cache = _Time2GraphCache(
        checkpoint_dir, paths, protocol, cache_args,
        {"settings": config, "fold": fold, "landmark_hours": args.landmark_hours},
        task="los", stage_files=stage_files,
    )
    train_y = np.asarray([label_by_stay[int(i)] for i in train_ids], dtype=int)
    val_y = np.asarray([label_by_stay[int(i)] for i in val_ids], dtype=int)
    test_y = np.asarray([label_by_stay[int(i)] for i in test_ids], dtype=int)

    def make_core(data_size):
        return LOSTime2GraphGAT(
            K=int(config["K"]), C=int(config["C"]),
            seg_length=int(config["seg_length"]), num_segment=int(config["num_segment"]),
            data_size=int(data_size), warp=int(config["warp"]),
            gpu_enable=device.type == "cuda", niter=int(config["niter"]),
            batch_size=int(config["core_batch_size"]), out_clf=True,
            n_class=NUM_LOS_CLASSES, njobs=shapelet_workers,
        )

    def save_core(core):
        path = cache.root / "los_core.pt"
        temporary = cache._temporary(path)
        core.save_los_core(temporary)
        os.replace(temporary, path)

    core_fit_stage = cache.ready("core_fit")
    if core_fit_stage is not None:
        result_stage = cache.ready("result", core_fit_stage["core_generation"])
        if result_stage is not None:
            print(f"[Time2Graph+ LOS] fold {fold}: resuming completed result", flush=True)
            return (
                result_stage["validation_metrics"], result_stage["test_metrics"],
                np.asarray(test_ids), cache.load_array("test_probabilities.npy"),
                result_stage["selected_tree_count"],
            )
        print(f"[Time2Graph+ LOS] fold {fold}: resuming core", flush=True)
        core = make_core(core_fit_stage["data_size"])
        core.load_los_core(cache.root / "los_core.pt", device)
        core_generation = core_fit_stage["core_generation"]
    else:
        train_ts, train_demo = matrix_arrays(
            store, train_ids, label_by_stay, args.model, config
        )
        core = make_core(train_ts.shape[-1])
        print(f"[Time2Graph+ LOS] fold {fold}: fitting core", flush=True)
        core.fit_los(train_ts, train_y)
        save_core(core)
        core_generation = uuid.uuid4().hex
        cache.finish("core_fit", data_size=int(train_ts.shape[-1]),
                     core_generation=core_generation)
        print(f"[Time2Graph+ LOS] fold {fold}: core checkpoint saved", flush=True)

    if cache.ready("core", core_generation) is None:
        if core_fit_stage is not None:
            train_ts, train_demo = matrix_arrays(
                store, train_ids, label_by_stay, args.model, config
            )
        train_x = np.concatenate([
            core.los_embeddings(train_ts, fit_feature_scaler=True), train_demo
        ], axis=1)
        save_core(core)
        cache.save_array("train_features.npy", train_x)
        cache.save_array("train_labels.npy", train_y)
        cache.finish("core", core_generation=core_generation)
        print(f"[Time2Graph+ LOS] fold {fold}: training features cached", flush=True)

    head_stage = cache.ready("head", core_generation)
    if head_stage is None:
        train_x = cache.load_array("train_features.npy")
        if cache.ready("validation", core_generation) is None:
            val_ts, val_demo = matrix_arrays(
                store, val_ids, label_by_stay, args.model, config
            )
            val_x = np.concatenate([core.los_embeddings(val_ts), val_demo], axis=1)
            cache.save_array("validation_features.npy", val_x)
            cache.save_array("validation_labels.npy", val_y)
            cache.finish("validation", core_generation=core_generation)
            print(f"[Time2Graph+ LOS] fold {fold}: validation features cached", flush=True)
        else:
            val_x = cache.load_array("validation_features.npy")
        model = build_xgb(config, args.seed + fold)
        model.fit(train_x, train_y, sample_weight=balanced_weights(train_y),
                  eval_set=[(val_x, val_y)], verbose=False)
        selected = int(model.best_iteration) + 1
        head_path = cache.root / "xgboost_head.json"
        temporary_head = cache.root / f"xgboost_head.tmp-{os.getpid()}.json"
        model.save_model(temporary_head)
        os.replace(temporary_head, head_path)
        cache.finish("head", core_generation=core_generation,
                     selected_tree_count=selected)
        print(f"[Time2Graph+ LOS] fold {fold}: XGBoost head cached", flush=True)
    else:
        model = build_xgb(config, args.seed + fold)
        model.load_model(cache.root / "xgboost_head.json")
        selected = int(head_stage["selected_tree_count"])
        val_x = cache.load_array("validation_features.npy")

    if list(model.classes_) != list(range(NUM_LOS_CLASSES)):
        raise AssertionError("Estimator class order differs from LOS class order")
    val_metrics = compute_los_metrics(
        val_y, model.predict_proba(val_x),
        targets.loc[list(val_ids), "remaining_los_days"].to_numpy(dtype=float),
        representatives,
    )
    if cache.ready("test", core_generation) is None:
        test_ts, test_demo = matrix_arrays(
            store, test_ids, label_by_stay, args.model, config
        )
        test_x = np.concatenate([core.los_embeddings(test_ts), test_demo], axis=1)
        cache.save_array("test_features.npy", test_x)
        cache.save_array("test_labels.npy", test_y)
        cache.finish("test", core_generation=core_generation)
        print(f"[Time2Graph+ LOS] fold {fold}: test features cached", flush=True)
    else:
        test_x = cache.load_array("test_features.npy")
    test_probs = model.predict_proba(test_x)
    test_metrics = compute_los_metrics(
        test_y, test_probs,
        targets.loc[list(test_ids), "remaining_los_days"].to_numpy(dtype=float),
        representatives,
    )
    core.xgb = model
    core.save_model(str(checkpoint_dir / "time2graph_core"))
    model.save_model(checkpoint_dir / "xgboost_head.json")
    cache.save_array("test_probabilities.npy", test_probs)
    cache.finish("result", core_generation=core_generation,
                 validation_metrics=val_metrics, test_metrics=test_metrics,
                 selected_tree_count=selected)
    return val_metrics, test_metrics, np.asarray(test_ids), test_probs, selected


def fit_classical(args, config, store, train_ids, val_ids, test_ids,
                  label_by_stay, targets, representatives, fold, checkpoint_dir,
                  device, label_path=None):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    train_y = np.asarray([label_by_stay[int(i)] for i in train_ids], dtype=int)
    val_y = np.asarray([label_by_stay[int(i)] for i in val_ids], dtype=int)
    if set(train_y) != set(range(NUM_LOS_CLASSES)):
        raise ValueError(f"Fold {fold} training split lacks at least one LOS class")
    if args.model == "time2graphplus":
        return fit_los_time2graphplus(
            args, config, store, train_ids, val_ids, test_ids,
            label_by_stay, targets, representatives, fold, checkpoint_dir,
            device, label_path,
        )
    else:
        train_x = matrix_arrays(store, train_ids, label_by_stay, args.model, config)
        val_x = matrix_arrays(store, val_ids, label_by_stay, args.model, config)
        if args.model == "lr":
            model = LogisticRegression(
                solver="lbfgs", max_iter=int(config["max_iter"]),
                C=float(config["C"]), class_weight="balanced",
                random_state=args.seed + fold,
            )
            model.fit(train_x, train_y)
        elif args.model == "svm":
            model = SVC(
                kernel="rbf", probability=True, C=float(config["C"]),
                class_weight="balanced", cache_size=1000,
                random_state=args.seed + fold,
            )
            model.fit(train_x, train_y)
        else:
            model = build_xgb(config, args.seed + fold)
            model.fit(train_x, train_y, sample_weight=balanced_weights(train_y),
                      eval_set=[(val_x, val_y)], verbose=False)
    val_probs = model.predict_proba(val_x)
    if list(model.classes_) != list(range(NUM_LOS_CLASSES)):
        raise AssertionError("Estimator class order differs from LOS class order")
    val_metrics = compute_los_metrics(
        val_y, val_probs,
        targets.loc[val_ids, "remaining_los_days"].to_numpy(dtype=float),
        representatives,
    )
    test_x = matrix_arrays(store, test_ids, label_by_stay, args.model, config)
    if args.model == "xgboost":
        model.save_model(checkpoint_dir / "selected_model.json")
    else:
        joblib.dump(model, checkpoint_dir / "selected_model.joblib")
    test_probs = model.predict_proba(test_x)
    test_metrics = compute_los_metrics(
        np.asarray([label_by_stay[int(i)] for i in test_ids], dtype=int),
        test_probs,
        targets.loc[test_ids, "remaining_los_days"].to_numpy(dtype=float),
        representatives,
    )
    selected_tree = int(model.best_iteration) + 1 if args.model == "xgboost" else None
    return val_metrics, test_metrics, np.asarray(test_ids), test_probs, selected_tree


def fit_neural(args, config, store, train_ids, val_ids, test_loader,
               label_by_stay, targets, representatives, fold, checkpoint_dir,
               device):
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    train_set = make_dataset(
        store, train_ids, label_by_stay, args.model, config, seed=None
    )
    val_set = make_dataset(
        store, val_ids, label_by_stay, args.model, config, seed=args.seed
    )
    batch_size = args.batch_size or int(config["batch_size"])
    train_loader = make_loader(train_set, args.model, batch_size, args.num_workers, True)
    val_loader = make_loader(val_set, args.model, args.eval_batch_size, args.num_workers, False)
    model = build_model(args.model, config, store, device)
    optim_class = torch.optim.AdamW if args.model.startswith("mstdgnn") else torch.optim.Adam
    optimizer = optim_class(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = args.epochs or int(config["epochs"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if args.model in {"mtgnn", "mstdgnn"} else None)
    best_metrics = None
    best_epoch = None
    checkpoint = checkpoint_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            if getattr(args, "max_train_steps", None) is not None and step > args.max_train_steps:
                break
            label = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.model == "mstdgnn_multiscale":
                branch_outputs = forward_logits(
                    args.model, model, batch, device, branch_logits=True
                )
                loss = sum(los_loss(output, label, args.ordinal_loss_weight)
                           for output in branch_outputs)
            else:
                output = forward_logits(args.model, model, batch, device)
                loss = los_loss(output, label, args.ordinal_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{args.model} produced non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.3 if args.model in SEQUENCE_MODELS else 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(label)
            total_count += len(label)
        if scheduler is not None:
            scheduler.step()
        val_metrics, _, _ = evaluate(
            args.model, model, val_loader, device, targets, representatives
        )
        if better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(), "best_epoch": epoch,
                 "validation_metrics": best_metrics, "config": config},
                checkpoint,
            )
        LOG.info(
            "model=%s fold=%d epoch=%d train_loss=%.4f val_kappa=%.4f "
            "val_rps=%.4f best_epoch=%d",
            args.model, fold, epoch, total_loss / total_count,
            val_metrics["linear_kappa"], val_metrics["rps"], best_epoch,
        )
    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    test_metrics, test_ids, test_probs = evaluate(
        args.model, model, test_loader, device, targets, representatives
    )
    return best_metrics, test_metrics, test_ids, test_probs, best_epoch


def write_results(output_dir, args, fold_rows, prediction_frames):
    prefix = f"{args.model}_{args.landmark_hours}h"
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / f"{prefix}_fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / f"{prefix}_test_predictions.csv", index=False
    )
    metrics = ["linear_kappa", "rps", "mae_days", "macro_f1",
               "accuracy", "within_1_bin_accuracy"]
    pd.DataFrame([
        {"dataset": args.dataset, "model": args.model, "metric": metric,
         "mean": folds[f"test_{metric}"].mean(),
         "std": folds[f"test_{metric}"].std(ddof=0),
         "n_folds": len(folds)}
        for metric in metrics
    ]).to_csv(output_dir / f"{prefix}_summary.csv", index=False)


def run(args):
    config = load_config(args.model, args.model_config)
    if args.landmark_hours != 24:
        raise ValueError("Main-table model inputs currently support only 24h landmarks")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.n_folds < 2 or not 1 <= args.window_size < args.n_folds:
        raise ValueError("Require n_folds >= 2 and 1 <= window_size < n_folds")
    if args.max_folds is not None and args.max_folds <= 0:
        raise ValueError("--max-folds must be positive")
    if args.model in CLASSICAL_MODELS and args.epochs is not None:
        raise ValueError("--epochs applies only to neural models")
    data_dir = Path(args.data_root) / DATASET_DIRS[args.dataset]
    pickle_path = Path(args.pkl_path) if args.pkl_path else data_dir / PICKLE_NAMES[args.dataset]
    stats_path = Path(args.statistics_path) if args.statistics_path else data_dir / "var_mean_std.json"
    if args.model == "raindrop" and not args.statistics_path:
        legacy = data_dir / "var_mean_std_old.json"
        if legacy.is_file():
            stats_path = legacy
    if args.model == "raindrop" and config["statistics_file"]:
        stats_path = Path(config["statistics_file"])
    label_path = (Path(args.los_labels_path) if args.los_labels_path
                  else data_dir / f"los_{args.landmark_hours}h_labels.csv")
    for path in (pickle_path, stats_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required LOS baseline input missing: {path}")
    train_ids, val_ids, test_ids = load_split_ids(args.dataset, data_dir, pickle_path)
    all_ids = np.concatenate((train_ids, val_ids, test_ids))
    labels = load_los_label_frame(
        label_path, all_ids, dataset_name=args.dataset,
        landmark_hours=args.landmark_hours,
    )
    targets = labels.set_index("ts_id", drop=False)
    label_by_stay = targets["los_class"].to_dict()
    patient_by_stay = targets["patient_id"].to_dict()
    assert_split_integrity(train_ids, val_ids, test_ids, patient_by_stay)
    folds = get_sliding_window_splits(
        np.concatenate((train_ids, val_ids)), patient_by_stay,
        n_splits=args.n_folds, window_size=args.window_size, seed=args.seed,
    )
    if args.max_folds:
        folds = folds[:args.max_folds]
    if getattr(args, "smoke_samples_per_class", None) is not None:
        test_ids = subset_per_class(test_ids, targets, args.smoke_samples_per_class)
        folds = [
            (
                subset_per_class(train, targets, args.smoke_samples_per_class, require_all=True),
                subset_per_class(validation, targets, args.smoke_samples_per_class),
            )
            for train, validation in folds
        ]
    LOG.info("dataset=%s model=%s stays=%d folds=%d label=%s",
             args.dataset, args.model, len(all_ids), len(folds), label_path)
    if args.check_only:
        return
    if "slot_minutes" in config and (int(config["slot_minutes"]) <= 0 or
                                     1440 % int(config["slot_minutes"])):
        raise ValueError("slot_minutes must divide 1440")
    if args.model == "mstdgnn_multiscale" and (
        not config["scales"] or any(
            int(scale) <= 0 or 1440 % int(scale) for scale in config["scales"]
        )
    ):
        raise ValueError("MSTD multiscale slots must divide 1440")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if args.model == "raindrop" and device.type != "cuda":
        raise RuntimeError("The existing Raindrop_v2 implementation requires CUDA")
    if args.model == "raindrop" and device.index not in (None, 0):
        raise RuntimeError("The existing Raindrop_v2 implementation uses CUDA device 0")
    store = LOSDataStore(pickle_path, stats_path, total_hours=24)
    output_dir = Path(args.results_root) / args.dataset / "los"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows, prediction_frames = [], []
    if args.model not in CLASSICAL_MODELS:
        test_set = make_dataset(
            store, test_ids, label_by_stay, args.model, config, seed=args.seed
        )
        test_loader = make_loader(
            test_set, args.model, args.eval_batch_size, args.num_workers, False
        )
    for fold, (fold_train, fold_val) in enumerate(folds, start=1):
        assert_split_integrity(fold_train, fold_val, test_ids, patient_by_stay)
        representatives = class_representative_days(
            targets.loc[fold_train, "los_class"],
            targets.loc[fold_train, "remaining_los_days"],
        )
        checkpoint_dir = (Path(args.checkpoint_root) / args.dataset /
                          args.model / f"fold_{fold}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.model in CLASSICAL_MODELS:
            val_metrics, test_metrics, ids, probs, selected = fit_classical(
                args, config, store, fold_train, fold_val, test_ids,
                label_by_stay, targets, representatives, fold,
                checkpoint_dir, device, label_path,
            )
            selection_key = "selected_tree_count"
        else:
            val_metrics, test_metrics, ids, probs, selected = fit_neural(
                args, config, store, fold_train, fold_val, test_loader,
                label_by_stay, targets, representatives, fold,
                checkpoint_dir, device,
            )
            selection_key = "selected_epoch"
        fold_rows.append({
            "dataset": args.dataset, "model": args.model,
            "landmark_hours": args.landmark_hours, "fold": fold,
            selection_key: selected,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        })
        prediction_frames.append(prediction_frame(
            args, fold, ids, probs, targets, representatives
        ))
        write_results(output_dir, args, fold_rows, prediction_frames)
        LOG.info(
            "fold=%d selected=%s test_kappa=%.4f test_rps=%.4f "
            "test_mae_days=%.4f test_within_1_bin_accuracy=%.4f",
            fold, selected, test_metrics["linear_kappa"], test_metrics["rps"],
            test_metrics["mae_days"], test_metrics["within_1_bin_accuracy"],
        )
    LOG.info("LOS results written to %s", output_dir.resolve())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_DIRS), default="mimic3")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--pkl-path")
    parser.add_argument("--statistics-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--model-config", help="JSON overrides for model defaults")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time2graph-workers", type=int, default=1,
                        help="Time2Graph+ shapelet workers; 1 avoids copying the full training matrix into subprocesses")
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--checkpoint-root", default="checkpoints/los")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-train-steps", type=int,
                        help="Smoke only: cap optimizer updates per epoch")
    parser.add_argument("--smoke-samples-per-class", type=int,
                        help="Smoke only: use at most this many stays per LOS class in each split")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    arguments = build_parser().parse_args()
    if arguments.ordinal_loss_weight < 0:
        raise ValueError("ordinal_loss_weight cannot be negative")
    run(arguments)
