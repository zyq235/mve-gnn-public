"""Fixed-split adapters for SAPS-II, flattened ML, and Time2Graph+."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
)
from evaluator import trapezoidal_pr_auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from patient_bootstrap import metric_values


TABULAR_MODELS = {"sapsii", "lr", "svm", "xgboost", "time2graphplus"}


class _Time2GraphCache:
    """Committed stage files for resuming an interrupted Time2Graph+ run."""

    _FILES = {
        "core": ("selected_t2g_core", "selected_t2g_core.xgboost",
                 "train_features.npy", "train_labels.npy"),
        "validation": ("validation_features.npy", "validation_labels.npy"),
        "head": ("selected_xgb_head.json",),
        "test": ("test_features.npy", "test_labels.npy"),
    }

    def __init__(self, output_dir, paths, protocol, args, config, *,
                 task="mortality", stage_files=None):
        self._stage_files = self._FILES if stage_files is None else stage_files
        source_files = {}
        for name in ("pickle_path", "metadata_path", "los_labels_path"):
            source = getattr(paths, name, None)
            if source is not None:
                source = Path(source).resolve()
                stat = source.stat()
                source_files[name] = {
                    "path": str(source), "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
        identity = {
            "version": 1, "dataset": getattr(args, "dataset", None),
            "seed": int(args.seed), "device": args.device,
            "config": config, "sources": source_files,
            "splits": {
                name: list(map(int, getattr(protocol, name)))
                for name in ("train_ids", "validation_ids", "test_ids")
            },
        }
        if task != "mortality":
            identity["task"] = task
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        self.fingerprint = hashlib.sha256(encoded).hexdigest()
        self.root = Path(output_dir) / "time2graphplus_intermediate" / self.fingerprint[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != self.fingerprint:
                raise ValueError(f"Time2Graph+ cache fingerprint mismatch: {manifest}")
        else:
            self._write_json(manifest, {"fingerprint": self.fingerprint, **identity})

    @staticmethod
    def _temporary(path):
        return path.with_name(f"{path.name}.tmp-{os.getpid()}")

    @classmethod
    def _write_json(cls, path, payload):
        temporary = cls._temporary(path)
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    def save_array(self, name, value):
        path = self.root / name
        temporary = self._temporary(path)
        with temporary.open("wb") as handle:
            np.save(handle, value)
        os.replace(temporary, path)

    def load_array(self, name):
        return np.load(self.root / name, mmap_mode="r", allow_pickle=False)

    def ready(self, stage, core_generation=None):
        marker = self.root / f"{stage}.done.json"
        if not marker.is_file():
            return None
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != self.fingerprint:
            raise ValueError(f"Time2Graph+ cache stage mismatch: {marker}")
        if core_generation is not None and metadata.get("core_generation") != core_generation:
            return None
        missing = [name for name in self._stage_files[stage] if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Time2Graph+ {stage} cache: {missing}")
        return metadata

    def finish(self, stage, **metadata):
        self._write_json(
            self.root / f"{stage}.done.json",
            {"fingerprint": self.fingerprint, **metadata},
        )

    def publish(self, name, output_dir):
        source = self.root / name
        target = Path(output_dir) / name
        temporary = self._temporary(target)
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)


def _config(supplied, defaults, name):
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
    return {**defaults, **supplied}


def _matrix_dataset(paths, ids, duration):
    from dataset.mimic3_MSTD_dataset import MIMIC3Dataset

    return MIMIC3Dataset(
        str(paths.pickle_path), str(paths.metadata_path),
        tsid_list=list(ids), slot_duration_mins=int(duration),
    )


def _matrix_arrays(dataset, batch_size, *, keep_time_axis=False):
    parts, demographics, labels = [], [], []
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    for x, demo, label in loader:
        matrix = x.transpose(1, 2) if keep_time_axis else x.flatten(start_dim=1)
        parts.append(matrix.numpy())
        demographics.append(demo.numpy())
        labels.append(label.numpy())
    if not parts:
        raise ValueError("Cannot extract an empty data split")
    x = np.concatenate(parts).astype(np.float32, copy=False)
    demo = np.concatenate(demographics).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.int64, copy=False)
    if keep_time_axis:
        return x, demo, y
    return np.concatenate([x, demo], axis=1), y


def _prediction(ids, labels, probabilities):
    frame = pd.DataFrame({
        "stay_id": np.asarray(ids, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
        "probability": np.asarray(probabilities, dtype=np.float64),
    })
    if len(frame) != len(ids):
        raise RuntimeError("Model predictions are not aligned to stay IDs")
    return frame


def _auc_prauc_loss(labels, probabilities):
    """XGBoost minimizes callable metrics, so negate validation AUROC+AUPRC."""
    return -(roc_auc_score(labels, probabilities) +
             trapezoidal_pr_auc(labels, probabilities))


def _boosted_head(config, seed, imbalance):
    import xgboost as xgb

    return xgb.XGBClassifier(
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        scale_pos_weight=float(imbalance),
        objective="binary:logistic", eval_metric=_auc_prauc_loss,
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        n_jobs=int(config["n_jobs"]), random_state=seed,
    )


def fit_tabular_baseline(name, paths, protocol, args, config, output_dir):
    """Fit on original train, select on validation, then infer on original test."""
    if name == "sapsii":
        return _fit_sapsii(paths, protocol, args, config, output_dir)
    if name == "time2graphplus":
        return _fit_time2graphplus(paths, protocol, args, config, output_dir)
    return _fit_flattened(name, paths, protocol, args, config, output_dir)


def _fit_sapsii(paths, protocol, args, supplied, output_dir):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from ML.train_sapsii import DATASET_CONFIGS, load_feature_table

    config = _config(supplied, {"C": 0.1, "max_iter": 1000}, "SAPS-II")
    choices = DATASET_CONFIGS[args.dataset]
    csv_path = args.sapsii_csv
    if csv_path is None:
        csv_path = next(
            (args.data_root / path.relative_to("data") for path in choices["csv_candidates"]
             if (args.data_root / path.relative_to("data")).is_file()),
            args.data_root / choices["csv_candidates"][0].relative_to("data"),
        )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"SAPS-II component CSV is missing: {csv_path}. "
            "Generate it before running this baseline or pass --sapsii-csv."
        )
    features, _ = load_feature_table(csv_path, choices["id_candidates"])
    for split_name, ids in (("train", protocol.train_ids),
                            ("validation", protocol.validation_ids),
                            ("test", protocol.test_ids)):
        missing = set(map(int, ids)) - set(features)
        if missing:
            raise ValueError(f"SAPS-II CSV lacks {len(missing)} {split_name} stays")

    def arrays(ids):
        return np.vstack([features[int(stay)] for stay in ids]), np.asarray(
            [protocol.label_by_stay[int(stay)] for stay in ids], dtype=int
        )

    x_train, y_train = arrays(protocol.train_ids)
    x_valid, y_valid = arrays(protocol.validation_ids)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=int(config["max_iter"]),
            C=float(config["C"]), random_state=args.seed,
        ),
    )
    model.fit(x_train, y_train)
    validation = metric_values(y_valid, model.predict_proba(x_valid)[:, 1])
    joblib.dump(model, output_dir / "selected_model.joblib")
    x_test, y_test = arrays(protocol.test_ids)
    prediction = _prediction(protocol.test_ids, y_test, model.predict_proba(x_test)[:, 1])
    return prediction, {"selection_rule": "fixed SAPS-II logistic regression; no epochs",
                        "selected_epoch": None, "validation_metrics": validation,
                        "effective_config": config, "sapsii_csv": str(csv_path)}


def _fit_flattened(name, paths, protocol, args, supplied, output_dir):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    defaults = {
        "slot_duration_mins": 120, "batch_size": 256,
        "max_depth": 5, "learning_rate": 0.05, "n_estimators": 500,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "early_stopping_rounds": 20, "n_jobs": 4,
    } if name == "xgboost" else {
        "slot_duration_mins": 120, "batch_size": 256,
        "C": 0.1 if name == "lr" else 1.0, "max_iter": 1000,
    }
    config = _config(supplied, defaults, name)
    train = _matrix_dataset(paths, protocol.train_ids, config["slot_duration_mins"])
    valid = _matrix_dataset(paths, protocol.validation_ids, config["slot_duration_mins"])
    x_train, y_train = _matrix_arrays(train, config["batch_size"])
    x_valid, y_valid = _matrix_arrays(valid, config["batch_size"])
    if name == "lr":
        model = LogisticRegression(
            solver="liblinear", class_weight="balanced",
            max_iter=int(config["max_iter"]), C=float(config["C"]),
            random_state=args.seed,
        )
        model.fit(x_train, y_train)
        selected = None
    elif name == "svm":
        model = SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            C=float(config["C"]), cache_size=1000, random_state=args.seed,
        )
        model.fit(x_train, y_train)
        selected = None
    elif name == "xgboost":
        imbalance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = _boosted_head(config, args.seed, imbalance)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        selected = int(model.best_iteration) + 1
    else:
        raise ValueError(f"Unknown flattened baseline: {name}")
    validation = metric_values(y_valid, model.predict_proba(x_valid)[:, 1])
    if name == "xgboost":
        model.save_model(output_dir / "selected_model.json")
    else:
        joblib.dump(model, output_dir / "selected_model.joblib")
    # The test split is constructed only after validation selection.
    test = _matrix_dataset(paths, protocol.test_ids, config["slot_duration_mins"])
    x_test, y_test = _matrix_arrays(test, config["batch_size"])
    prediction = _prediction(protocol.test_ids, y_test, model.predict_proba(x_test)[:, 1])
    return prediction, {
        "selection_rule": "maximize validation AUROC + AUPRC across XGBoost trees"
        if selected is not None else "fixed estimator; no epochs",
        "selected_epoch": None, "selected_tree_count": selected,
        "validation_metrics": validation,
        "effective_config": config,
    }


def _fit_time2graphplus(paths, protocol, args, supplied, output_dir):
    import faulthandler

    from models.time2graphplus.core.model_gat import Time2GraphGAT
    from time2graphplus.train_t2g_kfold import get_t2g_embeddings

    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)

    config = _config(supplied, {
        "slot_duration_mins": 120, "batch_size": 124, "core_batch_size": 200,
        "K": 30, "C": 300,
        "num_segment": 4, "seg_length": 3, "warp": 2, "niter": 30,
        "max_depth": 5, "learning_rate": 0.05, "n_estimators": 500,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "early_stopping_rounds": 20, "n_jobs": 4,
        "safe_small_batches": False,
    }, "Time2Graph+")
    slots = 1440 // int(config["slot_duration_mins"])
    if int(config["num_segment"]) * int(config["seg_length"]) != slots:
        raise ValueError("Time2Graph+ segment count and length must equal the time slots")
    cache = _Time2GraphCache(output_dir, paths, protocol, args, config)
    small_batch_options = {"safe_small_batches": True} if config["safe_small_batches"] else {}
    shapelet_workers = int(getattr(args, "time2graph_workers", 1))
    if shapelet_workers < 1:
        raise ValueError("--time2graph-workers must be positive")
    print(f"[Time2Graph+] shapelet_workers={shapelet_workers} "
          f"xgboost_threads={int(config['n_jobs'])}", flush=True)

    def make_core(data_size):
        return Time2GraphGAT(
            K=int(config["K"]), C=int(config["C"]),
            seg_length=int(config["seg_length"]),
            num_segment=int(config["num_segment"]), data_size=int(data_size),
            warp=int(config["warp"]), gpu_enable=args.device != "cpu",
            niter=int(config["niter"]), batch_size=int(config["core_batch_size"]),
            out_clf=True, njobs=shapelet_workers, **small_batch_options,
        )

    core_stage = cache.ready("core")
    if core_stage is None:
        train = _matrix_dataset(paths, protocol.train_ids, config["slot_duration_mins"])
        x_train, demo_train, y_train = _matrix_arrays(
            train, config["batch_size"], keep_time_axis=True
        )
        core = make_core(x_train.shape[2])
        print("[Time2Graph+] fitting core", flush=True)
        embedding_train = core.fit(
            x_train, y_train, n_splits=3, reset=True,
            return_train_embeddings=True,
        )
        fused_train = np.concatenate([embedding_train, demo_train], axis=1)
        print(f"[Time2Graph+] core fitted; caching train features {fused_train.shape}", flush=True)
        core_base = cache.root / "selected_t2g_core"
        temporary_core = cache._temporary(core_base)
        core.save_model(str(temporary_core))
        os.replace(Path(f"{temporary_core}.xgboost"), cache.root / "selected_t2g_core.xgboost")
        os.replace(temporary_core, core_base)
        cache.save_array("train_features.npy", fused_train)
        cache.save_array("train_labels.npy", y_train)
        core_generation = uuid.uuid4().hex
        cache.finish("core", data_size=int(x_train.shape[2]),
                     core_generation=core_generation)
        del train, x_train, demo_train, embedding_train
        print(f"[Time2Graph+] core checkpoint saved: {cache.root}", flush=True)
    else:
        print(f"[Time2Graph+] resuming saved core: {cache.root}", flush=True)
        core = make_core(core_stage["data_size"])
        core.load_model(str(cache.root / "selected_t2g_core"), map_location="cpu")
        core.gat = core.gat.to("cuda:0" if core.cuda else "cpu")
        y_train = cache.load_array("train_labels.npy")
        core_generation = core_stage["core_generation"]

    head_stage = cache.ready("head", core_generation)
    if head_stage is None:
        fused_train = cache.load_array("train_features.npy")
        validation_stage = cache.ready("validation", core_generation)
        if validation_stage is None:
            valid = _matrix_dataset(paths, protocol.validation_ids, config["slot_duration_mins"])
            x_valid, demo_valid, y_valid = _matrix_arrays(
                valid, config["batch_size"], keep_time_axis=True
            )
            print("[Time2Graph+] extracting validation embeddings", flush=True)
            embedding_valid = get_t2g_embeddings(core, x_valid)
            fused_valid = np.concatenate([embedding_valid, demo_valid], axis=1)
            cache.save_array("validation_features.npy", fused_valid)
            cache.save_array("validation_labels.npy", y_valid)
            cache.finish("validation", core_generation=core_generation)
            del valid, x_valid, demo_valid, embedding_valid
            print(f"[Time2Graph+] validation features cached {fused_valid.shape}", flush=True)
        else:
            print("[Time2Graph+] resuming saved validation features", flush=True)
            fused_valid = cache.load_array("validation_features.npy")
            y_valid = cache.load_array("validation_labels.npy")
    else:
        print("[Time2Graph+] resuming saved XGBoost head", flush=True)

    imbalance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    head = _boosted_head(config, args.seed, imbalance)
    if head_stage is None:
        print("[Time2Graph+] fitting validation-selected XGBoost head", flush=True)
        head.fit(fused_train, y_train, eval_set=[(fused_valid, y_valid)], verbose=False)
        selected = int(head.best_iteration) + 1
        validation = metric_values(y_valid, head.predict_proba(fused_valid)[:, 1])
        head_path = cache.root / "selected_xgb_head.json"
        temporary_head = cache.root / f"selected_xgb_head.tmp-{os.getpid()}.json"
        head.save_model(temporary_head)
        os.replace(temporary_head, head_path)
        cache.finish("head", core_generation=core_generation,
                     selected_tree_count=selected, validation_metrics=validation)
        del fused_train, fused_valid, y_valid
        print("[Time2Graph+] XGBoost head checkpoint saved", flush=True)
    else:
        head.load_model(cache.root / "selected_xgb_head.json")
        selected = int(head_stage["selected_tree_count"])
        validation = head_stage["validation_metrics"]

    test_stage = cache.ready("test", core_generation)
    if test_stage is None:
        print("[Time2Graph+] extracting test embeddings", flush=True)
        test = _matrix_dataset(paths, protocol.test_ids, config["slot_duration_mins"])
        x_test, demo_test, y_test = _matrix_arrays(test, config["batch_size"], keep_time_axis=True)
        embedding_test = get_t2g_embeddings(core, x_test)
        fused_test = np.concatenate([embedding_test, demo_test], axis=1)
        cache.save_array("test_features.npy", fused_test)
        cache.save_array("test_labels.npy", y_test)
        cache.finish("test", core_generation=core_generation)
        print(f"[Time2Graph+] test features cached {fused_test.shape}", flush=True)
    else:
        print("[Time2Graph+] resuming saved test features", flush=True)
        fused_test = cache.load_array("test_features.npy")
        y_test = cache.load_array("test_labels.npy")

    for name in ("selected_t2g_core", "selected_t2g_core.xgboost", "selected_xgb_head.json"):
        cache.publish(name, output_dir)
    prediction = _prediction(protocol.test_ids, y_test, head.predict_proba(fused_test)[:, 1])
    return prediction, {
        "selection_rule": "Time2Graph+ core fit on train; maximize validation AUROC + AUPRC across XGBoost head trees",
        "selected_epoch": None, "selected_tree_count": selected,
        "validation_metrics": validation,
        "effective_config": config,
        "shapelet_workers": shapelet_workers,
        "intermediate_cache": str(cache.root.resolve()),
    }
