"""Compare final MVE against baselines using saved checkpoints or test predictions.

No training or checkpoint selection occurs. The default reference averages
the type/time distilled students. All five baselines participate in one
patient-cluster paired bootstrap, with Holm correction across the 15 tests.
Each inference runs in a subprocess so a native Time2Graph+ crash is isolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
BASELINES = ("istrats", "xgboost", "grud", "mstdgnn_multiscale", "time2graphplus")
VIEWS = ("type", "diff", "time")
DATA_FILES = {
    "mimic3": ("mimic3", "mimic_iii.pkl"),
    "mimic4": ("mimic-iv", "mimic_iv.pkl"),
    "eicu": ("eicu", "eicu.pkl"),
}
FINAL_FILES = ("bootstrap_summary.json", "bootstrap_repetitions.csv",
               "model_metrics.csv", "paired_comparisons.csv", "run_metadata.json")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def assignments(values, choices, option):
    result = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or name not in choices or not directory or name in result:
            raise ValueError(f"{option} requires a distinct NAME=PATH with NAME in {choices}")
        result[name] = Path(directory).expanduser().resolve()
    return result


def file_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def baseline_metadata_path(source):
    training_metadata = source / "run_metadata.json"
    if training_metadata.is_file():
        return training_metadata
    return source / "checkpoint_bootstrap" / "run_metadata.json"


def baseline_source(args, base, model, checkpoint_names):
    override = args.model_dirs.get(model)
    default = override or base / model / f"seed_{args.seed}"
    candidates = (default,) if override else (default, default.parent, base)
    for source in candidates:
        metadata_path = baseline_metadata_path(source)
        saved_predictions = None
        if not metadata_path.is_file() or not all(
            (source / name).is_file() for name in checkpoint_names
        ):
            evaluation_dir = source / "checkpoint_bootstrap"
            metadata_path = evaluation_dir / "run_metadata.json"
            saved_predictions = evaluation_dir / "predictions.csv"
            if not metadata_path.is_file() or not saved_predictions.is_file():
                continue
        if source == base:
            # A flat dataset directory has room for only one best_validation.pt.
            metadata = read_json(metadata_path)
            if (metadata.get("dataset"), metadata.get("model_adapter")) != (args.dataset, model):
                continue
        return source, metadata_path, saved_predictions
    return default, baseline_metadata_path(default), None


def build_jobs(args):
    base = args.results_root / args.dataset
    jobs = []
    for model in args.baselines:
        names = ({"core": "selected_t2g_core", "head": "selected_xgb_head.json"}
                 if model == "time2graphplus" else
                 {"model": "selected_model.json" if model == "xgboost" else "best_validation.pt"})
        source, metadata, saved_predictions = baseline_source(args, base, model, names.values())
        jobs.append({"model": model, "adapter": model, "view": None,
                     "source_dir": str(source), "metadata_path": str(metadata),
                     "checkpoints": ({key: str(source / value) for key, value in names.items()}
                                     if saved_predictions is None else {}),
                     "source_predictions": str(saved_predictions) if saved_predictions else None})
    weight = f"{args.distillation_weight:g}".replace(".", "p")
    for view in args.mve_views:
        source = args.mve_view_dirs.get(
            view, base / f"mve_{view}_distilled_w{weight}" / f"seed_{args.seed}"
        )
        jobs.append({"model": f"mve_{view}_distilled", "adapter": "mve", "view": view,
                     "source_dir": str(source),
                     "metadata_path": str(source / "run_metadata.json"),
                     "checkpoints": {"model": str(source / "best_distilled_student.pt")}})

    directory, pickle_name = DATA_FILES[args.dataset]
    data_dir = args.data_root / directory
    data_files = [data_dir / pickle_name, data_dir / "var_mean_std.json"]
    if args.dataset == "eicu":
        data_files.append(data_dir / "patient_split.csv")
    # Vocabularies and the graph index also affect reconstructed MVE inputs.
    data_files.extend(path for folder in DATA_FILES.values() for path in (
        args.data_root / folder[0] / "var2ind.json",
    ) if path.is_file())
    index = data_dir / "processed" / "processed_files.json"
    if index.is_file():
        data_files.append(index)
    missing = []
    for job in jobs:
        source = Path(job["source_dir"])
        if source == args.output_dir or source.is_relative_to(args.output_dir):
            raise ValueError("Comparison output must not contain a source training directory")
        sources = [Path(job["metadata_path"]), *map(Path, job["checkpoints"].values()), *data_files]
        if job.get("source_predictions"):
            sources.append(Path(job["source_predictions"]))
        if job["adapter"] == "mve":
            sources.extend([args.settings_root / "pretrain_model_params.json",
                            args.settings_root / f"ft_param_{job['view']}.json"])
        absent = [str(path) for path in sources if not path.is_file()]
        missing.extend(absent)
        job.update({
            "dataset": args.dataset,
            "seed": args.seed, "device": args.device,
            "data_root": str(args.data_root), "settings_root": str(args.settings_root),
            "eval_batch_size": args.eval_batch_size, "num_workers": args.num_workers,
            "time2graph_workers": args.time2graph_workers,
            "distillation_weight": args.distillation_weight,
            "output_dir": str(args.output_dir / "predictions" / job["model"]),
        })
        if not absent:
            signature = {"inference_version": 1, "job": job,
                         "sources": [file_record(path) for path in sources]}
            job["fingerprint"] = hashlib.sha256(
                json.dumps(signature, sort_keys=True).encode("utf-8")
            ).hexdigest()
    return jobs, sorted(set(missing))


def worker_environment(requested):
    """Make legacy bare .cuda() and explicit cuda:0 use the requested GPU."""
    environment = os.environ.copy()
    if requested == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu", environment
    match = re.fullmatch(r"cuda:(\d+)", requested)
    if match:
        index = int(match.group(1))
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        tokens = None if visible is None else [part.strip() for part in visible.split(",")]
        if tokens is not None and (index >= len(tokens) or tokens[index] in ("", "-1")):
            raise ValueError(f"{requested} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        environment["CUDA_VISIBLE_DEVICES"] = str(index) if tokens is None else tokens[index]
        return "cuda:0", environment
    return requested, environment


def require_matches(payload, expected, description):
    differences = {key: (payload.get(key), value) for key, value in expected.items()
                   if payload.get(key) != value}
    if differences:
        raise ValueError(f"{description} does not match the requested experiment: {differences}")


def neural_prediction(job, metadata, paths, protocol, digest, device):
    import torch
    from main_table_adapters import (
        MSTDMultiscaleAdapter, StratsAdapter, TimeseriesBaselineAdapter,
    )
    from run_patient_bootstrap import MVEAdapter, make_loader, predict_model

    selected = torch.load(job["checkpoints"]["model"], map_location="cpu", weights_only=False)
    if not isinstance(selected, dict) or not isinstance(selected.get("model_state"), dict):
        raise ValueError("Checkpoint must contain the complete model_state")
    model_args = SimpleNamespace(
        dataset=job["dataset"], data_root=Path(job["data_root"]),
        settings_root=Path(job["settings_root"]), checkpoint_root=ROOT / "checkpoints",
        pool_mode="diff_pool" if job["view"] == "diff" else job["view"],
        pretrained_checkpoint=None, strats_pretrained_checkpoint=None,
    )
    config = dict(metadata.get("effective_config") or metadata.get("model_config") or {})
    if job["adapter"] == "mve":
        require_matches(metadata, {
            "stage": "mve_repeat5_mean_logit_distillation",
            "pool_mode": model_args.pool_mode,
            "distillation_weight": job["distillation_weight"],
        }, "MVE training metadata")
        require_matches(selected, {
            "dataset": job["dataset"], "pool_mode": model_args.pool_mode,
            "distillation_weight": job["distillation_weight"],
        }, "Distilled checkpoint")
        adapter = MVEAdapter(paths, {}, model_args, Path(job["output_dir"]), device,
                             initialize_pretrained=False)
    else:
        require_matches(selected, {
            "dataset": job["dataset"], "model": job["adapter"],
            "run_name": metadata["model"], "split_digest": digest,
        }, "Baseline checkpoint")
        if job["adapter"] == "istrats":
            adapter = StratsAdapter(paths, config, model_args, protocol,
                                    initialize_pretrained=False)
        elif job["adapter"] == "grud":
            # Full state is loaded below; a training initialization file is unnecessary.
            config["init_checkpoint"] = None
            adapter = TimeseriesBaselineAdapter("grud", paths, config, job["seed"])
        else:
            adapter = MSTDMultiscaleAdapter(paths, config)
    if int(selected.get("epoch", 0)) < 1:
        raise ValueError("Saved validation-selected epoch must be positive")
    test_set = adapter.dataset(protocol.test_ids)
    model = adapter.build_model(device, getattr(test_set, "demo_dim", 2))
    model.load_state_dict(selected["model_state"], strict=True)
    model.to(device)
    batch_size = (job["eval_batch_size"] or metadata.get("eval_batch_size")
                  or adapter.default_eval_batch_size)
    loader = make_loader(adapter, test_set, int(batch_size), job["num_workers"],
                         device, shuffle=False)
    prediction = predict_model(adapter, model, loader, device, protocol.test_ids)
    return prediction, {"selected_epoch": int(selected["epoch"]),
                        "effective_config": getattr(adapter, "effective_config", None),
                        "eval_batch_size": int(batch_size)}


def classical_prediction(job, metadata, paths, protocol, device):
    import numpy as np
    import torch
    from xgboost import XGBClassifier
    from main_table_classical import _matrix_arrays, _matrix_dataset, _prediction

    config = metadata.get("effective_config")
    if not isinstance(config, dict):
        raise ValueError("Classical training metadata must include effective_config")
    test_set = _matrix_dataset(paths, protocol.test_ids, config["slot_duration_mins"])
    if job["adapter"] == "xgboost":
        features, labels = _matrix_arrays(test_set, config["batch_size"])
        head = XGBClassifier()
        head.load_model(job["checkpoints"]["model"])
    else:
        from models.time2graphplus.core.model_gat import Time2GraphGAT
        from time2graphplus.train_t2g_kfold import get_t2g_embeddings

        state = torch.load(job["checkpoints"]["core"], map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or state.get("shapelets") is None or "gat" not in state:
            raise ValueError("Time2Graph+ core checkpoint lacks fitted shapelets/GAT")
        # save_model stores the complete fitted core dictionary except its inner
        # XGBoost. Inference uses the separately selected outer head instead.
        core = object.__new__(Time2GraphGAT)
        core.__dict__.update(state)
        core.cuda = core.gpu_enable = device.type == "cuda"
        core.gat.to(device).eval()
        core.kwargs = {**core.kwargs, "njobs": job["time2graph_workers"]}
        ts, demo, labels = _matrix_arrays(test_set, config["batch_size"], keep_time_axis=True)
        with torch.no_grad():
            features = np.concatenate([get_t2g_embeddings(core, ts), demo], axis=1)
        head = XGBClassifier()
        head.load_model(job["checkpoints"]["head"])
    if list(head.classes_) != [0, 1]:
        raise ValueError("Classifier class order must be [0, 1]")
    return _prediction(protocol.test_ids, labels, head.predict_proba(features)[:, 1]), {
        "selected_tree_count": metadata.get("selected_tree_count"), "effective_config": config,
    }


def infer_job(job):
    import faulthandler
    import torch
    from ehr_baseline_utils import resolve_device, set_seed
    from run_patient_bootstrap import _check_prediction_file, load_original_split

    faulthandler.enable(all_threads=True)
    metadata = read_json(job["metadata_path"])
    paths, protocol, digest = load_original_split(
        job["dataset"], Path(job["data_root"]),
        require_graph_cache=job["adapter"] == "mve", require_metadata=True,
    )
    require_matches(metadata, {"dataset": job["dataset"], "split_digest": digest}, "Training metadata")
    if job["adapter"] != "mve":
        require_matches(metadata, {"model_adapter": job["adapter"]}, "Baseline adapter")
    set_seed(job["seed"])
    device = resolve_device(job["worker_device"])
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        device = torch.device("cuda", torch.cuda.current_device())
    print(f"Inferring {job['model']} from {job['source_dir']} on {device}", flush=True)
    if job["adapter"] in ("xgboost", "time2graphplus"):
        frame, details = classical_prediction(job, metadata, paths, protocol, device)
    else:
        frame, details = neural_prediction(job, metadata, paths, protocol, digest, device)
    frame.insert(0, "patient_id", frame["stay_id"].map(
        lambda stay: str(protocol.patient_by_stay[int(stay)])
    ))
    frame.insert(0, "split_digest", digest)
    frame.insert(0, "model", job["model"])
    frame.insert(0, "dataset", job["dataset"])
    frame = _check_prediction_file(frame, SimpleNamespace(dataset=job["dataset"]), protocol, digest)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f"predictions.tmp-{os.getpid()}.csv"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output / "predictions.csv")
    # Commit the marker last so a crashed inference is never treated as reusable.
    write_json(output / "run_metadata.json", {
        "fingerprint": job["fingerprint"], "dataset": job["dataset"],
        "model": job["model"], "model_adapter": job["adapter"],
        "split_digest": digest, "source_metadata": job["metadata_path"],
        "source_checkpoints": job["checkpoints"], "test_stays": len(frame),
        "training": "none; infer from the saved validation-selected checkpoint", **details,
    })
    print(f"Saved {output / 'predictions.csv'}", flush=True)


def summarize(args, jobs):
    import numpy as np
    import pandas as pd
    from patient_bootstrap import METRICS, _aligned_predictions, bootstrap_predictions
    from run_patient_bootstrap import _read_prediction, load_original_split

    _, protocol, digest = load_original_split(args.dataset, args.data_root,
                                              require_graph_cache=False, require_metadata=False)
    view_frames, baselines = {}, {}
    for job in jobs:
        saved_predictions = job.get("source_predictions")
        path = Path(saved_predictions) if saved_predictions else Path(job["output_dir"]) / "predictions.csv"
        frame = _read_prediction(path, args, protocol, digest)
        if saved_predictions:
            require_matches(read_json(job["metadata_path"]), {
                "dataset": args.dataset, "split_digest": digest,
                "model_adapter": job["adapter"], "model": str(frame["model"].iloc[0]),
            }, f"Saved prediction metadata for {path}")
            frame["model"] = job["model"]
        (view_frames if job["adapter"] == "mve" else baselines)[job["model"]] = frame
    reference, probabilities = _aligned_predictions(view_frames)
    combined = reference.copy()
    combined["model"] = args.reference_name
    combined["probability"] = np.stack(list(probabilities.values())).mean(axis=0)
    combined_path = args.output_dir / "predictions" / args.reference_name / "predictions.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    frames = {args.reference_name: combined, **baselines}
    print(f"Paired patient bootstrap: {args.bootstrap_reps} draws, "
          f"MVE vs {len(baselines)} baselines", flush=True)
    summary, repetitions = bootstrap_predictions(
        frames, reference_model=args.reference_name,
        repetitions=args.bootstrap_reps, seed=args.bootstrap_seed,
    )
    write_json(args.output_dir / "bootstrap_summary.json", summary)
    repetitions.to_csv(args.output_dir / "bootstrap_repetitions.csv", index=False)
    model_rows, comparison_rows = [], []
    for model, metrics in summary["models"].items():
        for metric, result in metrics.items():
            model_rows.append({"model": model, "metric": metric, "point": result["point"],
                               "ci95_low": result["ci95"][0], "ci95_high": result["ci95"][1]})
    for baseline in args.baselines:
        comparison = summary["paired_differences"][f"{args.reference_name}_minus_{baseline}"]
        for metric in METRICS:
            result = comparison[metric]
            comparison_rows.append({
                "reference": args.reference_name, "baseline": baseline, "metric": metric,
                "reference_point": summary["models"][args.reference_name][metric]["point"],
                "baseline_point": summary["models"][baseline][metric]["point"],
                "delta": result["point"], "ci95_low": result["ci95"][0],
                "ci95_high": result["ci95"][1], "p_uncorrected": result["p_uncorrected"],
                "p_holm": result["p_holm"], "significant_holm_0p05": result["p_holm"] < 0.05,
            })
    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    write_json(args.output_dir / "run_metadata.json", {
        "dataset": args.dataset, "reference": args.reference_name, "mve_views": args.mve_views,
        "aggregation": "equal-weight mean of distilled test probabilities",
        "baselines": args.baselines, "split_digest": digest,
        "bootstrap_repetitions": args.bootstrap_reps, "bootstrap_seed": args.bootstrap_seed,
        "training_seed": args.seed, "distillation_weight": args.distillation_weight,
        "training": "none; checkpoint inference and any saved test predictions",
        "multiplicity_family": f"{len(args.baselines)} baselines x {len(METRICS)} metrics in this dataset",
        "sources": jobs,
    })
    print(f"Saved paired comparisons: {args.output_dir / 'paired_comparisons.csv'}", flush=True)
    return summary


def run(args, runner=subprocess.run):
    jobs, missing = build_jobs(args)
    if args.dry_run:
        for job in jobs:
            print(f"{job['model']}: {job.get('source_predictions') or job['checkpoints']}")
            print(f"  metadata: {job['metadata_path']}")
        for path in missing:
            print(f"MISSING: {path}")
        print(f"Comparison output: {args.output_dir}")
        return int(bool(missing))
    if missing:
        raise FileNotFoundError("Required saved experiment files are missing:\n" + "\n".join(missing))
    if (args.output_dir / "bootstrap_summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"Comparison already exists: {args.output_dir}; use --overwrite")
    worker_device, environment = worker_environment(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in FINAL_FILES:
            (args.output_dir / name).unlink(missing_ok=True)
    statuses = []
    for index, job in enumerate(jobs, start=1):
        if job.get("source_predictions"):
            print(f"[{index}/{len(jobs)}] Using saved predictions for {job['model']}: "
                  f"{job['source_predictions']}", flush=True)
            statuses.append({"model": job["model"], "status": "saved_predictions",
                             "predictions": job["source_predictions"]})
            write_json(args.output_dir / "inference_manifest.json", statuses)
            continue
        job["worker_device"] = worker_device
        output = Path(job["output_dir"])
        marker = output / "run_metadata.json"
        cached = False
        if not args.refresh_predictions and marker.is_file() and (output / "predictions.csv").is_file():
            cached = read_json(marker).get("fingerprint") == job["fingerprint"]
        log = args.output_dir / "logs" / f"{job['model']}.log"
        if cached:
            status = "reused"
            print(f"[{index}/{len(jobs)}] Reusing checkpoint inference: {job['model']}", flush=True)
        else:
            job_path = args.output_dir / "jobs" / f"{job['model']}.json"
            write_json(job_path, job)
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(jobs)}] {job['model']}; log={log}", flush=True)
            try:
                marker.unlink(missing_ok=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = runner([sys.executable, "-u", str(Path(__file__).resolve()),
                                     "--worker-job", str(job_path)], cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
                status = "completed" if result.returncode == 0 else f"failed: exit code {result.returncode}"
                if status == "completed" and (not (output / "predictions.csv").is_file() or
                        not marker.is_file() or
                        read_json(marker).get("fingerprint") != job["fingerprint"]):
                    status = "failed: predictions or inference completion marker is missing"
            except (OSError, ValueError) as error:
                status = f"failed: {error}"
        statuses.append({"model": job["model"], "status": status, "log": str(log)})
        write_json(args.output_dir / "inference_manifest.json", statuses)
    failures = [row for row in statuses if row["status"].startswith("failed")]
    if failures:
        print("Comparison not computed because requested models failed:", flush=True)
        for failure in failures:
            print(f"  {failure['model']}: {failure['status']}; {failure['log']}", flush=True)
        return 1
    summarize(args, jobs)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=tuple(DATA_FILES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "patient_bootstrap")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--settings-root", type=Path, default=ROOT / "settings")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--mve-views", nargs="+", choices=VIEWS, default=["type", "time"])
    parser.add_argument("--distillation-weight", type=float, default=0.75)
    parser.add_argument("--model-dir", action="append", default=[], metavar="MODEL=PATH",
                        help="Baseline directory containing the checkpoint or checkpoint_bootstrap/ results")
    parser.add_argument("--mve-view-dir", action="append", default=[], metavar="VIEW=PATH",
                        help="Override one distilled student's training directory")
    parser.add_argument("--eval-batch-size", type=int,
                        help="Default: use the saved training metadata or adapter default")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time2graph-workers", type=int, default=1)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--refresh-predictions", action="store_true",
                        help="Recompute inference even if a matching checkpoint inference cache exists")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace final comparison outputs; matching inference caches remain reusable")
    parser.add_argument("--dry-run", action="store_true", help="List required files without inference")
    args = parser.parse_args(argv)
    if args.worker_job is not None:
        return args
    if args.dataset is None:
        parser.error("--dataset is required")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", args.device):
        parser.error("--device must be auto, cpu, cuda or cuda:N")
    for name in ("baselines", "mve_views"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if len(args.mve_views) < 2:
        parser.error("--mve-views requires at least two views")
    args.mve_views = [view for view in VIEWS if view in args.mve_views]
    if not 0 <= args.distillation_weight <= 1:
        parser.error("--distillation-weight must be in [0, 1]")
    if args.num_workers < 0 or args.bootstrap_reps < 1 or args.time2graph_workers < 1:
        parser.error("bootstrap repetitions/workers must be positive; num-workers must be nonnegative")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    for name in ("results_root", "data_root", "settings_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    try:
        args.model_dirs = assignments(args.model_dir, args.baselines, "--model-dir")
        args.mve_view_dirs = assignments(args.mve_view_dir, args.mve_views, "--mve-view-dir")
    except ValueError as error:
        parser.error(str(error))
    args.reference_name = ("mve_combined" if args.mve_views == list(VIEWS) else
                           "mve_combined_" + "_".join(args.mve_views))
    args.output_dir = (args.output_dir or args.results_root / args.dataset /
                       "paired_mve_vs_baselines" / f"{'_'.join(args.mve_views)}_seed_{args.seed}")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_job is not None:
        infer_job(read_json(args.worker_job))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
