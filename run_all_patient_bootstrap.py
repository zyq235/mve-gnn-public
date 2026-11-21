"""Run every patient-bootstrap model serially, continuing after failures.

Each trainable model uses run_patient_bootstrap.py to train, evaluate on the
fixed test split, and compute its patient-level bootstrap results.  The
combined MVE model only reads previously distilled view predictions.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENTRY_POINT = ROOT / "run_patient_bootstrap.py"

# Edit this list to choose models and their execution order.
MODEL_NAMES = [
    "gct",
    "gram",
    "gru",
    "grud",
    "sand",
    "tcn",
    "istrats",
    "mtgnn",
    "raindrop",
    "mstdgnn",
    "mstdgnn_multiscale",
    "sapsii",
    "lr",
    "svm",
    "xgboost",
    "time2graphplus",
    "mve_combined",  # Requires prior type/diff/time MVE distillation runs.
]

# These estimators have no epoch parameter. MVE combined only reads predictions.
NO_EPOCH_MODELS = {"sapsii", "lr", "svm", "xgboost", "time2graphplus", "mve_combined"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mimic3", "mimic4", "eicu"))
    parser.add_argument("--device", default="auto", help="Passed to every model, e.g. cuda:0 or cpu")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs for each trainable neural-network model")
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--time2graph-workers", type=int, default=1,
                        help="Shapelet workers for Time2Graph+; default one limits GPU memory")
    parser.add_argument("--debug-numerics", action="store_true",
                        help="Pass per-batch numerical checks to each model")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow each model to replace its existing result")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.bootstrap_reps < 1 or args.time2graph_workers < 1:
        parser.error("--epochs, --bootstrap-reps and --time2graph-workers must be positive")
    return args


def _time2graph_cuda_index(model, requested_device):
    if model != "time2graphplus":
        return None
    match = re.fullmatch(r"cuda:(\d+)", requested_device)
    if match is None:
        return None
    index = int(match.group(1))
    return index if index > 0 else None


def _time2graph_child_env(model, requested_device, source_env=None):
    """Expose one requested GPU so legacy bare .cuda() resolves to cuda:0."""
    index = _time2graph_cuda_index(model, requested_device)
    if index is None:
        return None
    env = dict(os.environ if source_env is None else source_env)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(index)
    else:
        devices = [value.strip() for value in visible.split(",")]
        if index >= len(devices) or not devices[index]:
            raise ValueError(
                f"{requested_device} is not available within CUDA_VISIBLE_DEVICES={visible!r}"
            )
        selected = devices[index]
    env["CUDA_VISIBLE_DEVICES"] = selected
    return env


def command_for_model(model, args):
    child_device = "cuda:0" if _time2graph_cuda_index(model, args.device) is not None else args.device
    command = [
        sys.executable, str(ENTRY_POINT),
        "--mode", "train",
        "--model", model,
        "--dataset", args.dataset,
        "--device", child_device,
        "--bootstrap-reps", str(args.bootstrap_reps),
        "--seed", str(args.seed),
        "--bootstrap-seed", str(args.bootstrap_seed),
    ]
    if model not in NO_EPOCH_MODELS:
        command.extend(("--epochs", str(args.epochs)))
    if model == "time2graphplus":
        command.extend(("--time2graph-workers", str(args.time2graph_workers)))
    if args.overwrite:
        command.append("--overwrite")
    if args.debug_numerics:
        command.append("--debug-numerics")
    return command


def run_models(args, models=MODEL_NAMES, runner=subprocess.run):
    succeeded = []
    failed = []
    for index, model in enumerate(models, start=1):
        command = command_for_model(model, args)
        print(f"\n[{index}/{len(models)}] Starting {model}: {' '.join(command)}", flush=True)
        started = time.monotonic()
        try:
            child_env = _time2graph_child_env(model, args.device)
            if child_env is None:
                result = runner(command, cwd=ROOT, check=False)
            else:
                print(
                    f"  Time2Graph+ {args.device} -> child cuda:0 "
                    f"(CUDA_VISIBLE_DEVICES={child_env['CUDA_VISIBLE_DEVICES']})",
                    flush=True,
                )
                result = runner(command, cwd=ROOT, check=False, env=child_env)
            error = None if result.returncode == 0 else f"exit code {result.returncode}"
        except (OSError, ValueError) as exc:
            error = str(exc)
        elapsed = time.monotonic() - started
        if error is None:
            succeeded.append(model)
            print(f"[{model}] completed in {elapsed:.1f}s", flush=True)
        else:
            failed.append((model, error))
            print(f"[{model}] FAILED ({error}) after {elapsed:.1f}s; continuing", flush=True)

    print("\nFinished all requested models.", flush=True)
    print(f"Succeeded ({len(succeeded)}): {', '.join(succeeded) or 'none'}", flush=True)
    print(f"Failed ({len(failed)}):", flush=True)
    for model, error in failed:
        print(f"  {model}: {error}", flush=True)
    return 1 if failed else 0


def main(argv=None):
    args = parse_args(argv)
    return run_models(args)


if __name__ == "__main__":
    raise SystemExit(main())
