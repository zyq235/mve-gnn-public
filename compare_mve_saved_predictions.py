"""Run a paired patient bootstrap from existing MVE and baseline predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINES = ("istrats", "xgboost", "grud", "mstdgnn_multiscale")


def export_paired_comparisons(output_dir: Path) -> None:
    with (output_dir / "bootstrap_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    reference = summary["reference_model"]
    models = summary["models"]
    rows = []
    for baseline, metrics in models.items():
        if baseline == reference:
            continue
        comparison = summary["paired_differences"][f"{reference}_minus_{baseline}"]
        for metric, result in comparison.items():
            low, high = result["ci95"]
            rows.append({
                "reference": reference,
                "baseline": baseline,
                "metric": metric,
                "reference_point": models[reference][metric]["point"],
                "baseline_point": metrics[metric]["point"],
                "delta": result["point"],
                "ci95_low": low,
                "ci95_high": high,
                "p_uncorrected": result["p_uncorrected"],
                "p_holm": result["p_holm"],
                "significant_holm_0p05": result["p_holm"] < 0.05,
            })
    if not rows:
        raise ValueError("bootstrap_summary.json contains no paired comparisons")

    report = output_dir / "paired_comparisons.csv"
    temporary = output_dir / f".paired_comparisons.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(report)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved paired comparisons to {report}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=("mimic3", "mimic4", "eicu"), default="mimic3")
    parser.add_argument("--mve-model", choices=("mve_combined", "mve_combined_type_time"),
                        default="mve_combined", help="Saved final MVE prediction directory")
    parser.add_argument("--seed", type=int, default=443, help="Seed used in input directory names")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "patient_bootstrap")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Check and print selected paths")
    action.add_argument("--export-only", action="store_true",
                        help="Export paired_comparisons.csv from the existing bootstrap_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")

    dataset_dir = args.results_root.expanduser().resolve() / args.dataset
    output_name = "from_saved_csv" if args.mve_model == "mve_combined" else "from_saved_csv_type_time"
    output_dir = (args.output_dir or dataset_dir / "paired_mve_vs_baselines" / output_name).expanduser()
    if args.export_only:
        export_paired_comparisons(output_dir)
        return 0

    seed_dir = f"seed_{args.seed}"
    reference = dataset_dir / args.mve_model / seed_dir / "predictions.csv"
    missing = [] if reference.is_file() else [f"{args.mve_model}: {reference}"]
    predictions = {}
    for model in BASELINES:
        model_dir = dataset_dir / model / seed_dir
        candidates = (model_dir / "checkpoint_bootstrap" / "predictions.csv",
                      model_dir / "predictions.csv")
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is None:
            missing.append(f"{model}: {candidates[0]} or {candidates[1]}")
        else:
            predictions[model] = selected
    if missing:
        parser.error("Missing saved predictions:\n" + "\n".join(missing))

    print(f"Reference {args.mve_model}: {reference}", flush=True)
    for model, path in predictions.items():
        print(f"Baseline {model}: {path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    if args.dry_run:
        return 0

    command = [
        sys.executable, str(ROOT / "run_patient_bootstrap.py"),
        "--mode", "bootstrap", "--dataset", args.dataset,
        "--data-root", str(args.data_root.expanduser()),
        "--predictions-csv", str(reference),
        "--bootstrap-reps", str(args.bootstrap_reps),
        "--bootstrap-seed", str(args.bootstrap_seed),
        "--output-dir", str(output_dir),
    ]
    for path in predictions.values():
        command.extend(("--compare-to", str(path)))
    if args.overwrite:
        command.append("--overwrite")
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        export_paired_comparisons(output_dir)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
