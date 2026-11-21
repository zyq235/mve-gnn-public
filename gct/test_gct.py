"""Evaluate validation-selected GCT fold checkpoints on the fixed test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.ehr_cached_graph_dataset import (
    SUPPORTED_DATASETS,
    load_protocol_splits,
    resolve_ehr_dataset_paths,
)
from dataset.gct_dataset import GCTICUDataset, gct_collate_fn
from ehr_baseline_utils import (
    resolve_device,
    save_json,
    summarize_folds,
    write_fold_results_csv,
)
from gct.common import build_model, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="mimic3")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Defaults to results/<dataset>/gct.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=6)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    paths = resolve_ehr_dataset_paths(args.dataset, args.data_root)
    checkpoint_dir = args.checkpoint_dir or REPO_ROOT / "checkpoints" / f"gct_{args.dataset}"
    checkpoints = [
        checkpoint_dir / f"fold_{fold}_best.pt"
        for fold in range(1, args.n_folds + 1)
    ]
    existing = [path for path in checkpoints if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No GCT fold checkpoints found in {checkpoint_dir}")
    first = torch.load(existing[0], map_location="cpu", weights_only=False)
    if first.get("dataset") != args.dataset:
        raise ValueError("Checkpoint dataset does not match --dataset")
    protocol_config = first.get("protocol", {})
    protocol = load_protocol_splits(
        paths,
        n_splits=int(protocol_config.get("n_folds", args.n_folds)),
        window_size=int(protocol_config.get("window_size", 3)),
        seed=int(protocol_config.get("seed", 443)),
    )
    test_dataset = GCTICUDataset(
        paths.processed_dir,
        protocol.test_ids,
        index_path=paths.index_path,
        **first["dataset_kwargs"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=gct_collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    results = []
    for path in existing:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("dataset") != args.dataset:
            raise ValueError(f"Dataset mismatch in {path}")
        model = build_model(checkpoint["model_kwargs"], device)
        model.load_state_dict(checkpoint["model_state"])
        metrics = evaluate(
            model,
            test_loader,
            device,
            regularization_coefficient=float(
                checkpoint.get("regularization_coefficient", 0.1)
            ),
        )
        results.append({"fold": int(checkpoint["fold"]), **metrics})
        print(
            f"fold={checkpoint['fold']} auc={metrics['auc']:.4f} "
            f"prauc={metrics['prauc']:.4f}"
        )
    summary = summarize_folds(
        [{key: value for key, value in result.items() if key != "fold"} for result in results]
    )
    save_json(
        checkpoint_dir / "test_results.json",
        {"dataset": args.dataset, "folds": results, "summary": summary},
    )
    seed = int(protocol_config.get("seed", 443))
    num_folds = int(protocol_config.get("n_folds", args.n_folds))
    window_size = int(protocol_config.get("window_size", 3))
    train_sizes = {
        fold_index + 1: len(fold_train_ids)
        for fold_index, (fold_train_ids, _) in enumerate(protocol.folds)
    }
    output_path = getattr(args, "output_path", None) or (
        REPO_ROOT / "results" / args.dataset / "gct.csv"
    )
    try:
        feature_source = paths.index_path.relative_to(REPO_ROOT)
        processed_pkl = paths.pickle_path.relative_to(REPO_ROOT)
    except ValueError:
        feature_source = paths.index_path
        processed_pkl = paths.pickle_path
    write_fold_results_csv(
        results,
        output_path,
        dataset=args.dataset,
        model="gct",
        seed=seed,
        num_folds=num_folds,
        window_size=window_size,
        regularization_c=float(first.get("regularization_coefficient", 0.1)),
        train_sizes=train_sizes,
        test_size=len(protocol.test_ids),
        feature_csv_path=feature_source,
        processed_pkl_path=processed_pkl,
    )
    print("Test summary:", summary)


if __name__ == "__main__":
    main(parse_args())
