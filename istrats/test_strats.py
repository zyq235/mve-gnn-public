import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.mimic_strats_dataset import MIMICStratsDataset
from models.StraTS.modeling_strats import Strats
from evaluator import get_precision_at_target_recall

DATASET_CONFIGS = {
    'mimic3': ('mimic3', 'mimic_iii.pkl',
               ('ft_strats_kfold_mm3_extend', 'ft_strats_kfold_mm3',
                'ft_strats_kfold')),
    'mimic4': ('mimic-iv', 'mimic_iv.pkl', ('ft_strats_kfold_mm4',)),
    'eicu': ('eicu', 'eicu.pkl', ('ft_strats_kfold_eicu',)),
}
MODEL_CONFIG = {
    'model_type': 'istrats', 'D': 2, 'hid_dim': 64, 'num_layers': 2,
    'num_heads': 16, 'attention_dropout': 0.2, 'dropout': 0.2,
    'pretrain': 0, 'pos_class_weight': 1.0, 'M': 12,
    'kernel_size': 4, 'r': 24,
}


def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for values, times, varis, obs_mask, demo, labels in loader:
            values, times, varis = values.to(device), times.to(device), varis.to(device)
            obs_mask, demo, labels = obs_mask.to(device), demo.to(device), labels.to(device)

            probs = model(values, times, varis, obs_mask, demo, labels=None)
            preds = (probs > 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def calculate_metrics(y_true, y_probs, y_preds):
    auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
    prauc = trapezoidal_pr_auc(y_true, y_probs)
    pr, re, _ = precision_recall_curve(y_true, y_probs)
    minrp = np.minimum(pr, re).max()

    try:
        p_val1, _ = get_precision_at_target_recall(y_true, y_probs, target_recall=0.8)
        p_val2, _ = get_precision_at_target_recall(y_true, y_probs, target_recall=0.9)
    except ValueError:
        p_val1, p_val2 = 0.0, 0.0

    return {
        'auc': auc,
        'acc': acc,
        'f1': f1,
        'prauc': prauc,
        'minrp': minrp,
        'recall': recall,
        'precision': precision,
        'pr08': p_val1,
        'pr09': p_val2
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate six saved iStraTS mortality folds")
    parser.add_argument('--dataset', choices=DATASET_CONFIGS, required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:N')
    parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'data')
    parser.add_argument('--checkpoint-root', type=Path, default=REPO_ROOT / 'checkpoints')
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='Override the dataset-specific fold checkpoint directory')
    parser.add_argument('--statistics-path', type=Path,
                        help='Override var_mean_std.json for checkpoints trained with older metadata')
    parser.add_argument('--output-dir', type=Path,
                        help='Directory for per-fold predictions and test metrics')
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--eval-truncation', choices=('earliest', 'seeded'),
                        default='earliest',
                        help='How to select events when a stay exceeds 880 observations')
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1 or args.num_workers < 0:
        parser.error('--eval-batch-size must be positive and --num-workers nonnegative')
    return args


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_fold_state(path):
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or 'variable_emb.weight' not in state:
        raise ValueError(f'{path} is not an iStraTS fine-tuning state_dict')
    return state


def resolve_inputs(args):
    directory, pickle_name, checkpoint_names = DATASET_CONFIGS[args.dataset]
    data_dir = project_path(args.data_root) / directory
    pickle_path = data_dir / pickle_name
    if args.checkpoint_dir:
        checkpoint_dir = project_path(args.checkpoint_dir)
    else:
        candidates = [project_path(args.checkpoint_root) / name
                      for name in checkpoint_names]
        checkpoint_dir = next(
            (candidate for candidate in candidates if all(
                (candidate / f'best_model_fold{fold}.pt').is_file()
                for fold in range(1, 7)
            )), candidates[0]
        )
    old_mimic3 = args.dataset == 'mimic3' and checkpoint_dir.name in {
        'ft_strats_kfold_mm3', 'ft_strats_kfold'
    }
    statistics_path = (project_path(args.statistics_path) if args.statistics_path
                       else data_dir / ('var_mean_std_old.json' if old_mimic3
                                        else 'var_mean_std.json'))
    checkpoints = [checkpoint_dir / f'best_model_fold{fold}.pt'
                   for fold in range(1, 7)]
    missing = [path for path in (pickle_path, statistics_path, *checkpoints)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required data or fold checkpoints are missing:\n'
                                + '\n'.join(str(path) for path in missing))
    with statistics_path.open(encoding='utf-8') as source:
        statistics = json.load(source)
    if not {'Age', 'Gender'} <= set(statistics):
        raise ValueError(f'{statistics_path} needs Age and Gender entries')
    variable_count = len(statistics) - 2
    first_state = load_fold_state(checkpoints[0])
    checkpoint_count = int(first_state['variable_emb.weight'].shape[0])
    if variable_count != checkpoint_count:
        raise ValueError(
            f'{statistics_path} defines {variable_count} variables, but '
            f'{checkpoints[0]} expects {checkpoint_count}; use the metadata '
            'from training or pass --statistics-path'
        )
    return pickle_path, statistics_path, checkpoints, variable_count


def resolve_device(requested):
    device = torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                          else 'cpu' if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA was requested but is unavailable: {requested}')
    return device


def main(argv=None):
    args = parse_args(argv)
    pickle_path, statistics_path, checkpoints, variable_count = resolve_inputs(args)
    device = resolve_device(args.device)
    print(f'Dataset: {args.dataset} | Device: {device} | Variables: {variable_count}')
    print(f'Checkpoints: {checkpoints[0].parent}')

    # The saved pickle contains the unchanged held-out test IDs.
    with pickle_path.open('rb') as source:
        payload = pickle.load(source)
    if not isinstance(payload, (tuple, list)) or len(payload) != 5:
        raise ValueError(f'Expected a five-object dataset pickle: {pickle_path}')
    test_ids = [int(stay) for stay in payload[4]]
    del payload
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise ValueError('Held-out test IDs are empty or duplicated')

    test_set = MIMICStratsDataset(
        str(pickle_path), str(statistics_path), tsid_list=test_ids,
        total_hours=24, max_seq_len=880, truncation_seed=0,
    )
    if test_set.num_features != variable_count:
        raise ValueError('Dataset variable count differs from metadata and checkpoint')
    expected_labels = np.asarray([test_set.labels[stay] for stay in test_ids])
    if set(expected_labels.tolist()) != {0, 1}:
        raise ValueError('Held-out test set needs both mortality classes')
    if args.eval_truncation == 'earliest':
        # Match the deterministic evaluation rule used by the newer
        # multi-dataset iStraTS training code without changing Dataset itself.
        for stay, observations in test_set.grouped_data.items():
            if len(observations) > test_set.max_seq_len:
                test_set.grouped_data[stay] = observations.nsmallest(
                    test_set.max_seq_len, 'minute'
                )
    test_loader = DataLoader(
        test_set, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
    )
    model_args = SimpleNamespace(
        **MODEL_CONFIG, V=variable_count,
        load_ckpt_path=str(checkpoints[0]),  # Selects the fine-tuning head.
    )
    output_dir = (project_path(args.output_dir) if args.output_dir else
                  REPO_ROOT / 'results' / args.dataset / 'mortality' /
                  'istrats_test' / (datetime.now().strftime('%Y%m%d_%H%M%S')
                                    + f'_{os.getpid()}'))
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_metrics = []
    for fold_index, checkpoint_path in enumerate(checkpoints, start=1):
        state = load_fold_state(checkpoint_path)
        model = Strats(model_args)
        model.load_state_dict(state, strict=True)
        model.to(device)
        labels, probabilities, predictions = get_predictions(model, test_loader, device)
        labels = np.asarray(labels).reshape(-1)
        probabilities = np.asarray(probabilities).reshape(-1)
        predictions = np.asarray(predictions).reshape(-1)
        if (len(labels) != len(test_ids) or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError(f'Fold {fold_index} produced invalid test predictions')
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f'Fold {fold_index} test labels do not match the pickle')
        metrics = calculate_metrics(labels, probabilities, predictions)
        fold_metrics.append({'fold': fold_index, **metrics})
        with (output_dir / f'fold{fold_index}_test_predictions.csv').open(
            'w', newline='', encoding='utf-8'
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(('stay_id', 'label', 'probability'))
            writer.writerows(zip(test_ids, labels.astype(int), probabilities))
        print(f'Fold {fold_index}: AUC {metrics["auc"]:.4f} | '
              f'PR-AUC {metrics["prauc"]:.4f} | '
              f'P@R0.8 {metrics["pr08"]:.4f} | P@R0.9 {metrics["pr09"]:.4f}',
              flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    summary = {
        key: {'mean': float(np.mean([fold[key] for fold in fold_metrics])),
              'std': float(np.std([fold[key] for fold in fold_metrics]))}
        for key in ('auc', 'prauc', 'pr08', 'pr09', 'f1', 'minrp')
    }
    with (output_dir / 'fold_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_metrics[0].keys())
        writer.writeheader()
        writer.writerows(fold_metrics)
    (output_dir / 'summary.json').write_text(json.dumps({
        'dataset': args.dataset, 'test_stays': len(test_ids),
        'checkpoint_dir': str(checkpoints[0].parent),
        'statistics_path': str(statistics_path),
        'eval_truncation': args.eval_truncation,
        'fold_metrics': fold_metrics, 'summary': summary,
    }, indent=2), encoding='utf-8')
    print('\nSix-fold test results (mean ± std)')
    for key, values in summary.items():
        print(f'{key.upper():<10}: {values["mean"]:.4f} ± {values["std"]:.4f}')
    print(f'Saved: {output_dir}')


if __name__ == '__main__':
    main()
