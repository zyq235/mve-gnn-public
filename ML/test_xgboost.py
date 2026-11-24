import numpy as np
import os
import pickle
import xgboost as xgb
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc
from tqdm import tqdm

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset

CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std.json',
    'checkpoint_dir': 'checkpoints/xgboost_single_scale',

    'duration': 120,
    'seq_len': 12,

    'num_folds': 6,
    'batch_size': 256,
}


def extract_numpy_data(dataset, batch_size=256):
    """Convert the dataset to a two-dimensional NumPy feature matrix for XGBoost."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    X_list, y_list = [], []

    for batch_x, batch_demo, batch_y in tqdm(loader, desc="Extracting Data", leave=False):
        b_size = batch_x.shape[0]
        flat_x = batch_x.view(b_size, -1).numpy()
        demo = batch_demo.numpy()
        features = np.concatenate([flat_x, demo], axis=1)
        X_list.append(features)
        y_list.append(batch_y.numpy())

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def calculate_metrics(y_true, y_probs, y_preds):
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0

    acc = accuracy_score(y_true, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
    prauc = trapezoidal_pr_auc(y_true, y_probs)

    pr, re, _ = precision_recall_curve(y_true, y_probs)
    minrp = np.minimum(pr, re).max()

    return {'auc': auc, 'acc': acc, 'f1': f1, 'prauc': prauc, 'minrp': minrp, 'recall': recall, 'precision': precision}


def main():
    print(f"Testing XGBoost | Scale: {CONFIG['duration']} mins")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, _, _, _, test_list = pickle.load(f)

    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                             slot_duration_mins=CONFIG['duration'])
    X_test, y_test = extract_numpy_data(test_set, CONFIG['batch_size'])

    print(f"Test Data Shape: {X_test.shape}")

    fold_metrics_list = []

    for fold_idx in range(1, CONFIG['num_folds'] + 1):
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"best_model_fold{fold_idx}.json")

        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found. Skipping Fold {fold_idx}.")
            continue

        print(f"\nEvaluating Model from Fold {fold_idx}...")

        model = xgb.XGBClassifier()
        model.load_model(ckpt_path)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_probs, y_preds)
        fold_metrics_list.append(metrics)
        print(
            f"  -> AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | PR-AUC: {metrics['prauc']:.4f} | MinRP: {metrics['minrp']:.4f}")

    if not fold_metrics_list:
        print("No metrics collected.")
        return

    print(f"\n{'=' * 20} Final Results ({len(fold_metrics_list)} Folds) {'=' * 20}")
    keys = fold_metrics_list[0].keys()

    for k in keys:
        values = [m[k] for m in fold_metrics_list]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{k.upper().ljust(10)}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()