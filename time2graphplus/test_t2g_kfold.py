import torch
import numpy as np
import pickle
import os
import xgboost as xgb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score, \
    precision_recall_curve
from tqdm import tqdm

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.time2graphplus.core.model_gat import Time2GraphGAT
from evaluator import get_precision_at_target_recall

CONFIG = {
    'pkl_path': 'data/mimic-iv/mimic_iv.pkl',
    'json_path': 'data/mimic-iv/var_mean_std.json',
    'checkpoint_dir': 'checkpoints/t2g_optimized_mimic3',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'duration': 120,
    'seq_len': 12,
    'batch_size': 128,
    'num_folds': 6,

    'K': 30,
    'num_segment': 4,
    'seg_length': 3,
    'data_size': 129,
    'warp': 2,
}


def extract_numpy_data_separated(dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    X_ts_list, X_demo_list, y_list = [], [], []

    print("Extracting Test Data into Numpy Arrays...")
    for batch_x, batch_demo, batch_y in tqdm(loader, leave=False):
        x_ts_np = batch_x.transpose(1, 2).numpy()
        X_ts_list.append(x_ts_np)
        X_demo_list.append(batch_demo.numpy())
        y_list.append(batch_y.numpy())

    X_ts = np.concatenate(X_ts_list, axis=0).astype(np.float32)
    X_demo = np.concatenate(X_demo_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    return X_ts, X_demo, y


def get_t2g_embeddings(model, X):
    X_scale = model._Time2GraphGAT__preprocess_input_data(X)
    x_feat, adj = model.__gat_features__(X_scale)
    hidden_feat = model._Time2GraphGAT__gat_hidden_feature(feat=x_feat, adj=adj, X=X)
    return hidden_feat


def calculate_metrics(y_true, y_probs, y_preds):
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
    acc = accuracy_score(y_true, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
    prauc = average_precision_score(y_true, y_probs)
    pr, re, _ = precision_recall_curve(y_true, y_probs)
    minrp = np.minimum(pr, re).max()
    p_val1, thresh1 = get_precision_at_target_recall(y_true, y_probs, target_recall=0.8)
    p_val2, thresh2 = get_precision_at_target_recall(y_true, y_probs, target_recall=0.9)

    return {'auc': auc, 'acc': acc, 'f1': f1, 'prauc': prauc, 'minrp': minrp, 'recall': recall, 'precision': precision,
            'pr08': p_val1, 'pr09': p_val2}


def main():
    device = CONFIG['device']
    print(f"Testing Time2Graph+ on {device} | Scale: {CONFIG['duration']} mins")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, _, _, _, test_list = pickle.load(f)

    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                             slot_duration_mins=CONFIG['duration'])
    X_test_ts, X_test_demo, y_test = extract_numpy_data_separated(test_set, CONFIG['batch_size'])

    print(f"Test TS Shape: {X_test_ts.shape}, Demo Shape: {X_test_demo.shape}")

    fold_metrics_list = []
    all_folds_probs = []

    for fold_idx in range(1, CONFIG['num_folds'] + 1):
        t2g_ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"t2g_core_fold{fold_idx}")
        xgb_ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"xgb_head_fold{fold_idx}.json")

        if not os.path.exists(t2g_ckpt_path + ".pt") or not os.path.exists(xgb_ckpt_path):
            print(f"Warning: Checkpoints for Fold {fold_idx} not found. Skipping.")
            continue

        print(f"\nEvaluating Model from Fold {fold_idx}...")

        model = Time2GraphGAT(
            K=CONFIG['K'], seg_length=CONFIG['seg_length'], num_segment=CONFIG['num_segment'],
            data_size=CONFIG['data_size'], warp=CONFIG['warp'], gpu_enable=torch.cuda.is_available(),
            njobs=1
        )
        model.load_model(t2g_ckpt_path, map_location=device)

        xgb_clf = xgb.XGBClassifier()
        xgb_clf.load_model(xgb_ckpt_path)

        embed_test_t2g = get_t2g_embeddings(model, X_test_ts)
        embed_test_fused = np.concatenate([embed_test_t2g, X_test_demo], axis=1)

        y_probs = xgb_clf.predict_proba(embed_test_fused)[:, 1]
        y_preds = xgb_clf.predict(embed_test_fused)

        all_folds_probs.append(y_probs)

        metrics = calculate_metrics(y_test, y_probs, y_preds)
        fold_metrics_list.append(metrics)
        print(f"  -> AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | PR-AUC: {metrics['prauc']:.4f}")

    if not fold_metrics_list:
        print("No metrics collected.")
        return

    print(f"\n{'=' * 20} Single Model Average ({len(fold_metrics_list)} Folds) {'=' * 20}")
    keys = fold_metrics_list[0].keys()
    for k in keys:
        values = [m[k] for m in fold_metrics_list]
        print(f"{k.upper().ljust(10)}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    if len(all_folds_probs) > 1:
        print(f"\n{'=' * 20} Soft-Voting Ensemble Results {'=' * 20}")
        mean_probs = np.mean(all_folds_probs, axis=0)
        mean_preds = (mean_probs >= 0.5).astype(int)

        ensemble_metrics = calculate_metrics(y_test, mean_probs, mean_preds)
        for k, v in ensemble_metrics.items():
            print(f"{k.upper().ljust(10)}: {v:.4f}")
    # ====================================================================


if __name__ == "__main__":
    main()