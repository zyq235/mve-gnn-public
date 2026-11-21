import os
import pickle
import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score
from tqdm import tqdm

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.time2graphplus.core.model_gat import Time2GraphGAT
from evaluator import get_precision_at_target_recall
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_csv,
)

CONFIG = {
    'pkl_path': 'data/mimic-iv/mimic_iv.pkl',
    'json_path': 'data/mimic-iv/var_mean_std.json',
    'save_dir': 'checkpoints/t2g_optimized_mimic4',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'duration': 120,
    'seq_len': 12,

    'batch_size': 124,
    'num_folds': 6,
    'window_size': 3,

    'K': 30,
    'num_segment': 4,
    'seg_length': 3,
    'warp': 2,
    'niter': 30,

    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 20,
}


def extract_numpy_data_separated(dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    X_ts_list, X_demo_list, y_list = [], [], []

    print("Extracting Time-Series and Demographics separately...")
    for batch_x, batch_demo, batch_y in tqdm(loader, leave=False):
        x_ts_np = batch_x.transpose(1, 2).numpy()

        X_ts_list.append(x_ts_np)
        X_demo_list.append(batch_demo.numpy())  # [B, 2]
        y_list.append(batch_y.numpy())

    X_ts = np.concatenate(X_ts_list, axis=0).astype(np.float32)
    X_demo = np.concatenate(X_demo_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    return X_ts, X_demo, y


def get_t2g_embeddings(model, X):
    """Extract Time2Graph+ embeddings for the downstream classifier."""
    X_scale = model._Time2GraphGAT__preprocess_input_data(X)
    x_feat, adj = model.__gat_features__(X_scale)
    hidden_feat = model._Time2GraphGAT__gat_hidden_feature(feat=x_feat, adj=adj, X=X)
    return hidden_feat


def calculate_metrics(y_true, y_probs, y_preds):
    try:
        auc = roc_auc_score(y_true, y_probs)
        prauc = average_precision_score(y_true, y_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
        p_val1, _ = get_precision_at_target_recall(y_true, y_probs, target_recall=0.8)
        p_val2, _ = get_precision_at_target_recall(y_true, y_probs, target_recall=0.9)
    except ValueError:
        auc, prauc, f1, p_val1, p_val2 = 0.5, 0.0, 0.0, 0.0, 0.0
    return {'auc': auc, 'prauc': prauc, 'f1': f1, 'pr08': p_val1, 'pr09': p_val2}


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    print(f"Model: Time2Graph+ (Optimized Head) | Scale: {CONFIG['duration']} mins")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, _, train_list, val_list, test_list = pickle.load(f)
    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_csv(
        "data/mimic-iv/mimic-iv-3.1/icu/icustays.csv.gz"
    )

    folds_data = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=CONFIG['num_folds'],
        window_size=CONFIG['window_size'], seed=443
    )

    print("\nPreparing Test Data...")
    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                             slot_duration_mins=CONFIG['duration'])
    X_test_ts, X_test_demo, y_test = extract_numpy_data_separated(test_set, CONFIG['batch_size'])

    test_results = {"auc": [], "prauc": [], "f1": [], "pr08": [], "pr09": []}

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'=' * 10} Fold {fold_idx + 1} / {CONFIG['num_folds']} {'=' * 10}")

        train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids,
                                  slot_duration_mins=CONFIG['duration'])
        val_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids,
                                slot_duration_mins=CONFIG['duration'])

        X_train_ts, X_train_demo, y_train = extract_numpy_data_separated(train_set, CONFIG['batch_size'])
        X_val_ts, X_val_demo, y_val = extract_numpy_data_separated(val_set, CONFIG['batch_size'])

        dynamic_data_size = X_train_ts.shape[2]

        model = Time2GraphGAT(
            K=CONFIG['K'],
            seg_length=CONFIG['seg_length'],
            num_segment=CONFIG['num_segment'],
            data_size=dynamic_data_size,
            warp=CONFIG['warp'],
            gpu_enable=torch.cuda.is_available(),
            niter=CONFIG['niter'],
            out_clf=True,
            njobs=4
        )

        print("Stage 1: Training Time2Graph+ Core (Only Time-Series)...")
        model.fit(X_train_ts, y_train, n_splits=3, reset=True)

        print("Stage 2: Extracting Graph Embeddings...")
        embed_train_t2g = get_t2g_embeddings(model, X_train_ts)
        embed_val_t2g = get_t2g_embeddings(model, X_val_ts)
        embed_test_t2g = get_t2g_embeddings(model, X_test_ts)

        print("Stage 3: Fusing Embeddings with Demographics...")
        embed_train_fused = np.concatenate([embed_train_t2g, X_train_demo], axis=1)
        embed_val_fused = np.concatenate([embed_val_t2g, X_val_demo], axis=1)
        embed_test_fused = np.concatenate([embed_test_t2g, X_test_demo], axis=1)

        print("Stage 4: Training Advanced XGBoost Head...")
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = np.sum(y_train == 0) / max(pos_count, 1)

        xgb_clf = xgb.XGBClassifier(
            max_depth=CONFIG['max_depth'],
            learning_rate=CONFIG['learning_rate'],
            n_estimators=CONFIG['n_estimators'],
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=CONFIG['early_stopping_rounds'],
            n_jobs=4,
            use_label_encoder=False,
            random_state=443
        )

        xgb_clf.fit(embed_train_fused, y_train, eval_set=[(embed_val_fused, y_val)], verbose=False)

        test_probs = xgb_clf.predict_proba(embed_test_fused)[:, 1]
        test_preds = xgb_clf.predict(embed_test_fused)

        metrics = calculate_metrics(y_test, test_probs, test_preds)
        print(
            f"Fold {fold_idx + 1} Best Test -> AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['prauc']:.4f} | F1: {metrics['f1']:.4f} | PR@R0.9: {metrics['pr09']:.4f}")

        for k in test_results.keys():
            test_results[k].append(metrics[k])

        print(f"Stage 5: Saving Models for Fold {fold_idx + 1}...")
        t2g_save_path = os.path.join(CONFIG['save_dir'], f"t2g_core_fold{fold_idx + 1}")
        model.save_model(t2g_save_path)

        xgb_save_path = os.path.join(CONFIG['save_dir'], f"xgb_head_fold{fold_idx + 1}.json")
        xgb_clf.save_model(xgb_save_path)

    print("\n" + "=" * 30)
    print("Time2Graph+ (Optimized) 6-Fold Cross Validation Final Results:")
    print(f"Mean AUC:    {np.mean(test_results['auc']):.4f} ± {np.std(test_results['auc']):.4f}")
    print(f"Mean PR-AUC: {np.mean(test_results['prauc']):.4f} ± {np.std(test_results['prauc']):.4f}")
    print(f"Mean pr09: {np.mean(test_results['pr09']):.4f} ± {np.std(test_results['pr09']):.4f}")

    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()
