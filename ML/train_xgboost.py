import numpy as np
import os
import pickle
import xgboost as xgb
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc
from evaluator import get_precision_at_target_recall
from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_outcomes,
)

CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std.json',
    'save_dir': 'checkpoints/xgboost_single_scale_ext',

    'duration': 120,
    'seq_len': 12,

    'batch_size': 256,

    'num_folds': 6,
    'window_size': 3,

    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 20,
}



def extract_numpy_data(dataset, batch_size=256):
    """Flatten measurement arrays and append demographics for XGBoost."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
        prauc = trapezoidal_pr_auc(y_true, y_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
        p_val1, thresh1 = get_precision_at_target_recall(y_true, y_probs, target_recall=0.8)
        p_val2, thresh2 = get_precision_at_target_recall(y_true, y_probs, target_recall=0.9)
    except ValueError:
        auc, prauc, precision, recall, f1, p_val1, p_val2 = 0.5, 0.0, 0.0, 0.0, 0.0

    return {'auc': auc, 'prauc': prauc, 'f1': f1, 'pr08': p_val1, 'pr09': p_val2}


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    print(f"Model: XGBoost | Scale: {CONFIG['duration']} mins")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, lbl_df, train_list, val_list, test_list = pickle.load(f)
    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_outcomes(lbl_df)

    folds_data = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=CONFIG['num_folds'],
        window_size=CONFIG['window_size'], seed=443
    )

    test_results = {"auc": [], "prauc": [], "f1": [], "pr08": [], "pr09": []}

    print("\nPreparing Test Data...")
    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                             slot_duration_mins=CONFIG['duration'])
    X_test, y_test = extract_numpy_data(test_set, CONFIG['batch_size'])

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'=' * 10} Fold {fold_idx + 1} / {CONFIG['num_folds']} {'=' * 10}")

        train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids,
                                  slot_duration_mins=CONFIG['duration'])
        val_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids,
                                slot_duration_mins=CONFIG['duration'])

        X_train, y_train = extract_numpy_data(train_set, CONFIG['batch_size'])
        X_val, y_val = extract_numpy_data(val_set, CONFIG['batch_size'])

        print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = xgb.XGBClassifier(
            max_depth=CONFIG['max_depth'],
            learning_rate=CONFIG['learning_rate'],
            n_estimators=CONFIG['n_estimators'],
            subsample=CONFIG['subsample'],
            colsample_bytree=CONFIG['colsample_bytree'],
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=CONFIG['early_stopping_rounds'],
            n_jobs=-1,
            use_label_encoder=False,
            random_state=443
        )

        print("Training XGBoost...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        save_path = os.path.join(CONFIG['save_dir'], f"best_model_fold{fold_idx + 1}.json")
        model.save_model(save_path)

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = model.predict(X_test)

        metrics = calculate_metrics(y_test, test_probs, test_preds)
        print(
            f"Fold {fold_idx + 1} Best Test -> AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['prauc']:.4f} | F1: {metrics['f1']:.4f} | PR@R0.8: {metrics['pr08']:.4f} | PR@R0.9: {metrics['pr09']:.4f}")

        test_results["auc"].append(metrics['auc'])
        test_results["prauc"].append(metrics['prauc'])
        test_results["f1"].append(metrics['f1'])
        test_results["pr08"].append(metrics['pr08'])
        test_results["pr09"].append(metrics['pr09'])

    print("\n" + "=" * 30)
    print("XGBoost 6-Fold Cross Validation Final Results (Test Set):")
    print(f"Mean AUC:    {np.mean(test_results['auc']):.4f} ± {np.std(test_results['auc']):.4f}")
    print(f"Mean PR-AUC: {np.mean(test_results['prauc']):.4f} ± {np.std(test_results['prauc']):.4f}")
    print(f"Mean F1:     {np.mean(test_results['f1']):.4f} ± {np.std(test_results['f1']):.4f}")
    print(f"Mean PR@R0.8: {np.mean(test_results['pr08']):.4f} ± {np.std(test_results['pr08']):.4f}")
    print(f"Mean PR@R0.9: {np.mean(test_results['pr09']):.4f} ± {np.std(test_results['pr09']):.4f}")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()
