import numpy as np
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
    'duration': 120,
    'seq_len': 12,  # 24h / 2h = 12
    'num_folds': 6,
    'window_size': 3,
    'batch_size': 256,
    'num_workers': 4
}


def extract_features_and_labels(dataset):
    """Flatten the time-series features and append the two demographic features."""
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    X_list = []
    y_list = []

    print("Extracting features from dataset...")
    for batch_x, batch_demo, batch_y in tqdm(loader):
        # batch_x: [B, 129, 12] -> Flatten -> [B, 1548]
        batch_size = batch_x.shape[0]
        flat_x = batch_x.view(batch_size, -1).numpy()

        # batch_demo: [B, 2]
        demo = batch_demo.numpy()

        # Concatenate: [B, 1550]
        features = np.concatenate([flat_x, demo], axis=1)

        X_list.append(features)
        y_list.append(batch_y.numpy())

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def get_model(model_name):
    """Initialize the estimator with balanced class weights when supported."""
    if model_name == 'lr':
        # Logistic Regression
        return LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            C=0.1
        )
    elif model_name == 'svm':
        return SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            cache_size=1000
        )
    elif model_name == 'dt':
        # Decision Tree
        return DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=10,
            min_samples_leaf=5
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def calculate_metrics(y_true, y_probs, y_preds):
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
    acc = accuracy_score(y_true, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
    prauc = trapezoidal_pr_auc(y_true, y_probs)
    pr, re, _ = precision_recall_curve(y_true, y_probs)
    minrp = np.minimum(pr, re).max()
    p_val1, thresh1 = get_precision_at_target_recall(y_true, y_probs, target_recall=0.8)
    p_val2, thresh2 = get_precision_at_target_recall(y_true, y_probs, target_recall=0.9)
    return {'auc': auc, 'acc': acc, 'f1': f1, 'prauc': prauc, 'minrp': minrp, 'recall': recall, 'precision': precision,"pr08": p_val1, "pr09": p_val2}


def run_experiment(model_name):
    print(f"\n{'#' * 30}\nRunning Experiment: {model_name.upper()}\n{'#' * 30}")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, lbl_df, train_list, val_list, test_list = pickle.load(f)

    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_outcomes(lbl_df)

    print("Preparing Test Data...")
    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                             slot_duration_mins=CONFIG['duration'])
    X_test, y_test = extract_features_and_labels(test_set)


    folds = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=CONFIG['num_folds'],
        window_size=CONFIG['window_size'], seed=443
    )
    fold_results = []

    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        assert_split_integrity(train_ids, val_ids, test_list, id_to_patient)

        print(f"\n--- Fold {fold_idx + 1} / {CONFIG['num_folds']} ---")

        train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=train_ids,
                                  slot_duration_mins=CONFIG['duration'])
        X_train, y_train = extract_features_and_labels(train_set)

        print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

        clf = get_model(model_name)

        print(f"Training {model_name}...")
        clf.fit(X_train, y_train)

        print("Predicting...")
        if hasattr(clf, "predict_proba"):
            y_probs = clf.predict_proba(X_test)[:, 1]
        else:
            y_probs = clf.predict(X_test)

        y_preds = clf.predict(X_test)

        metrics = calculate_metrics(y_test, y_probs, y_preds)
        fold_results.append(metrics)
        print(f"  -> AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | prauc: {metrics['prauc']:.4f} | minrp: {metrics['minrp']:.4f} | pr08: {metrics['pr08']:.4f} | pr09: {metrics['pr09']:.4f}")

    print(f"\n{'=' * 20} Final {model_name.upper()} Results (6 Folds) {'=' * 20}")
    keys = fold_results[0].keys()

    for k in keys:
        values = [res[k] for res in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{k.upper().ljust(10)}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['lr', 'svm', 'dt', 'all'],
                        help='Choose model to run')
    args = parser.parse_args()

    models_to_run = ['lr'] if args.model == 'all' else [args.model]

    for m in models_to_run:
        run_experiment(m)
