import torch
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score, \
    precision_recall_curve

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.mtgnn.net import GTNetClassifier

CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std.json',
    'checkpoint_dir': 'checkpoints/gtnet_single_scale',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'duration': 120,
    'seq_len': 12,

    'num_folds': 6,
    'batch_size': 64,
    'num_classes': 2,

    'conv_channels': 32,
    'residual_channels': 32,
    'skip_channels': 64,
    'end_channels': 128,
    'layers': 3,
}


def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_demo, batch_y in loader:
            batch_x, batch_demo = batch_x.to(device), batch_demo.to(device)
            outputs = model(batch_x, batch_demo)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


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

    return {'auc': auc, 'acc': acc, 'f1': f1, 'prauc': prauc, 'minrp': minrp, 'recall': recall, 'precision': precision}


def main():
    device = CONFIG['device']
    print(f"Testing on {device} | Scale: {CONFIG['duration']} mins")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, _, _, _, test_list = pickle.load(f)

    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                             slot_duration_mins=CONFIG['duration'])
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    total_nodes = test_set.num_features + test_set.num_demo

    fold_metrics_list = []

    for fold_idx in range(1, CONFIG['num_folds'] + 1):
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"best_model_fold{fold_idx}.pt")

        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found. Skipping.")
            continue

        print(f"Evaluating Model from Fold {fold_idx}...")

        model = GTNetClassifier(
            num_nodes=total_nodes,
            seq_len=CONFIG['seq_len'],
            in_dim=1,
            out_dim=256,
            num_classes=2,
            dropout=0,
            device=device,
            layers=CONFIG['layers'],
            conv_channels=CONFIG['conv_channels'],
            residual_channels=CONFIG['residual_channels'],
            skip_channels=CONFIG['skip_channels'],
            end_channels=CONFIG['end_channels']
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        y_true, y_probs, y_preds = get_predictions(model, test_loader, device)

        metrics = calculate_metrics(y_true, y_probs, y_preds)
        fold_metrics_list.append(metrics)
        print(f"  -> AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | prauc: {metrics['prauc']:.4f} | minrp: {metrics['minrp']:.4f}")

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