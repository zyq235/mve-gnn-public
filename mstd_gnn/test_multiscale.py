import torch
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score, \
    precision_recall_curve

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.mstdgnn.STDGNN import GNNTest

CONFIG = {
    'pkl_path': 'data/mimic-iv/mimic_iv.pkl',
    'json_path': 'data/mimic-iv/var_mean_std.json',
    'checkpoint_dir': 'checkpoints/multiscale_ensemble_mimic_iv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'scales': [
        {'duration': 60, 'seq_len': 24},
        {'duration': 120, 'seq_len': 12},
        {'duration': 240, 'seq_len': 6}
    ],

    'num_folds': 6,

    'gnn_model_type': 'dyGIN2d',
    'num_layers': 3,
    'groups': 4,
    'pool_ratio': 0.2,
    'kern_size': [3, 3, 3],
    'in_dim': 64,
    'hidden_dim': 128,
    'out_dim': 256,
    'dropout': 0.0,
    'num_classes': 2,
    'batch_size': 32
}


def get_logits(model, loader, device):
    """Evaluate the model and return logits and labels."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_x, batch_demo, batch_y in loader:
            batch_x, batch_demo = batch_x.to(device), batch_demo.to(device)
            out = model(batch_x, batch_demo)
            logits_list.append(out.cpu().numpy())
            labels_list.append(batch_y.numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


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
    print(f"Testing on {device}")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, _, _, _, test_list = pickle.load(f)
    print(f"Test Set Size: {len(test_list)}")

    fold_ensemble_results = []

    all_scale_results = {scale['duration']: [] for scale in CONFIG['scales']}

    for fold_idx in range(1, CONFIG['num_folds'] + 1):

        print(f"\n{'=' * 20} Evaluating Fold {fold_idx} {'=' * 20}")

        ensemble_logits = None
        y_true_global = None

        for scale_cfg in CONFIG['scales']:
            duration = scale_cfg['duration']
            seq_len = scale_cfg['seq_len']

            test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                                     slot_duration_mins=duration)
            test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

            model = GNNTest(
                gnn_model_type=CONFIG['gnn_model_type'],
                num_layers=CONFIG['num_layers'],
                groups=CONFIG['groups'],
                pool_ratio=CONFIG['pool_ratio'],
                kern_size=CONFIG['kern_size'],
                in_dim=CONFIG['in_dim'],
                hidden_dim=CONFIG['hidden_dim'],
                out_dim=CONFIG['out_dim'],
                seq_len=seq_len,
                num_nodes=test_set.num_features + test_set.num_demo,
                num_classes=CONFIG['num_classes'],
                dropout=CONFIG['dropout']
            ).to(device)

            ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"fold{fold_idx}_scale{duration}.pt")
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint {ckpt_path} not found. Skipping this scale.")
                continue

            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            logits, labels = get_logits(model, test_loader, device)

            if y_true_global is None:
                y_true_global = labels

            probs_scale = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
            preds_scale = np.argmax(logits, axis=1)
            metrics_scale = calculate_metrics(y_true_global, probs_scale, preds_scale)

            all_scale_results[duration].append(metrics_scale)

            print(f"  [Scale {duration} mins] AUC: {metrics_scale['auc']:.4f} | "
                  f"PRAUC: {metrics_scale['prauc']:.4f} | "
                  f"MinRP: {metrics_scale['minrp']:.4f}")

            if ensemble_logits is None:
                ensemble_logits = logits
            else:
                ensemble_logits += logits

        if ensemble_logits is not None:
            avg_logits = ensemble_logits / len(CONFIG['scales'])

            probs_ens = torch.softmax(torch.tensor(avg_logits), dim=1)[:, 1].numpy()
            preds_ens = np.argmax(avg_logits, axis=1)

            metrics_ens = calculate_metrics(y_true_global, probs_ens, preds_ens)
            fold_ensemble_results.append(metrics_ens)

            print(f"  [Ensemble Result ] AUC: {metrics_ens['auc']:.4f} | "
                  f"PRAUC: {metrics_ens['prauc']:.4f} | "
                  f"MinRP: {metrics_ens['minrp']:.4f}")

    print(f"\n{'=' * 30} Final Cross-Validation Results (6 Folds) {'=' * 30}")

    metrics_to_report = ['auc', 'prauc', 'minrp']

    for scale_cfg in CONFIG['scales']:
        duration = scale_cfg['duration']
        res_list = all_scale_results[duration]
        if len(res_list) > 0:
            print(f"\n>> Scale {duration} mins:")
            for m in metrics_to_report:
                vals = [item[m] for item in res_list]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                print(f"   {m.upper().ljust(6)}: {mean_val:.4f} ± {std_val:.4f}")

    if len(fold_ensemble_results) > 0:
        print(f"\n>> Ensemble Model:")
        for m in metrics_to_report:
            vals = [item[m] for item in fold_ensemble_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"   {m.upper().ljust(6)}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()