import torch
import numpy as np
import pickle
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc

from dataset.mimic_raindrop_dataset import MIMIC3Dataset
from models.Raindrop.models_rd import Raindrop_v2
from evaluator import get_precision_at_target_recall

CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std_old.json',
    'checkpoint_dir': 'checkpoints/raindrop_kfold_mm3',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'num_folds': 6,
    'batch_size': 64,
    'num_classes': 2,

    'd_ob': 4,
    'nhead': 2,
    'nhid': 256,
    'nlayers': 2,
    'max_len': 200
}


def raindrop_collate_fn(batch):
    xs, masks, times, demos, ys = zip(*batch)

    lengths = torch.tensor([len(t) for t in times], dtype=torch.long)

    pad_x = pad_sequence(xs, batch_first=False, padding_value=0.0)
    pad_mask = pad_sequence(masks, batch_first=False, padding_value=0.0)
    pad_times = pad_sequence(times, batch_first=False, padding_value=0.0)

    T_max = pad_x.size(0)
    max_len = CONFIG['max_len']

    if T_max < max_len:
        pad_amount = max_len - T_max
        pad_x = F.pad(pad_x, (0, 0, 0, 0, 0, pad_amount), value=0.0)
        pad_mask = F.pad(pad_mask, (0, 0, 0, 0, 0, pad_amount), value=0.0)
        pad_times = F.pad(pad_times, (0, 0, 0, pad_amount), value=0.0)

    elif T_max > max_len:
        pad_x = pad_x[:max_len]
        pad_mask = pad_mask[:max_len]
        pad_times = pad_times[:max_len]

    lengths = torch.clamp(lengths, max=max_len)

    demos = torch.stack(demos)
    ys = torch.stack(ys)

    src = torch.cat([pad_x, pad_mask], dim=-1)

    return src, pad_times, demos, ys, lengths


def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for src, times, demo, y, lengths in loader:
            src = src.to(device)
            times = times.to(device)
            demo = demo.to(device)
            lengths = lengths.to(device)

            outputs, _, _ = model(src, demo, times, lengths)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


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
    return {'auc': auc, 'acc': acc, 'f1': f1, 'prauc': prauc, 'minrp': minrp, 'recall': recall, 'precision': precision,
            'pr08': p_val1, 'pr09': p_val2}


def main():
    device = CONFIG['device']
    print(f"Testing Raindrop on {device} | Max Length: {CONFIG['max_len']}")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, _, _, _, test_list = pickle.load(f)

    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list)

    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2,
                             collate_fn=raindrop_collate_fn)

    d_inp = test_set.num_features  # 129
    d_static = test_set.num_demo  # 2

    fold_metrics_list = []

    for fold_idx in range(1, CONFIG['num_folds'] + 1):
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"raindrop_fold{fold_idx}.pt")

        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found. Skipping.")
            continue

        print(f"Evaluating Raindrop Model from Fold {fold_idx}...")

        global_structure = torch.ones(d_inp, d_inp).to(device)

        model = Raindrop_v2(
            d_inp=d_inp,
            d_model=d_inp * CONFIG['d_ob'],
            nhead=CONFIG['nhead'],
            nhid=CONFIG['nhid'],
            nlayers=CONFIG['nlayers'],
            dropout=0.0,
            max_len=CONFIG['max_len'],
            d_static=d_static,
            n_classes=CONFIG['num_classes'],
            global_structure=global_structure
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        y_true, y_probs, y_preds = get_predictions(model, test_loader, device)

        metrics = calculate_metrics(y_true, y_probs, y_preds)
        fold_metrics_list.append(metrics)
        print(
            f"  -> AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | prauc: {metrics['prauc']:.4f} | minrp: {metrics['minrp']:.4f} | pr08: {metrics['pr08']:.4f} | pr09: {metrics['pr09']:.4f}")

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