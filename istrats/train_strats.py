import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc

from dataset.mimic_strats_dataset import MIMICStratsDataset
from models.StraTS.modeling_strats import Strats
from evaluator import get_precision_at_target_recall
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_outcomes,
)


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std.json',
    'save_dir': 'checkpoints/ft_strats_kfold_mm3_extend',
    'load_ckpt_path': 'checkpoints/finetune_strats_mm3_extend/checkpoint_best.bin',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'total_hours': 24,
    'max_seq_len': 880,
    'train_batch_size': 16,
    'eval_batch_size': 32,
    'epochs': 40,
    'lr': 5e-04,
    'weight_decay': 1e-4,
    'num_folds': 6,
    'window_size': 3,

    'model_type': 'istrats',
    'V': 142,
    'D': 2,
    'hid_dim': 64,
    'num_layers': 2,
    'num_heads': 16,
    'attention_dropout': 0.2,
    'dropout': 0.2,
    'pretrain': 0,
    'pos_class_weight': 1.0,

    'M': 12,
    'kernel_size': 4,
    'r': 24,
    'seed': 2023
}


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

    try:
        p_val1, _ = get_precision_at_target_recall(y_true, y_probs, target_recall=0.8)
        p_val2, _ = get_precision_at_target_recall(y_true, y_probs, target_recall=0.9)
    except:
        p_val1, p_val2 = 0.0, 0.0

    return {'auc': auc, 'acc': acc, 'f1': f1, 'prauc': prauc, 'minrp': minrp, 'recall': recall, 'precision': precision,
            'pr08': p_val1, 'pr09': p_val2}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for values, times, varis, obs_mask, demo, labels in tqdm(loader, desc="Train", leave=False):
        values, times, varis = values.to(device), times.to(device), varis.to(device)
        obs_mask, demo, labels = obs_mask.to(device), demo.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = model(values, times, varis, obs_mask, demo, labels=labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0

    criterion = nn.BCELoss()

    with torch.no_grad():
        for values, times, varis, obs_mask, demo, labels in loader:
            values, times, varis = values.to(device), times.to(device), varis.to(device)
            obs_mask, demo, labels = obs_mask.to(device), demo.to(device), labels.to(device)

            probs = model(values, times, varis, obs_mask, demo, labels=None)
            loss = criterion(probs, labels)
            total_loss += loss.item()

            preds = (probs > 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_probs, all_preds)
    return total_loss / len(loader), metrics


def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"Device: {device} | Model: istrats (Continuous Time Transformer)")

    with open(CONFIG['pkl_path'], "rb") as f:
        _, lbl_df, train_list, val_list, test_list = pickle.load(f)
    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_outcomes(lbl_df)

    t_lbls = [lbl_df[lbl_df.ts_id == tid].in_hospital_mortality.item() for tid in train_list]
    v_lbls = [lbl_df[lbl_df.ts_id == tid].in_hospital_mortality.item() for tid in val_list]
    full_label_dict = {**dict(zip(train_list, t_lbls)), **dict(zip(val_list, v_lbls))}

    folds_data = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=CONFIG['num_folds'],
        window_size=CONFIG['window_size'], seed=443
    )

    print("\nPreparing Test Data...")
    test_set = MIMICStratsDataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                                  total_hours=CONFIG['total_hours'], max_seq_len=CONFIG['max_seq_len'])
    test_loader = DataLoader(test_set, batch_size=CONFIG['eval_batch_size'], shuffle=False, num_workers=0)

    model_args = Args(
        model_type=CONFIG['model_type'], V=CONFIG['V'], D=CONFIG['D'], hid_dim=CONFIG['hid_dim'],
        num_layers=CONFIG['num_layers'], num_heads=CONFIG['num_heads'], attention_dropout=CONFIG['attention_dropout'],
        dropout=CONFIG['dropout'], pretrain=CONFIG['pretrain'], load_ckpt_path=CONFIG['load_ckpt_path'],
        pos_class_weight=CONFIG['pos_class_weight'], M=CONFIG['M'], kernel_size=CONFIG['kernel_size'], r=CONFIG['r']
    )

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'=' * 10} istrats Fold {fold_idx + 1} / {CONFIG['num_folds']} {'=' * 10}")

        train_set = MIMICStratsDataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids,
                                       total_hours=CONFIG['total_hours'], max_seq_len=CONFIG['max_seq_len'])
        val_set = MIMICStratsDataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids,
                                     total_hours=CONFIG['total_hours'], max_seq_len=CONFIG['max_seq_len'])

        curr_labels = [full_label_dict[uid] for uid in fold_train_ids]
        class_counts = np.bincount(curr_labels)
        weights = 1. / np.maximum(class_counts, 1)
        samples_weights = torch.tensor([weights[l] for l in curr_labels], dtype=torch.double)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        train_loader = DataLoader(train_set, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=CONFIG['train_batch_size'], shuffle=False, num_workers=0)

        model = Strats(model_args).to(device)

        if CONFIG['load_ckpt_path'] is not None and os.path.exists(CONFIG['load_ckpt_path']):
            print(f"  [Info] Loading pre-trained weights from {CONFIG['load_ckpt_path']}...")

            curr_state_dict = model.state_dict()
            pt_state_dict = torch.load(CONFIG['load_ckpt_path'], map_location=device)

            matched_keys = 0
            for k, v in pt_state_dict.items():
                if k in curr_state_dict:
                    curr_state_dict[k] = v
                    matched_keys += 1

            model.load_state_dict(curr_state_dict)
            print(f"  [Info] Successfully loaded {matched_keys} matching parameter tensors.")
        elif CONFIG['load_ckpt_path'] is not None:
            print(f"  [Warning] Pre-trained weights NOT found at {CONFIG['load_ckpt_path']}!")


        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

        best_val_auc = 0.0
        best_val_prauc = 0.0
        best_val_loss=1
        save_path = os.path.join(CONFIG['save_dir'], f"best_model_fold{fold_idx + 1}.pt")

        for epoch in range(CONFIG['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_metrics = evaluate(model, val_loader, device)
            # scheduler.step()

            if val_metrics['auc']+val_metrics['prauc'] > best_val_auc+best_val_prauc:
                best_val_auc = val_metrics['auc']
                best_val_prauc = val_metrics['prauc']
                torch.save(model.state_dict(), save_path)
                print(f"  Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val AUC {val_metrics['auc']:.4f} | Val PRAUC {val_metrics['prauc']:.4f} *Saved*")
            else:
                print(f"  Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val AUC {val_metrics['auc']:.4f}")

        model.load_state_dict(torch.load(save_path, map_location=device))
        _, test_metrics = evaluate(model, test_loader, device)
        print(
            f"Fold {fold_idx + 1} Finished. Best Val AUC: {best_val_auc:.4f} | "
            f"Test AUC: {test_metrics['auc']:.4f} | Test PR-AUC: {test_metrics['prauc']:.4f}"
        )


if __name__ == "__main__":
    main()
