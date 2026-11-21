import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import time
import os
import pickle
from tqdm import tqdm

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.mstdgnn.STDGNN import GNNTest
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_csv,
)

CONFIG = {
    'pkl_path': 'data/mimic-iv/mimic_iv.pkl',
    'json_path': 'data/mimic-iv/var_mean_std.json',
    'train_label_path': 'dataset/train_labels.pkl',
    'val_label_path': 'dataset/val_labels.pkl',
    'save_dir': 'checkpoints/multiscale_ensemble_mimic_iv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'batch_size': 16,
    'epochs': 10,
    'lr': 1e-4,
    'dropout': 0.3,

    'scales': [
        {'duration': 60, 'seq_len': 24},
        {'duration': 120, 'seq_len': 12},
        {'duration': 240, 'seq_len': 6}
    ],

    'gnn_model_type': 'dyGIN2d',
    'num_layers': 3,
    'groups': 4,
    'pool_ratio': 0.2,
    'kern_size': [3, 3, 3],
    'in_dim': 64,
    'hidden_dim': 128,
    'out_dim': 256,
    'num_classes': 2,

    'num_folds': 6,
    'window_size': 3,
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x, batch_demo, batch_y in tqdm(loader, desc="Train", leave=False):
        batch_x, batch_demo, batch_y = batch_x.to(device), batch_demo.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, batch_demo)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_demo, batch_y in loader:
            batch_x, batch_demo, batch_y = batch_x.to(device), batch_demo.to(device), batch_y.to(device)
            outputs = model(batch_x, batch_demo)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"Device: {device}")

    print("Loading Data Indices...")
    with open(CONFIG['pkl_path'], "rb") as f:
        _, lbl_df, train_list, val_list, test_list = pickle.load(f)

    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_csv(
        "data/mimic-iv/mimic-iv-3.1/icu/icustays.csv.gz"
    )

    # with open(CONFIG['train_label_path'], "rb") as f:
    #     t_lbls = pickle.load(f)
    # with open(CONFIG['val_label_path'], "rb") as f:
    #     v_lbls = pickle.load(f)
    t_lbls = [lbl_df[lbl_df.ts_id==tid].in_hospital_mortality.item() for tid in train_list]
    v_lbls = [lbl_df[lbl_df.ts_id==tid].in_hospital_mortality.item() for tid in val_list]

    full_label_dict = {**dict(zip(train_list, t_lbls)), **dict(zip(val_list, v_lbls))}

    folds_data = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=CONFIG['num_folds'],
        window_size=CONFIG['window_size'], seed=443
    )

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'#' * 30}\nStarting Fold {fold_idx + 1} / {CONFIG['num_folds']}\n{'#' * 30}")

        for scale_cfg in CONFIG['scales']:
            duration = scale_cfg['duration']
            seq_len = scale_cfg['seq_len']

            print(f"\n>>> Training Scale: {duration} mins (Seq Len: {seq_len})")

            train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids,
                                      slot_duration_mins=duration)
            val_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids,
                                    slot_duration_mins=duration)

            curr_labels = [full_label_dict[uid] for uid in fold_train_ids]
            class_counts = np.bincount(curr_labels)
            weights = 1. / np.maximum(class_counts, 1)
            samples_weights = torch.tensor([weights[l] for l in curr_labels], dtype=torch.double)
            sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

            train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

            # num_nodes = ts_features (129) + demo_features (2) = 131
            total_nodes = train_set.num_features + train_set.num_demo

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
                num_nodes=total_nodes,
                num_classes=CONFIG['num_classes'],
                dropout=CONFIG['dropout']
            ).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            save_name = f"fold{fold_idx + 1}_scale{duration}.pt"
            save_path = os.path.join(CONFIG['save_dir'], save_name)

            for epoch in range(CONFIG['epochs']):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = evaluate_loss(model, val_loader, criterion, device)

                print(f"  Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_path)

            print(f"  >>> Finished Scale {duration}m. Best Val Loss: {best_val_loss:.4f}. Saved to {save_name}")


if __name__ == "__main__":
    main()
