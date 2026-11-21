import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import pickle
from tqdm import tqdm

from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.mtgnn.net import GTNetClassifier
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_outcomes,
)

CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std.json',
    'save_dir': 'checkpoints/gtnet_single_scale',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'duration': 120,
    'seq_len': 12,

    'batch_size': 32,
    'epochs': 10,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.3,

    'num_folds': 6,
    'window_size': 3,

    'conv_channels': 32,
    'residual_channels': 32,
    'skip_channels': 64,
    'end_channels': 128,
    'layers': 3,
    'gcn_depth': 2
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
    print(f"Device: {device} | Scale: {CONFIG['duration']} mins")

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

    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'=' * 10} Fold {fold_idx + 1} / {CONFIG['num_folds']} {'=' * 10}")

        train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids,
                                  slot_duration_mins=CONFIG['duration'])
        val_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids,
                                slot_duration_mins=CONFIG['duration'])

        curr_labels = [full_label_dict[uid] for uid in fold_train_ids]
        class_counts = np.bincount(curr_labels)
        weights = 1. / np.maximum(class_counts, 1)
        samples_weights = torch.tensor([weights[l] for l in curr_labels], dtype=torch.double)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

        # num_nodes = TS Features (129) + Demo Features (2) = 131
        total_nodes = train_set.num_features + train_set.num_demo

        model = GTNetClassifier(
            num_nodes=total_nodes,
            seq_len=CONFIG['seq_len'],
            in_dim=1,
            out_dim=256,
            num_classes=2,
            dropout=CONFIG['dropout'],
            device=device,
            layers=CONFIG['layers'],
            conv_channels=CONFIG['conv_channels'],
            residual_channels=CONFIG['residual_channels'],
            skip_channels=CONFIG['skip_channels'],
            end_channels=CONFIG['end_channels']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        save_path = os.path.join(CONFIG['save_dir'], f"best_model_fold{fold_idx + 1}.pt")

        for epoch in range(CONFIG['epochs']):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"  Epoch {epoch + 1}: Train {train_loss:.4f} | Val {val_loss:.4f} *Saved*")
            else:
                print(f"  Epoch {epoch + 1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        print(f"Fold {fold_idx + 1} Finished. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
