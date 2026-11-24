import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
)
from evaluator import trapezoidal_pr_auc

from dataset.mimic_raindrop_dataset import MIMIC3Dataset
from models.Raindrop.models_rd import Raindrop_v2
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_outcomes,
)

CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',
    'json_path': 'data/mimic3/var_mean_std_old.json',
    'save_dir': 'checkpoints/raindrop_kfold_mm3_old',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'duration': 60,
    'total_hours': 24,
    'seq_len': 24,

    'batch_size': 32,
    'epochs': 20,
    'lr': 0.0001,
    'dropout': 0.2,

    'num_folds': 6,
    'window_size': 3,

    'd_ob': 4,
    'nhead': 2,
    'nhid': 256,
    'nlayers': 2
}


def prepare_input(x, mask):
    """Build Raindrop inputs with values followed by masks along the feature axis."""
    x = x.permute(2, 0, 1)  # [24, Batch, 129]
    mask = mask.permute(2, 0, 1)  # [24, Batch, 129]
    return torch.cat([x, mask], dim=-1)  # [24, Batch, 258]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, mask, times, demo, y in tqdm(loader, desc="Train", leave=False):
        x, mask, times, demo, y = x.to(device), mask.to(device), times.to(device), demo.to(device), y.to(device)

        src = prepare_input(x, mask)
        times_in = times.permute(1, 0)  # [Batch, Time] -> [Time, Batch]

        lengths = torch.sum((torch.sum(mask, dim=1) > 0), dim=1).long()

        optimizer.zero_grad()
        outputs, _, _ = model(src, demo, times_in, lengths)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, mask, times, demo, y in loader:
            x, mask, times, demo, y = x.to(device), mask.to(device), times.to(device), demo.to(device), y.to(device)
            src = prepare_input(x, mask)
            times_in = times.permute(1, 0)
            lengths = torch.sum((torch.sum(mask, dim=1) > 0), dim=1).long()

            outputs, _, _ = model(src, demo, times_in, lengths)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)

    try:
        val_auc = roc_auc_score(all_labels, all_probs)
        val_auprc = trapezoidal_pr_auc(all_labels, all_probs)
    except ValueError:
        val_auc = 0.5
        val_auprc = 0.0

    return avg_loss, val_auc, val_auprc


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']

    with open(CONFIG['pkl_path'], "rb") as f:
        _, lbl_df, train_list, val_list, test_list = pickle.load(f)
    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_outcomes(lbl_df)

    folds_data = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=CONFIG['num_folds'],
        window_size=CONFIG['window_size'], seed=443
    )
    test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list,
                              slot_duration_mins=CONFIG['duration'])
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'=' * 10} Raindrop Fold {fold_idx + 1} / {CONFIG['num_folds']} {'=' * 10}")

        train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids,
                                  slot_duration_mins=CONFIG['duration'])
        val_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids,
                                slot_duration_mins=CONFIG['duration'])

        train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

        d_inp = train_set.num_features  # 129
        d_static = train_set.num_demo  # 2

        global_structure = torch.ones(d_inp, d_inp).to(device)

        model = Raindrop_v2(
            d_inp=d_inp,
            d_model=d_inp * CONFIG['d_ob'],
            nhead=CONFIG['nhead'],
            nhid=CONFIG['nhid'],
            nlayers=CONFIG['nlayers'],
            dropout=CONFIG['dropout'],
            max_len=CONFIG['seq_len'],
            d_static=d_static,
            n_classes=2,
            global_structure=global_structure
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

        best_val_auc = 0.0
        save_path = os.path.join(CONFIG['save_dir'], f"raindrop_fold{fold_idx + 1}.pt")

        for epoch in range(CONFIG['epochs']):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc, val_auprc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), save_path)
                print(
                    f"  Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val AUROC {val_auc:.4f} | Val AUPRC {val_auprc:.4f} *Saved*")
            else:
                print(
                    f"  Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val AUROC {val_auc:.4f} | Val AUPRC {val_auprc:.4f}")

        model.load_state_dict(torch.load(save_path, map_location=device))
        _, test_auc, test_auprc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Fold {fold_idx + 1} Test AUROC {test_auc:.4f} | Test AUPRC {test_auprc:.4f}"
        )


if __name__ == "__main__":
    main()
