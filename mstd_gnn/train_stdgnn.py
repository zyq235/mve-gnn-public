import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from evaluator import trapezoidal_pr_auc
import time
import os
from torch.utils.data import WeightedRandomSampler
# Import your custom modules
# Assuming files are named mimic3_MSTD_dataset.py and STDGNN.py in the same directory
from dataset.mimic3_MSTD_dataset import MIMIC3Dataset
from models.mstdgnn.STDGNN import GNNTest
from tqdm import tqdm
import pickle
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_outcomes,
)

# --- Hyperparameters & Config ---
CONFIG = {
    'pkl_path': 'data/mimic3/mimic_iii.pkl',  # Modify path as needed
    'json_path': 'data/mimic3/var_mean_std.json',  # Modify path as needed
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'checkpoints/mstdgnn',

    # Model Architecture Params
    'gnn_model_type': 'dyGIN2d',
    'num_layers': 3,
    'groups': 4,
    'pool_ratio': 0.2,
    'kern_size': [3, 3, 3],  # Must match num_layers
    'in_dim': 64,
    'hidden_dim': 128,
    'out_dim': 256,
    'dropout': 0.3,
    'seq_len': 24,
    'num_classes': 2,
    'stage': 'train'
}


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_demo, batch_y in tqdm(loader):
        # Move to device
        batch_x = batch_x.to(device)  # [B, 129, 24]
        batch_demo = batch_demo.to(device)  # [B, 2]
        batch_y = batch_y.to(device)  # [B]

        optimizer.zero_grad()

        # Forward pass
        # Model expects (nume, demo)
        outputs = model(batch_x, batch_demo)  # [B, num_classes]

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store for rough accuracy calculation if needed
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(batch_y.detach().cpu().numpy())
    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / len(loader)
    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_x, batch_demo, batch_y in tqdm(loader):
            batch_x = batch_x.to(device)
            batch_demo = batch_demo.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, batch_demo)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Get probabilities for Class 1 (Mortality)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)

    # Calculate Metrics
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # Handle case with only one class in batch

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    prauc = trapezoidal_pr_auc(all_labels, all_probs)
    pr, re, thresholds = precision_recall_curve(all_labels, all_probs)
    minrp = np.minimum(pr, re).max()
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'prauc': prauc,
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'min_rp': minrp
    }
    return metrics


def main():
    print(f"Running on device: {CONFIG['device']}")

    # 1. Load Datasets
    print("Loading datasets...")

    mimic_pkl_path = CONFIG['pkl_path']
    with open(mimic_pkl_path, "rb") as f:
        _, labels_df, train_list, val_list, test_list = pickle.load(f)

    combined_ids = np.concatenate([train_list, val_list])
    id_to_patient = patient_mapping_from_outcomes(labels_df)
    combined_labels_dict = dict(
        zip(labels_df['ts_id'], labels_df['in_hospital_mortality'])
    )

    try:
        test_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=test_list)
    except FileNotFoundError:
        print("Error: Data files not found. Please check paths in CONFIG.")
        return
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    n_folds = 6
    folds_data = get_sliding_window_splits(
        combined_ids, id_to_patient, n_splits=n_folds,
        window_size=3, seed=443
    )
    os.makedirs(CONFIG['save_path'], exist_ok=True)
    for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(folds_data):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_list, id_to_patient)
        print(f"\n{'=' * 20} Start Fold {fold_idx + 1} / {n_folds} {'=' * 20}")
        train_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_train_ids)
        val_set = MIMIC3Dataset(CONFIG['pkl_path'], CONFIG['json_path'], tsid_list=fold_val_ids)

        curr_train_labels = np.array([combined_labels_dict[cid] for cid in fold_train_ids])
        class_count = np.bincount(curr_train_labels.astype(int))
        class_count[class_count == 0] = 1
        weights = 1. / class_count
        sample_weights = weights[curr_train_labels.astype(int)]
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

        num_ts_features = train_set.num_features  # 129
        num_demo_features = train_set.num_demo  # 2
        total_nodes = num_ts_features + num_demo_features  # 131

        print(f"Initializing Model with {total_nodes} nodes ({num_ts_features} TS + {num_demo_features} Demo)...")

        model = GNNTest(
            gnn_model_type=CONFIG['gnn_model_type'],
            num_layers=CONFIG['num_layers'],
            groups=CONFIG['groups'],
            pool_ratio=CONFIG['pool_ratio'],
            kern_size=CONFIG['kern_size'],
            in_dim=CONFIG['in_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            out_dim=CONFIG['out_dim'],
            seq_len=CONFIG['seq_len'],
            num_nodes=total_nodes,  # <--- Critical Update
            num_classes=CONFIG['num_classes'],
            dropout=CONFIG['dropout']
        )

        model = model.to(CONFIG['device'])
        # 3. Setup Optimizer and Loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00015, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=CONFIG['epochs'],)

        # Optional: Weighted Loss if dataset is very imbalanced
        # pos_weight = torch.tensor([ratio]).to(device)
        criterion = nn.CrossEntropyLoss()

        if CONFIG['stage'] == 'train':
            # 4. Training Loop
            best_val_auc = 0.0
            print("Start Training...")

            for epoch in range(CONFIG['epochs']):
                start_time = time.time()

                # Train
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,CONFIG['device'])

                # Validate
                val_metrics = evaluate(model, val_loader, criterion, CONFIG['device'])

                end_time = time.time()
                epoch_mins = (end_time - start_time) / 60

                print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Time: {epoch_mins:.2f}m")
                print(f"    Train Loss: {train_loss:.4f}")
                print(f"    Val Loss:   {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | PRAUC: {val_metrics['prauc']:.4f}")

                # Save Best Model
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    torch.save(model.state_dict(), os.path.join(CONFIG['save_path'], f'best_model_fold{fold_idx + 1}.pt'))
                    print(f"    >>> Best model saved (AUC: {best_val_auc:.4f})")

            # 5. Testing Loop
            print("\nTraining Finished. Testing Best Model...")

            # Load best weights
            best_path = os.path.join(
                CONFIG['save_path'], f'best_model_fold{fold_idx + 1}.pt'
            )
            model.load_state_dict(torch.load(best_path, map_location=CONFIG['device']))

            test_metrics = evaluate(model, test_loader, criterion, CONFIG['device'])

            print("=" * 30)
            print("FINAL TEST RESULTS")
            print("=" * 30)
            print(f"AUC:       {test_metrics['auc']:.4f}")
            print(f"Accuracy:  {test_metrics['acc']:.4f}")
            print(f"F1 Score:  {test_metrics['f1']:.4f}")
            print(f"Precision: {test_metrics['precision']:.4f}")
            print(f"Recall:    {test_metrics['recall']:.4f}")
            print(f"PRAUC:    {test_metrics['prauc']:.4f}")
            print(f"Loss:      {test_metrics['loss']:.4f}")
            print(f"minrp:      {test_metrics['min_rp']:.4f}")
            print("=" * 30)
        else:
            print("Evaluation Mode: Loading Best Model and Evaluating on Test Set...")
            model.load_state_dict(torch.load(os.path.join(CONFIG['save_path'], f'best_model_fold{fold_idx + 1}.pt')))
            test_metrics = evaluate(model, test_loader, criterion, CONFIG['device'])

            print("=" * 30)
            print("TEST RESULTS")
            print("=" * 30)
            print(f"AUC:       {test_metrics['auc']:.4f}")
            print(f"Accuracy:  {test_metrics['acc']:.4f}")
            print(f"F1 Score:  {test_metrics['f1']:.4f}")
            print(f"Precision: {test_metrics['precision']:.4f}")
            print(f"Recall:    {test_metrics['recall']:.4f}")
            print(f"PRAUC:    {test_metrics['prauc']:.4f}")
            print(f"Loss:      {test_metrics['loss']:.4f}")
            print(f"minrp:      {test_metrics['min_rp']:.4f}")
            print("=" * 30)

if __name__ == "__main__":
    main()
