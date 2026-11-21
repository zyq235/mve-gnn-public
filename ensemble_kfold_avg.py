import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score, precision_recall_curve, fbeta_score
from sklearn.metrics import auc
from models.graphmae.finetune_graph_model import HierarchicalEHRModel
from dataset.mimic3_patient_dataset import ICULargeGraphDataset
from models.graphmae.utils import load_pretrain_profile
import pickle
import numpy as np
import os
from pathlib import Path
from types import SimpleNamespace
import argparse
import pandas as pd


def load_all_args(dataset):
    """Read the encoder architecture for the complete fine-tuned checkpoints."""
    settings_root = Path(__file__).resolve().parent / "settings"
    profile = load_pretrain_profile(settings_root / "pretrain_model_params.json", dataset)
    return SimpleNamespace(
        num_heads=profile.get("num_heads", 8),
        attn_drop=0.2,
        negative_slope=0.2,
        in_drop=0.2,
        knn=10,
        norm=profile.get("norm", "layernorm"),
        drop_edge_rate=0.0,
        num_features=64,
        num_hidden=profile["num_hidden"],
        num_layers=profile["num_layers"],
        activation="prelu",
        mask_rate=profile["mask_rate"],
        loss_lmb=profile.get("loss_lmb", 1.0),
        mse_lmb=profile.get("mse_lmb", 0.5),
        alpha_l=2.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cat_var=False,
        aggr_type="combine",
    )


def load_model(args, checkpoint_path, mode, device):

    print(f"Loading {mode} model from {checkpoint_path}...")
   
    model = HierarchicalEHRModel(
        input_dim=args.num_features,
        hidden_dim=args.num_hidden,
        args=args,
        num_classes_type=args.num_classes_type,
        num_clusters_k=args.num_clusters_k,
        num_demo=args.num_demo,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggr_type=args.aggr_type
    )

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def get_precision_at_target_recall(labels, probs, target_recall):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    idx = np.where(recalls>=target_recall)[0][-1]
    
    p_at_r = precisions[idx]
    corresponding_threshold = thresholds[idx] if idx<len(thresholds) else 1.0
    return p_at_r, corresponding_threshold

def eval_metrics(y_true, y_prob,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)
    minrp = np.minimum(precision, recall).max()
    # pr_auc = average_precision_score(y_true, final_probs)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    p_val1, thresh = get_precision_at_target_recall(y_true, y_prob, target_recall=0.8)
    p_val2, thresh = get_precision_at_target_recall(y_true, y_prob, target_recall=0.9)
    return {'auc':roc_auc,'pr_auc':pr_auc,'minrp':minrp,'pr08':p_val1,'pr09':p_val2}
    
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_ensemble(models_dict, test_loader, device, ensemble_mode="soft", coef=None):
    """
    Args:
        models_dict: {'type': model_obj, 'diff_pool': model_obj, ...}
        test_loader: DataLoader
        device: torch.device
        ensemble_mode: 'soft' 
    """
    y_true = []
    ts_ids = []
    probs_dict = {k: [] for k in models_dict.keys()}
    preds_dict = {k: [] for k in models_dict.keys()}
    final_res_dict = dict()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Ensemble Testing ({ensemble_mode})"):
            batch = batch.to(device)
            y_true.append(batch.y.cpu().numpy())
            ts_ids.append(batch.ts_id.cpu().numpy())

            for mode_name, model_list in models_dict.items():
                batch_logits = 0
                for model in model_list:
                    model.eval()
                    logits = model(batch, mode=mode_name)
                    batch_logits += logits

                avg_logits = batch_logits / len(model_list)
                probs = torch.sigmoid(avg_logits).cpu().numpy()
                probs_dict[mode_name].append(probs)

    y_true = np.concatenate(y_true).flatten()
    ts_ids = np.concatenate(ts_ids).flatten()
    
    for k in probs_dict:
        probs_dict[k] = np.concatenate(probs_dict[k]).flatten()
        preds_dict[k] = (probs_dict[k] > 0.5).astype(int)
    num_models = len(probs_dict)
    first_key = list(probs_dict.keys())[0]

    print("\n--- Individual Performance ---")
    for k, v in probs_dict.items():
        try:
            res_dict = eval_metrics(y_true, probs_dict[k], preds_dict[k])
            final_res_dict[k]=res_dict
            print(f"Model [{k}] AUC: {res_dict['auc']:.4f}, PRAUC: {res_dict['pr_auc']:.4f}")
        except Exception as e:
            print(f"Model [{k}] Error calculating AUC: {e}")

    if ensemble_mode == "soft":      
        avg_probs = np.zeros_like(probs_dict[first_key])
        if not coef:
            for v in probs_dict.values():
                avg_probs += v
       
            final_probs = avg_probs / num_models
        else:
            for i,v in enumerate(probs_dict.values()):
                avg_probs += v*coef[i]
            final_probs=avg_probs
        final_preds = (final_probs > 0.5).astype(int)

    elif ensemble_mode == "hard":
        vote_counts = np.zeros_like(probs_dict[first_key])
        # === Hard Voting ===
        for v in probs_dict.values():
            vote_counts += (v > 0.5).astype(int)
        final_preds = (vote_counts > (num_models / 2)).astype(int)
        final_probs = vote_counts / num_models

    else:
        raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}. Use 'soft' or 'hard'.")

    precision, recall, thresholds = precision_recall_curve(y_true, final_probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, final_probs)
    minrp = np.minimum(precision, recall).max()
    # pr_auc = average_precision_score(y_true, final_probs)
    f1 = f1_score(y_true, final_preds)
    prec = precision_score(y_true, final_preds)
    rec = recall_score(y_true, final_preds)
    p_val1, thresh = get_precision_at_target_recall(y_true, final_probs, target_recall=0.8)
    p_val2, thresh = get_precision_at_target_recall(y_true, final_probs, target_recall=0.9)

    print("\n========================================")
    print(f"Ensemble Result | Mode: {ensemble_mode.upper()} | Models: {num_models}")
    print("========================================")
    print(f"AUC:       {roc_auc:.4f}")
    print(f"PR AUC:       {pr_auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"P@R08:    {p_val1:.4f}")
    print(f"P@R09:    {p_val2:.4f}")
    print(f"thresh:    {thresh:.4f}")
    print(f"min_rp:    {minrp:.4f}")
    print("----------------------------------------")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, final_preds))
    print("========================================")
    
    type_only_correct_ids = []
    all_correct_ids = []

    if all(k in preds_dict for k in ['type', 'diff_pool', 'time']):
        correct_type = (preds_dict['type'] == y_true)
        correct_diff = (preds_dict['diff_pool'] == y_true)
        correct_time = (preds_dict['time'] == y_true)

        mask_type_only = correct_type & (~correct_diff) & (~correct_time)
        type_only_correct_ids = ts_ids[mask_type_only].tolist()

        mask_all = correct_type & correct_diff & correct_time
        all_correct_ids = ts_ids[mask_all].tolist()
        
        print(f"\n Type correct only: {len(type_only_correct_ids)}")
        print(f"\n All correct: {len(all_correct_ids)}")
    return {"AUC":roc_auc, "PRAUC":pr_auc,"P@R08":p_val1,"P@R09":p_val2,"MIN_RP":minrp, "res_dict":final_res_dict, 
    "type_only_ids": type_only_correct_ids, "all_correct_ids": all_correct_ids }


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mimic3', 'mimic4'), default='mimic3')
    parser.add_argument('--device', type=str, default='cuda')
    margs = parser.parse_args()
    
    if margs.dataset=="mimic4":
        root_path = "data/mimic-iv/"
        mimic_pkl_path = "data/mimic-iv/mimic_iv.pkl"
        var2ind_path = "data/mimic-iv/var2ind.json"
        var_mean_std_path = "data/mimic-iv/var_mean_std.json"
    elif margs.dataset=="mimic3":
        root_path = "data/mimic3/"
        mimic_pkl_path = "data/mimic3/mimic_iii.pkl"
        var2ind_path = "data/mimic3/var2ind.json"
        var_mean_std_path = "data/mimic3/var_mean_std.json"

    args = load_all_args(margs.dataset)

    print("Loading test dataset...")
    with open(mimic_pkl_path, "rb") as f:
        _, _, _, _, test_list = pickle.load(f)
    del _

    test_dataset = ICULargeGraphDataset(
        root=root_path,
        ts_id=test_list,
        var2ind_path=var2ind_path,
        var_mean_std_path=var_mean_std_path,
        max_nodes=1000,
        time_limit=24,
        pkl_path=mimic_pkl_path)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    models_dict = {}
    
    #fold_turn=[[1,1,1], #type, time, diff
    #          [1,1,1],
    #          [1,1,1],
    #          [1,1,1],
    #          [1,1,1],
    #          [1,1,1]
    #          ]
    if margs.dataset=="mimic4":
        fold_turn=[[3,4,2], #type, time, diff
                  [1,4,4],
                  [1,3,5],
                  [5,5,2],
                  [4,5,4],
                  [2,5,5]
                  ]
    else:
        fold_turn=[[5,1,3], #type, time, diff
                  [5,1,5],
                  [1,1,3],
                  [5,5,4],
                  [2,5,2],
                  [3,4,5]
                  ]
    total_res = {"AUC":[], "PRAUC":[],"P@R08":[],"P@R09":[],"MIN_RP":[]}
    seperate_res = {"time":[], "type":[], "diff_pool":[]}

    for fold in range(1, 7):
        print(f"\n========== Processing Fold {fold} ==========")

        models_dict = {'type': [], 'diff_pool': [], 'time': []}

        for turn in range(1, 6):
            if margs.dataset == "mimic4":
                args.ckpt_type = f"checkpoints/mm4_gat_ft_pool_type_fold{fold}_turn{turn}_auc/checkpoint_fold{fold}_turn{turn}_best.pt"
                args.ckpt_diff = f"checkpoints/mm4_gat_ft_pool_diff_fold{fold}_turn{turn}_auc/checkpoint_fold{fold}_turn{turn}_best.pt"
                args.ckpt_time = f"checkpoints/mm4_gat_ft_pool_time_fold{fold}_turn{turn}_auc/checkpoint_fold{fold}_turn{turn}_best.pt"
            else:
                args.ckpt_type = f"checkpoints/mm3_gat_ft_pool_type_fold{fold}_turn{turn}_auc/checkpoint_fold{fold}_turn{turn}_best.pt"
                args.ckpt_diff = f"checkpoints/mm3_gat_ft_pool_diff_fold{fold}_turn{turn}_auc/checkpoint_fold{fold}_turn{turn}_best.pt"
                args.ckpt_time = f"checkpoints/mm3_gat_ft_pool_time_fold{fold}_turn{turn}_auc/checkpoint_fold{fold}_turn{turn}_best.pt"

            if os.path.exists(args.ckpt_type):
                model_type = HierarchicalEHRModel(
                    input_dim=1, hidden_dim=args.num_hidden, args=args,
                    num_classes_type=11, num_clusters_k=10, num_demo=2,
                    num_heads=8, num_layers=args.num_layers, dropout=0.2, aggr_type="combine"
                )
                model_type.load_state_dict(torch.load(args.ckpt_type, map_location=args.device))
                model_type.to(args.device)
                models_dict['type'].append(model_type)

            if os.path.exists(args.ckpt_diff):
                model_diff = HierarchicalEHRModel(
                    input_dim=1, hidden_dim=args.num_hidden, args=args,
                    num_classes_type=11, num_clusters_k=12, num_demo=2,
                    num_heads=8, num_layers=args.num_layers, dropout=0.2, aggr_type="combine"
                )
                model_diff.load_state_dict(torch.load(args.ckpt_diff, map_location=args.device))
                model_diff.to(args.device)
                models_dict['diff_pool'].append(model_diff)

            if os.path.exists(args.ckpt_time):
                model_time = HierarchicalEHRModel(
                    input_dim=1, hidden_dim=args.num_hidden, args=args,
                    num_classes_type=11, num_clusters_k=10, num_demo=2,
                    num_heads=8, num_layers=args.num_layers, dropout=0.2, aggr_type="combine"
                )
                model_time.load_state_dict(torch.load(args.ckpt_time, map_location=args.device))
                model_time.to(args.device)
                models_dict['time'].append(model_time)

        models_dict = {k: v for k, v in models_dict.items() if len(v) > 0}

        if len(models_dict) == 0:
            print(f"Error: No checkpoints found for fold {fold}. Please check paths.")
            continue

        res = evaluate_ensemble(models_dict, test_loader, args.device, ensemble_mode="soft", coef=None)

        seperate_res["time"].append(res["res_dict"]["time"])
        seperate_res["type"].append(res["res_dict"]["type"])
        seperate_res["diff_pool"].append(res["res_dict"]["diff_pool"])

        total_res["AUC"].append(res["AUC"])
        total_res["PRAUC"].append(res["PRAUC"])
        total_res["P@R08"].append(res["P@R08"])
        total_res["P@R09"].append(res["P@R09"])
        total_res["MIN_RP"].append(res["MIN_RP"])


        del models_dict
        torch.cuda.empty_cache()
    for key in total_res.keys():
        total_res[key]=np.array(total_res[key])

    time_df = pd.DataFrame(seperate_res["time"])
    type_df = pd.DataFrame(seperate_res["type"])
    diff_df = pd.DataFrame(seperate_res["diff_pool"])
    print("========================================")
    print("Time:")
    for key in time_df.columns:
        print(f"{key:<10}:{time_df.mean()[key]:.4f} +- {time_df.std()[key]:.4f}")
    print("========================================")
    print("Type:")
    for key in type_df.columns:
        print(f"{key:<10}:{type_df.mean()[key]:.4f} +- {type_df.std()[key]:.4f}")
    print("========================================")
    print("Diff:")
    for key in diff_df.columns:
        print(f"{key:<10}:{diff_df.mean()[key]:.4f} +- {diff_df.std()[key]:.4f}")

    print("========================================")
    print("Ensemble:")
    print(f"AUC:       {total_res['AUC'].mean():.4f}+-{total_res['AUC'].std():.4f}")
    print(f"PR AUC:       {total_res['PRAUC'].mean():.4f}+-{total_res['PRAUC'].std():.4f}")
    print(f"P@R08:    {total_res['P@R08'].mean():.4f}+-{total_res['P@R08'].std():.4f}")
    print(f"P@R09:    {total_res['P@R09'].mean():.4f}+-{total_res['P@R09'].std():.4f}")
    print(f"min_rp:    {total_res['MIN_RP'].mean():.4f}+-{total_res['MIN_RP'].std():.4f}")
    print("========================================")
