"""Pretrain the MIMIC-III graph encoder with masked reconstruction."""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset.mimic3_patient_dataset import ICULargeGraphDataset
from models.graphmae.mimic_graph_extract import build_model
from models.graphmae.utils import (
    build_parser,
    create_optimizer,
    load_pretrain_profile,
    set_random_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = build_parser()
    parser.description = "Pretrain the graph encoder on MIMIC-III."
    parser.set_defaults(dataset="mimic3", knn=10, num_heads=8, norm="layernorm")
    parser.add_argument(
        "--settings-root", "--settings_root", type=Path, default=Path("settings"),
        help="Directory containing pretrain_model_params.json, relative to the project root.",
    )

    selection = parser.parse_args(argv)
    settings_root = PROJECT_ROOT / selection.settings_root.expanduser()
    config_path = settings_root / "pretrain_model_params.json"
    try:
        profile = load_pretrain_profile(config_path, selection.dataset)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    defaults = argparse.Namespace(**profile)
    defaults.dataset = selection.dataset
    defaults.settings_root = selection.settings_root
    return parser.parse_args(argv, namespace=defaults)


def validate(model, dataloader, device):
    model.eval()
    sce_loss_list = []
    mse_loss_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch_g = batch.to(device)
            _, loss_dict = model(batch_g)

            sce_loss_list.append(loss_dict["sce"])
            mse_loss_list.append(loss_dict["mse"])

    avg_sce = np.mean(sce_loss_list)
    avg_mse = np.mean(mse_loss_list)

    model.train()
    return avg_sce, avg_mse


def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, args, writer=None):
    train_loader, eval_loader = dataloaders
    epoch_iter = range(max_epoch)

    for epoch in tqdm(epoch_iter, desc="Epochs"):
        model.train()
        sce_loss_list = []
        mse_loss_list = []

        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        for i, batch in train_iter:
            global_step = epoch * len(train_loader) + i

            batch_g = batch.to(device)
            loss, loss_dict = model(batch_g)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            sce_loss_list.append(loss_dict["sce"])
            mse_loss_list.append(loss_dict["mse"])

            if i % 100 == 0 and writer is not None:
                writer.add_scalar("train/sce_loss", np.mean(sce_loss_list), global_step=global_step)
                writer.add_scalar("train/mse_loss", np.mean(mse_loss_list), global_step=global_step)
                sce_loss_list = []
                mse_loss_list = []

            if i % 2000 == 0:
                val_sce, val_mse = validate(model, eval_loader, device)
                if writer is not None:
                    writer.add_scalar("valid/sce_loss", val_sce, global_step=global_step)
                    writer.add_scalar("valid/mse_loss", val_mse, global_step=global_step)

                if args.save_model:
                    if not os.path.exists(f"checkpoints/{args.task_name}"):
                        os.makedirs(f"checkpoints/{args.task_name}")
                    torch.save(model.state_dict(), f"checkpoints/{args.task_name}/checkpoint_step_{global_step}.pt")

        if scheduler is not None:
            scheduler.step()

    return model


def load_and_lock_modules(model, state_dict):
    """Load and freeze the value and variable embeddings from a checkpoint."""
    module_mapping = {
        'cve_value': 'cve_value',
        'variable_emb': 'variable_emb'
    }

    print(f"Loading and freezing modules: {list(module_mapping.values())}")

    for dict_prefix, model_attr_name in module_mapping.items():
        if not hasattr(model, model_attr_name):
            print(f"Warning: model has no module named {model_attr_name!r}; skipping.")
            continue

        sub_module = getattr(model, model_attr_name)

        # Submodules expect checkpoint keys relative to their own namespace.
        sub_state_dict = {}
        prefix_len = len(dict_prefix)

        for k, v in state_dict.items():
            if k.startswith(dict_prefix):
                new_key = k[prefix_len:]
                if new_key.startswith('.'):
                    new_key = new_key[1:]
                sub_state_dict[new_key] = v

        if not sub_state_dict:
            print(f"Warning: checkpoint has no parameters for {dict_prefix!r}; skipping.")
            continue

        try:
            sub_module.load_state_dict(sub_state_dict, strict=True)
            print(f"Successfully loaded: {model_attr_name} (from {dict_prefix})")
        except Exception as e:
            print(f"Error loading {model_attr_name}: {e}")
            continue

        for param in sub_module.parameters():
            param.requires_grad = False


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    max_epoch = args.max_epoch

    lr = args.lr
    weight_decay = args.weight_decay
    load_model = args.load_model
    save_model = args.save_model
    use_scheduler = args.scheduler
    batch_size = args.batch_size

    writer = SummaryWriter(log_dir=f"logs/{args.task_name}")

    data_root = "data/mimic3"
    pkl_path = os.path.join(data_root, "mimic_iii.pkl")
    var2ind_path = os.path.join(data_root, "var2ind.json")
    mean_std_path = os.path.join(data_root, "var_mean_std.json")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            _, _, train_ids, valid_ids, _ = pickle.load(f)
    else:
        raise FileNotFoundError(f"Data split file not found at {pkl_path}")

    train_dataset = ICULargeGraphDataset(
        root=data_root,
        ts_id=train_ids,
        var2ind_path=var2ind_path,
        var_mean_std_path=mean_std_path,
        pkl_path=pkl_path,
    )

    val_dataset = ICULargeGraphDataset(
        root=data_root,
        ts_id=valid_ids,
        var2ind_path=var2ind_path,
        var_mean_std_path=mean_std_path,
        pkl_path=pkl_path,
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Valid Dataset Size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # The model uses num_features as the input embedding dimension.
    if not hasattr(args, 'num_features') or args.num_features is None:
        args.num_features = 64

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = build_model(args)
        model.to(device)
        print(model)

        optimizer = create_optimizer(args.optimizer, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use scheduler")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if load_model and args.load_ckpt:
            logging.info(f"Loading Model from {args.load_ckpt} ... ")
            model.load_state_dict(torch.load(f"checkpoints/{args.task_name}/{args.load_ckpt}"))

        if args.load_pre_embedding:
            st = torch.load(f"checkpoints/{args.load_pre_embedding}")
            load_and_lock_modules(model, st)
            print(st)

        model = pretrain(
            model=model,
            dataloaders=(train_loader, val_loader),
            optimizer=optimizer,
            max_epoch=max_epoch,
            device=device,
            scheduler=scheduler,
            args=args,
            writer=writer
        )

        if save_model:
            logging.info("Saving Final Model ...")
            if not os.path.exists(f"checkpoints/{args.task_name}"):
                os.makedirs(f"checkpoints/{args.task_name}")
            torch.save(model.state_dict(), f"checkpoints/{args.task_name}/checkpoint_final_seed{seed}.pt")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
