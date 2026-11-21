# MVE-GNN

MVE-GNN learns a graph encoder from irregular EHR measurements and fine-tunes it for in-hospital mortality prediction. The downstream model combines semantic, temporal, and structural views.

## Setup

Run the commands from the repository root. Use Python 3.10, PyTorch 2.6, and PyTorch Geometric 2.6.1. Install `torch-cluster`, `torch-scatter`, and `torch-sparse` builds that match your PyTorch and CPU/CUDA environment. Install the remaining dependencies with:

```bash
python -m pip install numpy pandas scipy scikit-learn tensorboard tqdm xgboost joblib transformers==4.52.4 dill pathos psutil
```

## Data and checkpoints

Obtain dataset access through PhysioNet for [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), or [eICU](https://physionet.org/content/eicu-crd/2.0/).

Place the preprocessed cohort pickle and its `var2ind.json` and `var_mean_std.json` in the matching dataset directory. The `preprocess/` directory contains the data preparation code.

| Dataset | Cohort pickle | Encoder checkpoint to use |
| --- | --- | --- |
| MIMIC-III | `data/mimic3/mimic_iii.pkl` | `checkpoints/mm3_pretrained.pt` |
| MIMIC-IV | `data/mimic-iv/mimic_iv.pkl` | `checkpoints/mm4_pretrained.pt` |
| eICU | `data/eicu/eicu.pkl` | `checkpoints/mm4_pretrained.pt` |

eICU reuses the MIMIC-IV encoder weights. When running the downstream commands with `--dataset eicu`, pass `--pretrained-checkpoint checkpoints/mm4_pretrained.pt` to load the shared file directly.

Each pickle contains `(events, outcomes, train_ids, val_ids, test_ids)`. The events table needs `ts_id`, `variable`, `minute`, and `value`, including `Age` and `Gender` rows. The outcomes table needs `ts_id` and `in_hospital_mortality`. The MIMIC baseline loader additionally uses a patient identifier such as `SUBJECT_ID`. Map measurement names to contiguous integer IDs in `var2ind.json`. Include `mean`, `std`, and `class` for the measurements and demographics in `var_mean_std.json`. Use patient-disjoint development and test sets and compute normalization statistics on the training partition. The graph loaders build missing caches under `processed/`.

Step 1 produces the MIMIC-III checkpoint. If you already have the compatible checkpoint for your dataset, place it at the path above and start at step 2.

## 1. Pretrain the MIMIC-III encoder

```bash
python pretrain_mimic3_large.py --device 0 --task_name mimic3_pretraining --save_model
python -c "from shutil import copy2; copy2('checkpoints/mimic3_pretraining/checkpoint_final_seed0.pt', 'checkpoints/mm3_pretrained.pt')"
```

Use `--device -1` for CPU pretraining. The encoder configuration and reconstruction weights are in `settings/pretrain_model_params.json`.

## 2. Fine-tune teachers and distill students

Train five teachers per fold and per view. Then distill one student per view from the mean teacher logits. The commands share the same development folds, fixed test set, and encoder checkpoint.

```bash
python finetune_kfold.py --dataset mimic3 --pool_mode type --device 0
python finetune_kfold.py --dataset mimic3 --pool_mode time --device 0
python finetune_kfold.py --dataset mimic3 --pool_mode diff --device 0

python finetune_kfold_ensemble_distillation.py --dataset mimic3 --pool_mode type --device 0
python finetune_kfold_ensemble_distillation.py --dataset mimic3 --pool_mode time --device 0
python finetune_kfold_ensemble_distillation.py --dataset mimic3 --pool_mode diff --device 0

python ensemble_kfold_distilled.py --dataset mimic3 --views type time diff --device 0
```

The final command averages the three student probabilities and writes fold metrics to `results/ensemble_kfold_distilled/default/` for MIMIC-III. For a shorter run, add `--max_folds 1 --num_workers 0` to each downstream command and `--max_epoch 1` to each training command.

The MIMIC-IV and eICU paths use the MIMIC-III vocabulary size for the embedding table, so provide `data/mimic3/var2ind.json` as well. The eICU transfer also uses `data/mimic-iv/var2ind.json` as the source vocabulary.

## 3. Run the mortality baselines

```bash
python run_all_mortality_baselines.py --dataset mimic3 --device 0
```

This launcher trains each baseline on the same six mortality folds. Supply the relevant SAPS-II component table as `mm3_sapsii.csv`, `mm4_sapsii.csv`, or `eicu_sapsii.csv` in the dataset directory. The eICU baseline loader uses `data/eicu/patient_split.csv` with `ts_id` and `patient_id` columns. For the full MIMIC-III run, supply the iSTraTS weights at `checkpoints/finetune_strats_mm3_extend/checkpoint_best.bin` or pass `--strats-pretrained-checkpoint`. GCT and GRAM read the graph cache built during MVE-GNN fine-tuning.

Add `--dry-run` to print the full set of baseline commands, or use `--device cpu --only lr gru --max-folds 1 --epochs 1` for a small training run.
