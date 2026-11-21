import torch
import pandas as pd
import numpy as np
import json
import pickle
import os
import shutil
from torch_geometric.data import Data, Dataset, Batch
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch


def generate_subclass_to_class_map(json_path):
    """
    Args:
        json_path (str): var_mean_std.json

    Returns:
        subclass_to_class_map (list)
        num_classes (int)
        num_sub_classes (int)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        var_meta = json.load(f)

    # Keep the class ordering consistent with the dataset constructor.
    all_classes = sorted(list(set(item['class'] for item in var_meta.values())))
    all_sub_classes = sorted(list(set(item['sub_class'] for item in var_meta.values())))

    class2ind = {c: i for i, c in enumerate(all_classes)}
    subclass2ind = {sc: i for i, sc in enumerate(all_sub_classes)}

    subclass_to_class_map = [0] * len(all_sub_classes)

    for meta in var_meta.values():
        sub_id = subclass2ind[meta['sub_class']]
        cls_id = class2ind[meta['class']]
        subclass_to_class_map[sub_id] = cls_id

    return subclass_to_class_map, len(all_classes), len(all_sub_classes)


class ICUGraphData(Data):
    """Patient graph with a fixed semantic-type batching stride.

    PyG's default rule for attributes containing ``"batch"`` uses the largest
    value present in each graph as the increment.  ``batch_type`` instead has a
    dataset-wide semantic vocabulary, so every patient must advance by the full
    vocabulary size even when some types are absent from that patient.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "batch_type":
            if not hasattr(self, "num_type_classes"):
                raise AttributeError(
                    "ICUGraphData requires num_type_classes to batch batch_type"
                )
            count = self.num_type_classes
            if isinstance(count, torch.Tensor):
                count = count.view(-1)[0].item()
            count = int(count)
            if count < 1:
                raise ValueError("num_type_classes must be positive")
            return count
        return super().__inc__(key, value, *args, **kwargs)


class ICULargeGraphDataset(Dataset):
    def __init__(self, root, ts_id, var2ind_path, var_mean_std_path, max_nodes=2000, transform=None,
                 pre_transform=None, time_limit=None, stage="train", pkl_path=None, time_slot=60,
                 *, filter_before_windowing=False):

        self.tsid_list = [int(value) for value in ts_id]
        self.tsid_set = set(self.tsid_list)
        self.max_nodes = max_nodes
        self.time_slot = time_slot
        self.time_limit = time_limit
        self.window_cutoff_minutes = (
            time_limit * 60 if filter_before_windowing and time_limit is not None else None
        )
        self.stage = stage
        self.pkl_path = self._resolve_pkl_path(root, pkl_path)
        self.events = None
        self.label_df = None
        self.label_by_tsid = {}
        self._labels = None

        with open(var2ind_path, 'r') as f:
            self.var2ind = json.load(f)

        with open(var_mean_std_path, 'r') as f:
            self.var_meta = json.load(f)

        all_classes = sorted(list(set(item['class'] for item in self.var_meta.values())))
        self.class2ind = {c: i for i, c in enumerate(all_classes)}
        self.num_type_classes = len(self.class2ind)
        self.variable_to_class = torch.full(
            (len(self.var2ind),), -1, dtype=torch.long
        )
        for variable, variable_index in self.var2ind.items():
            if variable not in self.var_meta:
                raise KeyError(f"{variable!r} is missing from var_mean_std metadata")
            self.variable_to_class[int(variable_index)] = self.class2ind[
                self.var_meta[variable]['class']
            ]
        if bool((self.variable_to_class < 0).any()):
            missing_indices = torch.nonzero(
                self.variable_to_class < 0, as_tuple=False
            ).view(-1).tolist()
            raise ValueError(
                f"Variable indices are not contiguous; unmapped indices: {missing_indices[:10]}"
            )

        self.all_processed_files = []

        super().__init__(root, transform, pre_transform)

        self._load_processed_index()
        present_ids = {
            parsed[0]
            for parsed in (self._parse_graph_filename(name) for name in self.all_processed_files)
            if parsed is not None
        }
        missing_ids = self.tsid_set - present_ids
        if missing_ids:
            print(f"Building graphs for {len(missing_ids)} ICU stays missing from the cache...")
            self._process_ids(missing_ids, append=True)
            self._load_processed_index()

        self.files_by_tsid = {}
        for file_name in self.all_processed_files:
            parsed = self._parse_graph_filename(file_name)
            if parsed is None:
                continue
            stay_id, slice_id = parsed
            if stay_id in self.tsid_set:
                self.files_by_tsid.setdefault(stay_id, []).append((slice_id, file_name))
        for stay_id in self.files_by_tsid:
            self.files_by_tsid[stay_id].sort(key=lambda item: item[0])

        missing_after_processing = self.tsid_set - set(self.files_by_tsid)
        if missing_after_processing:
            preview = sorted(missing_after_processing)[:10]
            raise RuntimeError(
                f"No graph was produced for {len(missing_after_processing)} requested ICU stays; "
                f"first IDs: {preview}"
            )

        if self.time_limit is not None:
            # One dataset item must remain one ICU stay.  All cached slices are
            # merged first and the time mask is then applied to the merged graph.
            self.active_stay_ids = list(dict.fromkeys(self.tsid_list))
            self.active_files = [
                file_name
                for stay_id in self.active_stay_ids
                for _, file_name in self.files_by_tsid[stay_id]
            ]
            print(
                f"Time Limit Mode ({self.time_limit}h): "
                f"{len(self.active_stay_ids)} ICU stays from {len(self.active_files)} cached slices."
            )
        else:
            self.active_files = [
                file_name
                for stay_id in dict.fromkeys(self.tsid_list)
                for _, file_name in self.files_by_tsid[stay_id]
            ]
            print(f"Full Mode: Using all slices. Total: {len(self.active_files)}")

        print(f"Dataset Loaded. Active Items: {self.len()}")

    @staticmethod
    def _resolve_pkl_path(root, pkl_path):
        if pkl_path is not None:
            resolved = os.fspath(pkl_path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"Dataset pickle not found: {resolved}")
            return resolved

        candidates = [
            os.path.join(os.fspath(root), "mimic_iii.pkl"),
            os.path.join(os.fspath(root), "mimic_iv.pkl"),
            os.path.join(os.fspath(root), "eicu.pkl"),
        ]
        existing = [path for path in candidates if os.path.exists(path)]
        if len(existing) != 1:
            raise ValueError(
                "pkl_path was not supplied and could not be inferred uniquely; "
                f"candidates found: {existing}"
            )
        return existing[0]

    @staticmethod
    def _parse_graph_filename(file_name):
        stem, extension = os.path.splitext(os.path.basename(file_name))
        parts = stem.split("_")
        if extension != ".pt" or len(parts) != 3 or parts[0] != "graph":
            return None
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            return None

    @property
    def processed_dir(self):
        if self.window_cutoff_minutes is None:
            return super().processed_dir
        return os.path.join(
            self.root, f"processed_lt_{self.window_cutoff_minutes}_slot_{self.time_slot}_v1"
        )

    @property
    def index_path(self):
        return self.processed_paths[0]

    def _load_processed_index(self):
        if not os.path.exists(self.index_path):
            self.all_processed_files = []
            return
        with open(self.index_path, "r") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("files", [])
        if not isinstance(payload, list):
            raise ValueError(f"Invalid processed graph index: {self.index_path}")
        self.all_processed_files = list(dict.fromkeys(payload))

    def _discover_processed_graphs(self):
        """Recover graph files left by an interrupted preprocessing run."""
        if not os.path.isdir(self.processed_dir):
            return []
        graph_files = []
        with os.scandir(self.processed_dir) as entries:
            for entry in entries:
                if not entry.is_file() or self._parse_graph_filename(entry.name) is None:
                    continue
                # A zero-byte file can be left behind by an abrupt termination
                # of an older, non-atomic torch.save call and must not count as
                # a completed ICU stay.
                if entry.stat().st_size > 0:
                    graph_files.append(entry.name)
        graph_files.sort(
            key=lambda name: self._parse_graph_filename(name) or (10**18, 10**18)
        )
        return graph_files

    def _write_processed_index(self, file_names):
        file_names = list(dict.fromkeys(file_names))
        file_names.sort(
            key=lambda name: self._parse_graph_filename(name) or (10**18, 10**18)
        )
        temporary_index = self.index_path + ".tmp"
        with open(temporary_index, "w") as handle:
            json.dump(file_names, handle)
        os.replace(temporary_index, self.index_path)
        self.processed_files = file_names
        self.all_processed_files = file_names

    @staticmethod
    def _completed_stay_ids(file_names):
        # Slice 0 is committed last by _save_stay_graphs.  Its presence is the
        # completion marker for a stay if preprocessing is interrupted.
        return {
            stay_id
            for parsed in (
                ICULargeGraphDataset._parse_graph_filename(name)
                for name in file_names
            )
            if parsed is not None
            for stay_id, slice_id in [parsed]
            if slice_id == 0
        }

    def _load_source(self, stay_ids):
        with open(self.pkl_path, "rb") as handle:
            events, label_df, _, _, _ = pickle.load(handle)
        stay_ids = set(map(int, stay_ids))
        selected = events["ts_id"].isin(stay_ids)
        if self.window_cutoff_minutes is not None:
            selected &= events["minute"].lt(self.window_cutoff_minutes)
        self.events = events.loc[selected].copy()
        if isinstance(label_df, dict):
            self.label_df = None
            self.label_by_tsid = {
                int(stay_id): label
                for stay_id, label in label_df.items()
                if int(stay_id) in stay_ids
            }
        else:
            required_columns = {"ts_id", "in_hospital_mortality"}
            if not required_columns.issubset(label_df.columns):
                raise ValueError(
                    "Outcome table must contain ts_id and in_hospital_mortality"
                )
            self.label_df = label_df[label_df["ts_id"].isin(stay_ids)].copy()
            self.label_by_tsid = dict(
                zip(
                    self.label_df["ts_id"].astype(np.int64),
                    self.label_df["in_hospital_mortality"],
                )
            )
        del events, label_df

    def _clear_source(self):
        self.events = None
        self.label_df = None
        self.label_by_tsid = {}

    def _process_ids(self, stay_ids, append):
        stay_ids = set(map(int, stay_ids))
        if not stay_ids:
            return
        self._load_source(stay_ids)
        try:
            existing = self.all_processed_files if append else []
            self.time_slot_segmentation(
                self.time_slot, stay_ids=stay_ids, existing_files=existing
            )
        finally:
            self._clear_source()

    def extract_demographics(self, single_event, extract_vars_list):
        df = single_event
        demographic_mask = df['variable'].isin(extract_vars_list)
        filtered_df = df.loc[demographic_mask].copy()
        df.drop(df.index[demographic_mask], inplace=True)
        ordered_rows = []
        for variable in extract_vars_list:
            rows = filtered_df[filtered_df["variable"].eq(variable)]
            if rows.empty:
                raise ValueError(f"Demographic variable {variable!r} is missing")
            ordered_rows.append(rows.iloc[[0]])
        return pd.concat(ordered_rows, ignore_index=True)

    def get_labels(self):
        if self.time_limit is not None:
            file_names = [
                self.files_by_tsid[stay_id][0][1] for stay_id in self.active_stay_ids
            ]
        else:
            file_names = self.active_files
        return [
            float(
                torch.load(
                    os.path.join(self.processed_dir, file_name), weights_only=False
                ).y.view(-1)[0]
            )
            for file_name in file_names
        ]

    @property
    def labels(self):
        """Labels aligned one-to-one with the active dataset items."""
        if self._labels is None:
            self._labels = self.get_labels()
        return self._labels


    def time_slot_segmentation(
        self, time_slot, min_points=8, stay_ids=None, existing_files=None
    ):
        df = self.events
        if stay_ids is not None:
            df = df[df["ts_id"].isin(set(stay_ids))]
        processed_files_list = list(existing_files or [])
        for ts_id, group in tqdm(df.groupby('ts_id'),desc="ts_id:"):
            group = group.sort_values(by='minute')
            min_time = group['minute'].min()
            max_time = group['minute'].max()
            time_periods = []
            start_time = min_time
            demo_df = self.extract_demographics(group,["Age","Gender"])
            while start_time <= max_time:
                end_time = start_time + time_slot - 1
                period_data = group[(group['minute'] >= start_time) & (group['minute'] <= end_time)]
                if len(period_data) == 0:
                    start_time = end_time + 1
                    continue
                while len(period_data) < min_points:
                    # Extend sparse windows until they contain enough observations.
                    next_start_time = end_time + 1
                    if next_start_time > group['minute'].max():
                        break

                    next_end_time = next_start_time + time_slot - 1
                    next_period_data = group[(group['minute'] >= next_start_time) & (group['minute'] <= next_end_time)]

                    period_data = pd.concat([period_data, next_period_data], ignore_index=True)
                    end_time = next_end_time
                time_periods.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'events': period_data,
                    'time_slot': end_time - start_time + 1
                })
                start_time = end_time + 1
            if len(time_periods) > 1 and len(time_periods[-1]['events']) < min_points:
                # Merge a sparse final window into the preceding one.
                last_period = time_periods.pop()
                second_last_period = time_periods[-1]

                second_last_period['events'] = pd.concat([second_last_period['events'], last_period['events']],
                                                         ignore_index=True)
                second_last_period['time_slot'] = last_period['start_time'] + last_period['time_slot'] - \
                                                  second_last_period['start_time']
            if int(ts_id) not in self.label_by_tsid:
                raise KeyError(f"No mortality label found for ICU stay {ts_id}")
            y_label = self.label_by_tsid[int(ts_id)]
            self.process_slice_events(processed_files_list, ts_id, time_periods, y_label, demo_df)

        self._write_processed_index(processed_files_list)
        print(f"Processing complete. Index saved to {self.index_path}")

        # result.set_index('ts_id', drop=False)
        return


    def trans_tsid2raw_file(self, tsid_list):
        raw_file_list = [f"timeseries_{int(tsid)}.pkl" for tsid in tsid_list]
        return raw_file_list

    @property
    def processed_file_names(self):
        return ["processed_files.json"]

    def _events_to_data(self, full_df, ts_id, y_label, demo, slice_idx, total_time_slots):
        # Do all mappings through ordinary numeric arrays.  In the eICU pickle
        # ``variable`` is categorical; assigning mapped floats back into that
        # dtype makes pandas attempt an unsafe category -> int cast.
        variable_names = full_df['variable'].astype('string')
        var_ids = variable_names.map(self.var2ind)
        unknown_variables = sorted(variable_names[var_ids.isna()].dropna().unique().tolist())
        if unknown_variables:
            raise KeyError(
                f"Variables are missing from var2ind.json: {unknown_variables[:10]}"
            )

        class_by_variable = {
            variable: metadata['class'] for variable, metadata in self.var_meta.items()
        }
        mean_by_variable = {
            variable: metadata['mean'] for variable, metadata in self.var_meta.items()
        }
        std_by_variable = {
            variable: metadata['std'] for variable, metadata in self.var_meta.items()
        }
        class_names = variable_names.map(class_by_variable)
        class_ids = class_names.map(self.class2ind)

        values = pd.to_numeric(full_df['value'], errors='coerce').to_numpy(dtype=np.float32)
        means = pd.to_numeric(variable_names.map(mean_by_variable), errors='coerce').to_numpy(
            dtype=np.float32
        )
        stds = pd.to_numeric(variable_names.map(std_by_variable), errors='coerce').to_numpy(
            dtype=np.float32
        )
        minutes = pd.to_numeric(full_df['minute'], errors='coerce').to_numpy(dtype=np.float32)
        time_slot_ids = pd.to_numeric(
            full_df['time_slot_idx'], errors='coerce'
        ).to_numpy(dtype=np.float64)

        invalid_mapping = var_ids.isna() | class_ids.isna()
        invalid_numeric = ~(
            np.isfinite(values)
            & np.isfinite(means)
            & np.isfinite(stds)
            & np.isfinite(minutes)
            & np.isfinite(time_slot_ids)
        )
        if invalid_mapping.any() or invalid_numeric.any():
            raise ValueError(
                f"ICU stay {int(ts_id)} contains unmapped or non-numeric graph features"
            )

        stds[(stds == 0) | ~np.isfinite(stds)] = 1.0
        normalized_values = (values - means) / stds
        num_nodes = len(full_df)

        x_var = torch.as_tensor(
            var_ids.to_numpy(dtype=np.int64), dtype=torch.long
        ).unsqueeze(1)
        x_val = torch.as_tensor(normalized_values, dtype=torch.float).unsqueeze(1)
        x_time = torch.as_tensor(minutes, dtype=torch.float).unsqueeze(1)
        x_raw = torch.as_tensor(values, dtype=torch.float).unsqueeze(1)

        batch_time = torch.as_tensor(time_slot_ids.astype(np.int64), dtype=torch.long)
        batch_type = torch.as_tensor(
            class_ids.to_numpy(dtype=np.int64), dtype=torch.long
        )

        y = torch.tensor([y_label], dtype=torch.float)

        data = ICUGraphData(
            x_var=x_var,
            x_val=x_val,
            x_time=x_time,
            x_raw=x_raw,
            y=y,
            batch_time=batch_time,
            batch_type=batch_type,
            demo=demo,
            ts_id=ts_id,
            slice_id=slice_idx,
            num_nodes=num_nodes,
            num_time_steps=total_time_slots,
            num_type_classes=self.num_type_classes,
        )
        return data
    def process_slice_events(self, processed_files_list, ts_id, time_periods, y_label, demo_df):
        if not demo_df.empty:
            age_rows = demo_df[demo_df["variable"] == "Age"]
            if not age_rows.empty:
                demo_df.loc[demo_df["variable"] == "Age", "value"] = (age_rows["value"] - self.var_meta["Age"]["mean"]) / self.var_meta["Age"]["std"]
            demo = torch.tensor(demo_df["value"].to_list(), dtype=torch.float)
        else:
            demo = torch.tensor([], dtype=torch.float)

        current_slice_events = []
        current_slice_nodes = 0
        slice_idx = 0

        valid_periods = []

        for t_idx, period in enumerate(time_periods):
            events = period['events']
            if events.empty:
                continue

            events = events[events['variable'].isin(self.var2ind.keys())]
            if events.empty:
                continue

            # eICU stores ``variable`` and ``TABLE`` as categoricals.  Repeated
            # concatenation of groupby slices can trigger a pandas extension-
            # block bug (``object has no attribute size``).  Rebuild only the
            # four graph columns from plain NumPy arrays before concatenating.
            events = pd.DataFrame({
                'variable': events['variable'].astype(object).to_numpy(copy=True),
                'value': pd.to_numeric(events['value'], errors='coerce').to_numpy(
                    dtype=np.float32, copy=True
                ),
                'minute': pd.to_numeric(events['minute'], errors='coerce').to_numpy(
                    dtype=np.float32, copy=True
                ),
                'time_slot_idx': np.full(len(events), t_idx, dtype=np.int64),
            })
            valid_periods.append(events)

        if not valid_periods:
            return

        stay_graphs = []
        for events in valid_periods:
            num_nodes_in_slot = len(events)

            if current_slice_events and (current_slice_nodes + num_nodes_in_slot > self.max_nodes):
                full_df = pd.concat(current_slice_events, ignore_index=True)
                data = self._events_to_data(full_df, ts_id, y_label, demo, slice_idx, len(time_periods))

                file_name = f'graph_{int(ts_id)}_{slice_idx}.pt'
                stay_graphs.append((slice_idx, file_name, data))

                slice_idx += 1
                current_slice_events = []
                current_slice_nodes = 0

            current_slice_events.append(events)
            current_slice_nodes += num_nodes_in_slot

        if current_slice_events:
            full_df = pd.concat(current_slice_events, ignore_index=True)
            data = self._events_to_data(full_df, ts_id, y_label, demo, slice_idx, len(time_periods))

            file_name = f'graph_{int(ts_id)}_{slice_idx}.pt'
            stay_graphs.append((slice_idx, file_name, data))

        self._save_stay_graphs(stay_graphs, processed_files_list)

    def _save_stay_graphs(self, stay_graphs, processed_files_list):
        """Atomically commit every slice, with slice 0 acting as completion marker."""
        if not stay_graphs:
            return

        temporary_paths = []
        try:
            for _, file_name, data in stay_graphs:
                final_path = os.path.join(self.processed_dir, file_name)
                temporary_path = final_path + ".tmp"
                torch.save(data, temporary_path)
                temporary_paths.append((file_name, temporary_path, final_path))

            # Higher slices are made visible first.  Slice 0 is renamed last,
            # so an interrupted commit is not mistaken for a completed stay.
            for file_name, temporary_path, final_path in reversed(temporary_paths):
                os.replace(temporary_path, final_path)
                processed_files_list.append(file_name)
        finally:
            for _, temporary_path, _ in temporary_paths:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)




    def process(self):
        recovered_files = self._discover_processed_graphs()
        self.all_processed_files = recovered_files
        completed_ids = self._completed_stay_ids(recovered_files)
        missing_ids = self.tsid_set - completed_ids

        if recovered_files:
            recovered_requested = len(self.tsid_set & completed_ids)
            print(
                f"Recovered {len(recovered_files)} graph files for "
                f"{recovered_requested} requested ICU stays; "
                f"{len(missing_ids)} stays remain."
            )

        if missing_ids:
            self._process_ids(missing_ids, append=True)
        else:
            self._write_processed_index(recovered_files)

    def len(self):
        if self.time_limit is not None:
            return len(self.active_stay_ids)
        return len(self.active_files)

    def get(self, idx):
        if self.time_limit is None:
            file_name = self.active_files[idx]
            processed_path = os.path.join(self.processed_dir, file_name)
            if not os.path.exists(processed_path):
                raise FileNotFoundError(f"Processed graph not found: {processed_path}")
            return self._as_icu_graph_data(
                torch.load(processed_path, weights_only=False)
            )

        stay_id = self.active_stay_ids[idx]
        slices = []
        for _, file_name in self.files_by_tsid[stay_id]:
            processed_path = os.path.join(self.processed_dir, file_name)
            if not os.path.exists(processed_path):
                raise FileNotFoundError(f"Processed graph not found: {processed_path}")
            slices.append(
                self._as_icu_graph_data(
                    torch.load(processed_path, weights_only=False)
                )
            )
        data = self._merge_slices(slices)

        limit_minutes = self.time_limit * 60.0
        mask = data.x_time.view(-1) < limit_minutes

        if not bool(mask.any()):
            return self._get_empty_data_with_meta(data)

        new_data = ICUGraphData(
            x_var = data.x_var[mask],
            x_val = data.x_val[mask],
            x_time = data.x_time[mask],
            x_raw = data.x_raw[mask],
            y = data.y,
            batch_time = data.batch_time[mask],
            batch_type = data.batch_type[mask],
            demo = data.demo,
            ts_id = data.ts_id,
            slice_id = -1,
            num_nodes = mask.sum().item(),
            num_time_steps = data.num_time_steps,
            num_type_classes = self.num_type_classes,
        )
        return new_data

    def _as_icu_graph_data(self, data):
        """Upgrade caches and apply the currently selected semantic taxonomy."""
        if isinstance(data, ICUGraphData):
            upgraded = data
        else:
            upgraded = ICUGraphData(**data.to_dict())
            upgraded.num_nodes = int(data.num_nodes)

        variable_ids = upgraded.x_var.view(-1).long()
        invalid = (variable_ids < 0) | (variable_ids >= len(self.variable_to_class))
        if bool(invalid.any()):
            invalid_ids = torch.unique(variable_ids[invalid]).tolist()
            raise ValueError(
                "Cached graph contains variable IDs outside the current vocabulary: "
                f"{invalid_ids}"
            )
        # ``batch_type`` is derived rather than trusted because existing graph
        # caches may have been built with an older 5-class taxonomy.  Rebuilding
        # it in memory makes taxonomy changes safe without rewriting large caches.
        upgraded.batch_type = self.variable_to_class[variable_ids]
        upgraded.num_type_classes = self.num_type_classes
        return upgraded

    def _merge_slices(self, slices):
        if not slices:
            raise ValueError("Cannot merge an empty graph-slice list")
        if len(slices) == 1:
            return self._as_icu_graph_data(slices[0])

        node_fields = ('x_var', 'x_val', 'x_time', 'x_raw', 'batch_time', 'batch_type')
        missing_fields = [
            field for field in node_fields if any(not hasattr(data, field) for data in slices)
        ]
        if missing_fields:
            raise ValueError(f"Cached graph slices are missing fields: {missing_fields}")

        return ICUGraphData(
            **{
                field: torch.cat([getattr(data, field) for data in slices], dim=0)
                for field in node_fields
            },
            y=slices[0].y,
            demo=slices[0].demo,
            ts_id=slices[0].ts_id,
            slice_id=-1,
            num_nodes=sum(int(data.num_nodes) for data in slices),
            num_time_steps=max(int(data.num_time_steps) for data in slices),
            num_type_classes=self.num_type_classes,
        )

    def _get_empty_data_with_meta(self, original_data):
        return ICUGraphData(
            x_var=torch.empty((0, 1), dtype=torch.long),
            x_val=torch.empty((0, 1), dtype=torch.float),
            x_time=torch.empty((0, 1), dtype=torch.float),
            x_raw=torch.empty((0, 1), dtype=torch.float),
            batch_time=torch.empty(0, dtype=torch.long),
            batch_type=torch.empty(0, dtype=torch.long),
            num_nodes=0,
            y=original_data.y,
            demo=original_data.demo,
            ts_id=original_data.ts_id,
            slice_id=-1,
            num_time_steps=original_data.num_time_steps,
            num_type_classes=self.num_type_classes,
        )

if __name__=="__main__":
    # pkl_pth = "data/mimic3/mimic_iii.pkl"
    data_root = "data/eicu"
    pkl_path = os.path.join(data_root, "eicu.pkl")
    var2ind_path = os.path.join(data_root, "var2ind.json")
    mean_std_path = os.path.join(data_root, "var_mean_std.json")
    with open(pkl_path, "rb") as f:
        data, label, train_list, val_list, test_list = pickle.load(f)
    del data, label
    total_list =np.hstack([train_list, val_list, test_list])
    ds = ICULargeGraphDataset(
        root=data_root,
        ts_id=total_list,
        var2ind_path=var2ind_path,
        var_mean_std_path=mean_std_path,
        pkl_path=pkl_path,
    )
    print(ds[0].x_time)
