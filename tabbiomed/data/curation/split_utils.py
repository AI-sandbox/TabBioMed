from .paths import CONFIG_DIR
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def load_split_config(dataset_name):
    return json.loads((CONFIG_DIR / "split_settings.json").read_text()).get(
        dataset_name
    )


def create_group_stratified_split(df, split_config):
    group_by = split_config.get("group_by", None)
    stratify_by = split_config.get("stratify_by", None)
    train_size = split_config.get("train_size", 0.7)
    val_size = split_config.get("val_size", 0.15)
    test_size = split_config.get("test_size", 0.15)
    random_state = split_config.get("random_state", 42)
    if abs(train_size + val_size + test_size - 1.0) > 1e-06:
        raise ValueError(
            f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
        )
    if group_by is not None:
        if group_by not in df.columns:
            raise ValueError(f"Group column '{group_by}' not found in DataFrame")
        if df[group_by].isna().any():
            raise ValueError(f"Group column '{group_by}' contains missing identifiers")
    if group_by is not None and stratify_by is not None:
        if stratify_by not in df.columns:
            raise ValueError(f"Stratify column '{stratify_by}' not found in DataFrame")
        group_df = df.groupby(group_by)[stratify_by].max().reset_index()
        group_df.columns = [group_by, stratify_by]
        print(f"Total groups: {len(group_df)}")
        if stratify_by:
            print(f"Groups with positive class: {group_df[stratify_by].sum()}")
        temp_size = val_size + test_size
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=random_state
        )
        for train_idx, temp_idx in splitter1.split(group_df, group_df[stratify_by]):
            group_train = group_df.iloc[train_idx]
            group_temp = group_df.iloc[temp_idx]
        test_ratio = test_size / temp_size
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        for val_idx, test_idx in splitter2.split(group_temp, group_temp[stratify_by]):
            group_val = group_temp.iloc[val_idx]
            group_test = group_temp.iloc[test_idx]
        train_group_ids = set(group_train[group_by].values)
        val_group_ids = set(group_val[group_by].values)
        test_group_ids = set(group_test[group_by].values)
        train_mask = df[group_by].isin(train_group_ids)
        val_mask = df[group_by].isin(val_group_ids)
        test_mask = df[group_by].isin(test_group_ids)
        train_indices = np.where(train_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
        print(f"\nSplit Statistics:")
        print(f"Train - Groups: {len(train_group_ids)}, Rows: {len(train_indices)}")
        if stratify_by:
            print(f"         Positive class: {train_df[stratify_by].sum()}")
        print(f"Val   - Groups: {len(val_group_ids)}, Rows: {len(val_indices)}")
        if stratify_by:
            print(f"         Positive class: {val_df[stratify_by].sum()}")
        print(f"Test  - Groups: {len(test_group_ids)}, Rows: {len(test_indices)}")
        if stratify_by:
            print(f"         Positive class: {test_df[stratify_by].sum()}")
    elif group_by is None and stratify_by is not None:
        if stratify_by not in df.columns:
            raise ValueError(f"Stratify column '{stratify_by}' not found in DataFrame")
        temp_size = val_size + test_size
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=random_state
        )
        for train_idx, temp_idx in splitter1.split(df, df[stratify_by]):
            train_indices = train_idx.tolist()
            temp_df = df.iloc[temp_idx]
            temp_indices = temp_idx
        test_ratio = test_size / temp_size
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        for val_idx, test_idx in splitter2.split(temp_df, temp_df[stratify_by]):
            val_indices = temp_indices[val_idx].tolist()
            test_indices = temp_indices[test_idx].tolist()
    elif group_by is not None and stratify_by is None:
        group_df = df.groupby(group_by).first().reset_index()
        temp_size = val_size + test_size
        splitter1 = ShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=random_state
        )
        for train_idx, temp_idx in splitter1.split(group_df):
            group_train = group_df.iloc[train_idx]
            group_temp = group_df.iloc[temp_idx]
        test_ratio = test_size / temp_size
        splitter2 = ShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        for val_idx, test_idx in splitter2.split(group_temp):
            group_val = group_temp.iloc[val_idx]
            group_test = group_temp.iloc[test_idx]
        train_group_ids = set(group_train[group_by].values)
        val_group_ids = set(group_val[group_by].values)
        test_group_ids = set(group_test[group_by].values)
        train_mask = df[group_by].isin(train_group_ids)
        val_mask = df[group_by].isin(val_group_ids)
        test_mask = df[group_by].isin(test_group_ids)
        train_indices = np.where(train_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()
    else:
        temp_size = val_size + test_size
        splitter1 = ShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=random_state
        )
        for train_idx, temp_idx in splitter1.split(df):
            train_indices = train_idx.tolist()
            temp_indices = temp_idx
        test_ratio = test_size / temp_size
        splitter2 = ShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        for val_idx, test_idx in splitter2.split(df.iloc[temp_indices]):
            val_indices = temp_indices[val_idx].tolist()
            test_indices = temp_indices[test_idx].tolist()
    split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}
    assigned = np.concatenate(list(split_indices.values()))
    if not np.array_equal(np.sort(assigned), np.arange(len(df))):
        raise ValueError("Every input row must belong to exactly one split")
    return split_indices
