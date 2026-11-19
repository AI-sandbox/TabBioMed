from __future__ import annotations
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

INNER_CV_PROTOCOL = "nested_inner_cv_v1"
ZERO_NONZERO = "zero_vs_nonzero"


def zero_nonzero_group_labels(y, groups):
    y, groups = (np.asarray(y), np.asarray(groups))
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("Regression stratification needs one target and group per row")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError(
            "Zero/nonzero stratification requires finite nonnegative targets"
        )
    unique, inverse = np.unique(groups, return_inverse=True)
    values = [np.unique(y[inverse == i] > 0) for i in range(len(unique))]
    if any((len(v) != 1 for v in values)):
        raise ValueError("Each source group must have a single zero/nonzero stratum")
    labels = np.asarray([v[0] for v in values], dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError(
            "Regression stratification requires both zero and nonzero groups"
        )
    return (unique, inverse, labels)


def validate_grouped_outer_cv(splits, groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Sample groups must be a nonempty one-dimensional array")
    folds = splits.get("cv_folds", [])
    if len(folds) < 2:
        raise ValueError("Grouped outer CV requires at least two folds")
    tested = []
    for fold in folds:
        parts = {k: np.asarray(fold[k], dtype=int) for k in ("train", "val", "test")}
        joined = np.concatenate(list(parts.values()))
        if any((not len(p) for p in parts.values())) or not np.array_equal(
            np.sort(joined), np.arange(len(groups))
        ):
            raise ValueError(
                "Each outer fold must partition all rows into nonempty train/val/test roles"
            )
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(groups[parts[a]]) & set(groups[parts[b]]):
                raise ValueError(f"Sample groups cross the outer {a}/{b} boundary")
        tested.extend(parts["test"])
    if not np.array_equal(np.sort(tested), np.arange(len(groups))):
        raise ValueError("Outer test folds must test every sample exactly once")


def _stratified_pool_split(pool, pool_labels, n_val, seed):
    rng = np.random.RandomState(seed)
    classes = np.unique(pool_labels)
    val_selected = []
    train_reserved = []
    remaining_by_class = {}
    for cls in classes:
        cls_groups = pool[pool_labels == cls].copy()
        rng.shuffle(cls_groups)
        val_selected.append(cls_groups[0])
        train_reserved.append(cls_groups[1])
        remaining_by_class[cls] = list(cls_groups[2:])
    extra_val_needed = n_val - len(val_selected)
    if extra_val_needed > 0:
        total_remaining = sum(len(v) for v in remaining_by_class.values())
        quotas = {
            cls: (len(v) * extra_val_needed) / total_remaining
            for cls, v in remaining_by_class.items()
        }
        takes = {
            cls: min(len(remaining_by_class[cls]), int(np.floor(quotas[cls])))
            for cls in classes
        }
        shortfall = extra_val_needed - sum(takes.values())
        order = sorted(
            classes,
            key=lambda c: (
                quotas[c] - takes[c],
                len(remaining_by_class[c]) - takes[c],
            ),
            reverse=True,
        )
        for cls in order:
            if shortfall <= 0:
                break
            if takes[cls] < len(remaining_by_class[cls]):
                takes[cls] += 1
                shortfall -= 1
        for cls in classes:
            k = takes[cls]
            val_selected.extend(remaining_by_class[cls][:k])
            train_reserved.extend(remaining_by_class[cls][k:])
    else:
        for cls in classes:
            train_reserved.extend(remaining_by_class[cls])
    return (
        np.asarray(train_reserved, dtype=pool.dtype),
        np.asarray(val_selected, dtype=pool.dtype),
    )


def make_grouped_outer_cv(
    groups, y, task, *, n_folds=5, seed=42, validation_stratification=None
):
    groups, y = (np.asarray(groups), np.asarray(y))
    if groups.ndim != 1 or y.ndim != 1 or len(groups) != len(y):
        raise ValueError("Groups and target must have one value per row")
    unique, inverse = np.unique(groups, return_inverse=True)
    if type(n_folds) is not int or n_folds < 2 or len(unique) < n_folds:
        raise ValueError("Grouped outer CV needs at least n_folds distinct groups")
    classification = task in ("binary", "classification")
    if validation_stratification not in (None, ZERO_NONZERO):
        raise ValueError("Unknown validation stratification policy")
    if validation_stratification is not None and task != "regression":
        raise ValueError("Zero/nonzero validation stratification is for regression")
    labels = None
    if classification:
        values = [np.unique(y[inverse == i]) for i in range(len(unique))]
        if any((len(v) != 1 for v in values)):
            raise ValueError(
                "Classification source groups must each have a single class"
            )
        labels = np.asarray([v[0] for v in values])
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    elif validation_stratification == ZERO_NONZERO:
        _, _, labels = zero_nonzero_group_labels(y, groups)
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_folds, shuffle=True, random_state=seed)
    folds = []
    for i, (pool, test) in enumerate(splitter.split(unique, labels)):
        n_val = max(1, int(np.ceil(len(pool) / 8)))
        if labels is not None:
            _, counts = np.unique(labels[pool], return_counts=True)
            n_classes = len(np.unique(labels))
            n_val = max(n_val, n_classes)
            if (
                len(counts) != n_classes
                or counts.min() < 2
                or len(pool) - n_val < n_classes
            ):
                raise ValueError(
                    f"Outer fold {i} lacks enough class groups for a disjoint final-fit validation set"
                )
        if n_val >= len(pool):
            raise ValueError(
                "Too few outer non-test groups for separate training and validation"
            )
        split_seed = (int(seed) + (i + 1) * 10007) % 2**31
        if labels is not None:
            train, val = _stratified_pool_split(pool, labels[pool], n_val, split_seed)
        else:
            train, val = train_test_split(
                pool,
                test_size=n_val,
                random_state=split_seed,
            )
        folds.append(
            {
                key: np.flatnonzero(np.isin(inverse, indices)).tolist()
                for key, indices in (("train", train), ("val", val), ("test", test))
            }
        )
    result = {
        "cv_folds": folds,
        "groups": groups.tolist(),
        "group_split_protocol": "grouped_outer_partition_v1",
    }
    if validation_stratification is not None:
        result["validation_stratification"] = validation_stratification
    validate_grouped_outer_cv(result, groups)
    return result


def load_sample_groups(data_dir, splits):
    groups = splits.get("groups")
    if groups is None:
        return None
    groups = np.asarray(groups)
    if "cv_folds" in splits:
        validate_grouped_outer_cv(splits, groups)
    if splits.get("validation_stratification") == ZERO_NONZERO:
        y = np.load(Path(data_dir) / "y.npy")
        zero_nonzero_group_labels(y, groups)
        for fold in splits["cv_folds"]:
            for role in ("train", "val"):
                if set(y[fold[role]] > 0) != {False, True}:
                    raise ValueError(
                        "Train and validation must contain both regression strata"
                    )
    return groups


def make_inner_cv_plan(
    y,
    task,
    outer_split,
    *,
    groups=None,
    requested_folds=3,
    seed=42,
    outer_fold_id=0,
    stratification=None,
):
    if type(requested_folds) is not int or requested_folds < 1:
        raise ValueError("cv_inner_folds must be a positive integer")
    y = np.asarray(y)
    parts = {
        key: np.asarray(outer_split[key], dtype=int) for key in ("train", "val", "test")
    }
    for key, indices in parts.items():
        if not len(indices) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Outer {key} indices must be nonempty and unique")
        if np.any(indices < 0) or np.any(indices >= len(y)):
            raise ValueError(f"Outer {key} indices are out of range")
    joined = np.concatenate(list(parts.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Outer training, validation and test rows overlap")
    pool = np.sort(np.r_[parts["train"], parts["val"]])
    local_y = y[pool]
    classification = task in ("binary", "classification")
    classes = np.unique(local_y) if classification else None
    groups = None if groups is None else np.asarray(groups)
    if groups is not None:
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("Sample groups must have one value per dataset row")
        if set(groups[pool]) & set(groups[parts["test"]]):
            raise ValueError("Sample groups cross the outer train/test boundary")
        if set(groups[parts["train"]]) & set(groups[parts["val"]]):
            raise ValueError(
                "Sample groups cross the stored final-fit train/validation boundary"
            )
    local_groups = None if groups is None else groups[pool]
    regression_labels = None
    if stratification not in (None, ZERO_NONZERO):
        raise ValueError("Unknown inner validation stratification policy")
    if stratification is not None:
        if task != "regression" or local_groups is None:
            raise ValueError(
                "Zero/nonzero inner stratification requires grouped regression"
            )
        _, _, regression_labels = zero_nonzero_group_labels(local_y, local_groups)
    split_seed = (int(seed) + int(outer_fold_id) * 10007) % 2**31
    min_class = (
        int(np.unique(local_y, return_counts=True)[1].min()) if classification else None
    )
    pool_has_all_classes = not classification or np.array_equal(classes, np.unique(y))
    chosen, splitter_name = (None, "holdout")
    actual_seed = split_seed
    for count in range(min(requested_folds, len(pool)), 1, -1):
        if not pool_has_all_classes:
            continue
        if classification and (len(classes) < 2 or min_class < count):
            continue
        if local_groups is not None and len(np.unique(local_groups)) < count:
            continue
        if (
            regression_labels is not None
            and np.bincount(regression_labels).min() < count
        ):
            continue
        retries = 16 if classification and local_groups is not None else 1
        for attempt in range(retries):
            actual_seed = (split_seed + attempt) % 2**31
            if local_groups is None:
                splitter = (
                    StratifiedKFold(count, shuffle=True, random_state=actual_seed)
                    if classification
                    else KFold(count, shuffle=True, random_state=actual_seed)
                )
                candidate = list(
                    splitter.split(pool, local_y if classification else None)
                )
                name = "stratified_kfold" if classification else "kfold"
            elif classification:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    candidate = list(
                        StratifiedGroupKFold(
                            count, shuffle=True, random_state=actual_seed
                        ).split(pool, local_y, local_groups)
                    )
                name = "stratified_group_kfold"
            else:
                unique_groups = np.unique(local_groups)
                candidate = []
                splitter = (
                    KFold(count, shuffle=True, random_state=actual_seed)
                    if regression_labels is None
                    else StratifiedKFold(count, shuffle=True, random_state=actual_seed)
                )
                for gt, gv in splitter.split(unique_groups, regression_labels):
                    candidate.append(
                        (
                            np.flatnonzero(np.isin(local_groups, unique_groups[gt])),
                            np.flatnonzero(np.isin(local_groups, unique_groups[gv])),
                        )
                    )
                name = (
                    "group_kfold"
                    if regression_labels is None
                    else "stratified_group_zero_nonzero_kfold"
                )
            valid = all(
                (
                    len(tr)
                    and len(va)
                    and (
                        not classification
                        or (
                            np.array_equal(np.unique(local_y[tr]), classes)
                            and np.array_equal(np.unique(local_y[va]), classes)
                        )
                    )
                    and (
                        local_groups is None
                        or not set(local_groups[tr]) & set(local_groups[va])
                    )
                    for tr, va in candidate
                )
            )
            if valid:
                chosen, splitter_name = (candidate, name)
                break
        if chosen is not None:
            break
    if chosen is None:
        inner_splits = [
            {"train": parts["train"].tolist(), "val": parts["val"].tolist()}
        ]
        actual_seed = split_seed
        if regression_labels is not None and any(
            (set(y[parts[role]] > 0) != {False, True} for role in ("train", "val"))
        ):
            raise ValueError(
                "Stratified regression fallback needs zero and nonzero groups in both roles"
            )
    else:
        inner_splits = [
            {"train": np.sort(pool[tr]).tolist(), "val": np.sort(pool[va]).tolist()}
            for tr, va in chosen
        ]
        validation_rows = np.concatenate([fold["val"] for fold in inner_splits])
        if not np.array_equal(np.sort(validation_rows), pool):
            raise ValueError(
                "Inner validation folds must cover the outer non-test pool exactly once"
            )
    actual = len(inner_splits)
    reason = None
    if actual < requested_folds:
        reason = (
            "No class-complete/group-safe two-fold partition is feasible; retained the stored holdout"
            if actual == 1
            else f"Class/group support permits {actual}, rather than {requested_folds}, inner folds"
        )
    plan = dict(
        protocol=INNER_CV_PROTOCOL,
        requested_folds=requested_folds,
        actual_folds=actual,
        splitter=splitter_name,
        fallback_reason=reason,
        seed=actual_seed,
        grouped=groups is not None,
        pool_size=len(pool),
        outer_test_size=len(parts["test"]),
        min_class_count=min_class,
        folds=inner_splits,
    )
    if stratification is not None:
        plan["stratification"] = stratification
    plan["sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    return plan
