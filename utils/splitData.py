from torch.utils.data import Subset
import numpy as np
from collections import defaultdict

def split_dataset(dataset, train_ratio=0.7, seed=42):
    """
    Split dataset into train and test sets.
    
    Args:
        dataset: dataset to split
        train_ratio: ratio of train set
        seed: random seed
        
    Returns:
        train_set, test_set
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    train_size = int(len(dataset) * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, test_set


def split_dataset_train_val_test(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: dataset to split
        train_ratio: ratio of train set (default 0.6)
        val_ratio: ratio of validation set (default 0.2)
        seed: random seed
        
    Returns:
        train_set, val_set, test_set
        
    Note:
        test_ratio is automatically 1 - train_ratio - val_ratio
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return train_set, val_set, test_set


def split_dataset_train_val_test_stratified(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Stratified split (by class label) into train/val/test.

    Why:
      Random split can move most rare-class samples (e.g. rolling/ball) into val/test,
      leaving train with too few positives and causing irreversible collapse.

    Requirements:
      - dataset must expose labels via `dataset.labels` (list/np array/torch tensor)

    Strategy:
      - For each class, shuffle indices deterministically
      - Ensure at least 1 sample goes to train if the class exists
      - For very small classes (<=2), prioritize train/val and allow empty test
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    if not hasattr(dataset, "labels"):
        raise AttributeError("Dataset must have 'labels' attribute for stratified split")

    labels = np.asarray(dataset.labels)
    n = len(labels)
    if n == 0:
        return Subset(dataset, []), Subset(dataset, []), Subset(dataset, [])

    rng = np.random.RandomState(seed)
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(labels.tolist()):
        class_to_indices[int(y)].append(int(idx))

    train_idx = []
    val_idx = []
    test_idx = []

    for cls, idxs in class_to_indices.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        c = int(idxs.size)
        if c <= 0:
            continue

        # Small-class handling: keep as much as possible in train
        if c == 1:
            n_train, n_val = 1, 0
        elif c == 2:
            # Keep 1 in train, 1 in val (test empty)
            n_train, n_val = 1, 1
        else:
            n_train = int(np.floor(c * float(train_ratio)))
            n_val = int(np.floor(c * float(val_ratio)))

            # Ensure train has at least 1 sample
            n_train = max(1, n_train)

            # Ensure we don't exceed c
            if n_train + n_val > c:
                n_val = max(0, c - n_train)

            # Keep at least 1 sample for test when possible
            if (c - (n_train + n_val)) == 0 and c >= 3:
                if n_val > 0:
                    n_val -= 1
                else:
                    n_train = max(1, n_train - 1)

        n_train = int(min(c, max(0, n_train)))
        n_val = int(min(c - n_train, max(0, n_val)))

        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    # Shuffle within split (deterministic) to avoid class blocks
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()), Subset(dataset, test_idx.tolist())


def sample_condition_keys_from_real_distribution(
    real_condition_keys,
    n_samples: int,
    seed: int = 42,
):
    """
    Sample synthetic-window condition keys to match the empirical distribution
    of real CWRU (or other domain) condition keys.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        return []
    keys = [str(k) for k in list(real_condition_keys)]
    if len(keys) == 0:
        return ["unknown"] * n_samples
    uniq, counts = np.unique(keys, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    rng = np.random.RandomState(int(seed))
    sampled = rng.choice(uniq, size=n_samples, replace=True, p=probs)
    return [str(x) for x in sampled.tolist()]


def _auto_split_condition_groups(condition_ids, seed=42):
    uniq = sorted(list(set([int(x) for x in condition_ids])))
    if len(uniq) < 2:
        raise ValueError("Need at least 2 condition groups for cross-condition split")
    rng = np.random.RandomState(seed)
    uniq_arr = np.asarray(uniq, dtype=np.int64)
    rng.shuffle(uniq_arr)
    split_at = int(np.ceil(float(len(uniq_arr)) * 0.5))
    split_at = min(max(split_at, 1), len(uniq_arr) - 1)
    train_groups = sorted(uniq_arr[:split_at].tolist())
    test_groups = sorted(uniq_arr[split_at:].tolist())
    return train_groups, test_groups


def subsample_train_indices_by_condition_and_class(
    dataset,
    train_idx,
    ratio: float,
    seed: int = 42,
    label=None,
):
    """
    Downsample training indices within each (condition_group, fault_class) bucket.

    Test/val sets are not modified. ratio=1.0 keeps all train indices; ratio=0.5 keeps ~50% per bucket
    (at least 1 sample per non-empty bucket).
    """
    ratio = float(ratio)
    if ratio >= 1.0 - 1e-9 or train_idx is None or len(train_idx) == 0:
        return np.asarray(train_idx, dtype=np.int64)

    if not hasattr(dataset, "labels"):
        raise AttributeError("Dataset must have 'labels' for stratified subsample")
    if not hasattr(dataset, "condition_ids"):
        raise AttributeError("Dataset must have 'condition_ids' for stratified subsample")

    labels = np.asarray(dataset.labels, dtype=np.int32)
    condition_ids = np.asarray(dataset.condition_ids, dtype=np.int32)
    train_idx = np.asarray(train_idx, dtype=np.int64)

    rng = np.random.RandomState(int(seed) + 880011)
    buckets = defaultdict(list)
    for gi in train_idx.tolist():
        buckets[(int(condition_ids[gi]), int(labels[gi]))].append(int(gi))

    kept = []
    n_before = int(train_idx.size)
    for _, idxs in buckets.items():
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        if arr.size <= 1:
            kept.extend(arr.tolist())
            continue
        n_keep = int(np.floor(arr.size * ratio))
        n_keep = max(1, min(n_keep, arr.size))
        kept.extend(arr[:n_keep].tolist())

    out = np.asarray(kept, dtype=np.int64)
    rng.shuffle(out)
    _tag = f" {label}" if label else ""
    print(
        f"[TrainSubsample]{_tag} ratio={ratio:.2f} per (condition,class): "
        f"{n_before} -> {int(out.size)} train windows"
    )
    return out


def split_by_condition_groups(
    dataset,
    condition_ids=None,
    train_groups=None,
    test_groups=None,
    val_ratio_from_train=0.2,
    seed=42,
    train_subsample_ratio=1.0,
    subsample_label=None,
):
    """
    Split dataset by operating-condition groups:
      - train/val from train_groups
      - test from test_groups (zero-shot within-domain evaluation)
    """
    if not hasattr(dataset, "labels"):
        raise AttributeError("Dataset must have 'labels' attribute")

    if condition_ids is None:
        if not hasattr(dataset, "condition_ids"):
            raise AttributeError("Dataset must provide condition_ids or pass condition_ids explicitly")
        condition_ids = np.asarray(dataset.condition_ids, dtype=np.int32)
    else:
        condition_ids = np.asarray(condition_ids, dtype=np.int32)

    if condition_ids.size != len(dataset):
        raise ValueError("condition_ids size must match dataset length")

    if train_groups is None or test_groups is None:
        auto_train, auto_test = _auto_split_condition_groups(condition_ids, seed=seed)
        if train_groups is None:
            train_groups = auto_train
        if test_groups is None:
            test_groups = auto_test

    train_groups = sorted(list(set([int(x) for x in train_groups])))
    test_groups = sorted(list(set([int(x) for x in test_groups])))
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        raise ValueError(f"train_groups and test_groups overlap: {sorted(list(overlap))}")

    train_groups_arr = np.asarray(train_groups, dtype=np.int32)
    test_groups_arr = np.asarray(test_groups, dtype=np.int32)
    in_train_g = np.isin(condition_ids, train_groups_arr)
    in_test_g = np.isin(condition_ids, test_groups_arr)

    # ATT503-B: synthetic windows never enter the test set (LOO / cross-condition eval).
    # Synthetic samples tagged with a held-out test condition are kept in train_pool only.
    is_syn = None
    if hasattr(dataset, "is_synthetic"):
        try:
            _syn = np.asarray(dataset.is_synthetic, dtype=bool)
            if _syn.shape[0] == condition_ids.size:
                is_syn = _syn
        except Exception:
            is_syn = None

    if is_syn is not None:
        test_idx = np.where(in_test_g & (~is_syn))[0]
        train_pool = np.where(in_train_g | is_syn)[0]
        _n_syn_in_test_cond = int(np.sum(is_syn & in_test_g))
    else:
        test_idx = np.where(in_test_g)[0]
        train_pool = np.where(in_train_g)[0]
        _n_syn_in_test_cond = 0

    if train_pool.size == 0 or test_idx.size == 0:
        raise ValueError("Condition split produced empty train pool or empty test set")

    if is_syn is not None and int(is_syn.sum()) > 0:
        print(
            f"[ATT503-Split] synthetic windows: total={int(is_syn.sum())}, "
            f"in_test_set=0, tagged_test_condition_added_to_train_pool={_n_syn_in_test_cond}"
        )

    # Stratified train/val split only inside train condition groups.
    labels = np.asarray(dataset.labels, dtype=np.int32)
    pool_labels = labels[train_pool]
    rng = np.random.RandomState(seed)
    class_to_pool = defaultdict(list)
    for i, y in enumerate(pool_labels.tolist()):
        class_to_pool[int(y)].append(int(train_pool[i]))

    train_idx = []
    val_idx = []
    for _, idxs in class_to_pool.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        
        # [ATT85] Support strict real-only validation if is_synthetic is present
        if hasattr(dataset, "is_synthetic"):
            # Check is_synthetic for these indices
            is_syn_mask = dataset.is_synthetic[idxs]
            real_idxs = idxs[~is_syn_mask]
            syn_idxs = idxs[is_syn_mask]
        else:
            real_idxs = idxs
            syn_idxs = np.array([], dtype=np.int64)

        # Calculate validation size based on REAL samples only
        c = int(real_idxs.size)
        if c <= 1:
            n_val = 0
        else:
            n_val = int(np.floor(float(c) * float(val_ratio_from_train)))
            n_val = min(max(n_val, 1), c - 1)
        
        # Split real samples
        val_subset = real_idxs[:n_val]
        train_subset_real = real_idxs[n_val:]
        
        # Validation gets only real samples
        val_idx.extend(val_subset.tolist())
        
        # Train gets remaining real + ALL synthetic
        train_idx.extend(train_subset_real.tolist())
        train_idx.extend(syn_idxs.tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    # SC1 / tiny single-condition pools: per-class n_val can be 0 (c<=1 real-only, or all-synthetic train).
    if val_idx.size == 0 and train_idx.size > 0:
        _hold_rng = np.random.RandomState(int(seed) + 99173)
        _labels_tr = labels[train_idx]
        _by_cls = defaultdict(list)
        for _pos, _gidx in enumerate(train_idx.tolist()):
            _by_cls[int(_labels_tr[_pos])].append(int(_gidx))
        _new_train, _new_val = [], []
        for _cls, _idxs in _by_cls.items():
            _idxs = np.asarray(_idxs, dtype=np.int64)
            _hold_rng.shuffle(_idxs)
            if _idxs.size <= 1:
                _new_train.extend(_idxs.tolist())
                continue
            _nv = int(np.floor(float(_idxs.size) * float(val_ratio_from_train)))
            _nv = min(max(_nv, 1), int(_idxs.size) - 1)
            _new_val.extend(_idxs[:_nv].tolist())
            _new_train.extend(_idxs[_nv:].tolist())
        if len(_new_val) > 0:
            train_idx = np.asarray(_new_train, dtype=np.int64)
            val_idx = np.asarray(_new_val, dtype=np.int64)

    _train_before_sub = int(train_idx.size)
    train_idx = subsample_train_indices_by_condition_and_class(
        dataset,
        train_idx,
        ratio=float(train_subsample_ratio),
        seed=int(seed),
        label=subsample_label,
    )

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    meta = {
        "train_groups": train_groups,
        "test_groups": test_groups,
        "train_size": int(train_idx.size),
        "train_size_before_subsample": _train_before_sub,
        "train_subsample_ratio": float(train_subsample_ratio),
        "val_size": int(val_idx.size),
        "test_size": int(test_idx.size),
        "synthetic_excluded_from_test": int(is_syn.sum()) if is_syn is not None else 0,
        "synthetic_reassigned_to_train_pool": _n_syn_in_test_cond,
    }
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
        meta,
    )


def subset_by_condition(dataset, condition_gid: int) -> Subset:
    """
    Extract samples belonging to one operating-condition group.

    Works on the root dataset or on a ``Subset`` (e.g. merged STMO test set).
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        base_indices = np.asarray(dataset.indices, dtype=np.int64)
    else:
        base = dataset
        base_indices = np.arange(len(dataset), dtype=np.int64)

    if not hasattr(base, "condition_ids"):
        raise AttributeError("Underlying dataset must provide condition_ids")

    cids = np.asarray(base.condition_ids, dtype=np.int32)
    if base_indices.size == 0:
        return Subset(base, [])

    cond_mask = cids[base_indices] == int(condition_gid)
    selected = base_indices[cond_mask].tolist()
    return Subset(base, selected)
