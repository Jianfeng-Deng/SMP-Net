"""
Episodic Cross-Condition sampler for domain generalization.
Splits each batch into support/query sets based on condition or domain keys.
Ref: MGADA (Machines 2025), Episodic DG-GCD (arXiv 2503.14897)
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict


class EpisodicConditionSampler(Sampler):
    """
    Yields batches where samples are tagged with group membership.
    At each iteration, one group is designated as "support" and the other as "query".
    Groups alternate every batch so both directions are covered each epoch.

    Works with any split key:
      - condition keys (gk2/gk4 for CWRU, gk6/gk7 for PU)
      - domain keys (CWRU/PU for Task2)
    """

    def __init__(self, dataset, split_key="condition", batch_size=64, drop_last=True, seed=42):
        """
        Args:
            dataset: MultimodalBearingDataset with .condition_keys or .domain attribute per sample
            split_key: "condition" to split by condition_keys, "domain" to split by domain
            batch_size: total batch size (split ~50/50 between support and query)
            drop_last: drop last incomplete batch
            seed: random seed for reproducibility
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        group_indices = defaultdict(list)
        if split_key == "domain":
            for i in range(len(dataset)):
                if hasattr(dataset, 'domain'):
                    group_indices[dataset.domain].append(i)
                else:
                    group_indices["default"].append(i)
        else:
            if hasattr(dataset, 'condition_keys'):
                for i, key in enumerate(dataset.condition_keys):
                    group_indices[str(key)].append(i)
            else:
                for i in range(len(dataset)):
                    group_indices["default"].append(i)

        self.groups = {k: np.array(v) for k, v in group_indices.items()}
        self.group_names = sorted(self.groups.keys())

        if len(self.group_names) < 2:
            print(f"[EpisodicSampler] WARNING: only {len(self.group_names)} group(s) found "
                  f"({self.group_names}). Falling back to standard random sampling.")
            self.fallback = True
            self.all_indices = np.arange(len(dataset))
        else:
            self.fallback = False
            print(f"[EpisodicSampler] groups={self.group_names}, "
                  f"sizes={[len(self.groups[g]) for g in self.group_names]}, "
                  f"batch_size={batch_size}")

        self.n_samples = len(dataset)

    def __iter__(self):
        if self.fallback:
            indices = self.rng.permutation(self.all_indices)
            for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                yield indices[start:start + self.batch_size].tolist()
            return

        half = self.batch_size // 2
        shuffled = {}
        for g in self.group_names:
            shuffled[g] = self.rng.permutation(self.groups[g])

        pointers = {g: 0 for g in self.group_names}
        direction = 0

        while True:
            g_support = self.group_names[direction % len(self.group_names)]
            g_query = self.group_names[(direction + 1) % len(self.group_names)]
            direction += 1

            p_s, p_q = pointers[g_support], pointers[g_query]
            arr_s, arr_q = shuffled[g_support], shuffled[g_query]

            if p_s + half > len(arr_s) or p_q + half > len(arr_q):
                break

            support_idx = arr_s[p_s:p_s + half]
            query_idx = arr_q[p_q:p_q + half]
            pointers[g_support] = p_s + half
            pointers[g_query] = p_q + half

            batch_idx = np.concatenate([support_idx, query_idx])
            yield batch_idx.tolist()

    def __len__(self):
        if self.fallback:
            return self.n_samples // self.batch_size
        min_group_size = min(len(self.groups[g]) for g in self.group_names)
        half = self.batch_size // 2
        return (min_group_size // half) * len(self.group_names) // 2


def split_batch_by_condition(batch, dataset_ref=None, split_key="condition", batch_idx=0):
    """
    Split a collated batch into support and query subsets based on condition/domain.
    The support group alternates with batch_idx to ensure bidirectional learning.

    Returns:
        support_mask: bool tensor [B], True for support samples
        query_mask: bool tensor [B], True for query samples
        group_ids: tensor [B] of integer group ids
    """
    conditions = batch.get("condition", None)
    domains = batch.get("domain", None)

    if split_key == "domain" and domains is not None:
        keys = domains
    elif conditions is not None:
        keys = conditions
    else:
        B = batch["signal"].shape[0]
        return (torch.ones(B, dtype=torch.bool),
                torch.zeros(B, dtype=torch.bool),
                torch.zeros(B, dtype=torch.long))

    unique_keys = sorted(set(keys))
    if len(unique_keys) < 2:
        B = len(keys)
        return (torch.ones(B, dtype=torch.bool),
                torch.zeros(B, dtype=torch.bool),
                torch.zeros(B, dtype=torch.long))

    key_to_id = {k: i for i, k in enumerate(unique_keys)}
    group_ids = torch.tensor([key_to_id[k] for k in keys], dtype=torch.long)

    support_group = batch_idx % len(unique_keys)
    support_mask = (group_ids == support_group)
    query_mask = ~support_mask

    return support_mask, query_mask, group_ids
