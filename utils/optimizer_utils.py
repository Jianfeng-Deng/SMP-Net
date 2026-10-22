"""Helpers to build PyTorch optimizers without duplicate parameters across groups."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def unique_parameters(params: Iterable[Any]) -> List[Any]:
    """Drop duplicate parameter tensors (by object id) while preserving order."""
    seen: Set[int] = set()
    out: List[Any] = []
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def assigned_parameter_ids(param_groups: List[Dict[str, Any]]) -> Set[int]:
    ids: Set[int] = set()
    for group in param_groups:
        for p in group.get("params", []):
            ids.add(id(p))
    return ids


def sanitize_optimizer_param_groups(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate parameters within and across groups.
    Later groups lose conflicting params (earlier groups win).
    """
    assigned: Set[int] = set()
    sanitized: List[Dict[str, Any]] = []
    for group in param_groups:
        params = unique_parameters(group.get("params", []))
        kept = [p for p in params if id(p) not in assigned]
        if not kept:
            continue
        for p in kept:
            assigned.add(id(p))
        new_group = dict(group)
        new_group["params"] = kept
        sanitized.append(new_group)
    return sanitized
