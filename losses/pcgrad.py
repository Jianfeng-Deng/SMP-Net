"""
PCGrad: Projected Conflicting Gradients for Multi-Task Learning.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

Resolves gradient conflicts between loss groups by projecting conflicting
gradients onto the normal plane of each other, ensuring no loss term
actively harms another during joint optimization.
"""

import torch
import random
from typing import List, Optional, Dict


def pcgrad_backward(
    losses: List[torch.Tensor],
    shared_params: List[torch.nn.Parameter],
    loss_names: Optional[List[str]] = None,
    retain_graph: bool = True,
) -> Dict[str, float]:
    """
    Apply PCGrad to a list of loss tensors and accumulate gradients on shared_params.

    Each loss's gradient is computed independently, then conflicting pairs
    are resolved by random-order projection. The final combined gradient
    is accumulated into param.grad.

    Args:
        losses: list of scalar loss tensors (must require grad)
        shared_params: parameters to compute gradients for
        loss_names: optional names for diagnostics
        retain_graph: whether to retain computation graph

    Returns:
        Dictionary with conflict diagnostics:
        - "n_conflicts": number of gradient pair conflicts detected
        - "n_pairs": total gradient pairs compared
        - "conflict_ratio": fraction of conflicting pairs
    """
    if not losses:
        return {"n_conflicts": 0, "n_pairs": 0, "conflict_ratio": 0.0}

    valid_losses = [(i, l) for i, l in enumerate(losses) if torch.is_tensor(l) and l.requires_grad]
    if not valid_losses:
        return {"n_conflicts": 0, "n_pairs": 0, "conflict_ratio": 0.0}

    param_shapes = [p.shape for p in shared_params]
    param_numels = [p.numel() for p in shared_params]
    total_dim = sum(param_numels)

    if total_dim == 0:
        return {"n_conflicts": 0, "n_pairs": 0, "conflict_ratio": 0.0}

    device = shared_params[0].device if shared_params else 'cpu'
    task_grads_flat = []

    for idx, l in valid_losses:
        grads = torch.autograd.grad(
            l, shared_params, retain_graph=retain_graph,
            create_graph=False, allow_unused=True
        )
        flat_parts = []
        for g, numel in zip(grads, param_numels):
            if g is not None:
                flat_parts.append(g.reshape(-1))
            else:
                flat_parts.append(torch.zeros(numel, device=device))
        task_grads_flat.append(torch.cat(flat_parts))

    n_tasks = len(task_grads_flat)
    if n_tasks <= 1:
        if task_grads_flat:
            _set_param_grads(task_grads_flat[0], shared_params, param_numels)
        return {"n_conflicts": 0, "n_pairs": 0, "conflict_ratio": 0.0}

    n_conflicts = 0
    n_pairs = 0

    projected = [g.clone() for g in task_grads_flat]

    order = list(range(n_tasks))
    random.shuffle(order)

    for i in order:
        others = [j for j in range(n_tasks) if j != i]
        random.shuffle(others)
        for j in others:
            dot = torch.dot(projected[i], task_grads_flat[j])
            n_pairs += 1
            if dot < 0:
                n_conflicts += 1
                norm_sq = torch.dot(task_grads_flat[j], task_grads_flat[j])
                if norm_sq > 1e-12:
                    projected[i] = projected[i] - (dot / norm_sq) * task_grads_flat[j]

    combined = torch.stack(projected).sum(dim=0)
    _set_param_grads(combined, shared_params, param_numels)

    conflict_ratio = float(n_conflicts) / max(1, n_pairs)
    return {
        "n_conflicts": n_conflicts,
        "n_pairs": n_pairs,
        "conflict_ratio": conflict_ratio,
    }


def _set_param_grads(
    flat_grad: torch.Tensor,
    params: List[torch.nn.Parameter],
    param_numels: List[int],
):
    offset = 0
    for param, numel in zip(params, param_numels):
        g = flat_grad[offset:offset + numel].reshape(param.shape)
        if param.grad is None:
            param.grad = g.clone()
        else:
            param.grad.add_(g)
        offset += numel
