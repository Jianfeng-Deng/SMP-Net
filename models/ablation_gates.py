"""
Learnable residual gates for ablation-aligned sub-modules (PIDR / GIDC / GCSM).

Each gate g in (0, 1) blends module output against an identity (bypass) path:
    out = identity + g * (module_out - identity)

g -> 0 : module fully bypassed (equivalent to wo_* ablation)
g -> 1 : module fully active (equivalent to Full model)
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn

# Keys aligned with experiments/run_ablation.sh module boundaries.
ABLATION_GATE_KEYS: List[str] = [
    # --- PIDR (Module A) ---
    "msstft",
    "freq_encoding",
    "physics_wst",
    "physics_env",
    "physics_phys_proto",
    "domain_cond_fs_film",
    "domain_cond_stat_film",
    "domain_cond_mixstyle_spec",
    "domain_cond_mixstyle_signal",
    "domain_cond_domain_film_spec",
    "domain_cond_domain_film_signal",
    "domain_cond_lora_signal",
    # --- GIDC (Module B) ---
    "gidc_dual_contract",
    "gidc_mean_residual",
    "gidc_cross_modal_align",
    # --- GCSM (Module C) ---
    "gcsm_fusion_align",
    "gcsm_metric_geometry",
    "gcsm_silhouette",
]

TensorLike = Union[torch.Tensor, float, int]


class LearnableAblationGate(nn.Module):
    """Scalar gate initialized so sigmoid(logit) ~= init_value."""

    def __init__(self, init: float = 0.5, name: str = ""):
        super().__init__()
        self.name = str(name)
        init = float(max(1e-4, min(1.0 - 1e-4, init)))
        logit = math.log(init / (1.0 - init))
        self.logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def value(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def forward(
        self,
        module_out: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        g = self.value()
        if module_out.shape != identity.shape:
            raise ValueError(
                f"[AblationGate:{self.name}] shape mismatch: "
                f"module={tuple(module_out.shape)} identity={tuple(identity.shape)}"
            )
        return identity + g * (module_out - identity)


class AblationGateBank(nn.Module):
    """Registry of learnable gates for all ablation sub-modules."""

    def __init__(
        self,
        keys: Optional[Iterable[str]] = None,
        init_default: float = 0.5,
        init_overrides: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        key_list = list(keys) if keys is not None else list(ABLATION_GATE_KEYS)
        overrides = dict(init_overrides or {})
        self._gates = nn.ModuleDict()
        for key in key_list:
            init_v = float(overrides.get(key, init_default))
            safe_key = key.replace(".", "_")
            self._gates[safe_key] = LearnableAblationGate(init=init_v, name=key)
        self._key_to_safe = {k: k.replace(".", "_") for k in key_list}
        self._safe_to_key = {v: k for k, v in self._key_to_safe.items()}

    def keys(self) -> List[str]:
        return list(self._key_to_safe.keys())

    def scalar(self, key: str) -> torch.Tensor:
        safe = self._key_to_safe.get(key)
        if safe is None:
            raise KeyError(f"Unknown ablation gate key: {key}")
        return self._gates[safe].value()

    def blend(
        self,
        key: str,
        module_out: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        safe = self._key_to_safe.get(key)
        if safe is None:
            return module_out
        return self._gates[safe](module_out, identity)

    def scale_loss(self, key: str, loss: TensorLike) -> TensorLike:
        """Multiply a loss term by gate scalar (0 = off, 1 = full)."""
        if not isinstance(loss, torch.Tensor):
            if loss == 0 or loss == 0.0:
                return loss
            loss = torch.tensor(float(loss), dtype=torch.float32)
        g = self.scalar(key)
        return g * loss

    def summary_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for safe, gate in self._gates.items():
            key = self._safe_to_key.get(safe, safe)
            out[key] = float(gate.value().detach().cpu().item())
        return out

    def log_summary(self, prefix: str = "[AblationGates]") -> None:
        parts = [f"{k}={v:.3f}" for k, v in sorted(self.summary_dict().items())]
        print(f"{prefix} " + ", ".join(parts))


def apply_residual_gate(
    gate_bank: Optional[AblationGateBank],
    key: str,
    module_out: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    if gate_bank is None:
        return module_out
    return gate_bank.blend(key, module_out, identity)


def scale_by_gate(
    gate_bank: Optional[AblationGateBank],
    key: str,
    value: TensorLike,
) -> TensorLike:
    if gate_bank is None:
        return value
    return gate_bank.scale_loss(key, value)
