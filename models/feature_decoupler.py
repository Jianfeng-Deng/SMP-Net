"""
ATT396: Feature Decoupling Module for Condition-Invariant Fault Diagnosis.

References:
    - DFDN (Reliability Engineering & System Safety, 2024):
      "A domain feature decoupling network for rotating machinery fault
       diagnosis under unseen operating conditions"
    - FDIDG (Pattern Recognition, 2024):
      "Feature decoupling integrated domain generalization network
       for bearing fault diagnosis under unknown operating conditions"

Architecture:
    z_enc ─┬─ FaultProjector ──→ z_fault   (condition-invariant, used for PACL/fusion)
           └─ ConditionProjector ──→ z_cond (condition-specific, auxiliary training signal)

Training Objectives:
    1. L_pacl(z_fault, prototypes) — fault classification (existing)
    2. L_cond(z_cond, cond_labels)  — condition classification (auxiliary)
    3. L_adv(GRL(z_fault), cond_labels) — adversarial: z_fault has NO condition info
    4. L_ortho(z_fault, z_cond)     — orthogonality: fault ⊥ condition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def set_lambda(self, val: float):
        self.lambda_ = float(val)

    def forward(self, x):
        return _GradientReversal.apply(x, self.lambda_)


class FeatureDecoupler(nn.Module):
    """
    Decouple encoder embedding into fault-relevant and condition-relevant subspaces.

    fault_proj: z → z_fault  (residual gated projection)
    condition_proj: z → z_cond (separate projection)
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim

        self.fault_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.fault_gate = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, D] encoder embedding
        Returns:
            z_fault: [B, D] condition-invariant features
            z_cond:  [B, D] condition-specific features
        """
        gate = torch.sigmoid(self.fault_gate)
        z_fault = z + gate * self.fault_proj(z)
        z_cond = self.condition_proj(z)
        return z_fault, z_cond


class ConditionClassifier(nn.Module):
    """Lightweight head that predicts operating-condition ID from features."""

    def __init__(self, embed_dim: int = 256, num_conditions: int = 4,
                 hidden_dim: int = 128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_conditions),
        )

    def forward_adversarial(self, z: torch.Tensor, grl_lambda: float = 1.0):
        """Adversarial path: GRL reverses gradients so encoder REMOVES condition info."""
        self.grl.set_lambda(grl_lambda)
        return self.head(self.grl(z))

    def forward_direct(self, z: torch.Tensor):
        """Direct path: normal gradients so condition_proj LEARNS condition info."""
        return self.head(z)


def encode_condition_ids(domains, conditions, device):
    """
    Fine-grained operating-condition encoding for Feature Decoupling.

    Mapping (training time):
        CWRU gk2  → 0
        CWRU gk4  → 1
        PU   gk6  → 2
        PU   gk7  → 3
    Unseen conditions (test time): not used.
    """
    ids = []
    for i, d in enumerate(domains):
        d_upper = str(d).upper()
        cond = str(conditions[i]).lower() if (conditions is not None and i < len(conditions)) else ""
        if d_upper == "CWRU":
            if "gk4" in cond or "gk_4" in cond or "hp3" in cond:
                ids.append(1)
            else:
                ids.append(0)
        elif d_upper == "PU":
            if "gk7" in cond:
                ids.append(3)
            else:
                ids.append(2)
        else:
            ids.append(0)
    return torch.tensor(ids, dtype=torch.long, device=device)


def compute_decouple_losses(
    z_fault_spec: torch.Tensor,
    z_cond_spec: torch.Tensor,
    z_fault_signal: torch.Tensor,
    z_cond_signal: torch.Tensor,
    condition_ids: torch.Tensor,
    condition_classifier: ConditionClassifier,
    grl_lambda: float = 1.0,
    w_cond: float = 0.1,
    w_adv: float = 0.1,
    w_ortho: float = 0.05,
):
    """
    Compute Feature Decoupling auxiliary losses.

    Returns:
        total_loss: scalar
        loss_dict: {str: float} for logging
    """
    loss_dict = {}
    total = torch.tensor(0.0, device=z_fault_spec.device)

    z_cond_cat = torch.cat([z_cond_spec, z_cond_signal], dim=0)
    z_fault_cat = torch.cat([z_fault_spec, z_fault_signal], dim=0)
    cond_ids_cat = torch.cat([condition_ids, condition_ids], dim=0)

    if w_cond > 0:
        logits_cond = condition_classifier.forward_direct(z_cond_cat)
        loss_cond = F.cross_entropy(logits_cond, cond_ids_cat)
        total = total + w_cond * loss_cond
        with torch.no_grad():
            acc_cond = (logits_cond.argmax(1) == cond_ids_cat).float().mean()
        loss_dict["decouple_cond_loss"] = float(loss_cond.item())
        loss_dict["decouple_cond_acc"] = float(acc_cond.item())

    if w_adv > 0:
        logits_adv = condition_classifier.forward_adversarial(z_fault_cat, grl_lambda)
        loss_adv = F.cross_entropy(logits_adv, cond_ids_cat)
        total = total + w_adv * loss_adv
        with torch.no_grad():
            acc_adv = (logits_adv.argmax(1) == cond_ids_cat).float().mean()
        loss_dict["decouple_adv_loss"] = float(loss_adv.item())
        loss_dict["decouple_adv_acc"] = float(acc_adv.item())

    if w_ortho > 0:
        cos_spec = F.cosine_similarity(z_fault_spec, z_cond_spec, dim=-1).abs().mean()
        cos_signal = F.cosine_similarity(z_fault_signal, z_cond_signal, dim=-1).abs().mean()
        loss_ortho = (cos_spec + cos_signal) * 0.5
        total = total + w_ortho * loss_ortho
        loss_dict["decouple_ortho_loss"] = float(loss_ortho.item())

    loss_dict["decouple_total_loss"] = float(total.item())
    return total, loss_dict
