"""
Prototype-Anchored Contrastive Loss (PACL).

Forces learned feature embeddings to align with fixed (or slowly updated) class prototypes.
Inspired by CLIP's contrastive alignment and Neural Collapse's ETF geometry.

The loss has two components:
  1. Cross-entropy anchoring: each sample should be closest to its class prototype
  2. Margin enforcement: minimum angular separation between different-class samples

Reference:
  - Radford et al. "Learning Transferable Visual Models..." (CLIP), ICML 2021.
  - Papyan et al. "Prevalence of neural collapse..." PNAS 2020.
  - Deng et al. "ArcFace: Additive Angular Margin Loss..." CVPR 2019.
"""

import torch
import torch.nn.functional as F


def prototype_anchor_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
    use_margin: bool = False,
    margin: float = 0.30,
) -> dict:
    """
    Compute prototype-anchored loss.

    Default: cross-entropy mode.
    ATT379(S4) margin mode: L = max(0, m - cos(z, p_pos) + cos(z, p_hardneg))
    Margin mode always provides gradient even when CE accuracy = 100%.

    Args:
        z: [B, D] feature embeddings (will be L2-normalized)
        labels: [B] integer class labels
        prototypes: [C, D] class prototypes (will be L2-normalized)
        temperature: softmax temperature (CE mode only)
        label_smoothing: label smoothing factor (CE mode only)
        use_margin: if True, use margin-based triplet loss instead of CE
        margin: margin value for triplet loss
    """
    z_norm = F.normalize(z.float(), dim=-1)
    p_norm = F.normalize(prototypes.float().to(z.device), dim=-1)

    cos_sim = torch.matmul(z_norm, p_norm.T)  # [B, C] raw cosine

    B = z_norm.size(0)
    C = p_norm.size(0)

    valid = (labels >= 0) & (labels < C)
    if valid.sum() == 0:
        return {
            "loss": torch.tensor(0.0, device=z.device, requires_grad=True),
            "acc": 0.0,
            "mean_pos_cos": 0.0,
            "mean_neg_cos": 0.0,
        }

    cos_valid = cos_sim[valid]
    labels_valid = labels[valid]

    if use_margin:
        pos_cos = cos_valid[torch.arange(cos_valid.size(0), device=cos_valid.device), labels_valid]
        neg_mask = torch.ones_like(cos_valid, dtype=torch.bool)
        neg_mask[torch.arange(cos_valid.size(0), device=cos_valid.device), labels_valid] = False
        hardneg_cos = cos_valid.masked_fill(~neg_mask, -1e9).max(dim=1).values
        loss = torch.clamp(margin - pos_cos + hardneg_cos, min=0.0).mean()
    else:
        logits_valid = cos_valid / max(temperature, 1e-6)
        loss = F.cross_entropy(
            logits_valid, labels_valid, label_smoothing=label_smoothing
        )

    with torch.no_grad():
        preds = cos_valid.argmax(dim=1)
        acc = (preds == labels_valid).float().mean().item()

        pos_mask_oh = F.one_hot(labels_valid, C).bool()
        mean_pos_cos = cos_valid[pos_mask_oh].mean().item()
        mean_neg_cos = cos_valid[~pos_mask_oh].mean().item()

        per_class_acc = {}
        per_class_pos_cos = {}
        for c in range(C):
            c_mask = labels_valid == c
            if c_mask.sum() > 0:
                per_class_acc[c] = (preds[c_mask] == c).float().mean().item()
                per_class_pos_cos[c] = cos_valid[c_mask, c].mean().item()

    return {
        "loss": loss,
        "acc": acc,
        "mean_pos_cos": mean_pos_cos,
        "mean_neg_cos": mean_neg_cos,
        "per_class_acc": per_class_acc,
        "per_class_pos_cos": per_class_pos_cos,
    }


def text_etf_alignment_loss(
    text_encoder,
    label_to_prompts: dict,
    etf_prototypes: torch.Tensor,
    device: str = "cpu",
    cosine_matrix_weight: float = 0.0,
) -> dict:
    """
    Align text-encoder outputs with ETF prototype positions.
    ATT377: optionally adds cosine-matrix matching loss (E1) and diagnostic monitoring (F2).
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    cos_vals = []
    n_classes = 0
    text_proto_list = []
    label_order = []

    for label, prompts in sorted(label_to_prompts.items(), key=lambda x: int(x[0])):
        label_int = int(label)
        if label_int >= etf_prototypes.size(0):
            continue
        target = F.normalize(etf_prototypes[label_int].to(device).float(), dim=-1)

        text_embs = text_encoder(prompts)
        if text_embs.dim() == 1:
            text_embs = text_embs.unsqueeze(0)
        mean_emb = F.normalize(text_embs.mean(dim=0), dim=-1)

        cos = F.cosine_similarity(mean_emb.unsqueeze(0), target.unsqueeze(0)).squeeze()
        total_loss = total_loss + (1.0 - cos)
        cos_vals.append(cos.detach().item())
        text_proto_list.append(mean_emb)
        label_order.append(label_int)
        n_classes += 1

    if n_classes > 0:
        total_loss = total_loss / n_classes

    mean_cos = sum(cos_vals) / max(1, len(cos_vals)) if cos_vals else 0.0

    # ATT377 F2: text prototype cosine matrix monitoring + E1: matrix matching loss
    text_cos_diag = 0.0
    text_cos_offdiag = 0.0
    text_cos_margin = 0.0
    loss_matrix_match = torch.tensor(0.0, device=device)
    if len(text_proto_list) >= 2:
        text_protos = torch.stack(text_proto_list, dim=0)  # [C, D]
        text_cos_mat = torch.matmul(text_protos, text_protos.t())  # [C, C]
        C = int(text_cos_mat.size(0))
        text_cos_diag = float(text_cos_mat.diag().mean().detach().cpu().item())
        offdiag_mask = ~torch.eye(C, dtype=torch.bool, device=text_cos_mat.device)
        text_cos_offdiag = float(text_cos_mat[offdiag_mask].mean().detach().cpu().item())
        text_cos_margin = text_cos_diag - text_cos_offdiag

        # E1: Frobenius norm matching between text cosine matrix and ETF cosine matrix
        if cosine_matrix_weight > 0.0 and len(label_order) >= 2:
            etf_sub = F.normalize(etf_prototypes[label_order].to(device).float(), dim=-1)
            etf_cos_mat = torch.matmul(etf_sub, etf_sub.t())
            loss_matrix_match = ((text_cos_mat - etf_cos_mat) ** 2).mean()
            total_loss = total_loss + cosine_matrix_weight * loss_matrix_match

    return {
        "loss": total_loss,
        "mean_cos": mean_cos,
        "text_cos_diag": text_cos_diag,
        "text_cos_offdiag": text_cos_offdiag,
        "text_cos_margin": text_cos_margin,
        "loss_matrix_match": float(loss_matrix_match.detach().cpu().item()),
    }


def prototype_separation_loss(
    prototypes: torch.Tensor,
    target_cos: float = -0.333,
) -> torch.Tensor:
    """
    Regularize prototypes toward target pairwise cosine (for learnable prototypes).
    For fixed ETF prototypes this is not needed.

    Args:
        prototypes: [C, D]
        target_cos: target off-diagonal cosine (-1/(C-1) for ETF)

    Returns:
        scalar loss
    """
    p = F.normalize(prototypes.float(), dim=-1)
    C = p.size(0)
    cos_mat = p @ p.T
    mask = ~torch.eye(C, dtype=torch.bool, device=p.device)
    off_diag = cos_mat[mask]
    loss = F.mse_loss(off_diag, torch.full_like(off_diag, target_cos))
    return loss
