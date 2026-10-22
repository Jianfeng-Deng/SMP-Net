import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    use_class_weights: bool = True,
    return_stats: bool = False,
):
    """
    Decoupled Supervised Contrastive Loss (DCL-SupCon).
    Att83: Replaced standard InfoNCE with DCL [Yeh et al., ECCV 2022].
    
    Key change: denominator sums over NEGATIVE samples only (excluding positives),
    eliminating the Negative-Positive-Coupling (NPC) effect that causes feature
    collapse in small batches (CWRU B≈45, where positives are ~25% of denominator).
    
    Args:
        z: [B, D] feature embeddings
        labels: [B] int class labels
        temperature: float
        use_class_weights: if True, reweight anchors by inverse class frequency (batch-level)
        return_stats: if True, also return dict of stats
    """
    if z is None or labels is None or z.dim() != 2:
        loss = torch.tensor(0.0, device=z.device if z is not None else "cpu")
        return (loss, {"pos_pairs": 0, "valid_anchors": 0}) if return_stats else loss
    if z.size(0) < 2:
        loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        return (loss, {"pos_pairs": 0, "valid_anchors": 0}) if return_stats else loss

    z = F.normalize(z.float(), dim=-1)
    labels = labels.long().view(-1, 1)
    device = z.device
    batch_size = z.size(0)

    # Similarity matrix
    logits = (z @ z.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Mask for positives (same class) excluding self
    mask = (labels == labels.t()).float()
    self_mask = torch.eye(batch_size, device=device, dtype=mask.dtype)
    mask = mask - self_mask

    # --- Att83 DCL: denominator uses NEGATIVE samples only ---
    # Before (InfoNCE): exp_logits = torch.exp(logits) * (1.0 - self_mask)
    # After  (DCL):     exp_logits = torch.exp(logits) * neg_mask
    neg_mask = 1.0 - mask - self_mask  # excludes both self AND positives
    exp_logits_neg = torch.exp(logits) * neg_mask
    log_prob = logits - torch.log(exp_logits_neg.sum(dim=1, keepdim=True).clamp(min=1e-6))

    pos_counts = mask.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        loss = torch.tensor(0.0, device=device, dtype=z.dtype)
        return (loss, {"pos_pairs": 0, "valid_anchors": 0}) if return_stats else loss

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1.0)
    loss_vec = -mean_log_prob_pos

    if use_class_weights:
        counts = torch.bincount(labels.view(-1), minlength=int(labels.max().item()) + 1).float()
        counts = counts.to(device=device)
        inv = 1.0 / counts.clamp(min=1.0)
        anchor_w = inv[labels.view(-1)]
        loss_vec = loss_vec * anchor_w

    loss = loss_vec[valid].mean()

    if not return_stats:
        return loss

    pos_pairs = int(mask.sum().detach().cpu().item())
    valid_anchors = int(valid.sum().detach().cpu().item())
    return loss, {"pos_pairs": pos_pairs, "valid_anchors": valid_anchors}

