import torch
import torch.nn.functional as F


def cosine_center_loss(z: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Cosine center loss on normalized embeddings.

    Args:
        z: [B, D] normalized embeddings (z_cls)
        labels: [B] int labels in [0, C-1]
        centers: [C, D] normalized centers
        class_weights: optional [C] weights for sample reweighting

    Returns:
        scalar loss
    """
    if z.numel() == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    z = F.normalize(z, dim=-1)
    centers = F.normalize(centers, dim=-1)

    c = centers[labels]  # [B, D]
    cos_sim = (z * c).sum(dim=1)  # [B]
    loss_vec = 1.0 - cos_sim

    if class_weights is not None:
        w = class_weights.to(z.device)[labels]
        w = w / (w.mean().clamp(min=1e-6))
        loss_vec = loss_vec * w

    return loss_vec.mean()


@torch.no_grad()
def update_ema_centers(
    centers: torch.Tensor,
    z: torch.Tensor,
    labels: torch.Tensor,
    momentum: float = 0.95
) -> torch.Tensor:
    """
    Update EMA centers with current batch embeddings.

    centers: [C, D]
    z: [B, D] (will be normalized)
    labels: [B]
    """
    z = F.normalize(z, dim=-1)
    C = centers.size(0)

    for c in range(C):
        mask = (labels == c)
        if mask.any():
            batch_mean = z[mask].mean(dim=0)
            centers[c] = momentum * centers[c] + (1.0 - momentum) * batch_mean

    centers = F.normalize(centers, dim=-1)
    return centers


