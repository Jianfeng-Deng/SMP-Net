import torch
import torch.nn.functional as F

def spec_clip_alignment_loss(spec_emb, clip_emb):
    """
    spec_emb: [B, D]
    clip_emb: [B, D]
    """
    spec_emb = F.normalize(spec_emb, dim=-1)
    clip_emb = F.normalize(clip_emb, dim=-1)

    return 1.0 - (spec_emb * clip_emb).sum(dim=1).mean()


def clip_style_batch_contrastive_loss(
    visual_emb,
    text_emb,
    temperature=0.07,
    labels=None,
):
    """
    CLIP-style symmetric batch contrastive loss.
    ATT377 F1: when `labels` is provided, use SupCLIP — samples sharing the
    same label form positive pairs (supervised multi-positive InfoNCE).
    """
    _zero = {"loss": None, "diag_cos_mean": 0.0, "offdiag_cos_mean": 0.0}
    if (
        visual_emb is None
        or text_emb is None
        or (not torch.is_tensor(visual_emb))
        or (not torch.is_tensor(text_emb))
        or visual_emb.dim() != 2
        or text_emb.dim() != 2
        or int(visual_emb.size(0)) <= 1
        or int(text_emb.size(0)) <= 1
        or int(visual_emb.size(0)) != int(text_emb.size(0))
        or int(visual_emb.size(1)) != int(text_emb.size(1))
    ):
        dev = None
        if torch.is_tensor(visual_emb):
            dev = visual_emb.device
        elif torch.is_tensor(text_emb):
            dev = text_emb.device
        if dev is None:
            dev = "cpu"
        _zero["loss"] = torch.tensor(0.0, device=dev, requires_grad=True)
        return _zero

    v = F.normalize(visual_emb.float(), dim=-1)
    t = F.normalize(text_emb.float().to(v.device), dim=-1)
    temp = max(float(temperature), 1e-6)
    B = int(v.size(0))

    logits_v2t = torch.matmul(v, t.t()) / temp  # [B, B]
    logits_t2v = logits_v2t.t()

    use_supclip = (labels is not None and torch.is_tensor(labels) and labels.numel() == B)

    if use_supclip:
        # SupCLIP: multi-positive cross-entropy
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
        pos_mask = pos_mask.float()
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)

        # v->t: for each visual, sum over positive texts
        log_softmax_v2t = logits_v2t - logits_v2t.logsumexp(dim=1, keepdim=True)
        loss_v = -(pos_mask * log_softmax_v2t).sum(dim=1) / pos_count
        loss_v = loss_v.mean()

        # t->v: symmetric
        log_softmax_t2v = logits_t2v - logits_t2v.logsumexp(dim=1, keepdim=True)
        loss_t = -(pos_mask.t() * log_softmax_t2v).sum(dim=1) / pos_count
        loss_t = loss_t.mean()
    else:
        targets = torch.arange(B, device=v.device, dtype=torch.long)
        loss_v = F.cross_entropy(logits_v2t, targets)
        loss_t = F.cross_entropy(logits_t2v, targets)

    loss = 0.5 * (loss_v + loss_t)

    with torch.no_grad():
        cos_mat = torch.matmul(v, t.t())
        diag = torch.diagonal(cos_mat)
        if B > 1:
            mask = ~torch.eye(B, dtype=torch.bool, device=cos_mat.device)
            offdiag = cos_mat[mask]
            offdiag_mean = float(offdiag.mean().item()) if int(offdiag.numel()) > 0 else 0.0
        else:
            offdiag_mean = 0.0
        diag_mean = float(diag.mean().item()) if int(diag.numel()) > 0 else 0.0

    return {
        "loss": loss,
        "diag_cos_mean": diag_mean,
        "offdiag_cos_mean": offdiag_mean,
    }
