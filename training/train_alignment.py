import os
import inspect

import math
import numpy as np
import torch
import torch.nn.functional as F

from config import Config, FAULT_PROMPT_TEMPLATES, parse_fusion_decoder_grad_live_ratio
from losses.alignment_loss import (
    multimodal_alignment_loss,
    fusion_ce_loss,
    fusion_metric_learnable_proto_ce_loss,
    cross_domain_metric_centroid_loss,
    nf_contrastive_loss,
    nf_margin_loss,
    normal_reference_loss,
    LDAMLoss,
    PrototypeSeparationLoss,
)
from losses.prototype_anchor_loss import prototype_anchor_loss, text_etf_alignment_loss
from utils.simplex_etf import generate_simplex_etf, print_etf_stats
from losses.batch_nuclear_norm import batch_nuclear_norm_loss  # Att77新增
from losses.supcon_loss import supervised_contrastive_loss
from utils.ablation_gating import get_gate_bank, prepare_batch_spec
from losses.clip_alignment_loss import spec_clip_alignment_loss, clip_style_batch_contrastive_loss
from losses.moe_loss import compute_moe_loss, compute_moe_loss_bayesian
from losses.metric_loss import cosine_center_loss, update_ema_centers
from utils.bayesian_prototypes import build_bayesian_text_prototypes, extract_posterior_prototypes
from utils.checkpoint import resolve_readable_checkpoint, save_checkpoint
from utils.signal_augment import amplitude_scale
from utils.ema import EMAManager
from utils.uncertainty import compute_ece_loss, compute_temperature_scaling_loss
from zero_shot_eval.zero_shot_eval_hust import zero_shot_eval_hust
from models.proto_adapter import ProtoSpaceAdapter
from models.learnable_prototypes import LearnablePrototypes


def _maybe_add_fusion_metric_learnable_proto_ce(
    loss,
    loss_dict,
    cfg,
    learnable_proto_module,
    hierarchical_parts,
    labels,
    class_weights_for_loss,
    current_epoch,
    batch_idx,
):
    """
    ATT486 (S1): auxiliary CE on cos(z_metric, learnable_proto)/T vs labels.
    Requires `LearnablePrototypes` module + dual-tower `hierarchical_parts['z_metric']`.
    """
    try:
        _en = bool(getattr(cfg, "FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE", False))
        _w = float(getattr(cfg, "FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT", 0.0))
        _t = float(getattr(cfg, "FUSION_METRIC_LEARNABLE_PROTO_CE_TEMP", 0.10))
        _ep0 = int(getattr(cfg, "FUSION_METRIC_LEARNABLE_PROTO_CE_START_EPOCH", 1))
    except Exception:
        return loss, loss_dict
    if not _en or _w <= 0.0 or learnable_proto_module is None:
        return loss, loss_dict
    if int(current_epoch) < int(_ep0):
        return loss, loss_dict
    if not isinstance(hierarchical_parts, dict):
        return loss, loss_dict
    _zm = hierarchical_parts.get("z_metric", None)
    if not torch.is_tensor(_zm):
        return loss, loss_dict
    try:
        _P = learnable_proto_module.forward()
        _raw = fusion_metric_learnable_proto_ce_loss(_zm, _P, labels, _t, class_weights_for_loss)
        if _raw is None:
            return loss, loss_dict
        _wt = _w * _raw
        loss = loss + _wt
        if isinstance(loss_dict, dict):
            loss_dict["metric_lp_ce"] = float(_raw.detach().cpu().item())
            loss_dict["metric_lp_ce_weighted"] = float(_wt.detach().cpu().item())
            loss_dict["total"] = float(loss.detach().cpu().item())
    except Exception as _e:
        if int(batch_idx) == 0 and int(current_epoch) <= 2:
            print(f"[ATT486-metric_lp_ce] error: {_e}")
    return loss, loss_dict


def _fusion_zcls_ce_disabled_now(cfg, stage1_mode_now):
    """ATT488: skip dead z_cls CE during pretraining while leaving explicit stage2 calibration available."""
    try:
        _en = bool(getattr(cfg, "FUSION_PRETRAIN_DISABLE_ZCLS_CE", False))
        _stage1_only = bool(getattr(cfg, "FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY", True))
    except Exception:
        return False
    if not _en:
        return False
    return (not _stage1_only) or bool(stage1_mode_now)


def _mark_fusion_zcls_ce_skipped(loss_dict):
    """Keep raw CE visible as an _info_ diagnostic but remove it from loss-budget accounting."""
    if not isinstance(loss_dict, dict):
        return loss_dict
    try:
        _raw = float(loss_dict.get("fusion_cls", 0.0))
        loss_dict["_info_fusion_cls_raw_skipped"] = _raw
        loss_dict["_info_fusion_cls"] = 0.0
        loss_dict["fusion_cls"] = 0.0
        loss_dict["fusion_cls_skipped"] = 1.0
    except Exception:
        pass
    return loss_dict


def update_adaptive_class_weights(current_weights, class_accuracies, momentum=0.1, min_w=0.5, max_w=3.0):
    """
    根据每个类别的准确率自适应调整权重。
    
    Args:
        current_weights: 当前的类别权重 [C] torch.Tensor
        class_accuracies: 上一个阶段每个类别的准确率 [C] torch.Tensor (0-1之间)
        momentum: 权重更新的动量 (0-1之间，越大更新越快)
        min_w, max_w: 权重的边界限制
    
    Returns:
        new_weights: 更新后的类别权重 [C] torch.Tensor
    """
    device = current_weights.device
    
    # 计算误差率，误差越高权重应该越大
    error_rates = 1.0 - class_accuracies
    
    # 归一化误差率作为调整量（相对于平均误差率）
    mean_error = torch.mean(error_rates)
    if mean_error < 1e-8:
        # 如果所有类别都完美，保持原权重
        return current_weights
    
    # 计算调整因子：误差率高的类别应该增加权重
    adjustment = error_rates / (mean_error + 1e-8)
    
    # 平滑更新：new_weight = current_weight * (1 + momentum * (adjustment - 1))
    # 这样误差率高的类别权重会增加，误差率低的会减少
    new_weights = current_weights * (1.0 + momentum * (adjustment - 1.0))
    
    # 限制范围
    new_weights = torch.clamp(new_weights, min=min_w, max=max_w)
    
    # 重新归一化，保持均值为1（避免整体权重漂移）
    new_weights = new_weights / (new_weights.mean() + 1e-8)
    
    return new_weights


def log_moe_router_audit(cfg_mod, current_epoch: int, batch_idx: int, moe_stats) -> None:
    """Periodic MoE routing diagnostics (ATT470): entropy_std≈0 & stable top1_mass ⇒ frozen router."""
    try:
        if not bool(getattr(cfg_mod, "ATT470_MOE_ROUTE_DIAG_ENABLE", True)):
            return
    except Exception:
        return
    if moe_stats is None or batch_idx != 0:
        return
    if current_epoch != 1 and current_epoch % 10 != 0:
        return
    try:
        _ee = moe_stats.get("expert_entropies")
        _ew = moe_stats.get("expert_weights")
        if not _ee:
            return
        _vals = []
        for x in _ee:
            if torch.is_tensor(x):
                _vals.append(float(x.detach().mean().cpu()))
            else:
                _vals.append(float(x))
        _mean_e = sum(_vals) / len(_vals)
        _std_e = (sum((v - _mean_e) ** 2 for v in _vals) / len(_vals)) ** 0.5
        _tm = []
        if _ew:
            for w in _ew:
                if torch.is_tensor(w):
                    _tm.append(float(w.max(dim=-1).values.mean().detach().cpu()))
        _tm_mean = sum(_tm) / len(_tm) if _tm else float("nan")
        print(
            f"[MoE-RouterAudit] E{current_epoch} band_entropy_mean={_mean_e:.6f} "
            f"band_entropy_std={_std_e:.6f} expert_top1_mass_mean={_tm_mean:.4f}"
        )
    except Exception:
        pass


def compute_diag_loss_with_focal(
    logits,
    labels,
    class_weight_tensor,
    use_focal_loss,
    focal_gamma,
    focal_alpha,
    label_smoothing,
    use_balanced_softmax=False,
    class_counts=None,
):
    """
    Compute either focal or vanilla cross-entropy for diagnosis logits.
    """
    logits_for_ce = logits
    class_weight_for_ce = class_weight_tensor

    # Stage1 path also needs prior correction; otherwise USE_BALANCED_SOFTMAX becomes a dead switch.
    if use_balanced_softmax and class_counts is not None:
        try:
            counts = torch.as_tensor(class_counts, device=logits.device, dtype=logits.dtype).view(-1)
            if int(counts.numel()) == int(logits.size(1)):
                logits_for_ce = logits + torch.log(counts.clamp(min=1.0)).view(1, -1)
                # Avoid double compensation when balanced-softmax prior is active.
                class_weight_for_ce = None
        except Exception:
            pass

    if use_focal_loss:
        ce_loss = F.cross_entropy(
            logits_for_ce,
            labels,
            reduction='none',
            label_smoothing=label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** focal_gamma

        alpha_t = None
        if focal_alpha is not None:
            if isinstance(focal_alpha, (list, tuple, torch.Tensor)):
                alpha_tensor = torch.tensor(focal_alpha, device=labels.device, dtype=ce_loss.dtype)
                alpha_t = alpha_tensor[labels]
            else:
                alpha_t = focal_alpha
        elif class_weight_for_ce is not None:
            weight_tensor = class_weight_for_ce.to(labels.device)
            alpha_t = weight_tensor[labels]

        if alpha_t is not None:
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()

    return F.cross_entropy(
        logits_for_ce,
        labels,
        weight=class_weight_for_ce,
        label_smoothing=label_smoothing
    )


def _average_state_dict_group(checkpoints, group_key, ref_index):
    """
    Average floating-point tensors in one checkpoint group (e.g., signal_encoder),
    while keeping non-floating tensors/buffers from a reference checkpoint.
    """
    ref_group = checkpoints[ref_index][group_key]
    out = {}
    for name, ref_tensor in ref_group.items():
        if not torch.is_tensor(ref_tensor):
            out[name] = ref_tensor
            continue
        if not ref_tensor.is_floating_point():
            out[name] = ref_tensor.clone()
            continue
        acc = None
        valid = 0
        for ck in checkpoints:
            cur_group = ck.get(group_key, None)
            if cur_group is None:
                continue
            cur_tensor = cur_group.get(name, None)
            if (cur_tensor is None) or (not torch.is_tensor(cur_tensor)) or (cur_tensor.shape != ref_tensor.shape):
                continue
            cur_fp32 = cur_tensor.detach().to(torch.float32)
            acc = cur_fp32 if acc is None else (acc + cur_fp32)
            valid += 1
        if acc is None or valid <= 0:
            out[name] = ref_tensor.clone()
        else:
            out[name] = (acc / float(valid)).to(ref_tensor.dtype)
    return out


def _best_checkpoint_filename(epoch, checkpoint_tag=None):
    """
    Phase-safe best checkpoint naming.
    Prefer `<checkpoint_tag>_epoch_<N>_best.pth` to avoid collisions across phases.
    """
    if checkpoint_tag:
        return f"{checkpoint_tag}_epoch_{int(epoch)}_best.pth"
    return f"epoch_{int(epoch)}_best.pth"


def _prebest_checkpoint_filename(epoch, checkpoint_tag=None):
    if checkpoint_tag:
        return f"{checkpoint_tag}_epoch_{int(epoch)}_pre_best.pth"
    return f"epoch_{int(epoch)}_pre_best.pth"


def build_swad_checkpoint(checkpoint_dir, best_epoch, checkpoint_tag, swad_window=2, swad_min_models=3):
    """
    Build a SWAD-like averaged checkpoint around best epoch using existing checkpoints.
    Returns: (swad_path_or_none, used_epochs_list)
    """
    if best_epoch is None:
        return None, []
    try:
        best_epoch = int(best_epoch)
    except Exception:
        return None, []

    candidate_epochs = list(range(max(1, best_epoch - int(swad_window)), best_epoch + int(swad_window) + 1))
    candidate_paths = []
    used_epochs = []
    for ep in candidate_epochs:
        p_best = os.path.join(checkpoint_dir, _best_checkpoint_filename(ep, checkpoint_tag))
        p_best_legacy = os.path.join(checkpoint_dir, _best_checkpoint_filename(ep, None))
        p_periodic = os.path.join(checkpoint_dir, f"epoch_{ep}_{checkpoint_tag}.pth")
        if os.path.exists(p_best):
            p = p_best
        elif os.path.exists(p_best_legacy):
            p = p_best_legacy
        elif os.path.exists(p_periodic):
            p = p_periodic
        else:
            p = None
        if p is None:
            continue
        candidate_paths.append(p)
        used_epochs.append(ep)

    if len(candidate_paths) < int(swad_min_models):
        return None, used_epochs

    checkpoints = []
    for p in candidate_paths:
        checkpoints.append(torch.load(p, map_location="cpu", weights_only=False))

    # Prefer using best epoch as reference (keeps non-float buffers/prototypes stable).
    ref_index = 0
    if best_epoch in used_epochs:
        ref_index = used_epochs.index(best_epoch)
    ref_ckpt = checkpoints[ref_index]

    swad_ckpt = {
        "epoch": best_epoch,
        "loss": ref_ckpt.get("loss", None),
        "signal_encoder": _average_state_dict_group(checkpoints, "signal_encoder", ref_index),
        "spec_encoder": _average_state_dict_group(checkpoints, "spec_encoder", ref_index),
        "text_encoder": _average_state_dict_group(checkpoints, "text_encoder", ref_index),
        "fusion_module": _average_state_dict_group(checkpoints, "fusion_module", ref_index),
        "fusion_decoder": _average_state_dict_group(checkpoints, "fusion_decoder", ref_index),
        "optimizer": None,
        "text_prototypes": ref_ckpt.get("text_prototypes", None),
    }
    # Keep optional states from the reference checkpoint.
    if "expert_tracker_state" in ref_ckpt:
        swad_ckpt["expert_tracker_state"] = ref_ckpt["expert_tracker_state"]
    if "ema_state" in ref_ckpt:
        swad_ckpt["ema_state"] = ref_ckpt["ema_state"]

    swad_path = os.path.join(checkpoint_dir, f"epoch_{best_epoch}_swad.pth")
    torch.save(swad_ckpt, swad_path)
    return swad_path, used_epochs


def compute_confusion_penalty_weights(confusion_stats, threshold=0.3):
    """
    根据混淆统计计算 Confusion-Aware 惩罚权重。
    如果 Class 2/3 被大量混淆到 Class 1，增加它们的损失权重。
    
    Args:
        confusion_stats: dict, {true_class: {pred_class: count, ...}, ...}
        threshold: float, 混淆到 Class 1 的阈值（超过此阈值则应用惩罚）
    
    Returns:
        penalty_weights: torch.Tensor [3], 分别对应 Class 1, 2, 3 的惩罚权重（默认 1.0）
    """
    penalty_weights = torch.ones(3)  # [Class 1, Class 2, Class 3] 默认无惩罚
    
    # 检查 Class 2 (原始标签 2，在 fault 中为索引 1) 和 Class 3 (原始标签 3，在 fault 中为索引 2)
    for true_class in [2, 3]:  # 原始标签
        if true_class in confusion_stats:
            total_samples = sum(confusion_stats[true_class].values())
            if total_samples > 0:
                # 统计混淆到 Class 1 的比例
                confused_to_class1 = confusion_stats[true_class].get(1, 0)
                confusion_ratio = confused_to_class1 / total_samples
                
                # 如果混淆率超过阈值，应用惩罚
                if confusion_ratio > threshold:
                    # 惩罚权重 = 1.0 + 混淆率（最高可达 2.0）
                    penalty_weight = 1.0 + min(confusion_ratio, 1.0)
                    # Class 2 对应 fault 索引 1, Class 3 对应 fault 索引 2
                    fault_idx = true_class - 1  # 2->1, 3->2
                    if 0 <= fault_idx < 3:
                        penalty_weights[fault_idx] = penalty_weight
    
    return penalty_weights


def compute_decorr_loss(z, var_target=1.0, cov_weight=1.0, eps=1e-6):
    """
    De-collapse loss: encourage per-dim variance and reduce off-diagonal covariance.
    Returns (loss, var_mean, cov_abs_mean).
    """
    if z is None or (not torch.is_tensor(z)) or z.dim() != 2 or z.size(0) < 2:
        return None, 0.0, 0.0
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    denom = max(1, z.size(0) - 1)
    cov = (z.t() @ z) / float(denom)
    var = torch.diag(cov)
    var_loss = F.relu(float(var_target) - var).mean()
    off_diag = cov - torch.diag(var)
    cov_loss = (off_diag ** 2).mean()
    loss = var_loss + float(cov_weight) * cov_loss
    try:
        var_mean = float(var.mean().detach().cpu().item())
        cov_abs_mean = float(off_diag.abs().mean().detach().cpu().item())
    except Exception:
        var_mean, cov_abs_mean = 0.0, 0.0
    return loss, var_mean, cov_abs_mean


def compute_vicreg_varcov_loss(z, std_target=1.0, var_weight=1.0, cov_weight=1.0, eps=1e-6):
    """
    VICReg-style variance/covariance regularizer (single-view).
    Returns (loss, var_mean, cov_abs_mean).
    """
    if z is None or (not torch.is_tensor(z)) or z.dim() != 2 or z.size(0) < 2:
        return None, 0.0, 0.0
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    n = int(z.size(0))
    std = torch.sqrt(z.var(dim=0, unbiased=False) + float(eps))
    var_loss = F.relu(float(std_target) - std).mean()
    cov = (z.t() @ z) / float(max(1, n - 1))
    cov = cov - torch.diag(torch.diag(cov))
    cov_loss = (cov ** 2).sum() / float(max(1, z.size(1)))
    loss = float(var_weight) * var_loss + float(cov_weight) * cov_loss
    try:
        var_mean = float((std ** 2).mean().detach().cpu().item())
        cov_abs_mean = float(cov.abs().mean().detach().cpu().item())
    except Exception:
        var_mean, cov_abs_mean = 0.0, 0.0
    return loss, var_mean, cov_abs_mean


def compute_fusion_var_floor_loss(z, min_var=0.001, eps=1e-6):
    """Penalize batch-level mean feature variance falling below min_var."""
    if z is None or (not torch.is_tensor(z)) or z.dim() != 2 or z.size(0) < 2:
        return None, 0.0
    z = z.float()
    fusion_var = z.var(dim=0, unbiased=False).mean()
    loss = F.relu(float(min_var) - fusion_var)
    return loss, float(fusion_var.detach().cpu().item())


def compute_fusion_inter_cos_margin_loss(z, labels, target_max=0.15, min_samples_per_class=2, eps=1e-6):
    """
    Soft hinge on mean inter-class centroid cosine similarity.
    Returns (loss, inter_cos_mean, inter_cos_max).
    """
    if (
        z is None
        or labels is None
        or (not torch.is_tensor(z))
        or (not torch.is_tensor(labels))
        or z.dim() != 2
        or z.size(0) < 2
    ):
        return None, 0.0, 0.0
    z = F.normalize(z.float(), dim=-1, eps=float(eps))
    labels = labels.view(-1).long()
    centroids = []
    for cls_id in labels.unique(sorted=True):
        mask = labels == cls_id
        if int(mask.sum().item()) < int(min_samples_per_class):
            continue
        centroids.append(F.normalize(z[mask].mean(dim=0, keepdim=True), dim=-1, eps=float(eps)))
    if len(centroids) < 2:
        return None, 0.0, 0.0
    c = torch.cat(centroids, dim=0)
    cos_mat = c @ c.t()
    off_mask = ~torch.eye(c.size(0), dtype=torch.bool, device=cos_mat.device)
    off_diag = cos_mat[off_mask]
    inter_mean = off_diag.mean()
    inter_max = off_diag.max()
    loss = F.relu(inter_mean - float(target_max))
    return loss, float(inter_mean.detach().cpu().item()), float(inter_max.detach().cpu().item())


def compute_fishr_proxy_loss(
    logits,
    labels,
    domain_ids,
    repr_tensor,
    source_domain_ids=(0, 2),
    min_samples_per_domain=8,
    allow_intra_batch_split=True,
):
    """
    Fishr-style gradient variance alignment on representation gradients.
    ATT421: When only one source domain is present in a batch, performs
    intra-batch random splitting to create two pseudo-environments.
    Returns (loss, used_domains, avg_grad_var, used_split).
    """
    if (
        logits is None
        or (not torch.is_tensor(logits))
        or logits.dim() != 2
        or labels is None
        or (not torch.is_tensor(labels))
        or domain_ids is None
        or (not torch.is_tensor(domain_ids))
        or repr_tensor is None
        or (not torch.is_tensor(repr_tensor))
        or (not repr_tensor.requires_grad)
    ):
        return None, 0, 0.0, False

    dom_ids = domain_ids.long()
    grad_var_by_domain = []
    used_split = False
    for _dom in source_domain_ids:
        _mask = (dom_ids == int(_dom))
        _n = int(_mask.sum().item())
        if _n < int(min_samples_per_domain):
            continue
        _loss_dom = F.cross_entropy(logits[_mask].float(), labels[_mask].long(), reduction="mean")
        _grad = torch.autograd.grad(
            _loss_dom,
            repr_tensor,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        if _grad is None:
            continue
        _grad_dom = _grad[_mask]
        if int(_grad_dom.size(0)) < 2:
            continue
        grad_var_by_domain.append(_grad_dom.float().var(dim=0, unbiased=False))

    # ATT421: intra-batch random split fallback for single-domain phases
    if len(grad_var_by_domain) < 2 and allow_intra_batch_split:
        _all_src_mask = torch.zeros(dom_ids.size(0), dtype=torch.bool, device=dom_ids.device)
        for _dom in source_domain_ids:
            _all_src_mask |= (dom_ids == int(_dom))
        _n_src = int(_all_src_mask.sum().item())
        _min_half = max(int(min_samples_per_domain), 4)
        if _n_src >= 2 * _min_half:
            _src_indices = torch.where(_all_src_mask)[0]
            _perm = torch.randperm(_n_src, device=dom_ids.device)
            _half = _n_src // 2
            _idx_a = _src_indices[_perm[:_half]]
            _idx_b = _src_indices[_perm[_half:]]
            grad_var_by_domain = []
            for _idx_group in (_idx_a, _idx_b):
                _loss_g = F.cross_entropy(logits[_idx_group].float(), labels[_idx_group].long(), reduction="mean")
                _grad_g = torch.autograd.grad(
                    _loss_g, repr_tensor, retain_graph=True, create_graph=True, allow_unused=True
                )[0]
                if _grad_g is None:
                    continue
                _grad_sel = _grad_g[_idx_group]
                if int(_grad_sel.size(0)) < 2:
                    continue
                grad_var_by_domain.append(_grad_sel.float().var(dim=0, unbiased=False))
            if len(grad_var_by_domain) >= 2:
                used_split = True

    if len(grad_var_by_domain) < 2:
        return None, len(grad_var_by_domain), 0.0, False

    pair_losses = []
    for i in range(len(grad_var_by_domain)):
        for j in range(i + 1, len(grad_var_by_domain)):
            pair_losses.append(F.mse_loss(grad_var_by_domain[i], grad_var_by_domain[j], reduction="mean"))
    if len(pair_losses) == 0:
        return None, len(grad_var_by_domain), 0.0, False

    loss = torch.stack(pair_losses).mean()
    try:
        avg_grad_var = float(torch.stack([v.mean() for v in grad_var_by_domain]).mean().detach().cpu().item())
    except Exception:
        avg_grad_var = 0.0
    return loss, len(grad_var_by_domain), avg_grad_var, used_split


def extract_pre_l2_feature(z: torch.Tensor):
    """
    Best-effort extractor for pre-L2 normalized representation.
    Falls back to the input tensor if no cached pre-L2 feature exists.
    """
    if z is None or (not torch.is_tensor(z)):
        return None
    try:
        z_pre = getattr(z, "_pre_l2_feature", None)
        if torch.is_tensor(z_pre) and z_pre.shape == z.shape:
            return z_pre
    except Exception:
        pass
    return z


def compute_ccpa_loss(
    repr_tensor,
    labels,
    domain_ids,
    source_domain_ids=(0, 2),
    min_samples_per_domain_class=2,
):
    """
    Class-Conditional Prototype Alignment (CCPA) across source domains.
    Align per-class prototypes between domains on cosine geometry.
    Returns (loss, used_pairs, classes_used, mean_cos_sim).
    """
    if (
        repr_tensor is None
        or (not torch.is_tensor(repr_tensor))
        or repr_tensor.dim() != 2
        or labels is None
        or (not torch.is_tensor(labels))
        or domain_ids is None
        or (not torch.is_tensor(domain_ids))
    ):
        return None, 0, 0, 0.0

    z = F.normalize(repr_tensor.float(), dim=-1, eps=1e-6)
    y = labels.long()
    d = domain_ids.long()
    unique_classes = torch.unique(y)

    pair_terms = []
    cos_terms = []
    used_pairs = 0
    classes_used = 0
    min_n = int(max(1, min_samples_per_domain_class))

    for cls in unique_classes.tolist():
        protos = []
        for dom in source_domain_ids:
            m = (y == int(cls)) & (d == int(dom))
            if int(m.sum().item()) < min_n:
                continue
            p = z[m].mean(dim=0)
            protos.append(F.normalize(p, dim=-1, eps=1e-6))
        if len(protos) < 2:
            continue
        classes_used += 1
        for i in range(len(protos)):
            for j in range(i + 1, len(protos)):
                cos_ij = torch.sum(protos[i] * protos[j], dim=-1)
                pair_terms.append(1.0 - cos_ij)
                cos_terms.append(cos_ij.detach())
                used_pairs += 1

    if len(pair_terms) == 0:
        return None, 0, classes_used, 0.0

    loss = torch.stack(pair_terms).mean()
    try:
        mean_cos = float(torch.stack(cos_terms).mean().cpu().item()) if len(cos_terms) > 0 else 0.0
    except Exception:
        mean_cos = 0.0
    return loss, used_pairs, classes_used, mean_cos


def compute_uniformity_loss(z, t=2.0):
    """
    Uniformity regularizer on normalized embeddings (Wang & Isola style):
      L = log( E[ exp(-t * ||zi-zj||^2) ] )
    Lower is better (more uniformly spread embeddings on hypersphere).
    """
    if z is None or (not torch.is_tensor(z)) or z.dim() != 2 or z.size(0) < 3:
        return None, 0.0
    z = F.normalize(z.float(), dim=-1)
    d2 = torch.pdist(z, p=2).pow(2)
    if d2.numel() == 0:
        return None, 0.0
    loss = torch.log(torch.exp(-float(t) * d2).mean() + 1e-12)
    try:
        mean_d2 = float(d2.mean().detach().cpu().item())
    except Exception:
        mean_d2 = 0.0
    return loss, mean_d2


def build_text_prototypes(text_encoder, label_to_prompts, device):
    """
    label_to_prompts: Dict[int, List[str]]
    """
    text_encoder.eval()
    prototypes = {}
    with torch.no_grad():
        for label, prompts in label_to_prompts.items():
            embs = []
            for p in prompts:
                emb = text_encoder([p]).to(device)  # assume returns [1, D] or [K, D] if batchable
                embs.append(F.normalize(emb.squeeze(0), dim=0))
            proto = torch.stack(embs, dim=0).mean(dim=0)
            prototypes[label] = F.normalize(proto, dim=0)
    try:
        from utils.build_text_prototypes_main import apply_semantic_proto_ablation
        prototypes = apply_semantic_proto_ablation(prototypes)
    except Exception as exc:
        print(f"[SemanticProtoAblation] skipped in training builder: {exc}")
    return prototypes


def apply_prompt_tuned_prototypes(fusion_decoder, prototypes, detach_output=False):
    """
    Apply learnable class-wise prompt offsets from fusion_decoder (if available).
    """
    if prototypes is None:
        return None
    try:
        import config as _cfg_local
        if not bool(getattr(_cfg_local, "USE_TEXT_PROMPT_TUNING", False)):
            return prototypes
    except Exception:
        return prototypes
    if fusion_decoder is None or (not hasattr(fusion_decoder, "apply_text_prompt_tuning")):
        return prototypes
    try:
        return fusion_decoder.apply_text_prompt_tuning(
            prototypes,
            detach_output=bool(detach_output)
        )
    except Exception:
        return prototypes


def coral_covariance_alignment_loss(feat_a: torch.Tensor, feat_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    CORAL loss between two feature sets using covariance alignment.
    loss = ||Cov(A) - Cov(B)||_F^2 / (4 * d^2)
    """
    if feat_a is None or feat_b is None:
        device = feat_a.device if torch.is_tensor(feat_a) else (feat_b.device if torch.is_tensor(feat_b) else "cpu")
        return torch.tensor(0.0, device=device)
    if feat_a.dim() != 2 or feat_b.dim() != 2:
        return torch.tensor(0.0, device=feat_a.device)
    if feat_a.size(0) < 2 or feat_b.size(0) < 2:
        return torch.tensor(0.0, device=feat_a.device)

    # Normalize per-sample feature scale before covariance matching to avoid domain-scale explosion.
    feat_a = F.normalize(feat_a, dim=-1, eps=1e-6)
    feat_b = F.normalize(feat_b, dim=-1, eps=1e-6)
    a = feat_a - feat_a.mean(dim=0, keepdim=True)
    b = feat_b - feat_b.mean(dim=0, keepdim=True)
    cov_a = (a.t() @ a) / (float(max(1, a.size(0) - 1)) + float(eps))
    cov_b = (b.t() @ b) / (float(max(1, b.size(0) - 1)) + float(eps))
    # ATT195(A): Use Frobenius-sum scale (instead of mean) after L2-normalization.
    # This restores CORAL to a numerically meaningful magnitude for training.
    return torch.sum((cov_a - cov_b) ** 2)


def multi_kernel_mmd_loss(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    kernel_scales=(0.5, 1.0, 2.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    MK-MMD loss on normalized features.
    Returns a non-negative discrepancy estimate between source and target batches.
    """
    if feat_src is None or feat_tgt is None:
        device = feat_src.device if torch.is_tensor(feat_src) else (feat_tgt.device if torch.is_tensor(feat_tgt) else "cpu")
        return torch.tensor(0.0, device=device)
    if feat_src.dim() != 2 or feat_tgt.dim() != 2:
        return torch.tensor(0.0, device=feat_src.device)
    if feat_src.size(0) < 2 or feat_tgt.size(0) < 2:
        return torch.tensor(0.0, device=feat_src.device)

    x = F.normalize(feat_src.float(), dim=-1, eps=1e-6)
    y = F.normalize(feat_tgt.float(), dim=-1, eps=1e-6)

    d_xx = torch.cdist(x, x, p=2).pow(2)
    d_yy = torch.cdist(y, y, p=2).pow(2)
    d_xy = torch.cdist(x, y, p=2).pow(2)

    with torch.no_grad():
        sigma2 = torch.median(d_xy.detach().reshape(-1))
        sigma2 = torch.clamp(sigma2, min=float(eps))

    k_xx = torch.zeros_like(d_xx)
    k_yy = torch.zeros_like(d_yy)
    k_xy = torch.zeros_like(d_xy)
    for s in kernel_scales:
        scale = max(float(s), float(eps))
        gamma = 1.0 / (2.0 * sigma2 * scale)
        k_xx = k_xx + torch.exp(-gamma * d_xx)
        k_yy = k_yy + torch.exp(-gamma * d_yy)
        k_xy = k_xy + torch.exp(-gamma * d_xy)

    loss = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return torch.clamp(loss, min=0.0)


def encode_domain_ids(domains, conditions=None):
    """
    将域名字符串列表转换为整数ID，支持负载感知 DANN（ATT112）。

    Domain mapping (ATT112 load-aware, num_domains=5):
        CWRU gk2 (1HP)  → 0   ← DANN阳性对：gk2 vs gk4 的负载对抗
        CWRU gk4 (3HP)  → 1   ← 迫使 z_gate 对负载不变
        PU              → 2
        HUST            → 3
        HIT             → 4
    For non-CWRU or unknown CWRU condition: falls back to CWRU=0.

    Args:
        domains:    List[str], batch['domain'], e.g. ["CWRU", "CWRU", "PU"]
        conditions: Optional List[str], batch['condition'], e.g. ["gk2", "gk4", "gk6"]
                    When provided and domain=="CWRU", uses gk-group for fine-grained domain_id.
    """
    try:
        cwru_gk2_id = int(getattr(Config, "CWRU_GK2_DOMAIN_ID", 0))
        cwru_gk4_id = int(getattr(Config, "CWRU_GK4_DOMAIN_ID", 1))
        pu_id = int(getattr(Config, "PU_DOMAIN_ID", 2))
        hust_id = int(getattr(Config, "HUST_DOMAIN_ID", 3))
        hit_id = int(getattr(Config, "HIT_DOMAIN_ID", 4))
    except Exception:
        cwru_gk2_id, cwru_gk4_id, pu_id, hust_id, hit_id = 0, 1, 2, 3, 4

    ids = []
    for i, d in enumerate(domains):
        d_upper = str(d).upper()
        cond = str(conditions[i]).lower() if (conditions is not None and i < len(conditions)) else ""
        if d_upper == "CWRU":
            # Load-aware: gk4 (3HP) -> CWRU_GK4_DOMAIN_ID, others -> CWRU_GK2_DOMAIN_ID
            if "gk4" in cond or "gk_4" in cond or "hp3" in cond:
                ids.append(cwru_gk4_id)
            else:
                ids.append(cwru_gk2_id)
        elif d_upper == "PU":
            ids.append(pu_id)
        elif d_upper == "HUST":
            ids.append(hust_id)
        elif d_upper == "HIT":
            ids.append(hit_id)
        else:
            ids.append(cwru_gk2_id)
    return torch.tensor(ids, dtype=torch.long)


def compute_progressive_weight_scale(epoch, num_epochs, initial_scale=0.5, schedule='cosine'):
    """
    计算渐进式权重缩放因子
    
    Args:
        epoch: 当前epoch（从1开始）
        num_epochs: 总epoch数
        initial_scale: 初始权重缩放因子（0.5表示从50%权重开始）
        schedule: 调度方式，'cosine' 或 'linear'
    
    Returns:
        weight_scale: float，当前epoch的权重缩放因子（从initial_scale到1.0）
    """
    if schedule == 'cosine':
        # 余弦调度：从initial_scale平滑增长到1.0
        progress = epoch / num_epochs
        weight_scale = initial_scale + (1.0 - initial_scale) * (1.0 - math.cos(math.pi * progress / 2.0))
    elif schedule == 'linear':
        # 线性调度：从initial_scale线性增长到1.0
        progress = epoch / num_epochs
        weight_scale = initial_scale + (1.0 - initial_scale) * progress
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return weight_scale


def compute_dann_alpha(current_epoch: int, num_epochs: int, start_epoch: int, max_alpha: float = 1.0, schedule: str = "linear") -> float:
    """
    Compute GRL alpha for DANN.
    Typical practice is to increase alpha from 0 -> max_alpha during training for stability.
    """
    try:
        e = int(current_epoch)
        n = int(num_epochs)
        s = int(start_epoch)
        ma = float(max_alpha)
    except Exception:
        return 0.0
    if n <= 0:
        return 0.0
    if e < s:
        return 0.0
    ma = max(0.0, ma)
    sch = str(schedule or "linear").lower()
    if sch == "constant":
        return ma
    # linear warmup over remaining epochs
    denom = max(1, (n - s + 1))
    prog = float(e - s + 1) / float(denom)
    prog = max(0.0, min(1.0, prog))
    return ma * prog


def build_allowed_diag_mask(domain_ids: torch.Tensor | None, num_classes: int, device) -> torch.Tensor | None:
    """
    Build per-sample allowed class mask for A2 partial-label loss.
    Current assumption (source/target dataset definitions):
      - PU (id=1): no rolling (class 3)
      - HIT (id=3): no rolling (class 3)
    For unknown domains, do NOT mask (allow all classes).
    """
    if domain_ids is None:
        return None
    if num_classes <= 0:
        return None
    try:
        domain_ids = domain_ids.to(device=device)
        B = int(domain_ids.numel())
        mask = torch.ones(B, int(num_classes), device=device, dtype=torch.bool)
        if int(num_classes) >= 4:
            pu_id = int(getattr(Config, "PU_DOMAIN_ID", 2))
            hit_id = int(getattr(Config, "HIT_DOMAIN_ID", 4))
            no_roll = (domain_ids == pu_id) | (domain_ids == hit_id)
            if no_roll.any():
                mask = mask.clone()
                mask[no_roll, 3] = False
        return mask
    except Exception:
        return None


def compute_class_metrics(y_true, y_pred, num_classes):
    """
    计算类别级别的性能指标：准确率、召回率、F1分数
    
    Args:
        y_true: [N] 真实标签
        y_pred: [N] 预测标签
        num_classes: 类别数
    
    Returns:
        metrics: dict，包含每个类别的准确率、召回率、F1分数
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[int(t), int(p)] += 1
    
    metrics = {}
    for c in range(num_classes):
        tp = confusion_matrix[c, c]  # True Positive
        fp = confusion_matrix[:, c].sum() - tp  # False Positive
        fn = confusion_matrix[c, :].sum() - tp  # False Negative
        tn = confusion_matrix.sum() - tp - fp - fn  # True Negative
        
        # 准确率（Precision）
        precision = tp / (tp + fp + 1e-8)
        
        # 召回率（Recall）
        recall = tp / (tp + fn + 1e-8)
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # 支持数（样本数）
        support = tp + fn
        
        metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    # 计算宏平均和加权平均
    precisions = [metrics[c]['precision'] for c in range(num_classes)]
    recalls = [metrics[c]['recall'] for c in range(num_classes)]
    f1s = [metrics[c]['f1'] for c in range(num_classes)]
    supports = [metrics[c]['support'] for c in range(num_classes)]
    total_support = sum(supports)
    
    metrics['macro'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }
    
    metrics['weighted'] = {
        'precision': sum(p * s for p, s in zip(precisions, supports)) / (total_support + 1e-8),
        'recall': sum(r * s for r, s in zip(recalls, supports)) / (total_support + 1e-8),
        'f1': sum(f * s for f, s in zip(f1s, supports)) / (total_support + 1e-8)
    }
    
    return metrics


def evaluate_on_val_loader(
        val_loader,
        signal_encoder,
        spec_encoder,
        fusion_module,
        fusion_decoder,
        text_encoder,
        device,
        precomputed_class_weights=None,
        use_focal_loss=False,
        focal_gamma=2.0,
        focal_alpha=None,
        label_smoothing=0.0,
        use_moe=False,
        text_prototypes=None,
        wst_encoder=None,
        _wst_bypass_active=False,
        env_encoder=None,
        _env_bypass_active=False,
        scd_encoder=None,
        _scd_bypass_active=False,
):
    """
    在验证集上评估模型性能，用于early-stop（不使用目标域）
    
    Returns:
        val_metrics: dict with keys ["val_loss", "val_acc", "val_spec_head_acc", "val_fusion_diag_acc"]
        val_spec_head_acc: spec_head MLP 独立分类准确率 (ATT132 P0)
        val_fusion_diag_acc: fusion_decoder logits_diag 路径准确率，与 val_acc 数值相同 (ATT475 命名)
    """
    _prev_modes = {
        "signal": signal_encoder.training,
        "spec": spec_encoder.training,
        "fusion": fusion_module.training,
        "decoder": fusion_decoder.training,
        "text": text_encoder.training,
    }
    signal_encoder.eval()
    spec_encoder.eval()
    if _wst_bypass_active and wst_encoder is not None:
        wst_encoder.eval()
    if _env_bypass_active and env_encoder is not None:
        env_encoder.eval()
    if _scd_bypass_active and scd_encoder is not None:
        scd_encoder.eval()
    fusion_module.eval()
    fusion_decoder.eval()
    text_encoder.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    # ATT132 P0: spec_head 独立路径准确率（与 fusion decoder 路径分离）
    spec_head_correct = 0
    spec_head_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch['signal'].to(device)
            specs = prepare_batch_spec(batch, device, spec_encoder, Config)
            signal_mask = batch['signal_mask'].to(device)
            spec_mask = batch['spec_mask'].to(device)
            labels = batch['label'].to(device)
            domains = batch.get('domain', None)
            conditions = batch.get('condition', None)
            _val_stat_vec = batch.get('stat_vec', None)
            if _val_stat_vec is not None:
                _val_stat_vec = _val_stat_vec.to(device)
            # ATT446/ATT449: bypass features in validation
            _val_z_wst = None
            if _wst_bypass_active:
                _val_wst_raw = batch.get("wst_spec", None)
                if _val_wst_raw is not None:
                    _val_z_wst = wst_encoder(_val_wst_raw.to(device))
            _val_z_env = None
            _val_ratio = None
            if _env_bypass_active:
                _val_env_raw = batch.get("env_spec", None)
                if _val_env_raw is not None:
                    _val_z_env = env_encoder(_val_env_raw.to(device))
                _val_ratio_raw = batch.get("bpfi_bpfo_ratio", None)
                if _val_ratio_raw is not None:
                    _val_ratio = _val_ratio_raw.to(device)
            _val_z_scd = None
            if _scd_bypass_active:
                _val_scd_raw = batch.get("scd_spec", None)
                if _val_scd_raw is not None:
                    _val_z_scd = scd_encoder(_val_scd_raw.to(device))
            
            # 编码domain_ids（用于域残差和 DANN）
            # ATT112: 传入 conditions 以启用负载感知 DANN（gk2→0, gk4→1, PU→2, ...）
            domain_ids = None
            if domains is not None:
                domain_ids = encode_domain_ids(domains, conditions=conditions).to(device)
            
            # 前向传播（支持分类头）
            use_signal_classification_head = getattr(signal_encoder, 'use_classification_head', False)
            if hasattr(signal_encoder, 'use_domain_residual') and signal_encoder.use_domain_residual:
                if use_signal_classification_head:
                    signal_feat, signal_logits = signal_encoder(signals, signal_mask, domain_ids=domain_ids, return_logits=True)
                else:
                    signal_feat = signal_encoder(signals, signal_mask, domain_ids=domain_ids)
                    signal_logits = None
            else:
                if use_signal_classification_head:
                    signal_feat, signal_logits = signal_encoder(signals, signal_mask, return_logits=True)
                else:
                    signal_feat = signal_encoder(signals, signal_mask)
                    signal_logits = None
            
            # 处理spec编码（支持MoE和域残差）
            is_moe = use_moe and hasattr(spec_encoder, 'num_bands')
            use_spec_classification_head = getattr(spec_encoder, 'use_classification_head', False)
            if is_moe:
                # MoE版本 — ATT449-BUGFIX: 传递bypass特征
                _val_moe_bypass = {}
                if _val_z_wst is not None:
                    _val_moe_bypass["wst_aux"] = _val_z_wst
                if _val_z_env is not None:
                    _val_moe_bypass["env_aux"] = _val_z_env
                if _val_ratio is not None:
                    _val_moe_bypass["ratio_aux"] = _val_ratio
                if _val_z_scd is not None:
                    _val_moe_bypass["scd_aux"] = _val_z_scd
                if use_spec_classification_head:
                    z_spec, moe_stats, spec_logits = spec_encoder(
                        specs, spec_mask,
                        domain_ids=domain_ids,
                        return_moe_stats=True,
                        return_logits=True,
                        stat_vec=_val_stat_vec,
                        **_val_moe_bypass,
                    )
                else:
                    z_spec, moe_stats = spec_encoder(
                        specs, spec_mask,
                        domain_ids=domain_ids,
                        return_moe_stats=True,
                        stat_vec=_val_stat_vec,
                        **_val_moe_bypass,
                    )
                    spec_logits = None
                spec_feat = z_spec
            else:
                spec_kwargs = {"spec": specs, "spec_mask": spec_mask, "domain_ids": domain_ids, "stat_vec": _val_stat_vec}
                if _val_z_wst is not None:
                    spec_kwargs["wst_aux"] = _val_z_wst
                if _val_z_env is not None:
                    spec_kwargs["env_aux"] = _val_z_env
                if _val_ratio is not None:
                    spec_kwargs["ratio_aux"] = _val_ratio
                if _val_z_scd is not None:
                    spec_kwargs["scd_aux"] = _val_z_scd
                if use_spec_classification_head:
                    spec_feat, spec_logits = spec_encoder(**spec_kwargs, return_logits=True)
                else:
                    spec_feat = spec_encoder(**spec_kwargs)
                    spec_logits = None
            
            # 提取MoE expert disagreement（如果可用）
            spec_moe_disagreement = None
            if is_moe and moe_stats is not None:
                if 'expert_disagreements' in moe_stats and len(moe_stats['expert_disagreements']) > 0:
                    # moe_stats['expert_disagreements'] 是 List of [B] (每个band一个)
                    # 聚合：取所有band的disagreement的平均值
                    disagreements_list = moe_stats['expert_disagreements']  # List of [B]
                    disagreements_tensor = torch.stack(disagreements_list, dim=0)  # [num_bands, B]
                    spec_moe_disagreement = disagreements_tensor.mean(dim=0)  # [B] (跨所有bands平均)
            
            # Fusion - 根据fusion_module类型使用正确的参数
            _dgl_detach_val = bool(getattr(Config, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
            _fus_fw_params = inspect.signature(fusion_module.forward).parameters
            _fus_detach_kw = {}
            if "dgl_detach_inputs" in _fus_fw_params:
                _fus_detach_kw["dgl_detach_inputs"] = _dgl_detach_val

            if hasattr(fusion_module, 'domain_embedding'):
                # DomainAwareCrossAttentionFusion
                z_fusion = fusion_module(
                    signal_feat,
                    spec_feat,
                    domain_ids=domain_ids,
                    text_prototypes=text_prototypes,
                    class_ids=None,
                    class_probs=None,
                    signal_logits=signal_logits,
                    spec_logits=spec_logits,
                    spec_moe_disagreement=spec_moe_disagreement,  # 传递MoE disagreement
                    **_fus_detach_kw,
                )
            else:
                # CrossGatedFusion或其他类型（不支持logits参数）
                if "dgl_detach_inputs" in _fus_fw_params:
                    z_fusion = fusion_module(signal_feat, spec_feat, dgl_detach_inputs=_dgl_detach_val)
                else:
                    z_fusion = fusion_module(signal_feat, spec_feat)
                
            # SOTA Fix: Integrate MoE Entropy Loss (if available)
            moe_entropy_loss = 0.0
            if is_moe and moe_stats is not None and 'entropy_loss' in moe_stats:
                # Sum entropy loss across bands
                moe_entropy_loss = moe_stats['entropy_loss']
                # Add to total loss with weight
                w_entropy = float(getattr(Config, 'MOE_ENTROPY_LOSS_WEIGHT', 0.1))
                # Note: moe_entropy_loss from SpecEncoderMoE is already a summation or mean tensor
                # We simply multiply by weight.
                moe_entropy_loss = w_entropy * moe_entropy_loss
            
            # ATT443: collect z_fusion stats for variance monitoring
            with torch.no_grad():
                _fus_norm = z_fusion.norm(dim=-1).cpu()
                _fus_std = z_fusion.std(dim=-1).cpu()
                if not hasattr(spec_encoder, '_val_fus_norms'):
                    spec_encoder._val_fus_norms = [_fus_norm]
                    spec_encoder._val_fus_stds = [_fus_std]
                else:
                    spec_encoder._val_fus_norms.append(_fus_norm)
                    spec_encoder._val_fus_stds.append(_fus_std)

            # Decoder
            logits4, parts = fusion_decoder(z_fusion, domain_ids=domain_ids, text_prototypes=text_prototypes, return_parts=True)
            # Two-stage training uses supervised diagnosis head logits as the main logits.
            logits_diag = parts.get("logits_diag_linear", None)
            if logits_diag is None:
                logits_diag = parts.get("logits_diag", logits4)

            # 计算诊断损失（HIT 使用 masked CE）
            diag_labels = labels.clone()
            device = logits_diag.device
            hit_domain_id = Config.HIT_DOMAIN_ID if hasattr(Config, 'HIT_DOMAIN_ID') else 4
            hit_mask = None
            if domain_ids is not None:
                hit_mask = domain_ids == hit_domain_id
            else:
                hit_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
            other_mask = ~hit_mask

            diag_loss_sum = 0.0
            diag_sample_count = 0

            def _accumulate_diag_loss(mask, num_classes, is_hit=False):
                if mask.sum().item() == 0:
                    return 0, 0.0
                subset_logits = logits_diag[mask][:, :num_classes]
                subset_labels = diag_labels[mask]
                if is_hit:
                    subset_labels = torch.clamp(subset_labels, max=num_classes - 1)
                if precomputed_class_weights is not None:
                    if precomputed_class_weights.shape[0] >= num_classes:
                        weight_slice = precomputed_class_weights[:num_classes]
                    else:
                        pad_val = precomputed_class_weights.mean()
                        weight_slice = torch.cat([
                            precomputed_class_weights,
                            pad_val.expand(num_classes - precomputed_class_weights.shape[0])
                        ])
                else:
                    weight_slice = None
                loss_value = compute_diag_loss_with_focal(
                    subset_logits,
                    subset_labels,
                    weight_slice,
                    use_focal_loss,
                    focal_gamma,
                    focal_alpha,
                    label_smoothing
                )
                count = subset_logits.size(0)
                return count, loss_value * count

            hit_classes = min(3, logits_diag.size(1))
            hit_count, hit_loss_sum = _accumulate_diag_loss(hit_mask, hit_classes, is_hit=True)
            other_count, other_loss_sum = _accumulate_diag_loss(other_mask, logits_diag.size(1), is_hit=False)

            diag_sample_count = hit_count + other_count
            if diag_sample_count == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                loss = (hit_loss_sum + other_loss_sum) / diag_sample_count
            
            # ===== (MTFD) Multi-Target Fault Decomposition Loss =====
            # Independent losses for NF, IO, and Rolling tasks to prevent collapse
            logits_mtfd_nf = parts.get("logits_mtfd_nf", None)
            logits_mtfd_io = parts.get("logits_mtfd_io", None)
            logits_mtfd_roll = parts.get("logits_mtfd_roll", None)
            
            mtfd_loss_sum = 0.0
            
            if logits_mtfd_nf is not None:
                # 1. NF Binary Loss (Normal vs Fault)
                # Targets: 0 -> 0 (Normal), 1/2/3 -> 1 (Fault)
                nf_targets = (labels > 0).long()
                w_nf = float(getattr(Config, "MTFD_NF_LOSS_WEIGHT", 1.0))
                l_nf = F.cross_entropy(logits_mtfd_nf, nf_targets)
                mtfd_loss_sum += w_nf * l_nf
                
            if logits_mtfd_io is not None:
                # 2. IO Binary Loss (Inner vs Outer) - Only on Inner(1)/Outer(2) samples
                # Targets: 1 -> 0, 2 -> 1
                io_mask = (labels == 1) | (labels == 2)
                if io_mask.any():
                    io_logits = logits_mtfd_io[io_mask]
                    io_targets = (labels[io_mask] - 1).long() # 1->0, 2->1
                    w_io = float(getattr(Config, "MTFD_IO_LOSS_WEIGHT", 1.0))
                    l_io = F.cross_entropy(io_logits, io_targets)
                    mtfd_loss_sum += w_io * l_io
                    
            if logits_mtfd_roll is not None:
                # 3. Rolling Binary Loss (Non-Roll vs Roll) - Only on Fault samples (1,2,3)
                # Targets: 1/2 -> 0 (Non-Roll), 3 -> 1 (Roll)
                # Note: We train this on ALL faults to learn "not rolling" semantics
                roll_mask = (labels > 0)
                if roll_mask.any():
                    roll_logits = logits_mtfd_roll[roll_mask]
                    roll_targets = (labels[roll_mask] == 3).long()
                    w_roll = float(getattr(Config, "MTFD_ROLLING_LOSS_WEIGHT", 2.0))
                    # (Attempt 13) Use log-odds + BCE with pos_weight to counter extreme rolling imbalance.
                    n_pos = roll_targets.sum().float()
                    n_neg = (roll_targets == 0).sum().float()
                    if n_pos < 1 or n_neg < 1:
                        l_roll = F.cross_entropy(roll_logits, roll_targets)
                    else:
                        pw_max = float(getattr(Config, "SPLIT_FAULT_ROLL_POS_WEIGHT_MAX", 20.0))
                        pos_weight = (n_neg / n_pos).clamp(max=pw_max)
                        roll_logit_diff = roll_logits[:, 1] - roll_logits[:, 0]  # log-odds
                        l_roll = F.binary_cross_entropy_with_logits(
                            roll_logit_diff,
                            roll_targets.float(),
                            pos_weight=pos_weight
                        )
                    mtfd_loss_sum += w_roll * l_roll

            total_loss += loss.item() * signals.size(0)
            # Add MTFD loss to backprop (loss tensor is accumulated in 'loss' variable? No, total_loss is float accumulator)
            # Wait, 'loss' is the tensor used for backward later?
            # Looking at context: total_loss is accumulator for logging.
            # I need to find where backward() is called.
            # It seems 'loss' variable holds the current batch loss tensor.
            # Add MTFD loss to backprop (loss tensor is accumulated in 'loss' variable? No, total_loss is float accumulator)
            loss = loss + mtfd_loss_sum
            
            # Add MoE Entropy Loss
            if isinstance(moe_entropy_loss, torch.Tensor):
                 loss = loss + moe_entropy_loss
                 total_loss += moe_entropy_loss.item() * signals.size(0)

            total_loss += mtfd_loss_sum.item() * signals.size(0) if isinstance(mtfd_loss_sum, torch.Tensor) else 0.0
            
            # 计算准确率（fusion decoder 路径）
            _, predicted = torch.max(logits_diag, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # ATT453: collect z_spec for class centroid diagnostics
            if spec_feat is not None and labels is not None:
                _sf_det = spec_feat.detach()
                for _ci in range(int(labels.max().item()) + 1):
                    _m = (labels == _ci)
                    if _m.any():
                        if not hasattr(spec_encoder, '_val_zspec_sum'):
                            spec_encoder._val_zspec_sum = {}
                            spec_encoder._val_zspec_cnt = {}
                        if _ci not in spec_encoder._val_zspec_sum:
                            spec_encoder._val_zspec_sum[_ci] = torch.zeros_like(_sf_det[0])
                            spec_encoder._val_zspec_cnt[_ci] = 0
                        spec_encoder._val_zspec_sum[_ci] += _sf_det[_m].sum(dim=0)
                        spec_encoder._val_zspec_cnt[_ci] += int(_m.sum().item())

            # ATT132 P0: spec_head 独立路径准确率
            if spec_logits is not None:
                _, spec_predicted = torch.max(spec_logits, 1)
                spec_head_correct += (spec_predicted == labels).sum().item()
                spec_head_total += labels.size(0)
                # ATT442: collect labels and preds for per-class diagnostic
                if not hasattr(spec_encoder, '_sh_labels'):
                    spec_encoder._sh_labels = labels.cpu()
                    spec_encoder._sh_preds = spec_predicted.cpu()
                else:
                    spec_encoder._sh_labels = torch.cat([spec_encoder._sh_labels, labels.cpu()])
                    spec_encoder._sh_preds = torch.cat([spec_encoder._sh_preds, spec_predicted.cpu()])
    
    avg_loss = total_loss / total
    avg_acc = correct / total
    avg_spec_head_acc = spec_head_correct / max(1, spec_head_total) if spec_head_total > 0 else float('nan')

    # ATT443: z_fusion variance monitoring
    if hasattr(spec_encoder, '_val_fus_norms') and len(spec_encoder._val_fus_norms) > 0:
        _norms = torch.cat(spec_encoder._val_fus_norms)
        _stds = torch.cat(spec_encoder._val_fus_stds)
        print(f"[ATT443-ValFusion] n={_norms.numel()} norm(fus)={_norms.mean():.4f}±{_norms.std():.4f} "
              f"std(fus)={_stds.mean():.4f}±{_stds.std():.4f}")
        del spec_encoder._val_fus_norms, spec_encoder._val_fus_stds

    # ATT442: per-class SpecHead diagnostic on validation set
    if spec_head_total > 0 and hasattr(spec_encoder, '_sh_labels') and hasattr(spec_encoder, '_sh_preds'):
        _sh_l = spec_encoder._sh_labels
        _sh_p = spec_encoder._sh_preds
        _n_classes = max(_sh_l.max().item(), _sh_p.max().item()) + 1
        _cls_names = ["Normal", "Inner", "Outer", "Ball"][:_n_classes]
        _parts = []
        for _ci in range(_n_classes):
            _mask_ci = (_sh_l == _ci)
            _ni = _mask_ci.sum().item()
            _corr_i = ((_sh_p[_mask_ci] == _ci).sum().item()) if _ni > 0 else 0
            _ri = _corr_i / _ni if _ni > 0 else 0.0
            _parts.append(f"{_cls_names[_ci] if _ci < len(_cls_names) else _ci}(R={_ri:.2f},N={_ni})")
        print(f"[ATT442-ValSpecHead] n={spec_head_total} acc={avg_spec_head_acc:.4f} "
              f"per_class: {', '.join(_parts)}")
        del spec_encoder._sh_labels, spec_encoder._sh_preds

    # ATT453: z_spec class centroid cosine diagnostics
    _zspec_centroid_cos = {}
    if hasattr(spec_encoder, '_val_zspec_sum') and len(spec_encoder._val_zspec_sum) > 0:
        _centroids = {}
        for _ci, _s in spec_encoder._val_zspec_sum.items():
            _cnt = spec_encoder._val_zspec_cnt[_ci]
            if _cnt > 0:
                _centroids[_ci] = F.normalize(_s / _cnt, dim=0)
        if len(_centroids) >= 2:
            _cids = sorted(_centroids.keys())
            _cos_parts = []
            for _i, _ci in enumerate(_cids):
                for _cj in _cids[_i + 1:]:
                    _cos_val = float(torch.dot(_centroids[_ci], _centroids[_cj]))
                    _cos_parts.append(f"({_ci},{_cj})={_cos_val:.4f}")
                    _zspec_centroid_cos[(_ci, _cj)] = _cos_val
            print(f"[ATT453-ValZSpec] centroid_cos=[{', '.join(_cos_parts)}]")
        del spec_encoder._val_zspec_sum, spec_encoder._val_zspec_cnt

    # ATT378c-FIX: restore training mode for all models
    if _prev_modes["signal"]:
        signal_encoder.train()
    if _prev_modes["spec"]:
        spec_encoder.train()
    if _prev_modes["fusion"]:
        fusion_module.train()
    if _prev_modes["decoder"]:
        fusion_decoder.train()
    if _prev_modes["text"]:
        text_encoder.train()
    if _wst_bypass_active and wst_encoder is not None:
        wst_encoder.train()
    if _env_bypass_active and env_encoder is not None:
        env_encoder.train()
    
    return {
        "val_loss": avg_loss,
        "val_acc": avg_acc,
        "val_spec_head_acc": avg_spec_head_acc,  # ATT132 P0: spec_head 路径独立指标
        "val_fusion_diag_acc": avg_acc,  # ATT475: 与 logits_diag 路径一致，命名上与 Spec contract 对齐对照
    }


def train_alignment(
        train_loader,
        signal_encoder,
        spec_encoder,
        text_encoder,
        fusion_module,
        fusion_decoder,
        optimizer,
        device,
        clip_model=None,
        lambda_clip=0.0,
        start_epoch=0,
        num_epochs=5,
        log_interval=1,
        hust_loader=None,                # 新增，用于可选的监控（不参与early-stop）
        hit_loader=None,                 # 新增，用于可选的监控（不参与early-stop）
        val_loader=None,                 # 新增，用于early-stop的源域验证集
        build_text_prototypes_fn=None,   # 新增，用于生成 text_prototypes
        fault_prompt_templates=None,     # 新增，用于生成 text_prototypes
        test_interval=5,                # 每几轮测试一次
        use_moe=False,                  # 是否使用MoE
        moe_loss_weight=0.01,           # MoE损失权重
        use_bayesian_prototypes=False,  # 是否使用贝叶斯原型更新
        bayesian_prior_strength=1.0,    # 贝叶斯先验强度
        bayesian_update_until_epoch=10, # 贝叶斯原型更新到第几轮（含），之后冻结不再更新
        enable_finetuning=True,        # 是否在验证阶段启用在线微调
        finetune_samples=50,            # 微调样本数
        finetune_lr=1e-4,               # 微调学习率
        finetune_steps=3,               # 微调步数
        use_consistency_regularization=False,  # 是否使用一致性正则
        consistency_weight=0.1,          # 一致性损失权重
        consistency_temperature=0.07,    # 一致性损失温度参数
        use_amp=False,                  # 是否使用混合精度训练（AMP）
        precomputed_class_weights=None,  # 预先计算的class_weights（避免每个epoch重复计算）
        # Schedule 和 Early stopping 配置
        use_schedule=True,               # 是否启用二段式 schedule
        schedule_epoch_threshold=5,      # schedule 切换的 epoch 阈值
        w_fusion_cls_early=0.1,          # 前N轮的 w_fusion_cls 权重
        w_fusion_cls_late=0.05,          # N轮后的 w_fusion_cls 权重
        lr_decay_factor=0.5,             # 学习率衰减因子
        use_early_stopping=True,         # 是否启用 early stopping
        early_stop_patience=2,           # early stopping 耐心值
        early_stop_metric="val_loss",    # early stopping 使用的指标（改为val_loss，不再使用目标域）
        early_stop_min_epochs=10,        # 至少训练N个epoch后才允许early stop（避免过早停止）
        early_stop_skip_prewindow_baseline=False,  # ATT168: True时在min_epochs处丢弃pre-window最优基线，从inf重新追踪（用于Task2消除E2噪声谷）
        checkpoint_dir=Config.CHECKPOINT_DIR,    # checkpoint 保存目录
        checkpoint_tag="withNewSpecEncoder_x2",  # checkpoint 文件名标签：epoch_{k}_{tag}.pth
        # 样本不均衡处理参数
        use_progressive_weights=True,    # 是否使用渐进式权重调整
        initial_weight_scale=0.5,        # 初始权重缩放因子（0.5表示从50%权重开始）
        weight_schedule='cosine',        # 权重调度方式：'cosine' 或 'linear'
        use_focal_loss=False,            # 是否使用Focal Loss
        focal_gamma=2.0,                 # Focal Loss的gamma参数
        focal_alpha=None,                # Focal Loss的alpha参数（None表示使用class_weights）
        label_smoothing=0.0,             # Label Smoothing系数
        enable_class_monitoring=True,    # 是否启用类别级性能监控
        # 类别权重策略配置
        class_weight_strategy="fixed",  # "fixed" | "adaptive" | "hybrid"
        class_weight_fixed_epochs=5,       # hybrid模式下，前N个epoch使用固定权重
        # sampler-first: if True, allow class_weights to participate in CE/Focal; otherwise use uniform loss weights
        use_class_weights_in_loss=True,
        # 渐进式学习率和类感知正则化参数
        use_progressive_lr=False,        # 是否使用渐进式学习率
        progressive_lr_schedule=None,    # 渐进式学习率调度表
        fusion_class_aware_regularization=0.0,  # 类感知权重的L2正则化系数
        # EMA 权重平均参数
        use_ema=False,                   # 是否使用 EMA 权重平均
        ema_decay=0.9999,                # EMA 衰减率
        ema_warmup_steps=1000,           # EMA 预热步数
        ema_update_after_step=100,       # 在此步数之后才开始更新 EMA
        # 校准损失参数（新增）
        use_calibration_loss=False,      # 是否使用校准损失（ECE）
        calibration_loss_weight=0.01,    # 校准损失权重
        calibration_n_bins=15,           # ECE分箱数量
        # 正交性正则化参数（新增）
        use_orthogonality_reg=False,     # 是否使用正交性正则化（Barlow-Twins风格）
        orthogonality_reg_weight=0.001,   # 正交性正则化权重
        # 元学习参数（新增）
        use_meta_learning=False,         # 是否启用元学习
        meta_learning_start_epoch=5,     # 从第几个epoch开始元学习
        meta_learning_interval=2,        # 每N个epoch执行一次元学习
        meta_lr=1e-4,                    # 元学习率
        meta_inner_lr=1e-3,              # 内循环学习率
        meta_inner_steps=3,              # 内循环步数
        meta_first_order=True,           # 是否使用一阶近似
        **kwargs                         # 接收其他额外参数
):
    """
    Train signal/spec encoders to align with text embeddings.
    Early-stop now uses source-domain validation set (val_loader) to avoid target-domain leakage.
    Performs optional monitoring on HUST/HIT datasets every `test_interval` epochs (not used for early-stop).

    Returns:
        history: dict with keys ["total", "sig_inst", "spec_inst", "sig_proto", "spec_proto",
                                  "signal_zero_shot_acc", "spec_zero_shot_acc", "val_loss", "val_acc"]
    """
    # ATT446: extract wst_encoder from kwargs
    wst_encoder = kwargs.pop("wst_encoder", None)
    _wst_bypass_active = wst_encoder is not None
    # ATT449: extract env_encoder from kwargs
    env_encoder = kwargs.pop("env_encoder", None)
    _env_bypass_active = env_encoder is not None
    # ATT464b: extract scd_encoder from kwargs
    scd_encoder = kwargs.pop("scd_encoder", None)
    _scd_bypass_active = scd_encoder is not None

    # ATT465: Multi-Level Spec-Text Alignment config
    import config as _cfg_mlsa
    _mlsa_active = bool(getattr(_cfg_mlsa, "MLSA_ENABLE", False))
    _mlsa_weight = float(getattr(_cfg_mlsa, "MLSA_WEIGHT", 0.1))
    _mlsa_warmup_epochs = int(getattr(_cfg_mlsa, "MLSA_WARMUP_EPOCHS", 5))
    _mlsa_margin = float(getattr(_cfg_mlsa, "MLSA_MARGIN", 0.2))
    if _mlsa_active:
        print(f"[ATT465-MLSA] Active: weight={_mlsa_weight}, warmup={_mlsa_warmup_epochs}, margin={_mlsa_margin}")

    # ATT447: Asymmetric Fusion→Spec Knowledge Distillation config
    import config as _cfg_distill
    _spec_distill_active = bool(getattr(_cfg_distill, "SPEC_DISTILL_FROM_FUSION", False))
    _distill_temperature = float(getattr(_cfg_distill, "SPEC_DISTILL_TEMPERATURE", 2.0))
    _distill_alpha = float(getattr(_cfg_distill, "SPEC_DISTILL_ALPHA", 0.3))
    _distill_start_epoch = int(getattr(_cfg_distill, "SPEC_DISTILL_START_EPOCH", 5))
    # ATT449-S2: FiGKD-inspired high-frequency only output KD
    _distill_hf_only = bool(getattr(_cfg_distill, "SPEC_DISTILL_HF_ONLY", False))
    if _spec_distill_active:
        _hf_tag = " HF-ONLY" if _distill_hf_only else ""
        print(f"[ATT447-Distill] ACTIVE: T={_distill_temperature}, alpha={_distill_alpha}, start_epoch={_distill_start_epoch}{_hf_tag}")
    # ATT448: Relational Knowledge Distillation config
    _rkd_enable = bool(getattr(_cfg_distill, "RKD_ENABLE", False))
    _rkd_dist_w = float(getattr(_cfg_distill, "RKD_DISTANCE_WEIGHT", 1.0))
    _rkd_angle_w = float(getattr(_cfg_distill, "RKD_ANGLE_WEIGHT", 2.0))
    _rkd_start_epoch = int(getattr(_cfg_distill, "RKD_START_EPOCH", 3))
    # ATT449-S1: Fault-Focused RKD
    _rkd_fault_focus_w = float(getattr(_cfg_distill, "RKD_FAULT_FOCUS_WEIGHT", 0.0))
    if _rkd_enable:
        _ff_tag = f", fault_focus={_rkd_fault_focus_w}" if _rkd_fault_focus_w > 0 else ""
        print(f"[ATT448-RKD] ACTIVE: dist_w={_rkd_dist_w}, angle_w={_rkd_angle_w}, start_epoch={_rkd_start_epoch}{_ff_tag}")

    history = {
        "total": [], "signal": [], "spec": [], "fusion": [], "margin": [], "fusion_cls": [],
        "fusion_proto_align": [], "fusion_supcon": [], "metric_lp_ce": [], "nr_inter_cos_margin": [], "nr_inter_cos_hard": [],
        "fusion_var_floor": [], "fusion_vicreg": [], "fusion_inter_cos_margin": [],
        "grad_norm": [],
        "signal_zero_shot_acc": [], "spec_zero_shot_acc": [], "fusion_zero_shot_acc": [],
        "fusion_decoder_acc": [], "val_loss": [], "val_acc": [], "val_spec_head_acc": [],
        "hust_best_fusion_decoder_acc": None,
        "hust_best_epoch": None,
        "hust_best_checkpoint_path": None,
    }
    
    # 如果使用MoE，添加MoE损失历史
    if use_moe:
        history["moe_total"] = []
        history["moe_load_balance"] = []
        history["moe_entropy"] = []
        history["moe_diversity"] = []
    
    # 如果使用一致性正则，添加一致性损失历史
    if use_consistency_regularization:
        history["consistency"] = []

    # Freeze text encoder backbone; selectively unfreeze proj when Text-ETF alignment is active
    for p in text_encoder.parameters():
        p.requires_grad = False

    import config as _cfg_freeze
    _unfreeze_proj = (
        bool(getattr(_cfg_freeze, "USE_SIMPLEX_ETF_PROTOTYPES", False))
        and bool(getattr(_cfg_freeze, "USE_TEXT_ETF_ALIGNMENT_LOSS", False))
    )
    if _unfreeze_proj and hasattr(text_encoder, "proj"):
        for p in text_encoder.proj.parameters():
            p.requires_grad = True
        _proj_param_count = sum(p.numel() for p in text_encoder.proj.parameters() if p.requires_grad)
        print(f"[ATT375] text_encoder.proj UNFROZEN for Text-ETF alignment ({_proj_param_count} params)")
        # Add proj parameters to optimizer
        if optimizer is not None:
            _proj_params = [p for p in text_encoder.proj.parameters() if p.requires_grad]
            if _proj_params:
                _proj_lr = optimizer.param_groups[0]["lr"] * 0.1
                optimizer.add_param_group({"params": _proj_params, "lr": _proj_lr})
                print(f"[ATT375] text_encoder.proj added to optimizer (lr={_proj_lr:.2e})")

    signal_encoder.train()
    spec_encoder.train()
    if _wst_bypass_active:
        wst_encoder.train()
    if _env_bypass_active:
        env_encoder.train()
    if _scd_bypass_active:
        scd_encoder.train()
    text_encoder.eval()

    # ===== (Margin Router) Track the current collapse target =====
    # This is updated after each HUST zero-shot evaluation and injected into hierarchical_parts,
    # so margin losses can focus on the current "absorber" class (vs_inner / vs_rolling / vs_normal / all).
    margin_router_target = "all"

    # ===== (Attempt 40) Track best HUST zero-shot checkpoint =====
    hust_best_fusion_decoder_acc = -1.0
    hust_best_epoch = None
    hust_best_checkpoint_path = None

    import config as _cfg_att475
    _att475_dual_ckpt = bool(getattr(_cfg_att475, "DUAL_CONTRACT_SAVE_EXTRA_CKPT", False))
    _att475_best_spec_v = -1.0
    _att475_best_fus_v = -1.0
    _att475_best_spec_path = None
    _att475_best_fus_path = None

    class_prompts = FAULT_PROMPT_TEMPLATES

    # ===== ATT374: Simplex ETF Prototypes =====
    import config as _cfg_etf
    etf_prototypes = None
    etf_prototypes_tensor = None
    use_etf = bool(getattr(_cfg_etf, "USE_SIMPLEX_ETF_PROTOTYPES", False))
    if use_etf:
        _num_cls_etf = len(Config.FAULT_LABEL_TO_SEMANTIC)
        etf_proto_raw = generate_simplex_etf(
            num_classes=_num_cls_etf,
            embed_dim=512,
            seed=int(getattr(_cfg_etf, "SIMPLEX_ETF_SEED", 42)),
            device=device,
        )
        etf_prototypes = {i: etf_proto_raw[i] for i in range(_num_cls_etf)}
        etf_prototypes_tensor = etf_proto_raw  # [C, 512]
        _names_list = None
        try:
            _names_list = [
                Config.FAULT_LABEL_TO_SEMANTIC.get(i, {}).get("component", str(i))
                for i in range(_num_cls_etf)
            ]
        except Exception:
            _names_list = None
        print_etf_stats(etf_proto_raw, class_names=_names_list)
        print(f"[ATT374] Simplex ETF prototypes generated: C={_num_cls_etf}, D=512")

    # ATT377-C: fix flat classifier to ETF prototypes (ATT378: disabled by default)
    _use_etf_fixed_cls = bool(getattr(Config, "USE_ETF_FIXED_CLASSIFIER", False))
    if use_etf and _use_etf_fixed_cls and etf_prototypes_tensor is not None:
        try:
            fusion_decoder.fix_classifier_to_etf(etf_prototypes_tensor)
        except Exception as _e:
            print(f"[ATT377-C] fix_classifier_to_etf failed: {_e}")
    else:
        _dec_trainable = sum(1 for p in fusion_decoder.parameters() if p.requires_grad)
        _dec_total = sum(1 for p in fusion_decoder.parameters())
        print(f"[ATT378] ETF-FIX disabled: classifier trainable. decoder params={_dec_trainable}/{_dec_total}")

    # 构建文本原型（如果使用贝叶斯更新，会在训练循环中更新）
    if use_bayesian_prototypes:
        initial_text_prototypes = build_text_prototypes(text_encoder, class_prompts, device)
        class_embeddings = {label: [] for label in initial_text_prototypes.keys()}
        text_prototypes = None
    else:
        text_prototypes = build_text_prototypes(text_encoder, class_prompts, device)
        class_embeddings = None
        initial_text_prototypes = None

    # ATT374/ATT390-S2: Prototype selection for training alignment
    text_prototypes_for_eval = text_prototypes  # save original text prototypes for DynMultiProto
    # ATT390-S2: Build a [C, D] tensor from whichever prototypes are used for training.
    # This tensor is needed by PACL, regardless of ETF vs Text.
    training_proto_tensor = None  # [C, D] tensor for PACL
    if use_etf and etf_prototypes is not None:
        text_prototypes = etf_prototypes
        training_proto_tensor = etf_prototypes_tensor
        print(f"[ATT374] Training prototypes: ETF (text prototypes preserved for eval)")
        if text_prototypes_for_eval is not None:
            for _lbl in sorted(etf_prototypes.keys()):
                if _lbl in text_prototypes_for_eval:
                    _cos = F.cosine_similarity(
                        etf_prototypes[_lbl].unsqueeze(0),
                        text_prototypes_for_eval[_lbl].unsqueeze(0),
                    ).item()
                    print(f"[ATT374] ETF↔Text cos for class {_lbl}: {_cos:.4f}")
    else:
        # ATT390-S2/FIX: Use text prototypes for training alignment (semantic zero-shot)
        # BUG FIX: When USE_BAYESIAN_PROTOTYPES=True, text_prototypes is None at init.
        # Must also check initial_text_prototypes (Bayesian prior).
        _proto_source = text_prototypes
        if _proto_source is None and initial_text_prototypes is not None:
            _proto_source = initial_text_prototypes
            print(f"[ATT390-S2] Using initial_text_prototypes (Bayesian prior) for training_proto_tensor")
        if _proto_source is not None and isinstance(_proto_source, dict) and len(_proto_source) > 0:
            _num_cls = len(Config.FAULT_LABEL_TO_SEMANTIC)
            _rows = []
            for _ci in range(_num_cls):
                if _ci in _proto_source and torch.is_tensor(_proto_source[_ci]):
                    _rows.append(F.normalize(_proto_source[_ci].to(device), dim=0))
                else:
                    _rows.append(torch.zeros(512, device=device))
            training_proto_tensor = torch.stack(_rows, dim=0)  # [C, D]
            print(f"[ATT390-S2] Training prototypes: TEXT SEMANTIC (C={_num_cls}, D={training_proto_tensor.shape[1]})")
            _cos_mat = torch.matmul(training_proto_tensor, training_proto_tensor.t())
            for _ri in range(_num_cls):
                _row_str = ", ".join([f"{_cos_mat[_ri, _ci].item():.4f}" for _ci in range(_num_cls)])
                print(f"[ATT390-S2] Text proto cosine row {_ri}: [{_row_str}]")
        else:
            print(f"[ATT390-S2] WARNING: text_prototypes AND initial_text_prototypes both None, PACL will be inactive")

    # ===== ATT390: Diagnostic — domain embedding init state =====
    if hasattr(fusion_module, 'domain_embedding'):
        _de_w = fusion_module.domain_embedding.weight.detach()
        _de_norms = _de_w.norm(dim=1)
        print(f"[ATT390-S1] Domain embedding init norms: {[f'{n:.6f}' for n in _de_norms.tolist()]}")
        print(f"[ATT390-S1] Domain embedding init: zero-init={'YES' if _de_norms.max().item() < 1e-6 else 'NO (WARNING!)'}")

    # ===== ATT390: Diagnostic — prototype mode summary =====
    print(f"[ATT390] Proto mode: use_etf={use_etf}, "
          f"training_proto_tensor={'['+str(training_proto_tensor.shape)+']' if training_proto_tensor is not None else 'None'}, "
          f"text_prototypes type={'ETF' if use_etf else 'TEXT'}")

    # ===== ATT391-S1: Proto Space Adapter (ETF → Text mapping) =====
    import config as _cfg_adapter
    _use_adapter = bool(getattr(_cfg_adapter, "USE_ETF_TEXT_ADAPTER", False))
    proto_adapter = None
    _adapter_text_proto_tensor = None  # [C, D] text prototypes for adapter loss (always text, never ETF)
    if _use_adapter and use_etf:
        proto_adapter = ProtoSpaceAdapter(dim=512).to(device)
        # Build text proto tensor for adapter target (must always be semantic text, NOT ETF)
        _adapter_src = initial_text_prototypes if initial_text_prototypes is not None else text_prototypes_for_eval
        if _adapter_src is None:
            _adapter_src = build_text_prototypes(text_encoder, class_prompts, device)
        if _adapter_src is not None and isinstance(_adapter_src, dict) and len(_adapter_src) > 0:
            _num_cls_a = len(Config.FAULT_LABEL_TO_SEMANTIC)
            _rows_a = []
            for _ci_a in range(_num_cls_a):
                if _ci_a in _adapter_src and torch.is_tensor(_adapter_src[_ci_a]):
                    _rows_a.append(F.normalize(_adapter_src[_ci_a].to(device), dim=0))
                else:
                    _rows_a.append(torch.zeros(512, device=device))
            _adapter_text_proto_tensor = torch.stack(_rows_a, dim=0)  # [C, D]
        _adapter_cos = torch.matmul(_adapter_text_proto_tensor, _adapter_text_proto_tensor.t()) if _adapter_text_proto_tensor is not None else None
        if _adapter_cos is not None:
            for _ri_a in range(_adapter_cos.size(0)):
                _row_str_a = ", ".join([f"{_adapter_cos[_ri_a, _ci_a].item():.4f}" for _ci_a in range(_adapter_cos.size(1))])
                print(f"[ATT391-S3] Improved text proto cosine row {_ri_a}: [{_row_str_a}]")
        # Store adapter on fusion_decoder for eval access
        fusion_decoder._proto_adapter = proto_adapter
        # Add adapter to optimizer
        if optimizer is not None:
            _adapter_lr_mult = float(getattr(_cfg_adapter, "ETF_TEXT_ADAPTER_LR_MULT", 1.0))
            _adapter_lr = optimizer.param_groups[0]["lr"] * _adapter_lr_mult
            optimizer.add_param_group({"params": list(proto_adapter.parameters()), "lr": _adapter_lr})
            print(f"[ATT391-S1] Adapter added to optimizer (lr={_adapter_lr:.2e}, lr_mult={_adapter_lr_mult})")
        print(f"[ATT391-S1] ProtoSpaceAdapter initialized (dim=512, identity-init), "
              f"adapter_text_proto={'['+str(_adapter_text_proto_tensor.shape)+']' if _adapter_text_proto_tensor is not None else 'None'}")
    else:
        if _use_adapter and not use_etf:
            print(f"[ATT391-S1] Adapter SKIPPED: use_etf=False, adapter only needed when training with ETF")
        fusion_decoder._proto_adapter = None

    # ===== ATT392-S1: Learnable Prototypes (replaces fixed text/ETF protos) =====
    import config as _cfg_lp
    _use_learnable_proto = bool(getattr(_cfg_lp, "USE_LEARNABLE_PROTOTYPES", False))
    semantic_proto_source = os.environ.get("SEMANTIC_PROTO_SOURCE", "text").strip().lower()
    semantic_relation_weight = float(os.environ.get("SEMANTIC_RELATION_WEIGHT", "0.05"))
    semantic_bayesian_blend = float(os.environ.get("SEMANTIC_BAYESIAN_BLEND", "0.0"))
    semantic_centroid_space = os.environ.get("NEW_IDEA_CENTROID_SPACE", "fusion").strip().lower()
    if semantic_centroid_space not in {"fusion", "metric"}:
        raise ValueError(f"Invalid NEW_IDEA_CENTROID_SPACE={semantic_centroid_space}")
    semantic_centroid_update = os.environ.get("NEW_IDEA_CENTROID_UPDATE", "each_epoch").strip().lower()
    if semantic_centroid_update not in {"each_epoch", "once"}:
        raise ValueError(f"Invalid NEW_IDEA_CENTROID_UPDATE={semantic_centroid_update}")
    semantic_centroid_start_epoch = int(os.environ.get("NEW_IDEA_CENTROID_START_EPOCH", "1"))
    if semantic_centroid_start_epoch < 1:
        raise ValueError("NEW_IDEA_CENTROID_START_EPOCH must be positive")
    learnable_proto_module = None  # type: LearnablePrototypes | None
    print(f"[VGSP] source={semantic_proto_source}, relation_weight={semantic_relation_weight:.4f}, bayesian_blend={semantic_bayesian_blend:.3f}")
    print(f"[VGSP] centroid_space={semantic_centroid_space}, centroid_update={semantic_centroid_update}, start_epoch={semantic_centroid_start_epoch}")
    if _use_learnable_proto:
        # Build text proto init tensor [C, D] from text encoder
        _lp_src = text_prototypes_for_eval  # always text, even if use_etf currently overrides text_prototypes
        if _lp_src is None:
            _lp_src = initial_text_prototypes
        if _lp_src is None:
            _lp_src = build_text_prototypes(text_encoder, class_prompts, device)
        _num_cls_lp = len(Config.FAULT_LABEL_TO_SEMANTIC)
        _rows_lp = []
        for _ci_lp in range(_num_cls_lp):
            if _ci_lp in _lp_src and torch.is_tensor(_lp_src[_ci_lp]):
                _rows_lp.append(F.normalize(_lp_src[_ci_lp].to(device), dim=0))
            else:
                _rows_lp.append(torch.zeros(512, device=device))
        _lp_init_tensor = torch.stack(_rows_lp, dim=0)  # [C, D]
        text_proto_control = os.environ.get("NEW_IDEA_TEXT_PROTO_CONTROL", "correct").strip().lower()
        if text_proto_control == "shuffled":
            _perm_lp = torch.roll(torch.arange(_num_cls_lp, device=device), shifts=-1)
            _lp_init_tensor = _lp_init_tensor[_perm_lp]
        elif text_proto_control != "correct":
            raise ValueError(f"Invalid NEW_IDEA_TEXT_PROTO_CONTROL={text_proto_control}")
        print(f"[VGSP] text_proto_control={text_proto_control}")

        _lp_min_cos = float(getattr(_cfg_lp, "LEARNABLE_PROTO_MIN_COS", -0.30))
        _lp_init_etf = bool(getattr(_cfg_lp, "LEARNABLE_PROTO_INIT_FROM_ETF", False))
        learnable_proto_module = LearnablePrototypes(
            text_proto_init=_lp_init_tensor,
            min_cos=_lp_min_cos,
            init_from_etf=_lp_init_etf,
        ).to(device)

        # Override training_proto_tensor to use learnable protos
        training_proto_tensor = learnable_proto_module.forward().detach()  # initial snapshot
        if semantic_proto_source == "signal_centroid":
            semantic_relation_control = os.environ.get("SEMANTIC_RELATION_CONTROL", "correct").strip().lower()
            _vgsp_target = training_proto_tensor @ training_proto_tensor.t()
            if semantic_relation_control == "shuffled":
                _perm = torch.roll(torch.arange(_vgsp_target.size(0), device=_vgsp_target.device), shifts=-1)
                _vgsp_target = _vgsp_target[_perm][:, _perm]
            elif semantic_relation_control == "one_hot":
                _vgsp_target = torch.eye(_vgsp_target.size(0), device=_vgsp_target.device)
            elif semantic_relation_control == "random":
                _gen = torch.Generator(device="cpu").manual_seed(int(os.environ.get("RANDOM_SEED", "2026")))
                _raw = torch.randn(_vgsp_target.size(0), _vgsp_target.size(1), generator=_gen)
                _raw = F.normalize(_raw.to(_vgsp_target.device), dim=-1)
                _vgsp_target = _raw @ _raw.t()
            elif semantic_relation_control not in {"correct", "ordinary"}:
                raise ValueError(f"unknown SEMANTIC_RELATION_CONTROL={semantic_relation_control}")
            learnable_proto_module.set_relation_target(_vgsp_target)
            print(f"[VGSP] relation_control={semantic_relation_control} target initialized")
        _init_method = "ETF" if _lp_init_etf else "text"
        print(f"[ATT432] LearnablePrototypes initialized: C={_num_cls_lp}, D=512, "
              f"min_cos={_lp_min_cos}, init={_init_method}")

        # Log initial prototype cosine matrix
        _lp_cos_init = learnable_proto_module.proto_cosine_matrix()
        for _ri_lp in range(_lp_cos_init.size(0)):
            _row_str_lp = ", ".join([f"{_lp_cos_init[_ri_lp, _ci_lp].item():.4f}" for _ci_lp in range(_lp_cos_init.size(1))])
            print(f"[ATT392-S1] Init proto cosine row {_ri_lp}: [{_row_str_lp}]")
        _lp_anchor_cos = learnable_proto_module.anchor_cosine()
        print(f"[ATT392-S1] Init anchor cos: [{', '.join([f'{v:.4f}' for v in _lp_anchor_cos.tolist()])}]")

        # Add to optimizer with higher LR
        if optimizer is not None:
            _lp_lr_mult = float(getattr(_cfg_lp, "LEARNABLE_PROTO_LR_MULT", 5.0))
            _lp_lr = optimizer.param_groups[0]["lr"] * _lp_lr_mult
            optimizer.add_param_group({"params": list(learnable_proto_module.parameters()), "lr": _lp_lr})
            print(f"[ATT392-S1] Learnable protos added to optimizer (lr={_lp_lr:.2e}, lr_mult={_lp_lr_mult})")

        # Store on fusion_decoder for eval access
        fusion_decoder._learnable_proto_module = learnable_proto_module
    else:
        fusion_decoder._learnable_proto_module = None
        print(f"[ATT392-S1] Learnable prototypes DISABLED")

    # ===== ATT432-δ: Physical Feature Prototype Module =====
    import config as _cfg_phys
    _use_phys_proto = bool(getattr(_cfg_phys, "USE_PHYS_FEATURE_PROTO", False))
    phys_proto_module = None
    if _use_phys_proto:
        from models.phys_feature_proto import PhysFeatureProtoModule, PHYS_FEATURE_DIM
        _phys_hidden = int(getattr(_cfg_phys, "PHYS_FEATURE_HIDDEN_DIM", 128))
        _phys_n_cls = len(Config.FAULT_LABEL_TO_SEMANTIC)
        phys_proto_module = PhysFeatureProtoModule(
            feat_dim=PHYS_FEATURE_DIM,
            embed_dim=512,
            num_classes=_phys_n_cls,
            hidden_dim=_phys_hidden,
        ).to(device)
        _phys_params = sum(p.numel() for p in phys_proto_module.parameters())
        if optimizer is not None:
            _phys_lr = optimizer.param_groups[0]["lr"]
            optimizer.add_param_group({"params": list(phys_proto_module.parameters()), "lr": _phys_lr})
        print(f"[ATT432-δ] PhysFeatureProtoModule ENABLED: feat_dim={PHYS_FEATURE_DIM}, "
              f"hidden={_phys_hidden}, n_cls={_phys_n_cls}, params={_phys_params}")
        fusion_decoder._phys_proto_module = phys_proto_module
    else:
        fusion_decoder._phys_proto_module = None
        print(f"[ATT432-δ] PhysFeatureProtoModule DISABLED")

    _ablation_gate_bank = get_gate_bank(spec_encoder)
    if _ablation_gate_bank is not None and optimizer is not None:
        _has_gate_group = any(
            str(g.get("name", "")) == "ablation_gates" for g in optimizer.param_groups
        )
        if not _has_gate_group:
            from utils.optimizer_utils import assigned_parameter_ids

            _gate_lr = float(optimizer.param_groups[0]["lr"]) * float(
                getattr(_cfg_phys, "ABLATION_GATE_LR_MULT", 0.5)
            )
            _assigned = assigned_parameter_ids(optimizer.param_groups)
            _gate_params = [
                p for p in _ablation_gate_bank.parameters() if id(p) not in _assigned
            ]
            if _gate_params:
                optimizer.add_param_group(
                    {
                        "params": _gate_params,
                        "lr": _gate_lr,
                        "name": "ablation_gates",
                    }
                )
                print(
                    f"[AblationGating] Optimizer group added: n_gates={len(_ablation_gate_bank.keys())}, "
                    f"lr={_gate_lr:.2e}"
                )
        else:
            print("[AblationGating] Optimizer group 'ablation_gates' already present (from main.py)")

    # ATT428 diagnostic: print MVZ configuration summary
    import config as _cfg_mvz
    _fishr_target = str(getattr(_cfg_mvz, "FISHR_TARGET_FEATURE", "z_fusion")).strip().lower()
    _fishr_w = float(getattr(_cfg_mvz, "FISHR_WEIGHT", 0.05))
    _fishr_enabled = bool(getattr(_cfg_mvz, "USE_FISHR_LOSS", False))
    _pacl_enabled = bool(getattr(_cfg_mvz, "USE_PROTOTYPE_ANCHOR_LOSS", False))
    _fpa_enabled = bool(getattr(_cfg_mvz, "USE_FUSION_PROTO_ALIGN_LOSS", False))
    _aux_ce_w = float(getattr(_cfg_mvz, "AUX_CE_WEIGHT", 0.0))
    _aux_prenorm = bool(getattr(_cfg_mvz, "AUX_PRENORM_CE_ENABLE", False))
    _sep_w = float(getattr(_cfg_mvz, "LEARNABLE_PROTO_SEP_WEIGHT", 0.0))
    _anchor_w = float(getattr(_cfg_mvz, "LEARNABLE_PROTO_ANCHOR_WEIGHT", 0.0))
    _dd_enabled = bool(getattr(_cfg_mvz, "USE_DUAL_DISENTANGLE", False))
    _dd_orth_w = float(getattr(_cfg_mvz, "DUAL_DISENTANGLE_ORTH_WEIGHT", 0.10))
    _dd_spe_w = float(getattr(_cfg_mvz, "DUAL_DISENTANGLE_DC_SPE_WEIGHT", 0.10))
    _dd_conf_w = float(getattr(_cfg_mvz, "DUAL_DISENTANGLE_CONFUSION_WEIGHT", 0.10))
    _var_enabled = bool(getattr(_cfg_mvz, "USE_SPEC_VARIANCE_CONSTRAINT", False))
    _var_w = float(getattr(_cfg_mvz, "SPEC_VARIANCE_WEIGHT", 0.10))
    _w_signal_align = float(getattr(_cfg_mvz, "W_SIGNAL_ALIGN", 0.3))
    print(f"[ATT431-MVZ] Fishr={_fishr_enabled}(w={_fishr_w}), "
          f"DualDisentangle={_dd_enabled}(orth={_dd_orth_w},spe={_dd_spe_w},conf={_dd_conf_w}), "
          f"SpecVariance={_var_enabled}(w={_var_w}), "
          f"AuxCE={_aux_ce_w:.2f}, PACL={_pacl_enabled}, "
          f"SignalAlign={_w_signal_align:.2f}")
    _active_losses = []
    if _fishr_enabled:
        _active_losses.append(f"Fishr(w={_fishr_w})")
    if _dd_enabled:
        _active_losses.append(f"DualDisentangle(orth={_dd_orth_w},spe={_dd_spe_w},conf={_dd_conf_w})")
    if _var_enabled:
        _active_losses.append(f"SpecVariance(w={_var_w})")
    _zcls_pretrain_disabled_cfg = bool(getattr(_cfg_mvz, "FUSION_PRETRAIN_DISABLE_ZCLS_CE", False))
    if _zcls_pretrain_disabled_cfg:
        _active_losses.append("fusion_cls(CE:pretrain_skipped)")
    else:
        _active_losses.append("fusion_cls(CE)")
    _active_losses.append("spec(CE)")
    _active_losses.append(f"signal(CE, w={_w_signal_align})")
    print(f"[ATT431-MVZ] Active losses ({len(_active_losses)}): {', '.join(_active_losses)}")

    # ATT431: Comprehensive structural audit at training start
    _use_cbp = bool(getattr(_cfg_mvz, "USE_COMPACT_BILINEAR_FUSION", False))
    _cbp_dim = int(getattr(_cfg_mvz, "CBP_DIM", 2048))
    _hier_spec = bool(getattr(_cfg_mvz, "USE_HIERARCHICAL_SPEC_HEAD", False))
    _learnable_proto = bool(getattr(_cfg_mvz, "USE_LEARNABLE_PROTOTYPES", False))
    _lp_sep_w = float(getattr(_cfg_mvz, "LEARNABLE_PROTO_SEP_WEIGHT", 0.0))
    _lp_anchor_w = float(getattr(_cfg_mvz, "LEARNABLE_PROTO_ANCHOR_WEIGHT", 0.0))
    _aux_ce_w_val = float(getattr(_cfg_mvz, "AUX_CE_WEIGHT", 0.0))
    _w_fusion_cls_e = float(getattr(_cfg_mvz, "W_FUSION_CLS_EARLY", 0.15))
    _w_fusion_cls_l = float(getattr(_cfg_mvz, "W_FUSION_CLS_LATE", 0.10))
    _use_gated = bool(getattr(_cfg_mvz, "USE_GATED_ADDITIVE_FUSION", False))
    _lp_init_etf = bool(getattr(_cfg_mvz, "LEARNABLE_PROTO_INIT_FROM_ETF", False))
    _mslr_sig = float(getattr(_cfg_mvz, "MSLR_SIGNAL_MULT", 1.0))
    _mslr_spe = float(getattr(_cfg_mvz, "MSLR_SPEC_MULT", 1.0))
    _mslr_fus = float(getattr(_cfg_mvz, "MSLR_FUSION_MULT", 3.0))
    _mslr_dec = float(getattr(_cfg_mvz, "MSLR_DECODER_MULT", 1.0))
    _use_shared_proj_lvl = bool(getattr(_cfg_mvz, "USE_SHARED_PROJECTION_FUSION", False))
    _cm_align_w = float(getattr(_cfg_mvz, "FUSION_CROSS_MODAL_ALIGN_WEIGHT", 0.20))
    _use_decision_lvl = bool(getattr(_cfg_mvz, "USE_DECISION_LEVEL_FUSION", False))
    _use_att472_lvl = bool(getattr(_cfg_mvz, "ATT472_TWO_END_SHARED_NECK_ENABLE", False))
    _use_att473_lvl = bool(getattr(_cfg_mvz, "ATT473_SPECIALIST_ARBITER_ENABLE", False))
    _fixed_eq_w = bool(getattr(_cfg_mvz, "FUSION_FIXED_EQUAL_WEIGHT", False))
    print(f"[ATT432-StructAudit] === Configuration Snapshot ===")
    print(f"  Fusion: SharedProj={_use_shared_proj_lvl}(align_w={_cm_align_w}), "
          f"GatedAdditive={_use_gated and not _use_att472_lvl and not _use_att473_lvl}, "
          f"ATT472SharedNeck={_use_att472_lvl}, ATT473SpecialistArbiter={_use_att473_lvl}, "
          f"DecisionLevel={_use_decision_lvl}(fixed_equal={_fixed_eq_w}), "
          f"CBP={_use_cbp}(dim={_cbp_dim}), "
          f"CrossAttn={not _use_gated and not _use_cbp and not _use_decision_lvl and not _use_shared_proj_lvl and not _use_att472_lvl and not _use_att473_lvl}")
    print(f"  DualDisentangle={_dd_enabled}, SpecVariance={_var_enabled}, HierarchicalSpec={_hier_spec}")
    print(f"  LearnableProto={_learnable_proto}(sep_w={_lp_sep_w}, anchor_w={_lp_anchor_w}, "
          f"init_etf={_lp_init_etf})")
    print(f"  MSLR: signal={_mslr_sig}x, spec={_mslr_spe}x, fusion={_mslr_fus}x, decoder={_mslr_dec}x")
    _gate_ent_w_cfg = float(getattr(_cfg_mvz, "GATE_ENTROPY_WEIGHT", 0.0))
    _proto_cls_flag = bool(getattr(_cfg_mvz, "USE_PROTO_CLASSIFIER", False))
    _proto_cls_T_cfg = float(getattr(_cfg_mvz, "PROTO_CLS_TEMPERATURE", 0.10))
    _proto_cos_align_flag = bool(getattr(_cfg_mvz, "USE_PROTO_COSINE_ALIGN", False))
    _proto_cos_align_w_cfg = float(getattr(_cfg_mvz, "PROTO_COSINE_ALIGN_WEIGHT", 0.10))
    print(f"  Loss weights: aux_CE={_aux_ce_w_val}, fusion_cls_early={_w_fusion_cls_e}, "
          f"fusion_cls_late={_w_fusion_cls_l}, signal_align={_w_signal_align}, "
          f"gate_ent={_gate_ent_w_cfg}")
    print(
        f"  ATT488-ZClsCE: pretrain_disable={_zcls_pretrain_disabled_cfg}, "
        f"stage1_only={bool(getattr(_cfg_mvz, 'FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY', True))} "
        "[z_cls kept for explicit finetune/calibration; pretrain budget moves to z_metric geometry]"
    )
    print(f"  ProtoClassifier={_proto_cls_flag}(T={_proto_cls_T_cfg}) [FALSIFIED: F_PROTO_CLS_SELF_TERMINATING]")
    print(f"  ProtoCosAlign={_proto_cos_align_flag}(w={_proto_cos_align_w_cfg})")
    print(f"  Fishr={_fishr_enabled}(w={_fishr_w}), PACL={_pacl_enabled}, FPA={_fpa_enabled}")
    _mslr_fusion = float(getattr(_cfg_mvz, "MSLR_FUSION_MULT", 3.0))
    _is_shared_proj = bool(getattr(_cfg_mvz, "USE_SHARED_PROJECTION_FUSION", False))
    _is_decision_level = bool(getattr(_cfg_mvz, "USE_DECISION_LEVEL_FUSION", False))
    print(f"  MSLR_FUSION={_mslr_fusion}x [ATT437-β: increased for detached-norm compensation]")
    if _is_shared_proj:
        _cm_align_w_diag = float(getattr(_cfg_mvz, "FUSION_CROSS_MODAL_ALIGN_WEIGHT", 0.20))
        print(f"  F.normalize → standard L2 [ATT460: shared-projection fusion, cross_modal_align_w={_cm_align_w_diag}, full gradient flow]")
    elif _is_decision_level:
        _fw_mode = "fixed_equal(0.5/0.5)" if bool(getattr(_cfg_mvz, "FUSION_FIXED_EQUAL_WEIGHT", False)) else "adaptive_softmax"
        _att_tag = "ATT459" if bool(getattr(_cfg_mvz, "FUSION_FIXED_EQUAL_WEIGHT", False)) else "ATT458"
        print(f"  F.normalize → standard L2 [{_att_tag}: decision-level fusion, weight={_fw_mode}, full gradient flow]")
    else:
        if bool(getattr(_cfg_mvz, "FUSION_USE_STANDARD_UNIT_VECTOR_NORM", False)):
            print(f"  F.normalize → standard L2 [GatedAdditive: full-gradient unit vector, FUSION_USE_STANDARD_UNIT_VECTOR_NORM=1]")
        else:
            print(f"  F.normalize → detached-norm [ATT437-α: gradient projection loss eliminated]")
    if bool(getattr(_cfg_mvz, "FUSION_MEAN_RESIDUAL_ENABLE", False)) and _use_gated and not _use_att472_lvl and not _use_att473_lvl:
        print(f"  GatedAdditive: mean+residual MLP (no tanh gate), last layer zero-init [FUSION_MEAN_RESIDUAL_ENABLE=1]")
    if _use_gated and (not bool(getattr(_cfg_mvz, "ATT472_TWO_END_SHARED_NECK_ENABLE", False))) and float(getattr(_cfg_mvz, "FUSION_CROSS_MODAL_ALIGN_WEIGHT", 0.0)) > 1e-8:
        print(f"  ATT470-GatedFusion: cross_modal_align on sig_proj/spe_proj "
              f"(w={float(getattr(_cfg_mvz, 'FUSION_CROSS_MODAL_ALIGN_WEIGHT', 0.0)):.4f})")
    _att472_en = bool(getattr(_cfg_mvz, "ATT472_TWO_END_SHARED_NECK_ENABLE", False))
    if _att472_en:
        print("  ATT472-SharedNeck: two-end-separated (signal/spec) + shared semantic bottleneck enabled")
    _att473_en = bool(getattr(_cfg_mvz, "ATT473_SPECIALIST_ARBITER_ENABLE", False))
    if _att473_en:
        print("  ATT473-SpecialistArbiter: dual semantic experts + sample-wise arbiter alpha enabled")
    _att470_en = bool(getattr(_cfg_mvz, "ATT470_STRUCTURE_ENABLE", False))
    _spec_be_audit = str(getattr(_cfg_mvz, "SPEC_ENCODER_BACKEND", ""))
    _moe_ra = bool(getattr(_cfg_mvz, "ATT470_MOE_ROUTE_DIAG_ENABLE", True))
    print(f"  ATT470-Preset: enable={_att470_en}, SPEC_ENCODER_BACKEND={_spec_be_audit}, MoE_RouterAudit={_moe_ra}")
    _att471_en = bool(getattr(_cfg_mvz, "ATT471_AGIT_ENABLE", False))
    _fus_fwd_detach = bool(getattr(_cfg_mvz, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
    print(f"  ATT471-AGIT: preset={_att471_en}, fusion_forward_detach_encoder_inputs={_fus_fwd_detach}")
    _att475_en = bool(getattr(_cfg_mvz, "DUAL_CONTRACT_ENABLE", False))
    _att475_ckpt = bool(getattr(_cfg_mvz, "DUAL_CONTRACT_SAVE_EXTRA_CKPT", False))
    _att475_grad = bool(getattr(_cfg_mvz, "DUAL_CONTRACT_FORCE_GRAD_DETAIL", False))
    print(
        f"  ATT475-DualContract: preset={_att475_en}, dual_ckpt_extra_save={_att475_ckpt}, "
        f"log_grad_detail_force={_att475_grad} "
        f"[SplitArch defaults off when env omitted + ATT470 off; fus_fwd_detach |= ATT475]"
    )
    print(
        f"  FusionGradFix: mean_residual={bool(getattr(_cfg_mvz, 'FUSION_MEAN_RESIDUAL_ENABLE', False))}, "
        f"standard_unit_L2={bool(getattr(_cfg_mvz, 'FUSION_USE_STANDARD_UNIT_VECTOR_NORM', False))} "
        f"(DUAL_CONTRACT=1 defaults both True; export =0 to A/B ablate)"
    )
    _fddt_audit = bool(getattr(_cfg_mvz, "FUSION_DECODER_DUAL_TOWER_ENABLE", False))
    _dual_space_audit = bool(getattr(_cfg_mvz, "USE_DUAL_SPACE_DECOUPLE", False))
    _dsm_detach_audit = bool(getattr(_cfg_mvz, "DUAL_SPACE_METRIC_DETACH", False))
    print(
        f"  ATT478-DecoderDualTower: FUSION_DECODER_DUAL_TOWER_ENABLE={_fddt_audit}, "
        f"USE_DUAL_SPACE_DECOUPLE={_dual_space_audit}, DUAL_SPACE_METRIC_DETACH={_dsm_detach_audit} "
        f"[proto fault + fusion_proto on z_metric; supervised head on z_cls; export FUSION_DECODER_DUAL_TOWER_ENABLE=0 to disable]"
    )
    _fsup_en = bool(getattr(_cfg_mvz, "FUSION_SUPCON_ENABLE", False))
    _fsup_w = float(getattr(_cfg_mvz, "FUSION_SUPCON_WEIGHT", 0.0))
    _fsup_T = float(getattr(_cfg_mvz, "FUSION_SUPCON_TEMPERATURE", 0.07))
    _fsup_fo = bool(getattr(_cfg_mvz, "FUSION_SUPCON_FAULT_ONLY", False))
    _fdec_stop = bool(getattr(_cfg_mvz, "FUSION_DECODER_INPUT_STOPGRAD_ENABLE", False))
    _sem_metric_live = bool(getattr(_cfg_mvz, "FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH", True))
    _fd_grad_live = getattr(_cfg_mvz, "FUSION_DECODER_INPUT_GRAD_LIVE_RATIO", None)
    _fus_geom = str(getattr(_cfg_mvz, "FUSION_GEOMETRY_LOSS_MODE", "silhouette"))
    _fus_sil_eps = float(getattr(_cfg_mvz, "FUSION_SILHOUETTE_EPS", 1e-6))
    print(
        f"  ATT479-FusionSupCon: FUSION_SUPCON_ENABLE={_fsup_en}, weight={_fsup_w}, T={_fsup_T}, "
        f"fault_only={_fsup_fo}, GEOM_MODE={_fus_geom}, SIL_EPS={_fus_sil_eps}, "
        f"FUSION_DECODER_INPUT_STOPGRAD_ENABLE={_fdec_stop}, "
        f"FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH={_sem_metric_live} "
        f"FUSION_DECODER_INPUT_GRAD_LIVE_RATIO={_fd_grad_live} "
        f"[export FUSION_SUPCON_ENABLE=1; ATT485: export FUSION_GEOMETRY_LOSS_MODE=silhouette(default when supcon on) or supcon(ATT484); "
        f"stopgrad defaults ON when SupCon ON — set FUSION_DECODER_INPUT_STOPGRAD_ENABLE=0 or "
        f"FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH=0 to ablate; ATT483: export FUSION_DECODER_INPUT_GRAD_LIVE_RATIO=0.1 "
        f"for fixed weak decoder->fusion grad (overrides ATT482 anneal)]"
    )
    _mlp_ce_en = bool(getattr(_cfg_mvz, "FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE", False))
    _mlp_ce_w = float(getattr(_cfg_mvz, "FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT", 0.0))
    _mlp_ce_t = float(getattr(_cfg_mvz, "FUSION_METRIC_LEARNABLE_PROTO_CE_TEMP", 0.10))
    _mlp_ce_e0 = int(getattr(_cfg_mvz, "FUSION_METRIC_LEARNABLE_PROTO_CE_START_EPOCH", 1))
    print(
        f"  ATT486-MetricLearnableProtoCE: enable={_mlp_ce_en}, w={_mlp_ce_w}, T={_mlp_ce_t}, start_epoch={_mlp_ce_e0} "
        f"[requires USE_LEARNABLE_PROTOTYPES=1 + dual-tower z_metric; "
        f"export FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE=1 to activate]"
    )
    _dgl_on = bool(getattr(_cfg_mvz, "DGL_ENABLE", False))
    _dgl_aux_w = float(getattr(_cfg_mvz, "DGL_AUX_FUSION_HEAD_WEIGHT", 0.0))
    _dgl_detach = bool(getattr(_cfg_mvz, "DGL_DETACH_ENCODER_INPUTS", False))
    _dgl_lr_mult = float(getattr(_cfg_mvz, "DGL_AUX_FUSION_HEAD_LR_MULT", 1.0))
    print(f"  DGL-AuxHead={_dgl_on}(aux_w={_dgl_aux_w}, detach_env_flag={_dgl_detach}, lr_mult={_dgl_lr_mult}) "
          f"[ATT438 aux CE path falsified; fusion detach independent of DGL_ENABLE]")
    _xkd_on = bool(getattr(_cfg_mvz, "CROSS_MODAL_KD_ENABLE", False))
    _xkd_w = float(getattr(_cfg_mvz, "CROSS_MODAL_KD_WEIGHT", 0.0))
    _xkd_T = float(getattr(_cfg_mvz, "CROSS_MODAL_KD_TEMPERATURE", 4.0))
    _hier_primary = bool(getattr(_cfg_mvz, "HIERARCHICAL_SPEC_HEAD_PRIMARY", False))
    _spechead_cosine = bool(getattr(_cfg_mvz, "SPECHEAD_USE_COSINE_PROTO", False))
    print(f"  XKD={_xkd_on}(w={_xkd_w}) HierSpec={_hier_primary} "
          f"[ATT439: FALSIFIED - XKD gradient hijack + HierSpec dead code]")
    print(f"  SpecHead_CosProto={_spechead_cosine} "
          f"[ATT440: FALSIFIED - text proto cos(IR,OR)=0.90]")
    _vicreg_on = bool(getattr(_cfg_mvz, "USE_GLOBAL_VICREG_LOSS", False))
    _decorr_on = bool(getattr(_cfg_mvz, "USE_GLOBAL_DECORR_LOSS", False))
    _decorr_w = float(getattr(_cfg_mvz, "GLOBAL_DECORR_WEIGHT", 0.0))
    _clip_w = float(getattr(_cfg_mvz, "CLIP_STYLE_CONTRASTIVE_WEIGHT", 0.0))
    _etf_on = bool(getattr(_cfg_mvz, "USE_SIMPLEX_ETF_PROTOTYPES", False))
    _text_etf_on = bool(getattr(_cfg_mvz, "USE_TEXT_ETF_ALIGNMENT_LOSS", False))
    _text_etf_w = float(getattr(_cfg_mvz, "TEXT_ETF_ALIGNMENT_WEIGHT", 0.0))
    print(f"  VICReg={_vicreg_on}(decorr={_decorr_on}, w={_decorr_w}) "
          f"CLIP_style(w={_clip_w}) ETF={_etf_on} text_etf={_text_etf_on}(w={_text_etf_w}) "
          f"[ATT446: WST Bypass Injection — WST→TinyEncoder→SpecHead only, "
          f"main spec path=7ch (ATT437 identical), wst_bypass={_wst_bypass_active}]")
    print(f"  SpecDistill={_spec_distill_active}(T={_distill_temperature}, alpha={_distill_alpha}, "
          f"start_epoch={_distill_start_epoch}) "
          f"[ATT447: Asymmetric Fusion→Spec KD — z_fusion.detach()→soft_target→KL(SpecHead)]")
    print(f"  RKD={_rkd_enable}(dist_w={_rkd_dist_w}, angle_w={_rkd_angle_w}, start={_rkd_start_epoch}, "
          f"fault_focus={_rkd_fault_focus_w}) "
          f"[ATT449: Relational KD + Fault-Focused RKD]")
    print(f"  EnvBypass={_env_bypass_active} [ATT449-S3: Envelope STFT + BPFI/BPFO ratio]")
    print(f"  HF-OKD={_distill_hf_only} [ATT449-S2: FiGKD-inspired high-frequency output KD]")
    _s1_wst_scale = getattr(spec_encoder, '_wst_scale', None)
    _s1_env_scale = getattr(spec_encoder, '_env_scale', None)
    _s1_ratio_scale = getattr(spec_encoder, '_ratio_scale', None)
    print(f"  ATT450-S1: BypassScaleNorm "
          f"wst_scale={'%.2f' % float(_s1_wst_scale.detach()) if _s1_wst_scale is not None else 'N/A'}, "
          f"env_scale={'%.2f' % float(_s1_env_scale.detach()) if _s1_env_scale is not None else 'N/A'}, "
          f"ratio_scale={'%.2f' % float(_s1_ratio_scale.detach()) if _s1_ratio_scale is not None else 'N/A'}")
    print(f"  ATT450-S2: RPM-aware BPFI/BPFO ratio (condition_to_rpm mapping)")
    _ogm_diag = bool(getattr(_cfg_mvz, "OGM_GE_ENABLE", False))
    _ogm_a_diag = float(getattr(_cfg_mvz, "OGM_GE_ALPHA", 1.0))
    _ogm_start_diag = int(getattr(_cfg_mvz, "OGM_GE_START_EPOCH", 2))
    print(f"  ATT461-OGM-GE: enable={_ogm_diag}, alpha={_ogm_a_diag}, start_epoch={_ogm_start_diag}")
    _res_skip_diag = bool(getattr(_cfg_mvz, "FUSION_RESIDUAL_SKIP_ENABLE", False))
    _res_alpha_diag = float(getattr(_cfg_mvz, "FUSION_RESIDUAL_SKIP_INIT_ALPHA", 0.1))
    print(f"  ATT462-ResSkip: enable={_res_skip_diag}, init_alpha={_res_alpha_diag}")
    _split_arch_diag = bool(getattr(_cfg_mvz, "SPLIT_ARCH_ENABLE", False))
    _split_arch_sig_w = float(getattr(_cfg_mvz, "SPLIT_ARCH_SIGNAL_WEIGHT", 1.0))
    print(f"  ATT463-SplitArch: enable={_split_arch_diag}, signal_weight={_split_arch_sig_w}")
    print(f"[ATT450-StructAudit] === End Configuration Snapshot ===")

    # ===== ATT396: Feature Decoupling — condition-invariant fault features =====
    from models.feature_decoupler import (
        FeatureDecoupler, ConditionClassifier, encode_condition_ids as _encode_cond_ids,
        compute_decouple_losses as _compute_decouple_losses,
    )
    import config as _cfg_decouple
    _use_decouple = bool(getattr(_cfg_decouple, "USE_FEATURE_DECOUPLE", False))
    decouple_spec = None
    decouple_signal = None
    condition_classifier = None
    if _use_decouple:
        _dc_embed = 512
        _dc_hidden = int(getattr(_cfg_decouple, "DECOUPLE_HIDDEN_DIM", _dc_embed))
        _dc_ncond = int(getattr(_cfg_decouple, "DECOUPLE_NUM_CONDITIONS", 4))
        decouple_spec = FeatureDecoupler(embed_dim=_dc_embed, hidden_dim=_dc_hidden).to(device)
        decouple_signal = FeatureDecoupler(embed_dim=_dc_embed, hidden_dim=_dc_hidden).to(device)
        condition_classifier = ConditionClassifier(
            embed_dim=_dc_embed, num_conditions=_dc_ncond, hidden_dim=128
        ).to(device)
        _dc_lr = float(getattr(optimizer, "defaults", {}).get("lr", 1e-4))
        optimizer.add_param_group({"params": list(decouple_spec.parameters()), "lr": _dc_lr})
        optimizer.add_param_group({"params": list(decouple_signal.parameters()), "lr": _dc_lr})
        optimizer.add_param_group({"params": list(condition_classifier.parameters()), "lr": _dc_lr})
        print(
            f"[ATT396][FeatureDecouple] ENABLED: embed_dim={_dc_embed}, "
            f"hidden={_dc_hidden}, num_cond={_dc_ncond}, "
            f"params_spec={sum(p.numel() for p in decouple_spec.parameters())}, "
            f"params_signal={sum(p.numel() for p in decouple_signal.parameters())}, "
            f"params_cond_cls={sum(p.numel() for p in condition_classifier.parameters())}"
        )
    else:
        print("[ATT396][FeatureDecouple] DISABLED")

    # ATT129 P1: 余弦分类器已回退为2层MLP，init_classifier_from_prototypes 不再适用
    # ATT128该调用在P1回退后为空操作（hasattr=False），此处显式移除以保持代码整洁

    # ===== 性能优化：预先计算class_weights（避免每个epoch重复遍历数据集） =====
    if precomputed_class_weights is None:
        all_labels = []
        for batch in train_loader:
            labels = batch["label"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            all_labels.extend(labels)
        all_labels = np.array(all_labels)
        num_classes = len(np.unique(all_labels))
        counts = np.zeros(num_classes, dtype=np.float32)
        for l in all_labels:
            counts[l] += 1
        class_weights = (1.0 / (counts + 1e-8)) ** 0.5
        class_weights = class_weights / class_weights.mean()
        precomputed_class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        # If provided, still compute counts once for nf pos_weight (required for hybrid nf stability)
        all_labels = []
        for batch in train_loader:
            labels = batch["label"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            all_labels.extend(labels)
        all_labels = np.array(all_labels)
        num_classes = int(np.max(all_labels)) + 1 if len(all_labels) > 0 else 4
        counts = np.zeros(num_classes, dtype=np.float32)
        for l in all_labels:
            counts[int(l)] += 1
    
    class_counts_tensor = torch.tensor(counts, dtype=torch.float32, device=device)
    try:
        import config as cfg_module
        use_balanced_softmax_init = bool(getattr(cfg_module, "USE_BALANCED_SOFTMAX", False))
    except Exception:
        use_balanced_softmax_init = False
    
    # 保存基础权重用于渐进式调整
    base_class_weights = precomputed_class_weights.clone()
    
    # 保存初始固定权重（用于 fixed 和 hybrid 模式）
    initial_fixed_class_weights = precomputed_class_weights.clone()
    
    # 输出权重策略信息
    try:
        import config as cfg_module
        adaptive_momentum = getattr(cfg_module, 'ADAPTIVE_WEIGHT_MOMENTUM', 0.1)
    except:
        adaptive_momentum = 0.1
    
    print(f"[Class Weight Strategy] Strategy: {class_weight_strategy}")
    print(f"[Balanced Softmax] Enabled: {use_balanced_softmax_init}, class_counts={counts}")
    if class_weight_strategy == "fixed":
        print(f"[Class Weight Strategy] Using fixed weights based on training set distribution")
        print(f"[Class Weight Strategy] Initial weights: {initial_fixed_class_weights.cpu().numpy()}")
    elif class_weight_strategy == "hybrid":
        print(f"[Class Weight Strategy] Using hybrid mode: fixed for first {class_weight_fixed_epochs} epochs, then adaptive")
        print(f"[Class Weight Strategy] Initial fixed weights: {initial_fixed_class_weights.cpu().numpy()}")
    elif class_weight_strategy == "adaptive":
        print(f"[Class Weight Strategy] Using adaptive weights (momentum={adaptive_momentum})")
    else:
        raise ValueError(f"Invalid class_weight_strategy: {class_weight_strategy}. Must be 'fixed', 'adaptive', or 'hybrid'")

    # ===== Hybrid nf pos_weight (neg/pos) for StageA stability =====
    # global nf: pos = fault_any, neg = normal
    normal_cnt = float(counts[0]) if len(counts) > 0 else 1.0
    fault_cnt = float(np.sum(counts[1:])) if len(counts) > 1 else 1.0
    nf_pos_weight_global = torch.tensor([normal_cnt / max(fault_cnt, 1.0)], dtype=torch.float32, device=device)

    # per-type nf_k: pos = class k, neg = all others
    nf_pos_weight_multi = []
    for k in [1, 2, 3]:
        pos = float(counts[k]) if k < len(counts) else 1.0
        neg = float(np.sum(counts) - pos) if len(counts) > 0 else 1.0
        nf_pos_weight_multi.append(neg / max(pos, 1.0))
    nf_pos_weight_multi = torch.tensor(nf_pos_weight_multi, dtype=torch.float32, device=device)  # [3]
    
    # ===== LDAM Loss + DRW 初始化 =====
    use_ldam_loss = False
    ldam_loss_fn = None
    use_drw = False
    drw_epoch = 10
    try:
        import config as cfg_module
        use_ldam_loss = bool(getattr(cfg_module, "USE_LDAM_LOSS", False))
        use_drw = bool(getattr(cfg_module, "USE_DRW", False))
        drw_epoch = int(getattr(cfg_module, "DRW_EPOCH", 10))
        ldam_max_margin = float(getattr(cfg_module, "LDAM_MAX_MARGIN", 0.5))
        ldam_scale = float(getattr(cfg_module, "LDAM_SCALE", 30.0))
        ldam_normalize_margins = bool(getattr(cfg_module, "LDAM_NORMALIZE_MARGIN", True))
    except Exception:
        use_ldam_loss = False
        use_drw = False
        ldam_max_margin = 0.5
        ldam_scale = 30.0
        ldam_normalize_margins = True
    
    if use_ldam_loss and counts is not None and len(counts) >= 4:
        ldam_loss_fn = LDAMLoss(
            class_counts=counts.tolist(),
            max_margin=ldam_max_margin,
            scale=ldam_scale,
            weight=None,  # Weight is handled by DRW
            normalize_margins=ldam_normalize_margins
        )
        print(f"[LDAM Loss] Enabled: max_margin={ldam_max_margin}, scale={ldam_scale}")
        print(f"[LDAM Loss] Normalize margins: {ldam_normalize_margins}")
        print(f"[LDAM Loss] Class-dependent margins computed from counts: {counts}")
        if use_drw:
            print(f"[DRW] Enabled: uniform weights for epochs 1-{drw_epoch}, then class re-weighting")
    
    # ===== Prototype Separation Loss 初始化 =====
    use_proto_sep_loss = False
    proto_sep_loss_fn = None
    proto_sep_weight = 0.1
    proto_sep_start_epoch = 3
    log_proto_distances = False
    try:
        import config as cfg_module
        use_proto_sep_loss = bool(getattr(cfg_module, "USE_PROTO_SEPARATION_LOSS", False))
        proto_sep_margin_delta = float(getattr(cfg_module, "PROTO_SEP_MARGIN_DELTA", 0.5))
        proto_sep_weight = float(getattr(cfg_module, "PROTO_SEP_LOSS_WEIGHT", 0.1))
        proto_sep_start_epoch = int(getattr(cfg_module, "PROTO_SEP_START_EPOCH", 3))
        proto_sep_momentum = float(getattr(cfg_module, "PROTO_SEP_MOMENTUM", 0.9))
        log_proto_distances = bool(getattr(cfg_module, "LOG_PROTO_DISTANCES", False))
        use_log_proto_sep = bool(getattr(cfg_module, "USE_LOG_PROTO_SEP", False))
        log_proto_sep_eps = float(getattr(cfg_module, "LOG_PROTO_SEP_EPS", 1e-6))
    except Exception:
        use_proto_sep_loss = False
        proto_sep_margin_delta = 0.5
        proto_sep_momentum = 0.9
        use_log_proto_sep = False
        log_proto_sep_eps = 1e-6
    
    # 注意：feature_dim 需要在模型构建后获取，这里先设为 None，后续初始化
    if use_proto_sep_loss:
        print(f"[Proto-Sep Loss] Enabled: delta={proto_sep_margin_delta}, weight={proto_sep_weight}, start_epoch={proto_sep_start_epoch}")
        print(f"[Proto-Sep Loss] Mode: {'log_ratio' if use_log_proto_sep else 'hinge'} (eps={log_proto_sep_eps})")
        print(f"[Proto-Sep Loss] Will initialize after first forward pass (need feature_dim)")

    # ===== SupCon Stage1 (flat head mainline) =====
    use_supcon_stage1 = False
    supcon_weight = 1.0
    supcon_spec_weight = 0.0
    supcon_temp = 0.07
    supcon_use_class_weights = True
    supcon_fault_only = False
    task2_coral_enable = False
    task2_coral_weight = 0.01
    task2_coral_start_epoch = 8
    task2_coral_warmup_epochs = 1
    task2_coral_min_samples = 4
    task2_target_coral_enable = False
    task2_target_coral_weight = 0.0
    task2_target_coral_every_n_steps = 1
    task2_target_coral_min_samples = 4
    task2_target_mmd_enable = False
    task2_target_mmd_weight = 0.0
    task2_target_mmd_start_epoch = 1
    task2_target_mmd_warmup_epochs = 5
    task2_target_mmd_every_n_steps = 1
    task2_target_mmd_min_samples = 8
    task2_target_mmd_kernel_scales = [0.5, 1.0, 2.0]
    try:
        import config as cfg_module
        use_supcon_stage1 = bool(getattr(cfg_module, "USE_SUPCON_STAGE1", False))
        supcon_weight = float(getattr(cfg_module, "SUPCON_STAGE1_WEIGHT", 1.0))
        supcon_spec_weight = float(getattr(cfg_module, "SUPCON_STAGE1_SPEC_WEIGHT", 0.0))
        supcon_temp = float(getattr(cfg_module, "SUPCON_TEMPERATURE", 0.07))
        supcon_use_class_weights = bool(getattr(cfg_module, "SUPCON_USE_CLASS_WEIGHTS", True))
        supcon_fault_only = bool(getattr(cfg_module, "SUPCON_FAULT_ONLY", False))
        task2_coral_enable = bool(getattr(cfg_module, "TARGET_TASK2_CORAL_ENABLE", False))
        task2_coral_weight = float(getattr(cfg_module, "TARGET_TASK2_CORAL_WEIGHT", 0.01))
        task2_coral_start_epoch = int(getattr(cfg_module, "TARGET_TASK2_CORAL_START_EPOCH", 8))
        task2_coral_warmup_epochs = int(getattr(cfg_module, "TARGET_TASK2_CORAL_WARMUP_EPOCHS", 1))
        task2_coral_min_samples = int(getattr(cfg_module, "TARGET_TASK2_CORAL_MIN_SAMPLES_PER_DOMAIN", 4))
        task2_target_coral_enable = bool(getattr(cfg_module, "TARGET_TASK2_TARGET_CORAL_ENABLE", False))
        task2_target_coral_weight = float(getattr(cfg_module, "TARGET_TASK2_TARGET_CORAL_WEIGHT", 0.0))
        task2_target_coral_every_n_steps = int(getattr(cfg_module, "TARGET_TASK2_TARGET_CORAL_EVERY_N_STEPS", 1))
        task2_target_coral_min_samples = int(getattr(cfg_module, "TARGET_TASK2_TARGET_CORAL_MIN_SAMPLES", 4))
        task2_target_mmd_enable = bool(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_ENABLE", False))
        task2_target_mmd_weight = float(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_WEIGHT", 0.0))
        task2_target_mmd_start_epoch = int(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_START_EPOCH", 1))
        task2_target_mmd_warmup_epochs = int(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_WARMUP_EPOCHS", 5))
        task2_target_mmd_every_n_steps = int(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_EVERY_N_STEPS", 1))
        task2_target_mmd_min_samples = int(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_MIN_SAMPLES", 8))
        task2_target_mmd_kernel_scales = list(getattr(cfg_module, "TARGET_TASK2_TARGET_MMD_KERNEL_SCALES", [0.5, 1.0, 2.0]))
    except Exception:
        use_supcon_stage1 = False
        supcon_weight = 1.0
        supcon_spec_weight = 0.0
        supcon_temp = 0.07
        supcon_use_class_weights = True
        supcon_fault_only = False
        task2_coral_enable = False
        task2_coral_weight = 0.01
        task2_coral_start_epoch = 8
        task2_coral_warmup_epochs = 1
        task2_coral_min_samples = 4
        task2_target_coral_enable = False
        task2_target_coral_weight = 0.0
        task2_target_coral_every_n_steps = 1
        task2_target_coral_min_samples = 4
        task2_target_mmd_enable = False
        task2_target_mmd_weight = 0.0
        task2_target_mmd_start_epoch = 1
        task2_target_mmd_warmup_epochs = 5
        task2_target_mmd_every_n_steps = 1
        task2_target_mmd_min_samples = 8
        task2_target_mmd_kernel_scales = [0.5, 1.0, 2.0]
    if use_supcon_stage1:
        print(
            f"[SupCon Stage1] Enabled: fusion_w={supcon_weight}, spec_w={supcon_spec_weight}, "
            f"temp={supcon_temp}, class_weights={supcon_use_class_weights}, fault_only={supcon_fault_only}"
        )
    if task2_coral_enable and task2_coral_weight > 0.0:
        print(
            f"[Task2 CORAL] Enabled: weight={task2_coral_weight}, "
            f"start_epoch={task2_coral_start_epoch}, warmup_epochs={task2_coral_warmup_epochs}, "
            f"min_samples/domain={task2_coral_min_samples}"
        )
    if task2_target_coral_enable and task2_target_coral_weight > 0.0:
        print(
            f"[Task2 Target-CORAL] Enabled: weight={task2_target_coral_weight}, "
            f"every_n_steps={max(1, task2_target_coral_every_n_steps)}, "
            f"min_samples={task2_target_coral_min_samples} (HUST/HIT unlabeled refs)"
        )
    if task2_target_mmd_enable and task2_target_mmd_weight > 0.0:
        print(
            f"[Task2 Target-MMD] Enabled: weight={task2_target_mmd_weight}, "
            f"start_epoch={task2_target_mmd_start_epoch}, warmup_epochs={task2_target_mmd_warmup_epochs}, "
            f"every_n_steps={max(1, task2_target_mmd_every_n_steps)}, min_samples={task2_target_mmd_min_samples}, "
            f"kernel_scales={task2_target_mmd_kernel_scales}"
        )

    def _extract_spec_feat_from_batch(_batch):
        """Return z_spec features for a target batch (no-grad reference branch)."""
        try:
            _spec = prepare_batch_spec(_batch, device, spec_encoder, Config)
            _spec_mask = _batch["spec_mask"].to(device)
        except Exception:
            return None
        _domains = _batch.get("domain", None)
        _conditions = _batch.get("condition", None)
        _domain_ids = None
        if _domains is not None:
            try:
                _domain_ids = encode_domain_ids(_domains, _conditions).to(device)
            except Exception:
                _domain_ids = None
        _sv = _batch.get("stat_vec", None)
        if _sv is not None:
            _sv = _sv.to(device)
        _is_moe = bool(use_moe and hasattr(spec_encoder, "num_bands"))
        _use_spec_head = bool(getattr(spec_encoder, "use_classification_head", False))
        try:
            if _is_moe:
                _texts = _batch.get("text", None)
                _z_text = text_encoder(_texts) if _texts is not None else None
                if _use_spec_head:
                    _out = spec_encoder(
                        _spec, _spec_mask, text_emb=_z_text, domain_ids=_domain_ids,
                        return_moe_stats=True, return_logits=True, stat_vec=_sv,
                    )
                    _z_spec = _out[0]
                else:
                    _out = spec_encoder(
                        _spec, _spec_mask, text_emb=_z_text, domain_ids=_domain_ids,
                        return_moe_stats=True, stat_vec=_sv,
                    )
                    _z_spec = _out[0]
            else:
                if _use_spec_head:
                    _z_spec, _ = spec_encoder(
                        spec=_spec, spec_mask=_spec_mask, domain_ids=_domain_ids, return_logits=True, stat_vec=_sv,
                    )
                else:
                    _z_spec = spec_encoder(spec=_spec, spec_mask=_spec_mask, domain_ids=_domain_ids, stat_vec=_sv)
            return _z_spec
        except Exception:
            return None
    
    # 初始化混合精度训练的scaler（如果启用）
    scaler = torch.amp.GradScaler("cuda") if use_amp and torch.cuda.is_available() else None

    # ===== Early stopping 和 Schedule 配置 =====
    if use_schedule:
        # 学习率 schedule（如果 optimizer 有初始 lr，从这里获取；否则用默认值）
        initial_lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-3
        lr_early = initial_lr  # 前N轮的学习率
        lr_late = initial_lr * lr_decay_factor  # N轮后的学习率
    else:
        # 不使用 schedule，使用固定的 w_fusion_cls_early 作为固定值
        w_fusion_cls_late = w_fusion_cls_early  # 保持相同值
        schedule_epoch_threshold = float('inf')  # 永远不会切换
        lr_early = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-3
        lr_late = lr_early
    
    # Early stopping 配置
    if use_early_stopping:
        # 根据early_stop_metric决定初始最佳值
        # ATT132 P0: 新增 val_spec_head_acc（accuracy，越大越好，与val_acc同类）
        if early_stop_metric in ["val_loss"]:
            best_metric_value = float('inf')  # 损失越小越好
        else:
            best_metric_value = -1.0  # 准确率越大越好（val_acc / val_spec_head_acc）
        best_metric_epoch = -1
        no_improve_count = 0
        # ATT130 A: track best checkpoint during pre-min_epochs window separately
        # to avoid resetting baseline to a suboptimal epoch value (ATT125/ATT129 bug)
        best_pre_window_value = float('inf') if early_stop_metric == "val_loss" else -1.0
        best_pre_window_epoch = -1
        best_pre_window_ckpt_path = None
    else:
        best_metric_value = None
        best_metric_epoch = None
        no_improve_count = None
        early_stop_patience = None

    # ===== stability: metric learning on diagnosis branch z_cls =====
    try:
        from config import FUSION_METRIC_LOSS, FUSION_METRIC_LOSS_WEIGHT, FUSION_CENTER_MOMENTUM
    except Exception:
        FUSION_METRIC_LOSS = "none"
        FUSION_METRIC_LOSS_WEIGHT = 0.0
        FUSION_CENTER_MOMENTUM = 0.95
    ema_centers = None  # EMA centers for z_cls, shape [C,D], lazy init

    # ===== EMA (Exponential Moving Average) 权重管理 =====
    ema_manager = None
    if use_ema:
        ema_manager = EMAManager(
            decay=ema_decay,
            warmup_steps=ema_warmup_steps,
            update_after_step=ema_update_after_step,
            device=device
        )
        # 注册需要 EMA 的模型
        ema_manager.register('signal_encoder', signal_encoder)
        ema_manager.register('spec_encoder', spec_encoder)
        ema_manager.register('fusion_module', fusion_module)
        ema_manager.register('fusion_decoder', fusion_decoder)
        print(f"[EMA] Enabled with decay={ema_decay}, warmup_steps={ema_warmup_steps}, update_after_step={ema_update_after_step}")
    
    # 全局步数计数器（用于 EMA warmup）
    global_step = 0
    
    # ===== 元学习（Meta-Learning）初始化 (DISABLED: USE_META_LEARNING=False) =====
    meta_learner = None
    
    # 检查并打印自适应权重调整配置
    import config as cfg
    _dual_contract_on = bool(getattr(cfg, "DUAL_CONTRACT_ENABLE", False))
    if _dual_contract_on:
        _dc_split_arch = bool(getattr(cfg, "SPLIT_ARCH_ENABLE", False))
        _dc_detach = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
        _dc_fusion_trainable = sum(p.numel() for p in fusion_module.parameters() if p.requires_grad)
        _dc_decoder_trainable = sum(p.numel() for p in fusion_decoder.parameters() if p.requires_grad)
        if _dc_split_arch:
            raise RuntimeError(
                "DualContract guard failed: SPLIT_ARCH_ENABLE must be False when DUAL_CONTRACT_ENABLE=True."
            )
        _allow_no_detach_ablation = bool(
            getattr(cfg, "DUAL_CONTRACT_ALLOW_NO_DETACH_ABLATION", False)
        )
        if not _dc_detach and not _allow_no_detach_ablation:
            raise RuntimeError(
                "DualContract guard failed: FUSION_FORWARD_DETACH_ENCODER_INPUTS must be True when DUAL_CONTRACT_ENABLE=True."
            )
        if _dc_fusion_trainable <= 0 or _dc_decoder_trainable <= 0:
            raise RuntimeError(
                "DualContract guard failed: fusion_module/fusion_decoder has no trainable parameters."
            )
        print(
            f"[DualContract-Guard] PASSED: split_arch={_dc_split_arch}, fus_fwd_detach={_dc_detach}, "
            f"no_detach_ablation={int(_allow_no_detach_ablation)}, "
            f"trainable(fusion/decoder)=({_dc_fusion_trainable}/{_dc_decoder_trainable})"
        )
    try:
        progressive_weight_total_epochs = int(
            getattr(cfg, "PROGRESSIVE_WEIGHT_TOTAL_EPOCHS", num_epochs)
        )
    except Exception:
        progressive_weight_total_epochs = int(num_epochs)
    progressive_weight_total_epochs = max(1, progressive_weight_total_epochs)
    if use_progressive_weights:
        print(
            "[Progressive Weights] "
            f"schedule_total_epochs={progressive_weight_total_epochs}, train_total_epochs={num_epochs}"
        )
    use_adaptive_weights = getattr(cfg, 'USE_ADAPTIVE_CLASS_WEIGHTS', True)
    # sampler-first: if WeightedSampler is enabled, do NOT also do adaptive class-weight updates
    # (it creates a second imbalance feedback loop and often triggers "all→Rolling" collapse).
    try:
        if bool(getattr(cfg, "USE_WEIGHTED_SAMPLER", False)):
            use_adaptive_weights = False
    except Exception:
        pass
    if enable_class_monitoring and use_adaptive_weights:
        adaptive_momentum = getattr(cfg, 'ADAPTIVE_WEIGHT_MOMENTUM', 0.1)
        adaptive_min_w = getattr(cfg, 'ADAPTIVE_WEIGHT_MIN', 0.5)
        adaptive_max_w = getattr(cfg, 'ADAPTIVE_WEIGHT_MAX', 3.0)
        print(f"[Adaptive Weights] Enabled: momentum={adaptive_momentum}, min={adaptive_min_w}, max={adaptive_max_w}")
        print(f"[Adaptive Weights] Initial class weights: {precomputed_class_weights.cpu().numpy()}")
    elif enable_class_monitoring:
        print(f"[Adaptive Weights] Class monitoring enabled but adaptive weights disabled (USE_ADAPTIVE_CLASS_WEIGHTS=False)")

    # ===== Dataflow audit (domain-id / label routing sanity check) =====
    dataflow_audit_enabled = bool(getattr(cfg, "TRAIN_LOG_DATAFLOW_AUDIT", False))
    dataflow_audit_stride = max(1, int(getattr(cfg, "TRAIN_DATAFLOW_AUDIT_EPOCH_STRIDE", 1)))
    domain_id_to_name = {
        int(getattr(cfg, "CWRU_GK2_DOMAIN_ID", 0)): "CWRU_GK2",
        int(getattr(cfg, "CWRU_GK4_DOMAIN_ID", 1)): "CWRU_GK4",
        int(getattr(cfg, "PU_DOMAIN_ID", 2)): "PU",
        int(getattr(cfg, "HUST_DOMAIN_ID", 3)): "HUST",
        int(getattr(cfg, "HIT_DOMAIN_ID", 4)): "HIT",
    }
    if dataflow_audit_enabled:
        domain_map_msg = ", ".join([f"{name}:{did}" for did, name in sorted(domain_id_to_name.items(), key=lambda x: x[0])])
        print(f"[DataflowAudit] enabled=True, epoch_stride={dataflow_audit_stride}, domain_map=({domain_map_msg})")

    grad_norm_log_enabled = bool(getattr(cfg, "TRAIN_LOG_GRAD_NORM", False))
    grad_norm_log_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_GRAD_NORM_STEP_STRIDE", 50)))
    grad_norm_detail_log_enabled = bool(getattr(cfg, "TRAIN_LOG_GRAD_NORM_DETAIL", False))
    grad_norm_detail_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_GRAD_NORM_DETAIL_STEP_STRIDE", 20)))
    if bool(getattr(cfg, "DUAL_CONTRACT_FORCE_GRAD_DETAIL", False)):
        grad_norm_detail_log_enabled = True
    if grad_norm_log_enabled:
        print(f"[GradAudit] enabled=True, step_stride={grad_norm_log_step_stride}")
    batch_loss_log_enabled = bool(getattr(cfg, "TRAIN_LOG_BATCH_LOSS", False))
    batch_loss_log_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_BATCH_LOSS_STEP_STRIDE", 50)))
    batch_loss_breakdown_log_enabled = bool(getattr(cfg, "TRAIN_LOG_BATCH_LOSS_BREAKDOWN", False))
    batch_loss_breakdown_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_BATCH_LOSS_BREAKDOWN_STEP_STRIDE", batch_loss_log_step_stride)))
    proto_cos_log_enabled = bool(getattr(cfg, "TRAIN_LOG_PROTOTYPE_COS_MATRIX", False))
    proto_cos_log_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_PROTOTYPE_COS_STEP_STRIDE", 100)))
    batch_io_log_enabled = bool(getattr(cfg, "TRAIN_LOG_BATCH_IO_STATS", False))
    batch_io_log_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_BATCH_IO_STEP_STRIDE", 100)))
    grad_contrib_log_enabled = bool(getattr(cfg, "TRAIN_LOG_GRAD_CONTRIB", False))
    grad_contrib_log_step_stride = max(1, int(getattr(cfg, "TRAIN_LOG_GRAD_CONTRIB_STEP_STRIDE", 100)))
    loss_budget_log_enabled = bool(getattr(cfg, "TRAIN_LOG_LOSS_BUDGET", False))
    loss_budget_epoch_stride = max(1, int(getattr(cfg, "TRAIN_LOG_LOSS_BUDGET_EPOCH_STRIDE", 1)))
    grad_clip_enabled = bool(getattr(cfg, "TRAIN_ENABLE_GRAD_CLIP", False))
    grad_clip_max_norm = float(getattr(cfg, "TRAIN_GRAD_CLIP_MAX_NORM", 5.0))
    grad_clip_log_enabled = bool(getattr(cfg, "TRAIN_LOG_GRAD_CLIP", False))
    if batch_loss_log_enabled:
        print(f"[BatchLossAudit] enabled=True, step_stride={batch_loss_log_step_stride}")
    if batch_loss_breakdown_log_enabled:
        print(f"[BatchLossBreakdownAudit] enabled=True, step_stride={batch_loss_breakdown_step_stride}")
    if proto_cos_log_enabled:
        print(f"[ProtoCosAudit] enabled=True, step_stride={proto_cos_log_step_stride}")
    if batch_io_log_enabled:
        print(f"[BatchIOAudit] enabled=True, step_stride={batch_io_log_step_stride}")
    if grad_contrib_log_enabled:
        print(f"[GradContribAudit] enabled=True, step_stride={grad_contrib_log_step_stride}")
    if grad_norm_detail_log_enabled:
        print(f"[GradNormDetailAudit] enabled=True, step_stride={grad_norm_detail_step_stride}")
    if loss_budget_log_enabled:
        print(f"[LossBudgetAudit] enabled=True, epoch_stride={loss_budget_epoch_stride}")
    if grad_clip_log_enabled or grad_clip_enabled:
        print(f"[GradClipAudit] enabled={int(grad_clip_enabled)}, max_norm={grad_clip_max_norm:.4f}")

    def _compute_global_grad_norm():
        sq_sum = 0.0
        for _mod in (signal_encoder, spec_encoder, text_encoder, fusion_module, fusion_decoder):
            if _mod is None:
                continue
            for _p in _mod.parameters():
                if _p.grad is None:
                    continue
                _g = _p.grad.detach()
                sq_sum += float((_g.float() * _g.float()).sum().item())
        return math.sqrt(max(0.0, sq_sum))

    def _compute_module_grad_norm(_mod):
        if _mod is None:
            return 0.0, 0
        _sq = 0.0
        _count = 0
        try:
            for _p in _mod.parameters():
                if _p.grad is None:
                    continue
                _g = _p.grad.detach()
                _sq += float((_g.float() * _g.float()).sum().item())
                _count += 1
        except Exception:
            return 0.0, 0
        return math.sqrt(max(0.0, _sq)), _count

    def _iter_optimizer_grad_params(_optimizer):
        for _group in _optimizer.param_groups:
            for _p in _group.get("params", []):
                if _p is not None and getattr(_p, "requires_grad", False) and (_p.grad is not None):
                    yield _p

    _signal_param_id_set = set(id(_p) for _p in signal_encoder.parameters()) if signal_encoder is not None else set()

    def _signal_optimizer_membership(_optimizer):
        _total = len(_signal_param_id_set)
        _req_grad = 0
        _in_opt = 0
        _in_opt_req = 0
        if signal_encoder is not None:
            for _p in signal_encoder.parameters():
                if bool(getattr(_p, "requires_grad", False)):
                    _req_grad += 1
        if _optimizer is not None:
            try:
                for _g in _optimizer.param_groups:
                    for _p in _g.get("params", []):
                        if _p is None:
                            continue
                        if id(_p) in _signal_param_id_set:
                            _in_opt += 1
                            if bool(getattr(_p, "requires_grad", False)):
                                _in_opt_req += 1
            except Exception:
                pass
        return {
            "total": int(_total),
            "requires_grad": int(_req_grad),
            "in_optimizer": int(_in_opt),
            "in_optimizer_requires_grad": int(_in_opt_req),
        }

    def _to_proto_matrix(_prototypes, _device, _dtype):
        if _prototypes is None:
            return None, None
        try:
            if isinstance(_prototypes, dict):
                _keys = sorted([int(k) for k in _prototypes.keys() if int(k) >= 0])
                _rows = []
                for _k in _keys:
                    _p = _prototypes.get(_k, None)
                    if torch.is_tensor(_p) and _p.dim() == 1:
                        _rows.append(_p.to(_device, dtype=_dtype))
                if len(_rows) >= 2:
                    return torch.stack(_rows, dim=0), _keys
            elif torch.is_tensor(_prototypes) and _prototypes.dim() == 2 and int(_prototypes.size(0)) >= 2:
                _keys = list(range(int(_prototypes.size(0))))
                return _prototypes.to(_device, dtype=_dtype), _keys
        except Exception:
            return None, None
        return None, None

    def _iter_grad_probe_params():
        for _mod in (fusion_decoder, fusion_module, spec_encoder, signal_encoder):
            if _mod is None:
                continue
            for _p in _mod.parameters():
                if _p.requires_grad and torch.is_tensor(_p):
                    yield _p

    def _grad_contrib_snapshot(_total_loss, _term_tensors):
        if (not torch.is_tensor(_total_loss)) or (not _total_loss.requires_grad):
            return None
        _probe_param = None
        _g_total_f = None
        _g_total_norm = 0.0
        for _cand in _iter_grad_probe_params():
            try:
                _g_total = torch.autograd.grad(
                    _total_loss, _cand, retain_graph=True, create_graph=False, allow_unused=True
                )[0]
            except Exception:
                _g_total = None
            if _g_total is None:
                continue
            _g_total_f_try = _g_total.detach().float().reshape(-1)
            _g_total_norm_try = float(torch.norm(_g_total_f_try, p=2).item())
            if _g_total_norm_try <= 0:
                continue
            _probe_param = _cand
            _g_total_f = _g_total_f_try
            _g_total_norm = _g_total_norm_try
            break
        if _probe_param is None or _g_total_f is None or _g_total_norm <= 0:
            return None
        _rows = []
        for _name, _term in _term_tensors.items():
            if (not torch.is_tensor(_term)) or (not _term.requires_grad):
                continue
            try:
                _g_term = torch.autograd.grad(
                    _term, _probe_param, retain_graph=True, create_graph=False, allow_unused=True
                )[0]
            except Exception:
                _g_term = None
            if _g_term is None:
                continue
            _g_term_f = _g_term.detach().float().reshape(-1)
            _g_term_norm = float(torch.norm(_g_term_f, p=2).item())
            if _g_term_norm <= 0:
                continue
            _cos = float(torch.dot(_g_term_f, _g_total_f).item() / (_g_term_norm * _g_total_norm + 1e-12))
            _rows.append((_name, _g_term_norm, _g_term_norm / (_g_total_norm + 1e-12), _cos))
        if len(_rows) == 0:
            return None
        _rows = sorted(_rows, key=lambda x: x[1], reverse=True)
        return _g_total_norm, _rows

    # ===== (Fault-type) Long-tail weights for {inner, outer, rolling} =====
    # Rolling-element is often extremely rare in mixed-source training, causing 0-recall collapse on HUST.
    # We compute a dedicated 3-way weight vector and pass it into the hierarchical fusion loss.
    fault_type_weights = None
    try:
        import config as cfg
        ft_enable = bool(getattr(cfg, "FAULT_TYPE_WEIGHT_ENABLE", True))
        ft_pow = float(getattr(cfg, "FAULT_TYPE_WEIGHT_POWER", 1.0))
        ft_max = float(getattr(cfg, "FAULT_TYPE_MAX_WEIGHT", 20.0))
    except Exception:
        ft_enable, ft_pow, ft_max = True, 1.0, 20.0

    if ft_enable:
        try:
            import numpy as _np
            from collections import Counter as _Counter
            from torch.utils.data import Subset as _Subset

            def _collect_labels_from_dataset(ds):
                labels_out = []
                # ConcatDataset-like
                if hasattr(ds, "datasets"):
                    for sub in ds.datasets:
                        if isinstance(sub, _Subset):
                            base = sub.dataset
                            idxs = sub.indices
                            if hasattr(base, "labels"):
                                labels_out.extend(_np.asarray(base.labels)[_np.asarray(idxs)].tolist())
                        else:
                            if hasattr(sub, "labels"):
                                labels_out.extend(_np.asarray(sub.labels).tolist())
                elif hasattr(ds, "labels"):
                    labels_out = _np.asarray(ds.labels).tolist()
                return labels_out

            ds_labels = _collect_labels_from_dataset(getattr(train_loader, "dataset", None))
            c = _Counter(int(x) for x in ds_labels)
            # labels: 0=normal, 1=inner, 2=outer, 3=rolling
            inner_n = max(1, int(c.get(1, 0)))
            outer_n = max(1, int(c.get(2, 0)))
            roll_n = max(1, int(c.get(3, 0)))
            # IMPORTANT: only up-weight the rare class (usually rolling), never down-weight inner/outer.
            # Use max_count / count so the most frequent class gets weight=1 (before power/clamp).
            max_n = float(max(inner_n, outer_n, roll_n))
            w = _np.array([max_n / inner_n, max_n / outer_n, max_n / roll_n], dtype=_np.float32)
            w = _np.power(w, float(ft_pow))
            w = _np.clip(w, a_min=0.0, a_max=float(ft_max))
            # Normalize so min weight = 1.0 (keeps inner/outer >= 1)
            w = w / (float(w.min()) + 1e-8)
            fault_type_weights = torch.tensor(w, dtype=torch.float32, device=device)
            print(f"[Fault-type Weights] inner/outer/rolling counts=({inner_n},{outer_n},{roll_n}), weights={w.tolist()}")
        except Exception:
            fault_type_weights = None

    # ===== (S-C2) Training health gate state =====
    # If we detect collapse on HUST zero-shot, we temporarily disable prototype-head training for stability.
    proto_train_cooldown_until = 0
    roll_decorr_last_reset_epoch = -999999
    freeze_signal_after_epoch = int(getattr(cfg, "FREEZE_SIGNAL_AFTER_EPOCH", -1))
    signal_encoder_frozen = False

    _decoupled_enable = bool(getattr(cfg, "DECOUPLED_TRAINING_ENABLE", False))
    _decoupled_s1_epochs = int(getattr(cfg, "DECOUPLED_STAGE1_EPOCHS", 8))
    _decoupled_freeze_enc = bool(getattr(cfg, "DECOUPLED_STAGE2_FREEZE_ENCODERS", True))
    _decoupled_encoders_frozen = False

    # ATT409: Auxiliary pre-norm classification head
    _aux_prenorm_enable = bool(getattr(cfg, "AUX_PRENORM_CE_ENABLE", False))
    _aux_prenorm_weight = float(getattr(cfg, "AUX_PRENORM_CE_WEIGHT", 0.3))
    _aux_prenorm_head = None
    if _aux_prenorm_enable:
        _embed_dim = int(getattr(cfg, "EMBED_DIM", 512))
        _n_cls = int(getattr(cfg, "NUM_CLASSES", 4))
        _aux_prenorm_head = torch.nn.Linear(_embed_dim, _n_cls, bias=False).to(device)
        torch.nn.init.xavier_uniform_(_aux_prenorm_head.weight)
        try:
            optimizer.add_param_group({"params": list(_aux_prenorm_head.parameters()), "lr": float(getattr(cfg, "LEARNING_RATE", 1e-4))})
        except Exception as _apg_e:
            print(f"[ATT409-AuxPrenorm] WARNING: add_param_group failed ({_apg_e}), params NOT optimized by main optimizer")
        print(f"[ATT409-AuxPrenorm] ENABLED: weight={_aux_prenorm_weight}, embed_dim={_embed_dim}, n_cls={_n_cls}")

    best_checkpoint_path = None

    for epoch in range(num_epochs - start_epoch):
        current_epoch = start_epoch + epoch + 1
        in_proto_cooldown = (proto_train_cooldown_until > 0) and (current_epoch <= int(proto_train_cooldown_until))

        _decoupled_is_stage1 = _decoupled_enable and (current_epoch <= _decoupled_s1_epochs)
        _decoupled_is_stage2 = _decoupled_enable and (current_epoch > _decoupled_s1_epochs)

        # ATT378c-FIX: ensure all trainable models are in train mode at the start
        # of each epoch. evaluate_on_val_loader (end of prev epoch) sets all to
        # eval and the old code never restored fusion_module / fusion_decoder,
        # breaking gradient flow through the decoder's weight-normalization cache.
        if not signal_encoder_frozen and not _decoupled_encoders_frozen:
            signal_encoder.train()
        if not _decoupled_encoders_frozen:
            spec_encoder.train()
            fusion_module.train()
        fusion_decoder.train()
        # text_encoder stays in eval mode (frozen)

        # ATT379c: Decoupled Stage 2 — freeze encoders once at transition
        if _decoupled_is_stage2 and _decoupled_freeze_enc and not _decoupled_encoders_frozen:
            for p in signal_encoder.parameters():
                p.requires_grad = False
            for p in spec_encoder.parameters():
                p.requires_grad = False
            for p in fusion_module.parameters():
                p.requires_grad = False
            signal_encoder.eval()
            spec_encoder.eval()
            fusion_module.eval()
            _decoupled_encoders_frozen = True
            _enc_params = sum(p.numel() for m in [signal_encoder, spec_encoder, fusion_module] for p in m.parameters())
            _dec_params = sum(p.numel() for p in fusion_decoder.parameters() if p.requires_grad)
            print(
                f"[ATT379c] Decoupled Stage 2 activated at epoch {current_epoch}: "
                f"frozen encoder params={_enc_params}, trainable decoder params={_dec_params}"
            )
        elif _decoupled_encoders_frozen:
            signal_encoder.eval()
            spec_encoder.eval()
            fusion_module.eval()

        if _decoupled_is_stage1 and current_epoch == 1:
            print(
                f"[ATT379c] Decoupled Stage 1 (contrastive-only): epochs 1-{_decoupled_s1_epochs}, "
                f"w_spec=0, w_fusion_cls=0, margins=off"
            )

        # Attempt 66: freeze signal branch after warm start to reduce source-overfit contamination.
        if (not signal_encoder_frozen) and (not _decoupled_encoders_frozen) and (freeze_signal_after_epoch > 0) and (current_epoch > freeze_signal_after_epoch):
            for p in signal_encoder.parameters():
                p.requires_grad = False
            signal_encoder.eval()
            signal_encoder_frozen = True
            print(f"[Signal Freeze] Enabled at epoch {current_epoch} (freeze_after={freeze_signal_after_epoch})")
        elif signal_encoder_frozen and not _decoupled_encoders_frozen:
            # Keep BN stats fixed when branch is frozen.
            signal_encoder.eval()

        # ===== (Stage) Two-stage training switch =====
        # FIX (Attempt 36): When USE_TWO_STAGE_TRAINING=False, the entire training should be Stage1
        # Previously: stage1_mode = use_two_stage AND ..., which was always False when two-stage disabled
        try:
            use_two_stage = bool(getattr(cfg, "USE_TWO_STAGE_TRAINING", True))
            stage1_epochs = int(getattr(cfg, "STAGE1_SUPERVISED_EPOCHS", 8))
        except Exception:
            use_two_stage = True
            stage1_epochs = 8
        # FIX: When two-stage disabled, always treat as Stage1 (SupCon + separation losses active)
        if use_two_stage:
            stage1_mode = bool(current_epoch <= int(stage1_epochs))
        else:
            stage1_mode = True  # Full training is Stage1
        stage2_mode = bool(use_two_stage and (not stage1_mode))
        # ATT371.1:
        # Decouple "enable SupCon Stage1" from "force CE-only path in Stage2/all epochs".
        # Old behavior (force_ce_stage2=use_supcon_stage1) can silently block multimodal_alignment_loss.
        try:
            force_ce_stage2 = bool(getattr(cfg, "FORCE_CE_STAGE2", False))
        except Exception:
            force_ce_stage2 = False
        # Keep CE-only path when explicitly forced, or in true Stage1 under two-stage regime.
        force_ce_path_now = bool(force_ce_stage2 or (use_two_stage and stage1_mode))
        
        # ===== Stage Mode Validation Log =====
        if current_epoch == 1 or current_epoch == _decoupled_s1_epochs + 1:
            print(f"[Stage Mode] epoch={current_epoch}, use_two_stage={use_two_stage}, "
                  f"stage1_epochs={stage1_epochs}, stage1_mode={stage1_mode}, stage2_mode={stage2_mode}, "
                  f"force_ce_stage2={force_ce_stage2}, force_ce_path_now={force_ce_path_now}")
            if _decoupled_enable:
                _dc_stage_name = "Stage1-Contrastive" if _decoupled_is_stage1 else "Stage2-Calibration"
                print(
                    f"[ATT379c-Decoupled] epoch={current_epoch}, stage={_dc_stage_name}, "
                    f"encoders_frozen={_decoupled_encoders_frozen}, "
                    f"s1_epochs={_decoupled_s1_epochs}, freeze_enc={_decoupled_freeze_enc}"
                )
            if current_epoch == 1:
                try:
                    _s1_clom = bool(getattr(cfg, "CORE_LOSS_ONLY_MODE", False))
                    _s1_flat = bool(getattr(cfg, "USE_FLAT_DIAG_HEAD", True))
                    _s1_hier = hasattr(fusion_decoder, 'hierarchical') and fusion_decoder.hierarchical
                    print(f"[ATT379d-StructAudit] CORE_LOSS_ONLY_MODE={_s1_clom}, "
                          f"USE_FLAT_DIAG_HEAD={_s1_flat}, decoder.hierarchical={_s1_hier}, "
                          f"DECOUPLED_TRAINING_ENABLE={_decoupled_enable}")
                    print(f"[ATT379d-StructAudit] w_signal will be: "
                          f"{'0.0 (CORE_LOSS_ONLY)' if _s1_clom else '0.3 (active)'}, "
                          f"margins: {'DISABLED (CORE_LOSS_ONLY)' if _s1_clom else 'ENABLED (from E5)'}")
                    _e_supcon_sw = float(getattr(cfg, "SUPCON_STAGE1_SPEC_WEIGHT", 0.0))
                    _e_clip_w = float(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_WEIGHT", 0.0))
                    _e_dclip_w = float(getattr(cfg, "DUAL_CLIP_SPEC_WEIGHT", 0.0))
                    _e_fpa_w = float(getattr(cfg, "FUSION_PROTO_ALIGN_WEIGHT", 0.0))
                    _e_pacl_sw = float(getattr(cfg, "PROTOTYPE_ANCHOR_WEIGHT_SPEC", 0.0))
                    _e_pacl_fw = float(getattr(cfg, "PROTOTYPE_ANCHOR_WEIGHT_FUSION", 0.0))
                    print(f"[ATT379e-LossBudgetAudit] supcon_spec_w={_e_supcon_sw:.2f}, "
                          f"clip_style_w={_e_clip_w:.2f}, dual_clip_w={_e_dclip_w:.2f}, "
                          f"fusion_proto_align_w={_e_fpa_w:.2f}, "
                          f"pacl_spec_w={_e_pacl_sw:.2f}, pacl_fus_w={_e_pacl_fw:.2f}")
                    _disabled_flags = []
                    for _flag_name in ("USE_SUPCON_LOSS", "USE_FISHR_LOSS", "USE_CCPA_LOSS",
                                       "USE_NF_SEP_LOSS", "USE_NR_INTER_COS_MARGIN_LOSS",
                                       "USE_NF_GATE_MARGIN_LOSS", "USE_SEMANTIC_DISTILL",
                                       "USE_CALIBRATION_LOSS", "USE_BATCH_NUCLEAR_NORM",
                                       "USE_DIAG_ANTICOLLAPSE_REG", "USE_TEXT_ETF_ALIGNMENT_LOSS",
                                       "USE_OUTER_MARGIN_LOSS", "USE_INNER_MARGIN_LOSS",
                                       "USE_NORMAL_MARGIN_LOSS", "USE_ROLLING_ABSORB_MARGIN_LOSS"):
                        _v = getattr(cfg, _flag_name, None)
                        _disabled_flags.append(f"{_flag_name}={_v}")
                    print(f"[ATT379f-LossSimplify] Disabled losses: {', '.join(_disabled_flags)}")
                    _pcgrad_on = bool(getattr(cfg, "USE_PCGRAD", False))
                    print(f"[ATT379f-PCGrad] enabled={_pcgrad_on}, log_interval={getattr(cfg, 'PCGRAD_LOG_INTERVAL', 50)}")
                except Exception:
                    pass
            try:
                _sig_opt = _signal_optimizer_membership(optimizer)
                print(
                    f"[SignalGradAudit] freeze_after={freeze_signal_after_epoch}, "
                    f"requires_grad={_sig_opt['requires_grad']}/{_sig_opt['total']}, "
                    f"in_optimizer={_sig_opt['in_optimizer']}/{_sig_opt['total']}, "
                    f"in_opt_requires_grad={_sig_opt['in_optimizer_requires_grad']}/{_sig_opt['total']}, "
                    f"task2_detach_decoder_input={int(task2_detach_fusion_for_decoder and _is_task2_phase)}, "
                    f"aux_ce_weight={float(getattr(cfg, 'AUX_CE_WEIGHT', 0.0)):.4f}, "
                    f"spec_obj_mode={str(getattr(cfg, 'SPEC_TRAIN_OBJECTIVE_MODE', 'legacy_aux_ce'))}"
                )
            except Exception:
                pass

        # ===== Attempt 58 pre-step: Spec semantic-alignment warmup =====
        # Rationale: after Spec branch recovers from collapse, its early gradients can still be noisy
        # and drag the stronger signal branch. We ramp spec-alignment weight from low to normal.
        try:
            spec_align_warmup_enable = bool(getattr(cfg, "SPEC_ALIGN_WARMUP_ENABLE", False))
            spec_align_warmup_start = int(getattr(cfg, "SPEC_ALIGN_WARMUP_START_EPOCH", 1))
            spec_align_warmup_end = int(getattr(cfg, "SPEC_ALIGN_WARMUP_END_EPOCH", 8))
            spec_align_warmup_min = float(getattr(cfg, "SPEC_ALIGN_WARMUP_MIN_SCALE", 0.0))
            spec_align_warmup_max = float(getattr(cfg, "SPEC_ALIGN_WARMUP_MAX_SCALE", 1.0))
            spec_align_base = float(getattr(cfg, "SPEC_ALIGN_BASE_WEIGHT", 0.3))
        except Exception:
            spec_align_warmup_enable = False
            spec_align_warmup_start = 1
            spec_align_warmup_end = 8
            spec_align_warmup_min = 0.0
            spec_align_warmup_max = 1.0
            spec_align_base = 0.3
        if spec_align_warmup_enable:
            if current_epoch <= spec_align_warmup_start:
                spec_align_scale = spec_align_warmup_min
            elif current_epoch >= spec_align_warmup_end:
                spec_align_scale = spec_align_warmup_max
            else:
                denom = max(1, spec_align_warmup_end - spec_align_warmup_start)
                p = float(current_epoch - spec_align_warmup_start) / float(denom)
                spec_align_scale = spec_align_warmup_min + p * (spec_align_warmup_max - spec_align_warmup_min)
            spec_align_scale = float(max(0.0, min(1.0, spec_align_scale)))
        else:
            spec_align_scale = 1.0
        current_w_spec_align = float(spec_align_base) * float(spec_align_scale)
        # ATT379b: 每 epoch 打印 w_spec（验证 warmup 禁用是否生效）
        if current_epoch <= 3 or current_epoch % 5 == 0:
            print(
                f"[Spec Align Warmup] enabled={int(spec_align_warmup_enable)} "
                f"w_spec={current_w_spec_align:.4f} "
                f"(base={spec_align_base:.4f}, scale={spec_align_scale:.4f}, "
                f"start={spec_align_warmup_start}, end={spec_align_warmup_end})"
            )

        # ===== Prompt/Prototype tuning schedule (shared Δproto) =====
        # Enable shared residual after PROMPT_TUNE_START_EPOCH to avoid early noisy updates.
        try:
            if hasattr(fusion_decoder, "set_prototype_residual_shared_requires_grad"):
                enable_shared = (not stage1_mode) and (not in_proto_cooldown) and bool(getattr(cfg, "USE_PROMPT_TUNING", True)) and (
                    current_epoch >= int(getattr(cfg, "PROMPT_TUNE_START_EPOCH", 3))
                )
                fusion_decoder.set_prototype_residual_shared_requires_grad(enable_shared)
            if hasattr(fusion_decoder, "set_prototype_residual_domain_requires_grad"):
                # Domain residual is reserved for target-domain supervised finetune
                fusion_decoder.set_prototype_residual_domain_requires_grad(False)
        except Exception:
            pass

        # ===== (S1) Multi-prototype extra bank schedule =====
        try:
            use_multi_proto = bool(getattr(cfg, "USE_MULTI_PROTOTYPE_HEAD", True))
            multi_start = int(getattr(cfg, "MULTI_PROTOTYPE_START_EPOCH", 5))
        except Exception:
            use_multi_proto = True
            multi_start = 5
        if use_multi_proto and hasattr(fusion_decoder, "set_multi_prototype_requires_grad"):
            fusion_decoder.set_multi_prototype_requires_grad((not stage1_mode) and (not in_proto_cooldown) and bool(current_epoch >= int(multi_start)))

        # ===== (S-B) Diagnosis class-bias schedule =====
        try:
            use_bias = bool(getattr(cfg, "USE_DIAG_CLASS_BIAS", True))
            bias_start = int(getattr(cfg, "DIAG_CLASS_BIAS_TRAIN_START_EPOCH", 10))
        except Exception:
            use_bias = True
            bias_start = 10
        if use_bias and hasattr(fusion_decoder, "set_diag_class_bias_requires_grad"):
            # Class-bias is a low-risk correction head; keep it trainable even during proto cooldown
            # so the model has a chance to escape "all→Normal/Inner" collapse.
            fusion_decoder.set_diag_class_bias_requires_grad((not stage1_mode) and bool(current_epoch >= int(bias_start)))

        # ===== (Train/Eval consistency) Prototype head schedule =====
        try:
            proto_train_start = int(getattr(cfg, "PROTOTYPE_HEAD_TRAIN_START_EPOCH", 3))
        except Exception:
            proto_train_start = 3
        # IMPORTANT (train/eval consistency): even during "cooldown", keep using the same head as evaluation.
        # Cooldown should freeze high-risk adapters (prompt/multi-proto/bias/linear), not switch the decision head.
        use_proto_train_epoch = bool(getattr(cfg, "USE_PROTOTYPE_HEAD", True)) and (current_epoch >= int(proto_train_start)) and (not stage1_mode)

        # ===== (D2) Late-enable linear residual diagnosis head (source training) =====
        # Goal: avoid early destabilization while still learning a domain-robust correction head.
        try:
            use_lin = bool(getattr(cfg, "USE_DIAG_LINEAR_RESIDUAL", True))
            lin_start = int(getattr(cfg, "DIAG_LINEAR_TRAIN_START_EPOCH", 8))
            lin_warm = int(getattr(cfg, "DIAG_LINEAR_WARMUP_EPOCHS", 5))
        except Exception:
            use_lin = True
            lin_start = 8
            lin_warm = 5
        if use_lin and hasattr(fusion_decoder, "set_diag_linear_requires_grad") and hasattr(fusion_decoder, "set_diag_linear_ramp"):
            # Two-stage policy:
            # - Stage1: train supervised diagnosis head from the start (pure supervised convergence)
            # - Stage2: freeze the head to stabilize while doing semantic alignment on embeddings
            if stage1_mode:
                fusion_decoder.set_diag_linear_requires_grad(True)
                fusion_decoder.set_diag_linear_ramp(1.0)
            elif stage2_mode:
                fusion_decoder.set_diag_linear_requires_grad(False)
                fusion_decoder.set_diag_linear_ramp(1.0)
            else:
                # Legacy schedule
                if current_epoch >= int(lin_start):
                    fusion_decoder.set_diag_linear_requires_grad(True)
                    denom = max(1, int(lin_warm))
                    ramp = float(current_epoch - int(lin_start) + 1) / float(denom)
                    fusion_decoder.set_diag_linear_ramp(max(0.0, min(1.0, ramp)))
                else:
                    fusion_decoder.set_diag_linear_requires_grad(False)
                    fusion_decoder.set_diag_linear_ramp(0.0)

        # ===== (S-SEM) Semantic adapter schedule =====
        try:
            use_sem_adapter = bool(getattr(cfg, "USE_SEM_ADAPTER", True))
            sem_start = int(getattr(cfg, "STAGE1_SUPERVISED_EPOCHS", 8)) + 1
        except Exception:
            use_sem_adapter, sem_start = True, 9
        if use_sem_adapter and hasattr(fusion_decoder, "set_sem_adapter_requires_grad"):
            # Enable semantic adapter only in Stage2 (after supervised head converges)
            fusion_decoder.set_sem_adapter_requires_grad(bool(stage2_mode and (current_epoch >= int(sem_start))))

        # ===== (2) Stage1 NF-only: freeze fault-type heads so Stage1 trains only Normal/Fault =====
        try:
            stage1_nf_only = bool(getattr(cfg, "STAGE1_NF_ONLY", False))
        except Exception:
            stage1_nf_only = False
        if stage1_mode and stage1_nf_only:
            fault_type_keywords = ("classifier_fault_io", "classifier_fault_roll", "classifier_normal", "classifier_fault_shared", "classifier_fault", "classifier_fault_hust", "classifier_fault_hit", "diag_linear_head", "diag_linear_gate")
            for _name, _p in fusion_decoder.named_parameters():
                if any(k in _name for k in fault_type_keywords):
                    _p.requires_grad = False
        elif not stage2_mode:
            # Stage1 but not NF-only, or no two-stage: ensure fault-type heads trainable
            for _name, _p in fusion_decoder.named_parameters():
                if "classifier_fault" in _name or "classifier_normal" in _name or "gate_adapter" in _name or "cls_adapter" in _name or "classifier_nf_global" in _name:
                    _p.requires_grad = True

        # ===== (Stage2) Stabilize rolling head by delayed unfreeze =====
        try:
            roll_freeze_epochs = int(getattr(cfg, "ROLLING_STAGE2_FREEZE_EPOCHS", 0))
        except Exception:
            roll_freeze_epochs = 0
        if stage2_mode and roll_freeze_epochs > 0:
            try:
                stage2_epoch_idx = max(1, int(current_epoch) - int(stage1_epochs))
            except Exception:
                stage2_epoch_idx = 1
            freeze_roll = bool(stage2_epoch_idx <= int(roll_freeze_epochs))
            for _name, _p in fusion_decoder.named_parameters():
                if ("classifier_fault_roll" in _name) or ("roll_adapter" in _name):
                    _p.requires_grad = (not freeze_roll)
            if freeze_roll:
                print(f"[Stage2] Freeze rolling head: epoch {current_epoch} (stage2 idx {stage2_epoch_idx}/{roll_freeze_epochs})")
        # ===== (Stage2) Optionally freeze diagnosis backbone (STAGE2_FREEZE_BACKBONE=False lets fault-type refine z) =====
        try:
            stage2_freeze_backbone = bool(getattr(cfg, "STAGE2_FREEZE_BACKBONE", True))
            if stage2_mode and stage2_freeze_backbone:
                for _m in (signal_encoder, spec_encoder, fusion_module):
                    try:
                        for _p in _m.parameters():
                            _p.requires_grad = False
                    except Exception:
                        pass
                allow_keywords = ("sem_adapter", "prototype_residual_shared", "prototype_residual_domain", "prototype_extra", "diag_class_bias")
                try:
                    use_flat_diag = bool(getattr(cfg, "USE_FLAT_DIAG_HEAD", False))
                except Exception:
                    use_flat_diag = False
                if use_flat_diag:
                    allow_keywords = allow_keywords + ("classifier",)
                if stage1_nf_only:
                    allow_keywords = allow_keywords + ("classifier_fault", "classifier_normal", "gate_adapter", "classifier_nf", "cls_adapter", "nf_calibrator")
                try:
                    for _name, _p in fusion_decoder.named_parameters():
                        if any(k in _name for k in allow_keywords):
                            continue
                        if ("nf_calibrator" in _name):
                            continue
                        _p.requires_grad = False
                except Exception:
                    pass
        except Exception:
            pass

        # ===== (Debug) Trainable-parameter sanity check =====
        # Purpose: detect "Stage2 freezing" caused by accidentally freezing ALL trainable params.
        # Log on epoch 1, every 5 epochs, and at Stage2 entry (stage1_epochs+1).
        if current_epoch == 1 or (current_epoch % 5 == 0) or (use_two_stage and (current_epoch == int(stage1_epochs) + 1)):
            def _count(_m):
                try:
                    total = 0
                    trainable = 0
                    for _p in _m.parameters():
                        n = int(_p.numel())
                        total += n
                        if bool(getattr(_p, "requires_grad", False)):
                            trainable += n
                    return total, trainable
                except Exception:
                    return 0, 0

            sig_total, sig_tr = _count(signal_encoder)
            spec_total, spec_tr = _count(spec_encoder)
            fus_total, fus_tr = _count(fusion_module)
            dec_total, dec_tr = _count(fusion_decoder)
            all_total = sig_total + spec_total + fus_total + dec_total
            all_tr = sig_tr + spec_tr + fus_tr + dec_tr

            _dc_tag = ""
            if _decoupled_enable:
                _dc_tag = f", decoupled={'S1-Contrastive' if _decoupled_is_stage1 else 'S2-Calibration'}"
            print(
                "[Trainable Params] "
                f"epoch={current_epoch}, stage1_mode={stage1_mode}, stage2_mode={stage2_mode}{_dc_tag} | "
                f"Signal={sig_tr}/{sig_total}, Spec={spec_tr}/{spec_total}, Fusion={fus_tr}/{fus_total}, "
                f"Decoder={dec_tr}/{dec_total} | All={all_tr}/{all_total}"
            )
            if stage2_mode and all_tr <= 0:
                print(
                    f"[Stage2-Freezing] WARNING: stage2_mode=True but ALL params are frozen at epoch {current_epoch}. "
                    "Training/evaluation metrics may become identically constant."
                )

            # ATT397: log SAM + Episodic config at epoch start
            if current_epoch <= 2:
                _sam_on = hasattr(optimizer, 'first_step')
                _ep_on = bool(getattr(cfg, "USE_EPISODIC_TRAINING", False))
                print(
                    f"[ATT397-Config] E{current_epoch} "
                    f"SAM={'ON(rho='+str(getattr(cfg,'SAM_RHO',0.05))+')' if _sam_on else 'OFF'}, "
                    f"Episodic={'ON(w='+str(getattr(cfg,'EPISODIC_PROTO_LOSS_WEIGHT',0.5))+',T='+str(getattr(cfg,'EPISODIC_TEMPERATURE',0.07))+')' if _ep_on else 'OFF'}, "
                    f"optimizer_type={type(optimizer).__name__}"
                )

        # 只在指定轮数内做贝叶斯原型更新，之后冻结
        bayesian_update_active = bool(use_bayesian_prototypes and (bayesian_update_until_epoch is None or current_epoch <= bayesian_update_until_epoch))
        
        # ===== 动态调整 w_fusion_cls 和学习率（支持渐进式学习率） =====
        # ATT423-P0: preserve differential lr ratios when updating schedule
        def _update_lr_preserving_ratios(_optimizer, _new_base_lr):
            """Update lr while preserving per-group lr ratios set by ATT423-P0."""
            if _optimizer is None:
                return
            _old_base = _optimizer.param_groups[0]["lr"]
            if abs(_old_base) < 1e-12:
                for _pg in _optimizer.param_groups:
                    _pg["lr"] = _new_base_lr
                return
            _ratio = _new_base_lr / _old_base
            for _pg in _optimizer.param_groups:
                _pg["lr"] = _pg["lr"] * _ratio

        # 优先使用渐进式学习率（如果启用）
        if use_progressive_lr and progressive_lr_schedule:
            current_lr = lr_early
            for epoch_threshold in sorted(progressive_lr_schedule.keys(), reverse=True):
                if current_epoch >= epoch_threshold:
                    current_lr = progressive_lr_schedule[epoch_threshold]
                    break
            _update_lr_preserving_ratios(optimizer, current_lr)
            if current_epoch <= schedule_epoch_threshold:
                current_w_fusion_cls = w_fusion_cls_early
            else:
                current_w_fusion_cls = w_fusion_cls_late
        elif use_schedule:
            if current_epoch <= schedule_epoch_threshold:
                current_w_fusion_cls = w_fusion_cls_early
                current_lr = lr_early
            else:
                current_w_fusion_cls = w_fusion_cls_late
                current_lr = lr_late
            _update_lr_preserving_ratios(optimizer, current_lr)
        else:
            current_w_fusion_cls = w_fusion_cls_early
            current_lr = lr_early

        # ATT463: Split Architecture — disable fusion_cls loss and increase signal CE
        _split_arch_on = bool(getattr(cfg, "SPLIT_ARCH_ENABLE", False))
        if _split_arch_on:
            current_w_fusion_cls = 0.0
            _w_signal_align = float(getattr(cfg, "SPLIT_ARCH_SIGNAL_WEIGHT", 1.0))
            if current_epoch == start_epoch:
                print(f"[ATT463-SplitArch] Active: w_fusion_cls=0.0, w_signal={_w_signal_align:.2f}")

        _att488_zcls_ce_skip_epoch = _fusion_zcls_ce_disabled_now(cfg, stage1_mode)
        if _att488_zcls_ce_skip_epoch:
            current_w_fusion_cls = 0.0
            if current_epoch == start_epoch:
                print(
                    "[ATT488-ZClsCE] pretraining z_cls/Fusion Cls CE disabled; "
                    "z_metric geometry paths remain active."
                )

        # ATT432: per-epoch diagnostic audit
        if current_epoch <= 3 or current_epoch % 5 == 0:
            _lr_parts = []
            for _pg in optimizer.param_groups:
                _pn = _pg.get("name", "?")
                _lr_parts.append(f"{_pn}={_pg['lr']:.2e}")
            _proto_cls_on = bool(getattr(cfg, "USE_PROTO_CLASSIFIER", False))
            _proto_cos_align_on = bool(getattr(cfg, "USE_PROTO_COSINE_ALIGN", False))
            _proto_cos_align_w_val = float(getattr(cfg, "PROTO_COSINE_ALIGN_WEIGHT", 0.10))
            _mslr_fusion_ep = float(getattr(cfg, 'MSLR_FUSION_MULT', 3.0))
            _sg_anneal_on = bool(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_ENABLE", False))
            _sg_st = int(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_START_EPOCH", 1))
            _sg_ed = int(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_END_EPOCH", 15))
            _sg_lr_s = max(0.0, min(1.0, float(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_START", 1.0))))
            _sg_lr_e = max(0.0, min(1.0, float(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_END", 0.0))))
            if _sg_ed < _sg_st:
                _sg_ed = _sg_st
            if current_epoch <= _sg_st:
                _sg_lr_now = _sg_lr_s
            elif current_epoch >= _sg_ed:
                _sg_lr_now = _sg_lr_e
            else:
                _sg_p = float(current_epoch - _sg_st) / float(max(1, _sg_ed - _sg_st))
                _sg_lr_now = _sg_lr_s + (_sg_lr_e - _sg_lr_s) * _sg_p
            _fd_grad_ratio_audit = parse_fusion_decoder_grad_live_ratio(
                getattr(cfg, "FUSION_DECODER_INPUT_GRAD_LIVE_RATIO", None)
            )
            if _fd_grad_ratio_audit is not None:
                _dec_grad_audit_str = (
                    f"dec_grad_live=fixed({_fd_grad_ratio_audit:.2f})"
                )
            else:
                _dec_grad_audit_str = (
                    f"dec_stopgrad_anneal={_sg_anneal_on}(E{_sg_st}->{_sg_ed},live={_sg_lr_now:.2f})"
                )
            print(
                f"[ATT437-EpochAudit] E{current_epoch} "
                f"proto_cls={_proto_cls_on}(DISABLED) "
                f"proto_cos_align={_proto_cos_align_on}(w={_proto_cos_align_w_val:.3f}) "
                f"w_fusion_cls={current_w_fusion_cls:.3f} "
                f"zcls_ce_skip={int(_att488_zcls_ce_skip_epoch)} "
                f"w_signal_align={float(getattr(cfg, 'W_SIGNAL_ALIGN', 0.3)):.3f} "
                f"phys_align_w={float(getattr(cfg, 'PHYS_FEATURE_ALIGN_WEIGHT', 0.0)):.3f} "
                f"gate_ent_w={float(getattr(cfg, 'GATE_ENTROPY_WEIGHT', 0.0)):.3f} "
                f"sep_w={float(getattr(cfg, 'LEARNABLE_PROTO_SEP_WEIGHT', 0.0)):.3f} "
                f"anchor_w={float(getattr(cfg, 'LEARNABLE_PROTO_ANCHOR_WEIGHT', 0.0)):.3f} "
                f"MSLR_fusion={_mslr_fusion_ep}x "
                f"norm_mode={('att473-specialist-arbiter' if bool(getattr(cfg, 'ATT473_SPECIALIST_ARBITER_ENABLE', False)) else ('att472-shared-neck' if bool(getattr(cfg, 'ATT472_TWO_END_SHARED_NECK_ENABLE', False)) else ('shared-proj' if bool(getattr(cfg, 'USE_SHARED_PROJECTION_FUSION', False)) else ('decision-level' if bool(getattr(cfg, 'USE_DECISION_LEVEL_FUSION', False)) else ('gated-mean-residual' if (bool(getattr(cfg, 'USE_GATED_ADDITIVE_FUSION', False)) and bool(getattr(cfg, 'FUSION_MEAN_RESIDUAL_ENABLE', False))) else 'gated-additive')))))} "
                f"OGM_GE={bool(getattr(cfg, 'OGM_GE_ENABLE', False))}(a={float(getattr(cfg, 'OGM_GE_ALPHA', 1.0)):.1f},start=E{int(getattr(cfg, 'OGM_GE_START_EPOCH', 2))}) "
                f"ResSkip={bool(getattr(cfg, 'FUSION_RESIDUAL_SKIP_ENABLE', False))}(init={float(getattr(cfg, 'FUSION_RESIDUAL_SKIP_INIT_ALPHA', 0.1)):.2f}) "
                f"SplitArch={bool(getattr(cfg, 'SPLIT_ARCH_ENABLE', False))}(sig_w={float(getattr(cfg, 'SPLIT_ARCH_SIGNAL_WEIGHT', 1.0)):.2f}) "
                f"fus_fwd_detach={bool(getattr(cfg, 'FUSION_FORWARD_DETACH_ENCODER_INPUTS', False))} "
                f"ATT479_fus_supcon={bool(getattr(cfg, 'FUSION_SUPCON_ENABLE', False))}(w={float(getattr(cfg, 'FUSION_SUPCON_WEIGHT', 1.0)):.2f},"
                f"geom={str(getattr(cfg, 'FUSION_GEOMETRY_LOSS_MODE', 'silhouette'))}) "
                f"dec_in_stopgrad={bool(getattr(cfg, 'FUSION_DECODER_INPUT_STOPGRAD_ENABLE', False))} "
                f"{_dec_grad_audit_str} "
                f"sem_metric_live={bool(getattr(cfg, 'FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH', True))} "
                f" metric_lp_ce={bool(getattr(cfg, 'FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE', False))}"
                f"(w={float(getattr(cfg, 'FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT', 0.0)):.2f}) "
                f"AGIT_preset={bool(getattr(cfg, 'ATT471_AGIT_ENABLE', False))} "
                f"ATT472_preset={bool(getattr(cfg, 'ATT472_TWO_END_SHARED_NECK_ENABLE', False))} "
                f"ATT473_preset={bool(getattr(cfg, 'ATT473_SPECIALIST_ARBITER_ENABLE', False))} "
                f"DualContract={bool(getattr(cfg, 'DUAL_CONTRACT_ENABLE', False))} "
                f"DualContractCkpt={bool(getattr(cfg, 'DUAL_CONTRACT_SAVE_EXTRA_CKPT', False))} "
                f"cm_align_w={float(getattr(cfg, 'FUSION_CROSS_MODAL_ALIGN_WEIGHT', 0.0)):.3f} "
                f"VICReg={bool(getattr(cfg, 'USE_GLOBAL_VICREG_LOSS', False))} "
                f"CLIP_w={float(getattr(cfg, 'CLIP_STYLE_CONTRASTIVE_WEIGHT', 0.0)):.2f} "
                f"ETF={bool(getattr(cfg, 'USE_SIMPLEX_ETF_PROTOTYPES', False))} "
                f"SH_CosProto={bool(getattr(cfg, 'SPECHEAD_USE_COSINE_PROTO', False))} "
                f"SPEC_HEAD_MODE={getattr(cfg, 'SPEC_HEAD_MODE', 'mlp')} "
                f"OrderNorm={bool(getattr(cfg, 'SPEC_ORDER_NORM_ENABLE', False))}(max={float(getattr(cfg, 'SPEC_ORDER_NORM_MAX_ORDERS', 200)):.0f}) "
                f"FreqAlign={bool(getattr(cfg, 'SPEC_FREQ_ALIGN_ENABLE', False))}(hz={float(getattr(cfg, 'SPEC_FREQ_ALIGN_MAX_HZ', 6000)):.0f}) "
                f"FusReg=[var_floor={bool(getattr(cfg, 'USE_FUSION_VAR_FLOOR_LOSS', True))}"
                f"(w={float(getattr(cfg, 'FUSION_VAR_FLOOR_WEIGHT', 0.05)):.2f},min={float(getattr(cfg, 'FUSION_VAR_FLOOR_MIN', 0.001)):.4f}),"
                f" vicreg={bool(getattr(cfg, 'USE_FUSION_VICREG_LOSS', True))},"
                f" inter_cos={bool(getattr(cfg, 'USE_FUSION_INTER_COS_MARGIN_LOSS', True))}"
                f"(w={float(getattr(cfg, 'FUSION_INTER_COS_MARGIN_WEIGHT', 0.08)):.2f},tgt={float(getattr(cfg, 'FUSION_INTER_COS_MARGIN_TARGET_MAX', 0.15)):.2f})] "
                f"PidrScale=[freq_enc={float(getattr(cfg, 'PHYSICAL_FREQ_POSITION_ENCODING_SCALE', 1.0)):.2f},"
                f" mixstyle_a={float(getattr(cfg, 'MIXSTYLE_ALPHA', 0.15)):.2f},"
                f" domain_cond={float(getattr(cfg, 'DOMAIN_COND_STRENGTH', 0.5)):.2f}] "
                f"lr=[{', '.join(_lr_parts)}]"
            )
            # ATT453: Proto SpecHead diagnostics
            if getattr(spec_encoder, 'spec_prototypes', None) is not None:
                try:
                    _sp = spec_encoder.spec_prototypes.detach()
                    _sp_n = F.normalize(_sp, dim=-1)
                    _n_cls = _sp.size(0)
                    _cos_mat = _sp_n @ _sp_n.T
                    _inter_cos = (_cos_mat.sum() - _cos_mat.trace()) / max(1, _n_cls * (_n_cls - 1))
                    _diag_cos = _cos_mat.diag()
                    _pairs = []
                    for _i in range(_n_cls):
                        for _j in range(_i + 1, _n_cls):
                            _pairs.append(f"({_i},{_j})={_cos_mat[_i,_j]:.4f}")
                    print(f"[ATT453-ProtoSpec] E{current_epoch} "
                          f"inter_cos={_inter_cos:.4f} "
                          f"proto_norm=[{', '.join(f'{_sp[i].norm():.3f}' for i in range(_n_cls))}] "
                          f"pairwise=[{', '.join(_pairs)}]")
                    if hasattr(spec_encoder, '_spec_proto_temperature'):
                        print(f"[ATT453-ProtoSpec] T={spec_encoder._spec_proto_temperature:.4f}")
                except Exception as _e:
                    print(f"[ATT453-ProtoSpec] diag error: {_e}")
            # Fusion weight / gate distribution
            if hasattr(fusion_module, '_last_weights') and fusion_module._last_weights is not None:
                _fw = fusion_module._last_weights
                _is_sp = isinstance(fusion_module, SharedProjectionFusion) if 'SharedProjectionFusion' in dir() else hasattr(fusion_module, '_last_cross_modal_cos')
                _fmode = "shared_proj" if _is_sp else ("fixed" if getattr(fusion_module, 'fixed_equal_weight', False) else "adaptive")
                _cm_cos_str = ""
                if hasattr(fusion_module, '_last_cross_modal_cos') and fusion_module._last_cross_modal_cos is not None:
                    _cm_cos = fusion_module._last_cross_modal_cos
                    _cm_cos_str = f" cm_cos={_cm_cos.mean():.4f}±{_cm_cos.std():.4f}"
                print(f"[ATT460-FusionWeight] E{current_epoch} mode={_fmode} "
                      f"w_sig={_fw[:, 0].mean():.4f}±{_fw[:, 0].std():.4f} "
                      f"w_spec={_fw[:, 1].mean():.4f}±{_fw[:, 1].std():.4f} "
                      f"w_sig_range=[{_fw[:, 0].min():.4f},{_fw[:, 0].max():.4f}]"
                      f"{_cm_cos_str}")
            if hasattr(fusion_module, '_last_gate_values') and fusion_module._last_gate_values is not None:
                _gv = fusion_module._last_gate_values
                _gate_l2_diag = (_gv ** 2).mean()
                _gate_abs = _gv.abs()
                print(f"[ATT437-Gate] E{current_epoch} gate_mean={_gv.mean():.4f} "
                      f"gate_std={_gv.std():.4f} gate_min={_gv.min():.4f} gate_max={_gv.max():.4f} "
                      f"abs_mean={_gate_abs.mean():.4f} l2={_gate_l2_diag:.4f}")
            # Learnable prototype drift from text anchor
            if learnable_proto_module is not None:
                try:
                    _lp = learnable_proto_module.protos.detach()
                    _ta = learnable_proto_module.text_anchor.detach()
                    _lp_n = F.normalize(_lp, dim=-1)
                    _ta_n = F.normalize(_ta, dim=-1)
                    _per_cls_cos = (_lp_n * _ta_n).sum(dim=-1)
                    _inter_cos = _lp_n @ _lp_n.t()
                    _inter_mask = ~torch.eye(_lp_n.size(0), dtype=torch.bool, device=_lp_n.device)
                    _inter_offdiag = _inter_cos[_inter_mask]
                    print(f"[ATT437-LearnableProto] E{current_epoch} "
                          f"anchor_cos={_per_cls_cos.tolist()} "
                          f"inter_cos_mean={_inter_offdiag.mean():.4f} "
                          f"inter_cos_max={_inter_offdiag.max():.4f}")
                except Exception:
                    pass
            # Physical feature prototype centroid quality
            if phys_proto_module is not None:
                try:
                    _pc_cos = phys_proto_module.centroid_cosine_matrix()
                    _pc_offdiag = _pc_cos[~torch.eye(_pc_cos.size(0), dtype=torch.bool, device=_pc_cos.device)]
                    print(f"[ATT437-PhysProto] E{current_epoch} centroid_cos_mean={_pc_offdiag.mean():.4f} "
                          f"centroid_cos_max={_pc_offdiag.max():.4f} counts={phys_proto_module.class_counts.tolist()}")
                except Exception:
                    pass

        # ===== 渐进式权重调整 =====
        if use_progressive_weights:
            weight_scale = compute_progressive_weight_scale(
                current_epoch, progressive_weight_total_epochs, initial_weight_scale, weight_schedule
            )
            # 将权重从基础值缩放到目标值：weight = base + (target - base) * scale
            # 这里base是1.0（无权重），target是precomputed_class_weights
            class_weights = 1.0 + (base_class_weights - 1.0) * weight_scale
            class_weights = class_weights.to(device)
        else:
            class_weights = precomputed_class_weights

        # ===== sampler-first loss weighting =====
        # If WeightedSampler is enabled, do NOT additionally use per-class loss weights (double compensation),
        # which is a common cause of "all → Rolling" collapse in this project.
        effective_use_class_weights_in_loss = bool(use_class_weights_in_loss)
        try:
            import config as cfg_module
            if bool(getattr(cfg_module, "USE_WEIGHTED_SAMPLER", False)):
                effective_use_class_weights_in_loss = False
        except Exception:
            pass
        class_weights_for_loss = class_weights if (effective_use_class_weights_in_loss and class_weights is not None) else None
        try:
            import config as cfg
            use_balanced_softmax = bool(getattr(cfg, "USE_BALANCED_SOFTMAX", False))
            core_loss_only_mode = bool(getattr(cfg, "CORE_LOSS_ONLY_MODE", False))
            core_keep_task1_cwru_invariance = bool(getattr(cfg, "CORE_KEEP_TASK1_CWRU_INVARIANCE", True))
            _task1_cwru_fishr_ids = tuple(getattr(cfg, "TASK1_CWRU_FISHR_SOURCE_DOMAIN_IDS", (0, 1)))
            _task1_cwru_ccpa_ids = tuple(getattr(cfg, "TASK1_CWRU_CCPA_SOURCE_DOMAIN_IDS", (0, 1)))
        except Exception:
            use_balanced_softmax = False
            core_loss_only_mode = False
            core_keep_task1_cwru_invariance = True
            _task1_cwru_fishr_ids = (0, 1)
            _task1_cwru_ccpa_ids = (0, 1)
        if use_balanced_softmax:
            class_weights_for_loss = None
        try:
            _phase_tag = str(checkpoint_tag).lower()
        except Exception:
            _phase_tag = ""
        is_task1_cwru_phase = ("cwru" in _phase_tag) and ("task2" not in _phase_tag)
        if core_loss_only_mode:
            # Keep fusion CE stable in core mode (avoid extra prior-shift interactions).
            use_balanced_softmax = False
            use_focal_loss = False
            if current_epoch == 1:
                print(
                    f"[CoreLossMode] enabled=1 phase={_phase_tag or 'unknown'} "
                    f"keep_task1_cwru_invariance={int(core_keep_task1_cwru_invariance)} "
                    f"allowed={getattr(cfg, 'CORE_LOSS_ALLOWED_TERMS', ('fusion','aux_spec','pacl_spec','pacl_fusion','global_decorr','text_etf_align','clip_style'))}"
                )
        
        # ===== DRW (Deferred Re-Weighting) for LDAM =====
        # Before DRW_EPOCH: use uniform weights; after: use class weights
        drw_class_weights = None
        if use_ldam_loss and use_drw:
            if current_epoch < drw_epoch:
                # Uniform weights for early epochs (representation learning phase)
                drw_class_weights = None
                if current_epoch == 1:
                    print(f"[DRW] Epoch {current_epoch}: using uniform weights (deferred re-weighting phase)")
            else:
                # After DRW_EPOCH: use class weights
                drw_class_weights = class_weights.to(device) if class_weights is not None else None
                if current_epoch == drw_epoch:
                    print(f"[DRW] Epoch {current_epoch}: switching to class re-weighting")
            # Update LDAM loss function's weight if needed
            if ldam_loss_fn is not None:
                ldam_loss_fn.weight = drw_class_weights
        
        epoch_total = 0.0
        epoch_signal = 0.0
        epoch_spec = 0.0
        epoch_fusion = 0.0
        epoch_fusion_cls = 0.0
        epoch_margin = 0.0
        epoch_fusion_proto_align = 0.0
        epoch_fusion_supcon = 0.0
        epoch_metric_lp_ce = 0.0
        epoch_pacl_spec = 0.0
        epoch_pacl_fusion = 0.0
        epoch_pacl_spec_acc = 0.0
        epoch_pacl_fusion_acc = 0.0
        epoch_pacl_batches = 0
        epoch_pacl_spec_per_class_acc = {}
        epoch_adapter_loss = 0.0
        epoch_adapter_acc = 0.0
        epoch_adapter_batches = 0
        # ATT429-P2: hierarchical CE tracking
        epoch_h_nf_ce = 0.0
        epoch_h_fault_ce = 0.0
        epoch_h_batches = 0
        # ATT431: disentangle/variance tracking removed (falsified)
        epoch_dual_orth = 0.0
        epoch_dual_dc_spe = 0.0
        epoch_dual_dc_inv_train = 0.0
        epoch_dual_confusion = 0.0
        epoch_spec_variance = 0.0
        epoch_dual_batches = 0
        epoch_pacl_spec_per_class_pos_cos = {}
        epoch_pacl_spec_per_class_count = {}
        epoch_clip_style = 0.0
        epoch_clip_style_diag_cos = 0.0
        epoch_clip_style_offdiag_cos = 0.0
        epoch_clip_style_batches = 0
        epoch_nr_inter_cos_margin = 0.0
        epoch_nr_inter_cos_hard = 0.0
        epoch_fusion_var_floor = 0.0
        epoch_fusion_vicreg = 0.0
        epoch_fusion_inter_cos_margin = 0.0
        epoch_pairwise_margin_sum = 0.0
        epoch_pairwise_margin_count = 0
        epoch_moe_total = 0.0
        epoch_moe_load_balance = 0.0
        epoch_moe_entropy = 0.0
        epoch_moe_diversity = 0.0
        epoch_consistency = 0.0
        epoch_roll_dann = 0.0
        roll_dann_expected_batches = 0
        roll_dann_used_batches = 0
        roll_dann_missing_batches = 0
        roll_dann_no_fault_batches = 0
        epoch_roll_decorr = 0.0
        roll_decorr_expected_batches = 0
        roll_decorr_used_batches = 0
        roll_decorr_missing_batches = 0
        roll_decorr_no_fault_batches = 0
        roll_decorr_small_batches = 0
        roll_decorr_var_sum = 0.0
        roll_decorr_cov_sum = 0.0
        epoch_roll_decorr_in = 0.0
        roll_decorr_in_expected_batches = 0
        roll_decorr_in_used_batches = 0
        roll_decorr_in_missing_batches = 0
        roll_decorr_in_no_fault_batches = 0
        roll_decorr_in_small_batches = 0
        roll_decorr_in_var_sum = 0.0
        roll_decorr_in_cov_sum = 0.0
        epoch_global_decorr = 0.0
        global_decorr_expected_batches = 0
        global_decorr_used_batches = 0
        global_decorr_missing_batches = 0
        global_decorr_small_batches = 0
        global_decorr_var_sum = 0.0
        global_decorr_cov_sum = 0.0
        global_decorr_pre_std_sum = 0.0
        global_decorr_pre_std_batches = 0
        epoch_fishr = 0.0
        fishr_expected_batches = 0
        fishr_used_batches = 0
        fishr_skipped_batches = 0
        fishr_grad_var_sum = 0.0
        fishr_split_batches = 0
        epoch_ccpa = 0.0
        ccpa_expected_batches = 0
        ccpa_used_batches = 0
        ccpa_skipped_batches = 0
        epoch_grad_norm_sum = 0.0
        epoch_grad_norm_steps = 0
        epoch_grad_norm_detail_steps = 0
        epoch_grad_norm_signal_sum = 0.0
        epoch_grad_norm_spec_sum = 0.0
        epoch_grad_norm_fusion_sum = 0.0
        epoch_grad_norm_decoder_sum = 0.0
        epoch_grad_norm_signal_count_sum = 0
        epoch_grad_norm_spec_count_sum = 0
        epoch_grad_norm_fusion_count_sum = 0
        epoch_grad_norm_decoder_count_sum = 0
        epoch_grad_clip_batches = 0
        epoch_grad_clip_clipped_batches = 0
        epoch_grad_clip_pre_norm_sum = 0.0
        epoch_grad_clip_post_norm_sum = 0.0
        epoch_grad_clip_param_count_sum = 0
        epoch_loss_term_sum = {}
        epoch_loss_term_count = {}
        epoch_loss_info_sum = {}
        epoch_loss_info_count = {}
        epoch_domain_hist = {}
        epoch_domain_label_hist = {}
        epoch_domain_cond_examples = {}
        ccpa_pair_sum = 0
        ccpa_class_sum = 0
        ccpa_cos_sum = 0.0
        epoch_global_uniformity = 0.0
        global_uniformity_expected_batches = 0
        global_uniformity_used_batches = 0
        global_uniformity_missing_batches = 0
        global_uniformity_small_batches = 0
        global_uniformity_d2_sum = 0.0
        epoch_z_cls_var_sum = 0.0
        epoch_z_res_var_sum = 0.0
        epoch_z_gate_var_sum = 0.0
        z_cls_var_batches = 0
        z_res_var_batches = 0
        z_gate_var_batches = 0
        
        # Prototype Separation Loss 统计
        epoch_proto_sep_loss = 0.0
        proto_sep_batches = 0
        # 类内方差统计（用于观察类内紧致度）
        class_var_sum = [0.0, 0.0, 0.0, 0.0]
        class_var_count = [0, 0, 0, 0]

        # SupCon Stage1 统计
        epoch_supcon_loss = 0.0
        supcon_batches = 0
        loss_path_supcon_batches = 0
        loss_path_ce_batches = 0
        loss_path_align_batches = 0
        epoch_coral_loss = 0.0
        coral_batches = 0
        epoch_target_coral_loss = 0.0
        target_coral_batches = 0
        epoch_target_mmd_loss = 0.0
        target_mmd_batches = 0
        supcon_pos_pairs = 0
        supcon_valid_anchors = 0
        # Stage1 CE loss (Attempt 28 diagnostic)
        epoch_stage1_ce_loss = 0.0
        stage1_ce_batches = 0
        stage1_ce_missing_hp_batches = 0
        stage1_ce_missing_logits_batches = 0
        # Stage1 CE per-class diagnostics (Attempt 30+): track which classes are not learning
        stage1_ce_class_loss_sum = [0.0, 0.0, 0.0, 0.0]
        stage1_ce_class_count = [0, 0, 0, 0]
        # Attempt 64.1: semantic fault-vs-normal anchor diagnostics (must run on Stage1 path)
        epoch_sem_fvn_loss = 0.0
        sem_fvn_batches = 0
        
        # Att77: Batch Nuclear Norm diagnostics
        epoch_bnn_loss = 0.0
        bnn_batches = 0

        # ===== (Attempt 31) Targeted IO↔Rolling separation + embedding centroid diagnostics =====
        epoch_io_roll_sep_loss = 0.0
        io_roll_sep_batches = 0
        io_roll_sep_sim13_sum = 0.0
        io_roll_sep_sim23_sum = 0.0
        io_roll_sep_sim13_cnt = 0
        io_roll_sep_sim23_cnt = 0
        # ===== (Attempt 32) Inner↔Outer separation diagnostics =====
        epoch_io_sep_loss = 0.0
        io_sep_batches = 0
        io_sep_sim12_sum = 0.0
        io_sep_sim12_cnt = 0
        # ===== (Attempt 33) Normal↔Fault separation diagnostics =====
        epoch_nf_sep_loss = 0.0
        nf_sep_batches = 0
        nf_sep_sim01_sum = 0.0
        nf_sep_sim02_sum = 0.0
        nf_sep_sim03_sum = 0.0
        nf_sep_sim01_cnt = 0
        nf_sep_sim02_cnt = 0
        nf_sep_sim03_cnt = 0
        # ===== (Attempt 34) Inner↔Rolling separation diagnostics =====
        epoch_ir_sep_loss = 0.0
        ir_sep_batches = 0
        ir_sep_sim13_sum = 0.0  # sim(Inner, Rolling) = sim(class1, class3)
        ir_sep_sim13_cnt = 0
        embed_centroid_sum = None  # torch.Tensor [4,D]
        embed_centroid_cnt = [0, 0, 0, 0]

        # 如果使用贝叶斯原型，在每个epoch开始时重置embeddings收集器
        if bayesian_update_active:
            class_embeddings = {label: [] for label in initial_text_prototypes.keys()}
        
        # 初始化类别统计（用于自适应权重调整）
        class_stats = {}  # 总是初始化，即使不使用也会被重置
        
        # 初始化混淆统计（用于 Confusion-Aware 动态惩罚）
        # confusion_stats[true_class][pred_class] = count
        confusion_stats = {}  # {true_class: {pred_class: count, ...}, ...}
        
        # ===== 计算自适应 pos_weight (用于Binary Loss的正样本权重) =====
        # 将故障类的权重反馈给第一阶段的 Normal-vs-Fault 二分类器
        # 使用 max() 策略：确保最难识别的故障也被充分放行（更激进）
        # 如果故障类权重普遍很高，说明模型识别故障困难，应增大 binary_loss 的 pos_weight
        if enable_class_monitoring and class_weights is not None:
            # class_weights: [normal, fault1, fault2, fault3]
            fault_weights = class_weights[1:]  # [fault1, fault2, fault3]
            
            # 策略：使用最大值而非平均值（更激进，确保最难识别的故障也被放行）
            # 这能更好地处理"某个故障类特别难识别"的情况
            adaptive_pos_weight_global = fault_weights.mean().unsqueeze(0)  # scalar -> [1]
            adaptive_pos_weight_multi = fault_weights  # [3] 每个故障类的独立权重

            # Clamp per-type pos_weight to avoid one fault (often rolling) dominating nf_multi gradients.
            # This is a STRUCTURAL stabilization knob for the hybrid NF head.
            try:
                import config as cfg
                clamp_on = bool(getattr(cfg, "CLAMP_ADAPTIVE_POS_WEIGHT_MULTI", True))
                clamp_max = float(getattr(cfg, "ADAPTIVE_POS_WEIGHT_MULTI_MAX", 1.2))
            except Exception:
                clamp_on = True
                clamp_max = 1.2
            if clamp_on and adaptive_pos_weight_multi is not None:
                adaptive_pos_weight_multi = torch.clamp(adaptive_pos_weight_multi, max=float(clamp_max))
            
            if current_epoch <= 3 or current_epoch % 5 == 0:
                print(f"[Adaptive pos_weight] Epoch {current_epoch}: global={adaptive_pos_weight_global.item():.4f}, per-type={adaptive_pos_weight_multi.cpu().numpy()}")
        else:
            adaptive_pos_weight_global = None
            adaptive_pos_weight_multi = None
        
        # 一致性正则优化参数
        consistency_batch_interval = 1  # 每2个batch做一次一致性正则
        consistency_sample_ratio = 0.7  # 只增强50%的样本

        # ===== stability: skip pathological batches (prevents Fusion Cls explosion → collapse) =====
        warned_bad_batch = False
        skipped_bad_batches = 0
        # ===== rolling head diagnostics (epoch-level) =====
        roll_pos_count = 0
        roll_neg_count = 0
        roll_pos_prob_sum = 0.0
        roll_neg_prob_sum = 0.0
        roll_loss_sum = 0.0
        roll_sample_count = 0
        roll_pos_weight_sum = 0.0
        roll_pos_weight_count = 0
        try:
            roll_loss_mode = "focal" if bool(getattr(cfg, "USE_ROLLING_FOCAL_LOSS", False)) else "bce"
        except Exception:
            roll_loss_mode = "bce"
        # --- Att82 fix: split into TWO independent try/except blocks ---
        # Rolling decorr config (independent failure domain)
        try:
            use_roll_decorr = bool(getattr(cfg, "USE_ROLLING_DECORR_LOSS", False))
            roll_decorr_weight = float(getattr(cfg, "ROLLING_DECORR_WEIGHT", 0.05))
            roll_decorr_start = int(getattr(cfg, "ROLLING_DECORR_START_EPOCH", 1))
            roll_decorr_var_target = float(getattr(cfg, "ROLLING_DECORR_VAR_TARGET", 1.0))
            roll_decorr_cov_weight = float(getattr(cfg, "ROLLING_DECORR_COV_WEIGHT", 1.0))
            use_roll_decorr_input = bool(getattr(cfg, "USE_ROLLING_DECORR_INPUT", False))
            roll_decorr_input_weight = float(getattr(cfg, "ROLLING_DECORR_INPUT_WEIGHT", roll_decorr_weight))
            roll_decorr_reset_on_zero = bool(getattr(cfg, "ROLLING_DECORR_RESET_ON_ZERO", False))
            roll_decorr_reset_eps = float(getattr(cfg, "ROLLING_DECORR_RESET_EPS", 1e-4))
            roll_decorr_reset_cooldown = int(getattr(cfg, "ROLLING_DECORR_RESET_COOLDOWN", 3))
        except Exception as e:
            print(f"[WARNING] Rolling decorr config failed: {e}")
            use_roll_decorr = False
            roll_decorr_weight = 0.0
            roll_decorr_start = 1
            roll_decorr_var_target = 1.0
            roll_decorr_cov_weight = 1.0
            use_roll_decorr_input = False
            roll_decorr_input_weight = 0.0
            roll_decorr_reset_on_zero = False
            roll_decorr_reset_eps = 1e-4
            roll_decorr_reset_cooldown = 3
        # Global decorr / VICReg config (independent failure domain — Att82 fix)
        try:
            use_global_decorr = bool(getattr(cfg, "USE_GLOBAL_DECORR_LOSS", False))
            global_decorr_weight = float(getattr(cfg, "GLOBAL_DECORR_WEIGHT", 0.05))
            global_decorr_weight_task2 = float(getattr(cfg, "GLOBAL_DECORR_WEIGHT_TASK2", global_decorr_weight))
            global_decorr_start = int(getattr(cfg, "GLOBAL_DECORR_START_EPOCH", 1))
            global_decorr_var_target = float(getattr(cfg, "GLOBAL_DECORR_VAR_TARGET", 1.0))
            global_decorr_cov_weight = float(getattr(cfg, "GLOBAL_DECORR_COV_WEIGHT", 1.0))
            use_global_vicreg = bool(getattr(cfg, "USE_GLOBAL_VICREG_LOSS", False))
            global_vicreg_var_weight = float(getattr(cfg, "GLOBAL_VICREG_VAR_WEIGHT", 1.0))
            global_vicreg_cov_weight = float(getattr(cfg, "GLOBAL_VICREG_COV_WEIGHT", 1.0))
            global_vicreg_std_target = float(getattr(cfg, "GLOBAL_VICREG_STD_TARGET", 1.0))
            use_global_uniformity = bool(getattr(cfg, "USE_GLOBAL_UNIFORMITY_LOSS", True))
            global_uniformity_weight = float(getattr(cfg, "GLOBAL_UNIFORMITY_WEIGHT", 1.0))
            global_uniformity_t = float(getattr(cfg, "GLOBAL_UNIFORMITY_T", 2.0))
            use_fusion_var_floor = bool(getattr(cfg, "USE_FUSION_VAR_FLOOR_LOSS", True))
            fusion_var_floor_min = float(getattr(cfg, "FUSION_VAR_FLOOR_MIN", 0.001))
            fusion_var_floor_weight = float(getattr(cfg, "FUSION_VAR_FLOOR_WEIGHT", 0.05))
            use_fusion_vicreg = bool(getattr(cfg, "USE_FUSION_VICREG_LOSS", True))
            fusion_vicreg_var_weight = float(getattr(cfg, "FUSION_VICREG_VAR_WEIGHT", 0.5))
            fusion_vicreg_cov_weight = float(getattr(cfg, "FUSION_VICREG_COV_WEIGHT", 0.05))
            fusion_vicreg_std_target = float(getattr(cfg, "FUSION_VICREG_STD_TARGET", 0.04))
            use_fusion_inter_cos = bool(getattr(cfg, "USE_FUSION_INTER_COS_MARGIN_LOSS", True))
            fusion_inter_cos_weight = float(getattr(cfg, "FUSION_INTER_COS_MARGIN_WEIGHT", 0.08))
            fusion_inter_cos_target = float(getattr(cfg, "FUSION_INTER_COS_MARGIN_TARGET_MAX", 0.15))
            fusion_inter_cos_start = int(getattr(cfg, "FUSION_INTER_COS_MARGIN_START_EPOCH", 1))
            fusion_inter_cos_min_samples = int(getattr(cfg, "FUSION_INTER_COS_MIN_SAMPLES_PER_CLASS", 2))
            use_fishr_loss = bool(getattr(cfg, "USE_FISHR_LOSS", False))
            fishr_weight = float(getattr(cfg, "FISHR_WEIGHT", 0.0))
            fishr_start_epoch = int(getattr(cfg, "FISHR_START_EPOCH", 1))
            fishr_min_samples_per_domain = int(getattr(cfg, "FISHR_MIN_SAMPLES_PER_DOMAIN", 8))
            fishr_source_domain_ids = getattr(cfg, "FISHR_SOURCE_DOMAIN_IDS", (0, 2))
            if isinstance(fishr_source_domain_ids, (list, tuple)):
                fishr_source_domain_ids = tuple(int(x) for x in fishr_source_domain_ids)
            else:
                fishr_source_domain_ids = (0, 2)
            use_ccpa_loss = bool(getattr(cfg, "USE_CCPA_LOSS", False))
            ccpa_weight = float(getattr(cfg, "CCPA_WEIGHT", 0.0))
            ccpa_start_epoch = int(getattr(cfg, "CCPA_START_EPOCH", 1))
            ccpa_min_samples_per_domain_class = int(getattr(cfg, "CCPA_MIN_SAMPLES_PER_DOMAIN_CLASS", 2))
            ccpa_source_domain_ids = getattr(cfg, "CCPA_SOURCE_DOMAIN_IDS", fishr_source_domain_ids)
            if isinstance(ccpa_source_domain_ids, (list, tuple)):
                ccpa_source_domain_ids = tuple(int(x) for x in ccpa_source_domain_ids)
            else:
                ccpa_source_domain_ids = tuple(fishr_source_domain_ids)
            # ATT376: phase-aware source-domain routing for Task1-CWRU invariance.
            if is_task1_cwru_phase:
                try:
                    fishr_source_domain_ids = tuple(int(x) for x in _task1_cwru_fishr_ids)
                except Exception:
                    fishr_source_domain_ids = (0, 1)
                try:
                    ccpa_source_domain_ids = tuple(int(x) for x in _task1_cwru_ccpa_ids)
                except Exception:
                    ccpa_source_domain_ids = (0, 1)
            if core_loss_only_mode:
                _allowed = set(getattr(cfg, "CORE_LOSS_ALLOWED_TERMS", ()))
                if not (is_task1_cwru_phase and core_keep_task1_cwru_invariance):
                    if "fishr" not in _allowed:
                        use_fishr_loss = False
                    use_ccpa_loss = False
                # ATT379(S5): Fishr stays enabled when "fishr" is in _allowed
                if "fishr" in _allowed and use_fishr_loss:
                    pass  # keep use_fishr_loss=True
                use_roll_decorr = False
                use_roll_decorr_input = False
                use_global_uniformity = False
                # ATT378-FIX: enforce CORE_LOSS_ALLOWED_TERMS for global_decorr
                if "global_decorr" not in _allowed:
                    use_global_decorr = False
                    use_global_vicreg = False
                    global_decorr_weight = 0.0
                    global_decorr_weight_task2 = 0.0
                if current_epoch == 1:
                    print(f"[ATT378-FIX] core_loss enforced: "
                          f"global_decorr={'global_decorr' in _allowed}, "
                          f"aux_spec={'aux_spec' in _allowed}, "
                          f"allowed={_allowed}")
            log_global_variance = bool(getattr(cfg, "LOG_GLOBAL_VARIANCE", False))
            task2_detach_fusion_for_decoder = bool(getattr(cfg, "TARGET_TASK2_DETACH_FUSION_FOR_DECODER", False))
        except Exception as e:
            print(f"[WARNING] Global decorr config failed: {e}")
            use_global_decorr = False
            global_decorr_weight = 0.0
            global_decorr_weight_task2 = 0.0
            global_decorr_start = 1
            global_decorr_var_target = 1.0
            global_decorr_cov_weight = 1.0
            use_global_vicreg = False
            global_vicreg_var_weight = 1.0
            global_vicreg_cov_weight = 1.0
            global_vicreg_std_target = 1.0
            use_global_uniformity = True
            global_uniformity_weight = 1.0
            global_uniformity_t = 2.0
            use_fusion_var_floor = True
            fusion_var_floor_min = 0.001
            fusion_var_floor_weight = 0.05
            use_fusion_vicreg = True
            fusion_vicreg_var_weight = 0.5
            fusion_vicreg_cov_weight = 0.05
            fusion_vicreg_std_target = 0.04
            use_fusion_inter_cos = True
            fusion_inter_cos_weight = 0.08
            fusion_inter_cos_target = 0.15
            fusion_inter_cos_start = 1
            fusion_inter_cos_min_samples = 2
            use_fishr_loss = False
            fishr_weight = 0.0
            fishr_start_epoch = 1
            fishr_min_samples_per_domain = 8
            fishr_source_domain_ids = (0, 2)
            use_ccpa_loss = False
            ccpa_weight = 0.0
            ccpa_start_epoch = 1
            ccpa_min_samples_per_domain_class = 2
            ccpa_source_domain_ids = (0, 2)
            log_global_variance = False
            task2_detach_fusion_for_decoder = False
        try:
            _is_task2_phase = "task2" in str(checkpoint_tag).lower()
        except Exception:
            _is_task2_phase = False
        effective_global_decorr_weight = (
            float(global_decorr_weight_task2) if _is_task2_phase else float(global_decorr_weight)
        )
        # ATT121: 仅 epoch 1 打印（之前每轮都打印，严重冗余）
        if current_epoch == 1:
            print(
                f"[Config] Global Decorr: enabled={use_global_decorr}, "
                f"weight={effective_global_decorr_weight}, base_weight={global_decorr_weight}, "
                f"task2_weight={global_decorr_weight_task2}, "
                f"phase={'Task2' if _is_task2_phase else 'Task1'}, "
                  f"start_epoch={global_decorr_start}, log_variance={log_global_variance}")
            print(
                f"[Config] Global VICReg: enabled={use_global_vicreg}, "
                f"var_w={global_vicreg_var_weight}, cov_w={global_vicreg_cov_weight}, "
                f"std_target={global_vicreg_std_target}"
            )
            print(
                f"[Config] Fishr: enabled={use_fishr_loss}, weight={fishr_weight}, "
                f"start_epoch={fishr_start_epoch}, min_samples={fishr_min_samples_per_domain}, "
                f"source_domains={list(fishr_source_domain_ids)}"
            )
            print(
                f"[Config] CCPA: enabled={use_ccpa_loss}, weight={ccpa_weight}, "
                f"start_epoch={ccpa_start_epoch}, min_samples_per_domain_class={ccpa_min_samples_per_domain_class}, "
                f"source_domains={list(ccpa_source_domain_ids)}"
            )
            print(f"[Config] Rolling Decorr: enabled={use_roll_decorr}, weight={roll_decorr_weight}")
            print(f"[Config] Task2 fusion detach: enabled={int(task2_detach_fusion_for_decoder and _is_task2_phase)}")
            try:
                import config as _cfg_log
                _sig_in = bool(getattr(_cfg_log, "FUSION_SIGNAL_INSTANCE_NORM", False))
                _spec_res = bool(getattr(_cfg_log, "FUSION_SPEC_RESIDUAL_ENABLE", False))
                print(f"[Config] ATT421: FUSION_SIGNAL_INSTANCE_NORM={int(_sig_in)}, FUSION_SPEC_RESIDUAL_ENABLE={int(_spec_res)}")
            except Exception:
                pass

        # Build per-epoch unlabeled target references for target-CORAL.
        target_coral_refs = []
        if (
            task2_target_coral_enable
            and float(task2_target_coral_weight) > 0.0
            and int(current_epoch) >= int(task2_coral_start_epoch)
        ):
            _enc_train_mode = bool(spec_encoder.training)
            _txt_train_mode = bool(text_encoder.training)
            try:
                spec_encoder.eval()
                text_encoder.eval()
                with torch.no_grad():
                    for _name, _loader in (("HUST", hust_loader), ("HIT", hit_loader)):
                        if _loader is None:
                            continue
                        _ref_chunks = []
                        try:
                            for _ref_batch in _loader:
                                _z_ref_chunk = _extract_spec_feat_from_batch(_ref_batch)
                                if torch.is_tensor(_z_ref_chunk) and _z_ref_chunk.dim() == 2:
                                    _ref_chunks.append(_z_ref_chunk.detach())
                        except Exception:
                            _ref_chunks = []
                        if len(_ref_chunks) > 0:
                            _z_ref = torch.cat(_ref_chunks, dim=0)
                            if int(_z_ref.size(0)) >= int(task2_target_coral_min_samples):
                                target_coral_refs.append((_name, _z_ref))
            finally:
                if _enc_train_mode:
                    spec_encoder.train()
                if _txt_train_mode:
                    text_encoder.train()
            if int(current_epoch) == int(task2_coral_start_epoch):
                _ref_sizes = ", ".join([f"{_n}:{int(_z.size(0))}" for _n, _z in target_coral_refs]) if len(target_coral_refs) > 0 else "none"
                print(f"[Task2 Target-CORAL] refs ready: {len(target_coral_refs)} domain(s) [{_ref_sizes}]")

        target_mmd_loaders = []
        if task2_target_mmd_enable and _is_task2_phase and float(task2_target_mmd_weight) > 0.0:
            if hust_loader is not None:
                target_mmd_loaders.append(("HUST", hust_loader))
            if hit_loader is not None:
                target_mmd_loaders.append(("HIT", hit_loader))
        target_mmd_iters = {name: iter(loader) for name, loader in target_mmd_loaders}

        for batch_idx, batch in enumerate(train_loader):
            # 计算 Confusion-Aware 惩罚权重（基于当前累积的混淆统计，动态更新）
            confusion_penalty_weights = None
            if len(confusion_stats) > 0:
                penalty_weights = compute_confusion_penalty_weights(confusion_stats, threshold=0.3)
                confusion_penalty_weights = penalty_weights.to(device)
                if batch_idx == 0 and current_epoch <= 3:
                    print(f"[Confusion-Aware] Epoch {current_epoch}, Batch {batch_idx}: penalty_weights={confusion_penalty_weights.cpu().numpy()}")
            signal = batch["signal"].to(device)
            spec = prepare_batch_spec(batch, device, spec_encoder, cfg)
            signal_mask = batch["signal_mask"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            texts = batch["text"]
            labels = batch["label"].to(device)
            domains = batch.get("domain", None)  # List[str] or None
            conditions = batch.get("condition", None)
            stat_vec = batch.get("stat_vec", None)
            if stat_vec is not None:
                stat_vec = stat_vec.to(device)
            phys_features = batch.get("phys_features", None)
            if phys_features is not None:
                phys_features = phys_features.to(device)
            # ATT446: WST bypass — encode WST features separately
            z_wst = None
            if _wst_bypass_active:
                _wst_raw = batch.get("wst_spec", None)
                if _wst_raw is not None:
                    z_wst = wst_encoder(_wst_raw.to(device))  # [B, wst_dim]
                    if batch_idx == 0 and current_epoch <= 3:
                        _zw = z_wst.detach()
                        print(f"[ATT446-WSTBypass] Epoch {current_epoch} "
                              f"z_wst: shape={tuple(_zw.shape)}, "
                              f"norm={float(_zw.norm(dim=-1).mean()):.4f}, "
                              f"std={float(_zw.std()):.4f}")
            # ATT449-S3: Envelope bypass — encode env features separately
            z_env = None
            ratio_aux = None
            if _env_bypass_active:
                _env_raw = batch.get("env_spec", None)
                if _env_raw is not None:
                    z_env = env_encoder(_env_raw.to(device))
                    if batch_idx == 0 and current_epoch <= 3:
                        _ze = z_env.detach()
                        print(f"[ATT449-EnvBypass] Epoch {current_epoch} "
                              f"z_env: shape={tuple(_ze.shape)}, "
                              f"norm={float(_ze.norm(dim=-1).mean()):.4f}, "
                              f"std={float(_ze.std()):.4f}")
                _ratio_raw = batch.get("bpfi_bpfo_ratio", None)
                if _ratio_raw is not None:
                    ratio_aux = _ratio_raw.to(device)
                    if batch_idx == 0 and current_epoch <= 3:
                        print(f"[ATT449-Ratio] Epoch {current_epoch} "
                              f"bpfi_bpfo_ratio: mean={float(ratio_aux.mean()):.4f}, "
                              f"std={float(ratio_aux.std()):.4f}")
            # ATT464b: SCD bypass — encode CMS features separately
            z_scd = None
            if _scd_bypass_active:
                _scd_raw = batch.get("scd_spec", None)
                if _scd_raw is not None:
                    z_scd = scd_encoder(_scd_raw.to(device))
                    if batch_idx == 0 and current_epoch <= 3:
                        _zs = z_scd.detach()
                        print(f"[ATT464b-SCDBypass] Epoch {current_epoch} "
                              f"z_scd: shape={tuple(_zs.shape)}, "
                              f"norm={float(_zs.norm(dim=-1).mean()):.4f}, "
                              f"std={float(_zs.std()):.4f}")
            # ATT450-S1: Log bypass scale parameters every epoch (batch_idx==0)
            if batch_idx == 0:
                _s_wst = getattr(spec_encoder, '_wst_scale', None)
                _s_env = getattr(spec_encoder, '_env_scale', None)
                _s_rat = getattr(spec_encoder, '_ratio_scale', None)
                if _s_wst is not None or _s_env is not None or _s_rat is not None:
                    _p = [f"[ATT450-Scale] E{current_epoch}"]
                    if _s_wst is not None:
                        _p.append(f"wst_scale={float(_s_wst.detach()):.4f}")
                    if _s_env is not None:
                        _p.append(f"env_scale={float(_s_env.detach()):.4f}")
                    if _s_rat is not None:
                        _p.append(f"ratio_scale={float(_s_rat.detach()):.4f}")
                    print(", ".join(_p))
            if batch_idx == 0 and current_epoch == 1:
                _pf_status = "PRESENT" if phys_features is not None else "MISSING"
                _pf_shape = tuple(phys_features.shape) if phys_features is not None else "N/A"
                print(f"[ATT437-PhysPipeline] phys_features={_pf_status}, shape={_pf_shape}")
            # Hard guard: training/eval heads assume 4-class labels {0,1,2,3}.
            if torch.any((labels < 0) | (labels > 3)):
                _bad = sorted(set([int(x) for x in labels.detach().cpu().tolist()]))
                raise RuntimeError(
                    f"[DataflowAudit] Invalid label ids detected: {_bad}. "
                    "Expected labels in {0,1,2,3}; please check dataset label mapping."
                )

            # Stage1/Stage2 flags (flat head needs explicit stage1 flag)
            stage1_mode_now = bool(stage1_mode)
            # ATT377: allow SupCon through core-only mode when "supcon" is in CORE_LOSS_ALLOWED_TERMS
            _supcon_core_pass = (not core_loss_only_mode) or ("supcon" in getattr(cfg, "CORE_LOSS_ALLOWED_TERMS", ()))
            supcon_stage1_now = bool(_supcon_core_pass and use_supcon_stage1 and stage1_mode_now and force_ce_path_now)
            if batch_idx == 0:
                print(
                    f"[ATT377] SupCon gate: core_only={int(core_loss_only_mode)}, "
                    f"core_pass={int(_supcon_core_pass)}, stg1_cfg={int(use_supcon_stage1)}, "
                    f"stg1_mode={int(stage1_mode_now)}, force_ce={int(force_ce_path_now)}, "
                    f"=> supcon_stage1_now={int(supcon_stage1_now)}, "
                    f"align_supcon={int(_supcon_core_pass and getattr(cfg, 'USE_SUPCON_LOSS', False))}"
                )
            grad_probe_terms = {}

            # 编码domain_ids（用于域残差）
            domain_ids = None
            if domains is not None:
                domain_ids = encode_domain_ids(domains, conditions).to(device)
                if torch.any((domain_ids < 0) | (domain_ids > 4)):
                    _bad_dom = sorted(set([int(x) for x in domain_ids.detach().cpu().tolist()]))
                    raise RuntimeError(
                        f"[DataflowAudit] Invalid domain ids detected: {_bad_dom}. "
                        "Expected domain ids in {0,1,2,3,4}; please check encode_domain_ids/config DOMAIN_ID_MAP."
                    )
            
            # 编码signal（支持域残差和分类头）
            use_signal_classification_head = getattr(signal_encoder, 'use_classification_head', False)
            if hasattr(signal_encoder, 'use_domain_residual') and signal_encoder.use_domain_residual:
                if use_signal_classification_head:
                    z_signal, signal_logits = signal_encoder(signal, signal_mask, domain_ids=domain_ids, return_logits=True)
                else:
                    z_signal = signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                    signal_logits = None
            else:
                if use_signal_classification_head:
                    z_signal, signal_logits = signal_encoder(signal, signal_mask, return_logits=True)
                else:
                    z_signal = signal_encoder(signal, signal_mask)
                    signal_logits = None
            
            z_text = text_encoder(texts)
            
            # 处理spec编码（支持MoE和域残差）
            # 检查是否是MoE版本（通过检查是否有num_bands属性）
            is_moe = use_moe and hasattr(spec_encoder, 'num_bands')
            
            # 检查是否启用分类头
            use_spec_classification_head = getattr(spec_encoder, 'use_classification_head', False)
            
            # ATT172 Proto Isolation: z_spec_clean 供 alignment loss 使用（无 MixStyle 干净通道）
            # 默认等于 z_spec，仅在 proto isolation 激活时覆盖。
            z_spec_clean = None

            if is_moe:
                # MoE版本（支持可选 return_tokens 用于多token fusion）
                # ATT449-BUGFIX: MoE路径必须传递bypass特征(wst_aux/env_aux/ratio_aux)
                _moe_bypass_kw = {}
                if z_wst is not None:
                    _moe_bypass_kw["wst_aux"] = z_wst
                if z_env is not None:
                    _moe_bypass_kw["env_aux"] = z_env
                if ratio_aux is not None:
                    _moe_bypass_kw["ratio_aux"] = ratio_aux
                if z_scd is not None:
                    _moe_bypass_kw["scd_aux"] = z_scd
                want_spec_tokens = bool(getattr(fusion_module, "use_multi_token", False))
                if want_spec_tokens:
                    if use_spec_classification_head:
                        z_spec, z_spec_tokens, spec_mask_tokens, moe_stats, spec_logits = spec_encoder(
                            spec, spec_mask,
                            text_emb=z_text,
                            domain_ids=domain_ids,
                            return_moe_stats=True,
                            return_tokens=True,
                            return_logits=True,
                            stat_vec=stat_vec,
                            **_moe_bypass_kw,
                        )
                    else:
                        z_spec, z_spec_tokens, spec_mask_tokens, moe_stats = spec_encoder(
                            spec, spec_mask,
                            text_emb=z_text,
                            domain_ids=domain_ids,
                            return_moe_stats=True,
                            return_tokens=True,
                            stat_vec=stat_vec,
                            **_moe_bypass_kw,
                        )
                        spec_logits = None
                else:
                    if use_spec_classification_head:
                        z_spec, moe_stats, spec_logits = spec_encoder(
                            spec, spec_mask,
                            text_emb=z_text,
                            domain_ids=domain_ids,
                            return_moe_stats=True,
                            return_logits=True,
                            stat_vec=stat_vec,
                            **_moe_bypass_kw,
                        )
                    else:
                        z_spec, moe_stats = spec_encoder(
                            spec, spec_mask,
                            text_emb=z_text,
                            domain_ids=domain_ids,
                            return_moe_stats=True,
                            stat_vec=stat_vec,
                            **_moe_bypass_kw,
                        )
                        spec_logits = None
                    z_spec_tokens, spec_mask_tokens = None, None
            else:
                spec_kwargs = {"spec": spec, "spec_mask": spec_mask, "domain_ids": domain_ids, "stat_vec": stat_vec}
                if z_wst is not None:
                    spec_kwargs["wst_aux"] = z_wst
                if z_env is not None:
                    spec_kwargs["env_aux"] = z_env
                if ratio_aux is not None:
                    spec_kwargs["ratio_aux"] = ratio_aux
                if z_scd is not None:
                    spec_kwargs["scd_aux"] = z_scd
                if use_spec_classification_head:
                    z_spec, spec_logits = spec_encoder(**spec_kwargs, return_logits=True)
                else:
                    z_spec = spec_encoder(**spec_kwargs)
                    spec_logits = None
                moe_stats = None
                z_spec_tokens, spec_mask_tokens = None, None

                # ATT172 Proto Isolation：在 alignment loss 中用无 MixStyle 的干净 z_spec_clean 替代
                _proto_iso = (
                    current_w_spec_align > 0.0
                    and getattr(cfg, "MIXSTYLE_PROTO_ISOLATION", False)
                    and hasattr(spec_encoder, "mixstyle")
                    and spec_encoder.mixstyle is not None
                )
                if _proto_iso:
                    _ms_saved = spec_encoder.mixstyle
                    spec_encoder.mixstyle = None
                    if use_spec_classification_head:
                        z_spec_clean, _ = spec_encoder(**spec_kwargs, return_logits=True)
                    else:
                        z_spec_clean = spec_encoder(**spec_kwargs)
                    spec_encoder.mixstyle = _ms_saved
                    if current_epoch == 1 and batch_idx == 0:
                        print("[ProtoIsolation] ATT172: clean z_spec_clean branch active for alignment loss.")

            # fallback: 若 proto isolation 未激活（MoE 路径或 isolation 关闭），用 z_spec 本身
            if z_spec_clean is None:
                z_spec_clean = z_spec

            # ===== ATT396: Feature Decoupling =====
            _z_spec_fault, _z_spec_cond = None, None
            _z_signal_fault, _z_signal_cond = None, None
            _dc_active = (
                _use_decouple
                and decouple_spec is not None
                and decouple_signal is not None
                and current_epoch >= int(getattr(cfg, "DECOUPLE_START_EPOCH", 1))
            )
            if _dc_active:
                _z_spec_fault, _z_spec_cond = decouple_spec(z_spec_clean)
                _z_signal_fault, _z_signal_cond = decouple_signal(z_signal)
                z_spec_clean = _z_spec_fault
                z_signal = _z_signal_fault
                if batch_idx == 0 and current_epoch <= 3:
                    _gate_spec = float(torch.sigmoid(decouple_spec.fault_gate).item())
                    _gate_sig = float(torch.sigmoid(decouple_signal.fault_gate).item())
                    print(
                        f"[ATT396][Decouple] E{current_epoch} gate_spec={_gate_spec:.4f}, "
                        f"gate_signal={_gate_sig:.4f}, "
                        f"z_spec_fault_norm={float(_z_spec_fault.detach().norm(dim=-1).mean()):.4f}, "
                        f"z_spec_cond_norm={float(_z_spec_cond.detach().norm(dim=-1).mean()):.4f}"
                    )

            z_spec_for_fusion = z_spec_clean if _dc_active else z_spec
            try:
                _use_spec_pre_l2_for_fusion = bool(getattr(cfg, "FUSION_USE_SPEC_PRE_L2_INPUT", True))
            except Exception:
                _use_spec_pre_l2_for_fusion = True
            if _use_spec_pre_l2_for_fusion and not _dc_active:
                _z_spec_pre_l2 = extract_pre_l2_feature(z_spec)
                if torch.is_tensor(_z_spec_pre_l2) and _z_spec_pre_l2.shape == z_spec.shape:
                    z_spec_for_fusion = _z_spec_pre_l2

            # ATT377: Fusion input scale normalization — prevent modality scale mismatch
            _fusion_input_norm = bool(getattr(cfg, "FUSION_INPUT_LAYERNORM", True))
            if _fusion_input_norm:
                z_signal_for_fusion = F.layer_norm(z_signal, [z_signal.size(-1)])
                z_spec_for_fusion = F.layer_norm(z_spec_for_fusion, [z_spec_for_fusion.size(-1)])
            else:
                z_signal_for_fusion = z_signal
            if batch_idx == 0 and current_epoch <= 3:
                _sig_norm = float(z_signal.detach().norm(dim=-1).mean().cpu().item())
                _spec_norm = float(z_spec_for_fusion.detach().norm(dim=-1).mean().cpu().item())
                _sig_fn = float((z_signal_for_fusion if _fusion_input_norm else z_signal).detach().norm(dim=-1).mean().cpu().item())
                _spec_fn = float(z_spec_for_fusion.detach().norm(dim=-1).mean().cpu().item())
                print(
                    f"[ATT377][FusionInput] raw_sig_norm={_sig_norm:.4f}, raw_spec_norm={_spec_norm:.4f}, "
                    f"post_ln_sig={_sig_fn:.4f}, post_ln_spec={_spec_fn:.4f}, "
                    f"layernorm={int(_fusion_input_norm)}"
                )

            # 融合模块（支持多token和推理一致的类感知）
            # 检查fusion_module是否支持这些参数
            if hasattr(fusion_module, 'domain_embedding'):
                # DomainAwareCrossAttentionFusion支持这些参数
                
                # 渐进式训练逻辑（逐步启用高级特性）
                from config import FUSION_PROGRESSIVE_TRAINING, FUSION_CONFIDENCE_START_EPOCH, FUSION_CLASS_AWARE_START_EPOCH
                
                if FUSION_PROGRESSIVE_TRAINING:
                    # 动态控制置信度特征的使用
                    use_confidence_now = (current_epoch >= FUSION_CONFIDENCE_START_EPOCH)
                    if hasattr(fusion_module, 'use_confidence'):
                        fusion_module.use_confidence = use_confidence_now
                    
                    # ===== HARD DISABLE class-aware during training =====
                    # Class-aware causes distribution shift that destabilizes gate training.
                    # It will be available only for inference/fine-tuning in later stages.
                    use_class_aware_now = False  # Hard disable during training
                else:
                    # 非渐进式训练：直接使用配置
                    use_confidence_now = fusion_module.use_confidence if hasattr(fusion_module, 'use_confidence') else False
                    # ===== HARD DISABLE class-aware during training =====
                    use_class_aware_now = False  # Hard disable during training
                
                # 检查是否使用多token模式（MoE 现已支持 return_tokens）
                use_multi_token = bool(getattr(fusion_module, 'use_multi_token', False))
                
                if use_multi_token:
                    # 多token模式：从encoder获取token序列
                    if hasattr(signal_encoder, 'use_domain_residual') and signal_encoder.use_domain_residual:
                        z_signal_tokens, signal_mask_tokens = signal_encoder(signal, signal_mask, domain_ids=domain_ids, return_tokens=True)
                    else:
                        z_signal_tokens, signal_mask_tokens = signal_encoder(signal, signal_mask, return_tokens=True)
                    
                    # Spec tokens:
                    # - MoE 版本：上面已在 spec_encoder 前向里按需计算过 tokens
                    # - 普通版本：此处再取 tokens
                    if is_moe:
                        if z_spec_tokens is None or spec_mask_tokens is None:
                            # 理论上不应发生：want_spec_tokens=True 时已计算
                            z_spec_tokens = z_spec.unsqueeze(1)
                            spec_mask_tokens = torch.ones(z_spec_tokens.size(0), z_spec_tokens.size(1), device=z_spec_tokens.device)
                    else:
                        z_spec_tokens, spec_mask_tokens = spec_encoder(
                            spec,
                            spec_mask,
                            domain_ids=domain_ids,
                            return_tokens=True,
                            stat_vec=stat_vec,
                        )
                else:
                    # 单token模式：使用pooled embedding
                    # ATT377: use scale-normalized versions for fusion input
                    z_signal_tokens = z_signal_for_fusion if _fusion_input_norm else z_signal
                    z_spec_tokens = z_spec_for_fusion
                    signal_mask_tokens, spec_mask_tokens = None, None
                
                # 两步融合：如果使用类感知，先做一次普通融合得到预测
                if use_class_aware_now and fusion_module.use_class_aware:
                    # 第一步：普通融合（不使用类感知）得到预测
                    z_fusion_step1 = fusion_module(
                        z_signal_tokens,
                        z_spec_tokens,
                        domain_ids=domain_ids,
                        text_prototypes=text_prototypes,
                        class_ids=None,  # 不使用类感知
                        class_probs=None,
                        signal_token_mask=signal_mask_tokens if use_multi_token else None,
                        spec_token_mask=spec_mask_tokens if use_multi_token else None,
                        signal_logits=signal_logits,
                        spec_logits=spec_logits
                    )
                    
                    # 获取预测的类别概率分布
                    if hasattr(fusion_decoder, 'hierarchical') and fusion_decoder.hierarchical:
                        logits_step1, _ = fusion_decoder(z_fusion_step1, return_parts=True)
                    else:
                        logits_step1 = fusion_decoder(z_fusion_step1)
                    
                    # 计算类别概率分布（用于推理一致的类感知）
                    class_probs = F.softmax(logits_step1, dim=1)  # [B, C]
                    
                    # 第二步：使用预测分布做类感知融合
                    _dgl_detach_ca = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
                    z_fusion = fusion_module(
                        z_signal_tokens,
                        z_spec_tokens,
                        domain_ids=domain_ids,
                        text_prototypes=text_prototypes,
                        class_ids=None,  # 推理一致：不使用GT标签
                        class_probs=class_probs,  # 使用预测分布
                        signal_token_mask=signal_mask_tokens if use_multi_token else None,
                        spec_token_mask=spec_mask_tokens if use_multi_token else None,
                        signal_logits=signal_logits,
                        spec_logits=spec_logits,
                        dgl_detach_inputs=_dgl_detach_ca,
                    )
                else:
                    # 不使用类感知：直接融合
                    _dgl_detach_now = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
                    z_fusion = fusion_module(
                        z_signal_tokens,
                        z_spec_tokens,
                        domain_ids=domain_ids,
                        text_prototypes=text_prototypes,
                        class_ids=None,
                        class_probs=None,
                        signal_token_mask=signal_mask_tokens if use_multi_token else None,
                        spec_token_mask=spec_mask_tokens if use_multi_token else None,
                        signal_logits=signal_logits,
                        spec_logits=spec_logits,
                        dgl_detach_inputs=_dgl_detach_now,
                    )
            else:
                _fus_sig = z_signal_for_fusion if _fusion_input_norm else z_signal
                _fus_spe = z_spec_for_fusion
                _dgl_detach_now = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
                if hasattr(fusion_module, 'domain_embedding'):
                    z_fusion = fusion_module(_fus_sig, _fus_spe, domain_ids=domain_ids, dgl_detach_inputs=_dgl_detach_now)
                elif hasattr(fusion_module, 'use_class_aware') and fusion_module.use_class_aware:
                    _fs = _fus_sig.detach() if _dgl_detach_now else _fus_sig
                    _fe = _fus_spe.detach() if _dgl_detach_now else _fus_spe
                    z_fusion = fusion_module(_fs, _fe, class_ids=None)
                else:
                    _fs = _fus_sig.detach() if _dgl_detach_now else _fus_sig
                    _fe = _fus_spe.detach() if _dgl_detach_now else _fus_spe
                    z_fusion = fusion_module(_fs, _fe)

            # VGSP: collect source-only fused vibration centroids from the actual forward path.
            if (semantic_proto_source == "signal_centroid" and semantic_centroid_space == "fusion"
                    and torch.is_tensor(z_fusion) and labels is not None and labels.dim() == 1):
                _vgsp_z = F.normalize(z_fusion.detach().float(), dim=-1, eps=1e-6)
                if embed_centroid_sum is None:
                    embed_centroid_sum = torch.zeros((4, int(_vgsp_z.size(1))), device=_vgsp_z.device, dtype=torch.float32)
                for _c in range(4):
                    _m = labels == int(_c)
                    if bool(_m.any()):
                        embed_centroid_sum[_c] += _vgsp_z[_m].sum(dim=0)
                        embed_centroid_cnt[_c] += int(_m.sum().item())

            # Determine current prototypes for this batch.
            # ATT392-S1: Learnable protos override all other sources.
            if learnable_proto_module is not None:
                current_prototypes = learnable_proto_module.as_dict()
            else:
                current_prototypes = text_prototypes
                if use_bayesian_prototypes and text_prototypes is None:
                    current_prototypes = initial_text_prototypes
            current_prototypes = apply_prompt_tuned_prototypes(
                fusion_decoder=fusion_decoder,
                prototypes=current_prototypes,
                detach_output=False
            )

            # Get fusion logits (with hierarchical parts if available)
            # Always define `hierarchical_parts` to avoid NameError in later Stage1 diagnostics.
            hierarchical_parts = None
            # ATT210: decoder input detach policy should be defined for both hierarchical/flat decoders.
            try:
                _is_task2_phase = "task2" in str(checkpoint_tag).lower()
            except Exception:
                _is_task2_phase = False
            detach_decoder_input_now = bool(task2_detach_fusion_for_decoder and _is_task2_phase)
            _fus_dec_stop = bool(getattr(cfg, "FUSION_DECODER_INPUT_STOPGRAD_ENABLE", False))
            _fd_grad_ratio_cfg = parse_fusion_decoder_grad_live_ratio(
                getattr(cfg, "FUSION_DECODER_INPUT_GRAD_LIVE_RATIO", None)
            )
            _stopgrad_anneal_on = bool(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_ENABLE", False))
            _sg_start_ep = int(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_START_EPOCH", 1))
            _sg_end_ep = int(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_END_EPOCH", 15))
            _sg_live_start = float(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_START", 1.0))
            _sg_live_end = float(getattr(cfg, "FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_END", 0.0))

            def _clamp01(x):
                return max(0.0, min(1.0, float(x)))

            _sg_live_start = _clamp01(_sg_live_start)
            _sg_live_end = _clamp01(_sg_live_end)
            if _sg_end_ep < _sg_start_ep:
                _sg_end_ep = _sg_start_ep
            if current_epoch <= _sg_start_ep:
                _dec_live_ratio_now = _sg_live_start
            elif current_epoch >= _sg_end_ep:
                _dec_live_ratio_now = _sg_live_end
            else:
                _den = max(1, _sg_end_ep - _sg_start_ep)
                _p = float(current_epoch - _sg_start_ep) / float(_den)
                _dec_live_ratio_now = _sg_live_start + (_sg_live_end - _sg_live_start) * _p
            _dec_live_ratio_now = _clamp01(_dec_live_ratio_now)

            if detach_decoder_input_now:
                z_fusion_for_decoder = z_fusion.detach()
            elif _fd_grad_ratio_cfg is not None:
                # ATT483: fixed weak decoder->fusion backprop; forward equals z_fusion.
                _r_fix = _fd_grad_ratio_cfg
                z_fusion_for_decoder = _r_fix * z_fusion + (1.0 - _r_fix) * z_fusion.detach()
            elif _fus_dec_stop:
                # ATT482: keep forward unchanged, scale backward grad by live ratio.
                # value: z, grad: live_ratio * grad(z)
                if _stopgrad_anneal_on:
                    z_fusion_for_decoder = (
                        _dec_live_ratio_now * z_fusion + (1.0 - _dec_live_ratio_now) * z_fusion.detach()
                    )
                else:
                    z_fusion_for_decoder = z_fusion.detach()
            else:
                z_fusion_for_decoder = z_fusion
            z_fusion_for_reg = extract_pre_l2_feature(z_fusion)
            if hasattr(fusion_decoder, 'hierarchical') and fusion_decoder.hierarchical:
                # ===== (Stage) Decouple prototype diagnosis from trunk =====
                # We always use the supervised diagnosis head logits (diag_linear) for the main CE/Focal term.
                # Prototype/text is only passed when Stage2 semantic alignment is enabled, mainly for auxiliary losses.
                # FIX (Attempt 36): Match epoch-level stage1_mode logic
                try:
                    import config as cfg
                    use_two_stage = bool(getattr(cfg, "USE_TWO_STAGE_TRAINING", True))
                    stage1_epochs = int(getattr(cfg, "STAGE1_SUPERVISED_EPOCHS", 8))
                    # FIX: When two-stage disabled, always Stage1
                    if use_two_stage:
                        stage1_mode_now = bool(current_epoch <= int(stage1_epochs))
                    else:
                        stage1_mode_now = True
                except Exception:
                    use_two_stage = True
                    stage1_mode_now = True  # Default to Stage1

                # DANN alpha (GRL)
                try:
                    import config as cfg
                    use_dann = bool(getattr(cfg, "USE_DANN", False))
                    dann_lambda = float(getattr(cfg, "DANN_LAMBDA", 0.10))
                    dann_start = int(getattr(cfg, "DANN_START_EPOCH", 1))
                    dann_max_alpha = float(getattr(cfg, "DANN_MAX_ALPHA", 1.0))
                    dann_sched = str(getattr(cfg, "DANN_ALPHA_SCHEDULE", "linear"))
                    use_roll_dann = bool(getattr(cfg, "USE_ROLLING_DANN", False))
                    roll_dann_lambda = float(getattr(cfg, "ROLLING_DANN_LAMBDA", 0.0))
                    roll_dann_start = int(getattr(cfg, "ROLLING_DANN_START_EPOCH", 1))
                    roll_dann_max_alpha = float(getattr(cfg, "ROLLING_DANN_MAX_ALPHA", 1.0))
                    roll_dann_sched = str(getattr(cfg, "ROLLING_DANN_ALPHA_SCHEDULE", "linear"))
                    if core_loss_only_mode:
                        use_dann = False
                        dann_lambda = 0.0
                        use_roll_dann = False
                        roll_dann_lambda = 0.0
                except Exception:
                    use_dann, dann_lambda, dann_start, dann_max_alpha, dann_sched = False, 0.0, 1, 1.0, "linear"
                    use_roll_dann, roll_dann_lambda, roll_dann_start, roll_dann_max_alpha, roll_dann_sched = False, 0.0, 1, 1.0, "linear"
                dann_alpha_now = 0.0
                if use_dann and dann_lambda > 0.0 and current_epoch >= int(dann_start):
                    dann_alpha_now = compute_dann_alpha(
                        current_epoch=current_epoch,
                        num_epochs=num_epochs,
                        start_epoch=int(dann_start),
                        max_alpha=float(dann_max_alpha),
                        schedule=str(dann_sched)
                    )
                roll_dann_alpha_now = 0.0
                if use_roll_dann and roll_dann_lambda > 0.0 and current_epoch >= int(roll_dann_start):
                    roll_dann_alpha_now = compute_dann_alpha(
                        current_epoch=current_epoch,
                        num_epochs=num_epochs,
                        start_epoch=int(roll_dann_start),
                        max_alpha=float(roll_dann_max_alpha),
                        schedule=str(roll_dann_sched)
                    )

                # Forward fusion decoder (optionally with prototypes for auxiliary terms)
                if (not stage1_mode_now) and use_proto_train_epoch and (current_prototypes is not None) and (domain_ids is not None):
                    logits4_unused, hierarchical_parts = fusion_decoder(
                        z_fusion_for_decoder,
                        domain_ids=domain_ids,
                        text_prototypes=current_prototypes,
                        return_parts=True,
                        dann_alpha=(dann_alpha_now if dann_alpha_now > 0.0 else None),
                        rolling_dann_alpha=(roll_dann_alpha_now if roll_dann_alpha_now > 0.0 else None),
                        z_signal=z_signal if (str(getattr(cfg, 'DANN_FEATURE', 'z_gate')).lower() == 'z_signal') else None,
                    )
                else:
                    logits4_unused, hierarchical_parts = fusion_decoder(
                        z_fusion_for_decoder,
                        domain_ids=domain_ids,
                        return_parts=True,
                        dann_alpha=(dann_alpha_now if dann_alpha_now > 0.0 else None),
                        rolling_dann_alpha=(roll_dann_alpha_now if roll_dann_alpha_now > 0.0 else None),
                        z_signal=z_signal if (str(getattr(cfg, 'DANN_FEATURE', 'z_gate')).lower() == 'z_signal') else None,
                    )

                # Use supervised diagnosis logits as the "main" logits for CE/Focal.
                # IMPORTANT: in two-stage training, prototype logits should not dominate the supervised trunk.
                # Prefer `logits_diag_supervised` (pure learnable heads) when available.
                fusion_logits = None
                try:
                    if isinstance(hierarchical_parts, dict):
                        fusion_logits = hierarchical_parts.get("logits_diag_supervised", None)
                except Exception:
                    fusion_logits = None
                if fusion_logits is None:
                    try:
                        fusion_logits = hierarchical_parts.get("logits_diag_linear", None)
                    except Exception:
                        fusion_logits = None
                if fusion_logits is None:
                    fusion_logits = logits4_unused

                # Attach hierarchical parts to logits for auxiliary losses / diagnostics
                if isinstance(hierarchical_parts, dict):
                    hierarchical_parts["nf_pos_weight_global"] = nf_pos_weight_global
                    hierarchical_parts["roll_dann_lambda"] = float(roll_dann_lambda)
                    hierarchical_parts["roll_dann_alpha"] = float(roll_dann_alpha_now)
                    hierarchical_parts["nf_pos_weight_multi"] = nf_pos_weight_multi
                    if 'fault_type_weights' in locals() and fault_type_weights is not None:
                        hierarchical_parts["fault_type_weights"] = fault_type_weights
                    hierarchical_parts["domain_ids"] = domain_ids
                    hierarchical_parts["current_epoch"] = current_epoch
                    hierarchical_parts["margin_router_target"] = margin_router_target
                    # Key: tell loss_fusion_cls to use flat CE/Focal even if hierarchical logits exist
                    try:
                        use_flat_diag = bool(getattr(cfg, "USE_FLAT_DIAG_HEAD", False))
                    except Exception:
                        use_flat_diag = False
                    hierarchical_parts["force_flat_ce"] = bool(use_two_stage or use_flat_diag)
                    hierarchical_parts["dann_lambda"] = float(dann_lambda)
                    hierarchical_parts["dann_alpha"] = float(dann_alpha_now)
                    # A2: partial-label mask for diagnosis CE (PU/HIT have no class-3)
                    try:
                        hierarchical_parts["allowed_diag_mask"] = build_allowed_diag_mask(
                            domain_ids=domain_ids,
                            num_classes=int(fusion_logits.size(1)),
                            device=fusion_logits.device
                        )
                    except Exception:
                        hierarchical_parts["allowed_diag_mask"] = None
                    # Rolling head diagnostics (epoch-level)
                    try:
                        rl = hierarchical_parts.get("logits_fault_roll_raw", None)
                        if rl is not None:
                            if rl.dim() == 2 and rl.size(1) == 2:
                                roll_logit_diff = rl[:, 1] - rl[:, 0]
                            else:
                                roll_logit_diff = rl.view(-1)
                            roll_prob = torch.sigmoid(roll_logit_diff.detach())
                            try:
                                roll_fault_only = bool(getattr(cfg, "ROLLING_TRAIN_ON_FAULT_ONLY", True))
                            except Exception:
                                roll_fault_only = True
                            roll_mask = (labels > 0) if roll_fault_only else torch.ones_like(labels, dtype=torch.bool)
                            if roll_mask.any():
                                roll_targets = (labels[roll_mask] == 3).long()
                                roll_probs = roll_prob[roll_mask]
                                pos_mask = (roll_targets == 1)
                                neg_mask = (roll_targets == 0)
                                if pos_mask.any():
                                    roll_pos_count += int(pos_mask.sum().item())
                                    roll_pos_prob_sum += float(roll_probs[pos_mask].sum().item())
                                if neg_mask.any():
                                    roll_neg_count += int(neg_mask.sum().item())
                                    roll_neg_prob_sum += float(roll_probs[neg_mask].sum().item())
                                roll_sample_count += int(roll_targets.numel())
                    except Exception:
                        pass
                fusion_logits.hierarchical_parts = hierarchical_parts
            else:
                # Flat head can also return parts; we need them for Stage1 SupCon+CE diagnostics/training.
                try:
                    fusion_logits, hierarchical_parts = fusion_decoder(
                        z_fusion_for_decoder,
                        domain_ids=domain_ids,
                        text_prototypes=current_prototypes,
                        return_parts=True,
                        dann_alpha=(dann_alpha_now if ("dann_alpha_now" in locals()) and (dann_alpha_now > 0.0) else None),
                        rolling_dann_alpha=(roll_dann_alpha_now if ("roll_dann_alpha_now" in locals()) and (roll_dann_alpha_now > 0.0) else None),
                        z_signal=z_signal if ("z_signal" in locals() and z_signal is not None and str(getattr(cfg, 'DANN_FEATURE', 'z_gate')).lower() == 'z_signal') else None,
                    )
                except TypeError:
                    # Backward compatibility: if signature mismatch, fallback to logits-only.
                    fusion_logits = fusion_decoder(z_fusion_for_decoder, domain_ids=domain_ids)
                    hierarchical_parts = None
                except Exception:
                    fusion_logits = fusion_decoder(z_fusion_for_decoder, domain_ids=domain_ids)
                    hierarchical_parts = None
                # Best-effort attach for any downstream legacy usage (do not rely on it for Stage1-CE).
                try:
                    if isinstance(hierarchical_parts, dict):
                        # ATT371.1 hotfix:
                        # Keep flat-head branch consistent with hierarchical branch so
                        # multimodal_alignment_loss receives complete runtime context.
                        hierarchical_parts["nf_pos_weight_global"] = nf_pos_weight_global
                        hierarchical_parts["roll_dann_lambda"] = float(roll_dann_lambda)
                        hierarchical_parts["roll_dann_alpha"] = float(roll_dann_alpha_now)
                        hierarchical_parts["nf_pos_weight_multi"] = nf_pos_weight_multi
                        if 'fault_type_weights' in locals() and fault_type_weights is not None:
                            hierarchical_parts["fault_type_weights"] = fault_type_weights
                        hierarchical_parts["domain_ids"] = domain_ids
                        hierarchical_parts["current_epoch"] = current_epoch
                        hierarchical_parts["margin_router_target"] = margin_router_target
                        try:
                            use_flat_diag = bool(getattr(cfg, "USE_FLAT_DIAG_HEAD", False))
                        except Exception:
                            use_flat_diag = False
                        hierarchical_parts["force_flat_ce"] = bool(use_two_stage or use_flat_diag)
                        hierarchical_parts["dann_lambda"] = float(dann_lambda)
                        hierarchical_parts["dann_alpha"] = float(dann_alpha_now)
                        try:
                            hierarchical_parts["allowed_diag_mask"] = build_allowed_diag_mask(
                                domain_ids=domain_ids,
                                num_classes=int(fusion_logits.size(1)),
                                device=fusion_logits.device
                            )
                        except Exception:
                            hierarchical_parts["allowed_diag_mask"] = None
                        fusion_logits.hierarchical_parts = hierarchical_parts
                except Exception:
                    pass

            # ===== stability guard: skip pathological batches =====
            # 1) If fusion embedding is non-finite, gradients will explode unpredictably.
            if not torch.isfinite(z_fusion).all():
                skipped_bad_batches += 1
                if not warned_bad_batch:
                    warned_bad_batch = True
                    try:
                        zmin = float(torch.nan_to_num(z_fusion).min().detach().cpu())
                        zmax = float(torch.nan_to_num(z_fusion).max().detach().cpu())
                    except Exception:
                        zmin, zmax = float("nan"), float("nan")
                    print(f"[Stability] Skip batch (non-finite z_fusion) at epoch {current_epoch}, batch {batch_idx}. "
                          f"z_fusion[min,max]=({zmin:.3f},{zmax:.3f})")
                continue

            # Source vibration centroids must live in the same space as prototype CE and evaluation.
            if semantic_proto_source == "signal_centroid" and semantic_centroid_space == "metric":
                _vgsp_z = hierarchical_parts.get("z_metric") if isinstance(hierarchical_parts, dict) else None
                if not torch.is_tensor(_vgsp_z):
                    raise RuntimeError("Metric centroid requested but decoder did not return z_metric")
                _vgsp_z = F.normalize(_vgsp_z.detach().float(), dim=-1, eps=1e-6)
                if embed_centroid_sum is None:
                    embed_centroid_sum = torch.zeros((4, int(_vgsp_z.size(1))), device=_vgsp_z.device, dtype=torch.float32)
                for _c in range(4):
                    _m = labels == int(_c)
                    if bool(_m.any()):
                        embed_centroid_sum[_c] += _vgsp_z[_m].sum(dim=0)
                        embed_centroid_cnt[_c] += int(_m.sum().item())

            # ===== (S-SEM2) Update EMA visual prototypes on source training =====
            # This provides more data-grounded prototypes for semantic scoring under domain shift.
            try:
                import config as cfg
                use_vp = bool(getattr(cfg, "USE_VISUAL_PROTOTYPES", True))
            except Exception:
                use_vp = True
            if use_vp and hasattr(fusion_decoder, "update_visual_prototypes") and hasattr(fusion_decoder, "get_sem_embedding"):
                try:
                    import config as cfg
                    use_z_sem = bool(getattr(cfg, "SEMANTIC_SCORE_USE_Z_SEM", True))
                except Exception:
                    use_z_sem = True
                try:
                    with torch.no_grad():
                        _z_metric_upd = None
                        if isinstance(hierarchical_parts, dict):
                            _z_metric_upd = hierarchical_parts.get("z_metric", None)
                        _z_base = _z_metric_upd if torch.is_tensor(_z_metric_upd) else z_fusion
                        z_upd = fusion_decoder.get_sem_embedding(_z_base) if use_z_sem else _z_base
                    fusion_decoder.update_visual_prototypes(z_upd, labels)
                except Exception:
                    pass

            # 2) If prototype-head produces masked/invalid true-class logits (≈ -1e4), CE can explode (Fusion Cls 300+).
            try:
                if hasattr(fusion_logits, "hierarchical_parts"):
                    _parts = getattr(fusion_logits, "hierarchical_parts", None)
                    if isinstance(_parts, dict):
                        logits_diag = _parts.get("logits_diag", None)
                        if logits_diag is not None and logits_diag.dim() == 2 and logits_diag.size(0) == labels.size(0):
                            with torch.no_grad():
                                y = labels.clamp(min=0, max=logits_diag.size(1) - 1)
                                idx = torch.arange(labels.size(0), device=labels.device)
                                true_logits = logits_diag[idx, y]
                                bad = (true_logits < -1e3)
                                if bad.any():
                                    skipped_bad_batches += 1
                                    if not warned_bad_batch:
                                        warned_bad_batch = True
                                        frac = float(bad.float().mean().detach().cpu())
                                        tl_min = float(true_logits.min().detach().cpu())
                                        tl_max = float(true_logits.max().detach().cpu())
                                        ld_min = float(logits_diag.min().detach().cpu())
                                        ld_max = float(logits_diag.max().detach().cpu())
                                        dom_u = "n/a"
                                        try:
                                            if domain_ids is not None:
                                                dom_u = str(sorted(set(int(x) for x in domain_ids.detach().cpu().tolist())))
                                        except Exception:
                                            pass
                                        print(f"[Stability] Skip batch (true-class logits masked) at epoch {current_epoch}, batch {batch_idx}. "
                                              f"bad_frac={frac:.3f}, true_logit[min,max]=({tl_min:.1f},{tl_max:.1f}), "
                                              f"logits_diag[min,max]=({ld_min:.1f},{ld_max:.1f}), domain_ids={dom_u}")
                                    continue
            except Exception:
                # Never crash training because of guard logic.
                pass
            
            # ===== 一致性正则：每个batch都执行 =====
            z_fusion_aug = None
            augment_indices = None
            if use_consistency_regularization and signal_encoder.training:
                batch_size = signal.size(0)
                n_augment = max(2, int(batch_size * consistency_sample_ratio))  # 至少增强2个样本（BatchNorm需要）
                
                # 如果batch太小，确保不超过batch_size
                if n_augment > batch_size:
                    n_augment = batch_size
                
                # 只有当至少有2个样本时才做一致性正则（BatchNorm要求）
                if n_augment >= 2:
                    # 随机选择部分样本进行增强
                    augment_indices = torch.randperm(batch_size, device=signal.device)[:n_augment]

                    # ===== (A) 一致性保护：仅在高置信样本上施加一致性约束 =====
                    try:
                        import config as cfg
                        protect_on = bool(getattr(cfg, "CONSISTENCY_PROTECT_ENABLE", True))
                        min_conf = float(getattr(cfg, "CONSISTENCY_PROTECT_MIN_CONF", 0.60))
                    except Exception:
                        protect_on = True
                        min_conf = 0.60

                    if protect_on and augment_indices is not None and augment_indices.numel() > 0:
                        try:
                            with torch.no_grad():
                                probs = torch.softmax(fusion_logits.detach(), dim=1)
                                conf = probs.max(dim=1).values
                                keep = conf[augment_indices] >= float(min_conf)
                            if keep.any():
                                augment_indices = augment_indices[keep]
                            else:
                                augment_indices = None
                        except Exception:
                            pass

                    # 经过“高置信筛选”后，可能只剩 1 个样本；BatchNorm 在 training 下要求每通道>1个值
                    # （典型触发：子batch大小=1 且/或时序长度被下采样到 1）。
                    if augment_indices is None or augment_indices.numel() < 2:
                        augment_indices = None
                        z_fusion_aug = None
                    else:
                        # 只对选中的样本做"最轻量"的增强：amplitude_scale
                        signal_selected = signal[augment_indices]
                        signal_mask_selected = signal_mask[augment_indices]
                        signal_selected_aug = amplitude_scale(signal_selected, scale_range=(0.85, 1.15))

                        # 仅对子batch重新编码signal
                        try:
                            if signal_selected_aug is not None and torch.is_tensor(signal_selected_aug):
                                if signal_selected_aug.dim() == 1:
                                    signal_selected_aug = signal_selected_aug.unsqueeze(0)
                            if signal_mask_selected is not None and torch.is_tensor(signal_mask_selected):
                                if signal_mask_selected.dim() == 1:
                                    signal_mask_selected = signal_mask_selected.unsqueeze(0)
                            # Final guard: BatchNorm requires >1 value per channel in training (N*L > 1).
                            # If sub-batch accidentally collapses to 1 sample, skip consistency branch.
                            if signal_selected_aug is None or (torch.is_tensor(signal_selected_aug) and signal_selected_aug.size(0) < 2):
                                augment_indices = None
                                z_fusion_aug = None
                            else:
                                z_signal_aug = signal_encoder(signal_selected_aug, signal_mask_selected)
                        except Exception:
                            augment_indices = None
                            z_fusion_aug = None

                        # 复用原始z_spec的子集（可选：detach以减少一致性项对spec分支的反传开销）
                        if augment_indices is not None:
                            z_spec_subset = z_spec_for_fusion[augment_indices].detach()
                        else:
                            z_spec_subset = None

                        # 仅对子batch计算增强版fusion embedding
                        if augment_indices is not None and z_spec_subset is not None and ("z_signal_aug" in locals()) and (z_signal_aug is not None):
                            z_fusion_aug = fusion_module(z_signal_aug, z_spec_subset)
                        else:
                            z_fusion_aug = None
            
            # 如果使用贝叶斯原型且处于更新阶段，收集类别embeddings
            if bayesian_update_active:
                with torch.no_grad():
                    # 批量收集：先收集到GPU，最后再一次性移到CPU
                    labels_np = labels.cpu().numpy()
                    for label_int in np.unique(labels_np):
                        mask = labels_np == label_int
                        if label_int in class_embeddings:
                            # 批量收集同一类的embeddings，减少循环开销
                            class_embeddings[label_int].append(z_fusion[mask].detach().cpu())

            # ===== 混合精度训练（AMP） =====
            if use_amp and scaler is not None:
                # 使用新的 torch.amp API（PyTorch 2.0+）
                if hasattr(torch.amp, 'autocast'):
                    autocast_context = torch.amp.autocast('cuda')
                else:
                    # 兼容旧版本
                    autocast_context = torch.cuda.amp.autocast()
                with autocast_context:
                    # ===== Pre-initialize z_for_sem to avoid UnboundLocalError in non-Stage2 paths =====
                    z_for_sem = z_fusion
                    z_for_sem_aug = z_fusion_aug
                    
                    # ===== (2) Stage1 NF-only: only NF + 3a+3b; no 4-class CE =====
                    stage1_nf_only_now = False
                    try:
                        stage1_nf_only_now = bool(locals().get("stage1_mode_now", False)) and bool(getattr(cfg, "STAGE1_NF_ONLY", False))
                    except Exception:
                        pass
                    if supcon_stage1_now:
                        z_for_supcon = z_fusion
                        # NOTE (stability): do NOT rely on attaching attributes to Tensor (may be lost under AMP/autocast).
                        # Use the explicit `hierarchical_parts` dict captured from fusion_decoder(return_parts=True).
                        hp = hierarchical_parts if isinstance(hierarchical_parts, dict) else getattr(fusion_logits, "hierarchical_parts", None)
                        try:
                            if isinstance(hp, dict) and hp.get("z_cls", None) is not None:
                                z_for_supcon = hp.get("z_cls", z_fusion)
                        except Exception:
                            z_for_supcon = z_fusion
                        try:
                            if hasattr(fusion_decoder, "project_for_supcon"):
                                z_for_supcon = fusion_decoder.project_for_supcon(z_for_supcon)
                        except Exception:
                            pass
                        sup_labels = labels
                        if supcon_fault_only:
                            mask = labels > 0
                            z_for_supcon = z_for_supcon[mask]
                            sup_labels = labels[mask]
                        loss_supcon, sup_stats = supervised_contrastive_loss(
                            z_for_supcon,
                            sup_labels,
                            temperature=supcon_temp,
                            use_class_weights=supcon_use_class_weights,
                            return_stats=True
                        )
                        loss = supcon_weight * loss_supcon
                        loss_dict = {"supcon": float(loss_supcon.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                        epoch_supcon_loss += float(loss_supcon.detach().cpu().item())
                        supcon_batches += 1
                        loss_path_supcon_batches += 1
                        # ATT254(S2): enforce variance floor on pre-normalized z_cls to counter
                        # stage1 dimensional collapse before L2 normalization in SupCon pipeline.
                        try:
                            use_supcon_prenorm_var = bool(getattr(cfg, "USE_SUPCON_PRENORM_VAR_LOSS", False))
                            supcon_prenorm_var_w = float(getattr(cfg, "SUPCON_PRENORM_VAR_WEIGHT", 0.0))
                            supcon_prenorm_std_target = float(getattr(cfg, "SUPCON_PRENORM_STD_TARGET", 0.10))
                        except Exception:
                            use_supcon_prenorm_var = False
                            supcon_prenorm_var_w = 0.0
                            supcon_prenorm_std_target = 0.10
                        if use_supcon_prenorm_var and supcon_prenorm_var_w > 0.0 and isinstance(hp, dict):
                            z_cls_pre_norm = hp.get("z_cls_pre_norm", None)
                            if torch.is_tensor(z_cls_pre_norm):
                                if supcon_fault_only:
                                    _pre_mask = labels > 0
                                    z_cls_pre_norm = z_cls_pre_norm[_pre_mask]
                                if int(z_cls_pre_norm.size(0)) >= 2:
                                    z_pre = z_cls_pre_norm.float()
                                    z_pre = z_pre - z_pre.mean(dim=0, keepdim=True)
                                    std_pre = torch.sqrt(z_pre.var(dim=0, unbiased=False) + 1e-6)
                                    l_pre_var = F.relu(float(supcon_prenorm_std_target) - std_pre).mean()
                                    loss = loss + float(supcon_prenorm_var_w) * l_pre_var
                                    loss_dict["supcon_pre_var"] = float((float(supcon_prenorm_var_w) * l_pre_var).detach().cpu().item())
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                                    global_decorr_pre_std_sum += float(std_pre.mean().detach().cpu().item())
                                    global_decorr_pre_std_batches += 1
                        if float(supcon_spec_weight) > 0.0:
                            z_spec_supcon = z_spec
                            try:
                                if hasattr(spec_encoder, "project_for_supcon"):
                                    z_spec_supcon = spec_encoder.project_for_supcon(z_spec_supcon)
                            except Exception:
                                pass
                            spec_labels = labels
                            if supcon_fault_only:
                                spec_mask = labels > 0
                                z_spec_supcon = z_spec_supcon[spec_mask]
                                spec_labels = spec_labels[spec_mask]
                            loss_supcon_spec, _ = supervised_contrastive_loss(
                                z_spec_supcon,
                                spec_labels,
                                temperature=supcon_temp,
                                use_class_weights=supcon_use_class_weights,
                                return_stats=True
                            )
                            loss = loss + float(supcon_spec_weight) * loss_supcon_spec
                            loss_dict["supcon_spec"] = float(loss_supcon_spec.detach().cpu().item())
                            loss_dict["total"] = float(loss.detach().cpu().item())
                        try:
                            supcon_pos_pairs += int(sup_stats.get("pos_pairs", 0))
                            supcon_valid_anchors += int(sup_stats.get("valid_anchors", 0))
                        except Exception:
                            pass
                        # ===== (Attempt 28 Fix) Add lightweight CE loss in Stage1 to train classifier =====
                        supcon_ce_w = float(getattr(cfg, "SUPCON_STAGE1_CE_WEIGHT", 0.3))
                        if supcon_ce_w > 0:
                            if not isinstance(hp, dict):
                                stage1_ce_missing_hp_batches += 1
                            else:
                                diag_logits = hp.get("logits_diag_supervised", None)
                                if diag_logits is None:
                                    diag_logits = hp.get("logits_diag", None)
                                if diag_logits is None or (not torch.is_tensor(diag_logits)) or diag_logits.dim() != 2 or diag_logits.size(1) < 4:
                                    stage1_ce_missing_logits_batches += 1
                                else:
                                    stage1_ce_weights = None
                                    try:
                                        if bool(getattr(cfg, "STAGE1_CE_USE_CLASS_WEIGHTS", False)) and torch.is_tensor(class_weights):
                                            if int(class_weights.numel()) == int(diag_logits.size(1)):
                                                stage1_ce_weights = class_weights.to(device=diag_logits.device, dtype=torch.float32)
                                    except Exception:
                                        stage1_ce_weights = None
                                    stage1_use_focal = bool(getattr(cfg, "STAGE1_CE_USE_FOCAL", False))
                                    if core_loss_only_mode:
                                        stage1_use_focal = False
                                    stage1_focal_gamma = float(getattr(cfg, "STAGE1_CE_FOCAL_GAMMA", 2.0))
                                    # Att77: Focal延迟启动，防止早期Signal欠训练
                                    focal_start_epoch = int(getattr(cfg, "FOCAL_START_EPOCH", 1))
                                    if stage1_use_focal and current_epoch < focal_start_epoch:
                                        stage1_use_focal = False  # 在达到启动epoch前使用普通CE
                                    stage1_focal_use_class_alpha = bool(
                                        getattr(cfg, "STAGE1_CE_FOCAL_USE_CLASS_ALPHA", True)
                                    )
                                    stage1_loss_weights = (
                                        stage1_ce_weights
                                        if (not stage1_use_focal or stage1_focal_use_class_alpha)
                                        else None
                                    )
                                    loss_ce_stage1 = compute_diag_loss_with_focal(
                                        logits=diag_logits.float(),
                                        labels=labels,
                                        class_weight_tensor=stage1_loss_weights,
                                        use_focal_loss=stage1_use_focal,
                                        focal_gamma=stage1_focal_gamma,
                                        focal_alpha=None,
                                        label_smoothing=0.0,
                                        use_balanced_softmax=use_balanced_softmax,
                                        class_counts=class_counts_tensor,
                                    )
                                    loss = loss + supcon_ce_w * loss_ce_stage1
                                    loss_dict["stage1_ce"] = float(loss_ce_stage1.detach().cpu().item())
                                    loss_dict["_info_fusion_cls"] = float(loss_ce_stage1.detach().cpu().item())  # ATT374: _info_ prefix
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                                    # Accumulate Stage1 CE statistics
                                    epoch_stage1_ce_loss += float(loss_ce_stage1.detach().cpu().item())
                                    stage1_ce_batches += 1
                                    # Per-class CE diagnostics (mean over samples per class)
                                    try:
                                        if labels is not None and labels.ndim == 1 and labels.numel() > 0:
                                            ce_vec = F.cross_entropy(diag_logits.float(), labels, reduction="none")
                                            for _c in range(4):
                                                _m = (labels == int(_c))
                                                if bool(_m.any()):
                                                    stage1_ce_class_loss_sum[_c] += float(ce_vec[_m].detach().mean().cpu().item())
                                                    stage1_ce_class_count[_c] += 1
                                    except Exception:
                                        pass
                        # ATT141: Reverted ATT139 Stage1 semantic CE injection.
                        # ATT139 (w=0.3) and ATT140 (w=0.03) both caused degradation: text prototypes
                        # are anti-aligned with z_spec space (spec_proto_diag_acc < random baseline 0.25),
                        # so any prototype CE gradient actively misdirects z_cls features.
                        # The "no man's land" effect at w=0.03: too weak to find prototype-aligned basin,
                        # too strong to stay in ATT134's discriminative basin → val_loss explodes E6+.

                        # ===== (Attempt 35) IO↔Rolling separation loss & centroid logging (Stage1 only, AMP branch) =====
                        try:
                            import config as _cfg
                            use_sep = bool(getattr(_cfg, "USE_IO_ROLL_SEP_LOSS", False))
                            sep_w = float(getattr(_cfg, "IO_ROLL_SEP_WEIGHT", 0.0))
                            sep_max_sim = float(getattr(_cfg, "IO_ROLL_SEP_MAX_SIM", 0.35))
                            sep_start = int(getattr(_cfg, "IO_ROLL_SEP_START_EPOCH", 1))
                            log_centroid = bool(getattr(_cfg, "LOG_EMBED_CENTROID_SIM", False))
                            # Attempt 32: Inner↔Outer separation (primary fix for IO collapse)
                            use_io_sep = bool(getattr(_cfg, "USE_IO_SEP_LOSS", False))
                            io_sep_w = float(getattr(_cfg, "IO_SEP_WEIGHT", 0.0))
                            io_sep_max_sim = float(getattr(_cfg, "IO_SEP_MAX_SIM", 0.30))
                            io_sep_start = int(getattr(_cfg, "IO_SEP_START_EPOCH", 1))
                            # Attempt 33: Normal↔Fault separation (primary fix for Normal absorption)
                            use_nf_sep = bool(getattr(_cfg, "USE_NF_SEP_LOSS", False))
                            nf_sep_w = float(getattr(_cfg, "NF_SEP_WEIGHT", 0.0))
                            nf_sep_max_sim = float(getattr(_cfg, "NF_SEP_MAX_SIM", 0.20))
                            nf_sep_start = int(getattr(_cfg, "NF_SEP_START_EPOCH", 1))
                            # Att78: small-sample regime — disable NF-Sep when batches per epoch <= threshold (high-variance centroids).
                            nf_sep_min_batches = int(getattr(_cfg, "NF_SEP_MIN_BATCHES", 5))
                            if len(train_loader) < nf_sep_min_batches:
                                nf_sep_w = 0.0
                            # Attempt 35: Inner↔Rolling separation (fix gradient vanishing + relax threshold)
                            use_ir_sep = bool(getattr(_cfg, "USE_IR_SEP_LOSS", False))
                            ir_sep_w = float(getattr(_cfg, "IR_SEP_WEIGHT", 0.0))
                            ir_sep_max_sim = float(getattr(_cfg, "IR_SEP_MAX_SIM", 0.15))  # relaxed from 0.0
                            ir_sep_start = int(getattr(_cfg, "IR_SEP_START_EPOCH", 1))
                            ir_sep_maint_w = float(getattr(_cfg, "IR_SEP_MAINTENANCE_WEIGHT", 0.01))  # NEW
                        except Exception:
                            use_sep, sep_w, sep_max_sim, sep_start, log_centroid = False, 0.0, 0.35, 1, False
                            use_io_sep, io_sep_w, io_sep_max_sim, io_sep_start = False, 0.0, 0.30, 1
                            use_nf_sep, nf_sep_w, nf_sep_max_sim, nf_sep_start = False, 0.0, 0.20, 1
                            use_ir_sep, ir_sep_w, ir_sep_max_sim, ir_sep_start = False, 0.0, 0.15, 1
                            ir_sep_maint_w = 0.01  # default maintenance weight
                        if core_loss_only_mode:
                            use_sep = use_io_sep = use_nf_sep = use_ir_sep = False
                            sep_w = io_sep_w = nf_sep_w = ir_sep_w = 0.0

                        try:
                            z_sep_full = None
                            if isinstance(hp, dict):
                                z_sep_full = hp.get("z_cls", None)
                            if z_sep_full is None:
                                z_sep_full = z_fusion
                            if torch.is_tensor(z_sep_full) and z_sep_full.dim() == 2 and labels is not None and labels.dim() == 1:
                                z_sep_full = F.normalize(z_sep_full.float(), dim=-1, eps=1e-6)

                                # Epoch-level centroid accumulation (for diagnostics)
                                if (log_centroid or semantic_proto_source == "signal_centroid") and not (semantic_proto_source == "signal_centroid" and semantic_centroid_space == "metric"):
                                    if embed_centroid_sum is None:
                                        embed_centroid_sum = torch.zeros((4, int(z_sep_full.size(1))), device=z_sep_full.device, dtype=torch.float32)
                                    for _c in range(4):
                                        _m = (labels == int(_c))
                                        if bool(_m.any()):
                                            embed_centroid_sum[_c] += z_sep_full[_m].sum(dim=0)
                                            embed_centroid_cnt[_c] += int(_m.sum().item())

                                # Targeted separation: penalize high cosine sim for specific centroid pairs
                                need_mu = (
                                    (use_sep and (sep_w > 0.0) and (current_epoch >= int(sep_start)))
                                    or (use_io_sep and (io_sep_w > 0.0) and (current_epoch >= int(io_sep_start)))
                                    or (use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start)))
                                )
                                if need_mu:
                                    mu0 = mu1 = mu2 = mu3 = mu3_diag = None
                                    m0 = (labels == 0)
                                    m1 = (labels == 1)
                                    m2 = (labels == 2)
                                    m3 = (labels == 3)
                                    if (use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start))) and bool(m0.any()):
                                        mu0 = F.normalize(z_sep_full[m0].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    if bool(m1.any()):
                                        mu1 = F.normalize(z_sep_full[m1].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    if bool(m2.any()):
                                        mu2 = F.normalize(z_sep_full[m2].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    if (
                                        (use_sep and (sep_w > 0.0) and (current_epoch >= int(sep_start)))
                                        and bool(m3.any())
                                    ):
                                        mu3 = F.normalize(z_sep_full[m3].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    elif (use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start))) and bool(m3.any()):
                                        mu3_diag = F.normalize(z_sep_full[m3].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)

                                    updated = False
                                    # (A) IO↔Rolling separation loss
                                    if use_sep and (sep_w > 0.0) and (current_epoch >= int(sep_start)):
                                        sep_loss = None
                                        loss_terms = []
                                        if (mu1 is not None) and (mu3 is not None):
                                            sim13 = float((mu1 * mu3).sum().detach().cpu().item())
                                            io_roll_sep_sim13_sum += sim13
                                            io_roll_sep_sim13_cnt += 1
                                            loss_terms.append(F.relu((mu1 * mu3).sum() - float(sep_max_sim)))
                                        if (mu2 is not None) and (mu3 is not None):
                                            sim23 = float((mu2 * mu3).sum().detach().cpu().item())
                                            io_roll_sep_sim23_sum += sim23
                                            io_roll_sep_sim23_cnt += 1
                                            loss_terms.append(F.relu((mu2 * mu3).sum() - float(sep_max_sim)))
                                        if len(loss_terms) > 0:
                                            sep_loss = torch.stack(loss_terms).mean()
                                            loss = loss + float(sep_w) * sep_loss
                                            loss_dict["io_roll_sep"] = float((float(sep_w) * sep_loss).detach().cpu().item())
                                            epoch_io_roll_sep_loss += float(sep_loss.detach().cpu().item())
                                            io_roll_sep_batches += 1
                                            updated = True

                                    # (B) Inner↔Outer separation loss (Attempt 32): penalize high cosine sim(Inner, Outer)
                                    if use_io_sep and (io_sep_w > 0.0) and (current_epoch >= int(io_sep_start)):
                                        if (mu1 is not None) and (mu2 is not None):
                                            sim12 = float((mu1 * mu2).sum().detach().cpu().item())
                                            io_sep_sim12_sum += sim12
                                            io_sep_sim12_cnt += 1
                                            io_sep_loss = F.relu((mu1 * mu2).sum() - float(io_sep_max_sim))
                                            loss = loss + float(io_sep_w) * io_sep_loss
                                            loss_dict["io_sep"] = float((float(io_sep_w) * io_sep_loss).detach().cpu().item())
                                            epoch_io_sep_loss += float(io_sep_loss.detach().cpu().item())
                                            io_sep_batches += 1
                                            updated = True

                                    # (C) Normal↔Fault separation loss (Attempt 33): penalize high cosine sim(N, fault_k)
                                    if use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start)):
                                        if (mu0 is not None) and ((mu1 is not None) or (mu2 is not None) or (mu3 is not None)):
                                            nf_terms = []
                                            if mu1 is not None:
                                                sim01 = float((mu0 * mu1).sum().detach().cpu().item())
                                                nf_sep_sim01_sum += sim01
                                                nf_sep_sim01_cnt += 1
                                                nf_terms.append(F.relu((mu0 * mu1).sum() - float(nf_sep_max_sim)))
                                            if mu2 is not None:
                                                sim02 = float((mu0 * mu2).sum().detach().cpu().item())
                                                nf_sep_sim02_sum += sim02
                                                nf_sep_sim02_cnt += 1
                                                nf_terms.append(F.relu((mu0 * mu2).sum() - float(nf_sep_max_sim)))
                                            if mu3 is not None:
                                                sim03 = float((mu0 * mu3).sum().detach().cpu().item())
                                                nf_sep_sim03_sum += sim03
                                                nf_sep_sim03_cnt += 1
                                                nf_terms.append(F.relu((mu0 * mu3).sum() - float(nf_sep_max_sim)))
                                            if len(nf_terms) > 0:
                                                nf_loss = torch.stack(nf_terms).mean()
                                                loss = loss + float(nf_sep_w) * nf_loss
                                                loss_dict["nf_sep"] = float((float(nf_sep_w) * nf_loss).detach().cpu().item())
                                                epoch_nf_sep_loss += float(nf_loss.detach().cpu().item())
                                                nf_sep_batches += 1
                                                updated = True

                                    # Diagnostic-only cos_sim(N,R): do not add to loss (Att70: mu3 in loss causes high-variance gradient)
                                    if mu0 is not None and mu3_diag is not None:
                                        sim03_diag = float((mu0 * mu3_diag).sum().detach().cpu().item())
                                        nf_sep_sim03_sum += sim03_diag
                                        nf_sep_sim03_cnt += 1

                                    # (D) Inner↔Rolling separation loss (Attempt 35): penalize high cosine sim(I, R)
                                    # Fixed: Add maintenance gradient to prevent hinge loss gradient vanishing
                                    if use_ir_sep and (ir_sep_w > 0.0) and (current_epoch >= int(ir_sep_start)):
                                        if (mu1 is not None) and (mu3 is not None):
                                            sim_val = (mu1 * mu3).sum()
                                            sim13 = float(sim_val.detach().cpu().item())
                                            ir_sep_sim13_sum += sim13
                                            ir_sep_sim13_cnt += 1
                                            # Hinge loss: main constraint (zero gradient when target achieved)
                                            ir_hinge = F.relu(sim_val - float(ir_sep_max_sim))
                                            # Maintenance gradient: continuous small gradient even after target achieved
                                            # This prevents other losses (SupCon/CE) from pulling I-R back together
                                            ir_maintenance = ir_sep_maint_w * sim_val  # always pulls toward lower sim
                                            ir_loss = ir_hinge + ir_maintenance
                                            loss = loss + float(ir_sep_w) * ir_loss
                                            loss_dict["ir_sep"] = float((float(ir_sep_w) * ir_loss).detach().cpu().item())
                                            loss_dict["ir_hinge"] = float(ir_hinge.detach().cpu().item())
                                            loss_dict["ir_maint"] = float(ir_maintenance.detach().cpu().item())
                                            epoch_ir_sep_loss += float(ir_loss.detach().cpu().item())
                                            ir_sep_batches += 1
                                            updated = True

                                    if updated:
                                        loss_dict["total"] = float(loss.detach().cpu().item())
                        except Exception:
                            pass
                        # ATT177: 回退 ATT176 — Stage1 不加入 SpecAlign（ATT139/140/141/176 证伪）

                    else:
                        parts_now = getattr(fusion_logits, "hierarchical_parts", None)
                        if stage1_nf_only_now and isinstance(parts_now, dict):
                            logits_nf_global = parts_now.get("logits_nf_global", None)
                            z_gate = parts_now.get("z_gate", None)
                            if logits_nf_global is not None and logits_nf_global.dim() == 2 and logits_nf_global.size(1) == 2:
                                labels_nf = (labels > 0).long()
                                pos_w = parts_now.get("nf_pos_weight_global", None)
                                pw = float(pos_w) if pos_w is not None else 1.0
                                w_nf = torch.tensor([1.0, max(1.0, pw)], device=logits_nf_global.device, dtype=torch.float32)
                                loss_nf_ce = F.cross_entropy(logits_nf_global.float(), labels_nf, weight=w_nf)
                                nf_w_prim = float(getattr(cfg, "STAGE_NF_LOSS_WEIGHT", 0.10))
                                loss = loss_nf_ce * nf_w_prim
                                loss_dict = {"nf_global": float(loss.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                                if z_gate is not None and getattr(cfg, "USE_NF_CONTRASTIVE_LOSS", True):
                                    w_c = float(getattr(cfg, "NF_CONTRASTIVE_WEIGHT", 0.15))
                                    tau = float(getattr(cfg, "NF_CONTRASTIVE_TEMPERATURE", 0.07))
                                    l_c = nf_contrastive_loss(z_gate, labels_nf, temperature=tau)
                                    loss = loss + w_c * l_c
                                    loss_dict["nf_contrastive"] = float((w_c * l_c).detach().cpu().item())
                                if getattr(cfg, "USE_NF_MARGIN_LOSS", True) and current_epoch >= int(getattr(cfg, "NF_MARGIN_START_EPOCH", 1)):
                                    w_m = float(getattr(cfg, "NF_MARGIN_WEIGHT", 0.10))
                                    margin = float(getattr(cfg, "NF_MARGIN", 0.25))
                                    l_m = nf_margin_loss(logits_nf_global, labels_nf, margin=margin)
                                    loss = loss + w_m * l_m
                                    loss_dict["nf_margin"] = float((w_m * l_m).detach().cpu().item())
                                if z_gate is not None and getattr(cfg, "USE_NORMAL_REFERENCE_LOSS", True) and hasattr(fusion_decoder, "normal_reference"):
                                    ref = fusion_decoder.normal_reference
                                    if ref is not None and ref.numel() == z_gate.size(-1):
                                        w_ref = float(getattr(cfg, "NORMAL_REFERENCE_WEIGHT", 0.10))
                                        l_ref = normal_reference_loss(z_gate, labels_nf, ref, weight_fault_push=1.0)
                                        loss = loss + w_ref * l_ref
                                        loss_dict["normal_ref"] = float((w_ref * l_ref).detach().cpu().item())
                                loss_dict["total"] = float(loss.detach().cpu().item())
                            else:
                                # Fallback: if nf logits missing, avoid zero-grad crash.
                                try:
                                    allowed = None
                                    if isinstance(parts_now, dict):
                                        allowed = parts_now.get("allowed_diag_mask", None)
                                    loss = fusion_ce_loss(
                                        fusion_logits,
                                        labels,
                                        class_weights_for_loss,
                                        allowed_mask=allowed,
                                        use_focal_loss=use_focal_loss,
                                        focal_gamma=focal_gamma,
                                        focal_alpha=focal_alpha,
                                        label_smoothing=label_smoothing,
                                        use_balanced_softmax=use_balanced_softmax,
                                        class_counts=class_counts_tensor,
                                    )
                                    loss_dict = {
                                        "diag_ce": float(loss.detach().cpu().item()),
                                        "total": float(loss.detach().cpu().item()),
                                        "_info_fusion_cls": float(loss.detach().cpu().item()),
                                        "_info_nf_global_missing": 1.0,
                                    }
                                except Exception:
                                    loss = z_fusion.sum() * 0.0
                                    loss_dict = {"total": 0.0, "nf_global_missing": 1.0}
                        elif (current_prototypes is None) or force_ce_path_now:
                            loss_path_ce_batches += 1
                            # (New scheme 1) Two-stage fault loss: NF CE (all) + 3-class CE on fault samples only
                            use_two_stage_fault = bool(getattr(cfg, "USE_TWO_STAGE_FAULT_LOSS", False))
                            if core_loss_only_mode:
                                use_two_stage_fault = False
                            hp = getattr(fusion_logits, "hierarchical_parts", None)
                            logits_3way = hp.get("logits_3way", None) if isinstance(hp, dict) else None
                            logits_nf_global = hp.get("logits_nf_global", None) if isinstance(hp, dict) else None
                            if use_two_stage_fault and logits_3way is not None and logits_nf_global is not None:
                                labels_nf = (labels > 0).long()
                                loss_nf = F.cross_entropy(logits_nf_global.float(), labels_nf, reduction="mean")
                                fault_mask = labels > 0
                                if fault_mask.any():
                                    loss_3way = F.cross_entropy(
                                        logits_3way[fault_mask].float(),
                                        (labels[fault_mask] - 1).long(),
                                        reduction="mean",
                                    )
                                else:
                                    loss_3way = torch.tensor(0.0, device=logits_3way.device, dtype=logits_3way.dtype)
                                w_3way = float(getattr(cfg, "TWO_STAGE_FAULT_LOSS_WEIGHT", 1.0))
                                loss = loss_nf + w_3way * loss_3way
                                loss_dict = {
                                    "diag_ce": float(loss.detach().cpu().item()),
                                    "total": float(loss.detach().cpu().item()),
                                    "_info_fusion_cls": float(loss.detach().cpu().item()),
                                    "nf_ce": float(loss_nf.detach().cpu().item()),
                                    "fault_3way_ce": float(loss_3way.detach().cpu().item()),
                                }
                            else:
                                allowed = None
                                try:
                                    if isinstance(hp, dict):
                                        allowed = hp.get("allowed_diag_mask", None)
                                except Exception:
                                    allowed = None
                                
                                # ===== LDAM Loss or standard CE =====
                                if use_ldam_loss and ldam_loss_fn is not None:
                                    # Use LDAM Loss (with DRW-controlled weights)
                                    loss = ldam_loss_fn(fusion_logits, labels)
                                    loss_dict = {"diag_ce": float(loss.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                                    loss_dict["ldam_loss"] = float(loss.detach().cpu().item())
                                else:
                                    loss = fusion_ce_loss(
                                        fusion_logits,
                                        labels,
                                        class_weights_for_loss,
                                        allowed_mask=allowed,
                                        use_focal_loss=use_focal_loss,
                                        focal_gamma=focal_gamma,
                                        focal_alpha=focal_alpha,
                                        label_smoothing=label_smoothing,
                                        use_balanced_softmax=use_balanced_softmax,
                                        class_counts=class_counts_tensor,
                                    )
                                    loss_dict = {"diag_ce": float(loss.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                                loss_dict["_info_fusion_cls"] = loss_dict.get("diag_ce", 0.0)  # ATT374: moved to _info_ to fix budget double-count
                            # ATT177: 回退 ATT176 — Stage1 不加入 SpecAlign（ATT139/140/141/176 证伪）
                        else:
                            # Stage2: use z_sem for semantic alignment, keeping diagnosis space stable.
                            z_for_sem = z_fusion
                            z_for_sem_aug = z_fusion_aug
                            try:
                                _use_dual_space_now = bool(getattr(cfg, "USE_DUAL_SPACE_DECOUPLE", False))
                                _sem_metric_live_now = bool(
                                    getattr(cfg, "FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH", True)
                                )
                                _bypass_sem_adapter_now = bool(
                                    getattr(cfg, "FUSION_SEMANTIC_BYPASS_SEM_ADAPTER_ON_DIRECT_PATH", True)
                                )
                                if isinstance(hierarchical_parts, dict):
                                    _z_metric_now = hierarchical_parts.get("z_metric", None)
                                    if _use_dual_space_now and _sem_metric_live_now and hasattr(fusion_decoder, "get_metric_embedding"):
                                        # ATT480: keep metric semantic path connected to live z_fusion even when
                                        # decoder input is detached by FUSION_DECODER_INPUT_STOPGRAD_ENABLE.
                                        z_for_sem = fusion_decoder.get_metric_embedding(z_fusion)
                                    elif _use_dual_space_now and torch.is_tensor(_z_metric_now):
                                        z_for_sem = _z_metric_now
                                    if _use_dual_space_now and z_fusion_aug is not None and hasattr(fusion_decoder, "get_metric_embedding"):
                                        z_for_sem_aug = fusion_decoder.get_metric_embedding(z_fusion_aug)
                                _skip_sem_adapter_now = bool(
                                    _use_dual_space_now and _sem_metric_live_now and _bypass_sem_adapter_now
                                )
                                if (not _skip_sem_adapter_now) and hasattr(fusion_decoder, "get_sem_embedding"):
                                    z_for_sem = fusion_decoder.get_sem_embedding(z_for_sem)
                                    if z_fusion_aug is not None:
                                        z_for_sem_aug = fusion_decoder.get_sem_embedding(z_for_sem_aug if z_for_sem_aug is not None else z_fusion_aug)
                            except Exception:
                                z_for_sem = z_fusion
                                z_for_sem_aug = z_fusion_aug

                        # (Attempt 11) If using two-stage fault loss, do NOT let multimodal_alignment_loss
                        # apply its own 4-way CE on fusion logits (it causes vote-dilution under domain shift).
                        # Instead: set w_fusion_cls=0 and add NF CE + 3-way CE (fault-only) explicitly.
                        skip_stage_aux_nf = False
                        w_fusion_cls_for_align = current_w_fusion_cls
                        use_two_stage_fault_stage2 = False
                        hp_stage2 = getattr(fusion_logits, "hierarchical_parts", None)
                        logits_3way_stage2 = hp_stage2.get("logits_3way", None) if isinstance(hp_stage2, dict) else None
                        logits_nf_global_stage2 = hp_stage2.get("logits_nf_global", None) if isinstance(hp_stage2, dict) else None
                        try:
                            use_two_stage_fault_stage2 = bool(getattr(cfg, "USE_TWO_STAGE_FAULT_LOSS", False))
                        except Exception:
                            use_two_stage_fault_stage2 = False
                        if core_loss_only_mode:
                            use_two_stage_fault_stage2 = False
                        if use_two_stage_fault_stage2 and logits_3way_stage2 is not None and logits_nf_global_stage2 is not None:
                            w_fusion_cls_for_align = 0.0
                            skip_stage_aux_nf = True

                        # ATT379c: Decoupled two-stage weight override (training path)
                        _dc_w_fusion_cls = w_fusion_cls_for_align
                        _dc_w_spec = current_w_spec_align if (core_loss_only_mode and "aux_spec" in set(getattr(cfg, "CORE_LOSS_ALLOWED_TERMS", ()))) else (0.0 if core_loss_only_mode else current_w_spec_align)
                        # ATT431: restore signal CE for signal encoder supervision
                        # ATT463: split_arch increases signal weight
                        _dc_w_signal = float(getattr(cfg, "SPLIT_ARCH_SIGNAL_WEIGHT", 1.0)) if _split_arch_on else float(getattr(cfg, "W_SIGNAL_ALIGN", 0.3))
                        _dc_use_focal = False if core_loss_only_mode else use_focal_loss
                        _dc_margin_override = False
                        if _decoupled_is_stage1:
                            _dc_w_fusion_cls = 0.0
                            _dc_w_spec = 0.0
                            _dc_w_signal = 0.0
                            _dc_use_focal = False
                        elif _decoupled_is_stage2:
                            _dc_w_fusion_cls = float(getattr(cfg, "W_FUSION_CLS", 0.5))
                            _dc_w_spec = 0.0
                            _dc_w_signal = 0.0
                            _dc_margin_override = bool(getattr(cfg, "DECOUPLED_STAGE2_ENABLE_MARGINS", True))
                        _att488_skip_zcls_ce_batch = _fusion_zcls_ce_disabled_now(
                            cfg, locals().get("stage1_mode_now", stage1_mode)
                        )
                        if _att488_skip_zcls_ce_batch and not _decoupled_is_stage2:
                            _dc_w_fusion_cls = 0.0

                        loss_path_align_batches += 1
                        _att479_fus_sc = bool(getattr(cfg, "FUSION_SUPCON_ENABLE", False))
                        loss, loss_dict = multimodal_alignment_loss(
                            z_signal=z_signal,
                            z_spec=z_spec_clean,
                            z_fusion=z_for_sem,
                            z_fusion_raw=z_for_sem,
                            fusion_logits=fusion_logits,
                            hierarchical_parts=hierarchical_parts,
                            labels=labels,
                            text_prototypes=current_prototypes,
                            temperature=float(getattr(cfg, "SEMANTIC_ALIGNMENT_TEMPERATURE", 0.07)),
                            class_weights=class_weights_for_loss,
                            z_fusion_aug=z_for_sem_aug,
                            augment_indices=augment_indices,
                            w_signal=_dc_w_signal,
                            w_margin=0.0,  # ATT428(S1): margin loss disabled
                            w_consistency=0.0 if core_loss_only_mode else (consistency_weight if use_consistency_regularization else 0.0),
                            consistency_temperature=consistency_temperature,
                            w_fusion_cls=_dc_w_fusion_cls,
                            w_spec=_dc_w_spec,
                            use_focal_loss=_dc_use_focal,
                            focal_gamma=focal_gamma,
                            focal_alpha=focal_alpha,
                            label_smoothing=label_smoothing,
                            adaptive_pos_weight_global=adaptive_pos_weight_global,
                            adaptive_pos_weight_multi=adaptive_pos_weight_multi,
                            confusion_penalty_weights=confusion_penalty_weights,
                            use_supcon_loss=_supcon_core_pass and getattr(cfg, "USE_SUPCON_LOSS", True),
                            supcon_weight=getattr(cfg, "SUPCON_WEIGHT", 0.05) if _supcon_core_pass else 0.0,
                            supcon_temperature=getattr(cfg, "SUPCON_TEMPERATURE", 0.07),
                            supcon_fault_only=getattr(cfg, "SUPCON_FAULT_ONLY", True) if _supcon_core_pass else False,
                            use_fusion_supcon_loss=_att479_fus_sc,
                            fusion_supcon_weight=float(getattr(cfg, "FUSION_SUPCON_WEIGHT", 1.0)) if _att479_fus_sc else 0.0,
                            fusion_supcon_temperature=float(getattr(cfg, "FUSION_SUPCON_TEMPERATURE", 0.07)),
                            fusion_supcon_fault_only=bool(getattr(cfg, "FUSION_SUPCON_FAULT_ONLY", False)),
                            z_fusion_supcon=(z_fusion_for_reg if _att479_fus_sc else None),
                            fusion_geometry_loss_mode=str(getattr(cfg, "FUSION_GEOMETRY_LOSS_MODE", "silhouette")),
                            fusion_silhouette_eps=float(getattr(cfg, "FUSION_SILHOUETTE_EPS", 1e-6)),
                            use_proto_sep_loss=False if (core_loss_only_mode and not _dc_margin_override) else getattr(cfg, "USE_PROTO_SEP_LOSS", True),
                            proto_sep_weight=0.0 if (core_loss_only_mode and not _dc_margin_override) else getattr(cfg, "PROTO_SEP_WEIGHT", 0.02),
                            use_outer_margin_loss=False if (core_loss_only_mode and not _dc_margin_override) else getattr(cfg, "USE_OUTER_MARGIN_LOSS", True),
                            outer_margin_weight=0.0 if (core_loss_only_mode and not _dc_margin_override) else getattr(cfg, "OUTER_MARGIN_WEIGHT", 0.05),
                            outer_margin=getattr(cfg, "OUTER_MARGIN", 0.2),
                            outer_margin_start_epoch=1 if _dc_margin_override else getattr(cfg, "OUTER_MARGIN_START_EPOCH", 5),
                            use_inner_margin_loss=False if (core_loss_only_mode and not _dc_margin_override) else getattr(cfg, "USE_INNER_MARGIN_LOSS", True),
                            inner_margin_weight=0.0 if (core_loss_only_mode and not _dc_margin_override) else getattr(cfg, "INNER_MARGIN_WEIGHT", 0.05),
                            inner_margin=getattr(cfg, "INNER_MARGIN", 0.2),
                            inner_margin_start_epoch=1 if _dc_margin_override else getattr(cfg, "INNER_MARGIN_START_EPOCH", 5),
                            use_balanced_softmax=use_balanced_softmax,
                            class_counts=class_counts_tensor,
                            decoupled_stage2_margin_override=_dc_margin_override,
                            ablation_gate_bank=_ablation_gate_bank,
                        )
                        if _att488_skip_zcls_ce_batch and not _decoupled_is_stage2:
                            loss_dict = _mark_fusion_zcls_ce_skipped(loss_dict)

                        loss, loss_dict = _maybe_add_fusion_metric_learnable_proto_ce(
                            loss,
                            loss_dict,
                            cfg,
                            learnable_proto_module,
                            hierarchical_parts,
                            labels,
                            class_weights_for_loss,
                            current_epoch,
                            batch_idx,
                        )

                        # ATT435 diagnostic: verify proto_cls gradient source on first batch
                        if current_epoch == 1 and batch_idx == 0 and bool(getattr(cfg, "USE_PROTO_CLASSIFIER", False)):
                            _has_z_metric = isinstance(hierarchical_parts, dict) and torch.is_tensor(hierarchical_parts.get("z_metric"))
                            _z_metric_grad = hierarchical_parts.get("z_metric").requires_grad if _has_z_metric else False
                            _z_for_sem_grad = z_for_sem.requires_grad if z_for_sem is not None else False
                            print(f"[ATT437-ProtoClsGrad] z_metric_available={_has_z_metric} "
                                  f"z_metric_requires_grad={_z_metric_grad} "
                                  f"z_for_sem_requires_grad={_z_for_sem_grad} "
                                  f"proto_cls_via={'z_metric(decoder)' if _has_z_metric else 'z_for_sem(STOPGRAD WARNING)'}")

                        # Stage2 semantic warmup: gradually ramp semantic alignment to avoid stage shock.
                        try:
                            sem_warm = int(getattr(cfg, "STAGE2_SEMANTIC_WARMUP_EPOCHS", 0))
                        except Exception:
                            sem_warm = 0
                        if sem_warm > 0:
                            ramp_start = int(getattr(cfg, "STAGE1_SUPERVISED_EPOCHS", 8)) + 1
                            progress = float(current_epoch - int(ramp_start) + 1) / float(max(1, sem_warm))
                            progress = max(0.0, min(1.0, progress))
                            sem_ramp = 0.5 * (1.0 - math.cos(math.pi * progress))
                        else:
                            sem_ramp = 1.0
                        if sem_ramp < 1.0:
                            loss = loss * sem_ramp
                            try:
                                for _k, _v in list(loss_dict.items()):
                                    if isinstance(_v, (int, float)):
                                        loss_dict[_k] = float(_v) * sem_ramp
                                loss_dict["semantic_ramp"] = float(sem_ramp)
                            except Exception:
                                pass

                        if use_two_stage_fault_stage2 and logits_3way_stage2 is not None and logits_nf_global_stage2 is not None:
                            labels_nf = (labels > 0).long()
                            loss_nf = F.cross_entropy(logits_nf_global_stage2.float(), labels_nf, reduction="mean")
                            fault_mask = labels > 0
                            if fault_mask.any():
                                loss_3way = F.cross_entropy(
                                    logits_3way_stage2[fault_mask].float(),
                                    (labels[fault_mask] - 1).long(),
                                    reduction="mean",
                                )
                            else:
                                loss_3way = torch.tensor(0.0, device=logits_3way_stage2.device, dtype=logits_3way_stage2.dtype)
                            w_3way = float(getattr(cfg, "TWO_STAGE_FAULT_LOSS_WEIGHT", 1.0))
                            loss_diag = loss_nf + w_3way * loss_3way
                            loss = loss + loss_diag
                            try:
                                loss_dict["diag_ce"] = float(loss_diag.detach().cpu().item())
                                loss_dict["_info_fusion_cls"] = float(loss_diag.detach().cpu().item())  # ATT374: _info_ prefix
                                loss_dict["nf_ce"] = float(loss_nf.detach().cpu().item())
                                loss_dict["fault_3way_ce"] = float(loss_3way.detach().cpu().item())
                            except Exception:
                                pass

                    # ATT178: Detached SpecAlign Projection Head (safe Stage1 semantic alignment)
                    # 关键设计：z_spec.detach() -> align_proj -> CE(text_prototypes)，
                    # 只更新投影头，不影响 SpecEncoder/Fusion 主干梯度。
                    try:
                        _use_dsap = bool(getattr(cfg, "USE_DETACHED_SPEC_ALIGN_PROJ", False))
                        _dsap_w = float(getattr(cfg, "DETACHED_SPEC_ALIGN_WEIGHT", 0.0))
                        _dsap_t = float(getattr(cfg, "DETACHED_SPEC_ALIGN_TEMPERATURE", 0.1))
                        _dsap_start = int(getattr(cfg, "DETACHED_SPEC_ALIGN_START_EPOCH", 1))
                        _dsap_use_cw = bool(getattr(cfg, "DETACHED_SPEC_ALIGN_USE_CLASS_WEIGHTS", True))
                    except Exception:
                        _use_dsap, _dsap_w, _dsap_t, _dsap_start, _dsap_use_cw = False, 0.0, 0.1, 1, True
                    if _use_dsap and _dsap_w > 0.0 and current_epoch >= _dsap_start:
                        try:
                            _z_for_align = locals().get("z_spec_clean", None)
                            if _z_for_align is None:
                                _z_for_align = z_spec
                            if _z_for_align is not None and torch.is_tensor(_z_for_align):
                                _spec_align_attached = bool(getattr(cfg, "SPEC_ALIGN_ATTACHED", False))
                                _detach_input = not _spec_align_attached
                                if hasattr(spec_encoder, "project_for_proto"):
                                    _z_proj = spec_encoder.project_for_proto(_z_for_align, detach_input=_detach_input)
                                else:
                                    _z_proj = F.normalize(_z_for_align.detach() if _detach_input else _z_for_align, dim=-1)
                                # ATT428-FIX: extract proto tensor; .detach() to avoid
                                # inplace-modified-variable error when learnable_proto_module
                                # refreshes training_proto_tensor later in the same iteration.
                                _proto = None
                                if torch.is_tensor(training_proto_tensor):
                                    _proto = training_proto_tensor.detach().to(_z_proj.device, dtype=_z_proj.dtype)
                                elif isinstance(current_prototypes, dict):
                                    _rows = [current_prototypes[k].detach() for k in sorted(current_prototypes.keys()) if torch.is_tensor(current_prototypes[k])]
                                    if _rows:
                                        _proto = torch.stack(_rows, dim=0).to(_z_proj.device, dtype=_z_proj.dtype)
                                elif torch.is_tensor(current_prototypes):
                                    _proto = current_prototypes.detach().to(_z_proj.device, dtype=_z_proj.dtype)
                                if _proto is not None and _proto.dim() == 2 and _proto.size(-1) == _z_proj.size(-1):
                                    _logits_dsap = torch.matmul(_z_proj, _proto.t()) / max(_dsap_t, 1e-6)
                                    _cw = class_weights_for_loss if _dsap_use_cw else None
                                    _loss_dsap = F.cross_entropy(_logits_dsap.float(), labels, weight=_cw)
                                    loss = loss + float(_dsap_w) * _loss_dsap
                                    if isinstance(loss_dict, dict):
                                        loss_dict["detached_spec_align"] = float((float(_dsap_w) * _loss_dsap).detach().cpu().item())
                                        loss_dict["total"] = float(loss.detach().cpu().item())
                                elif batch_idx == 0 and current_epoch <= 2:
                                    print(f"[DSAP-WARN] proto extraction failed: type={type(current_prototypes).__name__}, "
                                          f"training_proto_tensor={'set' if training_proto_tensor is not None else 'None'}")
                        except Exception as _dsap_err:
                            if batch_idx == 0:
                                print(f"[DSAP-ERR] {type(_dsap_err).__name__}: {_dsap_err}")

                    # ===== (Stage aux) NF loss (keep AUC_nf stable); skip when Stage1 NF-only already applied =====
                    try:
                        import config as cfg
                        nf_w = float(getattr(cfg, "STAGE_NF_LOSS_WEIGHT", 0.10))
                    except Exception:
                        nf_w = 0.10
                    try:
                        parts_now = getattr(fusion_logits, "hierarchical_parts", None)
                    except Exception:
                        parts_now = None
                    if not isinstance(parts_now, dict):
                        try:
                            if isinstance(locals().get("hierarchical_parts", None), dict):
                                parts_now = hierarchical_parts
                        except Exception:
                            pass
                    if (not core_loss_only_mode) and (not stage1_nf_only_now) and (not bool(locals().get("skip_stage_aux_nf", False))) and nf_w > 0.0 and isinstance(parts_now, dict):
                        logits_nf_global = parts_now.get("logits_nf_global", None)
                        if logits_nf_global is not None and logits_nf_global.dim() == 2 and logits_nf_global.size(1) == 2:
                            labels_nf = (labels > 0).long()
                            pos_w = None
                            try:
                                pos_w = parts_now.get("nf_pos_weight_global", None)
                            except Exception:
                                pos_w = None
                            try:
                                pw = float(pos_w) if pos_w is not None else 1.0
                            except Exception:
                                pw = 1.0
                            w_nf = torch.tensor([1.0, max(1.0, pw)], device=logits_nf_global.device, dtype=torch.float32)
                            loss_nf = F.cross_entropy(logits_nf_global.float(), labels_nf, weight=w_nf)
                            loss = loss + float(nf_w) * loss_nf
                            loss_dict["nf_global"] = float((float(nf_w) * loss_nf).detach().cpu().item())
                            z_gate = parts_now.get("z_gate", None)
                            if z_gate is not None and getattr(cfg, "USE_NF_CONTRASTIVE_LOSS", True):
                                w_c = float(getattr(cfg, "NF_CONTRASTIVE_WEIGHT", 0.15))
                                tau = float(getattr(cfg, "NF_CONTRASTIVE_TEMPERATURE", 0.07))
                                l_c = nf_contrastive_loss(z_gate, labels_nf, temperature=tau)
                                loss = loss + w_c * l_c
                                loss_dict["nf_contrastive"] = float((w_c * l_c).detach().cpu().item())
                            if getattr(cfg, "USE_NF_MARGIN_LOSS", True) and current_epoch >= int(getattr(cfg, "NF_MARGIN_START_EPOCH", 1)):
                                w_m = float(getattr(cfg, "NF_MARGIN_WEIGHT", 0.10))
                                margin = float(getattr(cfg, "NF_MARGIN", 0.25))
                                l_m = nf_margin_loss(logits_nf_global, labels_nf, margin=margin)
                                loss = loss + w_m * l_m
                                loss_dict["nf_margin"] = float((w_m * l_m).detach().cpu().item())
                            if z_gate is not None and getattr(cfg, "USE_NORMAL_REFERENCE_LOSS", True) and hasattr(fusion_decoder, "normal_reference"):
                                ref = fusion_decoder.normal_reference
                                if ref is not None and ref.numel() == z_gate.size(-1):
                                    w_ref = float(getattr(cfg, "NORMAL_REFERENCE_WEIGHT", 0.10))
                                    l_ref = normal_reference_loss(z_gate, labels_nf, ref, weight_fault_push=1.0)
                                    loss = loss + w_ref * l_ref
                                    loss_dict["normal_ref"] = float((w_ref * l_ref).detach().cpu().item())

                    # ===== (ATT173) Spec Domain-Invariant Disentangle Losses =====
                    # L_orth: 跨协方差正交约束，迫使 z_inv / z_spe 子空间解耦
                    # L_domain_spe: dc_spe 域识别（z_spe 保留域信息）
                    # L_domain_adv: dc_inv 域对抗（z_inv 通过 GRL 迫使编码器剔除域信息）
                    try:
                        _use_dis = bool(getattr(cfg, "USE_SPEC_DISENTANGLE", False))
                        _dis_start = int(getattr(cfg, "SPEC_DISENTANGLE_START_EPOCH", 5))
                        _dis_max_alpha = float(getattr(cfg, "SPEC_DISENTANGLE_MAX_ALPHA", 0.3))
                        _dis_alpha_epochs = int(getattr(cfg, "SPEC_DISENTANGLE_ALPHA_EPOCHS", 10))
                        _w_orth = float(getattr(cfg, "SPEC_DISENTANGLE_ORTH_WEIGHT", 0.05))
                        _w_dom_spe = float(getattr(cfg, "SPEC_DISENTANGLE_DOMAIN_SPE_WEIGHT", 0.10))
                        _w_dom_adv = float(getattr(cfg, "SPEC_DISENTANGLE_DOMAIN_ADV_WEIGHT", 0.05))
                    except Exception:
                        _use_dis = False; _dis_start = 5; _dis_max_alpha = 0.3; _dis_alpha_epochs = 10
                        _w_orth = 0.05; _w_dom_spe = 0.10; _w_dom_adv = 0.05
                    _z_inv_raw = getattr(spec_encoder, "_last_z_inv_raw", None)
                    _z_spe_raw = getattr(spec_encoder, "_last_z_spe_raw", None)
                    if (
                        (not core_loss_only_mode)
                        and
                        _use_dis
                        and _z_inv_raw is not None
                        and _z_spe_raw is not None
                        and domain_ids is not None
                        and current_epoch >= _dis_start
                    ):
                        # 1. 正交损失：最小化 z_inv 与 z_spe 的跨协方差（batch 维度归一化）
                        _zi_n = F.normalize(_z_inv_raw.detach(), dim=0)   # [B, D_inv], 归一化到 batch 方向
                        _zs_n = F.normalize(_z_spe_raw.detach(), dim=0)   # [B, D_spe]
                        _cross_cov = _zi_n.T @ _zs_n                       # [D_inv, D_spe]
                        _loss_orth = _cross_cov.pow(2).sum() / (_zi_n.size(1) * _zs_n.size(1))
                        # 注：正交 loss 用 detached 特征评估协方差结构，但梯度需要通过未 detach 的版本传回
                        _zi_ng = F.normalize(_z_inv_raw, dim=0)
                        _zs_ng = F.normalize(_z_spe_raw, dim=0)
                        _loss_orth_grad = (_zi_ng.T @ _zs_ng).pow(2).sum() / (_zi_ng.size(1) * _zs_ng.size(1))

                        # 2. 域分类损失（z_spe → dc_spe）：训练 dc_spe 和编码器使 z_spe 保留域信息
                        if hasattr(spec_encoder, "dc_spe") and spec_encoder.dc_spe is not None:
                            _logits_spe = spec_encoder.dc_spe(_z_spe_raw)
                            _loss_dom_spe = F.cross_entropy(_logits_spe.float(), domain_ids.long())
                        else:
                            _loss_dom_spe = torch.tensor(0.0, device=loss.device)

                        # 3. 域对抗损失（z_inv → GRL → dc_inv）：编码器迫使 z_inv 域不可分
                        _adv_progress = float(current_epoch - _dis_start) / float(max(1, _dis_alpha_epochs))
                        _adv_progress = min(1.0, max(0.0, _adv_progress))
                        _adv_alpha = _dis_max_alpha * _adv_progress
                        if hasattr(spec_encoder, "dc_inv") and spec_encoder.dc_inv is not None and _adv_alpha > 0:
                            from models.grl import grad_reverse as _grl
                            _z_inv_rev = _grl(_z_inv_raw, _adv_alpha)
                            _logits_inv = spec_encoder.dc_inv(_z_inv_rev)
                            _loss_dom_adv = F.cross_entropy(_logits_inv.float(), domain_ids.long())
                        else:
                            _loss_dom_adv = torch.tensor(0.0, device=loss.device)

                        _loss_dis_total = _w_orth * _loss_orth_grad + _w_dom_spe * _loss_dom_spe + _w_dom_adv * _loss_dom_adv
                        loss = loss + _loss_dis_total
                        loss_dict["dis_orth"] = float((_w_orth * _loss_orth_grad).detach().cpu().item())
                        loss_dict["dis_dom_spe"] = float((_w_dom_spe * _loss_dom_spe).detach().cpu().item())
                        loss_dict["dis_dom_adv"] = float((_w_dom_adv * _loss_dom_adv).detach().cpu().item())
                        if current_epoch == _dis_start and batch_idx == 0:
                            print(f"[SpecDisentangle] ATT173 active: alpha={_adv_alpha:.3f}, "
                                  f"w_orth={_w_orth}, w_spe={_w_dom_spe}, w_adv={_w_dom_adv}")

                    # ===== (DANN) Domain adversarial loss =====
                    if isinstance(parts_now, dict):
                        try:
                            dann_lambda = float(parts_now.get("dann_lambda", 0.0))
                        except Exception:
                            dann_lambda = 0.0
                        dom_logits = parts_now.get("domain_logits", None)
                        dom_ids = parts_now.get("domain_ids", None)
                        if (not core_loss_only_mode) and dann_lambda > 0.0 and dom_logits is not None and dom_ids is not None:
                            loss_dann = F.cross_entropy(dom_logits.float(), dom_ids.long())
                            loss = loss + float(dann_lambda) * loss_dann
                            loss_dict["dann"] = float((float(dann_lambda) * loss_dann).detach().cpu().item())

                        # ===== (Rolling-DANN) Domain adversarial on rolling branch =====
                        try:
                            roll_dann_lambda = float(parts_now.get("roll_dann_lambda", 0.0))
                        except Exception:
                            roll_dann_lambda = 0.0
                        roll_dom_logits = parts_now.get("roll_domain_logits", None)
                        try:
                            roll_dann_alpha_now = float(parts_now.get("roll_dann_alpha", 0.0))
                        except Exception:
                            roll_dann_alpha_now = 0.0
                        roll_dann_expected_now = bool((not core_loss_only_mode) and roll_dann_lambda > 0.0 and roll_dann_alpha_now > 0.0)
                        if roll_dann_expected_now:
                            roll_dann_expected_batches += 1
                            if roll_dom_logits is None or dom_ids is None:
                                roll_dann_missing_batches += 1
                            else:
                                roll_mask = (labels > 0)
                                if roll_mask.any():
                                    loss_roll_dann = F.cross_entropy(
                                        roll_dom_logits[roll_mask].float(),
                                        dom_ids[roll_mask].long()
                                    )
                                    loss = loss + float(roll_dann_lambda) * loss_roll_dann
                                    loss_dict["roll_dann"] = float((float(roll_dann_lambda) * loss_roll_dann).detach().cpu().item())
                                    roll_dann_used_batches += 1
                                else:
                                    roll_dann_no_fault_batches += 1

                        # ===== (Rolling De-collapse) variance + decorrelation on rolling branch =====
                        roll_decorr_expected_now = bool(use_roll_decorr and roll_decorr_weight > 0.0 and current_epoch >= int(roll_decorr_start))
                        if roll_decorr_expected_now:
                            roll_decorr_expected_batches += 1
                            z_roll = parts_now.get("z_roll", None)
                            if z_roll is None:
                                roll_decorr_missing_batches += 1
                            else:
                                try:
                                    roll_fault_only = bool(getattr(cfg, "ROLLING_TRAIN_ON_FAULT_ONLY", True))
                                except Exception:
                                    roll_fault_only = True
                                roll_mask = (labels > 0) if roll_fault_only else torch.ones_like(labels, dtype=torch.bool)
                                if roll_mask.any():
                                    if int(roll_mask.sum().item()) >= 2:
                                        z_roll_sub = z_roll[roll_mask]
                                        l_decorr, var_mean, cov_mean = compute_decorr_loss(
                                            z_roll_sub,
                                            var_target=roll_decorr_var_target,
                                            cov_weight=roll_decorr_cov_weight
                                        )
                                        if l_decorr is not None:
                                            loss = loss + float(roll_decorr_weight) * l_decorr
                                            loss_dict["roll_decorr"] = float((float(roll_decorr_weight) * l_decorr).detach().cpu().item())
                                            roll_decorr_used_batches += 1
                                            roll_decorr_var_sum += float(var_mean)
                                            roll_decorr_cov_sum += float(cov_mean)
                                    else:
                                        roll_decorr_small_batches += 1
                                else:
                                    roll_decorr_no_fault_batches += 1

                        # ===== (Rolling De-collapse Input) variance + decorrelation on z_res =====
                        roll_decorr_in_expected_now = bool(use_roll_decorr_input and roll_decorr_input_weight > 0.0 and current_epoch >= int(roll_decorr_start))
                        if roll_decorr_in_expected_now:
                            roll_decorr_in_expected_batches += 1
                            z_res = parts_now.get("z_res", None)
                            if z_res is None:
                                roll_decorr_in_missing_batches += 1
                            else:
                                try:
                                    roll_fault_only = bool(getattr(cfg, "ROLLING_TRAIN_ON_FAULT_ONLY", True))
                                except Exception:
                                    roll_fault_only = True
                                roll_mask = (labels > 0) if roll_fault_only else torch.ones_like(labels, dtype=torch.bool)
                                if roll_mask.any():
                                    if int(roll_mask.sum().item()) >= 2:
                                        z_res_sub = z_res[roll_mask]
                                        l_decorr_in, var_in, cov_in = compute_decorr_loss(
                                            z_res_sub,
                                            var_target=roll_decorr_var_target,
                                            cov_weight=roll_decorr_cov_weight
                                        )
                                        if l_decorr_in is not None:
                                            loss = loss + float(roll_decorr_input_weight) * l_decorr_in
                                            loss_dict["roll_decorr_in"] = float((float(roll_decorr_input_weight) * l_decorr_in).detach().cpu().item())
                                            roll_decorr_in_used_batches += 1
                                            roll_decorr_in_var_sum += float(var_in)
                                            roll_decorr_in_cov_sum += float(cov_in)
                                    else:
                                        roll_decorr_in_small_batches += 1
                                else:
                                    roll_decorr_in_no_fault_batches += 1

                        # ===== (Global De-collapse) variance + decorrelation =====
                        # ATT284(P1-fix): for VICReg branch, use pre-normalization features to avoid
                        # L2-norm geometric ceiling (post-norm variance is naturally pinned around 1/D).
                        global_decorr_expected_now = bool(use_global_decorr and effective_global_decorr_weight > 0.0 and current_epoch >= int(global_decorr_start))
                        if global_decorr_expected_now:
                            global_decorr_expected_batches += 1
                            if use_global_vicreg:
                                z_global = z_fusion_for_reg
                                if z_global is None and isinstance(parts_now, dict):
                                    z_global = parts_now.get("z_cls_pre_norm", None)
                                if z_global is None:
                                    z_global = parts_now.get("z_cls", None) if isinstance(parts_now, dict) else None
                            else:
                                z_global = parts_now.get("z_cls", None) if isinstance(parts_now, dict) else None
                            if z_global is None:
                                global_decorr_missing_batches += 1
                            else:
                                if int(z_global.size(0)) >= 2:
                                    if use_global_vicreg:
                                        try:
                                            std_pre = torch.sqrt(z_global.var(dim=0, unbiased=False) + 1e-6)
                                            global_decorr_pre_std_sum += float(std_pre.mean().detach().cpu().item())
                                            global_decorr_pre_std_batches += 1
                                        except Exception:
                                            pass
                                    if use_global_vicreg:
                                        l_global, var_g, cov_g = compute_vicreg_varcov_loss(
                                            z_global,
                                            std_target=global_vicreg_std_target,
                                            var_weight=global_vicreg_var_weight,
                                            cov_weight=global_vicreg_cov_weight
                                        )
                                    else:
                                        l_global, var_g, cov_g = compute_decorr_loss(
                                            z_global,
                                            var_target=global_decorr_var_target,
                                            cov_weight=global_decorr_cov_weight
                                        )
                                    if l_global is not None:
                                        _global_weighted = float(effective_global_decorr_weight) * l_global
                                        loss = loss + _global_weighted
                                        loss_dict["global_decorr"] = float(_global_weighted.detach().cpu().item())
                                        grad_probe_terms["global_decorr_weighted"] = _global_weighted
                                        global_decorr_used_batches += 1
                                        global_decorr_var_sum += float(var_g)
                                        global_decorr_cov_sum += float(cov_g)
                                else:
                                    global_decorr_small_batches += 1

                        # ===== (Global Uniformity) spread normalized z_cls on hypersphere =====
                        global_uniformity_expected_now = bool(
                            use_global_uniformity
                            and global_uniformity_weight > 0.0
                            and current_epoch >= int(global_decorr_start)
                        )
                        if global_uniformity_expected_now:
                            global_uniformity_expected_batches += 1
                            z_cls = parts_now.get("z_cls", None)
                            if z_cls is None:
                                global_uniformity_missing_batches += 1
                            else:
                                if int(z_cls.size(0)) >= 3:
                                    l_unif, mean_d2 = compute_uniformity_loss(z_cls, t=global_uniformity_t)
                                    if l_unif is not None:
                                        loss = loss + float(global_uniformity_weight) * l_unif
                                        loss_dict["global_uniformity"] = float(
                                            (float(global_uniformity_weight) * l_unif).detach().cpu().item()
                                        )
                                        global_uniformity_used_batches += 1
                                        global_uniformity_d2_sum += float(mean_d2)
                                else:
                                    global_uniformity_small_batches += 1

                        # ===== (ATT502) Fusion representation anti-collapse =====
                        if z_fusion_for_reg is not None and torch.is_tensor(z_fusion_for_reg):
                            if use_fusion_var_floor and fusion_var_floor_weight > 0.0:
                                l_fvf, _fv = compute_fusion_var_floor_loss(
                                    z_fusion_for_reg, min_var=fusion_var_floor_min
                                )
                                if l_fvf is not None:
                                    _wfv = float(fusion_var_floor_weight) * l_fvf
                                    loss = loss + _wfv
                                    loss_dict["fusion_var_floor"] = float(_wfv.detach().cpu().item())
                            if use_fusion_vicreg and (fusion_vicreg_var_weight > 0.0 or fusion_vicreg_cov_weight > 0.0):
                                if int(z_fusion_for_reg.size(0)) >= 2:
                                    l_fvr, _, _ = compute_vicreg_varcov_loss(
                                        z_fusion_for_reg,
                                        std_target=fusion_vicreg_std_target,
                                        var_weight=fusion_vicreg_var_weight,
                                        cov_weight=fusion_vicreg_cov_weight,
                                    )
                                    if l_fvr is not None:
                                        loss = loss + l_fvr
                                        loss_dict["fusion_vicreg"] = float(l_fvr.detach().cpu().item())
                            if (
                                use_fusion_inter_cos
                                and fusion_inter_cos_weight > 0.0
                                and current_epoch >= int(fusion_inter_cos_start)
                            ):
                                _z_sep = z_fusion_for_reg
                                if isinstance(parts_now, dict):
                                    _z_metric = parts_now.get("z_metric", None)
                                    if _z_metric is not None and torch.is_tensor(_z_metric):
                                        _z_sep = extract_pre_l2_feature(_z_metric)
                                l_fic, _, _ = compute_fusion_inter_cos_margin_loss(
                                    _z_sep,
                                    labels,
                                    target_max=fusion_inter_cos_target,
                                    min_samples_per_class=fusion_inter_cos_min_samples,
                                )
                                if l_fic is not None:
                                    _wic = float(fusion_inter_cos_weight) * l_fic
                                    loss = loss + _wic
                                    loss_dict["fusion_inter_cos_margin"] = float(_wic.detach().cpu().item())

                        # ===== Fishr proxy: align gradient variance across source domains =====
                        # ATT427: configurable target feature — z_spec for better cross-domain invariance
                        fishr_expected_now = bool(
                            use_fishr_loss
                            and fishr_weight > 0.0
                            and current_epoch >= int(fishr_start_epoch)
                        )
                        if fishr_expected_now:
                            fishr_expected_batches += 1
                            _fishr_target = str(getattr(cfg, "FISHR_TARGET_FEATURE", "z_fusion")).strip().lower()
                            if _fishr_target == "z_spec" and z_spec is not None and torch.is_tensor(z_spec):
                                _fishr_repr = z_spec
                                # ATT428: use spec_logits for Fishr CE when targeting z_spec
                                # fusion_logits→z_spec gradient path is too long (cross-attention),
                                # causing grad_var≈0. spec_logits has direct gradient to z_spec.
                                _fishr_logits = spec_logits if (spec_logits is not None and torch.is_tensor(spec_logits)) else fusion_logits
                            else:
                                _fishr_repr = z_fusion_for_reg
                                _fishr_logits = fusion_logits
                            fishr_loss_now, fishr_used_domains, fishr_grad_var_now, fishr_used_split = compute_fishr_proxy_loss(
                                logits=_fishr_logits,
                                labels=labels,
                                domain_ids=domain_ids,
                                repr_tensor=_fishr_repr,
                                source_domain_ids=fishr_source_domain_ids,
                                min_samples_per_domain=fishr_min_samples_per_domain,
                                allow_intra_batch_split=True,
                            )
                            if fishr_loss_now is not None and fishr_used_domains >= 2:
                                loss = loss + float(fishr_weight) * fishr_loss_now
                                loss_dict["fishr"] = float((float(fishr_weight) * fishr_loss_now).detach().cpu().item())
                                fishr_used_batches += 1
                                fishr_grad_var_sum += float(fishr_grad_var_now)
                                if fishr_used_split:
                                    fishr_split_batches += 1
                            else:
                                fishr_skipped_batches += 1

                            # ATT428: Fishr diagnostic logging
                            _fishr_diag = bool(getattr(cfg, "FISHR_DIAG_LOG_ENABLED", False))
                            _fishr_diag_stride = int(getattr(cfg, "FISHR_DIAG_LOG_STRIDE", 20))
                            if _fishr_diag and batch_idx % _fishr_diag_stride == 0:
                                _fr = _fishr_repr
                                _fr_rg = _fr.requires_grad if torch.is_tensor(_fr) else None
                                _fr_gfn = type(_fr.grad_fn).__name__ if (torch.is_tensor(_fr) and _fr.grad_fn is not None) else "None"
                                _fr_std = float(_fr.std().item()) if torch.is_tensor(_fr) else 0.0
                                _fr_mean = float(_fr.mean().item()) if torch.is_tensor(_fr) else 0.0
                                _fr_norm = float(_fr.norm(dim=-1).mean().item()) if torch.is_tensor(_fr) else 0.0
                                _fl_val = float(fishr_loss_now.item()) if (fishr_loss_now is not None and torch.is_tensor(fishr_loss_now)) else 0.0
                                _gv = float(fishr_grad_var_now) if fishr_grad_var_now is not None else 0.0
                                _dom_counts = {}
                                if domain_ids is not None and torch.is_tensor(domain_ids):
                                    for _did in domain_ids.cpu().tolist():
                                        _dom_counts[_did] = _dom_counts.get(_did, 0) + 1
                                print(f"[FishrDiag][E{current_epoch}B{batch_idx}] "
                                      f"target={_fishr_target}, repr_rg={_fr_rg}, grad_fn={_fr_gfn}, "
                                      f"repr_std={_fr_std:.6f}, repr_mean={_fr_mean:.6f}, repr_norm={_fr_norm:.4f}, "
                                      f"raw_loss={_fl_val:.8f}, grad_var={_gv:.8f}, "
                                      f"domains_used={fishr_used_domains}, dom_counts={_dom_counts}")

                        # ===== CCPA: class-conditional prototype alignment across source domains =====
                        ccpa_expected_now = bool(
                            use_ccpa_loss
                            and ccpa_weight > 0.0
                            and current_epoch >= int(ccpa_start_epoch)
                        )
                        if ccpa_expected_now:
                            ccpa_expected_batches += 1
                            z_ccpa = None
                            if isinstance(parts_now, dict):
                                z_ccpa = parts_now.get("z_cls", None)
                            if z_ccpa is None:
                                z_ccpa = z_fusion_for_decoder
                            ccpa_loss_now, ccpa_pairs_now, ccpa_classes_now, ccpa_mean_cos_now = compute_ccpa_loss(
                                repr_tensor=z_ccpa,
                                labels=labels,
                                domain_ids=domain_ids,
                                source_domain_ids=ccpa_source_domain_ids,
                                min_samples_per_domain_class=ccpa_min_samples_per_domain_class,
                            )
                            if ccpa_loss_now is not None and ccpa_pairs_now > 0:
                                loss = loss + float(ccpa_weight) * ccpa_loss_now
                                loss_dict["ccpa"] = float((float(ccpa_weight) * ccpa_loss_now).detach().cpu().item())
                                ccpa_used_batches += 1
                                ccpa_pair_sum += int(ccpa_pairs_now)
                                ccpa_class_sum += int(ccpa_classes_now)
                                ccpa_cos_sum += float(ccpa_mean_cos_now)
                            else:
                                ccpa_skipped_batches += 1

                        # ===== Global variance logging (z_cls / z_res / z_gate) =====
                        if log_global_variance:
                            try:
                                z_cls_log = parts_now.get("z_cls", None)
                                if z_cls_log is not None and int(z_cls_log.size(0)) >= 2:
                                    v = z_cls_log.detach().float().var(dim=0, unbiased=False).mean().item()
                                    epoch_z_cls_var_sum += float(v)
                                    z_cls_var_batches += 1
                                z_res_log = parts_now.get("z_res", None)
                                if z_res_log is not None and int(z_res_log.size(0)) >= 2:
                                    v = z_res_log.detach().float().var(dim=0, unbiased=False).mean().item()
                                    epoch_z_res_var_sum += float(v)
                                    z_res_var_batches += 1
                                z_gate_log = parts_now.get("z_gate", None)
                                if z_gate_log is not None and int(z_gate_log.size(0)) >= 2:
                                    v = z_gate_log.detach().float().var(dim=0, unbiased=False).mean().item()
                                    epoch_z_gate_var_sum += float(v)
                                    z_gate_var_batches += 1
                            except Exception:
                                pass

                    # (3b) EMA update normal reference
                    if (not core_loss_only_mode) and getattr(cfg, "USE_NORMAL_REFERENCE_LOSS", True) and hasattr(fusion_decoder, "update_normal_reference"):
                        _hp = getattr(fusion_logits, "hierarchical_parts", None)
                        _zg = _hp.get("z_gate", None) if isinstance(_hp, dict) else None
                        if _zg is not None:
                            with torch.no_grad():
                                mom = getattr(cfg, "NORMAL_REFERENCE_MOMENTUM", 0.99)
                                if current_epoch == 1:
                                    first_epoch_mom = getattr(cfg, "NORMAL_REFERENCE_FIRST_EPOCH_MOMENTUM", None)
                                    if first_epoch_mom is not None:
                                        mom = float(first_epoch_mom)
                                fusion_decoder.update_normal_reference(_zg, labels, momentum=mom)

                    # 添加signal/spec aux CE损失（辅助监督）
                    import config as cfg
                    aux_ce_weight = getattr(cfg, 'AUX_CE_WEIGHT', 0.01)
                    aux_ce_start_epoch = getattr(cfg, 'AUX_CE_START_EPOCH', 1)
                    _spec_obj_mode = str(getattr(cfg, "SPEC_TRAIN_OBJECTIVE_MODE", "legacy_aux_ce")).strip().lower()
                    _spec_use_legacy_ce = _spec_obj_mode in ("legacy_aux_ce", "hybrid")
                    _spec_use_proto_ce = _spec_obj_mode in ("proto_ce", "hybrid")
                    _spec_proto_ce_w = float(getattr(cfg, "SPEC_PROTO_CE_WEIGHT", 1.0))
                    _spec_proto_ce_t = float(getattr(cfg, "SPEC_PROTO_CE_TEMPERATURE", 0.10))
                    _spec_proto_ce_t = max(_spec_proto_ce_t, 1e-4)

                    if aux_ce_weight > 0 and current_epoch >= aux_ce_start_epoch:
                        aux_loss = 0.0
                        aux_weight_sum = 0.0

                        # ATT124: read focal config once per iteration (cheap getattr)
                        _aux_use_focal = getattr(cfg, 'AUX_CE_USE_FOCAL', False)
                        _aux_focal_gamma = float(getattr(cfg, 'AUX_CE_FOCAL_GAMMA', 2.0))

                        # Signal aux CE (standard CE; focal is spec-head-specific)
                        if (not core_loss_only_mode) and use_signal_classification_head and signal_logits is not None:
                            aux_signal_loss = F.cross_entropy(signal_logits, labels, reduction='mean')
                            aux_loss += aux_signal_loss
                            aux_weight_sum += 1.0
                            loss_dict['aux_signal'] = aux_signal_loss.item()

                        # ATT378-FIX: block aux_spec when not in CORE_LOSS_ALLOWED_TERMS
                        # ATT379c: also disable aux_spec during decoupled Stage 1 (contrastive-only)
                        _aux_spec_allowed = (not core_loss_only_mode) or ("aux_spec" in set(getattr(cfg, "CORE_LOSS_ALLOWED_TERMS", ())))
                        if _decoupled_is_stage1:
                            _aux_spec_allowed = False
                        if _spec_use_legacy_ce and _aux_spec_allowed and use_spec_classification_head and spec_logits is not None:
                            if _aux_use_focal:
                                _log_probs = F.log_softmax(spec_logits, dim=1)
                                _ce_per = F.nll_loss(_log_probs, labels, reduction='none')  # [B]
                                _pt = torch.exp(-_ce_per)                                   # p_t ∈ (0,1]
                                _focal_w = (1.0 - _pt) ** _aux_focal_gamma                  # (1-p_t)^γ
                                aux_spec_loss = (_focal_w * _ce_per).mean()
                            else:
                                aux_spec_loss = F.cross_entropy(spec_logits, labels, reduction='mean',
                                                                label_smoothing=0.05)
                            aux_loss += aux_spec_loss
                            aux_weight_sum += 1.0
                            loss_dict['aux_spec'] = aux_spec_loss.item()

                        # ATT489(P1): prototype-consistent spec objective in z_spec space.
                        if _spec_use_proto_ce and _aux_spec_allowed and (z_spec_clean is not None) and (training_proto_tensor is not None):
                            try:
                                # Use align_proj branch (or clone) to avoid sharing the same
                                # normalized z_spec tensor with multimodal_alignment_loss / clip_style.
                                if hasattr(spec_encoder, "project_for_proto"):
                                    _spec_z = spec_encoder.project_for_proto(
                                        z_spec_clean, detach_input=False
                                    )
                                else:
                                    _spec_z = F.normalize(z_spec_clean.float().clone(), dim=-1)
                                # Detach + refresh protos to avoid inplace conflict when
                                # learnable_proto_module.forward() is called again later (ATT428).
                                _proto_src = training_proto_tensor
                                if learnable_proto_module is not None:
                                    _proto_src = learnable_proto_module.forward().detach()
                                elif torch.is_tensor(_proto_src):
                                    _proto_src = _proto_src.detach()
                                _proto_mat = F.normalize(_proto_src.to(_spec_z.device).float(), dim=-1)
                                _proto_logits = torch.matmul(_spec_z, _proto_mat.t()) / _spec_proto_ce_t
                                _spec_proto_ce = F.cross_entropy(_proto_logits, labels, reduction='mean')
                                _w_proto = max(0.0, _spec_proto_ce_w)
                                if _w_proto > 0:
                                    aux_loss += _w_proto * _spec_proto_ce
                                    aux_weight_sum += _w_proto
                                    loss_dict['spec_proto_ce'] = float(_spec_proto_ce.detach().cpu().item())
                            except Exception as _e_proto_ce:
                                if batch_idx == 0 and (current_epoch <= 3 or current_epoch % 10 == 0):
                                    print(f"[ATT489-SpecProtoCE] skip due to {type(_e_proto_ce).__name__}: {_e_proto_ce}")

                        # ATT439: Hierarchical SpecHead loss (NF + Fault subtype)
                        _hier_primary = bool(getattr(cfg, "HIERARCHICAL_SPEC_HEAD_PRIMARY", False))
                        if _hier_primary and _aux_spec_allowed and hasattr(spec_encoder, '_last_nf_logits'):
                            _nf_logits_h = getattr(spec_encoder, '_last_nf_logits', None)
                            _fault_logits_h = getattr(spec_encoder, '_last_fault_logits', None)
                            if _nf_logits_h is not None and labels is not None:
                                try:
                                    _nf_labels = (labels > 0).long()
                                    _nf_ce = F.cross_entropy(_nf_logits_h, _nf_labels, reduction='mean')
                                    aux_loss += _nf_ce
                                    aux_weight_sum += 1.0
                                    loss_dict['spec_nf_ce'] = float(_nf_ce.detach().cpu().item())
                                except Exception:
                                    pass
                            if _fault_logits_h is not None and labels is not None:
                                try:
                                    _fault_mask = (labels > 0)
                                    if _fault_mask.any():
                                        _fl = _fault_logits_h[_fault_mask]
                                        _ft_labels = (labels[_fault_mask] - 1).clamp(min=0)
                                        _n_ft_cls = _fl.size(-1)
                                        _ft_labels = _ft_labels.clamp(max=_n_ft_cls - 1)
                                        _gamma_ft = 2.0
                                        _log_p_ft = F.log_softmax(_fl, dim=1)
                                        _ce_ft = F.nll_loss(_log_p_ft, _ft_labels, reduction='none')
                                        _pt_ft = torch.exp(-_ce_ft)
                                        _focal_ft = ((1.0 - _pt_ft) ** _gamma_ft * _ce_ft).mean()
                                        aux_loss += _focal_ft
                                        aux_weight_sum += 1.0
                                        loss_dict['spec_fault_ce'] = float(_focal_ft.detach().cpu().item())
                                        if batch_idx == 0 and (current_epoch <= 3 or current_epoch % 5 == 0):
                                            with torch.no_grad():
                                                _ft_pred = _fl.argmax(dim=-1)
                                                _ft_acc = (_ft_pred == _ft_labels).float().mean().item()
                                                _per_cls = []
                                                for _ci in range(_n_ft_cls):
                                                    _cm = (_ft_labels == _ci)
                                                    if _cm.any():
                                                        _per_cls.append(f"c{_ci}R={(_ft_pred[_cm] == _ci).float().mean().item():.2f}")
                                                    else:
                                                        _per_cls.append(f"c{_ci}R=N/A")
                                            print(f"[ATT439-HierSpec][E{current_epoch}B{batch_idx}] "
                                                  f"fault_focal={float(_focal_ft.detach().cpu().item()):.4f} "
                                                  f"fault_acc={_ft_acc:.4f} {' '.join(_per_cls)}")
                                except Exception as _hier_e:
                                    if batch_idx == 0:
                                        print(f"[ATT439-HierSpec] fault_ce error: {_hier_e}")

                        if aux_weight_sum > 0:
                            aux_loss = aux_loss / max(aux_weight_sum, 1e-12)  # weighted mean
                            loss = loss + aux_ce_weight * aux_loss
                            loss_dict['aux_ce'] = aux_ce_weight * aux_loss.item()

                        # ATT431: Hierarchical NF/Fault CE, Dual Disentangle, Spec Variance
                        # all removed (falsified in ATT430: F_DD, F_SV, USE_HIERARCHICAL_SPEC_HEAD=0)

                    # ATT431: Per-batch diagnostic (simplified)
                    if batch_idx % 50 == 0:
                        _diag_parts = []
                        for _dk in ("aux_ce",):
                            if _dk in loss_dict:
                                _diag_parts.append(f"{_dk}={loss_dict[_dk]:.6f}")
                        if _diag_parts:
                            _z_pre = getattr(spec_encoder, '_last_z_spec_pre_norm', None)
                            _z_spe = getattr(spec_encoder, '_last_z_spe_dual', None)
                            _diag_extra = ""
                            if _z_pre is not None:
                                _diag_extra += f" z_inv_std={float(_z_pre.detach().std()):.4f}"
                            if _z_spe is not None:
                                _diag_extra += f" z_spe_std={float(_z_spe.detach().std()):.4f}"
                            print(f"[ATT430-Diag][E{current_epoch}B{batch_idx}] "
                                  f"{', '.join(_diag_parts)}{_diag_extra}")

                    # 添加MoE损失
                    if (not core_loss_only_mode) and use_moe and moe_stats is not None:
                        num_experts_per_band = getattr(spec_encoder, 'num_experts_per_band', 4)
                        num_bands = getattr(spec_encoder, 'num_bands', 3)
                        expert_top_k = getattr(spec_encoder, 'expert_top_k', num_experts_per_band)
                        
                        # Check if using Bayesian router or disagreement
                        use_bayesian = getattr(spec_encoder, 'use_bayesian_router', False)
                        use_disagreement = getattr(spec_encoder, 'use_expert_disagreement', False)
                        
                        if use_bayesian or use_disagreement:
                            # Use extended loss function with Bayesian and disagreement terms
                            import config as cfg
                            moe_loss, moe_loss_dict = compute_moe_loss_bayesian(
                                moe_stats,
                                num_experts_per_band,
                                num_bands,
                                w_load_balance=3.0,
                                w_entropy=0.01,
                                w_diversity=0.2,
                                w_router_reg=0.05,
                                use_v2_losses=True,
                                diversity_margin=0.3,
                                expert_top_k=expert_top_k,
                                w_sparse_balance=1.0,
                                # Bayesian/disagreement specific
                                w_kl_div=cfg.BAYESIAN_ROUTER_KL_WEIGHT,
                                w_disagreement_reg=cfg.EXPERT_DISAGREEMENT_REG_WEIGHT,
                                disagreement_target=cfg.EXPERT_DISAGREEMENT_TARGET,
                                disagreement_margin=cfg.EXPERT_DISAGREEMENT_MARGIN
                            )
                        else:
                            # Use standard MoE loss
                            moe_loss, moe_loss_dict = compute_moe_loss(
                                moe_stats, 
                                num_experts_per_band, 
                                num_bands,
                                w_load_balance=3.0,
                                w_entropy=0.01,
                                w_diversity=0.2,
                                w_router_reg=0.05,
                                use_v2_losses=True,
                                diversity_margin=0.3,
                                expert_top_k=expert_top_k,
                                w_sparse_balance=1.0
                            )
                        
                        loss = loss + moe_loss_weight * moe_loss
                        loss_dict.update(moe_loss_dict)
                        _bs_w = float(getattr(Config, "SPEC_MOE_BAND_SPARSITY_WEIGHT", 0.0))
                        if _bs_w > 0 and isinstance(moe_stats, dict):
                            _bst = moe_stats.get("band_sparsity_loss")
                            if torch.is_tensor(_bst) and _bst.requires_grad:
                                _bw = _bs_w * _bst
                                loss = loss + _bw
                                loss_dict["moe_band_sparsity"] = float(_bw.detach().cpu().item())
                        log_moe_router_audit(Config, current_epoch, batch_idx, moe_stats)
                    
                    # 添加类感知权重正则化
                    if (not core_loss_only_mode) and fusion_class_aware_regularization > 0 and hasattr(fusion_module, 'class_modal_weights'):
                        class_aware_reg = fusion_class_aware_regularization * torch.norm(fusion_module.class_modal_weights)
                        loss = loss + class_aware_reg
                        loss_dict['class_aware_reg'] = class_aware_reg.item()

                    # 添加注意力门控熵正则化（防坍缩）
                    # 关键：使用可反传的 reg_loss，而不是 detach 的版本
                    if (not core_loss_only_mode) and hasattr(fusion_module, 'get_reg_loss'):
                        reg_loss = fusion_module.get_reg_loss()
                        if reg_loss is not None:
                            # 使用配置中的权重，默认 0.01
                            import config as cfg
                            attention_reg_weight = getattr(cfg, 'ATTENTION_REG_WEIGHT', 0.01)
                            loss = loss + attention_reg_weight * reg_loss
                            loss_dict['attention_reg'] = attention_reg_weight * reg_loss.item()
                    
                    # 添加正交性/互补性正则化（Barlow-Twins风格）
                    if (not core_loss_only_mode) and use_orthogonality_reg and orthogonality_reg_weight > 0:
                        if hasattr(fusion_module, 'compute_orthogonality_loss'):
                            # IMPORTANT (stability): do NOT use cached tensors that may reference a previous graph.
                            # Use current-batch embeddings directly to avoid "backward through the graph a second time".
                            orthogonality_loss = fusion_module.compute_orthogonality_loss(z_signal, z_spec)
                            loss = loss + orthogonality_reg_weight * orthogonality_loss
                            loss_dict['orthogonality_reg'] = orthogonality_reg_weight * orthogonality_loss.item()

                    if not supcon_stage1_now:
                        # ===== metric learning (EMA center loss) on diagnosis branch z_cls =====
                        if FUSION_METRIC_LOSS_WEIGHT > 0 and FUSION_METRIC_LOSS == "center":
                            hp = getattr(fusion_logits, "hierarchical_parts", None)
                            if hp is not None and "z_cls" in hp:
                                z_cls = hp["z_cls"]
                                num_classes = int(getattr(fusion_decoder, "num_classes", 4))
                                if ema_centers is None or ema_centers.size(0) != num_classes or ema_centers.size(1) != z_cls.size(1):
                                    ema_centers = torch.zeros(num_classes, z_cls.size(1), device=z_cls.device, dtype=z_cls.dtype)
                                    ema_centers = F.normalize(ema_centers + 1e-3, dim=-1)

                                center_loss_val = cosine_center_loss(
                                    z_cls, labels, ema_centers,
                                    class_weights=class_weights_for_loss if class_weights_for_loss is not None else None
                                )
                                loss = loss + FUSION_METRIC_LOSS_WEIGHT * center_loss_val
                                loss_dict["metric_center"] = float(center_loss_val.detach().cpu().item())

                                ema_centers = update_ema_centers(
                                    ema_centers, z_cls.detach(), labels.detach(),
                                    momentum=float(FUSION_CENTER_MOMENTUM)
                                )

                        if (not core_loss_only_mode) and clip_model is not None and float(lambda_clip) > 0.0:
                            clip_emb = clip_model.encode_spec(spec)
                            clip_loss = spec_clip_alignment_loss(z_spec, clip_emb)
                            loss = loss + lambda_clip * clip_loss
                        # CLIP-style symmetric contrastive alignment (visual-text).
                        try:
                            _clip_style_active = bool(getattr(cfg, "USE_CLIP_STYLE_CONTRASTIVE", False))
                            _clip_style_w = float(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_WEIGHT", 0.0))
                            _clip_style_t = float(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_TEMPERATURE", 0.07))
                            _clip_style_start = int(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_START_EPOCH", 1))
                        except Exception:
                            _clip_style_active, _clip_style_w, _clip_style_t, _clip_style_start = False, 0.0, 0.07, 1
                        if _clip_style_active and _clip_style_w > 0.0 and current_epoch >= _clip_style_start:
                            try:
                                _txt_batch = text_encoder(list(texts))
                                _clip_out = clip_style_batch_contrastive_loss(
                                    z_spec_clean,
                                    _txt_batch,
                                    temperature=_clip_style_t,
                                    labels=labels,
                                )
                                _clip_tensor = _clip_out.get("loss", None)
                                if torch.is_tensor(_clip_tensor) and _clip_tensor.requires_grad:
                                    loss = loss + float(_clip_style_w) * _clip_tensor
                                    loss_dict["clip_style"] = float((float(_clip_style_w) * _clip_tensor).detach().cpu().item())
                                    loss_dict["_info_clip_style_diag_cos"] = float(_clip_out.get("diag_cos_mean", 0.0))
                                    loss_dict["_info_clip_style_offdiag_cos"] = float(_clip_out.get("offdiag_cos_mean", 0.0))
                                    epoch_clip_style += float(loss_dict["clip_style"])
                                    epoch_clip_style_diag_cos += float(_clip_out.get("diag_cos_mean", 0.0))
                                    epoch_clip_style_offdiag_cos += float(_clip_out.get("offdiag_cos_mean", 0.0))
                                    epoch_clip_style_batches += 1
                            except Exception:
                                pass

                        # ===== ATT379(S2): Dual CLIP on z_fusion for direct FusionProto alignment =====
                        try:
                            _dual_clip_active = bool(getattr(cfg, "USE_DUAL_CLIP_SPEC", False))
                            _dual_clip_w = float(getattr(cfg, "DUAL_CLIP_SPEC_WEIGHT", 0.0))
                        except Exception:
                            _dual_clip_active, _dual_clip_w = False, 0.0
                        if (_dual_clip_active and _dual_clip_w > 0.0
                                and _clip_style_active and current_epoch >= _clip_style_start):
                            try:
                                _z_fus_for_clip = z_fusion_for_decoder
                                if _z_fus_for_clip is None:
                                    _z_fus_for_clip = z_fusion
                                _z_fus_norm = F.normalize(_z_fus_for_clip.float(), dim=-1)
                                if "_txt_batch" not in dir():
                                    _txt_batch = text_encoder(list(texts))
                                _dc_out = clip_style_batch_contrastive_loss(
                                    _z_fus_norm,
                                    _txt_batch,
                                    temperature=_clip_style_t,
                                    labels=labels,
                                )
                                _dc_tensor = _dc_out.get("loss", None)
                                if torch.is_tensor(_dc_tensor) and _dc_tensor.requires_grad:
                                    loss = loss + float(_dual_clip_w) * _dc_tensor
                                    loss_dict["dual_clip_fus"] = float((float(_dual_clip_w) * _dc_tensor).detach().cpu().item())
                                    loss_dict["_info_dual_clip_fus_diag_cos"] = float(_dc_out.get("diag_cos_mean", 0.0))
                            except Exception:
                                pass

                        # 添加校准损失（ECE）
                        if (not core_loss_only_mode) and use_calibration_loss and calibration_loss_weight > 0:
                            # 使用fusion_logits计算ECE
                            if hasattr(fusion_logits, 'hierarchical_parts'):
                                # 对于层次化模型，使用4类logits
                                logits_for_ece = fusion_logits
                            else:
                                logits_for_ece = fusion_logits
                            
                            ece_loss = compute_ece_loss(logits_for_ece, labels, n_bins=calibration_n_bins)
                            loss = loss + calibration_loss_weight * ece_loss
                            loss_dict['ece'] = ece_loss.item()
                        
                        # ===== Prototype Separation Loss =====
                        if use_proto_sep_loss and current_epoch >= proto_sep_start_epoch:
                            # Get z_cls features for prototype update (from hierarchical_parts or fusion_logits)
                            z_cls_for_proto = None
                            try:
                                hp = getattr(fusion_logits, "hierarchical_parts", None)
                                if isinstance(hp, dict):
                                    z_cls_for_proto = hp.get("z_cls", None)
                                if z_cls_for_proto is None:
                                    z_cls_for_proto = z_fusion  # Fallback to fusion embedding
                            except Exception:
                                z_cls_for_proto = z_fusion
                            
                            if z_cls_for_proto is not None:
                                # Initialize proto_sep_loss_fn if not done yet
                                if proto_sep_loss_fn is None:
                                    feature_dim = z_cls_for_proto.size(-1)
                                    proto_sep_loss_fn = PrototypeSeparationLoss(
                                        num_classes=4,
                                        feature_dim=feature_dim,
                                        delta=proto_sep_margin_delta,
                                        momentum=proto_sep_momentum,
                                        use_log_ratio=use_log_proto_sep,
                                        log_eps=log_proto_sep_eps
                                    ).to(device)
                                    print(f"[Proto-Sep Loss] Initialized with feature_dim={feature_dim}")
                                
                                # Compute prototype separation loss
                                proto_sep_loss_val, proto_dists = proto_sep_loss_fn(z_cls_for_proto, labels)
                                loss = loss + proto_sep_weight * proto_sep_loss_val
                                loss_dict["proto_sep"] = float(proto_sep_loss_val.detach().cpu().item())
                                epoch_proto_sep_loss += float(proto_sep_loss_val.detach().cpu().item())
                                proto_sep_batches += 1
                                # 类内方差统计（L2，batch-level）
                                try:
                                    with torch.no_grad():
                                        for c in range(4):
                                            m = (labels == c)
                                            if m.any():
                                                feats_c = z_cls_for_proto[m]
                                                mu_c = feats_c.mean(dim=0, keepdim=True)
                                                var_c = (feats_c - mu_c).pow(2).sum(dim=1).mean()
                                                class_var_sum[c] += float(var_c.detach().cpu().item())
                                                class_var_count[c] += 1
                                except Exception:
                                    pass

                # Attempt 64.1: semantic anchor must be injected on Stage1 path
                # (USE_TWO_STAGE_TRAINING=False => multimodal_alignment_loss branch is not executed).
                try:
                    import config as _cfg_sem
                    use_sem_fvn = bool(getattr(_cfg_sem, "USE_SEMANTIC_FAULT_VS_NORMAL_ANCHOR", False))
                    sem_fvn_weight = float(getattr(_cfg_sem, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_WEIGHT", 0.0))
                    sem_fvn_margin = float(getattr(_cfg_sem, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_MARGIN", 0.15))
                    sem_fvn_start_epoch = int(getattr(_cfg_sem, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_START_EPOCH", 1))
                except Exception:
                    use_sem_fvn, sem_fvn_weight, sem_fvn_margin, sem_fvn_start_epoch = False, 0.0, 0.15, 1
                if (
                    (not core_loss_only_mode)
                    and
                    bool(locals().get("stage1_mode_now", False))
                    and use_sem_fvn
                    and (sem_fvn_weight > 0.0)
                    and (current_epoch >= int(sem_fvn_start_epoch))
                    and isinstance(current_prototypes, dict)
                    and (0 in current_prototypes)
                ):
                    try:
                        z_anchor = F.normalize(z_fusion.float(), dim=-1, eps=1e-6)
                        p_normal = F.normalize(current_prototypes[0].to(z_anchor.device).float(), dim=-1, eps=1e-6)
                        per_cls_losses = []
                        for _c in (1, 2, 3):
                            if _c not in current_prototypes:
                                continue
                            _m = (labels == int(_c))
                            if not bool(_m.any()):
                                continue
                            p_c = F.normalize(current_prototypes[_c].to(z_anchor.device).float(), dim=-1, eps=1e-6)
                            z_c = z_anchor[_m]
                            sim_pos = torch.matmul(z_c, p_c)
                            sim_neg_n = torch.matmul(z_c, p_normal)
                            per_cls_losses.append(F.relu(float(sem_fvn_margin) - (sim_pos - sim_neg_n)).mean())
                        if len(per_cls_losses) > 0:
                            loss_sem_fvn = torch.stack(per_cls_losses).mean()
                            loss = loss + float(sem_fvn_weight) * loss_sem_fvn
                            if isinstance(loss_dict, dict):
                                loss_dict["sem_fault_vs_normal_anchor"] = float(loss_sem_fvn.detach().cpu().item())
                                loss_dict["total"] = float(loss.detach().cpu().item())
                            epoch_sem_fvn_loss += float(loss_sem_fvn.detach().cpu().item())
                            sem_fvn_batches += 1
                    except Exception:
                        pass

                # Guard: avoid backward on no-grad loss
                if (
                    bool(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_ENABLE", False))
                    and _is_task2_phase
                    and int(current_epoch) >= int(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_START_EPOCH", 1))
                    and isinstance(hierarchical_parts, dict)
                    and torch.is_tensor(domain_ids)
                ):
                    try:
                        _zm = hierarchical_parts.get("z_metric", None)
                        _cd_loss = cross_domain_metric_centroid_loss(
                            _zm,
                            domain_ids=domain_ids,
                            labels=labels,
                            sep_margin=float(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_SEP_MARGIN", 0.10)),
                        )
                        if torch.is_tensor(_cd_loss):
                            _cd_w = float(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_WEIGHT", 0.10))
                            if _cd_w > 0.0:
                                loss = loss + (_cd_w * _cd_loss)
                                if isinstance(loss_dict, dict):
                                    loss_dict["cross_domain_metric_align"] = float(_cd_loss.detach().cpu().item())
                                    loss_dict["cross_domain_metric_align_weight"] = float(_cd_w)
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                    except Exception:
                        pass

                # Guard: avoid backward on no-grad loss
                if (
                    bool(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_ENABLE", False))
                    and _is_task2_phase
                    and int(current_epoch) >= int(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_START_EPOCH", 1))
                    and isinstance(hierarchical_parts, dict)
                    and torch.is_tensor(domain_ids)
                ):
                    try:
                        _zm = hierarchical_parts.get("z_metric", None)
                        _cd_loss = cross_domain_metric_centroid_loss(
                            _zm,
                            domain_ids=domain_ids,
                            labels=labels,
                            sep_margin=float(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_SEP_MARGIN", 0.10)),
                        )
                        if torch.is_tensor(_cd_loss):
                            _cd_w = float(getattr(cfg, "CROSS_DOMAIN_METRIC_ALIGN_WEIGHT", 0.10))
                            if _cd_w > 0.0:
                                loss = loss + (_cd_w * _cd_loss)
                                if isinstance(loss_dict, dict):
                                    loss_dict["cross_domain_metric_align"] = float(_cd_loss.detach().cpu().item())
                                    loss_dict["cross_domain_metric_align_weight"] = float(_cd_w)
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                    except Exception:
                        pass

                # Guard: avoid backward on no-grad loss
                if (
                    task2_coral_enable
                    and float(task2_coral_weight) > 0.0
                    and int(current_epoch) >= int(task2_coral_start_epoch)
                    and torch.is_tensor(z_spec)
                    and torch.is_tensor(domain_ids)
                ):
                    try:
                        mask_cwru = (domain_ids == 0) | (domain_ids == 1)
                        mask_pu = (domain_ids == 2)
                        if bool(mask_cwru.any()) and bool(mask_pu.any()):
                            if int(mask_cwru.sum().item()) >= int(task2_coral_min_samples) and int(mask_pu.sum().item()) >= int(task2_coral_min_samples):
                                loss_coral = coral_covariance_alignment_loss(z_spec[mask_cwru], z_spec[mask_pu])
                                coral_warmup_ratio = min(
                                    1.0,
                                    max(0.0, float(int(current_epoch) - int(task2_coral_start_epoch) + 1))
                                    / float(max(1, int(task2_coral_warmup_epochs))),
                                )
                                effective_coral_weight = float(task2_coral_weight) * coral_warmup_ratio
                                loss = loss + effective_coral_weight * loss_coral
                                if isinstance(loss_dict, dict):
                                    loss_dict["coral_spec"] = float(loss_coral.detach().cpu().item())
                                    loss_dict["coral_spec_weight"] = float(effective_coral_weight)
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                                epoch_coral_loss += float(loss_coral.detach().cpu().item())
                                coral_batches += 1
                                if int(current_epoch) == int(task2_coral_start_epoch) and int(batch_idx) == 0:
                                    print(
                                        f"[Task2 CORAL] active: n_cwru={int(mask_cwru.sum().item())}, "
                                        f"n_pu={int(mask_pu.sum().item())}, "
                                        f"w_base={float(task2_coral_weight):.4f}, w_eff={float(effective_coral_weight):.4f}"
                                    )
                                # ATT197(B): source-to-target CORAL with unlabeled HUST/HIT reference batches.
                                if (
                                    task2_target_coral_enable
                                    and float(task2_target_coral_weight) > 0.0
                                    and len(target_coral_refs) > 0
                                    and (int(batch_idx) % max(1, int(task2_target_coral_every_n_steps)) == 0)
                                ):
                                    src_mask = mask_cwru | mask_pu
                                    if bool(src_mask.any()) and int(src_mask.sum().item()) >= int(task2_target_coral_min_samples):
                                        src_feat = z_spec[src_mask]
                                        loss_target_coral = None
                                        used_refs = 0
                                        for _, _ref_feat in target_coral_refs:
                                            _lt = coral_covariance_alignment_loss(src_feat, _ref_feat)
                                            if torch.is_tensor(_lt):
                                                loss_target_coral = _lt if loss_target_coral is None else (loss_target_coral + _lt)
                                                used_refs += 1
                                        if loss_target_coral is not None and used_refs > 0:
                                            loss_target_coral = loss_target_coral / float(max(1, used_refs))
                                            effective_target_coral_weight = float(task2_target_coral_weight) * coral_warmup_ratio
                                            loss = loss + effective_target_coral_weight * loss_target_coral
                                            if isinstance(loss_dict, dict):
                                                loss_dict["coral_target_spec"] = float(loss_target_coral.detach().cpu().item())
                                                loss_dict["coral_target_spec_weight"] = float(effective_target_coral_weight)
                                                loss_dict["total"] = float(loss.detach().cpu().item())
                                            epoch_target_coral_loss += float(loss_target_coral.detach().cpu().item())
                                            target_coral_batches += 1
                        # ATT221(S1): source->target MK-MMD with unlabeled HUST/HIT online batches.
                        if (
                            task2_target_mmd_enable
                            and _is_task2_phase
                            and float(task2_target_mmd_weight) > 0.0
                            and int(current_epoch) >= int(task2_target_mmd_start_epoch)
                            and len(target_mmd_loaders) > 0
                            and (int(batch_idx) % max(1, int(task2_target_mmd_every_n_steps)) == 0)
                        ):
                            src_mask = mask_cwru | mask_pu
                            src_count = int(src_mask.sum().item())
                            if bool(src_mask.any()) and src_count >= int(task2_target_mmd_min_samples):
                                pick_idx = int(batch_idx) % int(len(target_mmd_loaders))
                                tgt_name, tgt_loader = target_mmd_loaders[pick_idx]
                                tgt_iter = target_mmd_iters.get(tgt_name, None)
                                if tgt_iter is None:
                                    tgt_iter = iter(tgt_loader)
                                try:
                                    tgt_batch = next(tgt_iter)
                                except StopIteration:
                                    tgt_iter = iter(tgt_loader)
                                    tgt_batch = next(tgt_iter)
                                except Exception:
                                    tgt_batch = None
                                target_mmd_iters[tgt_name] = tgt_iter
                                if tgt_batch is not None:
                                    _enc_mode = bool(spec_encoder.training)
                                    _txt_mode = bool(text_encoder.training)
                                    try:
                                        spec_encoder.eval()
                                        text_encoder.eval()
                                        with torch.no_grad():
                                            z_tgt = _extract_spec_feat_from_batch(tgt_batch)
                                    finally:
                                        if _enc_mode:
                                            spec_encoder.train()
                                        if _txt_mode:
                                            text_encoder.train()
                                    if torch.is_tensor(z_tgt) and z_tgt.dim() == 2 and int(z_tgt.size(0)) >= int(task2_target_mmd_min_samples):
                                        mmd_loss = multi_kernel_mmd_loss(
                                            z_spec[src_mask],
                                            z_tgt.detach(),
                                            kernel_scales=task2_target_mmd_kernel_scales,
                                        )
                                        mmd_warmup_ratio = min(
                                            1.0,
                                            max(0.0, float(int(current_epoch) - int(task2_target_mmd_start_epoch) + 1))
                                            / float(max(1, int(task2_target_mmd_warmup_epochs))),
                                        )
                                        effective_mmd_weight = float(task2_target_mmd_weight) * mmd_warmup_ratio
                                        loss = loss + effective_mmd_weight * mmd_loss
                                        if isinstance(loss_dict, dict):
                                            loss_dict["mmd_target_spec"] = float(mmd_loss.detach().cpu().item())
                                            loss_dict["mmd_target_spec_weight"] = float(effective_mmd_weight)
                                            loss_dict["total"] = float(loss.detach().cpu().item())
                                        epoch_target_mmd_loss += float(mmd_loss.detach().cpu().item())
                                        target_mmd_batches += 1
                                        if int(current_epoch) == int(task2_target_mmd_start_epoch) and int(batch_idx) == 0:
                                            print(
                                                f"[Task2 Target-MMD] active: src_n={src_count}, tgt={tgt_name}, "
                                                f"tgt_n={int(z_tgt.size(0))}, w_base={float(task2_target_mmd_weight):.4f}, "
                                                f"w_eff={float(effective_mmd_weight):.4f}"
                                            )
                    except Exception:
                        pass

                # ===== ATT392-S1: Refresh training_proto_tensor from learnable protos =====
                if learnable_proto_module is not None:
                    training_proto_tensor = learnable_proto_module.forward()  # [C, D] with grad

                # ===== ATT374/ATT390-S2: Prototype-Anchored Contrastive Loss (PACL) =====
                if training_proto_tensor is not None:
                    _pacl_active = bool(getattr(cfg, "USE_PROTOTYPE_ANCHOR_LOSS", False))
                    _pacl_start = int(getattr(cfg, "PROTOTYPE_ANCHOR_START_EPOCH", 1))
                    if _pacl_active and current_epoch >= _pacl_start:
                        _pacl_temp = float(getattr(cfg, "PROTOTYPE_ANCHOR_TEMPERATURE", 0.07))
                        _pacl_ls = float(getattr(cfg, "PROTOTYPE_ANCHOR_LABEL_SMOOTHING", 0.05))
                        _pacl_w_spec = float(getattr(cfg, "PROTOTYPE_ANCHOR_WEIGHT_SPEC", 0.5))
                        _pacl_w_fus = float(getattr(cfg, "PROTOTYPE_ANCHOR_WEIGHT_FUSION", 0.5))
                        _pacl_use_margin = bool(getattr(cfg, "PACL_USE_MARGIN", False))
                        _pacl_margin = float(getattr(cfg, "PACL_MARGIN", 0.30))
                        try:
                            if _pacl_w_spec > 0.0 and torch.is_tensor(z_spec_clean) and z_spec_clean.size(-1) == training_proto_tensor.size(-1):
                                _pacl_spec = prototype_anchor_loss(
                                    z_spec_clean, labels, training_proto_tensor,
                                    temperature=_pacl_temp, label_smoothing=_pacl_ls,
                                    use_margin=_pacl_use_margin, margin=_pacl_margin,
                                )
                                if torch.is_tensor(_pacl_spec["loss"]) and _pacl_spec["loss"].requires_grad:
                                    _pacl_spec_wt = _pacl_w_spec * _pacl_spec["loss"]
                                    loss = loss + _pacl_spec_wt
                                    loss_dict["pacl_spec"] = float(_pacl_spec_wt.detach().cpu().item())
                                    loss_dict["_info_pacl_spec_acc"] = _pacl_spec["acc"]
                                    loss_dict["_info_pacl_spec_pos_cos"] = _pacl_spec["mean_pos_cos"]
                                    loss_dict["_info_pacl_spec_neg_cos"] = _pacl_spec["mean_neg_cos"]
                                    loss_dict["_info_pacl_spec_per_class_acc"] = _pacl_spec.get("per_class_acc", {})
                                    loss_dict["_info_pacl_spec_per_class_pos_cos"] = _pacl_spec.get("per_class_pos_cos", {})
                                    if isinstance(loss_dict.get("_tensor_terms"), dict):
                                        loss_dict["_tensor_terms"]["pacl_spec_weighted"] = _pacl_spec_wt
                        except Exception:
                            pass
                        try:
                            if _pacl_w_fus > 0.0 and torch.is_tensor(z_fusion) and z_fusion.size(-1) == training_proto_tensor.size(-1):
                                _pacl_fus = prototype_anchor_loss(
                                    z_fusion, labels, training_proto_tensor,
                                    temperature=_pacl_temp, label_smoothing=_pacl_ls,
                                    use_margin=_pacl_use_margin, margin=_pacl_margin,
                                )
                                if torch.is_tensor(_pacl_fus["loss"]) and _pacl_fus["loss"].requires_grad:
                                    _pacl_fus_wt = _pacl_w_fus * _pacl_fus["loss"]
                                    loss = loss + _pacl_fus_wt
                                    loss_dict["pacl_fusion"] = float(_pacl_fus_wt.detach().cpu().item())
                                    if isinstance(loss_dict.get("_tensor_terms"), dict):
                                        loss_dict["_tensor_terms"]["pacl_fus_weighted"] = _pacl_fus_wt
                                    loss_dict["_info_pacl_fus_acc"] = _pacl_fus["acc"]
                                    loss_dict["_info_pacl_fus_pos_cos"] = _pacl_fus["mean_pos_cos"]
                                    loss_dict["_info_pacl_fus_neg_cos"] = _pacl_fus["mean_neg_cos"]
                        except Exception:
                            pass
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT392-S1: Learnable Prototype Regularization (separation + anchor) =====
                # ATT428: skip computation entirely when weights=0 to avoid inplace-modified-variable error
                if learnable_proto_module is not None:
                    _lp_start = int(getattr(cfg, "LEARNABLE_PROTO_START_EPOCH", 0))
                    _lp_sep_w = float(getattr(cfg, "LEARNABLE_PROTO_SEP_WEIGHT", 0.0))
                    _lp_anc_w = float(getattr(cfg, "LEARNABLE_PROTO_ANCHOR_WEIGHT", 0.0))
                    if current_epoch >= _lp_start and (_lp_sep_w > 0.0 or _lp_anc_w > 0.0):
                        if _lp_sep_w > 0.0:
                            _sep_loss = learnable_proto_module.separation_loss()
                            if torch.is_tensor(_sep_loss) and _sep_loss.requires_grad:
                                _sep_wt = _lp_sep_w * _sep_loss
                                loss = loss + _sep_wt
                                loss_dict["proto_sep"] = float(_sep_wt.detach().cpu().item())
                        if _lp_anc_w > 0.0:
                            _anc_loss = learnable_proto_module.anchor_loss()
                            if torch.is_tensor(_anc_loss) and _anc_loss.requires_grad:
                                _anc_wt = _lp_anc_w * _anc_loss
                                loss = loss + _anc_wt
                                loss_dict["proto_anchor"] = float(_anc_wt.detach().cpu().item())
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # VGSP semantic geometry is applied to the learnable prototype set.
                if (semantic_proto_source == "signal_centroid" and
                        semantic_relation_weight > 0.0 and
                        learnable_proto_module is not None):
                    _vgsp_rel = learnable_proto_module.semantic_relation_loss()
                    if torch.is_tensor(_vgsp_rel) and _vgsp_rel.requires_grad:
                        loss = loss + semantic_relation_weight * _vgsp_rel
                        loss_dict["vgsp_relation"] = float(_vgsp_rel.detach().cpu().item())

                # ===== ATT434: Gate L2 Regularization (tanh gate → push toward 0 = balanced) =====
                _gate_ent_w = float(getattr(cfg, "GATE_ENTROPY_WEIGHT", 0.0))
                if _gate_ent_w > 0.0 and hasattr(fusion_module, '_gate_for_reg') and fusion_module._gate_for_reg is not None:
                    _gv = fusion_module._gate_for_reg.float()
                    _gate_l2 = (_gv ** 2).mean()
                    _gate_l2_penalty = _gate_ent_w * _gate_l2
                    if _gate_l2_penalty.requires_grad:
                        loss = loss + _gate_l2_penalty
                        loss_dict["gate_ent_reg"] = float(_gate_l2_penalty.detach().cpu().item())
                        loss_dict["_info_gate_l2"] = float(_gate_l2.detach().cpu().item())
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT432-δ: Physical Feature Alignment Loss =====
                if phys_proto_module is not None and phys_features is not None:
                    _phys_w = float(getattr(cfg, "PHYS_FEATURE_ALIGN_WEIGHT", 0.20))
                    if _phys_w > 0.0:
                        try:
                            phys_proto_module.train()
                            z_phys = phys_proto_module(phys_features, class_ids=labels)
                            if torch.is_tensor(z_fusion) and z_fusion.dim() >= 1:
                                _z_fus_phys = z_fusion
                                if hasattr(z_fusion, "_pre_l2_feature"):
                                    _z_fus_phys = F.normalize(z_fusion._pre_l2_feature, dim=-1)
                                _phys_cos = (z_phys * _z_fus_phys.detach()).sum(dim=-1)
                                _phys_align_loss = (1.0 - _phys_cos).mean()
                                _phys_loss_wt = _phys_w * _phys_align_loss
                                if _ablation_gate_bank is not None:
                                    _phys_loss_wt = _ablation_gate_bank.scale_loss(
                                        "physics_phys_proto", _phys_loss_wt
                                    )
                                loss = loss + _phys_loss_wt
                                loss_dict["phys_align"] = float(_phys_loss_wt.detach().cpu().item())
                                loss_dict["_info_phys_cos_mean"] = float(_phys_cos.mean().detach().cpu().item())
                                loss_dict["total"] = float(loss.detach().cpu().item())
                        except Exception as _phys_e:
                            if batch_idx == 0:
                                print(f"[ATT432-δ] phys_align error: {_phys_e}")

                # ===== ATT438-DGL: Auxiliary Fusion Head CE Loss =====
                _dgl_on_now = bool(getattr(cfg, "DGL_ENABLE", False))
                _dgl_aux_w = float(getattr(cfg, "DGL_AUX_FUSION_HEAD_WEIGHT", 0.30))
                if _dgl_on_now and _dgl_aux_w > 0.0 and hasattr(fusion_module, 'get_aux_logits'):
                    _dgl_aux_logits = fusion_module.get_aux_logits()
                    if _dgl_aux_logits is not None and labels is not None:
                        try:
                            _dgl_labels = labels.long()
                            _dgl_n_cls = _dgl_aux_logits.size(-1)
                            _dgl_cw = None
                            if class_weights is not None and class_weights.numel() == _dgl_n_cls:
                                _dgl_cw = class_weights.to(_dgl_aux_logits.device)
                            _dgl_ce = F.cross_entropy(_dgl_aux_logits, _dgl_labels, weight=_dgl_cw)
                            _dgl_loss_wt = _dgl_aux_w * _dgl_ce
                            if _dgl_loss_wt.requires_grad:
                                loss = loss + _dgl_loss_wt
                                loss_dict["dgl_aux_ce"] = float(_dgl_loss_wt.detach().cpu().item())
                                loss_dict["_info_dgl_aux_ce_raw"] = float(_dgl_ce.detach().cpu().item())
                                with torch.no_grad():
                                    _dgl_preds = _dgl_aux_logits.argmax(dim=-1)
                                    _dgl_acc = (_dgl_preds == _dgl_labels).float().mean().item()
                                loss_dict["_info_dgl_aux_acc"] = _dgl_acc
                                loss_dict["total"] = float(loss.detach().cpu().item())
                                if batch_idx == 0 and (current_epoch <= 3 or current_epoch % 5 == 0):
                                    print(f"[ATT438-DGL][E{current_epoch}B{batch_idx}] "
                                          f"aux_ce_raw={float(_dgl_ce.detach().cpu().item()):.4f} "
                                          f"aux_ce_wt={float(_dgl_loss_wt.detach().cpu().item()):.4f} "
                                          f"aux_acc={_dgl_acc:.4f}")
                        except Exception as _dgl_e:
                            if batch_idx == 0:
                                print(f"[ATT438-DGL] aux_ce error: {_dgl_e}")

                # ===== ATT439: Cross-Modal Knowledge Distillation (Signal → Spec) =====
                _xkd_on = bool(getattr(cfg, "CROSS_MODAL_KD_ENABLE", False))
                _xkd_w = float(getattr(cfg, "CROSS_MODAL_KD_WEIGHT", 0.50))
                _xkd_T = float(getattr(cfg, "CROSS_MODAL_KD_TEMPERATURE", 4.0))
                if _xkd_on and _xkd_w > 0.0 and signal_logits is not None and spec_logits is not None:
                    try:
                        _sig_n = signal_logits.size(-1)
                        _spe_n = spec_logits.size(-1)
                        _kd_n = min(_sig_n, _spe_n)
                        _sig_kd = signal_logits[:, :_kd_n].detach().float()
                        _spe_kd = spec_logits[:, :_kd_n].float()
                        _p_teacher = F.softmax(_sig_kd / _xkd_T, dim=-1)
                        _log_p_student = F.log_softmax(_spe_kd / _xkd_T, dim=-1)
                        _kd_loss = F.kl_div(_log_p_student, _p_teacher, reduction='batchmean') * (_xkd_T ** 2)
                        if _kd_loss.requires_grad:
                            _kd_loss_wt = _xkd_w * _kd_loss
                            loss = loss + _kd_loss_wt
                            loss_dict["xkd_sig2spec"] = float(_kd_loss_wt.detach().cpu().item())
                            loss_dict["_info_xkd_raw"] = float(_kd_loss.detach().cpu().item())
                            loss_dict["total"] = float(loss.detach().cpu().item())
                            if batch_idx == 0 and (current_epoch <= 3 or current_epoch % 5 == 0):
                                with torch.no_grad():
                                    _sig_pred = signal_logits[:, :_kd_n].argmax(dim=-1)
                                    _spe_pred = spec_logits[:, :_kd_n].argmax(dim=-1)
                                    _agree = (_sig_pred == _spe_pred).float().mean().item()
                                print(f"[ATT439-XKD][E{current_epoch}B{batch_idx}] "
                                      f"kd_raw={float(_kd_loss.detach().cpu().item()):.4f} "
                                      f"kd_wt={float(_kd_loss_wt.detach().cpu().item()):.4f} "
                                      f"T={_xkd_T:.1f} agree={_agree:.4f} "
                                      f"sig_cls={_sig_n} spec_cls={_spe_n} kd_cls={_kd_n}")
                    except Exception as _xkd_e:
                        if batch_idx == 0:
                            print(f"[ATT439-XKD] error: {_xkd_e}")

                # ===== ATT391-S1: Adapter PACL (ETF→Text bridge) =====
                if proto_adapter is not None and _adapter_text_proto_tensor is not None:
                    _ada_start = int(getattr(cfg, "ETF_TEXT_ADAPTER_START_EPOCH", 2))
                    _ada_w = float(getattr(cfg, "ETF_TEXT_ADAPTER_WEIGHT", 0.30))
                    _ada_t = float(getattr(cfg, "ETF_TEXT_ADAPTER_TEMPERATURE", 0.07))
                    if current_epoch >= _ada_start and _ada_w > 0.0:
                        try:
                            _z_fus_det = z_fusion.detach()
                            _z_adapted = proto_adapter(_z_fus_det)
                            _ada_result = prototype_anchor_loss(
                                _z_adapted, labels, _adapter_text_proto_tensor,
                                temperature=_ada_t, label_smoothing=0.05,
                            )
                            if torch.is_tensor(_ada_result["loss"]) and _ada_result["loss"].requires_grad:
                                _ada_loss_wt = _ada_w * _ada_result["loss"]
                                loss = loss + _ada_loss_wt
                                loss_dict["adapter_pacl"] = float(_ada_loss_wt.detach().cpu().item())
                                loss_dict["_info_adapter_acc"] = _ada_result["acc"]
                                loss_dict["_info_adapter_pos_cos"] = _ada_result["mean_pos_cos"]
                                loss_dict["_info_adapter_neg_cos"] = _ada_result["mean_neg_cos"]
                        except Exception:
                            pass
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT374+ATT377: Text-ETF Alignment Loss + F2/E1 cosine matrix =====
                _text_etf_interval = int(getattr(cfg, "TEXT_ETF_ALIGNMENT_BATCH_INTERVAL", (1 if core_loss_only_mode else 10**9)))
                if use_etf and etf_prototypes_tensor is not None and (batch_idx % max(1, _text_etf_interval) == 0):
                    _text_etf_active = bool(getattr(cfg, "USE_TEXT_ETF_ALIGNMENT_LOSS", False))
                    _text_etf_start = int(getattr(cfg, "TEXT_ETF_ALIGNMENT_START_EPOCH", 1))
                    if _text_etf_active and current_epoch >= _text_etf_start:
                        _text_etf_w = float(getattr(cfg, "TEXT_ETF_ALIGNMENT_WEIGHT", 0.3))
                        _text_cos_matrix_w = float(getattr(cfg, "TEXT_ETF_COSINE_MATRIX_WEIGHT", 0.10))
                        try:
                            _tea_result = text_etf_alignment_loss(
                                text_encoder, class_prompts, etf_prototypes_tensor, device=device,
                                cosine_matrix_weight=_text_cos_matrix_w,
                            )
                            if torch.is_tensor(_tea_result["loss"]) and _tea_result["loss"].requires_grad:
                                loss = loss + _text_etf_w * _tea_result["loss"]
                                loss_dict["text_etf_align"] = float((_text_etf_w * _tea_result["loss"]).detach().cpu().item())
                                loss_dict["_info_text_etf_cos"] = _tea_result["mean_cos"]
                                loss_dict["_info_text_cos_margin"] = _tea_result.get("text_cos_margin", 0.0)
                                loss_dict["total"] = float(loss.detach().cpu().item())
                            if batch_idx == 0:
                                print(
                                    f"[TextProtoCos][E{current_epoch}] diag={_tea_result.get('text_cos_diag', 0.0):.4f}, "
                                    f"offdiag={_tea_result.get('text_cos_offdiag', 0.0):.4f}, "
                                    f"margin={_tea_result.get('text_cos_margin', 0.0):.4f}, "
                                    f"matrix_loss={_tea_result.get('loss_matrix_match', 0.0):.6f}"
                                )
                        except Exception:
                            pass

                _batch_should_log_loss = bool(batch_loss_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % batch_loss_log_step_stride == 0)))
                _batch_should_log_proto = bool(proto_cos_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % proto_cos_log_step_stride == 0)))
                _batch_should_log_io = bool(batch_io_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % batch_io_log_step_stride == 0)))
                _batch_should_log_grad = bool(grad_contrib_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % grad_contrib_log_step_stride == 0)))
                if isinstance(loss_dict, dict):
                    _tensor_terms = loss_dict.get("_tensor_terms", None)
                    if isinstance(_tensor_terms, dict):
                        for _name, _term in _tensor_terms.items():
                            if torch.is_tensor(_term):
                                grad_probe_terms[_name] = _term
                    for _k_default in (
                        "total",
                        "diag_ce",
                        "nf_ce",
                        "fault_3way_ce",
                        "supcon",
                        "global_decorr",
                        "clip_style",
                        "fusion_proto_align",
                        "fusion_supcon",
                        "nr_inter_cos_margin",
                        "text_prompt_sep",
                        "nr_inter_cos_hard",
                        "fishr",
                        "ccpa",
                    ):
                        if _k_default not in loss_dict:
                            loss_dict[_k_default] = 0.0
                    for _k_l, _v_l in loss_dict.items():
                        if not isinstance(_v_l, (int, float)):
                            continue
                        _vf = float(_v_l)
                        if (not math.isfinite(_vf)):
                            continue
                        if str(_k_l).startswith("_info_"):
                            epoch_loss_info_sum[_k_l] = epoch_loss_info_sum.get(_k_l, 0.0) + _vf
                            epoch_loss_info_count[_k_l] = epoch_loss_info_count.get(_k_l, 0) + 1
                        else:
                            epoch_loss_term_sum[_k_l] = epoch_loss_term_sum.get(_k_l, 0.0) + _vf
                            epoch_loss_term_count[_k_l] = epoch_loss_term_count.get(_k_l, 0) + 1
                if _batch_should_log_loss and isinstance(loss_dict, dict):
                    _keys = [
                        "total", "_info_fusion_cls", "_info_fusion_cls_raw_skipped", "fusion_cls_skipped",
                        "diag_ce", "nf_ce", "fault_3way_ce",
                        "supcon", "fusion_supcon", "global_decorr", "clip_style", "fusion_proto_align", "nr_inter_cos_margin",
                        "text_prompt_sep", "nr_inter_cos_hard", "fishr", "ccpa", "proto_sep", "proto_anchor"
                    ]
                    _parts = []
                    for _k in _keys:
                        try:
                            _parts.append(f"{_k}={float(loss_dict.get(_k, 0.0)):.4f}")
                        except Exception:
                            pass
                    try:
                        _lbl_hist = torch.bincount(labels.detach().cpu().long(), minlength=4).tolist()
                    except Exception:
                        _lbl_hist = []
                    try:
                        _dom_hist = torch.bincount(domain_ids.detach().cpu().long(), minlength=5).tolist() if domain_ids is not None else []
                    except Exception:
                        _dom_hist = []
                    print(
                        f"[BatchLoss][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                        + " | ".join(_parts)
                        + f" | y_hist={_lbl_hist} | dom_hist={_dom_hist}"
                    )
                    if batch_loss_breakdown_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % batch_loss_breakdown_step_stride == 0)):
                        _num_terms = {}
                        _info_terms = {}
                        for _k_bd, _v_bd in loss_dict.items():
                            if not isinstance(_v_bd, (int, float)):
                                continue
                            _v_bd_f = float(_v_bd)
                            if not math.isfinite(_v_bd_f):
                                continue
                            if str(_k_bd).startswith("_info_"):
                                _info_terms[_k_bd] = _v_bd_f
                            elif _k_bd != "total":
                                _num_terms[_k_bd] = _v_bd_f
                        _total_bd = float(loss_dict.get("total", 0.0))
                        _sum_non_total = float(sum(_num_terms.values()))
                        _residual = float(_total_bd - _sum_non_total)
                        _info_sum = float(sum(_info_terms.values()))
                        _top_terms = sorted(_num_terms.items(), key=lambda _x: abs(_x[1]), reverse=True)[:8]
                        _top_info = sorted(_info_terms.items(), key=lambda _x: abs(_x[1]), reverse=True)[:6]
                        _top_s = ", ".join([f"{_k}={_v:.4f}" for _k, _v in _top_terms]) if _top_terms else "none"
                        _top_info_s = ", ".join([f"{_k}={_v:.4f}" for _k, _v in _top_info]) if _top_info else "none"
                        print(
                            f"[BatchLossBreakdown][AMP=ON][E{current_epoch}B{batch_idx}] "
                            f"total={_total_bd:.4f}, sum_non_total={_sum_non_total:.4f}, residual={_residual:.4f}, "
                            f"terms={len(_num_terms)}, info_terms={len(_info_terms)}, info_sum={_info_sum:.4f}, "
                            f"top={_top_s}, info_top={_top_info_s}"
                        )
                        # ATT375: full term dump for first batch of each task
                        if batch_idx == 0 and current_epoch == 1:
                            _all_sorted = sorted(_num_terms.items(), key=lambda _x: abs(_x[1]), reverse=True)
                            _all_s = ", ".join([f"{_k}={_v:.4f}" for _k, _v in _all_sorted])
                            print(f"[LossBudgetAudit][E{current_epoch}B0] ALL {len(_num_terms)} terms: {_all_s}")
                            print(
                                f"[LossBudgetAudit] NOTE: residual!=0 is EXPECTED. "
                                f"alignment_loss stores RAW values, total is WEIGHTED sum. "
                                f"This is NOT a double-counting bug."
                            )
                if _batch_should_log_proto:
                    _proto_mat, _proto_keys = _to_proto_matrix(
                        current_prototypes,
                        z_fusion.device if torch.is_tensor(z_fusion) else device,
                        z_fusion.dtype if torch.is_tensor(z_fusion) else torch.float32,
                    )
                    if _proto_mat is not None:
                        try:
                            _proto_mat = F.normalize(_proto_mat.float(), dim=-1, eps=1e-6)
                            _cos = torch.matmul(_proto_mat, _proto_mat.t()).detach().cpu().numpy()
                            _rows = []
                            for _i, _k in enumerate(_proto_keys):
                                _vals = ",".join([f"{float(_v):.3f}" for _v in _cos[_i].tolist()])
                                _rows.append(f"{int(_k)}:[{_vals}]")
                            print(
                                f"[ProtoCos][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                + " | ".join(_rows)
                            )
                        except Exception:
                            pass
                if _batch_should_log_io:
                    try:
                        _z_spec_pre = extract_pre_l2_feature(z_spec_clean) if torch.is_tensor(z_spec_clean) else None
                        _z_fus_pre = extract_pre_l2_feature(z_fusion) if torch.is_tensor(z_fusion) else None
                        _sig_std = float(z_signal.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(z_signal) else 0.0
                        _spec_std = float(z_spec_clean.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(z_spec_clean) else 0.0
                        _fus_std = float(z_fusion.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(z_fusion) else 0.0
                        _sig_norm = float(z_signal.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(z_signal) else 0.0
                        _spec_norm = float(z_spec_clean.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(z_spec_clean) else 0.0
                        _fus_norm = float(z_fusion.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(z_fusion) else 0.0
                        _spec_pre_std = float(_z_spec_pre.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(_z_spec_pre) else 0.0
                        _fus_pre_std = float(_z_fus_pre.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(_z_fus_pre) else 0.0
                        _spec_pre_norm = float(_z_spec_pre.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(_z_spec_pre) else 0.0
                        _fus_pre_norm = float(_z_fus_pre.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(_z_fus_pre) else 0.0
                        _spec_fus_pre_cos = float("nan")
                        if torch.is_tensor(_z_spec_pre) and torch.is_tensor(_z_fus_pre):
                            if _z_spec_pre.shape == _z_fus_pre.shape and _z_spec_pre.dim() == 2:
                                _spec_fus_pre_cos = float(
                                    F.cosine_similarity(
                                        F.normalize(_z_spec_pre.detach().float(), dim=-1),
                                        F.normalize(_z_fus_pre.detach().float(), dim=-1),
                                        dim=-1
                                    ).mean().cpu().item()
                                )
                        print(
                            f"[BatchIO][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"std(sig/spec/fus)=({_sig_std:.4f},{_spec_std:.4f},{_fus_std:.4f}) "
                            f"norm(sig/spec/fus)=({_sig_norm:.4f},{_spec_norm:.4f},{_fus_norm:.4f})"
                        )
                        print(
                            f"[BatchIOPreL2][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"std(spec/fus)=({_spec_pre_std:.4f},{_fus_pre_std:.4f}) "
                            f"norm(spec/fus)=({_spec_pre_norm:.4f},{_fus_pre_norm:.4f}) "
                            f"cos(spec,fus)={_spec_fus_pre_cos:.4f}"
                        )
                        try:
                            if signal_logits is not None and torch.is_tensor(signal_logits):
                                with torch.no_grad():
                                    _p_sig_b = F.softmax(signal_logits.float(), dim=1)
                                    _H_sig_b = -torch.sum(_p_sig_b * torch.log(torch.clamp(_p_sig_b, min=1e-8)), dim=1)
                                    _H_max_b = math.log(max(2, _p_sig_b.size(1)))
                                    _info_sig_b = (1.0 - _H_sig_b / _H_max_b).clamp(min=0.0, max=1.0)
                                    _seg_gate_b = getattr(fusion_module, '_last_signal_entropy_gate', None)
                                    print(
                                        f"[SignalEntropy][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                        f"H_norm={float(_H_sig_b.mean()/_H_max_b):.4f} info_score={float(_info_sig_b.mean()):.4f} "
                                        f"gate={_seg_gate_b if _seg_gate_b is not None else 'N/A'}"
                                    )
                        except Exception:
                            pass
                        if bool(getattr(cfg, "USE_DUAL_SPACE_DECOUPLE", False)) and isinstance(hierarchical_parts, dict):
                            _z_cls_now = hierarchical_parts.get("z_cls", None)
                            _z_metric_now = hierarchical_parts.get("z_metric", None)
                            if torch.is_tensor(_z_cls_now) and torch.is_tensor(_z_metric_now):
                                _z_cls_n = F.normalize(_z_cls_now.detach().float(), dim=-1)
                                _z_met_n = F.normalize(_z_metric_now.detach().float(), dim=-1)
                                _dual_cos = float(F.cosine_similarity(_z_cls_n, _z_met_n, dim=-1).mean().cpu().item())
                                _dual_cls_norm = float(_z_cls_now.detach().float().norm(dim=-1).mean().cpu().item())
                                _dual_met_norm = float(_z_metric_now.detach().float().norm(dim=-1).mean().cpu().item())
                                print(
                                    f"[DualSpace][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                    f"norm(cls/metric)=({_dual_cls_norm:.4f},{_dual_met_norm:.4f}) "
                                    f"cos(cls,metric)={_dual_cos:.4f}"
                                )
                        if bool(getattr(cfg, "USE_TEXT_PROMPT_TUNING", False)) and isinstance(current_prototypes, dict) and 1 in current_prototypes and 2 in current_prototypes:
                            _p1 = F.normalize(current_prototypes[1].detach().float(), dim=-1)
                            _p2 = F.normalize(current_prototypes[2].detach().float(), dim=-1)
                            _io_cos = float(torch.sum(_p1 * _p2).cpu().item())
                            _prompt_stats = {"scale": 0.0, "offset_norm_mean": 0.0, "offset_norm_max": 0.0}
                            if hasattr(fusion_decoder, "get_text_prompt_offset_norms"):
                                _prompt_stats = fusion_decoder.get_text_prompt_offset_norms()
                            print(
                                f"[PromptTuning][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                f"inner_outer_cos={_io_cos:.4f} "
                                f"scale={float(_prompt_stats.get('scale', 0.0)):.4f} "
                                f"offset_norm_mean={float(_prompt_stats.get('offset_norm_mean', 0.0)):.4f} "
                                f"offset_norm_max={float(_prompt_stats.get('offset_norm_max', 0.0)):.4f}"
                            )
                        _spec_freq_pos = batch.get("spec_freq_pos", None)
                        if bool(getattr(cfg, "PHYSICAL_FREQ_LOG_STATS", True)) and torch.is_tensor(_spec_freq_pos):
                            _fp = _spec_freq_pos.detach().float()
                            _pos_scale = float(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_SCALE", 0.0))
                            _fp_delta = torch.abs(_fp * _pos_scale)
                            _spec_abs = torch.abs(spec.detach().float()) if torch.is_tensor(spec) else None
                            _energy_ratio = float("nan")
                            if torch.is_tensor(_spec_abs):
                                _spec_mean = float(_spec_abs.mean().cpu().item())
                                _delta_mean = float(_fp_delta.mean().cpu().item())
                                _energy_ratio = _delta_mean / max(1e-8, _spec_mean)
                            print(
                                f"[FreqPosEnc][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                f"min/max/mean=({float(_fp.min().cpu().item()):.4f},"
                                f"{float(_fp.max().cpu().item()):.4f},"
                                f"{float(_fp.mean().cpu().item()):.4f}) "
                                f"delta_scale={_pos_scale:.3f} energy_ratio={_energy_ratio:.4f}"
                            )
                    except Exception:
                        pass
                if _batch_should_log_grad:
                    _grad_terms = {
                        "fusion_cls": grad_probe_terms.get("fusion_cls_weighted", None),
                        "supcon": grad_probe_terms.get("supcon_weighted", None),
                        "fusion_supcon": grad_probe_terms.get("fusion_supcon_weighted", None),
                        "vicreg_or_decorr": grad_probe_terms.get("global_decorr_weighted", None),
                        "fusion_proto_align": grad_probe_terms.get("fusion_proto_align_weighted", None),
                        "nr_inter_cos_margin": grad_probe_terms.get("nr_inter_cos_margin_weighted", None),
                        "text_prompt_sep": grad_probe_terms.get("text_prompt_sep_weighted", None),
                    }
                    _grad_snapshot = _grad_contrib_snapshot(loss, _grad_terms)
                    if _grad_snapshot is not None:
                        _g_total_norm, _rows = _grad_snapshot
                        _rows_str = " | ".join(
                            [f"{_n}:g={_gn:.3e},r={_r:.3f},cos={_c:.3f}" for _n, _gn, _r, _c in _rows[:5]]
                        )
                        print(
                            f"[GradContrib][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"g_total={_g_total_norm:.3e} | {_rows_str}"
                        )
                    else:
                        _active_terms = [
                            _n for _n, _t in _grad_terms.items()
                            if torch.is_tensor(_t) and _t.requires_grad
                        ]
                        print(
                            f"[GradContrib][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"unavailable(active_terms={_active_terms})"
                        )
                if (not torch.is_tensor(loss)) or (not loss.requires_grad):
                    param = None
                    try:
                        for _m in (fusion_decoder, fusion_module, signal_encoder, spec_encoder):
                            for _p in _m.parameters():
                                if _p.requires_grad:
                                    param = _p
                                    break
                            if param is not None:
                                break
                    except Exception:
                        param = None
                    if param is not None:
                        loss = param.sum() * 0.0
                    else:
                        try:
                            loss = z_fusion.sum() * 0.0
                        except Exception:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                    if isinstance(loss_dict, dict):
                        loss_dict["no_grad_loss"] = 1.0
                if batch_idx == 0 and current_epoch <= 3:
                    try:
                        _d = []
                        _d.append(f"loss.req_grad={loss.requires_grad}")
                        _d.append(f"loss.grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else 'None'}")
                        if hasattr(fusion_decoder, 'classifier'):
                            _cls_w = fusion_decoder.classifier.weight
                            _d.append(f"cls_w.req_grad={_cls_w.requires_grad}")
                        _dec_in_opt = 0
                        _dec_id_set = set(id(p) for p in fusion_decoder.parameters())
                        for _og in optimizer.param_groups:
                            for _op in _og.get("params", []):
                                if id(_op) in _dec_id_set:
                                    _dec_in_opt += 1
                        _d.append(f"dec_in_opt={_dec_in_opt}/{len(_dec_id_set)}")
                        if hasattr(fusion_logits, 'requires_grad'):
                            _d.append(f"flogits.req_grad={fusion_logits.requires_grad}")
                            _d.append(f"flogits.grad_fn={type(fusion_logits.grad_fn).__name__ if fusion_logits.grad_fn else 'None'}")
                        _fwd_cls = loss_dict.get("fusion_cls", 0.0) if isinstance(loss_dict, dict) else 0.0
                        _d.append(f"fusion_cls_val={_fwd_cls:.6f}")
                        print(f"[ATT378c-PreBackward][E{current_epoch}B{batch_idx}] {', '.join(_d)}")
                    except Exception as _e:
                        print(f"[ATT378c-PreBackward] diag error: {_e}")

                # ===== ATT396: Add Feature Decoupling auxiliary losses =====
                if _dc_active and _z_spec_fault is not None and _z_signal_fault is not None:
                    try:
                        _cond_ids = _encode_cond_ids(domains, conditions, device)
                        _grl_lam = float(getattr(cfg, "DECOUPLE_ADV_LAMBDA_MAX", 1.0))
                        if current_epoch < int(getattr(cfg, "DECOUPLE_START_EPOCH", 1)) + 5:
                            _progress = float(current_epoch - int(getattr(cfg, "DECOUPLE_START_EPOCH", 1))) / 5.0
                            _grl_lam *= max(0.0, min(1.0, _progress))
                        _dc_loss, _dc_dict = _compute_decouple_losses(
                            z_fault_spec=_z_spec_fault,
                            z_cond_spec=_z_spec_cond,
                            z_fault_signal=_z_signal_fault,
                            z_cond_signal=_z_signal_cond,
                            condition_ids=_cond_ids,
                            condition_classifier=condition_classifier,
                            grl_lambda=_grl_lam,
                            w_cond=float(getattr(cfg, "DECOUPLE_COND_WEIGHT", 0.10)),
                            w_adv=float(getattr(cfg, "DECOUPLE_ADV_WEIGHT", 0.10)),
                            w_ortho=float(getattr(cfg, "DECOUPLE_ORTHO_WEIGHT", 0.05)),
                        )
                        loss = loss + _dc_loss
                        if isinstance(loss_dict, dict):
                            loss_dict.update(_dc_dict)
                        if batch_idx == 0 and current_epoch <= 5:
                            print(
                                f"[ATT396][Decouple-Loss] E{current_epoch} "
                                f"cond={_dc_dict.get('decouple_cond_loss', 0):.4f}/"
                                f"acc={_dc_dict.get('decouple_cond_acc', 0):.3f}, "
                                f"adv={_dc_dict.get('decouple_adv_loss', 0):.4f}/"
                                f"acc={_dc_dict.get('decouple_adv_acc', 0):.3f}, "
                                f"ortho={_dc_dict.get('decouple_ortho_loss', 0):.4f}, "
                                f"grl_lam={_grl_lam:.3f}"
                            )
                    except Exception as _dc_e:
                        if batch_idx == 0:
                            print(f"[ATT396][Decouple-Loss] Error: {_dc_e}")

                # ATT397: Episodic Cross-Condition Prototype Loss (AMP path)
                _use_episodic = bool(getattr(cfg, "USE_EPISODIC_TRAINING", False))
                if _use_episodic and conditions is not None:
                    try:
                        from training.episodic_sampler import split_batch_by_condition
                        _ep_split = str(getattr(cfg, "EPISODIC_SPLIT_KEY", "auto"))
                        if _ep_split == "auto":
                            _ep_split = "condition"
                        _s_mask, _q_mask, _g_ids = split_batch_by_condition(
                            batch, split_key=_ep_split, batch_idx=batch_idx)
                        _s_mask = _s_mask.to(device)
                        _q_mask = _q_mask.to(device)
                        if _s_mask.sum() >= 2 and _q_mask.sum() >= 2:
                            _ep_feat = z_spec if z_spec is not None else z_signal
                            _ep_feat_norm = F.normalize(_ep_feat.detach(), dim=-1)
                            _ep_feat_q = F.normalize((z_spec if z_spec is not None else z_signal), dim=-1)
                            _s_feat = _ep_feat_norm[_s_mask]
                            _s_labels = labels[_s_mask]
                            _q_feat = _ep_feat_q[_q_mask]
                            _q_labels = labels[_q_mask]
                            _ep_classes = _s_labels.unique()
                            _protos = []
                            _proto_labels = []
                            for _c in _ep_classes:
                                _c_mask = (_s_labels == _c)
                                if _c_mask.sum() > 0:
                                    _protos.append(_s_feat[_c_mask].mean(dim=0))
                                    _proto_labels.append(_c)
                            if len(_protos) >= 2:
                                _protos_t = torch.stack(_protos)
                                _proto_labels_t = torch.stack(_proto_labels)
                                _ep_temp = float(getattr(cfg, "EPISODIC_TEMPERATURE", 0.07))
                                _ep_sims = _q_feat @ _protos_t.T / _ep_temp
                                _ep_targets = torch.zeros(_q_feat.size(0), dtype=torch.long, device=device)
                                for _pi, _pl in enumerate(_proto_labels_t):
                                    _ep_targets[_q_labels == _pl] = _pi
                                _valid_q = torch.zeros(_q_feat.size(0), dtype=torch.bool, device=device)
                                for _pl in _proto_labels_t:
                                    _valid_q |= (_q_labels == _pl)
                                if _valid_q.sum() >= 2:
                                    _ep_loss = F.cross_entropy(_ep_sims[_valid_q], _ep_targets[_valid_q])
                                    _ep_w = float(getattr(cfg, "EPISODIC_PROTO_LOSS_WEIGHT", 0.5))
                                    loss = loss + _ep_w * _ep_loss
                                    if isinstance(loss_dict, dict):
                                        loss_dict["episodic_proto_loss"] = float(_ep_loss.detach())
                                    if batch_idx == 0:
                                        print(
                                            f"[ATT397-Episodic] E{current_epoch}B{batch_idx} "
                                            f"ep_loss={float(_ep_loss.detach()):.4f} "
                                            f"n_support={int(_s_mask.sum())} n_query={int(_q_mask.sum())} "
                                            f"n_proto_classes={len(_protos)} valid_query={int(_valid_q.sum())}"
                                        )
                    except Exception as _ep_e:
                        if batch_idx == 0:
                            print(f"[ATT397-Episodic] Error: {_ep_e}")

                # ===== ATT405/407: Training-time domain alignment on z_fusion (MMD/CORAL) =====
                _fda_enable = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_ENABLE", False))
                _fda_disable_task1 = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1", True))
                _is_task1 = ("task2" not in _phase_tag) and len(_phase_tag) > 0
                if _fda_disable_task1 and _is_task1:
                    _fda_enable = False
                    if batch_idx == 0 and current_epoch == 1:
                        print(f"[ATT407-DomainAlign] SKIPPED for Task1 phase={_phase_tag}")
                if _fda_enable and z_fusion is not None and domain_ids is not None:
                    try:
                        _fda_start = int(getattr(cfg, "FUSION_DOMAIN_ALIGN_START_EPOCH", 3))
                        if current_epoch >= _fda_start:
                            _fda_w_base = float(getattr(cfg, "FUSION_DOMAIN_ALIGN_WEIGHT", 0.05))
                            _fda_warmup = int(getattr(cfg, "FUSION_DOMAIN_ALIGN_WARMUP_EPOCHS", 3))
                            _fda_progress = min(1.0, float(current_epoch - _fda_start) / max(1, _fda_warmup))
                            _fda_w = _fda_w_base * _fda_progress
                            _fda_detach = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_DETACH", False))
                            _fda_min_n = int(getattr(cfg, "FUSION_DOMAIN_ALIGN_MIN_SAMPLES", 4))
                            _fda_mode = str(getattr(cfg, "FUSION_DOMAIN_ALIGN_LOSS", "mmd")).lower()

                            _zf = z_fusion.detach() if _fda_detach else z_fusion
                            _unique_doms = domain_ids.unique()
                            _fda_losses = []
                            for _i_d in range(len(_unique_doms)):
                                for _j_d in range(_i_d + 1, len(_unique_doms)):
                                    _mask_i = (domain_ids == _unique_doms[_i_d])
                                    _mask_j = (domain_ids == _unique_doms[_j_d])
                                    if _mask_i.sum() >= _fda_min_n and _mask_j.sum() >= _fda_min_n:
                                        _fi = _zf[_mask_i]
                                        _fj = _zf[_mask_j]
                                        if _fda_mode == "coral":
                                            _fda_losses.append(coral_covariance_alignment_loss(_fi, _fj))
                                        else:
                                            _fda_losses.append(multi_kernel_mmd_loss(_fi, _fj))
                            if len(_fda_losses) > 0:
                                _fda_total = sum(_fda_losses) / len(_fda_losses)
                                _fda_loss = _fda_w * _fda_total
                                loss = loss + _fda_loss
                                if isinstance(loss_dict, dict):
                                    loss_dict["fusion_domain_align"] = float(_fda_loss.detach())
                                if batch_idx == 0:
                                    print(
                                        f"[ATT406-DomainAlign] E{current_epoch}B{batch_idx} "
                                        f"mode={_fda_mode} pairs={len(_fda_losses)} "
                                        f"raw={float(_fda_total.detach()):.6f} "
                                        f"weighted={float(_fda_loss.detach()):.6f} "
                                        f"w={_fda_w:.4f}(warmup={_fda_progress:.2f})"
                                    )
                    except Exception as _fda_e:
                        if batch_idx == 0:
                            print(f"[ATT406-DomainAlign] Error: {_fda_e}")

                # ATT447: Asymmetric Fusion→Spec Knowledge Distillation
                if _spec_distill_active and current_epoch >= _distill_start_epoch and spec_logits is not None and z_fusion is not None:
                    try:
                        _proto_matrix = None
                        if learnable_proto_module is not None:
                            _proto_matrix = learnable_proto_module.forward().detach()
                        elif isinstance(current_prototypes, dict):
                            _rows = [current_prototypes[k].detach() for k in sorted(current_prototypes.keys())]
                            _proto_matrix = torch.stack(_rows, dim=0)
                        elif torch.is_tensor(current_prototypes):
                            _proto_matrix = current_prototypes.detach()

                        if _proto_matrix is not None:
                            _z_fus_d = z_fusion.detach()
                            _fus_cos = torch.matmul(F.normalize(_z_fus_d, dim=-1), F.normalize(_proto_matrix, dim=-1).t())
                            _soft_targets = F.softmax(_fus_cos / _distill_temperature, dim=-1)

                            # ATT449-S2: FiGKD-inspired high-frequency output KD
                            if _distill_hf_only:
                                _fus_hf = _fus_cos - _fus_cos.mean(dim=-1, keepdim=True)
                                _spec_hf = spec_logits - spec_logits.mean(dim=-1, keepdim=True)
                                _soft_targets_hf = F.softmax(_fus_hf / _distill_temperature, dim=-1)
                                _spec_log_p = F.log_softmax(_spec_hf / _distill_temperature, dim=-1)
                                _kl = F.kl_div(_spec_log_p, _soft_targets_hf, reduction='batchmean')
                            else:
                                _spec_log_p = F.log_softmax(spec_logits / _distill_temperature, dim=-1)
                                _kl = F.kl_div(_spec_log_p, _soft_targets, reduction='batchmean')
                            _distill_loss = _distill_alpha * (_distill_temperature ** 2) * _kl
                            loss = loss + _distill_loss
                            if isinstance(loss_dict, dict):
                                loss_dict["spec_distill_kl"] = float(_distill_loss.detach().cpu().item())
                                loss_dict["spec_distill_kl_raw"] = float(_kl.detach().cpu().item())

                            if batch_idx % 50 == 0:
                                _t_tgt = _soft_targets_hf if _distill_hf_only else _soft_targets
                                _teacher_ent = float((-_t_tgt * (_t_tgt + 1e-8).log()).sum(-1).mean())
                                _student_ent = float((-_spec_log_p.exp() * _spec_log_p).sum(-1).mean())
                                _teacher_conf = float(_t_tgt.max(-1)[0].mean())
                                _agree = float((_t_tgt.argmax(-1) == spec_logits.argmax(-1)).float().mean())
                                _hf_tag = " HF" if _distill_hf_only else ""
                                print(f"[ATT449-Distill{_hf_tag}][E{current_epoch}B{batch_idx}] "
                                      f"KL_raw={float(_kl):.4f} weighted={float(_distill_loss):.4f} "
                                      f"T_ent={_teacher_ent:.3f} S_ent={_student_ent:.3f} "
                                      f"T_conf={_teacher_conf:.3f} agree={_agree:.3f}")
                    except Exception as _distill_e:
                        if batch_idx == 0:
                            print(f"[ATT447-Distill] Error: {_distill_e}")

                # ATT448/ATT449-S1: Relational Knowledge Distillation with Fault-Focused component
                if _rkd_enable and current_epoch >= _rkd_start_epoch and z_fusion is not None and z_spec is not None:
                    try:
                        _z_fus_rkd = z_fusion.detach()
                        _z_spec_rkd = extract_pre_l2_feature(z_spec)
                        if _z_spec_rkd is None:
                            _z_spec_rkd = z_spec

                        _B_rkd = _z_fus_rkd.size(0)
                        if _B_rkd >= 4:
                            _rkd_loss_total = torch.tensor(0.0, device=_z_fus_rkd.device, dtype=_z_fus_rkd.dtype)
                            _rkd_d_loss = None
                            _rkd_a_loss = None
                            _rkd_ff_d_loss = None

                            # --- Global RKD (all pairs) ---
                            _global_w = 1.0 - _rkd_fault_focus_w if _rkd_fault_focus_w > 0 else 1.0
                            if _rkd_dist_w > 0:
                                _d_fus = torch.cdist(_z_fus_rkd, _z_fus_rkd, p=2)
                                _d_spec = torch.cdist(_z_spec_rkd, _z_spec_rkd, p=2)
                                _mu_fus = _d_fus.mean().clamp(min=1e-6)
                                _mu_spec = _d_spec.mean().clamp(min=1e-6)
                                _rkd_d_loss = F.smooth_l1_loss(_d_spec / _mu_spec, (_d_fus / _mu_fus).detach())
                                _rkd_loss_total = _rkd_loss_total + _global_w * _rkd_dist_w * _rkd_d_loss

                            if _rkd_angle_w > 0 and _B_rkd >= 3:
                                _n_triplets = min(_B_rkd, 32)
                                _idx = torch.randperm(_B_rkd, device=_z_fus_rkd.device)[:_n_triplets]
                                _idx_a = _idx
                                _idx_b = _idx[torch.randperm(_n_triplets, device=_z_fus_rkd.device)]
                                _idx_c = _idx[torch.randperm(_n_triplets, device=_z_fus_rkd.device)]
                                _valid = (_idx_a != _idx_b) & (_idx_a != _idx_c) & (_idx_b != _idx_c)
                                if _valid.sum() >= 3:
                                    _ia, _ib, _ic = _idx_a[_valid], _idx_b[_valid], _idx_c[_valid]
                                    _e_fus_ab = F.normalize(_z_fus_rkd[_ib] - _z_fus_rkd[_ia], dim=-1)
                                    _e_fus_ac = F.normalize(_z_fus_rkd[_ic] - _z_fus_rkd[_ia], dim=-1)
                                    _cos_fus = (_e_fus_ab * _e_fus_ac).sum(-1)
                                    _e_spec_ab = F.normalize(_z_spec_rkd[_ib] - _z_spec_rkd[_ia], dim=-1)
                                    _e_spec_ac = F.normalize(_z_spec_rkd[_ic] - _z_spec_rkd[_ia], dim=-1)
                                    _cos_spec = (_e_spec_ab * _e_spec_ac).sum(-1)
                                    _rkd_a_loss = F.smooth_l1_loss(_cos_spec, _cos_fus.detach())
                                    _rkd_loss_total = _rkd_loss_total + _global_w * _rkd_angle_w * _rkd_a_loss

                            # --- ATT449-S1: Fault-Focused RKD (fault-only pairs) ---
                            if _rkd_fault_focus_w > 0 and _rkd_dist_w > 0:
                                _fault_mask = (labels != 0)
                                _n_fault = int(_fault_mask.sum())
                                if _n_fault >= 4:
                                    _zf_fus_ff = _z_fus_rkd[_fault_mask]
                                    _zf_spec_ff = _z_spec_rkd[_fault_mask]
                                    _df_fus = torch.cdist(_zf_fus_ff, _zf_fus_ff, p=2)
                                    _df_spec = torch.cdist(_zf_spec_ff, _zf_spec_ff, p=2)
                                    _mf_fus = _df_fus.mean().clamp(min=1e-6)
                                    _mf_spec = _df_spec.mean().clamp(min=1e-6)
                                    _rkd_ff_d_loss = F.smooth_l1_loss(_df_spec / _mf_spec, (_df_fus / _mf_fus).detach())
                                    _rkd_loss_total = _rkd_loss_total + _rkd_fault_focus_w * _rkd_dist_w * _rkd_ff_d_loss
                                    if _rkd_angle_w > 0 and _n_fault >= 3:
                                        _nft = min(_n_fault, 32)
                                        _fix = torch.randperm(_n_fault, device=_zf_fus_ff.device)[:_nft]
                                        _fia, _fib = _fix, _fix[torch.randperm(_nft, device=_zf_fus_ff.device)]
                                        _fic = _fix[torch.randperm(_nft, device=_zf_fus_ff.device)]
                                        _fv = (_fia != _fib) & (_fia != _fic) & (_fib != _fic)
                                        if _fv.sum() >= 3:
                                            _fia2, _fib2, _fic2 = _fia[_fv], _fib[_fv], _fic[_fv]
                                            _ef_ab = F.normalize(_zf_fus_ff[_fib2] - _zf_fus_ff[_fia2], dim=-1)
                                            _ef_ac = F.normalize(_zf_fus_ff[_fic2] - _zf_fus_ff[_fia2], dim=-1)
                                            _cf = (_ef_ab * _ef_ac).sum(-1)
                                            _es_ab = F.normalize(_zf_spec_ff[_fib2] - _zf_spec_ff[_fia2], dim=-1)
                                            _es_ac = F.normalize(_zf_spec_ff[_fic2] - _zf_spec_ff[_fia2], dim=-1)
                                            _cs = (_es_ab * _es_ac).sum(-1)
                                            _ff_a = F.smooth_l1_loss(_cs, _cf.detach())
                                            _rkd_loss_total = _rkd_loss_total + _rkd_fault_focus_w * _rkd_angle_w * _ff_a

                            loss = loss + _rkd_loss_total
                            if isinstance(loss_dict, dict):
                                loss_dict["rkd_total"] = float(_rkd_loss_total.detach().cpu().item())
                                if _rkd_d_loss is not None:
                                    loss_dict["rkd_dist"] = float((_rkd_dist_w * _rkd_d_loss).detach().cpu().item())
                                if _rkd_a_loss is not None:
                                    loss_dict["rkd_angle"] = float((_rkd_angle_w * _rkd_a_loss).detach().cpu().item())
                                if _rkd_ff_d_loss is not None:
                                    loss_dict["rkd_ff_dist"] = float((_rkd_fault_focus_w * _rkd_dist_w * _rkd_ff_d_loss).detach().cpu().item())

                            if batch_idx % 50 == 0:
                                _spec_intra = float(_d_spec.mean()) if _rkd_d_loss is not None else 0
                                _fus_intra = float(_d_fus.mean()) if _rkd_d_loss is not None else 0
                                _rkd_d_val = float(_rkd_d_loss) if _rkd_d_loss is not None else 0
                                _rkd_a_val = float(_rkd_a_loss) if _rkd_a_loss is not None else 0
                                _ff_d_val = float(_rkd_ff_d_loss) if _rkd_ff_d_loss is not None else 0
                                print(f"[ATT449-RKD][E{current_epoch}B{batch_idx}] "
                                      f"total={float(_rkd_loss_total):.4f} "
                                      f"global_d={_rkd_d_val:.4f} global_a={_rkd_a_val:.4f} "
                                      f"ff_d={_ff_d_val:.4f} "
                                      f"d_spec={_spec_intra:.3f} d_fus={_fus_intra:.3f} "
                                      f"B={_B_rkd}")
                    except Exception as _rkd_e:
                        if batch_idx == 0:
                            print(f"[ATT448-RKD] Error: {_rkd_e}")

                # ATT460 / ATT470: Cross-modal cosine alignment (SharedProjectionFusion or GatedAdditiveFusion)
                _cm_align_w_t1 = float(getattr(cfg, "FUSION_CROSS_MODAL_ALIGN_WEIGHT", 0.0))
                if _cm_align_w_t1 > 0 and hasattr(fusion_module, 'get_cross_modal_align_loss'):
                    try:
                        _cm_loss = fusion_module.get_cross_modal_align_loss()
                        if torch.is_tensor(_cm_loss) and _cm_loss.requires_grad:
                            _cm_loss_weighted = _cm_align_w_t1 * _cm_loss
                            if _ablation_gate_bank is not None:
                                _cm_loss_weighted = _ablation_gate_bank.scale_loss(
                                    "gidc_cross_modal_align", _cm_loss_weighted
                                )
                            loss = loss + _cm_loss_weighted
                            if isinstance(loss_dict, dict):
                                loss_dict["cross_modal_align"] = float(_cm_loss_weighted.detach().cpu().item())
                                loss_dict["_info_cm_cos_raw"] = float((1.0 - _cm_loss).detach().cpu().item())
                            if batch_idx % 50 == 0:
                                _cm_cos_val = float((1.0 - _cm_loss).detach().cpu().item())
                                _fus_nm = type(fusion_module).__name__
                                print(f"[CrossModalAlign][{_fus_nm}][E{current_epoch}B{batch_idx}] "
                                      f"cos={_cm_cos_val:.4f} loss={float(_cm_loss):.4f} "
                                      f"weighted={float(_cm_loss_weighted):.4f}")
                    except Exception as _cm_e:
                        if batch_idx == 0:
                            print(f"[ATT460-CMAlign] Error: {_cm_e}")

                _fus_detach_b0 = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
                if _fus_detach_b0 and batch_idx == 0 and (current_epoch == 1 or current_epoch % 10 == 0):
                    print(
                        f"[ATT471-AGIT][E{current_epoch}B0] fusion_forward_detach_encoder_inputs=1 "
                        f"=> fusion_cls/decoder/CMAlign(grad) do not backprop to signal/spec encoders "
                        f"(encoders updated via spec/signal/MLSA paths only)"
                    )
                _att475_dc = bool(getattr(cfg, "DUAL_CONTRACT_ENABLE", False))
                if _att475_dc and batch_idx == 0 and (current_epoch == 1 or current_epoch % 10 == 0):
                    print(
                        f"[ATT475-DualContract][E{current_epoch}B0] SplitArch={bool(getattr(cfg, 'SPLIT_ARCH_ENABLE', False))} "
                        f"fus_fwd_detach={_fus_detach_b0} dual_ckpt={bool(getattr(cfg, 'DUAL_CONTRACT_SAVE_EXTRA_CKPT', False))} "
                        f"| spec_contract≈spec_head CE path; fusion_contract≈fusion+decoder on detached enc outputs"
                    )
                if bool(getattr(cfg, "ATT472_TWO_END_SHARED_NECK_ENABLE", False)) and batch_idx == 0 and (current_epoch == 1 or current_epoch % 10 == 0):
                    try:
                        _ss = getattr(fusion_module, "_last_shared_semantic", None)
                        _sg = getattr(fusion_module, "_last_signal_pooled", None)
                        _sp = getattr(fusion_module, "_last_spec_pooled", None)
                        if _ss is not None and _sg is not None and _sp is not None:
                            _cos_sg_sp = float(F.cosine_similarity(_sg.detach(), _sp.detach(), dim=-1).mean().cpu().item())
                            _ss_std = float(_ss.detach().std().cpu().item())
                            _sg_std = float(_sg.detach().std().cpu().item())
                            _sp_std = float(_sp.detach().std().cpu().item())
                            print(
                                f"[ATT472-NeckAudit][E{current_epoch}B0] "
                                f"cos(sig_deep,spec_deep)={_cos_sg_sp:.4f} "
                                f"std(shared/sig_deep/spec_deep)=({_ss_std:.4f},{_sg_std:.4f},{_sp_std:.4f})"
                            )
                    except Exception as _att472_e:
                        print(f"[ATT472-NeckAudit] Error: {_att472_e}")
                if bool(getattr(cfg, "ATT473_SPECIALIST_ARBITER_ENABLE", False)) and batch_idx == 0 and (current_epoch == 1 or current_epoch % 10 == 0):
                    try:
                        _fa = getattr(fusion_module, "_last_arbiter_alpha", None)
                        _sg = getattr(fusion_module, "_last_signal_pooled", None)
                        _sp = getattr(fusion_module, "_last_spec_pooled", None)
                        if _fa is not None and _sg is not None and _sp is not None:
                            _alpha_mean = float(_fa.detach().mean().cpu().item())
                            _alpha_std = float(_fa.detach().std().cpu().item())
                            _cos_sg_sp = float(F.cosine_similarity(_sg.detach(), _sp.detach(), dim=-1).mean().cpu().item())
                            print(
                                f"[ATT473-ArbiterAudit][E{current_epoch}B0] "
                                f"alpha(mean/std)=({_alpha_mean:.4f},{_alpha_std:.4f}) "
                                f"cos(fusion_spec,expert_spec)={_cos_sg_sp:.4f}"
                            )
                    except Exception as _att473_e:
                        print(f"[ATT473-ArbiterAudit] Error: {_att473_e}")

                optimizer.zero_grad()
                _pcgrad_active = bool(getattr(cfg, "USE_PCGRAD", False))
                _pcgrad_applied = False
                if _pcgrad_active and (not supcon_stage1_now) and isinstance(loss_dict, dict):
                    try:
                        from losses.pcgrad import pcgrad_backward as _pcgrad_fn
                        _tt = loss_dict.get("_tensor_terms", {})
                        _g_ce_parts = []
                        for _k in ("fusion_cls_weighted", "signal_ce_weighted", "spec_ce_weighted"):
                            _t = _tt.get(_k)
                            if torch.is_tensor(_t) and _t.requires_grad:
                                _g_ce_parts.append(_t)
                        _g_pacl_parts = []
                        for _k in ("fusion_proto_align_weighted", "fusion_alignment_weighted",
                                    "pacl_spec_weighted", "pacl_fus_weighted"):
                            _t = _tt.get(_k)
                            if torch.is_tensor(_t) and _t.requires_grad:
                                _g_pacl_parts.append(_t)
                        _group_ce = sum(_g_ce_parts) if _g_ce_parts else None
                        _group_pacl = sum(_g_pacl_parts) if _g_pacl_parts else None
                        _groups = [g for g in [_group_ce, _group_pacl] if g is not None]
                        if len(_groups) >= 2:
                            _ = scaler.scale(loss)
                            _scale_f = scaler.get_scale()
                            _scaled = [_scale_f * g for g in _groups]
                            _shared_p = [p for pg in optimizer.param_groups for p in pg["params"] if p.requires_grad]
                            _pcg_stats = _pcgrad_fn(_scaled, _shared_p, loss_names=["CE", "PACL"], retain_graph=True)
                            _pcgrad_applied = True
                            _pcg_log_interval = int(getattr(cfg, "PCGRAD_LOG_INTERVAL", 50))
                            if batch_idx % _pcg_log_interval == 0:
                                print(f"[PCGrad][E{current_epoch}B{batch_idx}] conflicts={_pcg_stats['n_conflicts']}/{_pcg_stats['n_pairs']} ratio={_pcg_stats['conflict_ratio']:.3f}")
                    except Exception as _pcg_e:
                        if batch_idx == 0:
                            print(f"[PCGrad] fallback to standard backward: {_pcg_e}")
                        _pcgrad_applied = False
                # ATT409: Auxiliary pre-norm CE loss (bypasses F.normalize gradient bottleneck)
                if _aux_prenorm_enable and _aux_prenorm_head is not None and z_fusion_for_reg is not None:
                    try:
                        _apn_logits = _aux_prenorm_head(z_fusion_for_reg)
                        _apn_loss = torch.nn.functional.cross_entropy(_apn_logits, labels)
                        _apn_weighted = _aux_prenorm_weight * _apn_loss
                        loss = loss + _apn_weighted
                        loss_dict["aux_prenorm_ce"] = float(_apn_weighted.detach().cpu().item())
                        if batch_idx == 0 and current_epoch == 1:
                            print(f"[ATT409-AuxPrenorm] loss={float(_apn_loss):.4f}, weighted={float(_apn_weighted):.4f}")
                    except Exception as _apn_e:
                        if batch_idx == 0:
                            print(f"[ATT409-AuxPrenorm] Error: {_apn_e}")

                # ===== ATT465-MLSA: Multi-Level Spec-Text Alignment Loss =====
                if _mlsa_active and text_prototypes is not None:
                    _z_mid = getattr(spec_encoder, '_last_z_mid', None)
                    if _z_mid is not None and torch.is_tensor(_z_mid) and _z_mid.requires_grad:
                        _mlsa_warmup_frac = min(1.0, max(0.0, float(current_epoch) / max(1, _mlsa_warmup_epochs)))
                        _mlsa_w_eff = _mlsa_weight * _mlsa_warmup_frac
                        if _mlsa_w_eff > 0.0:
                            _z_mid_n = F.normalize(_z_mid, dim=-1)
                            _proto_keys = sorted(text_prototypes.keys())
                            _proto_mat = torch.stack([text_prototypes[k] for k in _proto_keys]).to(_z_mid.device)
                            _proto_mat_n = F.normalize(_proto_mat, dim=-1)
                            _cos_sim = torch.mm(_z_mid_n, _proto_mat_n.t())
                            _mlsa_labels_np = labels.detach().cpu().tolist()
                            _mlsa_pos = torch.stack([_cos_sim[i, _mlsa_labels_np[i]] for i in range(len(_mlsa_labels_np))])
                            _mlsa_neg_mask = torch.ones_like(_cos_sim, dtype=torch.bool)
                            for i in range(len(_mlsa_labels_np)):
                                _mlsa_neg_mask[i, _mlsa_labels_np[i]] = False
                            _mlsa_neg_max = _cos_sim.masked_fill(~_mlsa_neg_mask, -1e9).max(dim=-1).values
                            _mlsa_margin_loss = F.relu(_mlsa_margin - (_mlsa_pos - _mlsa_neg_max)).mean()
                            _mlsa_weighted = _mlsa_w_eff * _mlsa_margin_loss
                            loss = loss + _mlsa_weighted
                            loss_dict["mlsa_mid"] = float(_mlsa_weighted.detach().cpu().item())
                            if batch_idx == 0 and current_epoch <= 3:
                                print(f"[ATT465-MLSA] E{current_epoch}B0 loss={float(_mlsa_margin_loss):.4f}, w_eff={_mlsa_w_eff:.4f}, warmup={_mlsa_warmup_frac:.2f}")

                if not _pcgrad_applied:
                    scaler.scale(loss).backward()
                if batch_idx == 0 and current_epoch <= 3:
                    try:
                        _post = []
                        if hasattr(fusion_decoder, 'classifier'):
                            _cls_w = fusion_decoder.classifier.weight
                            _g = _cls_w.grad
                            _post.append(f"cls_w.grad={'None' if _g is None else f'norm={float(_g.norm()):.8f}'}")
                        _any_grad = sum(1 for p in fusion_decoder.parameters() if p.grad is not None)
                        _total_p = sum(1 for _ in fusion_decoder.parameters())
                        _post.append(f"dec_any_grad={_any_grad}/{_total_p}")
                        print(f"[ATT378c-PostBackward][E{current_epoch}B{batch_idx}] {', '.join(_post)}")
                    except Exception as _e:
                        print(f"[ATT378c-PostBackward] diag error: {_e}")
                _ogm_ge_on = bool(getattr(cfg, "OGM_GE_ENABLE", False))
                _need_unscale = bool(grad_norm_log_enabled or grad_norm_detail_log_enabled or grad_clip_enabled or _ogm_ge_on)
                if _need_unscale:
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                if grad_norm_log_enabled:
                    try:
                        _grad_norm = float(_compute_global_grad_norm())
                        epoch_grad_norm_sum += _grad_norm
                        epoch_grad_norm_steps += 1
                        if isinstance(loss_dict, dict) and (batch_idx == 0 or ((batch_idx + 1) % grad_norm_log_step_stride == 0)):
                            loss_dict["grad_norm"] = _grad_norm
                    except Exception:
                        pass
                if grad_norm_detail_log_enabled:
                    try:
                        _detail_sampled = (batch_idx == 0 or ((batch_idx + 1) % grad_norm_detail_step_stride == 0))
                        if _detail_sampled:
                            _g_sig, _c_sig = _compute_module_grad_norm(signal_encoder)
                            _g_spec, _c_spec = _compute_module_grad_norm(spec_encoder)
                            _g_fus, _c_fus = _compute_module_grad_norm(fusion_module)
                            _g_dec, _c_dec = _compute_module_grad_norm(fusion_decoder)
                            epoch_grad_norm_signal_sum += _g_sig
                            epoch_grad_norm_spec_sum += _g_spec
                            epoch_grad_norm_fusion_sum += _g_fus
                            epoch_grad_norm_decoder_sum += _g_dec
                            epoch_grad_norm_signal_count_sum += _c_sig
                            epoch_grad_norm_spec_count_sum += _c_spec
                            epoch_grad_norm_fusion_count_sum += _c_fus
                            epoch_grad_norm_decoder_count_sum += _c_dec
                            epoch_grad_norm_detail_steps += 1
                            if _c_dec == 0 and batch_idx == 0:
                                _dec_diag = []
                                try:
                                    _cls_w = fusion_decoder.classifier.weight
                                    _dec_diag.append(f"cls.w.grad={'None' if _cls_w.grad is None else f'{float(_cls_w.grad.norm()):.6f}'}")
                                    _dec_diag.append(f"req_grad={_cls_w.requires_grad}")
                                except Exception:
                                    _dec_diag.append("cls.w=N/A")
                                _dec_diag.append(f"total_req={sum(1 for p in fusion_decoder.parameters() if p.requires_grad)}")
                                _dec_diag.append(f"total_has_grad={sum(1 for p in fusion_decoder.parameters() if p.grad is not None)}")
                                print(f"[ATT378b-DecoderGradDiag][E{current_epoch}B{batch_idx}] decoder_n=0! {', '.join(_dec_diag)}")
                    except Exception:
                        pass
                # ATT461: OGM-GE Gradient Modulation — Task1 (after unscale, before clip/step)
                if _ogm_ge_on and current_epoch >= int(getattr(cfg, "OGM_GE_START_EPOCH", 2)):
                    try:
                        import math as _math_ogm
                        _ogm_alpha = float(getattr(cfg, "OGM_GE_ALPHA", 1.0))
                        _g_sig_sq = sum(
                            p.grad.data.norm() ** 2
                            for p in signal_encoder.parameters() if p.grad is not None
                        )
                        _g_spec_sq = sum(
                            p.grad.data.norm() ** 2
                            for p in spec_encoder.parameters() if p.grad is not None
                        )
                        _g_sig_n = float(_g_sig_sq.sqrt()) if torch.is_tensor(_g_sig_sq) else float(_g_sig_sq ** 0.5)
                        _g_spec_n = float(_g_spec_sq.sqrt()) if torch.is_tensor(_g_spec_sq) else float(_g_spec_sq ** 0.5)
                        _ratio_raw = _g_sig_n / (_g_spec_n + 1e-8)
                        _r_sig = _ratio_raw / (1.0 + _ratio_raw)
                        _r_spec = 1.0 / (1.0 + _ratio_raw)
                        _excess_sig = max(_r_sig - 0.5, 0.0)
                        _excess_spec = max(_r_spec - 0.5, 0.0)
                        _coeff_sig = 1.0 - _math_ogm.tanh(_ogm_alpha * _excess_sig)
                        _coeff_spec = 1.0 - _math_ogm.tanh(_ogm_alpha * _excess_spec)
                        for p in signal_encoder.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(_coeff_sig)
                        for p in spec_encoder.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(_coeff_spec)
                                _ge_std = (1.0 - _coeff_spec ** 2) ** 0.5
                                if _ge_std > 1e-6:
                                    _pnorm = float(p.grad.data.norm())
                                    if _pnorm > 1e-10:
                                        p.grad.data.add_(torch.randn_like(p.grad.data) * (_ge_std * _pnorm * 0.01))
                        if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                            print(f"[ATT461-OGM-GE][E{current_epoch}B{batch_idx}] "
                                  f"g_sig={_g_sig_n:.4f} g_spec={_g_spec_n:.4f} "
                                  f"ratio={_ratio_raw:.3f} "
                                  f"coeff_sig={_coeff_sig:.4f} coeff_spec={_coeff_spec:.4f}")
                    except Exception as _ogm_e:
                        if batch_idx == 0:
                            print(f"[ATT461-OGM-GE] Error: {_ogm_e}")
                if grad_clip_enabled:
                    try:
                        _clip_params = list(_iter_optimizer_grad_params(optimizer))
                        if len(_clip_params) > 0:
                            _pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=float(grad_clip_max_norm)))
                            _post_clip_norm = float(_compute_global_grad_norm())
                            epoch_grad_clip_batches += 1
                            epoch_grad_clip_pre_norm_sum += _pre_clip_norm
                            epoch_grad_clip_post_norm_sum += _post_clip_norm
                            epoch_grad_clip_param_count_sum += int(len(_clip_params))
                            if _pre_clip_norm > float(grad_clip_max_norm) + 1e-8:
                                epoch_grad_clip_clipped_batches += 1
                            if isinstance(loss_dict, dict) and (batch_idx == 0 or ((batch_idx + 1) % grad_norm_log_step_stride == 0)):
                                loss_dict["grad_clip_pre_norm"] = _pre_clip_norm
                                loss_dict["grad_clip_post_norm"] = _post_clip_norm
                    except Exception:
                        pass
                # ATT397: SAM two-step optimization
                _is_sam = hasattr(optimizer, 'first_step') and hasattr(optimizer, 'second_step')
                if _is_sam:
                    if not _need_unscale:
                        try:
                            scaler.unscale_(optimizer)
                        except Exception:
                            pass
                    optimizer.first_step(zero_grad=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        _sam_z_sig = signal_encoder(signal, signal_mask, domain_ids=domain_ids) if (
                            hasattr(signal_encoder, 'use_domain_residual') and signal_encoder.use_domain_residual
                        ) else signal_encoder(signal, signal_mask)
                        if isinstance(_sam_z_sig, tuple):
                            _sam_z_sig = _sam_z_sig[0]
                        _sam_z_spec_pair = spec_encoder(spec, spec_mask, domain_ids=domain_ids) if (
                            hasattr(spec_encoder, 'use_domain_residual') and spec_encoder.use_domain_residual
                        ) else spec_encoder(spec, spec_mask)
                        _sam_z_spec = _sam_z_spec_pair[0] if isinstance(_sam_z_spec_pair, tuple) else _sam_z_spec_pair
                        _sam_sig_feat = _sam_z_sig
                        _sam_spec_feat = _sam_z_spec
                        if hasattr(fusion_module, 'domain_embedding'):
                            _sam_z_fus = fusion_module(_sam_sig_feat, _sam_spec_feat, domain_ids=domain_ids)
                        else:
                            _sam_z_fus = fusion_module(_sam_sig_feat, _sam_spec_feat)
                        _sam_logits, _ = fusion_decoder(_sam_z_fus, domain_ids=domain_ids, return_parts=True)
                        _sam_loss = F.cross_entropy(_sam_logits, labels)
                    scaler.scale(_sam_loss).backward()
                    optimizer.restore_weights()
                    scaler.unscale_(optimizer.base_optimizer)
                    if grad_clip_enabled:
                        _clip_params = list(_iter_optimizer_grad_params(optimizer))
                        if len(_clip_params) > 0:
                            torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=float(grad_clip_max_norm))
                    scaler.step(optimizer.base_optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    if batch_idx == 0 and current_epoch <= 3:
                        print(f"[ATT397-SAM] E{current_epoch}B{batch_idx} sam_loss={float(_sam_loss):.4f}")
                else:
                    scaler.step(optimizer)
                    scaler.update()
                
                # 更新 EMA 权重（每步更新）
                if use_ema and ema_manager is not None:
                    global_step += 1
                    ema_manager.update_all(global_step, {
                        'signal_encoder': signal_encoder,
                        'spec_encoder': spec_encoder,
                        'fusion_module': fusion_module,
                        'fusion_decoder': fusion_decoder
                    })
                
                # 收集类别预测统计（用于自适应权重调整）- AMP分支
                if enable_class_monitoring:
                    import config as cfg
                    use_adaptive_weights = getattr(cfg, 'USE_ADAPTIVE_CLASS_WEIGHTS', True)
                    
                    if use_adaptive_weights:
                        with torch.no_grad():
                            # 获取预测结果
                            try:
                                if hasattr(fusion_logits, 'hierarchical_parts'):
                                    # 层次化模型：使用4类logits
                                    preds = torch.argmax(fusion_logits, dim=1)
                                else:
                                    preds = torch.argmax(fusion_logits, dim=1)
                                
                                # 更新每个类别的统计
                                unique_labels = labels.unique()
                                if len(unique_labels) > 0:
                                    for label_val in unique_labels:
                                        label_val = label_val.item()
                                        mask = (labels == label_val)
                                        if mask.sum() > 0:
                                            correct = (preds[mask] == labels[mask]).sum().item()
                                            total = mask.sum().item()
                                            if label_val not in class_stats:
                                                class_stats[label_val] = {"correct": 0, "total": 0}
                                            class_stats[label_val]["correct"] += correct
                                            class_stats[label_val]["total"] += total
                                            
                                            # 更新混淆统计（Confusion-Aware 动态惩罚）
                                            if label_val not in confusion_stats:
                                                confusion_stats[label_val] = {}
                                            preds_batch = preds[mask].cpu().numpy()
                                            for pred_val in preds_batch:
                                                pred_val = int(pred_val)
                                                if pred_val not in confusion_stats[label_val]:
                                                    confusion_stats[label_val][pred_val] = 0
                                                confusion_stats[label_val][pred_val] += 1
                            except Exception as e:
                                # 如果出现错误，记录但不中断训练
                                if current_epoch <= 3 and batch_idx == 0:
                                    print(f"[Adaptive Weights] Warning: Error collecting stats in epoch {current_epoch}, batch {batch_idx}: {e}")
            else:
                # 标准精度训练
                # ===== Pre-initialize z_for_sem to avoid UnboundLocalError in non-Stage2 paths =====
                z_for_sem = z_fusion
                z_for_sem_aug = z_fusion_aug
                
                # ===== (2) Stage1 NF-only: only NF + 3a+3b =====
                stage1_nf_only_now = False
                try:
                    stage1_nf_only_now = bool(locals().get("stage1_mode_now", False)) and bool(getattr(cfg, "STAGE1_NF_ONLY", False))
                except Exception:
                    pass
                if supcon_stage1_now:
                    z_for_supcon = z_fusion
                    # NOTE (stability): do NOT rely on attaching attributes to Tensor (may be lost under AMP/autocast).
                    hp = hierarchical_parts if isinstance(hierarchical_parts, dict) else getattr(fusion_logits, "hierarchical_parts", None)
                    try:
                        if isinstance(hp, dict) and hp.get("z_cls", None) is not None:
                            z_for_supcon = hp.get("z_cls", z_fusion)
                    except Exception:
                        z_for_supcon = z_fusion
                    try:
                        if hasattr(fusion_decoder, "project_for_supcon"):
                            z_for_supcon = fusion_decoder.project_for_supcon(z_for_supcon)
                    except Exception:
                        pass
                    sup_labels = labels
                    if supcon_fault_only:
                        mask = labels > 0
                        z_for_supcon = z_for_supcon[mask]
                        sup_labels = labels[mask]
                    loss_supcon, sup_stats = supervised_contrastive_loss(
                        z_for_supcon,
                        sup_labels,
                        temperature=supcon_temp,
                        use_class_weights=supcon_use_class_weights,
                        return_stats=True
                    )
                    loss = supcon_weight * loss_supcon
                    loss_dict = {"supcon": float(loss_supcon.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                    epoch_supcon_loss += float(loss_supcon.detach().cpu().item())
                    supcon_batches += 1
                    loss_path_supcon_batches += 1
                    # ATT254(S2): enforce variance floor on pre-normalized z_cls to counter
                    # stage1 dimensional collapse before L2 normalization in SupCon pipeline.
                    try:
                        use_supcon_prenorm_var = bool(getattr(cfg, "USE_SUPCON_PRENORM_VAR_LOSS", False))
                        supcon_prenorm_var_w = float(getattr(cfg, "SUPCON_PRENORM_VAR_WEIGHT", 0.0))
                        supcon_prenorm_std_target = float(getattr(cfg, "SUPCON_PRENORM_STD_TARGET", 0.10))
                    except Exception:
                        use_supcon_prenorm_var = False
                        supcon_prenorm_var_w = 0.0
                        supcon_prenorm_std_target = 0.10
                    if use_supcon_prenorm_var and supcon_prenorm_var_w > 0.0 and isinstance(hp, dict):
                        z_cls_pre_norm = hp.get("z_cls_pre_norm", None)
                        if torch.is_tensor(z_cls_pre_norm):
                            if supcon_fault_only:
                                _pre_mask = labels > 0
                                z_cls_pre_norm = z_cls_pre_norm[_pre_mask]
                            if int(z_cls_pre_norm.size(0)) >= 2:
                                z_pre = z_cls_pre_norm.float()
                                z_pre = z_pre - z_pre.mean(dim=0, keepdim=True)
                                std_pre = torch.sqrt(z_pre.var(dim=0, unbiased=False) + 1e-6)
                                l_pre_var = F.relu(float(supcon_prenorm_std_target) - std_pre).mean()
                                loss = loss + float(supcon_prenorm_var_w) * l_pre_var
                                loss_dict["supcon_pre_var"] = float((float(supcon_prenorm_var_w) * l_pre_var).detach().cpu().item())
                                loss_dict["total"] = float(loss.detach().cpu().item())
                                global_decorr_pre_std_sum += float(std_pre.mean().detach().cpu().item())
                                global_decorr_pre_std_batches += 1
                    if float(supcon_spec_weight) > 0.0:
                        z_spec_supcon = z_spec
                        try:
                            if hasattr(spec_encoder, "project_for_supcon"):
                                z_spec_supcon = spec_encoder.project_for_supcon(z_spec_supcon)
                        except Exception:
                            pass
                        spec_labels = labels
                        if supcon_fault_only:
                            spec_mask = labels > 0
                            z_spec_supcon = z_spec_supcon[spec_mask]
                            spec_labels = spec_labels[spec_mask]
                        loss_supcon_spec, _ = supervised_contrastive_loss(
                            z_spec_supcon,
                            spec_labels,
                            temperature=supcon_temp,
                            use_class_weights=supcon_use_class_weights,
                            return_stats=True
                        )
                        loss = loss + float(supcon_spec_weight) * loss_supcon_spec
                        loss_dict["supcon_spec"] = float(loss_supcon_spec.detach().cpu().item())
                        loss_dict["total"] = float(loss.detach().cpu().item())
                    try:
                        supcon_pos_pairs += int(sup_stats.get("pos_pairs", 0))
                        supcon_valid_anchors += int(sup_stats.get("valid_anchors", 0))
                    except Exception:
                        pass
                    # ===== (Attempt 28 Fix) Add lightweight CE loss in Stage1 to train classifier =====
                    supcon_ce_w = float(getattr(cfg, "SUPCON_STAGE1_CE_WEIGHT", 0.3))
                    if supcon_ce_w > 0:
                        if not isinstance(hp, dict):
                            stage1_ce_missing_hp_batches += 1
                        else:
                            diag_logits = hp.get("logits_diag_supervised", None)
                            if diag_logits is None:
                                diag_logits = hp.get("logits_diag", None)
                            if diag_logits is None or (not torch.is_tensor(diag_logits)) or diag_logits.dim() != 2 or diag_logits.size(1) < 4:
                                stage1_ce_missing_logits_batches += 1
                            else:
                                stage1_ce_weights = None
                                try:
                                    if bool(getattr(cfg, "STAGE1_CE_USE_CLASS_WEIGHTS", False)) and torch.is_tensor(class_weights):
                                        if int(class_weights.numel()) == int(diag_logits.size(1)):
                                            stage1_ce_weights = class_weights.to(device=diag_logits.device, dtype=torch.float32)
                                except Exception:
                                    stage1_ce_weights = None
                                stage1_use_focal = bool(getattr(cfg, "STAGE1_CE_USE_FOCAL", False))
                                if core_loss_only_mode:
                                    stage1_use_focal = False
                                stage1_focal_gamma = float(getattr(cfg, "STAGE1_CE_FOCAL_GAMMA", 2.0))
                                # Att77: Focal延迟启动，防止早期Signal欠训练
                                focal_start_epoch = int(getattr(cfg, "FOCAL_START_EPOCH", 1))
                                if stage1_use_focal and current_epoch < focal_start_epoch:
                                    stage1_use_focal = False  # 在达到启动epoch前使用普通CE
                                stage1_focal_use_class_alpha = bool(
                                    getattr(cfg, "STAGE1_CE_FOCAL_USE_CLASS_ALPHA", True)
                                )
                                stage1_loss_weights = (
                                    stage1_ce_weights
                                    if (not stage1_use_focal or stage1_focal_use_class_alpha)
                                    else None
                                )
                                loss_ce_stage1 = compute_diag_loss_with_focal(
                                    logits=diag_logits.float(),
                                    labels=labels,
                                    class_weight_tensor=stage1_loss_weights,
                                    use_focal_loss=stage1_use_focal,
                                    focal_gamma=stage1_focal_gamma,
                                    focal_alpha=None,
                                    label_smoothing=0.0,
                                    use_balanced_softmax=use_balanced_softmax,
                                    class_counts=class_counts_tensor,
                                )
                                loss = loss + supcon_ce_w * loss_ce_stage1
                                loss_dict["stage1_ce"] = float(loss_ce_stage1.detach().cpu().item())
                                loss_dict["_info_fusion_cls"] = float(loss_ce_stage1.detach().cpu().item())  # ATT374: _info_ prefix
                                loss_dict["total"] = float(loss.detach().cpu().item())
                                # Accumulate Stage1 CE statistics
                                epoch_stage1_ce_loss += float(loss_ce_stage1.detach().cpu().item())
                                stage1_ce_batches += 1
                    # ATT141: Reverted ATT139 validation semantic CE injection (symmetric with training revert).
                else:
                    parts_now = getattr(fusion_logits, "hierarchical_parts", None)
                    if stage1_nf_only_now and isinstance(parts_now, dict):
                        logits_nf_global = parts_now.get("logits_nf_global", None)
                        z_gate = parts_now.get("z_gate", None)
                        if logits_nf_global is not None and logits_nf_global.dim() == 2 and logits_nf_global.size(1) == 2:
                            labels_nf = (labels > 0).long()
                            pos_w = parts_now.get("nf_pos_weight_global", None)
                            pw = float(pos_w) if pos_w is not None else 1.0
                            w_nf = torch.tensor([1.0, max(1.0, pw)], device=logits_nf_global.device, dtype=torch.float32)
                            loss_nf_ce = F.cross_entropy(logits_nf_global.float(), labels_nf, weight=w_nf)
                            nf_w_prim = float(getattr(cfg, "STAGE_NF_LOSS_WEIGHT", 0.10))
                            loss = loss_nf_ce * nf_w_prim
                            loss_dict = {"nf_global": float(loss.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                            if z_gate is not None and getattr(cfg, "USE_NF_CONTRASTIVE_LOSS", True):
                                w_c = float(getattr(cfg, "NF_CONTRASTIVE_WEIGHT", 0.15))
                                tau = float(getattr(cfg, "NF_CONTRASTIVE_TEMPERATURE", 0.07))
                                l_c = nf_contrastive_loss(z_gate, labels_nf, temperature=tau)
                                loss = loss + w_c * l_c
                                loss_dict["nf_contrastive"] = float((w_c * l_c).detach().cpu().item())
                            if getattr(cfg, "USE_NF_MARGIN_LOSS", True) and current_epoch >= int(getattr(cfg, "NF_MARGIN_START_EPOCH", 1)):
                                w_m = float(getattr(cfg, "NF_MARGIN_WEIGHT", 0.10))
                                margin = float(getattr(cfg, "NF_MARGIN", 0.25))
                                l_m = nf_margin_loss(logits_nf_global, labels_nf, margin=margin)
                                loss = loss + w_m * l_m
                                loss_dict["nf_margin"] = float((w_m * l_m).detach().cpu().item())
                            if z_gate is not None and getattr(cfg, "USE_NORMAL_REFERENCE_LOSS", True) and hasattr(fusion_decoder, "normal_reference"):
                                ref = fusion_decoder.normal_reference
                                if ref is not None and ref.numel() == z_gate.size(-1):
                                    w_ref = float(getattr(cfg, "NORMAL_REFERENCE_WEIGHT", 0.10))
                                    l_ref = normal_reference_loss(z_gate, labels_nf, ref, weight_fault_push=1.0)
                                    loss = loss + w_ref * l_ref
                                    loss_dict["normal_ref"] = float((w_ref * l_ref).detach().cpu().item())
                            loss_dict["total"] = float(loss.detach().cpu().item())
                        else:
                            # Fallback: if nf logits missing, avoid zero-grad crash.
                            try:
                                allowed = None
                                if isinstance(parts_now, dict):
                                    allowed = parts_now.get("allowed_diag_mask", None)
                                loss = fusion_ce_loss(
                                    fusion_logits,
                                    labels,
                                    class_weights_for_loss,
                                    allowed_mask=allowed,
                                    use_focal_loss=use_focal_loss,
                                    focal_gamma=focal_gamma,
                                    focal_alpha=focal_alpha,
                                    label_smoothing=label_smoothing,
                                    use_balanced_softmax=use_balanced_softmax,
                                    class_counts=class_counts_tensor,
                                )
                                loss_dict = {
                                    "diag_ce": float(loss.detach().cpu().item()),
                                    "total": float(loss.detach().cpu().item()),
                                    "_info_fusion_cls": float(loss.detach().cpu().item()),
                                    "_info_nf_global_missing": 1.0,
                                }
                            except Exception:
                                loss = z_fusion.sum() * 0.0
                                loss_dict = {"total": 0.0, "_info_nf_global_missing": 1.0}
                    elif (current_prototypes is None) or force_ce_path_now:
                        loss_path_ce_batches += 1
                        use_two_stage_fault = bool(getattr(cfg, "USE_TWO_STAGE_FAULT_LOSS", False))
                        if core_loss_only_mode:
                            use_two_stage_fault = False
                        hp = getattr(fusion_logits, "hierarchical_parts", None)
                        logits_3way = hp.get("logits_3way", None) if isinstance(hp, dict) else None
                        logits_nf_global = hp.get("logits_nf_global", None) if isinstance(hp, dict) else None
                        if use_two_stage_fault and logits_3way is not None and logits_nf_global is not None:
                            labels_nf = (labels > 0).long()
                            loss_nf = F.cross_entropy(logits_nf_global.float(), labels_nf, reduction="mean")
                            fault_mask = labels > 0
                            if fault_mask.any():
                                loss_3way = F.cross_entropy(
                                    logits_3way[fault_mask].float(),
                                    (labels[fault_mask] - 1).long(),
                                    reduction="mean",
                                )
                            else:
                                loss_3way = torch.tensor(0.0, device=logits_3way.device, dtype=logits_3way.dtype)
                            w_3way = float(getattr(cfg, "TWO_STAGE_FAULT_LOSS_WEIGHT", 1.0))
                            loss = loss_nf + w_3way * loss_3way
                            loss_dict = {
                                "diag_ce": float(loss.detach().cpu().item()),
                                "total": float(loss.detach().cpu().item()),
                                "_info_fusion_cls": float(loss.detach().cpu().item()),
                                "nf_ce": float(loss_nf.detach().cpu().item()),
                                "fault_3way_ce": float(loss_3way.detach().cpu().item()),
                            }
                        else:
                            allowed = None
                            try:
                                if isinstance(hp, dict):
                                    allowed = hp.get("allowed_diag_mask", None)
                            except Exception:
                                allowed = None
                            loss = fusion_ce_loss(
                                fusion_logits,
                                labels,
                                class_weights_for_loss,
                                allowed_mask=allowed,
                                use_focal_loss=use_focal_loss,
                                focal_gamma=focal_gamma,
                                focal_alpha=focal_alpha,
                                label_smoothing=label_smoothing,
                                use_balanced_softmax=use_balanced_softmax,
                                class_counts=class_counts_tensor,
                            )
                            loss_dict = {"diag_ce": float(loss.detach().cpu().item()), "total": float(loss.detach().cpu().item())}
                            loss_dict["_info_fusion_cls"] = loss_dict.get("diag_ce", 0.0)  # ATT374: fix budget double-count
                        # ATT177: 回退 ATT176 — 非 AMP 分支 Stage1 不加入 SpecAlign
                    else:
                        z_for_sem = z_fusion
                        z_for_sem_aug = z_fusion_aug
                        try:
                            _use_dual_space_now = bool(getattr(cfg, "USE_DUAL_SPACE_DECOUPLE", False))
                            _sem_metric_live_now = bool(
                                getattr(cfg, "FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH", True)
                            )
                            _bypass_sem_adapter_now = bool(
                                getattr(cfg, "FUSION_SEMANTIC_BYPASS_SEM_ADAPTER_ON_DIRECT_PATH", True)
                            )
                            if isinstance(hierarchical_parts, dict):
                                _z_metric_now = hierarchical_parts.get("z_metric", None)
                                if _use_dual_space_now and _sem_metric_live_now and hasattr(fusion_decoder, "get_metric_embedding"):
                                    # ATT480: same as AMP path, keep proto semantic gradient flowing to fusion.
                                    z_for_sem = fusion_decoder.get_metric_embedding(z_fusion)
                                elif _use_dual_space_now and torch.is_tensor(_z_metric_now):
                                    z_for_sem = _z_metric_now
                                if _use_dual_space_now and z_fusion_aug is not None and hasattr(fusion_decoder, "get_metric_embedding"):
                                    z_for_sem_aug = fusion_decoder.get_metric_embedding(z_fusion_aug)
                            _skip_sem_adapter_now = bool(
                                _use_dual_space_now and _sem_metric_live_now and _bypass_sem_adapter_now
                            )
                            if (not _skip_sem_adapter_now) and hasattr(fusion_decoder, "get_sem_embedding"):
                                z_for_sem = fusion_decoder.get_sem_embedding(z_for_sem)
                                if z_fusion_aug is not None:
                                    z_for_sem_aug = fusion_decoder.get_sem_embedding(z_for_sem_aug if z_for_sem_aug is not None else z_fusion_aug)
                        except Exception:
                            z_for_sem = z_fusion
                            z_for_sem_aug = z_fusion_aug
                        # (Attempt 11) Same as training: avoid vote-dilution CE inside alignment loss when enabled.
                        skip_stage_aux_nf = False
                        w_fusion_cls_for_align = current_w_fusion_cls
                        use_two_stage_fault_stage2 = False
                        hp_stage2 = getattr(fusion_logits, "hierarchical_parts", None)
                        logits_3way_stage2 = hp_stage2.get("logits_3way", None) if isinstance(hp_stage2, dict) else None
                        logits_nf_global_stage2 = hp_stage2.get("logits_nf_global", None) if isinstance(hp_stage2, dict) else None
                        try:
                            use_two_stage_fault_stage2 = bool(getattr(cfg, "USE_TWO_STAGE_FAULT_LOSS", False))
                        except Exception:
                            use_two_stage_fault_stage2 = False
                        if core_loss_only_mode:
                            use_two_stage_fault_stage2 = False
                        if use_two_stage_fault_stage2 and logits_3way_stage2 is not None and logits_nf_global_stage2 is not None:
                            w_fusion_cls_for_align = 0.0
                            skip_stage_aux_nf = True

                        # ATT379c: validation uses same decoupled-stage weights as training
                        _vdc_w_fusion_cls = w_fusion_cls_for_align
                        _vdc_w_spec = current_w_spec_align if (core_loss_only_mode and "aux_spec" in set(getattr(cfg, "CORE_LOSS_ALLOWED_TERMS", ()))) else (0.0 if core_loss_only_mode else current_w_spec_align)
                        # ATT431: restore signal CE for signal encoder supervision
                        # ATT463: split_arch increases signal weight
                        _vdc_w_signal = float(getattr(cfg, "SPLIT_ARCH_SIGNAL_WEIGHT", 1.0)) if _split_arch_on else float(getattr(cfg, "W_SIGNAL_ALIGN", 0.3))
                        _vdc_use_focal = False if core_loss_only_mode else use_focal_loss
                        _vdc_margin_override = False
                        if _decoupled_is_stage1:
                            _vdc_w_fusion_cls = 0.0
                            _vdc_w_spec = 0.0
                            _vdc_w_signal = 0.0
                            _vdc_use_focal = False
                        elif _decoupled_is_stage2:
                            _vdc_w_fusion_cls = float(getattr(cfg, "W_FUSION_CLS", 0.5))
                            _vdc_w_spec = 0.0
                            _vdc_w_signal = 0.0
                            _vdc_margin_override = bool(getattr(cfg, "DECOUPLED_STAGE2_ENABLE_MARGINS", True))
                        _att488_skip_zcls_ce_batch = _fusion_zcls_ce_disabled_now(
                            cfg, locals().get("stage1_mode_now", stage1_mode)
                        )
                        if _att488_skip_zcls_ce_batch and not _decoupled_is_stage2:
                            _vdc_w_fusion_cls = 0.0

                        loss_path_align_batches += 1
                        _att479_fus_sc_b = bool(getattr(cfg, "FUSION_SUPCON_ENABLE", False))
                        loss, loss_dict = multimodal_alignment_loss(
                            z_signal=z_signal,
                            z_spec=z_spec_clean,
                            z_fusion=z_for_sem,
                            z_fusion_raw=z_for_sem,
                            fusion_logits=fusion_logits,
                            hierarchical_parts=hierarchical_parts,
                            labels=labels,
                            text_prototypes=current_prototypes,
                            temperature=float(getattr(cfg, "SEMANTIC_ALIGNMENT_TEMPERATURE", 0.07)),
                            class_weights=class_weights_for_loss,
                            z_fusion_aug=z_for_sem_aug,
                            augment_indices=augment_indices,
                            w_signal=_vdc_w_signal,
                            w_margin=0.0,  # ATT428(S1): margin loss disabled
                            w_consistency=0.0 if core_loss_only_mode else (consistency_weight if use_consistency_regularization else 0.0),
                            consistency_temperature=consistency_temperature,
                            w_fusion_cls=_vdc_w_fusion_cls,
                            w_spec=_vdc_w_spec,
                            use_focal_loss=_vdc_use_focal,
                            focal_gamma=focal_gamma,
                            focal_alpha=focal_alpha,
                            label_smoothing=label_smoothing,
                            adaptive_pos_weight_global=adaptive_pos_weight_global,
                            adaptive_pos_weight_multi=adaptive_pos_weight_multi,
                            confusion_penalty_weights=confusion_penalty_weights,
                            use_supcon_loss=_supcon_core_pass and getattr(cfg, "USE_SUPCON_LOSS", True),
                            supcon_weight=getattr(cfg, "SUPCON_WEIGHT", 0.05) if _supcon_core_pass else 0.0,
                            supcon_temperature=getattr(cfg, "SUPCON_TEMPERATURE", 0.07),
                            supcon_fault_only=getattr(cfg, "SUPCON_FAULT_ONLY", True) if _supcon_core_pass else False,
                            use_fusion_supcon_loss=_att479_fus_sc_b,
                            fusion_supcon_weight=float(getattr(cfg, "FUSION_SUPCON_WEIGHT", 1.0)) if _att479_fus_sc_b else 0.0,
                            fusion_supcon_temperature=float(getattr(cfg, "FUSION_SUPCON_TEMPERATURE", 0.07)),
                            fusion_supcon_fault_only=bool(getattr(cfg, "FUSION_SUPCON_FAULT_ONLY", False)),
                            z_fusion_supcon=(z_fusion_for_reg if _att479_fus_sc_b else None),
                            fusion_geometry_loss_mode=str(getattr(cfg, "FUSION_GEOMETRY_LOSS_MODE", "silhouette")),
                            fusion_silhouette_eps=float(getattr(cfg, "FUSION_SILHOUETTE_EPS", 1e-6)),
                            use_proto_sep_loss=False if (core_loss_only_mode and not _vdc_margin_override) else getattr(cfg, "USE_PROTO_SEP_LOSS", True),
                            proto_sep_weight=0.0 if (core_loss_only_mode and not _vdc_margin_override) else getattr(cfg, "PROTO_SEP_WEIGHT", 0.02),
                            use_outer_margin_loss=False if (core_loss_only_mode and not _vdc_margin_override) else getattr(cfg, "USE_OUTER_MARGIN_LOSS", True),
                            outer_margin_weight=0.0 if (core_loss_only_mode and not _vdc_margin_override) else getattr(cfg, "OUTER_MARGIN_WEIGHT", 0.05),
                            outer_margin=getattr(cfg, "OUTER_MARGIN", 0.2),
                            outer_margin_start_epoch=1 if _vdc_margin_override else getattr(cfg, "OUTER_MARGIN_START_EPOCH", 5),
                            use_inner_margin_loss=False if (core_loss_only_mode and not _vdc_margin_override) else getattr(cfg, "USE_INNER_MARGIN_LOSS", True),
                            inner_margin_weight=0.0 if (core_loss_only_mode and not _vdc_margin_override) else getattr(cfg, "INNER_MARGIN_WEIGHT", 0.05),
                            inner_margin=getattr(cfg, "INNER_MARGIN", 0.2),
                            inner_margin_start_epoch=1 if _vdc_margin_override else getattr(cfg, "INNER_MARGIN_START_EPOCH", 5),
                            use_balanced_softmax=use_balanced_softmax,
                            class_counts=class_counts_tensor,
                            decoupled_stage2_margin_override=_vdc_margin_override,
                            ablation_gate_bank=_ablation_gate_bank,
                        )
                        if _att488_skip_zcls_ce_batch and not _decoupled_is_stage2:
                            loss_dict = _mark_fusion_zcls_ce_skipped(loss_dict)

                        loss, loss_dict = _maybe_add_fusion_metric_learnable_proto_ce(
                            loss,
                            loss_dict,
                            cfg,
                            learnable_proto_module,
                            hierarchical_parts,
                            labels,
                            class_weights_for_loss,
                            current_epoch,
                            batch_idx,
                        )

                        if use_two_stage_fault_stage2 and logits_3way_stage2 is not None and logits_nf_global_stage2 is not None:
                            labels_nf = (labels > 0).long()
                            loss_nf = F.cross_entropy(logits_nf_global_stage2.float(), labels_nf, reduction="mean")
                            fault_mask = labels > 0
                            if fault_mask.any():
                                loss_3way = F.cross_entropy(
                                    logits_3way_stage2[fault_mask].float(),
                                    (labels[fault_mask] - 1).long(),
                                    reduction="mean",
                                )
                            else:
                                loss_3way = torch.tensor(0.0, device=logits_3way_stage2.device, dtype=logits_3way_stage2.dtype)
                            w_3way = float(getattr(cfg, "TWO_STAGE_FAULT_LOSS_WEIGHT", 1.0))
                            loss_diag = loss_nf + w_3way * loss_3way
                            loss = loss + loss_diag
                            try:
                                loss_dict["diag_ce"] = float(loss_diag.detach().cpu().item())
                                loss_dict["_info_fusion_cls"] = float(loss_diag.detach().cpu().item())  # ATT374: _info_ prefix
                                loss_dict["nf_ce"] = float(loss_nf.detach().cpu().item())
                                loss_dict["fault_3way_ce"] = float(loss_3way.detach().cpu().item())
                            except Exception:
                                pass

                        # ===== (Attempt 35) IO↔Rolling separation loss & centroid logging (Stage1 only, Non-AMP branch) =====
                        try:
                            import config as _cfg
                            use_sep = bool(getattr(_cfg, "USE_IO_ROLL_SEP_LOSS", False))
                            sep_w = float(getattr(_cfg, "IO_ROLL_SEP_WEIGHT", 0.0))
                            sep_max_sim = float(getattr(_cfg, "IO_ROLL_SEP_MAX_SIM", 0.35))
                            sep_start = int(getattr(_cfg, "IO_ROLL_SEP_START_EPOCH", 1))
                            log_centroid = bool(getattr(_cfg, "LOG_EMBED_CENTROID_SIM", False))
                            # Attempt 32: Inner↔Outer separation (primary fix for IO collapse)
                            use_io_sep = bool(getattr(_cfg, "USE_IO_SEP_LOSS", False))
                            io_sep_w = float(getattr(_cfg, "IO_SEP_WEIGHT", 0.0))
                            io_sep_max_sim = float(getattr(_cfg, "IO_SEP_MAX_SIM", 0.30))
                            io_sep_start = int(getattr(_cfg, "IO_SEP_START_EPOCH", 1))
                            # Attempt 33: Normal↔Fault separation (primary fix for Normal absorption)
                            use_nf_sep = bool(getattr(_cfg, "USE_NF_SEP_LOSS", False))
                            nf_sep_w = float(getattr(_cfg, "NF_SEP_WEIGHT", 0.0))
                            nf_sep_max_sim = float(getattr(_cfg, "NF_SEP_MAX_SIM", 0.20))
                            nf_sep_start = int(getattr(_cfg, "NF_SEP_START_EPOCH", 1))
                            # Att78: small-sample regime — disable NF-Sep when batches per epoch <= threshold (high-variance centroids).
                            nf_sep_min_batches = int(getattr(_cfg, "NF_SEP_MIN_BATCHES", 5))
                            if len(train_loader) < nf_sep_min_batches:
                                nf_sep_w = 0.0
                            # Attempt 35: Inner↔Rolling separation (fix gradient vanishing + relax threshold)
                            use_ir_sep = bool(getattr(_cfg, "USE_IR_SEP_LOSS", False))
                            ir_sep_w = float(getattr(_cfg, "IR_SEP_WEIGHT", 0.0))
                            ir_sep_max_sim = float(getattr(_cfg, "IR_SEP_MAX_SIM", 0.15))  # relaxed from 0.0
                            ir_sep_start = int(getattr(_cfg, "IR_SEP_START_EPOCH", 1))
                            ir_sep_maint_w = float(getattr(_cfg, "IR_SEP_MAINTENANCE_WEIGHT", 0.01))  # NEW
                        except Exception:
                            use_sep, sep_w, sep_max_sim, sep_start, log_centroid = False, 0.0, 0.35, 1, False
                            use_io_sep, io_sep_w, io_sep_max_sim, io_sep_start = False, 0.0, 0.30, 1
                            use_nf_sep, nf_sep_w, nf_sep_max_sim, nf_sep_start = False, 0.0, 0.20, 1
                            use_ir_sep, ir_sep_w, ir_sep_max_sim, ir_sep_start = False, 0.0, 0.15, 1
                            ir_sep_maint_w = 0.01  # default maintenance weight
                        if core_loss_only_mode:
                            use_sep = use_io_sep = use_nf_sep = use_ir_sep = False
                            sep_w = io_sep_w = nf_sep_w = ir_sep_w = 0.0

                        try:
                            z_sep_full = None
                            if isinstance(hp, dict):
                                z_sep_full = hp.get("z_cls", None)
                            if z_sep_full is None:
                                z_sep_full = z_fusion
                            if torch.is_tensor(z_sep_full) and z_sep_full.dim() == 2 and labels is not None and labels.dim() == 1:
                                z_sep_full = F.normalize(z_sep_full.float(), dim=-1, eps=1e-6)

                                # Epoch-level centroid accumulation (for diagnostics)
                                if (log_centroid or semantic_proto_source == "signal_centroid") and not (semantic_proto_source == "signal_centroid" and semantic_centroid_space == "metric"):
                                    if embed_centroid_sum is None:
                                        embed_centroid_sum = torch.zeros((4, int(z_sep_full.size(1))), device=z_sep_full.device, dtype=torch.float32)
                                    for _c in range(4):
                                        _m = (labels == int(_c))
                                        if bool(_m.any()):
                                            embed_centroid_sum[_c] += z_sep_full[_m].sum(dim=0)
                                            embed_centroid_cnt[_c] += int(_m.sum().item())

                                # Targeted separation: penalize high cosine sim for specific centroid pairs
                                need_mu = (
                                    (use_sep and (sep_w > 0.0) and (current_epoch >= int(sep_start)))
                                    or (use_io_sep and (io_sep_w > 0.0) and (current_epoch >= int(io_sep_start)))
                                    or (use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start)))
                                )
                                if need_mu:
                                    mu0 = mu1 = mu2 = mu3 = mu3_diag = None
                                    m0 = (labels == 0)
                                    m1 = (labels == 1)
                                    m2 = (labels == 2)
                                    m3 = (labels == 3)
                                    if (use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start))) and bool(m0.any()):
                                        mu0 = F.normalize(z_sep_full[m0].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    if bool(m1.any()):
                                        mu1 = F.normalize(z_sep_full[m1].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    if bool(m2.any()):
                                        mu2 = F.normalize(z_sep_full[m2].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    if (
                                        (use_sep and (sep_w > 0.0) and (current_epoch >= int(sep_start)))
                                        and bool(m3.any())
                                    ):
                                        mu3 = F.normalize(z_sep_full[m3].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
                                    elif (use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start))) and bool(m3.any()):
                                        mu3_diag = F.normalize(z_sep_full[m3].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)

                                    updated = False
                                    # (A) IO↔Rolling separation loss
                                    if use_sep and (sep_w > 0.0) and (current_epoch >= int(sep_start)):
                                        sep_loss = None
                                        loss_terms = []
                                        if (mu1 is not None) and (mu3 is not None):
                                            sim13 = float((mu1 * mu3).sum().detach().cpu().item())
                                            io_roll_sep_sim13_sum += sim13
                                            io_roll_sep_sim13_cnt += 1
                                            loss_terms.append(F.relu((mu1 * mu3).sum() - float(sep_max_sim)))
                                        if (mu2 is not None) and (mu3 is not None):
                                            sim23 = float((mu2 * mu3).sum().detach().cpu().item())
                                            io_roll_sep_sim23_sum += sim23
                                            io_roll_sep_sim23_cnt += 1
                                            loss_terms.append(F.relu((mu2 * mu3).sum() - float(sep_max_sim)))
                                        if len(loss_terms) > 0:
                                            sep_loss = torch.stack(loss_terms).mean()
                                            loss = loss + float(sep_w) * sep_loss
                                            loss_dict["io_roll_sep"] = float((float(sep_w) * sep_loss).detach().cpu().item())
                                            epoch_io_roll_sep_loss += float(sep_loss.detach().cpu().item())
                                            io_roll_sep_batches += 1
                                            updated = True

                                    # (B) Inner↔Outer separation loss (Attempt 32): penalize high cosine sim(Inner, Outer)
                                    if use_io_sep and (io_sep_w > 0.0) and (current_epoch >= int(io_sep_start)):
                                        if (mu1 is not None) and (mu2 is not None):
                                            sim12 = float((mu1 * mu2).sum().detach().cpu().item())
                                            io_sep_sim12_sum += sim12
                                            io_sep_sim12_cnt += 1
                                            io_sep_loss = F.relu((mu1 * mu2).sum() - float(io_sep_max_sim))
                                            loss = loss + float(io_sep_w) * io_sep_loss
                                            loss_dict["io_sep"] = float((float(io_sep_w) * io_sep_loss).detach().cpu().item())
                                            epoch_io_sep_loss += float(io_sep_loss.detach().cpu().item())
                                            io_sep_batches += 1
                                            updated = True

                                    # (C) Normal↔Fault separation loss (Attempt 33): penalize high cosine sim(N, fault_k)
                                    if use_nf_sep and (nf_sep_w > 0.0) and (current_epoch >= int(nf_sep_start)):
                                        if (mu0 is not None) and ((mu1 is not None) or (mu2 is not None) or (mu3 is not None)):
                                            nf_terms = []
                                            if mu1 is not None:
                                                sim01 = float((mu0 * mu1).sum().detach().cpu().item())
                                                nf_sep_sim01_sum += sim01
                                                nf_sep_sim01_cnt += 1
                                                nf_terms.append(F.relu((mu0 * mu1).sum() - float(nf_sep_max_sim)))
                                            if mu2 is not None:
                                                sim02 = float((mu0 * mu2).sum().detach().cpu().item())
                                                nf_sep_sim02_sum += sim02
                                                nf_sep_sim02_cnt += 1
                                                nf_terms.append(F.relu((mu0 * mu2).sum() - float(nf_sep_max_sim)))
                                            if mu3 is not None:
                                                sim03 = float((mu0 * mu3).sum().detach().cpu().item())
                                                nf_sep_sim03_sum += sim03
                                                nf_sep_sim03_cnt += 1
                                                nf_terms.append(F.relu((mu0 * mu3).sum() - float(nf_sep_max_sim)))
                                            if len(nf_terms) > 0:
                                                nf_loss = torch.stack(nf_terms).mean()
                                                loss = loss + float(nf_sep_w) * nf_loss
                                                loss_dict["nf_sep"] = float((float(nf_sep_w) * nf_loss).detach().cpu().item())
                                                epoch_nf_sep_loss += float(nf_loss.detach().cpu().item())
                                                nf_sep_batches += 1
                                                updated = True

                                    # Diagnostic-only cos_sim(N,R): do not add to loss (Att70: mu3 in loss causes high-variance gradient)
                                    if mu0 is not None and mu3_diag is not None:
                                        sim03_diag = float((mu0 * mu3_diag).sum().detach().cpu().item())
                                        nf_sep_sim03_sum += sim03_diag
                                        nf_sep_sim03_cnt += 1

                                    # (D) Inner↔Rolling separation loss (Attempt 35): penalize high cosine sim(I, R)
                                    # Fixed: Add maintenance gradient to prevent hinge loss gradient vanishing
                                    if use_ir_sep and (ir_sep_w > 0.0) and (current_epoch >= int(ir_sep_start)):
                                        if (mu1 is not None) and (mu3 is not None):
                                            sim_val = (mu1 * mu3).sum()
                                            sim13 = float(sim_val.detach().cpu().item())
                                            ir_sep_sim13_sum += sim13
                                            ir_sep_sim13_cnt += 1
                                            # Hinge loss: main constraint (zero gradient when target achieved)
                                            ir_hinge = F.relu(sim_val - float(ir_sep_max_sim))
                                            # Maintenance gradient: continuous small gradient even after target achieved
                                            ir_maintenance = ir_sep_maint_w * sim_val  # always pulls toward lower sim
                                            ir_loss = ir_hinge + ir_maintenance
                                            loss = loss + float(ir_sep_w) * ir_loss
                                            loss_dict["ir_sep"] = float((float(ir_sep_w) * ir_loss).detach().cpu().item())
                                            loss_dict["ir_hinge"] = float(ir_hinge.detach().cpu().item())
                                            loss_dict["ir_maint"] = float(ir_maintenance.detach().cpu().item())
                                            epoch_ir_sep_loss += float(ir_loss.detach().cpu().item())
                                            ir_sep_batches += 1
                                            updated = True

                                    if updated:
                                        loss_dict["total"] = float(loss.detach().cpu().item())
                        except Exception:
                            pass

                # ATT178: Detached SpecAlign Projection Head (safe Stage1 semantic alignment)
                # 关键设计：z_spec.detach() -> align_proj -> CE(text_prototypes)，
                # 只更新投影头，不影响 SpecEncoder/Fusion 主干梯度。
                try:
                    _use_dsap = bool(getattr(cfg, "USE_DETACHED_SPEC_ALIGN_PROJ", False))
                    _dsap_w = float(getattr(cfg, "DETACHED_SPEC_ALIGN_WEIGHT", 0.0))
                    _dsap_t = float(getattr(cfg, "DETACHED_SPEC_ALIGN_TEMPERATURE", 0.1))
                    _dsap_start = int(getattr(cfg, "DETACHED_SPEC_ALIGN_START_EPOCH", 1))
                    _dsap_use_cw = bool(getattr(cfg, "DETACHED_SPEC_ALIGN_USE_CLASS_WEIGHTS", True))
                except Exception:
                    _use_dsap, _dsap_w, _dsap_t, _dsap_start, _dsap_use_cw = False, 0.0, 0.1, 1, True
                if _use_dsap and _dsap_w > 0.0 and current_epoch >= _dsap_start:
                    try:
                        _z_for_align = locals().get("z_spec_clean", None)
                        if _z_for_align is None:
                            _z_for_align = z_spec
                        if _z_for_align is not None and torch.is_tensor(_z_for_align):
                            _spec_align_attached = bool(getattr(cfg, "SPEC_ALIGN_ATTACHED", False))
                            _detach_input = not _spec_align_attached
                            if hasattr(spec_encoder, "project_for_proto"):
                                _z_proj = spec_encoder.project_for_proto(_z_for_align, detach_input=_detach_input)
                            else:
                                _z_proj = F.normalize(_z_for_align.detach() if _detach_input else _z_for_align, dim=-1)
                            # ATT428-FIX: extract proto tensor; .detach() to avoid inplace error
                            _proto = None
                            if torch.is_tensor(training_proto_tensor):
                                _proto = training_proto_tensor.detach().to(_z_proj.device, dtype=_z_proj.dtype)
                            elif isinstance(current_prototypes, dict):
                                _rows = [current_prototypes[k].detach() for k in sorted(current_prototypes.keys()) if torch.is_tensor(current_prototypes[k])]
                                if _rows:
                                    _proto = torch.stack(_rows, dim=0).to(_z_proj.device, dtype=_z_proj.dtype)
                            elif torch.is_tensor(current_prototypes):
                                _proto = current_prototypes.detach().to(_z_proj.device, dtype=_z_proj.dtype)
                            if _proto is not None and _proto.dim() == 2 and _proto.size(-1) == _z_proj.size(-1):
                                _logits_dsap = torch.matmul(_z_proj, _proto.t()) / max(_dsap_t, 1e-6)
                                _cw = class_weights_for_loss if _dsap_use_cw else None
                                _loss_dsap = F.cross_entropy(_logits_dsap.float(), labels, weight=_cw)
                                loss = loss + float(_dsap_w) * _loss_dsap
                                if isinstance(loss_dict, dict):
                                    loss_dict["detached_spec_align"] = float((float(_dsap_w) * _loss_dsap).detach().cpu().item())
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                            elif batch_idx == 0 and current_epoch <= 2:
                                print(f"[DSAP-WARN] V2: proto extraction failed: type={type(current_prototypes).__name__}")
                    except Exception as _dsap_err:
                        if batch_idx == 0:
                            print(f"[DSAP-ERR] V2: {type(_dsap_err).__name__}: {_dsap_err}")

                # ===== (Stage aux) NF loss (keep AUC_nf stable); skip when Stage1 NF-only already applied =====
                try:
                    import config as cfg
                    nf_w = float(getattr(cfg, "STAGE_NF_LOSS_WEIGHT", 0.10))
                except Exception:
                    nf_w = 0.10
                try:
                    parts_now = getattr(fusion_logits, "hierarchical_parts", None)
                except Exception:
                    parts_now = None
                if (not core_loss_only_mode) and (not stage1_nf_only_now) and (not bool(locals().get("skip_stage_aux_nf", False))) and nf_w > 0.0 and isinstance(parts_now, dict):
                    logits_nf_global = parts_now.get("logits_nf_global", None)
                    if logits_nf_global is not None and logits_nf_global.dim() == 2 and logits_nf_global.size(1) == 2:
                        labels_nf = (labels > 0).long()
                        pos_w = parts_now.get("nf_pos_weight_global", None)
                        try:
                            pw = float(pos_w) if pos_w is not None else 1.0
                        except Exception:
                            pw = 1.0
                        w_nf = torch.tensor([1.0, max(1.0, pw)], device=logits_nf_global.device, dtype=torch.float32)
                        loss_nf = F.cross_entropy(logits_nf_global.float(), labels_nf, weight=w_nf)
                        loss = loss + float(nf_w) * loss_nf
                        loss_dict["nf_global"] = float((float(nf_w) * loss_nf).detach().cpu().item())
                        z_gate = parts_now.get("z_gate", None)
                        if z_gate is not None and getattr(cfg, "USE_NF_CONTRASTIVE_LOSS", True):
                            w_c = float(getattr(cfg, "NF_CONTRASTIVE_WEIGHT", 0.15))
                            tau = float(getattr(cfg, "NF_CONTRASTIVE_TEMPERATURE", 0.07))
                            l_c = nf_contrastive_loss(z_gate, labels_nf, temperature=tau)
                            loss = loss + w_c * l_c
                            loss_dict["nf_contrastive"] = float((w_c * l_c).detach().cpu().item())
                        if getattr(cfg, "USE_NF_MARGIN_LOSS", True) and current_epoch >= int(getattr(cfg, "NF_MARGIN_START_EPOCH", 1)):
                            w_m = float(getattr(cfg, "NF_MARGIN_WEIGHT", 0.10))
                            margin = float(getattr(cfg, "NF_MARGIN", 0.25))
                            l_m = nf_margin_loss(logits_nf_global, labels_nf, margin=margin)
                            loss = loss + w_m * l_m
                            loss_dict["nf_margin"] = float((w_m * l_m).detach().cpu().item())
                        if z_gate is not None and getattr(cfg, "USE_NORMAL_REFERENCE_LOSS", True) and hasattr(fusion_decoder, "normal_reference"):
                            ref = fusion_decoder.normal_reference
                            if ref is not None and ref.numel() == z_gate.size(-1):
                                w_ref = float(getattr(cfg, "NORMAL_REFERENCE_WEIGHT", 0.10))
                                l_ref = normal_reference_loss(z_gate, labels_nf, ref, weight_fault_push=1.0)
                                loss = loss + w_ref * l_ref
                                loss_dict["normal_ref"] = float((w_ref * l_ref).detach().cpu().item())

                # (3b) EMA update normal reference (first epoch can use lower momentum for warm-start)
                if (not core_loss_only_mode) and getattr(cfg, "USE_NORMAL_REFERENCE_LOSS", True) and hasattr(fusion_decoder, "update_normal_reference"):
                    _hp = getattr(fusion_logits, "hierarchical_parts", None)
                    _zg = _hp.get("z_gate", None) if isinstance(_hp, dict) else None
                    if _zg is not None:
                        with torch.no_grad():
                            mom = getattr(cfg, "NORMAL_REFERENCE_MOMENTUM", 0.99)
                            if current_epoch == 1:
                                first_epoch_mom = getattr(cfg, "NORMAL_REFERENCE_FIRST_EPOCH_MOMENTUM", None)
                                if first_epoch_mom is not None:
                                    mom = float(first_epoch_mom)
                            fusion_decoder.update_normal_reference(_zg, labels, momentum=mom)

                # ===== (ATT173) Spec Domain-Invariant Disentangle Losses (Stage2 path) =====
                try:
                    _use_dis2 = bool(getattr(cfg, "USE_SPEC_DISENTANGLE", False))
                    _dis_start2 = int(getattr(cfg, "SPEC_DISENTANGLE_START_EPOCH", 5))
                    _dis_max_alpha2 = float(getattr(cfg, "SPEC_DISENTANGLE_MAX_ALPHA", 0.3))
                    _dis_alpha_epochs2 = int(getattr(cfg, "SPEC_DISENTANGLE_ALPHA_EPOCHS", 10))
                    _w_orth2 = float(getattr(cfg, "SPEC_DISENTANGLE_ORTH_WEIGHT", 0.05))
                    _w_dom_spe2 = float(getattr(cfg, "SPEC_DISENTANGLE_DOMAIN_SPE_WEIGHT", 0.10))
                    _w_dom_adv2 = float(getattr(cfg, "SPEC_DISENTANGLE_DOMAIN_ADV_WEIGHT", 0.05))
                except Exception:
                    _use_dis2 = False; _dis_start2 = 5; _dis_max_alpha2 = 0.3; _dis_alpha_epochs2 = 10
                    _w_orth2 = 0.05; _w_dom_spe2 = 0.10; _w_dom_adv2 = 0.05
                _z_inv_raw2 = getattr(spec_encoder, "_last_z_inv_raw", None)
                _z_spe_raw2 = getattr(spec_encoder, "_last_z_spe_raw", None)
                if (
                    _use_dis2
                    and _z_inv_raw2 is not None
                    and _z_spe_raw2 is not None
                    and domain_ids is not None
                    and current_epoch >= _dis_start2
                ):
                    _zi_ng2 = F.normalize(_z_inv_raw2, dim=0)
                    _zs_ng2 = F.normalize(_z_spe_raw2, dim=0)
                    _loss_orth2 = (_zi_ng2.T @ _zs_ng2).pow(2).sum() / (_zi_ng2.size(1) * _zs_ng2.size(1))
                    if hasattr(spec_encoder, "dc_spe") and spec_encoder.dc_spe is not None:
                        _loss_dom_spe2 = F.cross_entropy(spec_encoder.dc_spe(_z_spe_raw2).float(), domain_ids.long())
                    else:
                        _loss_dom_spe2 = torch.tensor(0.0, device=loss.device)
                    _adv_prog2 = min(1.0, max(0.0, float(current_epoch - _dis_start2) / float(max(1, _dis_alpha_epochs2))))
                    _adv_alpha2 = _dis_max_alpha2 * _adv_prog2
                    if hasattr(spec_encoder, "dc_inv") and spec_encoder.dc_inv is not None and _adv_alpha2 > 0:
                        from models.grl import grad_reverse as _grl2
                        _loss_dom_adv2 = F.cross_entropy(spec_encoder.dc_inv(_grl2(_z_inv_raw2, _adv_alpha2)).float(), domain_ids.long())
                    else:
                        _loss_dom_adv2 = torch.tensor(0.0, device=loss.device)
                    loss = loss + _w_orth2 * _loss_orth2 + _w_dom_spe2 * _loss_dom_spe2 + _w_dom_adv2 * _loss_dom_adv2
                    loss_dict["dis_orth"] = float((_w_orth2 * _loss_orth2).detach().cpu().item())
                    loss_dict["dis_dom_spe"] = float((_w_dom_spe2 * _loss_dom_spe2).detach().cpu().item())
                    loss_dict["dis_dom_adv"] = float((_w_dom_adv2 * _loss_dom_adv2).detach().cpu().item())

                # ===== (DANN) Domain adversarial loss =====
                if isinstance(parts_now, dict):
                    try:
                        dann_lambda = float(parts_now.get("dann_lambda", 0.0))
                    except Exception:
                        dann_lambda = 0.0
                    dom_logits = parts_now.get("domain_logits", None)
                    dom_ids = parts_now.get("domain_ids", None)
                    if (not core_loss_only_mode) and dann_lambda > 0.0 and dom_logits is not None and dom_ids is not None:
                        loss_dann = F.cross_entropy(dom_logits.float(), dom_ids.long())
                        loss = loss + float(dann_lambda) * loss_dann
                        loss_dict["dann"] = float((float(dann_lambda) * loss_dann).detach().cpu().item())

                    # ===== (Rolling-DANN) Domain adversarial on rolling branch =====
                    try:
                        roll_dann_lambda = float(parts_now.get("roll_dann_lambda", 0.0))
                    except Exception:
                        roll_dann_lambda = 0.0
                    roll_dom_logits = parts_now.get("roll_domain_logits", None)
                    try:
                        roll_dann_alpha_now = float(parts_now.get("roll_dann_alpha", 0.0))
                    except Exception:
                        roll_dann_alpha_now = 0.0
                    roll_dann_expected_now = bool((not core_loss_only_mode) and roll_dann_lambda > 0.0 and roll_dann_alpha_now > 0.0)
                    if roll_dann_expected_now:
                        roll_dann_expected_batches += 1
                        if roll_dom_logits is None or dom_ids is None:
                            roll_dann_missing_batches += 1
                        else:
                            roll_mask = (labels > 0)
                            if roll_mask.any():
                                loss_roll_dann = F.cross_entropy(
                                    roll_dom_logits[roll_mask].float(),
                                    dom_ids[roll_mask].long()
                                )
                                loss = loss + float(roll_dann_lambda) * loss_roll_dann
                                loss_dict["roll_dann"] = float((float(roll_dann_lambda) * loss_roll_dann).detach().cpu().item())
                                roll_dann_used_batches += 1
                            else:
                                roll_dann_no_fault_batches += 1

                    # ===== (Rolling De-collapse) variance + decorrelation on rolling branch =====
                    roll_decorr_expected_now = bool(use_roll_decorr and roll_decorr_weight > 0.0 and current_epoch >= int(roll_decorr_start))
                    if roll_decorr_expected_now:
                        roll_decorr_expected_batches += 1
                        z_roll = parts_now.get("z_roll", None)
                        if z_roll is None:
                            roll_decorr_missing_batches += 1
                        else:
                            try:
                                roll_fault_only = bool(getattr(cfg, "ROLLING_TRAIN_ON_FAULT_ONLY", True))
                            except Exception:
                                roll_fault_only = True
                            roll_mask = (labels > 0) if roll_fault_only else torch.ones_like(labels, dtype=torch.bool)
                            if roll_mask.any():
                                if int(roll_mask.sum().item()) >= 2:
                                    z_roll_sub = z_roll[roll_mask]
                                    l_decorr, var_mean, cov_mean = compute_decorr_loss(
                                        z_roll_sub,
                                        var_target=roll_decorr_var_target,
                                        cov_weight=roll_decorr_cov_weight
                                    )
                                    if l_decorr is not None:
                                        loss = loss + float(roll_decorr_weight) * l_decorr
                                        loss_dict["roll_decorr"] = float((float(roll_decorr_weight) * l_decorr).detach().cpu().item())
                                        roll_decorr_used_batches += 1
                                        roll_decorr_var_sum += float(var_mean)
                                        roll_decorr_cov_sum += float(cov_mean)
                                else:
                                    roll_decorr_small_batches += 1
                            else:
                                roll_decorr_no_fault_batches += 1

                    # ===== (Rolling De-collapse Input) variance + decorrelation on z_res =====
                    roll_decorr_in_expected_now = bool(use_roll_decorr_input and roll_decorr_input_weight > 0.0 and current_epoch >= int(roll_decorr_start))
                    if roll_decorr_in_expected_now:
                        roll_decorr_in_expected_batches += 1
                        z_res = parts_now.get("z_res", None)
                        if z_res is None:
                            roll_decorr_in_missing_batches += 1
                        else:
                            try:
                                roll_fault_only = bool(getattr(cfg, "ROLLING_TRAIN_ON_FAULT_ONLY", True))
                            except Exception:
                                roll_fault_only = True
                            roll_mask = (labels > 0) if roll_fault_only else torch.ones_like(labels, dtype=torch.bool)
                            if roll_mask.any():
                                if int(roll_mask.sum().item()) >= 2:
                                    z_res_sub = z_res[roll_mask]
                                    l_decorr_in, var_in, cov_in = compute_decorr_loss(
                                        z_res_sub,
                                        var_target=roll_decorr_var_target,
                                        cov_weight=roll_decorr_cov_weight
                                    )
                                    if l_decorr_in is not None:
                                        loss = loss + float(roll_decorr_input_weight) * l_decorr_in
                                        loss_dict["roll_decorr_in"] = float((float(roll_decorr_input_weight) * l_decorr_in).detach().cpu().item())
                                        roll_decorr_in_used_batches += 1
                                        roll_decorr_in_var_sum += float(var_in)
                                        roll_decorr_in_cov_sum += float(cov_in)
                                else:
                                    roll_decorr_in_small_batches += 1
                            else:
                                roll_decorr_in_no_fault_batches += 1

                # 添加MoE损失
                if (not core_loss_only_mode) and use_moe and moe_stats is not None:
                    num_experts_per_band = getattr(spec_encoder, 'num_experts_per_band', 4)
                    num_bands = getattr(spec_encoder, 'num_bands', 3)
                    expert_top_k = getattr(spec_encoder, 'expert_top_k', num_experts_per_band)
                    
                    # Check if using Bayesian router or disagreement
                    use_bayesian = getattr(spec_encoder, 'use_bayesian_router', False)
                    use_disagreement = getattr(spec_encoder, 'use_expert_disagreement', False)
                    
                    if use_bayesian or use_disagreement:
                        # Use extended loss function
                        import config as cfg
                        moe_loss, moe_loss_dict = compute_moe_loss_bayesian(
                            moe_stats,
                            num_experts_per_band,
                            num_bands,
                            w_load_balance=3.0,
                            w_entropy=0.01,
                            w_diversity=0.2,
                            w_router_reg=0.05,
                            use_v2_losses=True,
                            diversity_margin=0.3,
                            expert_top_k=expert_top_k,
                            w_sparse_balance=1.0,
                            w_kl_div=cfg.BAYESIAN_ROUTER_KL_WEIGHT,
                            w_disagreement_reg=cfg.EXPERT_DISAGREEMENT_REG_WEIGHT,
                            disagreement_target=cfg.EXPERT_DISAGREEMENT_TARGET,
                            disagreement_margin=cfg.EXPERT_DISAGREEMENT_MARGIN
                        )
                    else:
                        moe_loss, moe_loss_dict = compute_moe_loss(
                            moe_stats, 
                            num_experts_per_band, 
                            num_bands,
                            w_load_balance=3.0,
                            w_entropy=0.01,
                            w_diversity=0.2,
                            w_router_reg=0.05,
                            use_v2_losses=True,
                            diversity_margin=0.3,
                            expert_top_k=expert_top_k,
                            w_sparse_balance=1.0
                        )
                    
                    loss = loss + moe_loss_weight * moe_loss
                    loss_dict.update(moe_loss_dict)
                    _bs_w = float(getattr(Config, "SPEC_MOE_BAND_SPARSITY_WEIGHT", 0.0))
                    if _bs_w > 0 and isinstance(moe_stats, dict):
                        _bst = moe_stats.get("band_sparsity_loss")
                        if torch.is_tensor(_bst) and _bst.requires_grad:
                            _bw = _bs_w * _bst
                            loss = loss + _bw
                            loss_dict["moe_band_sparsity"] = float(_bw.detach().cpu().item())
                    log_moe_router_audit(Config, current_epoch, batch_idx, moe_stats)
                
                # 添加类感知权重正则化
                if (not core_loss_only_mode) and fusion_class_aware_regularization > 0 and hasattr(fusion_module, 'class_modal_weights'):
                    class_aware_reg = fusion_class_aware_regularization * torch.norm(fusion_module.class_modal_weights)
                    loss = loss + class_aware_reg
                    loss_dict['class_aware_reg'] = class_aware_reg.item()

                if not supcon_stage1_now:
                    # ===== metric learning (EMA center loss) on diagnosis branch z_cls =====
                    if FUSION_METRIC_LOSS_WEIGHT > 0 and FUSION_METRIC_LOSS == "center":
                        hp = getattr(fusion_logits, "hierarchical_parts", None)
                        if hp is not None and "z_cls" in hp:
                            z_cls = hp["z_cls"]
                            num_classes = int(getattr(fusion_decoder, "num_classes", 4))
                            if ema_centers is None or ema_centers.size(0) != num_classes or ema_centers.size(1) != z_cls.size(1):
                                ema_centers = torch.zeros(num_classes, z_cls.size(1), device=z_cls.device, dtype=z_cls.dtype)
                                ema_centers = F.normalize(ema_centers + 1e-3, dim=-1)

                            center_loss_val = cosine_center_loss(
                                z_cls, labels, ema_centers,
                                class_weights=class_weights_for_loss if class_weights_for_loss is not None else None
                            )
                            loss = loss + FUSION_METRIC_LOSS_WEIGHT * center_loss_val
                            loss_dict["metric_center"] = float(center_loss_val.detach().cpu().item())

                            ema_centers = update_ema_centers(
                                ema_centers, z_cls.detach(), labels.detach(),
                                momentum=float(FUSION_CENTER_MOMENTUM)
                            )

                    if (not core_loss_only_mode) and clip_model is not None and float(lambda_clip) > 0.0:
                        clip_emb = clip_model.encode_spec(spec)
                        clip_loss = spec_clip_alignment_loss(z_spec, clip_emb)
                        loss = loss + lambda_clip * clip_loss
                    try:
                        _clip_style_active = bool(getattr(cfg, "USE_CLIP_STYLE_CONTRASTIVE", False))
                        _clip_style_w = float(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_WEIGHT", 0.0))
                        _clip_style_t = float(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_TEMPERATURE", 0.07))
                        _clip_style_start = int(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_START_EPOCH", 1))
                    except Exception:
                        _clip_style_active, _clip_style_w, _clip_style_t, _clip_style_start = False, 0.0, 0.07, 1
                    if _clip_style_active and _clip_style_w > 0.0 and current_epoch >= _clip_style_start:
                        try:
                            _txt_batch = text_encoder(list(texts))
                            _clip_out = clip_style_batch_contrastive_loss(
                                z_spec_clean,
                                _txt_batch,
                                temperature=_clip_style_t,
                                labels=labels,
                            )
                            _clip_tensor = _clip_out.get("loss", None)
                            if torch.is_tensor(_clip_tensor) and _clip_tensor.requires_grad:
                                loss = loss + float(_clip_style_w) * _clip_tensor
                                loss_dict["clip_style"] = float((float(_clip_style_w) * _clip_tensor).detach().cpu().item())
                                loss_dict["_info_clip_style_diag_cos"] = float(_clip_out.get("diag_cos_mean", 0.0))
                                loss_dict["_info_clip_style_offdiag_cos"] = float(_clip_out.get("offdiag_cos_mean", 0.0))
                                epoch_clip_style += float(loss_dict["clip_style"])
                                epoch_clip_style_diag_cos += float(_clip_out.get("diag_cos_mean", 0.0))
                                epoch_clip_style_offdiag_cos += float(_clip_out.get("offdiag_cos_mean", 0.0))
                                epoch_clip_style_batches += 1
                        except Exception:
                            pass

                    # ===== ATT379(S2): Dual CLIP on z_fusion (Task2 path) =====
                    try:
                        _dual_clip_active = bool(getattr(cfg, "USE_DUAL_CLIP_SPEC", False))
                        _dual_clip_w = float(getattr(cfg, "DUAL_CLIP_SPEC_WEIGHT", 0.0))
                    except Exception:
                        _dual_clip_active, _dual_clip_w = False, 0.0
                    if (_dual_clip_active and _dual_clip_w > 0.0
                            and _clip_style_active and current_epoch >= _clip_style_start):
                        try:
                            _z_fus_for_clip = z_fusion_for_decoder if z_fusion_for_decoder is not None else z_fusion
                            _z_fus_norm = F.normalize(_z_fus_for_clip.float(), dim=-1)
                            if "_txt_batch" not in dir():
                                _txt_batch = text_encoder(list(texts))
                            _dc_out = clip_style_batch_contrastive_loss(
                                _z_fus_norm, _txt_batch,
                                temperature=_clip_style_t, labels=labels,
                            )
                            _dc_tensor = _dc_out.get("loss", None)
                            if torch.is_tensor(_dc_tensor) and _dc_tensor.requires_grad:
                                loss = loss + float(_dual_clip_w) * _dc_tensor
                                loss_dict["dual_clip_fus"] = float((float(_dual_clip_w) * _dc_tensor).detach().cpu().item())
                                loss_dict["_info_dual_clip_fus_diag_cos"] = float(_dc_out.get("diag_cos_mean", 0.0))
                        except Exception:
                            pass

                    # 添加校准损失（ECE）
                    if (not core_loss_only_mode) and use_calibration_loss and calibration_loss_weight > 0:
                        # 使用fusion_logits计算ECE
                        if hasattr(fusion_logits, 'hierarchical_parts'):
                            # 对于层次化模型，使用4类logits
                            logits_for_ece = fusion_logits
                        else:
                            logits_for_ece = fusion_logits
                        
                        ece_loss = compute_ece_loss(logits_for_ece, labels, n_bins=calibration_n_bins)
                        loss = loss + calibration_loss_weight * ece_loss
                        loss_dict['ece'] = ece_loss.item()
                    
                    # ===== Prototype Separation Loss (non-AMP branch) =====
                    if use_proto_sep_loss and current_epoch >= proto_sep_start_epoch:
                        z_cls_for_proto = None
                        try:
                            hp = getattr(fusion_logits, "hierarchical_parts", None)
                            if isinstance(hp, dict):
                                z_cls_for_proto = hp.get("z_cls", None)
                            if z_cls_for_proto is None:
                                z_cls_for_proto = z_fusion
                        except Exception:
                            z_cls_for_proto = z_fusion
                        
                        if z_cls_for_proto is not None:
                            if proto_sep_loss_fn is None:
                                feature_dim = z_cls_for_proto.size(-1)
                                proto_sep_loss_fn = PrototypeSeparationLoss(
                                    num_classes=4,
                                    feature_dim=feature_dim,
                                    delta=proto_sep_margin_delta,
                                    momentum=proto_sep_momentum,
                                    use_log_ratio=use_log_proto_sep,
                                    log_eps=log_proto_sep_eps
                                ).to(device)
                                print(f"[Proto-Sep Loss] Initialized with feature_dim={feature_dim}")
                            
                            proto_sep_loss_val, proto_dists = proto_sep_loss_fn(z_cls_for_proto, labels)
                            loss = loss + proto_sep_weight * proto_sep_loss_val
                            loss_dict["proto_sep"] = float(proto_sep_loss_val.detach().cpu().item())
                            epoch_proto_sep_loss += float(proto_sep_loss_val.detach().cpu().item())
                            proto_sep_batches += 1
                            # 类内方差统计（L2，batch-level）
                            try:
                                with torch.no_grad():
                                    for c in range(4):
                                        m = (labels == c)
                                        if m.any():
                                            feats_c = z_cls_for_proto[m]
                                            mu_c = feats_c.mean(dim=0, keepdim=True)
                                            var_c = (feats_c - mu_c).pow(2).sum(dim=1).mean()
                                            class_var_sum[c] += float(var_c.detach().cpu().item())
                                            class_var_count[c] += 1
                            except Exception:
                                pass

                # Attempt 64.1: semantic anchor must be injected on Stage1 path
                # (USE_TWO_STAGE_TRAINING=False => multimodal_alignment_loss branch is not executed).
                try:
                    import config as _cfg_sem
                    use_sem_fvn = bool(getattr(_cfg_sem, "USE_SEMANTIC_FAULT_VS_NORMAL_ANCHOR", False))
                    sem_fvn_weight = float(getattr(_cfg_sem, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_WEIGHT", 0.0))
                    sem_fvn_margin = float(getattr(_cfg_sem, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_MARGIN", 0.15))
                    sem_fvn_start_epoch = int(getattr(_cfg_sem, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_START_EPOCH", 1))
                except Exception:
                    use_sem_fvn, sem_fvn_weight, sem_fvn_margin, sem_fvn_start_epoch = False, 0.0, 0.15, 1
                if (
                    (not core_loss_only_mode)
                    and
                    bool(locals().get("stage1_mode_now", False))
                    and use_sem_fvn
                    and (sem_fvn_weight > 0.0)
                    and (current_epoch >= int(sem_fvn_start_epoch))
                    and isinstance(current_prototypes, dict)
                    and (0 in current_prototypes)
                ):
                    try:
                        z_anchor = F.normalize(z_fusion.float(), dim=-1, eps=1e-6)
                        p_normal = F.normalize(current_prototypes[0].to(z_anchor.device).float(), dim=-1, eps=1e-6)
                        per_cls_losses = []
                        for _c in (1, 2, 3):
                            if _c not in current_prototypes:
                                continue
                            _m = (labels == int(_c))
                            if not bool(_m.any()):
                                continue
                            p_c = F.normalize(current_prototypes[_c].to(z_anchor.device).float(), dim=-1, eps=1e-6)
                            z_c = z_anchor[_m]
                            sim_pos = torch.matmul(z_c, p_c)
                            sim_neg_n = torch.matmul(z_c, p_normal)
                            per_cls_losses.append(F.relu(float(sem_fvn_margin) - (sim_pos - sim_neg_n)).mean())
                        if len(per_cls_losses) > 0:
                            loss_sem_fvn = torch.stack(per_cls_losses).mean()
                            loss = loss + float(sem_fvn_weight) * loss_sem_fvn
                            if isinstance(loss_dict, dict):
                                loss_dict["sem_fault_vs_normal_anchor"] = float(loss_sem_fvn.detach().cpu().item())
                                loss_dict["total"] = float(loss.detach().cpu().item())
                            epoch_sem_fvn_loss += float(loss_sem_fvn.detach().cpu().item())
                            sem_fvn_batches += 1
                    except Exception:
                        pass

                # Att77新增：Batch Nuclear Norm Loss（防止特征维度坍缩）
                try:
                    import config as _cfg_bnn
                    use_bnn = bool(getattr(_cfg_bnn, "USE_BATCH_NUCLEAR_NORM", False))
                    bnn_weight = float(getattr(_cfg_bnn, "BATCH_NUCLEAR_NORM_WEIGHT", 0.01))
                    bnn_start_epoch = int(getattr(_cfg_bnn, "BATCH_NUCLEAR_NORM_START_EPOCH", 1))
                except Exception:
                    use_bnn, bnn_weight, bnn_start_epoch = False, 0.01, 1
                
                if use_bnn and bnn_weight > 0.0 and current_epoch >= bnn_start_epoch:
                    try:
                        if z_fusion is not None and labels is not None and z_fusion.dim() == 2 and labels.dim() == 1:
                            z_fusion_for_bnn = z_fusion.detach()
                            bnn_loss = batch_nuclear_norm_loss(z_fusion_for_bnn, labels, weight=bnn_weight)
                            if torch.is_tensor(bnn_loss) and bnn_loss.requires_grad:
                                loss = loss + bnn_loss
                                if isinstance(loss_dict, dict):
                                    loss_dict["batch_nuclear_norm"] = float(bnn_loss.detach().cpu().item())
                                epoch_bnn_loss += float(bnn_loss.detach().cpu().item())
                                bnn_batches += 1
                    except Exception as e:
                        pass

                # Guard: avoid backward on no-grad loss
                if (
                    task2_coral_enable
                    and float(task2_coral_weight) > 0.0
                    and int(current_epoch) >= int(task2_coral_start_epoch)
                    and torch.is_tensor(z_spec)
                    and torch.is_tensor(domain_ids)
                ):
                    try:
                        mask_cwru = (domain_ids == 0) | (domain_ids == 1)
                        mask_pu = (domain_ids == 2)
                        if bool(mask_cwru.any()) and bool(mask_pu.any()):
                            if int(mask_cwru.sum().item()) >= int(task2_coral_min_samples) and int(mask_pu.sum().item()) >= int(task2_coral_min_samples):
                                loss_coral = coral_covariance_alignment_loss(z_spec[mask_cwru], z_spec[mask_pu])
                                coral_warmup_ratio = min(
                                    1.0,
                                    max(0.0, float(int(current_epoch) - int(task2_coral_start_epoch) + 1))
                                    / float(max(1, int(task2_coral_warmup_epochs))),
                                )
                                effective_coral_weight = float(task2_coral_weight) * coral_warmup_ratio
                                loss = loss + effective_coral_weight * loss_coral
                                if isinstance(loss_dict, dict):
                                    loss_dict["coral_spec"] = float(loss_coral.detach().cpu().item())
                                    loss_dict["coral_spec_weight"] = float(effective_coral_weight)
                                    loss_dict["total"] = float(loss.detach().cpu().item())
                                epoch_coral_loss += float(loss_coral.detach().cpu().item())
                                coral_batches += 1
                                if int(current_epoch) == int(task2_coral_start_epoch) and int(batch_idx) == 0:
                                    print(
                                        f"[Task2 CORAL] active: n_cwru={int(mask_cwru.sum().item())}, "
                                        f"n_pu={int(mask_pu.sum().item())}, "
                                        f"w_base={float(task2_coral_weight):.4f}, w_eff={float(effective_coral_weight):.4f}"
                                    )
                                # ATT197(B): source-to-target CORAL with unlabeled HUST/HIT reference batches.
                                if (
                                    task2_target_coral_enable
                                    and float(task2_target_coral_weight) > 0.0
                                    and len(target_coral_refs) > 0
                                    and (int(batch_idx) % max(1, int(task2_target_coral_every_n_steps)) == 0)
                                ):
                                    src_mask = mask_cwru | mask_pu
                                    if bool(src_mask.any()) and int(src_mask.sum().item()) >= int(task2_target_coral_min_samples):
                                        src_feat = z_spec[src_mask]
                                        loss_target_coral = None
                                        used_refs = 0
                                        for _, _ref_feat in target_coral_refs:
                                            _lt = coral_covariance_alignment_loss(src_feat, _ref_feat)
                                            if torch.is_tensor(_lt):
                                                loss_target_coral = _lt if loss_target_coral is None else (loss_target_coral + _lt)
                                                used_refs += 1
                                        if loss_target_coral is not None and used_refs > 0:
                                            loss_target_coral = loss_target_coral / float(max(1, used_refs))
                                            effective_target_coral_weight = float(task2_target_coral_weight) * coral_warmup_ratio
                                            loss = loss + effective_target_coral_weight * loss_target_coral
                                            if isinstance(loss_dict, dict):
                                                loss_dict["coral_target_spec"] = float(loss_target_coral.detach().cpu().item())
                                                loss_dict["coral_target_spec_weight"] = float(effective_target_coral_weight)
                                                loss_dict["total"] = float(loss.detach().cpu().item())
                                            epoch_target_coral_loss += float(loss_target_coral.detach().cpu().item())
                                            target_coral_batches += 1
                        # ATT221(S1): source->target MK-MMD with unlabeled HUST/HIT online batches.
                        if (
                            task2_target_mmd_enable
                            and _is_task2_phase
                            and float(task2_target_mmd_weight) > 0.0
                            and int(current_epoch) >= int(task2_target_mmd_start_epoch)
                            and len(target_mmd_loaders) > 0
                            and (int(batch_idx) % max(1, int(task2_target_mmd_every_n_steps)) == 0)
                        ):
                            src_mask = mask_cwru | mask_pu
                            src_count = int(src_mask.sum().item())
                            if bool(src_mask.any()) and src_count >= int(task2_target_mmd_min_samples):
                                pick_idx = int(batch_idx) % int(len(target_mmd_loaders))
                                tgt_name, tgt_loader = target_mmd_loaders[pick_idx]
                                tgt_iter = target_mmd_iters.get(tgt_name, None)
                                if tgt_iter is None:
                                    tgt_iter = iter(tgt_loader)
                                try:
                                    tgt_batch = next(tgt_iter)
                                except StopIteration:
                                    tgt_iter = iter(tgt_loader)
                                    tgt_batch = next(tgt_iter)
                                except Exception:
                                    tgt_batch = None
                                target_mmd_iters[tgt_name] = tgt_iter
                                if tgt_batch is not None:
                                    _enc_mode = bool(spec_encoder.training)
                                    _txt_mode = bool(text_encoder.training)
                                    try:
                                        spec_encoder.eval()
                                        text_encoder.eval()
                                        with torch.no_grad():
                                            z_tgt = _extract_spec_feat_from_batch(tgt_batch)
                                    finally:
                                        if _enc_mode:
                                            spec_encoder.train()
                                        if _txt_mode:
                                            text_encoder.train()
                                    if torch.is_tensor(z_tgt) and z_tgt.dim() == 2 and int(z_tgt.size(0)) >= int(task2_target_mmd_min_samples):
                                        mmd_loss = multi_kernel_mmd_loss(
                                            z_spec[src_mask],
                                            z_tgt.detach(),
                                            kernel_scales=task2_target_mmd_kernel_scales,
                                        )
                                        mmd_warmup_ratio = min(
                                            1.0,
                                            max(0.0, float(int(current_epoch) - int(task2_target_mmd_start_epoch) + 1))
                                            / float(max(1, int(task2_target_mmd_warmup_epochs))),
                                        )
                                        effective_mmd_weight = float(task2_target_mmd_weight) * mmd_warmup_ratio
                                        loss = loss + effective_mmd_weight * mmd_loss
                                        if isinstance(loss_dict, dict):
                                            loss_dict["mmd_target_spec"] = float(mmd_loss.detach().cpu().item())
                                            loss_dict["mmd_target_spec_weight"] = float(effective_mmd_weight)
                                            loss_dict["total"] = float(loss.detach().cpu().item())
                                        epoch_target_mmd_loss += float(mmd_loss.detach().cpu().item())
                                        target_mmd_batches += 1
                                        if int(current_epoch) == int(task2_target_mmd_start_epoch) and int(batch_idx) == 0:
                                            print(
                                                f"[Task2 Target-MMD] active: src_n={src_count}, tgt={tgt_name}, "
                                                f"tgt_n={int(z_tgt.size(0))}, w_base={float(task2_target_mmd_weight):.4f}, "
                                                f"w_eff={float(effective_mmd_weight):.4f}"
                                            )
                    except Exception:
                        pass

                # ===== ATT392-S1: Refresh training_proto_tensor from learnable protos =====
                if learnable_proto_module is not None:
                    training_proto_tensor = learnable_proto_module.forward()  # [C, D] with grad

                # ===== ATT374/ATT390-S2: Prototype-Anchored Contrastive Loss (PACL) =====
                if training_proto_tensor is not None:
                    _pacl_active = bool(getattr(cfg, "USE_PROTOTYPE_ANCHOR_LOSS", False))
                    _pacl_start = int(getattr(cfg, "PROTOTYPE_ANCHOR_START_EPOCH", 1))
                    if _pacl_active and current_epoch >= _pacl_start:
                        _pacl_temp = float(getattr(cfg, "PROTOTYPE_ANCHOR_TEMPERATURE", 0.07))
                        _pacl_ls = float(getattr(cfg, "PROTOTYPE_ANCHOR_LABEL_SMOOTHING", 0.05))
                        _pacl_w_spec = float(getattr(cfg, "PROTOTYPE_ANCHOR_WEIGHT_SPEC", 0.5))
                        _pacl_w_fus = float(getattr(cfg, "PROTOTYPE_ANCHOR_WEIGHT_FUSION", 0.5))
                        _pacl_use_margin = bool(getattr(cfg, "PACL_USE_MARGIN", False))
                        _pacl_margin = float(getattr(cfg, "PACL_MARGIN", 0.30))
                        try:
                            if _pacl_w_spec > 0.0 and torch.is_tensor(z_spec_clean) and z_spec_clean.size(-1) == training_proto_tensor.size(-1):
                                _pacl_spec = prototype_anchor_loss(
                                    z_spec_clean, labels, training_proto_tensor,
                                    temperature=_pacl_temp, label_smoothing=_pacl_ls,
                                    use_margin=_pacl_use_margin, margin=_pacl_margin,
                                )
                                if torch.is_tensor(_pacl_spec["loss"]) and _pacl_spec["loss"].requires_grad:
                                    _pacl_spec_wt = _pacl_w_spec * _pacl_spec["loss"]
                                    loss = loss + _pacl_spec_wt
                                    loss_dict["pacl_spec"] = float(_pacl_spec_wt.detach().cpu().item())
                                    loss_dict["_info_pacl_spec_acc"] = _pacl_spec["acc"]
                                    loss_dict["_info_pacl_spec_pos_cos"] = _pacl_spec["mean_pos_cos"]
                                    loss_dict["_info_pacl_spec_neg_cos"] = _pacl_spec["mean_neg_cos"]
                                    loss_dict["_info_pacl_spec_per_class_acc"] = _pacl_spec.get("per_class_acc", {})
                                    loss_dict["_info_pacl_spec_per_class_pos_cos"] = _pacl_spec.get("per_class_pos_cos", {})
                                    if isinstance(loss_dict.get("_tensor_terms"), dict):
                                        loss_dict["_tensor_terms"]["pacl_spec_weighted"] = _pacl_spec_wt
                        except Exception:
                            pass
                        try:
                            if _pacl_w_fus > 0.0 and torch.is_tensor(z_fusion) and z_fusion.size(-1) == training_proto_tensor.size(-1):
                                _pacl_fus = prototype_anchor_loss(
                                    z_fusion, labels, training_proto_tensor,
                                    temperature=_pacl_temp, label_smoothing=_pacl_ls,
                                    use_margin=_pacl_use_margin, margin=_pacl_margin,
                                )
                                if torch.is_tensor(_pacl_fus["loss"]) and _pacl_fus["loss"].requires_grad:
                                    _pacl_fus_wt = _pacl_w_fus * _pacl_fus["loss"]
                                    loss = loss + _pacl_fus_wt
                                    loss_dict["pacl_fusion"] = float(_pacl_fus_wt.detach().cpu().item())
                                    if isinstance(loss_dict.get("_tensor_terms"), dict):
                                        loss_dict["_tensor_terms"]["pacl_fus_weighted"] = _pacl_fus_wt
                                    loss_dict["_info_pacl_fus_acc"] = _pacl_fus["acc"]
                                    loss_dict["_info_pacl_fus_pos_cos"] = _pacl_fus["mean_pos_cos"]
                                    loss_dict["_info_pacl_fus_neg_cos"] = _pacl_fus["mean_neg_cos"]
                        except Exception:
                            pass
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT392-S1: Learnable Prototype Regularization (separation + anchor) =====
                # ATT428: skip computation entirely when weights=0 to avoid inplace-modified-variable error
                if learnable_proto_module is not None:
                    _lp_start = int(getattr(cfg, "LEARNABLE_PROTO_START_EPOCH", 0))
                    _lp_sep_w = float(getattr(cfg, "LEARNABLE_PROTO_SEP_WEIGHT", 0.0))
                    _lp_anc_w = float(getattr(cfg, "LEARNABLE_PROTO_ANCHOR_WEIGHT", 0.0))
                    if current_epoch >= _lp_start and (_lp_sep_w > 0.0 or _lp_anc_w > 0.0):
                        if _lp_sep_w > 0.0:
                            _sep_loss = learnable_proto_module.separation_loss()
                            if torch.is_tensor(_sep_loss) and _sep_loss.requires_grad:
                                _sep_wt = _lp_sep_w * _sep_loss
                                loss = loss + _sep_wt
                                loss_dict["proto_sep"] = float(_sep_wt.detach().cpu().item())
                        if _lp_anc_w > 0.0:
                            _anc_loss = learnable_proto_module.anchor_loss()
                            if torch.is_tensor(_anc_loss) and _anc_loss.requires_grad:
                                _anc_wt = _lp_anc_w * _anc_loss
                                loss = loss + _anc_wt
                                loss_dict["proto_anchor"] = float(_anc_wt.detach().cpu().item())
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT434: Gate L2 Regularization (tanh gate → push toward 0 = balanced) =====
                _gate_ent_w = float(getattr(cfg, "GATE_ENTROPY_WEIGHT", 0.0))
                if _gate_ent_w > 0.0 and hasattr(fusion_module, '_gate_for_reg') and fusion_module._gate_for_reg is not None:
                    _gv = fusion_module._gate_for_reg.float()
                    _gate_l2 = (_gv ** 2).mean()
                    _gate_l2_penalty = _gate_ent_w * _gate_l2
                    if _gate_l2_penalty.requires_grad:
                        loss = loss + _gate_l2_penalty
                        loss_dict["gate_ent_reg"] = float(_gate_l2_penalty.detach().cpu().item())
                        loss_dict["_info_gate_l2"] = float(_gate_l2.detach().cpu().item())
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT432-δ: Physical Feature Alignment Loss =====
                if phys_proto_module is not None and phys_features is not None:
                    _phys_w = float(getattr(cfg, "PHYS_FEATURE_ALIGN_WEIGHT", 0.20))
                    if _phys_w > 0.0:
                        try:
                            phys_proto_module.train()
                            z_phys = phys_proto_module(phys_features, class_ids=labels)
                            if torch.is_tensor(z_fusion) and z_fusion.dim() >= 1:
                                _z_fus_phys = z_fusion
                                if hasattr(z_fusion, "_pre_l2_feature"):
                                    _z_fus_phys = F.normalize(z_fusion._pre_l2_feature, dim=-1)
                                _phys_cos = (z_phys * _z_fus_phys.detach()).sum(dim=-1)
                                _phys_align_loss = (1.0 - _phys_cos).mean()
                                _phys_loss_wt = _phys_w * _phys_align_loss
                                if _ablation_gate_bank is not None:
                                    _phys_loss_wt = _ablation_gate_bank.scale_loss(
                                        "physics_phys_proto", _phys_loss_wt
                                    )
                                loss = loss + _phys_loss_wt
                                loss_dict["phys_align"] = float(_phys_loss_wt.detach().cpu().item())
                                loss_dict["_info_phys_cos_mean"] = float(_phys_cos.mean().detach().cpu().item())
                                loss_dict["total"] = float(loss.detach().cpu().item())
                        except Exception as _phys_e:
                            if batch_idx == 0:
                                print(f"[ATT432-δ] phys_align error: {_phys_e}")

                # ===== ATT438-DGL: Auxiliary Fusion Head CE Loss =====
                _dgl_on_now = bool(getattr(cfg, "DGL_ENABLE", False))
                _dgl_aux_w = float(getattr(cfg, "DGL_AUX_FUSION_HEAD_WEIGHT", 0.30))
                if _dgl_on_now and _dgl_aux_w > 0.0 and hasattr(fusion_module, 'get_aux_logits'):
                    _dgl_aux_logits = fusion_module.get_aux_logits()
                    if _dgl_aux_logits is not None and labels is not None:
                        try:
                            _dgl_labels = labels.long()
                            _dgl_n_cls = _dgl_aux_logits.size(-1)
                            _dgl_cw = None
                            if class_weights is not None and class_weights.numel() == _dgl_n_cls:
                                _dgl_cw = class_weights.to(_dgl_aux_logits.device)
                            _dgl_ce = F.cross_entropy(_dgl_aux_logits, _dgl_labels, weight=_dgl_cw)
                            _dgl_loss_wt = _dgl_aux_w * _dgl_ce
                            if _dgl_loss_wt.requires_grad:
                                loss = loss + _dgl_loss_wt
                                loss_dict["dgl_aux_ce"] = float(_dgl_loss_wt.detach().cpu().item())
                                loss_dict["_info_dgl_aux_ce_raw"] = float(_dgl_ce.detach().cpu().item())
                                with torch.no_grad():
                                    _dgl_preds = _dgl_aux_logits.argmax(dim=-1)
                                    _dgl_acc = (_dgl_preds == _dgl_labels).float().mean().item()
                                loss_dict["_info_dgl_aux_acc"] = _dgl_acc
                                loss_dict["total"] = float(loss.detach().cpu().item())
                                if batch_idx == 0 and (current_epoch <= 3 or current_epoch % 5 == 0):
                                    print(f"[ATT438-DGL][E{current_epoch}B{batch_idx}] "
                                          f"aux_ce_raw={float(_dgl_ce.detach().cpu().item()):.4f} "
                                          f"aux_ce_wt={float(_dgl_loss_wt.detach().cpu().item()):.4f} "
                                          f"aux_acc={_dgl_acc:.4f}")
                        except Exception as _dgl_e:
                            if batch_idx == 0:
                                print(f"[ATT438-DGL] aux_ce error: {_dgl_e}")

                # ===== ATT439: Cross-Modal Knowledge Distillation (Signal → Spec) =====
                _xkd_on = bool(getattr(cfg, "CROSS_MODAL_KD_ENABLE", False))
                _xkd_w = float(getattr(cfg, "CROSS_MODAL_KD_WEIGHT", 0.50))
                _xkd_T = float(getattr(cfg, "CROSS_MODAL_KD_TEMPERATURE", 4.0))
                if _xkd_on and _xkd_w > 0.0 and signal_logits is not None and spec_logits is not None:
                    try:
                        _sig_n = signal_logits.size(-1)
                        _spe_n = spec_logits.size(-1)
                        _kd_n = min(_sig_n, _spe_n)
                        _sig_kd = signal_logits[:, :_kd_n].detach().float()
                        _spe_kd = spec_logits[:, :_kd_n].float()
                        _p_teacher = F.softmax(_sig_kd / _xkd_T, dim=-1)
                        _log_p_student = F.log_softmax(_spe_kd / _xkd_T, dim=-1)
                        _kd_loss = F.kl_div(_log_p_student, _p_teacher, reduction='batchmean') * (_xkd_T ** 2)
                        if _kd_loss.requires_grad:
                            _kd_loss_wt = _xkd_w * _kd_loss
                            loss = loss + _kd_loss_wt
                            loss_dict["xkd_sig2spec"] = float(_kd_loss_wt.detach().cpu().item())
                            loss_dict["_info_xkd_raw"] = float(_kd_loss.detach().cpu().item())
                            loss_dict["total"] = float(loss.detach().cpu().item())
                            if batch_idx == 0 and (current_epoch <= 3 or current_epoch % 5 == 0):
                                with torch.no_grad():
                                    _sig_pred = signal_logits[:, :_kd_n].argmax(dim=-1)
                                    _spe_pred = spec_logits[:, :_kd_n].argmax(dim=-1)
                                    _agree = (_sig_pred == _spe_pred).float().mean().item()
                                print(f"[ATT439-XKD][E{current_epoch}B{batch_idx}] "
                                      f"kd_raw={float(_kd_loss.detach().cpu().item()):.4f} "
                                      f"kd_wt={float(_kd_loss_wt.detach().cpu().item()):.4f} "
                                      f"T={_xkd_T:.1f} agree={_agree:.4f} "
                                      f"sig_cls={_sig_n} spec_cls={_spe_n} kd_cls={_kd_n}")
                    except Exception as _xkd_e:
                        if batch_idx == 0:
                            print(f"[ATT439-XKD] error: {_xkd_e}")

                # ===== ATT391-S1: Adapter PACL (ETF→Text bridge) =====
                if proto_adapter is not None and _adapter_text_proto_tensor is not None:
                    _ada_start = int(getattr(cfg, "ETF_TEXT_ADAPTER_START_EPOCH", 2))
                    _ada_w = float(getattr(cfg, "ETF_TEXT_ADAPTER_WEIGHT", 0.30))
                    _ada_t = float(getattr(cfg, "ETF_TEXT_ADAPTER_TEMPERATURE", 0.07))
                    if current_epoch >= _ada_start and _ada_w > 0.0:
                        try:
                            _z_fus_det = z_fusion.detach()
                            _z_adapted = proto_adapter(_z_fus_det)
                            _ada_result = prototype_anchor_loss(
                                _z_adapted, labels, _adapter_text_proto_tensor,
                                temperature=_ada_t, label_smoothing=0.05,
                            )
                            if torch.is_tensor(_ada_result["loss"]) and _ada_result["loss"].requires_grad:
                                _ada_loss_wt = _ada_w * _ada_result["loss"]
                                loss = loss + _ada_loss_wt
                                loss_dict["adapter_pacl"] = float(_ada_loss_wt.detach().cpu().item())
                                loss_dict["_info_adapter_acc"] = _ada_result["acc"]
                                loss_dict["_info_adapter_pos_cos"] = _ada_result["mean_pos_cos"]
                                loss_dict["_info_adapter_neg_cos"] = _ada_result["mean_neg_cos"]
                        except Exception:
                            pass
                        loss_dict["total"] = float(loss.detach().cpu().item())

                # ===== ATT374+ATT377: Text-ETF Alignment Loss + F2/E1 cosine matrix =====
                _text_etf_interval = int(getattr(cfg, "TEXT_ETF_ALIGNMENT_BATCH_INTERVAL", (1 if core_loss_only_mode else 10**9)))
                if use_etf and etf_prototypes_tensor is not None and (batch_idx % max(1, _text_etf_interval) == 0):
                    _text_etf_active = bool(getattr(cfg, "USE_TEXT_ETF_ALIGNMENT_LOSS", False))
                    _text_etf_start = int(getattr(cfg, "TEXT_ETF_ALIGNMENT_START_EPOCH", 1))
                    if _text_etf_active and current_epoch >= _text_etf_start:
                        _text_etf_w = float(getattr(cfg, "TEXT_ETF_ALIGNMENT_WEIGHT", 0.3))
                        _text_cos_matrix_w = float(getattr(cfg, "TEXT_ETF_COSINE_MATRIX_WEIGHT", 0.10))
                        try:
                            _tea_result = text_etf_alignment_loss(
                                text_encoder, class_prompts, etf_prototypes_tensor, device=device,
                                cosine_matrix_weight=_text_cos_matrix_w,
                            )
                            if torch.is_tensor(_tea_result["loss"]) and _tea_result["loss"].requires_grad:
                                loss = loss + _text_etf_w * _tea_result["loss"]
                                loss_dict["text_etf_align"] = float((_text_etf_w * _tea_result["loss"]).detach().cpu().item())
                                loss_dict["_info_text_etf_cos"] = _tea_result["mean_cos"]
                                loss_dict["_info_text_cos_margin"] = _tea_result.get("text_cos_margin", 0.0)
                                loss_dict["total"] = float(loss.detach().cpu().item())
                            if batch_idx == 0:
                                print(
                                    f"[TextProtoCos][E{current_epoch}] diag={_tea_result.get('text_cos_diag', 0.0):.4f}, "
                                    f"offdiag={_tea_result.get('text_cos_offdiag', 0.0):.4f}, "
                                    f"margin={_tea_result.get('text_cos_margin', 0.0):.4f}, "
                                    f"matrix_loss={_tea_result.get('loss_matrix_match', 0.0):.6f}"
                                )
                        except Exception:
                            pass

                _batch_should_log_loss = bool(batch_loss_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % batch_loss_log_step_stride == 0)))
                _batch_should_log_proto = bool(proto_cos_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % proto_cos_log_step_stride == 0)))
                _batch_should_log_io = bool(batch_io_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % batch_io_log_step_stride == 0)))
                _batch_should_log_grad = bool(grad_contrib_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % grad_contrib_log_step_stride == 0)))
                if isinstance(loss_dict, dict):
                    _tensor_terms = loss_dict.get("_tensor_terms", None)
                    if isinstance(_tensor_terms, dict):
                        for _name, _term in _tensor_terms.items():
                            if torch.is_tensor(_term):
                                grad_probe_terms[_name] = _term
                    for _k_default in (
                        "total",
                        "diag_ce",
                        "nf_ce",
                        "fault_3way_ce",
                        "supcon",
                        "global_decorr",
                        "clip_style",
                        "fusion_proto_align",
                        "fusion_supcon",
                        "nr_inter_cos_margin",
                        "text_prompt_sep",
                        "nr_inter_cos_hard",
                        "fishr",
                        "ccpa",
                    ):
                        if _k_default not in loss_dict:
                            loss_dict[_k_default] = 0.0
                    for _k_l, _v_l in loss_dict.items():
                        if not isinstance(_v_l, (int, float)):
                            continue
                        _vf = float(_v_l)
                        if (not math.isfinite(_vf)):
                            continue
                        if str(_k_l).startswith("_info_"):
                            epoch_loss_info_sum[_k_l] = epoch_loss_info_sum.get(_k_l, 0.0) + _vf
                            epoch_loss_info_count[_k_l] = epoch_loss_info_count.get(_k_l, 0) + 1
                        else:
                            epoch_loss_term_sum[_k_l] = epoch_loss_term_sum.get(_k_l, 0.0) + _vf
                            epoch_loss_term_count[_k_l] = epoch_loss_term_count.get(_k_l, 0) + 1
                if _batch_should_log_loss and isinstance(loss_dict, dict):
                    _keys = [
                        "total", "_info_fusion_cls", "diag_ce", "nf_ce", "fault_3way_ce",
                        "supcon", "fusion_supcon", "global_decorr", "clip_style", "fusion_proto_align", "nr_inter_cos_margin",
                        "text_prompt_sep", "nr_inter_cos_hard", "fishr", "ccpa", "proto_sep", "proto_anchor"
                    ]
                    _parts = []
                    for _k in _keys:
                        try:
                            _parts.append(f"{_k}={float(loss_dict.get(_k, 0.0)):.4f}")
                        except Exception:
                            pass
                    try:
                        _lbl_hist = torch.bincount(labels.detach().cpu().long(), minlength=4).tolist()
                    except Exception:
                        _lbl_hist = []
                    try:
                        _dom_hist = torch.bincount(domain_ids.detach().cpu().long(), minlength=5).tolist() if domain_ids is not None else []
                    except Exception:
                        _dom_hist = []
                    print(
                        f"[BatchLoss][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                        + " | ".join(_parts)
                        + f" | y_hist={_lbl_hist} | dom_hist={_dom_hist}"
                    )
                    if batch_loss_breakdown_log_enabled and (batch_idx == 0 or ((batch_idx + 1) % batch_loss_breakdown_step_stride == 0)):
                        _num_terms = {}
                        _info_terms = {}
                        for _k_bd, _v_bd in loss_dict.items():
                            if not isinstance(_v_bd, (int, float)):
                                continue
                            _v_bd_f = float(_v_bd)
                            if not math.isfinite(_v_bd_f):
                                continue
                            if str(_k_bd).startswith("_info_"):
                                _info_terms[_k_bd] = _v_bd_f
                            elif _k_bd != "total":
                                _num_terms[_k_bd] = _v_bd_f
                        _total_bd = float(loss_dict.get("total", 0.0))
                        _sum_non_total = float(sum(_num_terms.values()))
                        _residual = float(_total_bd - _sum_non_total)
                        _info_sum = float(sum(_info_terms.values()))
                        _top_terms = sorted(_num_terms.items(), key=lambda _x: abs(_x[1]), reverse=True)[:8]
                        _top_info = sorted(_info_terms.items(), key=lambda _x: abs(_x[1]), reverse=True)[:6]
                        _top_s = ", ".join([f"{_k}={_v:.4f}" for _k, _v in _top_terms]) if _top_terms else "none"
                        _top_info_s = ", ".join([f"{_k}={_v:.4f}" for _k, _v in _top_info]) if _top_info else "none"
                        print(
                            f"[BatchLossBreakdown][AMP=OFF][E{current_epoch}B{batch_idx}] "
                            f"total={_total_bd:.4f}, sum_non_total={_sum_non_total:.4f}, residual={_residual:.4f}, "
                            f"terms={len(_num_terms)}, info_terms={len(_info_terms)}, info_sum={_info_sum:.4f}, "
                            f"top={_top_s}, info_top={_top_info_s}"
                        )
                        if batch_idx == 0 and current_epoch == 1:
                            _all_sorted = sorted(_num_terms.items(), key=lambda _x: abs(_x[1]), reverse=True)
                            _all_s = ", ".join([f"{_k}={_v:.4f}" for _k, _v in _all_sorted])
                            print(f"[LossBudgetAudit][E{current_epoch}B0] ALL {len(_num_terms)} terms: {_all_s}")
                            print(
                                f"[LossBudgetAudit] NOTE: residual!=0 is EXPECTED. "
                                f"alignment_loss stores RAW values, total is WEIGHTED sum. "
                                f"This is NOT a double-counting bug."
                            )
                if _batch_should_log_proto:
                    _proto_mat, _proto_keys = _to_proto_matrix(
                        current_prototypes,
                        z_fusion.device if torch.is_tensor(z_fusion) else device,
                        z_fusion.dtype if torch.is_tensor(z_fusion) else torch.float32,
                    )
                    if _proto_mat is not None:
                        try:
                            _proto_mat = F.normalize(_proto_mat.float(), dim=-1, eps=1e-6)
                            _cos = torch.matmul(_proto_mat, _proto_mat.t()).detach().cpu().numpy()
                            _rows = []
                            for _i, _k in enumerate(_proto_keys):
                                _vals = ",".join([f"{float(_v):.3f}" for _v in _cos[_i].tolist()])
                                _rows.append(f"{int(_k)}:[{_vals}]")
                            print(
                                f"[ProtoCos][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                + " | ".join(_rows)
                            )
                        except Exception:
                            pass
                if _batch_should_log_io:
                    try:
                        _z_spec_pre = extract_pre_l2_feature(z_spec_clean) if torch.is_tensor(z_spec_clean) else None
                        _z_fus_pre = extract_pre_l2_feature(z_fusion) if torch.is_tensor(z_fusion) else None
                        _sig_std = float(z_signal.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(z_signal) else 0.0
                        _spec_std = float(z_spec_clean.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(z_spec_clean) else 0.0
                        _fus_std = float(z_fusion.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(z_fusion) else 0.0
                        _sig_norm = float(z_signal.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(z_signal) else 0.0
                        _spec_norm = float(z_spec_clean.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(z_spec_clean) else 0.0
                        _fus_norm = float(z_fusion.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(z_fusion) else 0.0
                        _spec_pre_std = float(_z_spec_pre.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(_z_spec_pre) else 0.0
                        _fus_pre_std = float(_z_fus_pre.detach().float().std(unbiased=False).cpu().item()) if torch.is_tensor(_z_fus_pre) else 0.0
                        _spec_pre_norm = float(_z_spec_pre.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(_z_spec_pre) else 0.0
                        _fus_pre_norm = float(_z_fus_pre.detach().float().norm(dim=-1).mean().cpu().item()) if torch.is_tensor(_z_fus_pre) else 0.0
                        _spec_fus_pre_cos = float("nan")
                        if torch.is_tensor(_z_spec_pre) and torch.is_tensor(_z_fus_pre):
                            if _z_spec_pre.shape == _z_fus_pre.shape and _z_spec_pre.dim() == 2:
                                _spec_fus_pre_cos = float(
                                    F.cosine_similarity(
                                        F.normalize(_z_spec_pre.detach().float(), dim=-1),
                                        F.normalize(_z_fus_pre.detach().float(), dim=-1),
                                        dim=-1
                                    ).mean().cpu().item()
                                )
                        print(
                            f"[BatchIO][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"std(sig/spec/fus)=({_sig_std:.4f},{_spec_std:.4f},{_fus_std:.4f}) "
                            f"norm(sig/spec/fus)=({_sig_norm:.4f},{_spec_norm:.4f},{_fus_norm:.4f})"
                        )
                        print(
                            f"[BatchIOPreL2][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"std(spec/fus)=({_spec_pre_std:.4f},{_fus_pre_std:.4f}) "
                            f"norm(spec/fus)=({_spec_pre_norm:.4f},{_fus_pre_norm:.4f}) "
                            f"cos(spec,fus)={_spec_fus_pre_cos:.4f}"
                        )
                        try:
                            if signal_logits is not None and torch.is_tensor(signal_logits):
                                with torch.no_grad():
                                    _p_sig_b = F.softmax(signal_logits.float(), dim=1)
                                    _H_sig_b = -torch.sum(_p_sig_b * torch.log(torch.clamp(_p_sig_b, min=1e-8)), dim=1)
                                    _H_max_b = math.log(max(2, _p_sig_b.size(1)))
                                    _info_sig_b = (1.0 - _H_sig_b / _H_max_b).clamp(min=0.0, max=1.0)
                                    _seg_gate_b = getattr(fusion_module, '_last_signal_entropy_gate', None)
                                    print(
                                        f"[SignalEntropy][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                        f"H_norm={float(_H_sig_b.mean()/_H_max_b):.4f} info_score={float(_info_sig_b.mean()):.4f} "
                                        f"gate={_seg_gate_b if _seg_gate_b is not None else 'N/A'}"
                                    )
                        except Exception:
                            pass
                        if bool(getattr(cfg, "USE_DUAL_SPACE_DECOUPLE", False)) and isinstance(hierarchical_parts, dict):
                            _z_cls_now = hierarchical_parts.get("z_cls", None)
                            _z_metric_now = hierarchical_parts.get("z_metric", None)
                            if torch.is_tensor(_z_cls_now) and torch.is_tensor(_z_metric_now):
                                _z_cls_n = F.normalize(_z_cls_now.detach().float(), dim=-1)
                                _z_met_n = F.normalize(_z_metric_now.detach().float(), dim=-1)
                                _dual_cos = float(F.cosine_similarity(_z_cls_n, _z_met_n, dim=-1).mean().cpu().item())
                                _dual_cls_norm = float(_z_cls_now.detach().float().norm(dim=-1).mean().cpu().item())
                                _dual_met_norm = float(_z_metric_now.detach().float().norm(dim=-1).mean().cpu().item())
                                print(
                                    f"[DualSpace][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                    f"norm(cls/metric)=({_dual_cls_norm:.4f},{_dual_met_norm:.4f}) "
                                    f"cos(cls,metric)={_dual_cos:.4f}"
                                )
                        if bool(getattr(cfg, "USE_TEXT_PROMPT_TUNING", False)) and isinstance(current_prototypes, dict) and 1 in current_prototypes and 2 in current_prototypes:
                            _p1 = F.normalize(current_prototypes[1].detach().float(), dim=-1)
                            _p2 = F.normalize(current_prototypes[2].detach().float(), dim=-1)
                            _io_cos = float(torch.sum(_p1 * _p2).cpu().item())
                            _prompt_stats = {"scale": 0.0, "offset_norm_mean": 0.0, "offset_norm_max": 0.0}
                            if hasattr(fusion_decoder, "get_text_prompt_offset_norms"):
                                _prompt_stats = fusion_decoder.get_text_prompt_offset_norms()
                            print(
                                f"[PromptTuning][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                f"inner_outer_cos={_io_cos:.4f} "
                                f"scale={float(_prompt_stats.get('scale', 0.0)):.4f} "
                                f"offset_norm_mean={float(_prompt_stats.get('offset_norm_mean', 0.0)):.4f} "
                                f"offset_norm_max={float(_prompt_stats.get('offset_norm_max', 0.0)):.4f}"
                            )
                        _spec_freq_pos = batch.get("spec_freq_pos", None)
                        if bool(getattr(cfg, "PHYSICAL_FREQ_LOG_STATS", True)) and torch.is_tensor(_spec_freq_pos):
                            _fp = _spec_freq_pos.detach().float()
                            _pos_scale = float(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_SCALE", 0.0))
                            _fp_delta = torch.abs(_fp * _pos_scale)
                            _spec_abs = torch.abs(spec.detach().float()) if torch.is_tensor(spec) else None
                            _energy_ratio = float("nan")
                            if torch.is_tensor(_spec_abs):
                                _spec_mean = float(_spec_abs.mean().cpu().item())
                                _delta_mean = float(_fp_delta.mean().cpu().item())
                                _energy_ratio = _delta_mean / max(1e-8, _spec_mean)
                            print(
                                f"[FreqPosEnc][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                                f"min/max/mean=({float(_fp.min().cpu().item()):.4f},"
                                f"{float(_fp.max().cpu().item()):.4f},"
                                f"{float(_fp.mean().cpu().item()):.4f}) "
                                f"delta_scale={_pos_scale:.3f} energy_ratio={_energy_ratio:.4f}"
                            )
                    except Exception:
                        pass
                if _batch_should_log_grad:
                    _grad_terms = {
                        "fusion_cls": grad_probe_terms.get("fusion_cls_weighted", None),
                        "supcon": grad_probe_terms.get("supcon_weighted", None),
                        "fusion_supcon": grad_probe_terms.get("fusion_supcon_weighted", None),
                        "vicreg_or_decorr": grad_probe_terms.get("global_decorr_weighted", None),
                        "fusion_proto_align": grad_probe_terms.get("fusion_proto_align_weighted", None),
                        "nr_inter_cos_margin": grad_probe_terms.get("nr_inter_cos_margin_weighted", None),
                        "text_prompt_sep": grad_probe_terms.get("text_prompt_sep_weighted", None),
                    }
                    _grad_snapshot = _grad_contrib_snapshot(loss, _grad_terms)
                    if _grad_snapshot is not None:
                        _g_total_norm, _rows = _grad_snapshot
                        _rows_str = " | ".join(
                            [f"{_n}:g={_gn:.3e},r={_r:.3f},cos={_c:.3f}" for _n, _gn, _r, _c in _rows[:5]]
                        )
                        print(
                            f"[GradContrib][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"g_total={_g_total_norm:.3e} | {_rows_str}"
                        )
                    else:
                        _active_terms = [
                            _n for _n, _t in _grad_terms.items()
                            if torch.is_tensor(_t) and _t.requires_grad
                        ]
                        print(
                            f"[GradContrib][Epoch {current_epoch}][Batch {batch_idx + 1}/{len(train_loader)}] "
                            f"unavailable(active_terms={_active_terms})"
                        )
                if (not torch.is_tensor(loss)) or (not loss.requires_grad):
                    param = None
                    try:
                        for _m in (fusion_decoder, fusion_module, signal_encoder, spec_encoder):
                            for _p in _m.parameters():
                                if _p.requires_grad:
                                    param = _p
                                    break
                            if param is not None:
                                break
                    except Exception:
                        param = None
                    if param is not None:
                        loss = param.sum() * 0.0
                    else:
                        try:
                            loss = z_fusion.sum() * 0.0
                        except Exception:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                    if isinstance(loss_dict, dict):
                        loss_dict["no_grad_loss"] = 1.0

                # ===== ATT396: Add Feature Decoupling auxiliary losses (non-AMP path) =====
                if _dc_active and _z_spec_fault is not None and _z_signal_fault is not None:
                    try:
                        _cond_ids = _encode_cond_ids(domains, conditions, device)
                        _grl_lam = float(getattr(cfg, "DECOUPLE_ADV_LAMBDA_MAX", 1.0))
                        if current_epoch < int(getattr(cfg, "DECOUPLE_START_EPOCH", 1)) + 5:
                            _progress = float(current_epoch - int(getattr(cfg, "DECOUPLE_START_EPOCH", 1))) / 5.0
                            _grl_lam *= max(0.0, min(1.0, _progress))
                        _dc_loss, _dc_dict = _compute_decouple_losses(
                            z_fault_spec=_z_spec_fault,
                            z_cond_spec=_z_spec_cond,
                            z_fault_signal=_z_signal_fault,
                            z_cond_signal=_z_signal_cond,
                            condition_ids=_cond_ids,
                            condition_classifier=condition_classifier,
                            grl_lambda=_grl_lam,
                            w_cond=float(getattr(cfg, "DECOUPLE_COND_WEIGHT", 0.10)),
                            w_adv=float(getattr(cfg, "DECOUPLE_ADV_WEIGHT", 0.10)),
                            w_ortho=float(getattr(cfg, "DECOUPLE_ORTHO_WEIGHT", 0.05)),
                        )
                        loss = loss + _dc_loss
                        if isinstance(loss_dict, dict):
                            loss_dict.update(_dc_dict)
                        if batch_idx == 0 and current_epoch <= 5:
                            print(
                                f"[ATT396][Decouple-Loss] E{current_epoch} "
                                f"cond={_dc_dict.get('decouple_cond_loss', 0):.4f}/"
                                f"acc={_dc_dict.get('decouple_cond_acc', 0):.3f}, "
                                f"adv={_dc_dict.get('decouple_adv_loss', 0):.4f}/"
                                f"acc={_dc_dict.get('decouple_adv_acc', 0):.3f}, "
                                f"ortho={_dc_dict.get('decouple_ortho_loss', 0):.4f}, "
                                f"grl_lam={_grl_lam:.3f}"
                            )
                    except Exception as _dc_e:
                        if batch_idx == 0:
                            print(f"[ATT396][Decouple-Loss] Error: {_dc_e}")

                # ATT397: Episodic Cross-Condition Prototype Loss (non-AMP path)
                _use_episodic = bool(getattr(cfg, "USE_EPISODIC_TRAINING", False))
                if _use_episodic and conditions is not None:
                    try:
                        from training.episodic_sampler import split_batch_by_condition
                        _ep_split = str(getattr(cfg, "EPISODIC_SPLIT_KEY", "auto"))
                        if _ep_split == "auto":
                            _ep_split = "condition"
                        _s_mask, _q_mask, _g_ids = split_batch_by_condition(
                            batch, split_key=_ep_split, batch_idx=batch_idx)
                        _s_mask = _s_mask.to(device)
                        _q_mask = _q_mask.to(device)
                        if _s_mask.sum() >= 2 and _q_mask.sum() >= 2:
                            _ep_feat = z_spec if z_spec is not None else z_signal
                            _ep_feat_norm = F.normalize(_ep_feat.detach(), dim=-1)
                            _ep_feat_q = F.normalize((z_spec if z_spec is not None else z_signal), dim=-1)
                            _s_feat = _ep_feat_norm[_s_mask]
                            _s_labels = labels[_s_mask]
                            _q_feat = _ep_feat_q[_q_mask]
                            _q_labels = labels[_q_mask]
                            _ep_classes = _s_labels.unique()
                            _protos = []
                            _proto_labels = []
                            for _c in _ep_classes:
                                _c_mask = (_s_labels == _c)
                                if _c_mask.sum() > 0:
                                    _protos.append(_s_feat[_c_mask].mean(dim=0))
                                    _proto_labels.append(_c)
                            if len(_protos) >= 2:
                                _protos_t = torch.stack(_protos)
                                _proto_labels_t = torch.stack(_proto_labels)
                                _ep_temp = float(getattr(cfg, "EPISODIC_TEMPERATURE", 0.07))
                                _ep_sims = _q_feat @ _protos_t.T / _ep_temp
                                _ep_targets = torch.zeros(_q_feat.size(0), dtype=torch.long, device=device)
                                for _pi, _pl in enumerate(_proto_labels_t):
                                    _ep_targets[_q_labels == _pl] = _pi
                                _valid_q = torch.zeros(_q_feat.size(0), dtype=torch.bool, device=device)
                                for _pl in _proto_labels_t:
                                    _valid_q |= (_q_labels == _pl)
                                if _valid_q.sum() >= 2:
                                    _ep_loss = F.cross_entropy(_ep_sims[_valid_q], _ep_targets[_valid_q])
                                    _ep_w = float(getattr(cfg, "EPISODIC_PROTO_LOSS_WEIGHT", 0.5))
                                    loss = loss + _ep_w * _ep_loss
                                    if isinstance(loss_dict, dict):
                                        loss_dict["episodic_proto_loss"] = float(_ep_loss.detach())
                                    if batch_idx == 0:
                                        print(
                                            f"[ATT397-Episodic] E{current_epoch}B{batch_idx} "
                                            f"ep_loss={float(_ep_loss.detach()):.4f} "
                                            f"n_support={int(_s_mask.sum())} n_query={int(_q_mask.sum())} "
                                            f"n_proto_classes={len(_protos)} valid_query={int(_valid_q.sum())}"
                                        )
                    except Exception as _ep_e:
                        if batch_idx == 0:
                            print(f"[ATT397-Episodic] Error: {_ep_e}")

                # ===== ATT405/407: Training-time domain alignment on z_fusion (MMD/CORAL) =====
                _fda_enable2 = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_ENABLE", False))
                _fda_disable_task1_2 = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1", True))
                _is_task1_2 = ("task2" not in _phase_tag) and len(_phase_tag) > 0
                if _fda_disable_task1_2 and _is_task1_2:
                    _fda_enable2 = False
                if _fda_enable2 and z_fusion is not None and domain_ids is not None:
                    try:
                        _fda_start2 = int(getattr(cfg, "FUSION_DOMAIN_ALIGN_START_EPOCH", 3))
                        if current_epoch >= _fda_start2:
                            _fda_w_base2 = float(getattr(cfg, "FUSION_DOMAIN_ALIGN_WEIGHT", 0.05))
                            _fda_warmup2 = int(getattr(cfg, "FUSION_DOMAIN_ALIGN_WARMUP_EPOCHS", 3))
                            _fda_progress2 = min(1.0, float(current_epoch - _fda_start2) / max(1, _fda_warmup2))
                            _fda_w2 = _fda_w_base2 * _fda_progress2
                            _fda_detach2 = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_DETACH", False))
                            _fda_min_n2 = int(getattr(cfg, "FUSION_DOMAIN_ALIGN_MIN_SAMPLES", 4))
                            _fda_mode2 = str(getattr(cfg, "FUSION_DOMAIN_ALIGN_LOSS", "mmd")).lower()

                            _zf2 = z_fusion.detach() if _fda_detach2 else z_fusion
                            _unique_doms2 = domain_ids.unique()
                            _fda_losses2 = []
                            for _i_d2 in range(len(_unique_doms2)):
                                for _j_d2 in range(_i_d2 + 1, len(_unique_doms2)):
                                    _mask_i2 = (domain_ids == _unique_doms2[_i_d2])
                                    _mask_j2 = (domain_ids == _unique_doms2[_j_d2])
                                    if _mask_i2.sum() >= _fda_min_n2 and _mask_j2.sum() >= _fda_min_n2:
                                        _fi2 = _zf2[_mask_i2]
                                        _fj2 = _zf2[_mask_j2]
                                        if _fda_mode2 == "coral":
                                            _fda_losses2.append(coral_covariance_alignment_loss(_fi2, _fj2))
                                        else:
                                            _fda_losses2.append(multi_kernel_mmd_loss(_fi2, _fj2))
                            if len(_fda_losses2) > 0:
                                _fda_total2 = sum(_fda_losses2) / len(_fda_losses2)
                                _fda_loss2 = _fda_w2 * _fda_total2
                                loss = loss + _fda_loss2
                                if isinstance(loss_dict, dict):
                                    loss_dict["fusion_domain_align"] = float(_fda_loss2.detach())
                                if batch_idx == 0:
                                    print(
                                        f"[ATT406-DomainAlign] E{current_epoch}B{batch_idx} "
                                        f"mode={_fda_mode2} pairs={len(_fda_losses2)} "
                                        f"raw={float(_fda_total2.detach()):.6f} "
                                        f"weighted={float(_fda_loss2.detach()):.6f} "
                                        f"w={_fda_w2:.4f}(warmup={_fda_progress2:.2f})"
                                    )
                    except Exception as _fda_e2:
                        if batch_idx == 0:
                            print(f"[ATT406-DomainAlign] Error: {_fda_e2}")

                # ATT447: Asymmetric Fusion→Spec Distillation (task2 branch)
                if _spec_distill_active and current_epoch >= _distill_start_epoch and spec_logits is not None and z_fusion is not None:
                    try:
                        _proto_matrix2 = None
                        if learnable_proto_module is not None:
                            _proto_matrix2 = learnable_proto_module.forward().detach()
                        elif isinstance(current_prototypes, dict):
                            _rows2 = [current_prototypes[k].detach() for k in sorted(current_prototypes.keys())]
                            _proto_matrix2 = torch.stack(_rows2, dim=0)
                        elif torch.is_tensor(current_prototypes):
                            _proto_matrix2 = current_prototypes.detach()
                        if _proto_matrix2 is not None:
                            _z_fus_d2 = z_fusion.detach()
                            _fus_cos2 = torch.matmul(F.normalize(_z_fus_d2, dim=-1), F.normalize(_proto_matrix2, dim=-1).t())
                            _soft_t2 = F.softmax(_fus_cos2 / _distill_temperature, dim=-1)
                            if _distill_hf_only:
                                _fus_hf2 = _fus_cos2 - _fus_cos2.mean(dim=-1, keepdim=True)
                                _spec_hf2 = spec_logits - spec_logits.mean(dim=-1, keepdim=True)
                                _soft_t2_hf = F.softmax(_fus_hf2 / _distill_temperature, dim=-1)
                                _spec_lp2 = F.log_softmax(_spec_hf2 / _distill_temperature, dim=-1)
                                _kl2 = F.kl_div(_spec_lp2, _soft_t2_hf, reduction='batchmean')
                            else:
                                _spec_lp2 = F.log_softmax(spec_logits / _distill_temperature, dim=-1)
                                _kl2 = F.kl_div(_spec_lp2, _soft_t2, reduction='batchmean')
                            _dl2 = _distill_alpha * (_distill_temperature ** 2) * _kl2
                            loss = loss + _dl2
                            if isinstance(loss_dict, dict):
                                loss_dict["spec_distill_kl"] = float(_dl2.detach().cpu().item())
                            if batch_idx % 50 == 0:
                                _hf2 = " HF" if _distill_hf_only else ""
                                print(f"[ATT449-Distill-T2{_hf2}][E{current_epoch}B{batch_idx}] KL={float(_kl2):.4f} weighted={float(_dl2):.4f}")
                    except Exception as _dl2_e:
                        if batch_idx == 0:
                            print(f"[ATT447-Distill-T2] Error: {_dl2_e}")

                # ATT448/ATT449: RKD + Fault-Focused RKD (task2 branch)
                if _rkd_enable and current_epoch >= _rkd_start_epoch and z_fusion is not None and z_spec is not None:
                    try:
                        _zfr2 = z_fusion.detach()
                        _zsr2 = extract_pre_l2_feature(z_spec)
                        if _zsr2 is None:
                            _zsr2 = z_spec
                        _Br2 = _zfr2.size(0)
                        if _Br2 >= 4:
                            _rkd2 = torch.tensor(0.0, device=_zfr2.device, dtype=_zfr2.dtype)
                            _gw2 = 1.0 - _rkd_fault_focus_w if _rkd_fault_focus_w > 0 else 1.0
                            if _rkd_dist_w > 0:
                                _df2 = torch.cdist(_zfr2, _zfr2, p=2)
                                _ds2 = torch.cdist(_zsr2, _zsr2, p=2)
                                _mf2 = _df2.mean().clamp(min=1e-6)
                                _ms2 = _ds2.mean().clamp(min=1e-6)
                                _rkd2 = _rkd2 + _gw2 * _rkd_dist_w * F.smooth_l1_loss(_ds2 / _ms2, (_df2 / _mf2).detach())
                            if _rkd_angle_w > 0 and _Br2 >= 3:
                                _nt2 = min(_Br2, 32)
                                _ix2 = torch.randperm(_Br2, device=_zfr2.device)[:_nt2]
                                _ia2 = _ix2
                                _ib2 = _ix2[torch.randperm(_nt2, device=_zfr2.device)]
                                _ic2 = _ix2[torch.randperm(_nt2, device=_zfr2.device)]
                                _v2 = (_ia2 != _ib2) & (_ia2 != _ic2) & (_ib2 != _ic2)
                                if _v2.sum() >= 3:
                                    _a2, _b2, _c2 = _ia2[_v2], _ib2[_v2], _ic2[_v2]
                                    _efab2 = F.normalize(_zfr2[_b2] - _zfr2[_a2], dim=-1)
                                    _efac2 = F.normalize(_zfr2[_c2] - _zfr2[_a2], dim=-1)
                                    _cf2 = (_efab2 * _efac2).sum(-1)
                                    _esab2 = F.normalize(_zsr2[_b2] - _zsr2[_a2], dim=-1)
                                    _esac2 = F.normalize(_zsr2[_c2] - _zsr2[_a2], dim=-1)
                                    _cs2 = (_esab2 * _esac2).sum(-1)
                                    _rkd2 = _rkd2 + _gw2 * _rkd_angle_w * F.smooth_l1_loss(_cs2, _cf2.detach())
                            # FF-RKD for task2
                            if _rkd_fault_focus_w > 0 and _rkd_dist_w > 0:
                                _fm2 = (labels != 0)
                                _nf2 = int(_fm2.sum())
                                if _nf2 >= 4:
                                    _zff2 = _zfr2[_fm2]
                                    _zsf2 = _zsr2[_fm2]
                                    _dff2 = torch.cdist(_zff2, _zff2, p=2)
                                    _dsf2 = torch.cdist(_zsf2, _zsf2, p=2)
                                    _mff2 = _dff2.mean().clamp(min=1e-6)
                                    _msf2 = _dsf2.mean().clamp(min=1e-6)
                                    _rkd2 = _rkd2 + _rkd_fault_focus_w * _rkd_dist_w * F.smooth_l1_loss(_dsf2 / _msf2, (_dff2 / _mff2).detach())
                            loss = loss + _rkd2
                            if isinstance(loss_dict, dict):
                                loss_dict["rkd_total"] = float(_rkd2.detach().cpu().item())
                            if batch_idx % 50 == 0:
                                print(f"[ATT449-RKD-T2][E{current_epoch}B{batch_idx}] total={float(_rkd2):.4f} B={_Br2}")
                    except Exception as _rkd2_e:
                        if batch_idx == 0:
                            print(f"[ATT449-RKD-T2] Error: {_rkd2_e}")

                # ATT460 / ATT470: Cross-modal cosine alignment (Task2 branch)
                _cm_align_w_t2 = float(getattr(cfg, "FUSION_CROSS_MODAL_ALIGN_WEIGHT", 0.0))
                if _cm_align_w_t2 > 0 and hasattr(fusion_module, 'get_cross_modal_align_loss'):
                    try:
                        _cm_loss2 = fusion_module.get_cross_modal_align_loss()
                        if torch.is_tensor(_cm_loss2) and _cm_loss2.requires_grad:
                            _cm_loss_wt2 = _cm_align_w_t2 * _cm_loss2
                            if _ablation_gate_bank is not None:
                                _cm_loss_wt2 = _ablation_gate_bank.scale_loss(
                                    "gidc_cross_modal_align", _cm_loss_wt2
                                )
                            loss = loss + _cm_loss_wt2
                            if isinstance(loss_dict, dict):
                                loss_dict["cross_modal_align"] = float(_cm_loss_wt2.detach().cpu().item())
                            if batch_idx % 50 == 0:
                                _fus_nm2 = type(fusion_module).__name__
                                print(f"[CrossModalAlign-T2][{_fus_nm2}][E{current_epoch}B{batch_idx}] "
                                      f"cos={float((1.0 - _cm_loss2)):.4f} "
                                      f"weighted={float(_cm_loss_wt2):.4f}")
                    except Exception as _cm2_e:
                        if batch_idx == 0:
                            print(f"[ATT460-CMAlign-T2] Error: {_cm2_e}")

                _fus_detach_b0_t2 = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
                if _fus_detach_b0_t2 and batch_idx == 0 and (current_epoch == 1 or current_epoch % 10 == 0):
                    print(
                        f"[ATT471-AGIT-T2][E{current_epoch}B0] fusion_forward_detach_encoder_inputs=1 "
                        f"=> fusion losses do not backprop to encoders"
                    )
                if bool(getattr(cfg, "DUAL_CONTRACT_ENABLE", False)) and batch_idx == 0 and (current_epoch == 1 or current_epoch % 10 == 0):
                    print(
                        f"[ATT475-DualContract-T2][E{current_epoch}B0] SplitArch={bool(getattr(cfg, 'SPLIT_ARCH_ENABLE', False))} "
                        f"fus_fwd_detach={_fus_detach_b0_t2} dual_ckpt={bool(getattr(cfg, 'DUAL_CONTRACT_SAVE_EXTRA_CKPT', False))}"
                    )

                optimizer.zero_grad()
                _pcgrad_active2 = bool(getattr(cfg, "USE_PCGRAD", False))
                _pcgrad_applied2 = False
                if _pcgrad_active2 and (not supcon_stage1_now) and isinstance(loss_dict, dict):
                    try:
                        from losses.pcgrad import pcgrad_backward as _pcgrad_fn2
                        _tt2 = loss_dict.get("_tensor_terms", {})
                        _g2_ce = []
                        for _k2 in ("fusion_cls_weighted", "signal_ce_weighted", "spec_ce_weighted"):
                            _t2 = _tt2.get(_k2)
                            if torch.is_tensor(_t2) and _t2.requires_grad:
                                _g2_ce.append(_t2)
                        _g2_pacl = []
                        for _k2 in ("fusion_proto_align_weighted", "fusion_alignment_weighted",
                                     "pacl_spec_weighted", "pacl_fus_weighted"):
                            _t2 = _tt2.get(_k2)
                            if torch.is_tensor(_t2) and _t2.requires_grad:
                                _g2_pacl.append(_t2)
                        _grp2_ce = sum(_g2_ce) if _g2_ce else None
                        _grp2_pacl = sum(_g2_pacl) if _g2_pacl else None
                        _groups2 = [g for g in [_grp2_ce, _grp2_pacl] if g is not None]
                        if len(_groups2) >= 2:
                            _sp2 = [p for pg in optimizer.param_groups for p in pg["params"] if p.requires_grad]
                            _pcg2 = _pcgrad_fn2(_groups2, _sp2, loss_names=["CE", "PACL"], retain_graph=True)
                            _pcgrad_applied2 = True
                            if batch_idx % int(getattr(cfg, "PCGRAD_LOG_INTERVAL", 50)) == 0:
                                print(f"[PCGrad][E{current_epoch}B{batch_idx}] conflicts={_pcg2['n_conflicts']}/{_pcg2['n_pairs']} ratio={_pcg2['conflict_ratio']:.3f}")
                    except Exception as _pcg2_e:
                        if batch_idx == 0:
                            print(f"[PCGrad] fallback: {_pcg2_e}")
                        _pcgrad_applied2 = False
                # ATT409: Auxiliary pre-norm CE loss (non-AMP path)
                if _aux_prenorm_enable and _aux_prenorm_head is not None:
                    try:
                        _apn_z2 = extract_pre_l2_feature(z_fusion) if torch.is_tensor(z_fusion) else None
                        if _apn_z2 is not None:
                            _apn_logits2 = _aux_prenorm_head(_apn_z2)
                            _apn_loss2 = torch.nn.functional.cross_entropy(_apn_logits2, labels)
                            _apn_weighted2 = _aux_prenorm_weight * _apn_loss2
                            loss = loss + _apn_weighted2
                            loss_dict["aux_prenorm_ce"] = float(_apn_weighted2.detach().cpu().item())
                    except Exception as _apn_e2:
                        if batch_idx == 0:
                            print(f"[ATT409-AuxPrenorm] non-AMP error: {_apn_e2}")

                # ===== ATT465-MLSA: Multi-Level Spec-Text Alignment Loss (non-AMP) =====
                if _mlsa_active and text_prototypes is not None:
                    _z_mid2 = getattr(spec_encoder, '_last_z_mid', None)
                    if _z_mid2 is not None and torch.is_tensor(_z_mid2) and _z_mid2.requires_grad:
                        _mlsa_warmup_frac2 = min(1.0, max(0.0, float(current_epoch) / max(1, _mlsa_warmup_epochs)))
                        _mlsa_w_eff2 = _mlsa_weight * _mlsa_warmup_frac2
                        if _mlsa_w_eff2 > 0.0:
                            _z_mid_n2 = F.normalize(_z_mid2, dim=-1)
                            _proto_keys2 = sorted(text_prototypes.keys())
                            _proto_mat2 = torch.stack([text_prototypes[k] for k in _proto_keys2]).to(_z_mid2.device)
                            _proto_mat_n2 = F.normalize(_proto_mat2, dim=-1)
                            _cos_sim2 = torch.mm(_z_mid_n2, _proto_mat_n2.t())
                            _mlsa_labels2 = labels.detach().cpu().tolist()
                            _mlsa_pos2 = torch.stack([_cos_sim2[i, _mlsa_labels2[i]] for i in range(len(_mlsa_labels2))])
                            _mlsa_neg_mask2 = torch.ones_like(_cos_sim2, dtype=torch.bool)
                            for i in range(len(_mlsa_labels2)):
                                _mlsa_neg_mask2[i, _mlsa_labels2[i]] = False
                            _mlsa_neg_max2 = _cos_sim2.masked_fill(~_mlsa_neg_mask2, -1e9).max(dim=-1).values
                            _mlsa_margin_loss2 = F.relu(_mlsa_margin - (_mlsa_pos2 - _mlsa_neg_max2)).mean()
                            _mlsa_weighted2 = _mlsa_w_eff2 * _mlsa_margin_loss2
                            loss = loss + _mlsa_weighted2
                            loss_dict["mlsa_mid"] = float(_mlsa_weighted2.detach().cpu().item())

                if not _pcgrad_applied2:
                    loss.backward()
                if grad_norm_log_enabled:
                    try:
                        _grad_norm = float(_compute_global_grad_norm())
                        epoch_grad_norm_sum += _grad_norm
                        epoch_grad_norm_steps += 1
                        if isinstance(loss_dict, dict) and (batch_idx == 0 or ((batch_idx + 1) % grad_norm_log_step_stride == 0)):
                            loss_dict["grad_norm"] = _grad_norm
                    except Exception:
                        pass
                if grad_norm_detail_log_enabled:
                    try:
                        _detail_sampled = (batch_idx == 0 or ((batch_idx + 1) % grad_norm_detail_step_stride == 0))
                        if _detail_sampled:
                            _g_sig, _c_sig = _compute_module_grad_norm(signal_encoder)
                            _g_spec, _c_spec = _compute_module_grad_norm(spec_encoder)
                            _g_fus, _c_fus = _compute_module_grad_norm(fusion_module)
                            _g_dec, _c_dec = _compute_module_grad_norm(fusion_decoder)
                            epoch_grad_norm_signal_sum += _g_sig
                            epoch_grad_norm_spec_sum += _g_spec
                            epoch_grad_norm_fusion_sum += _g_fus
                            epoch_grad_norm_decoder_sum += _g_dec
                            epoch_grad_norm_signal_count_sum += _c_sig
                            epoch_grad_norm_spec_count_sum += _c_spec
                            epoch_grad_norm_fusion_count_sum += _c_fus
                            epoch_grad_norm_decoder_count_sum += _c_dec
                            epoch_grad_norm_detail_steps += 1
                            if _c_dec == 0 and batch_idx == 0:
                                _dec_diag = []
                                try:
                                    _cls_w = fusion_decoder.classifier.weight
                                    _dec_diag.append(f"cls.w.grad={'None' if _cls_w.grad is None else f'{float(_cls_w.grad.norm()):.6f}'}")
                                    _dec_diag.append(f"req_grad={_cls_w.requires_grad}")
                                except Exception:
                                    _dec_diag.append("cls.w=N/A")
                                _dec_diag.append(f"total_req={sum(1 for p in fusion_decoder.parameters() if p.requires_grad)}")
                                _dec_diag.append(f"total_has_grad={sum(1 for p in fusion_decoder.parameters() if p.grad is not None)}")
                                print(f"[ATT378b-DecoderGradDiag][E{current_epoch}B{batch_idx}] decoder_n=0! {', '.join(_dec_diag)}")
                    except Exception:
                        pass
                # ATT461: OGM-GE Gradient Modulation — Task2 (non-AMP, after backward, before clip/step)
                _ogm_ge_on_t2 = bool(getattr(cfg, "OGM_GE_ENABLE", False))
                if _ogm_ge_on_t2 and current_epoch >= int(getattr(cfg, "OGM_GE_START_EPOCH", 2)):
                    try:
                        import math as _math_ogm2
                        _ogm_alpha2 = float(getattr(cfg, "OGM_GE_ALPHA", 1.0))
                        _g_sig_sq2 = sum(
                            p.grad.data.norm() ** 2
                            for p in signal_encoder.parameters() if p.grad is not None
                        )
                        _g_spec_sq2 = sum(
                            p.grad.data.norm() ** 2
                            for p in spec_encoder.parameters() if p.grad is not None
                        )
                        _g_sig_n2 = float(_g_sig_sq2.sqrt()) if torch.is_tensor(_g_sig_sq2) else float(_g_sig_sq2 ** 0.5)
                        _g_spec_n2 = float(_g_spec_sq2.sqrt()) if torch.is_tensor(_g_spec_sq2) else float(_g_spec_sq2 ** 0.5)
                        _ratio_raw2 = _g_sig_n2 / (_g_spec_n2 + 1e-8)
                        _r_sig2 = _ratio_raw2 / (1.0 + _ratio_raw2)
                        _r_spec2 = 1.0 / (1.0 + _ratio_raw2)
                        _excess_sig2 = max(_r_sig2 - 0.5, 0.0)
                        _excess_spec2 = max(_r_spec2 - 0.5, 0.0)
                        _coeff_sig2 = 1.0 - _math_ogm2.tanh(_ogm_alpha2 * _excess_sig2)
                        _coeff_spec2 = 1.0 - _math_ogm2.tanh(_ogm_alpha2 * _excess_spec2)
                        for p in signal_encoder.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(_coeff_sig2)
                        for p in spec_encoder.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(_coeff_spec2)
                                _ge_std2 = (1.0 - _coeff_spec2 ** 2) ** 0.5
                                if _ge_std2 > 1e-6:
                                    _pnorm2 = float(p.grad.data.norm())
                                    if _pnorm2 > 1e-10:
                                        p.grad.data.add_(torch.randn_like(p.grad.data) * (_ge_std2 * _pnorm2 * 0.01))
                        if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                            print(f"[ATT461-OGM-GE-T2][E{current_epoch}B{batch_idx}] "
                                  f"g_sig={_g_sig_n2:.4f} g_spec={_g_spec_n2:.4f} "
                                  f"ratio={_ratio_raw2:.3f} "
                                  f"coeff_sig={_coeff_sig2:.4f} coeff_spec={_coeff_spec2:.4f}")
                    except Exception as _ogm_e2:
                        if batch_idx == 0:
                            print(f"[ATT461-OGM-GE-T2] Error: {_ogm_e2}")
                if grad_clip_enabled:
                    try:
                        _clip_params = list(_iter_optimizer_grad_params(optimizer))
                        if len(_clip_params) > 0:
                            _pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=float(grad_clip_max_norm)))
                            _post_clip_norm = float(_compute_global_grad_norm())
                            epoch_grad_clip_batches += 1
                            epoch_grad_clip_pre_norm_sum += _pre_clip_norm
                            epoch_grad_clip_post_norm_sum += _post_clip_norm
                            epoch_grad_clip_param_count_sum += int(len(_clip_params))
                            if _pre_clip_norm > float(grad_clip_max_norm) + 1e-8:
                                epoch_grad_clip_clipped_batches += 1
                            if isinstance(loss_dict, dict) and (batch_idx == 0 or ((batch_idx + 1) % grad_norm_log_step_stride == 0)):
                                loss_dict["grad_clip_pre_norm"] = _pre_clip_norm
                                loss_dict["grad_clip_post_norm"] = _post_clip_norm
                    except Exception:
                        pass
                # ATT397: SAM two-step optimization (non-AMP path)
                _is_sam = hasattr(optimizer, 'first_step') and hasattr(optimizer, 'second_step')
                if _is_sam:
                    optimizer.first_step(zero_grad=True)
                    _sam_z_sig = signal_encoder(signal, signal_mask, domain_ids=domain_ids) if (
                        hasattr(signal_encoder, 'use_domain_residual') and signal_encoder.use_domain_residual
                    ) else signal_encoder(signal, signal_mask)
                    if isinstance(_sam_z_sig, tuple):
                        _sam_z_sig = _sam_z_sig[0]
                    _sam_z_spec_pair = spec_encoder(spec, spec_mask, domain_ids=domain_ids) if (
                        hasattr(spec_encoder, 'use_domain_residual') and spec_encoder.use_domain_residual
                    ) else spec_encoder(spec, spec_mask)
                    _sam_z_spec = _sam_z_spec_pair[0] if isinstance(_sam_z_spec_pair, tuple) else _sam_z_spec_pair
                    if hasattr(fusion_module, 'domain_embedding'):
                        _sam_z_fus = fusion_module(_sam_z_sig, _sam_z_spec, domain_ids=domain_ids)
                    else:
                        _sam_z_fus = fusion_module(_sam_z_sig, _sam_z_spec)
                    _sam_logits, _ = fusion_decoder(_sam_z_fus, domain_ids=domain_ids, return_parts=True)
                    _sam_loss = F.cross_entropy(_sam_logits, labels)
                    _sam_loss.backward()
                    if grad_clip_enabled:
                        _clip_params = list(_iter_optimizer_grad_params(optimizer))
                        if len(_clip_params) > 0:
                            torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=float(grad_clip_max_norm))
                    optimizer.second_step(zero_grad=True)
                    if batch_idx == 0 and current_epoch <= 3:
                        print(f"[ATT397-SAM] E{current_epoch}B{batch_idx} sam_loss={float(_sam_loss):.4f}")
                else:
                    optimizer.step()
                
                # 收集类别预测统计（用于自适应权重调整）
                if enable_class_monitoring:
                    import config as cfg
                    use_adaptive_weights = getattr(cfg, 'USE_ADAPTIVE_CLASS_WEIGHTS', True)
                    
                    if use_adaptive_weights:
                        with torch.no_grad():
                            # 获取预测结果
                            try:
                                if hasattr(fusion_logits, 'hierarchical_parts'):
                                    # 层次化模型：使用4类logits
                                    preds = torch.argmax(fusion_logits, dim=1)
                                else:
                                    preds = torch.argmax(fusion_logits, dim=1)
                                
                                # 更新每个类别的统计
                                unique_labels = labels.unique()
                                if len(unique_labels) > 0:
                                    for label_val in unique_labels:
                                        label_val = label_val.item()
                                        mask = (labels == label_val)
                                        if mask.sum() > 0:
                                            correct = (preds[mask] == labels[mask]).sum().item()
                                            total = mask.sum().item()
                                            if label_val not in class_stats:
                                                class_stats[label_val] = {"correct": 0, "total": 0}
                                            class_stats[label_val]["correct"] += correct
                                            class_stats[label_val]["total"] += total
                                            
                                            # 更新混淆统计（Confusion-Aware 动态惩罚）
                                            if label_val not in confusion_stats:
                                                confusion_stats[label_val] = {}
                                            preds_batch = preds[mask].cpu().numpy()
                                            for pred_val in preds_batch:
                                                pred_val = int(pred_val)
                                                if pred_val not in confusion_stats[label_val]:
                                                    confusion_stats[label_val][pred_val] = 0
                                                confusion_stats[label_val][pred_val] += 1
                            except Exception as e:
                                # 如果出现错误，记录但不中断训练
                                if current_epoch <= 3 and batch_idx == 0:
                                    print(f"[Adaptive Weights] Warning: Error collecting stats in epoch {current_epoch}, batch {batch_idx}: {e}")
                
                # Update expert tracker (if enabled)
                if use_moe and hasattr(spec_encoder, 'expert_tracker') and spec_encoder.expert_tracker is not None:
                    import config as cfg
                    if global_step % cfg.EXPERT_TRACKER_UPDATE_INTERVAL == 0:
                        # Update tracker with batch loss
                        for band_idx in range(len(moe_stats.get('expert_weights', []))):
                            spec_encoder.expert_tracker.update_from_batch(
                                domain_ids=domain_ids,
                                band_idx=band_idx,
                                expert_weights=moe_stats['expert_weights'][band_idx],
                                expert_outputs=moe_stats['expert_outputs'][band_idx],
                                batch_loss=loss.item()
                            )
                
                # 更新 EMA 权重（每步更新）
                if use_ema and ema_manager is not None:
                    global_step += 1
                    ema_manager.update_all(global_step, {
                        'signal_encoder': signal_encoder,
                        'spec_encoder': spec_encoder,
                        'fusion_module': fusion_module,
                        'fusion_decoder': fusion_decoder
                    })

            # ---- accumulate ----
            # Make accumulation robust for Stage1 (supervised-only) where semantic keys are absent.
            try:
                if "total" not in loss_dict:
                    loss_dict["total"] = float(loss.detach().cpu().item())
            except Exception:
                pass
            epoch_total += float(loss_dict.get("total", 0.0))
            epoch_signal += float(loss_dict.get("signal", 0.0))
            epoch_spec += float(loss_dict.get("spec", 0.0))
            epoch_fusion += float(loss_dict.get("fusion", 0.0))
            epoch_fusion_cls += float(loss_dict.get("_info_fusion_cls", loss_dict.get("fusion_cls", 0.0)))
            epoch_margin += float(loss_dict.get("margin", 0.0))
            epoch_fusion_proto_align += float(loss_dict.get("fusion_proto_align", 0.0))
            epoch_fusion_supcon += float(loss_dict.get("fusion_supcon", 0.0))
            epoch_metric_lp_ce += float(loss_dict.get("metric_lp_ce", 0.0))
            epoch_nr_inter_cos_margin += float(loss_dict.get("nr_inter_cos_margin", 0.0))
            epoch_nr_inter_cos_hard += float(loss_dict.get("nr_inter_cos_hard", 0.0))
            epoch_fusion_var_floor += float(loss_dict.get("fusion_var_floor", 0.0))
            epoch_fusion_vicreg += float(loss_dict.get("fusion_vicreg", 0.0))
            epoch_fusion_inter_cos_margin += float(loss_dict.get("fusion_inter_cos_margin", 0.0))
            _pm_keys = ("normal_margin", "rolling_absorb_margin", "rolling_vs_normal_margin",
                        "inner_vs_rolling_margin", "outer_vs_rolling_margin",
                        "inner_vs_normal_margin", "outer_vs_normal_margin",
                        "normal_vs_inner_margin", "outer_vs_inner_margin", "rolling_vs_inner_margin",
                        "outer_margin", "inner_margin")
            _pm_batch = sum(float(loss_dict.get(k, 0.0)) for k in _pm_keys)
            epoch_pairwise_margin_sum += _pm_batch
            epoch_pairwise_margin_count += 1
            if "roll_dann" in loss_dict:
                epoch_roll_dann += float(loss_dict.get("roll_dann", 0.0))
            if "roll_decorr" in loss_dict:
                epoch_roll_decorr += float(loss_dict.get("roll_decorr", 0.0))
            if "roll_decorr_in" in loss_dict:
                epoch_roll_decorr_in += float(loss_dict.get("roll_decorr_in", 0.0))
            if "global_decorr" in loss_dict:
                epoch_global_decorr += float(loss_dict.get("global_decorr", 0.0))
            if "fishr" in loss_dict:
                epoch_fishr += float(loss_dict.get("fishr", 0.0))
            if "ccpa" in loss_dict:
                epoch_ccpa += float(loss_dict.get("ccpa", 0.0))
            if "global_uniformity" in loss_dict:
                epoch_global_uniformity += float(loss_dict.get("global_uniformity", 0.0))
            
            if "pacl_spec" in loss_dict or "pacl_fusion" in loss_dict:
                epoch_pacl_spec += float(loss_dict.get("pacl_spec", 0.0))
                epoch_pacl_fusion += float(loss_dict.get("pacl_fusion", 0.0))
                epoch_pacl_spec_acc += float(loss_dict.get("_info_pacl_spec_acc", 0.0))
                epoch_pacl_fusion_acc += float(loss_dict.get("_info_pacl_fus_acc", 0.0))
                epoch_pacl_batches += 1
                _pc_acc = loss_dict.get("_info_pacl_spec_per_class_acc", {})
                _pc_cos = loss_dict.get("_info_pacl_spec_per_class_pos_cos", {})
                if isinstance(_pc_acc, dict):
                    for _c, _v in _pc_acc.items():
                        _c = int(_c)
                        epoch_pacl_spec_per_class_acc[_c] = epoch_pacl_spec_per_class_acc.get(_c, 0.0) + float(_v)
                        epoch_pacl_spec_per_class_count[_c] = epoch_pacl_spec_per_class_count.get(_c, 0) + 1
                if isinstance(_pc_cos, dict):
                    for _c, _v in _pc_cos.items():
                        _c = int(_c)
                        epoch_pacl_spec_per_class_pos_cos[_c] = epoch_pacl_spec_per_class_pos_cos.get(_c, 0.0) + float(_v)
            if "adapter_pacl" in loss_dict:
                epoch_adapter_loss += float(loss_dict.get("adapter_pacl", 0.0))
                epoch_adapter_acc += float(loss_dict.get("_info_adapter_acc", 0.0))
                epoch_adapter_batches += 1
            # ATT429-P2: accumulate hierarchical CE losses
            if "hierarchical_nf_ce" in loss_dict or "hierarchical_fault_ce" in loss_dict:
                epoch_h_nf_ce += float(loss_dict.get("hierarchical_nf_ce", 0.0))
                epoch_h_fault_ce += float(loss_dict.get("hierarchical_fault_ce", 0.0))
                epoch_h_batches += 1
            # ATT430-S1/S2/S3: accumulate disentangle + variance
            if any(k in loss_dict for k in ("dual_orth", "dual_dc_spe", "dual_confusion", "spec_variance")):
                epoch_dual_orth += float(loss_dict.get("dual_orth", 0.0))
                epoch_dual_dc_spe += float(loss_dict.get("dual_dc_spe", 0.0))
                epoch_dual_dc_inv_train += float(loss_dict.get("dual_dc_inv_train", 0.0))
                epoch_dual_confusion += float(loss_dict.get("dual_confusion", 0.0))
                epoch_spec_variance += float(loss_dict.get("spec_variance", 0.0))
                epoch_dual_batches += 1
            if use_moe and "moe_total" in loss_dict:
                epoch_moe_total += loss_dict["moe_total"]
                epoch_moe_load_balance += loss_dict["moe_load_balance"]
                epoch_moe_entropy += loss_dict["moe_entropy"]
                epoch_moe_diversity += loss_dict["moe_diversity"]
            
            if use_consistency_regularization and "consistency" in loss_dict:
                epoch_consistency += loss_dict["consistency"]

            if dataflow_audit_enabled and domain_ids is not None:
                try:
                    _did = domain_ids.detach().cpu().tolist()
                    _lbl = labels.detach().cpu().tolist()
                    _cond = conditions if isinstance(conditions, (list, tuple)) else None
                    for _i, _d in enumerate(_did):
                        _d = int(_d)
                        epoch_domain_hist[_d] = int(epoch_domain_hist.get(_d, 0)) + 1
                        _label_hist = epoch_domain_label_hist.setdefault(_d, {})
                        _y = int(_lbl[_i]) if _i < len(_lbl) else -1
                        _label_hist[_y] = int(_label_hist.get(_y, 0)) + 1
                        if _cond is not None and _i < len(_cond):
                            _cond_str = str(_cond[_i])
                            if _d not in epoch_domain_cond_examples:
                                epoch_domain_cond_examples[_d] = []
                            if len(epoch_domain_cond_examples[_d]) < 3 and _cond_str not in epoch_domain_cond_examples[_d]:
                                epoch_domain_cond_examples[_d].append(_cond_str)
                except Exception:
                    pass


        if skipped_bad_batches > 0:
            print(f"[Stability] Epoch {current_epoch}: skipped {skipped_bad_batches} pathological batch(es).")

        # 如果使用贝叶斯原型且处于更新阶段，在epoch结束时更新prototypes；否则保持冻结
        # ATT375: ETF prototypes are fixed geometric structures—never overwrite them with Bayesian posterior
        if bayesian_update_active and not use_etf:
            class_embeddings_device = {}
            for label, emb_list in class_embeddings.items():
                if len(emb_list) > 0:
                    all_tensors = []
                    for emb in emb_list:
                        if isinstance(emb, torch.Tensor):
                            all_tensors.append(emb)
                    if len(all_tensors) > 0:
                        stacked = torch.cat(all_tensors, dim=0).to(device)
                        class_embeddings_device[label] = [stacked]
                    else:
                        class_embeddings_device[label] = []
                else:
                    class_embeddings_device[label] = []
            
            bayesian_prototypes = build_bayesian_text_prototypes(
                text_encoder, class_prompts, class_embeddings_device, device,
                prior_strength=bayesian_prior_strength
            )
            
            text_prototypes = extract_posterior_prototypes(bayesian_prototypes)
            # ATT390-S2/FIX: Sync training_proto_tensor after Bayesian update
            if not use_etf and text_prototypes is not None and isinstance(text_prototypes, dict):
                _num_cls_sync = len(Config.FAULT_LABEL_TO_SEMANTIC)
                _rows_sync = []
                for _ci_sync in range(_num_cls_sync):
                    if _ci_sync in text_prototypes and torch.is_tensor(text_prototypes[_ci_sync]):
                        _rows_sync.append(F.normalize(text_prototypes[_ci_sync].to(device), dim=0))
                    else:
                        _rows_sync.append(torch.zeros(512, device=device))
                training_proto_tensor = torch.stack(_rows_sync, dim=0)
                if current_epoch <= 3:
                    print(f"[ATT390-S2] Bayesian update synced training_proto_tensor (epoch {current_epoch})")
        elif bayesian_update_active and use_etf:
            if current_epoch == 1:
                print(f"[ATT375] Bayesian update SKIPPED: ETF prototypes are immutable (use_etf=True)")

        n = len(train_loader)
        avg_total = epoch_total / n
        avg_signal = epoch_signal / n
        avg_spec = epoch_spec / n
        avg_fusion = epoch_fusion / n
        avg_fusion_cls = epoch_fusion_cls / n
        avg_margin = epoch_margin / n
        avg_fusion_proto_align = epoch_fusion_proto_align / n
        avg_fusion_supcon = epoch_fusion_supcon / n
        avg_metric_lp_ce = epoch_metric_lp_ce / n
        avg_nr_inter_cos_margin = epoch_nr_inter_cos_margin / n
        avg_nr_inter_cos_hard = epoch_nr_inter_cos_hard / n
        avg_fusion_var_floor = epoch_fusion_var_floor / n
        avg_fusion_vicreg = epoch_fusion_vicreg / n
        avg_fusion_inter_cos_margin = epoch_fusion_inter_cos_margin / n
        avg_grad_norm = epoch_grad_norm_sum / max(1, epoch_grad_norm_steps)

        history["total"].append(avg_total)
        history["signal"].append(avg_signal)
        history["spec"].append(avg_spec)
        history["fusion"].append(avg_fusion)
        history["fusion_cls"].append(avg_fusion_cls)
        history["margin"].append(avg_margin)
        history["fusion_proto_align"].append(avg_fusion_proto_align)
        history["fusion_supcon"].append(avg_fusion_supcon)
        history["metric_lp_ce"].append(avg_metric_lp_ce)
        history["nr_inter_cos_margin"].append(avg_nr_inter_cos_margin)
        history["nr_inter_cos_hard"].append(avg_nr_inter_cos_hard)
        history["fusion_var_floor"].append(avg_fusion_var_floor)
        history["fusion_vicreg"].append(avg_fusion_vicreg)
        history["fusion_inter_cos_margin"].append(avg_fusion_inter_cos_margin)
        history["grad_norm"].append(avg_grad_norm)
        
        if use_moe:
            avg_moe_total = epoch_moe_total / n
            avg_moe_load_balance = epoch_moe_load_balance / n
            avg_moe_entropy = epoch_moe_entropy / n
            avg_moe_diversity = epoch_moe_diversity / n
            history["moe_total"].append(avg_moe_total)
            history["moe_load_balance"].append(avg_moe_load_balance)
            history["moe_entropy"].append(avg_moe_entropy)
            history["moe_diversity"].append(avg_moe_diversity)
        
        if use_consistency_regularization:
            avg_consistency = epoch_consistency / n
            history["consistency"].append(avg_consistency)
        
        # ===== 类别权重更新（根据策略） =====
        if enable_class_monitoring:
            import config as cfg
            use_adaptive_weights = getattr(cfg, 'USE_ADAPTIVE_CLASS_WEIGHTS', True)
            try:
                if bool(getattr(cfg, "USE_WEIGHTED_SAMPLER", False)):
                    use_adaptive_weights = False
            except Exception:
                pass
            
            # 决定是否应该更新权重
            should_update_weights = False
            if class_weight_strategy == "fixed":
                # 固定模式：始终使用初始固定权重，不更新
                should_update_weights = False
                precomputed_class_weights = initial_fixed_class_weights.clone()
            elif class_weight_strategy == "hybrid":
                # 混合模式：前N个epoch固定，之后自适应
                if current_epoch <= class_weight_fixed_epochs:
                    should_update_weights = False
                    precomputed_class_weights = initial_fixed_class_weights.clone()
                    if current_epoch == class_weight_fixed_epochs:
                        print(f"[Class Weight Strategy] Epoch {current_epoch}: Switching from fixed to adaptive weights")
                else:
                    should_update_weights = use_adaptive_weights
            elif class_weight_strategy == "adaptive":
                # 自适应模式：根据配置决定
                should_update_weights = use_adaptive_weights
            else:
                raise ValueError(f"Invalid class_weight_strategy: {class_weight_strategy}")
            
            # 如果需要更新权重且class_stats有数据
            if should_update_weights and len(class_stats) > 0:
                # 调试信息：显示class_stats的内容
                if current_epoch <= 5:
                    print(f"[Adaptive Weights Debug] Epoch {current_epoch}: class_stats keys={list(class_stats.keys())}, len={len(class_stats)}")
                    if len(class_stats) > 0:
                        for k, v in class_stats.items():
                            print(f"  Class {k}: correct={v['correct']}, total={v['total']}, acc={v['correct']/v['total'] if v['total'] > 0 else 0:.4f}")
                
                # 计算每个类别的准确率
                num_classes = len(precomputed_class_weights)
                class_accuracies = torch.zeros(num_classes, device=device)
                
                for label_val in range(num_classes):
                    if label_val in class_stats:
                        total = class_stats[label_val]["total"]
                        if total > 0:
                            acc = class_stats[label_val]["correct"] / total
                            class_accuracies[label_val] = acc
                        else:
                            # 如果该类别没有样本，使用平均准确率
                            class_accuracies[label_val] = 0.5
                    else:
                        # 如果该类别没有出现在统计中，使用平均准确率
                        class_accuracies[label_val] = 0.5
                
                # 更新类别权重
                adaptive_momentum = getattr(cfg, 'ADAPTIVE_WEIGHT_MOMENTUM', 0.1)
                adaptive_min_w = getattr(cfg, 'ADAPTIVE_WEIGHT_MIN', 0.5)
                adaptive_max_w = getattr(cfg, 'ADAPTIVE_WEIGHT_MAX', 3.0)
                
                old_weights = precomputed_class_weights.clone()
                precomputed_class_weights = update_adaptive_class_weights(
                    precomputed_class_weights,
                    class_accuracies,
                    momentum=adaptive_momentum,
                    min_w=adaptive_min_w,
                    max_w=adaptive_max_w
                )
                
                # 打印更新信息（总是打印，不依赖log_interval）
                print(f"[Adaptive Weights] Epoch {current_epoch} - Updating weights based on class accuracies")
                print(f"  Class accuracies: {class_accuracies.cpu().numpy()}")
                print(f"  Old weights: {old_weights.cpu().numpy()}")
                print(f"  New weights: {precomputed_class_weights.cpu().numpy()}")
                print(f"  Weight changes: {(precomputed_class_weights - old_weights).cpu().numpy()}")
            elif should_update_weights and len(class_stats) == 0:
                # 如果class_stats为空，输出调试信息（前5个epoch都打印）
                if current_epoch <= 5:
                    print(f"[Adaptive Weights] Warning: No class statistics collected in epoch {current_epoch} (enable_class_monitoring={enable_class_monitoring}, USE_ADAPTIVE_CLASS_WEIGHTS={use_adaptive_weights})")
                    print(f"  class_stats keys: {list(class_stats.keys()) if 'class_stats' in locals() else 'not defined'}")
                    print(f"  class_stats length: {len(class_stats) if 'class_stats' in locals() else 0}")
            
            # 更新class_weights（用于loss计算）
            # 如果使用渐进式权重，需要基于base_class_weights重新计算
            if use_progressive_weights:
                base_class_weights = precomputed_class_weights.clone()
                weight_scale = compute_progressive_weight_scale(
                    current_epoch, progressive_weight_total_epochs, initial_weight_scale, weight_schedule
                )
                class_weights = 1.0 + (base_class_weights - 1.0) * weight_scale
                class_weights = class_weights.to(device)
            else:
                class_weights = precomputed_class_weights
            
            # 重置统计信息（为下一个epoch准备）
            class_stats = {}

        if dataflow_audit_enabled and (current_epoch % dataflow_audit_stride == 0):
            if len(epoch_domain_hist) == 0:
                print(f"[DataflowAudit][Epoch {current_epoch}] no domain_ids observed in this epoch.")
            else:
                _parts = []
                for _did in sorted(epoch_domain_hist.keys()):
                    _name = domain_id_to_name.get(int(_did), f"UNK_{int(_did)}")
                    _cnt = int(epoch_domain_hist.get(_did, 0))
                    _lh = epoch_domain_label_hist.get(_did, {})
                    _lh_str = ",".join([f"{int(k)}:{int(v)}" for k, v in sorted(_lh.items(), key=lambda x: x[0])]) if len(_lh) > 0 else "none"
                    _parts.append(f"{_name}(id={int(_did)},n={_cnt},labels={_lh_str})")
                print(f"[DataflowAudit][Epoch {current_epoch}] " + " | ".join(_parts))
                _cond_parts = []
                for _did in sorted(epoch_domain_cond_examples.keys()):
                    _name = domain_id_to_name.get(int(_did), f"UNK_{int(_did)}")
                    _examples = epoch_domain_cond_examples.get(_did, [])
                    if len(_examples) > 0:
                        _cond_parts.append(f"{_name}:{_examples}")
                if len(_cond_parts) > 0:
                    print(f"[DataflowAudit][Epoch {current_epoch}] condition_examples=" + " | ".join(_cond_parts))

        if (
            _ablation_gate_bank is not None
            and int(getattr(cfg, "ABLATION_GATE_LOG_INTERVAL", 1)) > 0
            and current_epoch % int(getattr(cfg, "ABLATION_GATE_LOG_INTERVAL", 1)) == 0
        ):
            _ablation_gate_bank.log_summary(prefix=f"[AblationGates][Epoch {current_epoch}]")

        if current_epoch % log_interval == 0:
            verbose_train_diag = bool(getattr(cfg, "TRAIN_LOG_VERBOSE_DIAGNOSTICS", False))
            # ATT120: Only print non-zero loss terms to reduce log noise.
            # Always show: Total, Fusion Cls (primary). Conditionally show others if non-zero.
            _nonzero_thresh = 1e-6
            log_parts = [f"[Epoch {current_epoch}/{num_epochs}] Total: {avg_total:.4f}"]
            if abs(avg_signal) > _nonzero_thresh:
                log_parts.append(f"Signal: {avg_signal:.4f}")
            if abs(avg_spec) > _nonzero_thresh:
                log_parts.append(f"Spec: {avg_spec:.4f}")
            if abs(avg_fusion) > _nonzero_thresh:
                log_parts.append(f"Fusion: {avg_fusion:.4f}")
            if bool(getattr(cfg, "FUSION_PRETRAIN_DISABLE_ZCLS_CE", False)):
                log_parts.append(f"Fusion Cls(SKIP): {avg_fusion_cls:.4f}")
            else:
                log_parts.append(f"Fusion Cls: {avg_fusion_cls:.4f}")
            if abs(avg_margin) > _nonzero_thresh:
                log_parts.append(f"Margin: {avg_margin:.4f}")
            if abs(avg_fusion_proto_align) > _nonzero_thresh:
                log_parts.append(f"FusionProtoAlign: {avg_fusion_proto_align:.4f}")
            if abs(avg_fusion_supcon) > _nonzero_thresh:
                _geom_ep = str(getattr(cfg, "FUSION_GEOMETRY_LOSS_MODE", "silhouette")).strip().lower()
                _fus_reg_label = "FusionSilhouette" if _geom_ep == "silhouette" else "FusionSupCon"
                log_parts.append(f"{_fus_reg_label}: {avg_fusion_supcon:.4f}")
            if abs(avg_metric_lp_ce) > _nonzero_thresh:
                log_parts.append(f"MetricLpCE: {avg_metric_lp_ce:.4f}")
            if abs(avg_nr_inter_cos_margin) > _nonzero_thresh:
                log_parts.append(f"NRCosMargin: {avg_nr_inter_cos_margin:.4f}")
            if abs(avg_nr_inter_cos_hard) > _nonzero_thresh:
                log_parts.append(f"NRCosHard: {avg_nr_inter_cos_hard:.4f}")
            if abs(avg_fusion_var_floor) > _nonzero_thresh:
                log_parts.append(f"FusVarFloor: {avg_fusion_var_floor:.4f}")
            if abs(avg_fusion_vicreg) > _nonzero_thresh:
                log_parts.append(f"FusVICReg: {avg_fusion_vicreg:.4f}")
            if abs(avg_fusion_inter_cos_margin) > _nonzero_thresh:
                log_parts.append(f"FusInterCos: {avg_fusion_inter_cos_margin:.4f}")
            if epoch_pairwise_margin_count > 0:
                _avg_pm = epoch_pairwise_margin_sum / epoch_pairwise_margin_count
                if abs(_avg_pm) > _nonzero_thresh:
                    log_parts.append(f"PairwiseMargins: {_avg_pm:.4f}")
            if grad_norm_log_enabled and epoch_grad_norm_steps > 0:
                log_parts.append(f"GradNorm: {avg_grad_norm:.4f}")
            if use_moe and abs(avg_moe_total) > _nonzero_thresh:
                log_parts.append(
                    f"MoE: {avg_moe_total:.4f} "
                    f"(LB: {avg_moe_load_balance:.4f}, "
                    f"Ent: {avg_moe_entropy:.4f}, "
                    f"Div: {avg_moe_diversity:.4f})"
                )
            if use_consistency_regularization and abs(avg_consistency) > _nonzero_thresh:
                log_parts.append(f"Consistency: {avg_consistency:.4f}")
            avg_supcon = 0.0
            if supcon_batches > 0:
                avg_supcon = epoch_supcon_loss / max(1, supcon_batches)
                log_parts.append(f"SupCon: {avg_supcon:.4f}")
            if coral_batches > 0:
                avg_coral = epoch_coral_loss / max(1, coral_batches)
                log_parts.append(f"CORAL: {avg_coral:.8e}")
            if target_coral_batches > 0:
                avg_target_coral = epoch_target_coral_loss / max(1, target_coral_batches)
                log_parts.append(f"CORAL_TGT: {avg_target_coral:.8e}")
            if target_mmd_batches > 0:
                avg_target_mmd = epoch_target_mmd_loss / max(1, target_mmd_batches)
                log_parts.append(f"MMD_TGT: {avg_target_mmd:.8e}")
            log_str = " | ".join(log_parts)
            print(log_str)
            if bool(getattr(cfg, "TRAIN_LOG_LOSS_PATH_AUDIT", True)):
                print(
                    f"[LossPathAudit][Epoch {current_epoch}] "
                    f"supcon_path={loss_path_supcon_batches} "
                    f"ce_path={loss_path_ce_batches} "
                    f"align_path={loss_path_align_batches} "
                    f"stage1_mode={stage1_mode} stage2_mode={stage2_mode} "
                    f"force_ce_stage2={force_ce_stage2} force_ce_path_now={force_ce_path_now} "
                    f"zcls_ce_skip={int(bool(locals().get('_att488_zcls_ce_skip_epoch', False)))}"
                )
                if (not force_ce_path_now) and (loss_path_align_batches <= 0):
                    print(
                        f"[LossPathAudit][WARN] epoch={current_epoch} alignment path was not entered "
                        "while force_ce_path_now=0; check current_prototypes / branch conditions."
                    )
            if loss_budget_log_enabled and (current_epoch % loss_budget_epoch_stride == 0):
                _avg_terms = {}
                for _k_l, _s_l in epoch_loss_term_sum.items():
                    _c_l = max(1, int(epoch_loss_term_count.get(_k_l, 0)))
                    _avg_terms[_k_l] = float(_s_l) / float(_c_l)
                _avg_info = {}
                for _k_i, _s_i in epoch_loss_info_sum.items():
                    _c_i = max(1, int(epoch_loss_info_count.get(_k_i, 0)))
                    _avg_info[_k_i] = float(_s_i) / float(_c_i)
                _abs_total = max(1e-8, abs(float(avg_total)))
                _sup_f_raw = float(_avg_terms.get("signal", avg_signal))
                _sup_s_raw = float(_avg_terms.get("spec", avg_spec))
                _ce_raw = float(_avg_terms.get("_info_fusion_cls", _avg_terms.get("fusion_cls", avg_fusion_cls)))
                _ce_raw_skipped = float(_avg_info.get("_info_fusion_cls_raw_skipped", 0.0))
                _align_ce_raw = float(_avg_terms.get("fusion", avg_fusion))
                _fusion_align_raw = float(_avg_terms.get("fusion_proto_align", avg_fusion_proto_align))
                _proto_reg_raw = float(_avg_terms.get("margin", avg_margin))
                _align_share = abs(_align_ce_raw) / _abs_total
                print(
                    f"[LossShare] align={_align_share:.3f} (raw={_align_ce_raw:.4f}) of |Total|={_abs_total:.4f} | "
                    f"sup_f={_sup_f_raw:.4f}, sup_s={_sup_s_raw:.4f}(raw={_sup_s_raw:.4f}), ce={_ce_raw:.4f}, "
                    f"ce_raw_skipped={_ce_raw_skipped:.4f}, align_ce={_align_ce_raw:.4f}, "
                    f"fusion_align={_fusion_align_raw:.4f}, proto_reg={_proto_reg_raw:.4f}"
                )
                _term_wo_total = {k: v for k, v in _avg_terms.items() if k != "total" and math.isfinite(float(v))}
                _pos_sum = float(sum([v for v in _term_wo_total.values() if v >= 0.0]))
                _neg_sum = float(sum([v for v in _term_wo_total.values() if v < 0.0]))
                _recon = _pos_sum + _neg_sum
                _recon_gap = float(avg_total - _recon)
                _neg_ratio = abs(_neg_sum) / _abs_total
                _untracked_ratio = abs(_recon_gap) / _abs_total
                _top_pos = sorted([(k, v) for k, v in _term_wo_total.items() if v >= 0.0], key=lambda _x: _x[1], reverse=True)[:3]
                _top_neg = sorted([(k, v) for k, v in _term_wo_total.items() if v < 0.0], key=lambda _x: _x[1])[:3]
                _top_pos_s = ", ".join([f"{k}={v:.4f}" for k, v in _top_pos]) if _top_pos else "none"
                _top_neg_s = ", ".join([f"{k}={v:.4f}" for k, v in _top_neg]) if _top_neg else "none"
                print(
                    f"[LossBudget] total={avg_total:.4f}, pos_sum={_pos_sum:.4f}, neg_sum={_neg_sum:.4f}, recon={_recon:.4f}, "
                    f"recon_gap={_recon_gap:.4e}, neg_ratio={_neg_ratio:.3f}, untracked_ratio={_untracked_ratio:.3f}, "
                    f"top_pos={_top_pos_s}, top_neg={_top_neg_s}"
                )
                print(
                    f"[LossShareDetail] sup_f_raw={_sup_f_raw:.3f}, sup_s_raw={_sup_s_raw:.3f}, ce_raw={_ce_raw:.3f}, "
                    f"align_ce_raw={_align_ce_raw:.3f}, fusion_align_raw={_fusion_align_raw:.3f}, proto_reg_raw={_proto_reg_raw:.3f}"
                )
                # ATT379e: weighted loss percentage breakdown
                _wl_items = []
                for _wk, _wv in sorted(_term_wo_total.items(), key=lambda x: abs(x[1]), reverse=True):
                    _wl_pct = abs(_wv) / max(1e-8, _pos_sum) * 100.0
                    if _wl_pct >= 0.5:
                        _wl_items.append(f"{_wk}={_wv:.4f}({_wl_pct:.1f}%)")
                if _wl_items:
                    print(f"[ATT379e-LossPercent] " + ", ".join(_wl_items))
                # ATT428: MVZ core 3-loss monitor
                _mvz_spec = float(_avg_terms.get("spec", 0.0))
                _mvz_fce = float(_avg_terms.get("fusion_cls", 0.0))
                _mvz_fishr = float(_avg_terms.get("fishr", 0.0))
                _mvz_total = float(_avg_terms.get("total", 0.0))
                _mvz_core_sum = _mvz_spec + _mvz_fce + _mvz_fishr
                _mvz_noncore = _mvz_total - _mvz_core_sum if _mvz_total > 0 else 0.0
                _n_active = sum(1 for _k, _v in _term_wo_total.items() if abs(_v) > 1e-6 and not str(_k).startswith("_info_"))
                print(f"[ATT428-CoreLoss] spec_CE={_mvz_spec:.4f}, fusion_cls_CE={_mvz_fce:.4f}, "
                      f"Fishr={_mvz_fishr:.6f}, core_sum={_mvz_core_sum:.4f}, "
                      f"non_core={_mvz_noncore:.4f}, active_terms={_n_active}")
                _reg_keys = [
                    "global_decorr", "spec_vicreg", "spec_uniformity", "global_uniformity",
                    "spec_coral", "fishr", "sdac", "par", "task2_inter_cos",
                    "text_proto_inter_cos_reg", "ccpa", "orthogonality"
                ]
                _active_reg = [(k, float(_avg_terms.get(k, 0.0))) for k in _reg_keys if abs(float(_avg_terms.get(k, 0.0))) > 1e-6]
                if _active_reg:
                    print("[LossShareReg] " + ", ".join(f"{k}={v:.3f}" for k, v in _active_reg))
                # ATT428: margin detail only if any active (skip noise in MVZ mode)
                if current_epoch >= 5 or current_epoch == 1:
                    _margin_keys = [
                        "normal_margin", "rolling_absorb_margin", "rolling_vs_normal_margin",
                        "inner_vs_rolling_margin", "outer_vs_rolling_margin",
                        "inner_vs_normal_margin", "outer_vs_normal_margin",
                        "normal_vs_inner_margin", "outer_vs_inner_margin", "rolling_vs_inner_margin",
                        "outer_margin", "inner_margin"
                    ]
                    _mp = []
                    for _mk in _margin_keys:
                        _mv = float(_avg_terms.get(_mk, 0.0))
                        if abs(_mv) > 1e-7:
                            _mp.append(f"{_mk}={_mv:.4f}")
                    if _mp:
                        print(f"[ATT379d-MarginDetail] " + ", ".join(_mp))
                # ATT374: PACL epoch summary
                if epoch_pacl_batches > 0:
                    _avg_pacl_spec = epoch_pacl_spec / max(1, epoch_pacl_batches)
                    _avg_pacl_fus = epoch_pacl_fusion / max(1, epoch_pacl_batches)
                    _avg_pacl_spec_acc = epoch_pacl_spec_acc / max(1, epoch_pacl_batches)
                    _avg_pacl_fus_acc = epoch_pacl_fusion_acc / max(1, epoch_pacl_batches)
                    print(
                        f"[PACL][Epoch {current_epoch}] "
                        f"spec_loss={_avg_pacl_spec:.4f}, spec_acc={_avg_pacl_spec_acc:.4f}, "
                        f"fus_loss={_avg_pacl_fus:.4f}, fus_acc={_avg_pacl_fus_acc:.4f}, "
                        f"batches={epoch_pacl_batches}"
                    )
                    _cname = {0: "Norm", 1: "Inner", 2: "Outer", 3: "Roll"}
                    _pc_parts = []
                    for _c in sorted(epoch_pacl_spec_per_class_count.keys()):
                        _cnt = epoch_pacl_spec_per_class_count.get(_c, 0)
                        if _cnt > 0:
                            _ca = epoch_pacl_spec_per_class_acc.get(_c, 0.0) / _cnt
                            _cc = epoch_pacl_spec_per_class_pos_cos.get(_c, 0.0) / _cnt
                            _pc_parts.append(f"{_cname.get(_c, f'c{_c}')}(acc={_ca:.3f},cos={_cc:.3f})")
                    if _pc_parts:
                        print(f"[PACL-PerClass][Epoch {current_epoch}] {' | '.join(_pc_parts)}")
                # ATT391-S1: Adapter epoch summary
                if epoch_adapter_batches > 0:
                    _avg_ada_loss = epoch_adapter_loss / max(1, epoch_adapter_batches)
                    _avg_ada_acc = epoch_adapter_acc / max(1, epoch_adapter_batches)
                    _ada_dev = proto_adapter.weight_deviation_from_identity() if proto_adapter is not None else 0.0
                    print(
                        f"[ATT391-Adapter][Epoch {current_epoch}] "
                        f"loss={_avg_ada_loss:.4f}, acc={_avg_ada_acc:.4f}, "
                        f"weight_deviation={_ada_dev:.6f}, batches={epoch_adapter_batches}"
                    )
                # ATT429-P2: Hierarchical CE epoch summary
                if epoch_h_batches > 0:
                    _avg_h_nf = epoch_h_nf_ce / max(1, epoch_h_batches)
                    _avg_h_fault = epoch_h_fault_ce / max(1, epoch_h_batches)
                    print(
                        f"[ATT429-P2][Epoch {current_epoch}] "
                        f"nf_ce={_avg_h_nf:.6f}, fault_ce={_avg_h_fault:.6f}, "
                        f"batches={epoch_h_batches}/{n}"
                    )
                # ATT430-S0: MLP/Proto classifier diagnostics
                try:
                    if hasattr(spec_encoder, 'classifier') and spec_encoder.classifier is not None:
                        _cls_params = list(spec_encoder.classifier.parameters())
                        _cls_grad_norms = []
                        for _cp in _cls_params:
                            if _cp.grad is not None:
                                _cls_grad_norms.append(float(_cp.grad.norm().item()))
                        _cls_w_std = float(_cls_params[-1].data.std().item()) if len(_cls_params) > 0 else 0.0
                        _repr_std = 0.0
                        if hasattr(spec_encoder, '_last_z_spec_pre_norm') and spec_encoder._last_z_spec_pre_norm is not None:
                            _repr_std = float(spec_encoder._last_z_spec_pre_norm.detach().float().std().item())
                        print(
                            f"[ATT430-S0][Epoch {current_epoch}] "
                            f"classifier_w_std={_cls_w_std:.6f}, repr_std={_repr_std:.6f}, "
                            f"cls_grad_norms={[f'{g:.6f}' for g in _cls_grad_norms[:4]]}"
                        )
                    elif getattr(spec_encoder, 'spec_prototypes', None) is not None:
                        _sp = spec_encoder.spec_prototypes
                        _sp_grad_norm = float(_sp.grad.norm().item()) if _sp.grad is not None else 0.0
                        _repr_std = 0.0
                        if hasattr(spec_encoder, '_last_z_spec_pre_norm') and spec_encoder._last_z_spec_pre_norm is not None:
                            _repr_std = float(spec_encoder._last_z_spec_pre_norm.detach().float().std().item())
                        print(
                            f"[ATT453-ProtoGrad][Epoch {current_epoch}] "
                            f"proto_grad_norm={_sp_grad_norm:.6f}, repr_std={_repr_std:.6f}, "
                            f"proto_norm_mean={_sp.data.norm(dim=-1).mean():.4f}"
                        )
                except Exception:
                    pass
                # ATT430-S1/S2/S3: Disentangle + variance epoch summary
                if epoch_dual_batches > 0:
                    _n_db = max(1, epoch_dual_batches)
                    print(
                        f"[ATT430-S1/S2/S3][Epoch {current_epoch}] "
                        f"orth={epoch_dual_orth/_n_db:.6f}, dc_spe={epoch_dual_dc_spe/_n_db:.6f}, "
                        f"dc_inv={epoch_dual_dc_inv_train/_n_db:.6f}, confusion={epoch_dual_confusion/_n_db:.6f}, "
                        f"spec_var={epoch_spec_variance/_n_db:.6f}, batches={epoch_dual_batches}/{n}"
                    )
                    # Disentangle module diagnostics
                    try:
                        if hasattr(spec_encoder, '_last_z_spe_dual') and spec_encoder._last_z_spe_dual is not None:
                            _z_spe = spec_encoder._last_z_spe_dual.detach().float()
                            _z_inv = spec_encoder._last_z_spec_pre_norm.detach().float()
                            print(
                                f"[ATT430-S1][Epoch {current_epoch}] "
                                f"z_inv_std={float(_z_inv.std()):.6f}, z_inv_norm={float(_z_inv.norm(dim=-1).mean()):.4f}, "
                                f"z_spe_std={float(_z_spe.std()):.6f}, z_spe_norm={float(_z_spe.norm(dim=-1).mean()):.4f}"
                            )
                    except Exception:
                        pass
                # ATT392-S1: Learnable Prototype epoch summary
                if learnable_proto_module is not None:
                    _lp_cos_mat = learnable_proto_module.proto_cosine_matrix()
                    _lp_anc_cos = learnable_proto_module.anchor_cosine()
                    _lp_drift = learnable_proto_module.drift_from_anchor()
                    _lp_sep_val = float(learnable_proto_module.separation_loss().detach().item())
                    _lp_anc_val = float(learnable_proto_module.anchor_loss().detach().item())
                    _cos_strs = []
                    for _ri_lp in range(_lp_cos_mat.size(0)):
                        for _ci_lp in range(_ri_lp + 1, _lp_cos_mat.size(1)):
                            _cos_strs.append(f"({_ri_lp},{_ci_lp})={_lp_cos_mat[_ri_lp, _ci_lp].item():.4f}")
                    print(
                        f"[ATT392-LearnProto][Epoch {current_epoch}] "
                        f"sep_loss={_lp_sep_val:.6f}, anchor_loss={_lp_anc_val:.6f}, "
                        f"drift={_lp_drift:.4f}, "
                        f"pairwise_cos=[{', '.join(_cos_strs)}], "
                        f"anchor_cos=[{', '.join([f'{v:.4f}' for v in _lp_anc_cos.tolist()])}]"
                    )
                if epoch_clip_style_batches > 0:
                    _avg_clip_style = epoch_clip_style / max(1, epoch_clip_style_batches)
                    _avg_clip_diag = epoch_clip_style_diag_cos / max(1, epoch_clip_style_batches)
                    _avg_clip_off = epoch_clip_style_offdiag_cos / max(1, epoch_clip_style_batches)
                    print(
                        f"[ClipStyle][Epoch {current_epoch}] "
                        f"loss={_avg_clip_style:.4f}, diag_cos={_avg_clip_diag:.4f}, "
                        f"offdiag_cos={_avg_clip_off:.4f}, batches={epoch_clip_style_batches}"
                    )
                # ATT379: Log new structural modifications
                try:
                    _att379_parts = []
                    _att379_parts.append(f"dual_clip={'ON' if bool(getattr(cfg, 'USE_DUAL_CLIP_SPEC', False)) else 'OFF'}")
                    _att379_parts.append(f"supclip={'ON' if bool(getattr(cfg, 'USE_SUPCLIP', False)) else 'OFF'}")
                    _att379_parts.append(f"pacl_margin={'ON' if bool(getattr(cfg, 'PACL_USE_MARGIN', False)) else 'OFF'}(m={float(getattr(cfg, 'PACL_MARGIN', 0.3)):.2f})")
                    _att379_parts.append(f"aux_spec={'ON' if 'aux_spec' in set(getattr(cfg, 'CORE_LOSS_ALLOWED_TERMS', ())) else 'OFF'}")
                    _att379_parts.append(f"fishr={'ON' if ('fishr' in set(getattr(cfg, 'CORE_LOSS_ALLOWED_TERMS', ())) and bool(getattr(cfg, 'USE_FISHR_LOSS', False))) else 'OFF'}")
                    if current_epoch == 1:
                        print(f"[ATT379][Epoch {current_epoch}] {' | '.join(_att379_parts)}")
                except Exception:
                    pass
                if len(_avg_info) > 0:
                    _info_sum = float(sum(_avg_info.values()))
                    _resid_vals = []
                    for _k_res in ("residual", "residual_abs", "residual_abs_p95", "residual_abs_max"):
                        if _k_res in _avg_info:
                            _resid_vals.append(float(_avg_info.get(_k_res, 0.0)))
                    _resid_mean = float(np.mean(_resid_vals)) if len(_resid_vals) > 0 else 0.0
                    _resid_abs_mean = float(np.mean([abs(v) for v in _resid_vals])) if len(_resid_vals) > 0 else 0.0
                    _resid_p95 = float(np.percentile(np.array([abs(v) for v in _resid_vals], dtype=np.float64), 95)) if len(_resid_vals) > 0 else 0.0
                    _resid_max = float(np.max(np.array([abs(v) for v in _resid_vals], dtype=np.float64))) if len(_resid_vals) > 0 else 0.0
                    print(
                        f"[LossBreakdown-Audit] samples={len(_avg_terms)}, residual_mean={_resid_mean:.4f}, "
                        f"residual_abs_mean={_resid_abs_mean:.4f}, residual_abs_p95={_resid_p95:.4f}, "
                        f"residual_abs_max={_resid_max:.4f}, info_sum_mean={_info_sum:.4f}"
                    )
            if grad_norm_detail_log_enabled and epoch_grad_norm_detail_steps > 0:
                _g_sig = epoch_grad_norm_signal_sum / max(1, epoch_grad_norm_detail_steps)
                _g_spec = epoch_grad_norm_spec_sum / max(1, epoch_grad_norm_detail_steps)
                _g_fus = epoch_grad_norm_fusion_sum / max(1, epoch_grad_norm_detail_steps)
                _g_dec = epoch_grad_norm_decoder_sum / max(1, epoch_grad_norm_detail_steps)
                _n_sig = epoch_grad_norm_signal_count_sum / max(1, epoch_grad_norm_detail_steps)
                _n_spec = epoch_grad_norm_spec_count_sum / max(1, epoch_grad_norm_detail_steps)
                _n_fus = epoch_grad_norm_fusion_count_sum / max(1, epoch_grad_norm_detail_steps)
                _n_dec = epoch_grad_norm_decoder_count_sum / max(1, epoch_grad_norm_detail_steps)
                _ratio_dec_fus = (_g_dec / _g_fus) if abs(_g_fus) > 1e-12 else float("nan")
                _ratio_fus_spec = (_g_fus / _g_spec) if abs(_g_spec) > 1e-12 else float("nan")
                _ratio_spec_sig = (_g_spec / _g_sig) if abs(_g_sig) > 1e-12 else float("nan")
                print(
                    f"[GradNorm] sampled_batches={epoch_grad_norm_detail_steps}, stride={grad_norm_detail_step_stride} | "
                    f"signal={_g_sig:.4f}(n={_n_sig:.0f}), spec={_g_spec:.4f}(n={_n_spec:.0f}), "
                    f"fusion={_g_fus:.4f}(n={_n_fus:.0f}), decoder={_g_dec:.4f}(n={_n_dec:.0f}) | "
                    f"ratio(dec/fusion)={_ratio_dec_fus:.3f}, ratio(fusion/spec)={_ratio_fus_spec:.3f}, ratio(spec/signal)={_ratio_spec_sig:.3f}"
                )
                # ATT437-α diagnostic: compare dec/fusion ratio to ATT436 baseline (265x@E31)
                if _ratio_dec_fus < 5.0:
                    print(f"[ATT437-DetachedNorm-OK] dec/fusion={_ratio_dec_fus:.1f}x (<5x, gradient bottleneck resolved)")
                elif _ratio_dec_fus < 20.0:
                    print(f"[ATT437-DetachedNorm-PARTIAL] dec/fusion={_ratio_dec_fus:.1f}x (improved from ATT436 265x)")
                else:
                    print(f"[ATT437-DetachedNorm-WARN] dec/fusion={_ratio_dec_fus:.1f}x (still high, needs investigation)")
                try:
                    _sig_opt = _signal_optimizer_membership(optimizer)
                    _sig_has_grad = int(_n_sig > 0 and _g_sig > 1e-10)
                    print(
                        f"[SignalGradAudit][Epoch {current_epoch}] has_grad={_sig_has_grad} "
                        f"grad={_g_sig:.4e} grad_params={_n_sig:.0f} "
                        f"requires_grad={_sig_opt['requires_grad']}/{_sig_opt['total']} "
                        f"in_optimizer={_sig_opt['in_optimizer']}/{_sig_opt['total']} "
                        f"in_opt_requires_grad={_sig_opt['in_optimizer_requires_grad']}/{_sig_opt['total']}"
                    )
                except Exception:
                    pass
                # ATT438-DGL: audit aux_cls_head gradient norms
                if bool(getattr(cfg, "DGL_ENABLE", False)) and hasattr(fusion_module, 'aux_cls_head') and fusion_module.aux_cls_head is not None:
                    _aux_grad_sum = 0.0
                    _aux_n = 0
                    for _ap in fusion_module.aux_cls_head.parameters():
                        if _ap.grad is not None:
                            _aux_grad_sum += _ap.grad.data.norm(2).item()
                            _aux_n += 1
                    _aux_grad_avg = _aux_grad_sum / max(1, _aux_n)
                    _ratio_aux_fus = (_aux_grad_avg / _g_fus) if abs(_g_fus) > 1e-12 else float("nan")
                    print(
                        f"[ATT438-DGL-GradAudit] E{current_epoch} "
                        f"aux_head_grad={_aux_grad_avg:.6f}(n={_aux_n}), "
                        f"fusion_body_grad={_g_fus:.6f}, "
                        f"ratio(aux/fusion_body)={_ratio_aux_fus:.2f}"
                    )
            if grad_clip_log_enabled:
                if grad_clip_enabled and epoch_grad_clip_batches > 0:
                    _clip_frac = float(epoch_grad_clip_clipped_batches) / float(max(1, epoch_grad_clip_batches))
                    _pre_mean = epoch_grad_clip_pre_norm_sum / max(1, epoch_grad_clip_batches)
                    _post_mean = epoch_grad_clip_post_norm_sum / max(1, epoch_grad_clip_batches)
                    _grad_params = epoch_grad_clip_param_count_sum / max(1, epoch_grad_clip_batches)
                    print(
                        f"[GradClip] enabled=1, max_norm={grad_clip_max_norm:.4f}, batches={epoch_grad_clip_batches}, "
                        f"clipped_batches={epoch_grad_clip_clipped_batches}, clip_frac={_clip_frac:.3f}, "
                        f"grad_params={_grad_params:.0f}, pre_norm_mean={_pre_mean:.4f}, post_norm_mean={_post_mean:.4f}"
                    )
                else:
                    print(
                        f"[GradClip] enabled={int(grad_clip_enabled)}, max_norm={grad_clip_max_norm:.4f}, "
                        "batches=0, clipped_batches=0, clip_frac=0.000, grad_params=0, pre_norm_mean=0.0000, post_norm_mean=0.0000"
                    )
            # ATT416: Log signal entropy gate stats
            try:
                if hasattr(fusion_module, '_last_signal_entropy_gate') and fusion_module._last_signal_entropy_gate is not None:
                    _seg_val = fusion_module._last_signal_entropy_gate
                    print(f"[SignalEntropyGate][Epoch {current_epoch}] gate_mean={_seg_val:.4f} (floor={getattr(fusion_module, '_signal_entropy_gate_floor', 0.1):.2f})")
            except Exception:
                pass
            # ATT420-P2: Log spec-preserving residual alpha
            try:
                if hasattr(fusion_module, '_last_spec_residual_alpha') and fusion_module._last_spec_residual_alpha is not None:
                    _sra = fusion_module._last_spec_residual_alpha
                    print(f"[SpecResidual][Epoch {current_epoch}] alpha={_sra:.4f} (spec_weight={_sra:.4f}, ca_weight={1-_sra:.4f})")
            except Exception:
                pass
            # ATT462: Log fusion residual identity skip alpha
            try:
                if hasattr(fusion_module, '_last_residual_alpha') and fusion_module._last_residual_alpha is not None:
                    _fra = fusion_module._last_residual_alpha
                    print(f"[ATT462-ResSkip][Epoch {current_epoch}] alpha={_fra:.4f} "
                          f"(identity_weight=1.0, fusion_weight={_fra:.4f})")
            except Exception:
                pass
            # ATT164: Log FreqAttnPool gate weight norm to track attention learning speed
            try:
                _gate_w_norm = spec_encoder.freq_attn_pool.gate.weight.abs().mean().item()
                print(f"[FreqAttnPool] gate_w_norm={_gate_w_norm:.6f}")
            except AttributeError:
                pass
            if verbose_train_diag and supcon_batches > 0:
                avg_pos_pairs = supcon_pos_pairs / max(1, supcon_batches)
                avg_valid_anchors = supcon_valid_anchors / max(1, supcon_batches)
                print(f"[SupCon] batches={supcon_batches}, pos_pairs_avg={avg_pos_pairs:.1f}, valid_anchors_avg={avg_valid_anchors:.1f}")
                try:
                    bs_now = int(batch_size)
                except Exception:
                    bs_now = None
                try:
                    if bs_now is not None and avg_valid_anchors < 0.5 * bs_now:
                        print(f"[SupCon] Warning: valid_anchors_avg too low (< 0.5*batch_size={0.5*bs_now:.1f})")
                    if bs_now is not None and avg_pos_pairs < max(1.0, 0.5 * bs_now):
                        print(f"[SupCon] Warning: pos_pairs_avg too low (< 0.5*batch_size={0.5*bs_now:.1f})")
                except Exception:
                    pass
            # Stage1 CE diagnostic (Attempt 28)
            if verbose_train_diag and stage1_ce_batches > 0:
                avg_stage1_ce = epoch_stage1_ce_loss / max(1, stage1_ce_batches)
                print(
                    f"[Stage1-CE] batches={stage1_ce_batches}, avg_ce_loss={avg_stage1_ce:.4f}, "
                    f"missing_hp={stage1_ce_missing_hp_batches}, missing_logits={stage1_ce_missing_logits_batches} "
                    f"(classifier being trained)"
                )
                # Per-class CE diagnostics: helps detect IO collapse (Inner/Outer not learning)
                try:
                    cls_msgs = []
                    for _c in range(4):
                        if int(stage1_ce_class_count[_c]) > 0:
                            cls_msgs.append(f"c{_c}={stage1_ce_class_loss_sum[_c]/max(1,int(stage1_ce_class_count[_c])):.4f}")
                        else:
                            cls_msgs.append(f"c{_c}=NA")
                    print("[Stage1-CE-Class] " + ", ".join(cls_msgs))
                except Exception:
                    pass
            if sem_fvn_batches > 0:
                avg_sem_fvn = epoch_sem_fvn_loss / max(1, sem_fvn_batches)
                print(f"[Sem-FvN] batches={sem_fvn_batches}, avg_loss={avg_sem_fvn:.4f}")
                if not verbose_train_diag and semantic_proto_source != "signal_centroid":
                    # Keep core training indicators concise while preserving optimization behavior.
                    io_roll_sep_batches = 0
                    io_sep_batches = 0
                    nf_sep_batches = 0
                    ir_sep_batches = 0
                    embed_centroid_sum = None
                # Attempt 31: IO↔Rolling separation diagnostics
                try:
                    import config as _cfg_sep
                    use_sep_flag = bool(getattr(_cfg_sep, "USE_IO_ROLL_SEP_LOSS", False))
                    log_centroid_flag = bool(getattr(_cfg_sep, "LOG_EMBED_CENTROID_SIM", False))
                    amp_indicator = "[AMP=ON]" if (use_amp and scaler is not None) else "[AMP=OFF]"
                    if use_sep_flag or log_centroid_flag:
                        if io_roll_sep_batches > 0:
                            avg_sep = epoch_io_roll_sep_loss / max(1, io_roll_sep_batches)
                            sim13 = (io_roll_sep_sim13_sum / max(1, io_roll_sep_sim13_cnt)) if io_roll_sep_sim13_cnt > 0 else float("nan")
                            sim23 = (io_roll_sep_sim23_sum / max(1, io_roll_sep_sim23_cnt)) if io_roll_sep_sim23_cnt > 0 else float("nan")
                            print(f"[IO-Roll-Sep] {amp_indicator} batches={io_roll_sep_batches}, loss={avg_sep:.4f}, cos_sim(inner,roll)={sim13:.4f}, cos_sim(outer,roll)={sim23:.4f}")
                        elif supcon_batches > 0:
                            # SupCon stage active but io_roll_sep_batches == 0 → code path not executed
                            print(f"[IO-Roll-Sep] {amp_indicator} WARNING: supcon_batches={supcon_batches} but io_roll_sep_batches=0 (sep/centroid code path not entered; check AMP branch parity)")
                except Exception:
                    pass
                # Attempt 32: Inner↔Outer separation diagnostics
                try:
                    import config as _cfg_io
                    use_io_flag = bool(getattr(_cfg_io, "USE_IO_SEP_LOSS", False))
                    log_io_flag = bool(getattr(_cfg_io, "LOG_IO_SEP", False))
                    amp_indicator_io = "[AMP=ON]" if (use_amp and scaler is not None) else "[AMP=OFF]"
                    if use_io_flag or log_io_flag:
                        if io_sep_batches > 0:
                            avg_io = epoch_io_sep_loss / max(1, io_sep_batches)
                            sim12 = (io_sep_sim12_sum / max(1, io_sep_sim12_cnt)) if io_sep_sim12_cnt > 0 else float("nan")
                            print(f"[IO-Sep] {amp_indicator_io} batches={io_sep_batches}, loss={avg_io:.4f}, cos_sim(inner,outer)={sim12:.4f}")
                        elif supcon_batches > 0:
                            print(f"[IO-Sep] {amp_indicator_io} WARNING: supcon_batches={supcon_batches} but io_sep_batches=0 (io-sep code path not entered; check AMP branch parity)")
                except Exception:
                    pass
                # Attempt 33: Normal↔Fault separation diagnostics
                # ATT167 log reduction: print at E1 and every 5 epochs only.
                try:
                    import config as _cfg_nf
                    use_nf_flag = bool(getattr(_cfg_nf, "USE_NF_SEP_LOSS", False))
                    log_nf_flag = bool(getattr(_cfg_nf, "LOG_NF_SEP", False))
                    amp_indicator_nf = "[AMP=ON]" if (use_amp and scaler is not None) else "[AMP=OFF]"
                    if use_nf_flag or log_nf_flag:
                        _log_nf_this_epoch = (current_epoch == 1 or current_epoch % 5 == 0)
                        if nf_sep_batches > 0 and _log_nf_this_epoch:
                            avg_nf = epoch_nf_sep_loss / max(1, nf_sep_batches)
                            s01 = (nf_sep_sim01_sum / max(1, nf_sep_sim01_cnt)) if nf_sep_sim01_cnt > 0 else float("nan")
                            s02 = (nf_sep_sim02_sum / max(1, nf_sep_sim02_cnt)) if nf_sep_sim02_cnt > 0 else float("nan")
                            s03 = (nf_sep_sim03_sum / max(1, nf_sep_sim03_cnt)) if nf_sep_sim03_cnt > 0 else float("nan")
                            print(f"[NF-Sep] {amp_indicator_nf} batches={nf_sep_batches}, loss={avg_nf:.4f}, cos_sim(N,I)={s01:.4f}, cos_sim(N,O)={s02:.4f}, cos_sim(N,R)={s03:.4f}")
                        elif supcon_batches > 0 and _log_nf_this_epoch:
                            print(f"[NF-Sep] {amp_indicator_nf} WARNING: supcon_batches={supcon_batches} but nf_sep_batches=0 (nf-sep code path not entered; check AMP branch parity)")
                except Exception:
                    pass
                # Attempt 34: Inner↔Rolling separation diagnostics
                try:
                    import config as _cfg_ir
                    use_ir_flag = bool(getattr(_cfg_ir, "USE_IR_SEP_LOSS", False))
                    log_ir_flag = bool(getattr(_cfg_ir, "LOG_IR_SEP", False))
                    amp_indicator_ir = "[AMP=ON]" if (use_amp and scaler is not None) else "[AMP=OFF]"
                    if use_ir_flag or log_ir_flag:
                        if ir_sep_batches > 0:
                            avg_ir = epoch_ir_sep_loss / max(1, ir_sep_batches)
                            s13 = (ir_sep_sim13_sum / max(1, ir_sep_sim13_cnt)) if ir_sep_sim13_cnt > 0 else float("nan")
                            print(f"[IR-Sep] {amp_indicator_ir} batches={ir_sep_batches}, loss={avg_ir:.4f}, cos_sim(inner,rolling)={s13:.4f}")
                        elif supcon_batches > 0:
                            print(f"[IR-Sep] {amp_indicator_ir} WARNING: supcon_batches={supcon_batches} but ir_sep_batches=0 (ir-sep code path not entered; check AMP branch parity)")
                except Exception:
                    pass
                try:
                    import config as _cfg_cent
                    amp_indicator_cent = "[AMP=ON]" if (use_amp and scaler is not None) else "[AMP=OFF]"
                    if bool(getattr(_cfg_cent, "LOG_EMBED_CENTROID_SIM", False)):
                        if embed_centroid_sum is not None and any(c > 0 for c in embed_centroid_cnt):
                            cents = []
                            for _c in range(4):
                                if int(embed_centroid_cnt[_c]) > 0:
                                    v = embed_centroid_sum[_c] / float(max(1, int(embed_centroid_cnt[_c])))
                                    cents.append(F.normalize(v, dim=-1, eps=1e-6))
                                else:
                                    cents.append(None)
                            # Print key sims (focus on IO↔Rolling and IO↔IO)
                            def _cos(a, b):
                                if a is None or b is None:
                                    return None
                                return float((a * b).sum().detach().cpu().item())
                            s12 = _cos(cents[1], cents[2])
                            s13 = _cos(cents[1], cents[3])
                            s23 = _cos(cents[2], cents[3])
                            s01 = _cos(cents[0], cents[1])
                            s02 = _cos(cents[0], cents[2])
                            s03 = _cos(cents[0], cents[3])
                            def _fmt(x):
                                return "NA" if x is None else f"{x:.3f}"
                            print(
                                f"[Embed-CentroidCos] {amp_indicator_cent} "
                                f"sim(N,I)={_fmt(s01)}, sim(N,O)={_fmt(s02)}, sim(N,R)={_fmt(s03)}, "
                                f"sim(I,O)={_fmt(s12)}, sim(I,R)={_fmt(s13)}, sim(O,R)={_fmt(s23)}"
                            )
                        else:
                            # LOG enabled but no centroids accumulated → path not executed
                            print(f"[Embed-CentroidCos] {amp_indicator_cent} WARNING: LOG enabled but embed_centroid_sum is empty (path not entered; check AMP branch parity)")
                except Exception:
                    pass
            if semantic_proto_source == "signal_centroid":
                print(f"[VGSP-DEBUG] epoch={current_epoch} centroid_none={embed_centroid_sum is None} counts={embed_centroid_cnt}")
            # VGSP: optionally wait until the source metric space has matured before anchoring.
            if (semantic_proto_source == "signal_centroid" and
                    int(current_epoch) >= semantic_centroid_start_epoch and
                    (semantic_centroid_update == "each_epoch" or int(current_epoch) == semantic_centroid_start_epoch) and
                    learnable_proto_module is not None and
                    embed_centroid_sum is not None and
                    all(int(c) > 0 for c in embed_centroid_cnt)):
                _centroids = []
                for _c in range(4):
                    _v = embed_centroid_sum[_c] / float(embed_centroid_cnt[_c])
                    _centroids.append(F.normalize(_v, dim=-1))
                _centroids = torch.stack(_centroids, dim=0)
                _centroid_cos = _centroids @ _centroids.t()
                _offdiag = ~torch.eye(_centroid_cos.size(0), dtype=torch.bool, device=_centroid_cos.device)
                print(f"[VGSP] epoch={current_epoch} source_centroid_pairwise_cos_mean={_centroid_cos[_offdiag].mean().item():.4f} max={_centroid_cos[_offdiag].max().item():.4f}")
                learnable_proto_module.set_from_signal_centroids(_centroids, semantic_blend=semantic_bayesian_blend)
                training_proto_tensor = learnable_proto_module.forward().detach()
                print(f"[VGSP] epoch={current_epoch} source vibration centroid calibration applied (bayesian_blend={semantic_bayesian_blend:.3f})")

            # Keep NF-Sep observable even when semantic FvN anchor is disabled.
            # (sem_fvn_batches==0 is the normal path in the current DG mainline)
            # ATT167 log reduction: print at E1 and every 5 epochs only.
            if sem_fvn_batches <= 0:
                try:
                    import config as _cfg_nf
                    use_nf_flag = bool(getattr(_cfg_nf, "USE_NF_SEP_LOSS", False))
                    log_nf_flag = bool(getattr(_cfg_nf, "LOG_NF_SEP", False))
                    amp_indicator_nf = "[AMP=ON]" if (use_amp and scaler is not None) else "[AMP=OFF]"
                    if use_nf_flag or log_nf_flag:
                        _log_nf2_this_epoch = (current_epoch == 1 or current_epoch % 5 == 0)
                        if nf_sep_batches > 0 and _log_nf2_this_epoch:
                            avg_nf = epoch_nf_sep_loss / max(1, nf_sep_batches)
                            s01 = (nf_sep_sim01_sum / max(1, nf_sep_sim01_cnt)) if nf_sep_sim01_cnt > 0 else float("nan")
                            s02 = (nf_sep_sim02_sum / max(1, nf_sep_sim02_cnt)) if nf_sep_sim02_cnt > 0 else float("nan")
                            s03 = (nf_sep_sim03_sum / max(1, nf_sep_sim03_cnt)) if nf_sep_sim03_cnt > 0 else float("nan")
                            print(f"[NF-Sep] {amp_indicator_nf} batches={nf_sep_batches}, loss={avg_nf:.4f}, cos_sim(N,I)={s01:.4f}, cos_sim(N,O)={s02:.4f}, cos_sim(N,R)={s03:.4f}")
                        elif supcon_batches > 0 and _log_nf2_this_epoch:
                            print(f"[NF-Sep] {amp_indicator_nf} WARNING: supcon_batches={supcon_batches} but nf_sep_batches=0 (nf-sep code path not entered; check AMP branch parity)")
                except Exception:
                    pass
            elif verbose_train_diag and supcon_batches > 0:
                print(
                    f"[Stage1-CE] WARNING: supcon_batches={supcon_batches} but stage1_ce_batches=0 - classifier NOT training! "
                    f"(missing_hp={stage1_ce_missing_hp_batches}, missing_logits={stage1_ce_missing_logits_batches})"
                )
            # Rolling diagnostics (epoch-level)
            if not verbose_train_diag:
                roll_sample_count = 0
                roll_dann_expected_batches = 0
                roll_decorr_expected_batches = 0
                roll_decorr_in_expected_batches = 0
            if roll_sample_count > 0:
                pos_mean = (roll_pos_prob_sum / max(1, roll_pos_count)) if roll_pos_count > 0 else 0.0
                neg_mean = (roll_neg_prob_sum / max(1, roll_neg_count)) if roll_neg_count > 0 else 0.0
                print(
                    f"[Rolling Stats] samples={roll_sample_count}, pos={roll_pos_count}, neg={roll_neg_count}, "
                    f"p_pos_mean={pos_mean:.4f}, p_neg_mean={neg_mean:.4f}, loss_mode={roll_loss_mode}"
                )
            if roll_dann_expected_batches > 0:
                avg_roll_dann = epoch_roll_dann / max(1, roll_dann_used_batches)
                print(
                    f"[Rolling-DANN] alpha={roll_dann_alpha_now:.4f}, lambda={roll_dann_lambda:.4f}, "
                    f"loss={avg_roll_dann:.4f}, used={roll_dann_used_batches}/{roll_dann_expected_batches}, "
                    f"missing={roll_dann_missing_batches}, no_fault={roll_dann_no_fault_batches}"
                )
            if roll_decorr_expected_batches > 0:
                avg_roll_decorr = epoch_roll_decorr / max(1, roll_decorr_used_batches)
                var_mean = roll_decorr_var_sum / max(1, roll_decorr_used_batches)
                cov_mean = roll_decorr_cov_sum / max(1, roll_decorr_used_batches)
                print(
                    f"[Rolling-DeCollapse] weight={roll_decorr_weight:.4f}, loss={avg_roll_decorr:.4f}, "
                    f"var_mean={var_mean:.4f}, cov_abs={cov_mean:.4f}, "
                    f"used={roll_decorr_used_batches}/{roll_decorr_expected_batches}, "
                    f"missing={roll_decorr_missing_batches}, no_fault={roll_decorr_no_fault_batches}, "
                    f"small_batch={roll_decorr_small_batches}"
                )
                roll_decorr_var_mean_log = float(var_mean)
            else:
                roll_decorr_var_mean_log = None
            if roll_decorr_in_expected_batches > 0:
                avg_roll_decorr_in = epoch_roll_decorr_in / max(1, roll_decorr_in_used_batches)
                var_in = roll_decorr_in_var_sum / max(1, roll_decorr_in_used_batches)
                cov_in = roll_decorr_in_cov_sum / max(1, roll_decorr_in_used_batches)
                print(
                    f"[Rolling-DeCollapse-Input] weight={roll_decorr_input_weight:.4f}, loss={avg_roll_decorr_in:.4f}, "
                    f"var_mean={var_in:.4f}, cov_abs={cov_in:.4f}, "
                    f"used={roll_decorr_in_used_batches}/{roll_decorr_in_expected_batches}, "
                    f"missing={roll_decorr_in_missing_batches}, no_fault={roll_decorr_in_no_fault_batches}, "
                    f"small_batch={roll_decorr_in_small_batches}"
                )
                roll_decorr_in_var_mean_log = float(var_in)
            else:
                roll_decorr_in_var_mean_log = None

            # ATT167 log reduction: [Global-DeCollapse] and [Global-Var] printed at E1 and every 5 epochs only.
            _log_decorr_this_epoch = (current_epoch == 1 or current_epoch % 5 == 0)
            if global_decorr_expected_batches > 0:
                avg_global_decorr = epoch_global_decorr / max(1, global_decorr_used_batches)
                var_g = global_decorr_var_sum / max(1, global_decorr_used_batches)
                cov_g = global_decorr_cov_sum / max(1, global_decorr_used_batches)
                if _log_decorr_this_epoch:
                    _global_reg_tag = "Global-VICReg" if use_global_vicreg else "Global-DeCollapse"
                    _pre_std_msg = ""
                    if use_global_vicreg and global_decorr_pre_std_batches > 0:
                        _pre_std = global_decorr_pre_std_sum / max(1, global_decorr_pre_std_batches)
                        _pre_std_msg = f", pre_std_mean={_pre_std:.4f}"
                    print(
                        f"[{_global_reg_tag}] weight={effective_global_decorr_weight:.4f}, loss={avg_global_decorr:.4f}, "
                        f"var_mean={var_g:.4f}, cov_abs={cov_g:.4f}{_pre_std_msg}, "
                        f"used={global_decorr_used_batches}/{global_decorr_expected_batches}, "
                        f"missing={global_decorr_missing_batches}, small_batch={global_decorr_small_batches}"
                    )
                if global_decorr_used_batches == 0:
                    _global_reg_tag = "Global-VICReg" if use_global_vicreg else "Global-DeCollapse"
                    print(
                        f"[{_global_reg_tag}] Warning: no usable batches this epoch "
                        f"(missing={global_decorr_missing_batches}, small_batch={global_decorr_small_batches})"
                    )

            if global_uniformity_expected_batches > 0:
                avg_global_uniformity = epoch_global_uniformity / max(1, global_uniformity_used_batches)
                mean_d2 = global_uniformity_d2_sum / max(1, global_uniformity_used_batches)
                if _log_decorr_this_epoch:
                    print(
                        f"[Global-Uniformity] weight={global_uniformity_weight:.4f}, "
                        f"loss={avg_global_uniformity:.4f}, mean_pair_d2={mean_d2:.4f}, "
                        f"used={global_uniformity_used_batches}/{global_uniformity_expected_batches}, "
                        f"missing={global_uniformity_missing_batches}, small_batch={global_uniformity_small_batches}, "
                        f"t={global_uniformity_t:.2f}"
                    )
                if global_uniformity_used_batches == 0:
                    print(
                        f"[Global-Uniformity] Warning: no usable batches this epoch "
                        f"(missing={global_uniformity_missing_batches}, small_batch={global_uniformity_small_batches})"
                    )

            if fishr_expected_batches > 0 and _log_decorr_this_epoch:
                avg_fishr = epoch_fishr / max(1, fishr_used_batches)
                avg_fishr_grad_var = fishr_grad_var_sum / max(1, fishr_used_batches)
                _fishr_target_diag = str(getattr(cfg, "FISHR_TARGET_FEATURE", "z_fusion")).strip().lower()
                _fishr_status = "ACTIVE" if avg_fishr > 1e-8 else "DEAD(loss≈0)"
                print(
                    f"[Fishr] status={_fishr_status}, weight={fishr_weight:.4f}, loss={avg_fishr:.6f}, "
                    f"grad_var_mean={avg_fishr_grad_var:.8f}, "
                    f"used={fishr_used_batches}/{fishr_expected_batches}, skipped={fishr_skipped_batches}, "
                    f"intra_split={fishr_split_batches}, "
                    f"source_domains={list(fishr_source_domain_ids)}, target_feature={_fishr_target_diag}"
                )
                if avg_fishr < 1e-8 and fishr_used_batches > 0:
                    print(f"[Fishr-WARN] loss≈0 indicates gradient variance between domains is negligible. "
                          f"Possible causes: (1) repr features collapsed, (2) logits→repr grad path broken, "
                          f"(3) single-domain batch. Check [FishrDiag] batch-level logs.")
            if ccpa_expected_batches > 0 and _log_decorr_this_epoch:
                avg_ccpa = epoch_ccpa / max(1, ccpa_used_batches)
                avg_ccpa_pairs = float(ccpa_pair_sum) / max(1, ccpa_used_batches)
                avg_ccpa_classes = float(ccpa_class_sum) / max(1, ccpa_used_batches)
                avg_ccpa_cos = ccpa_cos_sum / max(1, ccpa_used_batches)
                print(
                    f"[CCPA] weight={ccpa_weight:.4f}, loss={avg_ccpa:.4f}, mean_cos={avg_ccpa_cos:.4f}, "
                    f"pairs={avg_ccpa_pairs:.2f}, classes={avg_ccpa_classes:.2f}, "
                    f"used={ccpa_used_batches}/{ccpa_expected_batches}, skipped={ccpa_skipped_batches}, "
                    f"source_domains={list(ccpa_source_domain_ids)}"
                )

            if log_global_variance and _log_decorr_this_epoch:
                z_cls_var = (epoch_z_cls_var_sum / max(1, z_cls_var_batches)) if z_cls_var_batches > 0 else 0.0
                z_res_var = (epoch_z_res_var_sum / max(1, z_res_var_batches)) if z_res_var_batches > 0 else 0.0
                z_gate_var = (epoch_z_gate_var_sum / max(1, z_gate_var_batches)) if z_gate_var_batches > 0 else 0.0
                print(
                    f"[Global-Var] z_cls={z_cls_var:.6f}, z_res={z_res_var:.6f}, z_gate={z_gate_var:.6f} "
                    f"(batches: cls={z_cls_var_batches}, res={z_res_var_batches}, gate={z_gate_var_batches})"
                )
                # ATT437-α: verify detached-norm is active (no F.normalize in forward)
                try:
                    import config as _cfg_dn
                    _has_ln = hasattr(fusion_module, 'out_norm')
                    print(f"[ATT437-FusionLN] LayerNorm={'PRESENT' if _has_ln else 'MISSING'}")
                    if bool(getattr(_cfg_dn, "FUSION_USE_STANDARD_UNIT_VECTOR_NORM", False)):
                        print(f"[ATT437-FusionNorm] torch.nn.functional.normalize(dim=-1) — full L2 gradient (FUSION_USE_STANDARD_UNIT_VECTOR_NORM=1)")
                    else:
                        print(f"[ATT437-DetachedNorm] F.normalize replaced with z/||z||.detach() "
                              f"(gradient projection loss eliminated)")
                except Exception:
                    pass
            
            # ===== Prototype Separation Loss 日志 =====
            if use_proto_sep_loss and proto_sep_batches > 0:
                avg_proto_sep = epoch_proto_sep_loss / max(1, proto_sep_batches)
                print(
                    f"[Proto-Sep] weight={proto_sep_weight:.4f}, loss={avg_proto_sep:.4f}, "
                    f"batches={proto_sep_batches}"
                )
                # 打印类原型距离
                if log_proto_distances and proto_sep_loss_fn is not None:
                    proto_dists = proto_sep_loss_fn.get_centroid_distances()
                    if proto_dists:
                        dist_str = ", ".join([f"{k}={v:.4f}" for k, v in proto_dists.items()])
                        print(f"[Proto-Sep] Centroid distances: {dist_str}")
                        # 特别关注 Inner(1) vs Outer(2) 的距离
                        io_dist = proto_dists.get("dist_1_2", None)
                        if io_dist is not None:
                            if io_dist < proto_sep_margin_delta:
                                print(f"[Proto-Sep] Warning: Inner-Outer distance ({io_dist:.4f}) < delta ({proto_sep_margin_delta:.4f})")
                # 类内方差统计日志
                if any(c > 0 for c in class_var_count):
                    var_items = []
                    for c in range(4):
                        if class_var_count[c] > 0:
                            var_mean = class_var_sum[c] / max(1, class_var_count[c])
                            var_items.append(f"class_{c}={var_mean:.4f}")
                    if var_items:
                        print(f"[Class-Var] " + ", ".join(var_items))

            if roll_decorr_reset_on_zero:
                effective_var = None
                if roll_decorr_in_var_mean_log is not None:
                    effective_var = roll_decorr_in_var_mean_log
                elif roll_decorr_var_mean_log is not None:
                    effective_var = roll_decorr_var_mean_log
                if effective_var is not None and effective_var <= float(roll_decorr_reset_eps):
                    if (current_epoch - roll_decorr_last_reset_epoch) >= int(roll_decorr_reset_cooldown):
                        did_reset = False
                        if hasattr(fusion_decoder, "reset_roll_adapter"):
                            did_reset = bool(fusion_decoder.reset_roll_adapter())
                        if did_reset:
                            roll_decorr_last_reset_epoch = int(current_epoch)
                            print(
                                f"[Rolling-Reset] Adapter reset triggered at epoch {current_epoch} "
                                f"(var_mean={effective_var:.6f}, eps={float(roll_decorr_reset_eps):.6f})"
                            )
            
            # 打印类感知权重（如果启用）
            if hasattr(fusion_module, 'class_modal_weights'):
                with torch.no_grad():
                    class_weights_sigmoid = torch.sigmoid(fusion_module.class_modal_weights)
                    # 归一化权重
                    class_weights_norm = class_weights_sigmoid / class_weights_sigmoid.sum(dim=1, keepdim=True)
                    print(f"[Class-Aware Weights] Raw sigmoid: {class_weights_sigmoid.cpu().numpy()}")
                    print(f"[Class-Aware Weights] Normalized: {class_weights_norm.cpu().numpy()}")
            
            # 打印可靠性门控信息（默认关闭，仅在需要时打开，减少日志 token）
            log_reliability_gate = bool(getattr(cfg, "LOG_RELIABILITY_GATE", False))
            if log_reliability_gate and hasattr(fusion_module, 'use_reliability_gating') and fusion_module.use_reliability_gating:
                gate_info = fusion_module.get_gate_info()
                if gate_info['uncertainties'] is not None:
                    uncertainties = gate_info['uncertainties']
                    gate_info_dict = gate_info.get('gate_info', {})
                    
                    # 获取权重（优先使用final_weights，否则使用weights）
                    if 'final_weights' in gate_info_dict:
                        weights = gate_info_dict['final_weights']
                    elif 'weights' in gate_info_dict:
                        weights = gate_info_dict['weights']
                    else:
                        weights = None
                    
                    if weights is not None:
                        print(f"[Reliability Gate] Uncertainties (Signal, Spec): "
                              f"mean=({uncertainties[:, 0].mean():.4f}, {uncertainties[:, 1].mean():.4f}), "
                              f"std=({uncertainties[:, 0].std():.4f}, {uncertainties[:, 1].std():.4f})")
                        print(f"[Reliability Gate] Weights (Signal, Spec): "
                              f"mean=({weights[:, 0].mean():.4f}, {weights[:, 1].mean():.4f}), "
                              f"std=({weights[:, 0].std():.4f}, {weights[:, 1].std():.4f})")
                        
                        # 如果使用注意力gate，显示额外信息
                        if hasattr(fusion_module.reliability_gate, 'use_attention') and fusion_module.reliability_gate.use_attention:
                            alpha = gate_info_dict.get('attention_weight_alpha', 0.0)
                            if alpha > 0 and 'attention_weights' in gate_info_dict:
                                attn_weights = gate_info_dict['attention_weights']
                                uncert_weights = gate_info_dict.get('uncertainty_weights', weights)
                                print(f"[Attention Gate] Alpha={alpha:.3f}, "
                                      f"Uncertainty weights mean=({uncert_weights[:, 0].mean():.4f}, {uncert_weights[:, 1].mean():.4f}), "
                                      f"Attention weights mean=({attn_weights[:, 0].mean():.4f}, {attn_weights[:, 1].mean():.4f})")

        # ---- 每 test_interval 轮进行目标域监控（默认关闭，避免训练期泄露）----
        metrics = {}
        try:
            enable_target_monitor = bool(getattr(cfg, "TRAIN_ENABLE_TARGET_MONITOR", False))
        except Exception:
            enable_target_monitor = False
        if enable_target_monitor and hust_loader is not None and build_text_prototypes_fn is not None and current_epoch % test_interval == 0:
            # 保存当前训练模式状态（zero_shot_eval_hust 会将模型设置为 eval 模式）
            signal_was_training = signal_encoder.training
            spec_was_training = spec_encoder.training
            fusion_was_training = fusion_module.training
            decoder_was_training = fusion_decoder.training
            
            # ATT392-S1: Learnable protos used directly for eval (zero-shot).
            # Fallback chain: learnable_proto → adapter+text → ETF → text.
            if learnable_proto_module is not None:
                eval_prototypes = learnable_proto_module.as_dict()
                _eval_proto_type = "LEARNABLE"
            elif proto_adapter is not None and _adapter_text_proto_tensor is not None:
                eval_prototypes = build_text_prototypes_fn(
                    text_encoder=text_encoder,
                    label_to_prompts=fault_prompt_templates,
                    device=device
                )
                _eval_proto_type = "TEXT(adapter)"
            elif use_etf and etf_prototypes is not None:
                eval_prototypes = etf_prototypes
                _eval_proto_type = "ETF"
            else:
                eval_prototypes = text_prototypes if text_prototypes is not None else initial_text_prototypes
                if not use_bayesian_prototypes:
                    eval_prototypes = build_text_prototypes_fn(
                        text_encoder=text_encoder,
                        label_to_prompts=fault_prompt_templates,
                        device=device
                    )
                _eval_proto_type = "TEXT"
            eval_prototypes = apply_prompt_tuned_prototypes(
                fusion_decoder=fusion_decoder,
                prototypes=eval_prototypes,
                detach_output=True
            )
            # ATT390: Domain embedding norm monitoring at eval time
            if hasattr(fusion_module, 'domain_embedding') and hasattr(fusion_module, '_trained_domain_ids'):
                _de_w = fusion_module.domain_embedding.weight.detach()
                _de_norms = _de_w.norm(dim=1)
                _trained = sorted(fusion_module._trained_domain_ids)
                print(f"[ATT390-S1-Eval] Domain emb norms: {[f'{n:.4f}' for n in _de_norms.tolist()]}, trained_ids={_trained}")
            print(f"[ATT391-Eval] eval_prototypes type={_eval_proto_type}, n_classes={len(eval_prototypes) if isinstance(eval_prototypes, dict) else 'N/A'}")
            # ATT391-S1: Log adapter state at eval
            if proto_adapter is not None:
                _ada_dev = proto_adapter.weight_deviation_from_identity()
                print(f"[ATT391-S1-Eval] Adapter weight deviation from identity: {_ada_dev:.6f}")

            # ===== 应用 EMA 权重进行评估（如果启用）=====
            if use_ema and ema_manager is not None:
                ema_manager.apply_all({
                    'signal_encoder': signal_encoder,
                    'spec_encoder': spec_encoder,
                    'fusion_module': fusion_module,
                    'fusion_decoder': fusion_decoder
                })

            # 在线微调配置（可以通过参数传入）
            # 注意：在线微调会修改模型参数，在训练阶段的验证中使用时需要谨慎
            # 建议只在最终评估时启用，避免影响训练过程
            metrics = zero_shot_eval_hust(
                signal_encoder=signal_encoder,
                spec_encoder=spec_encoder,
                fusion_module=fusion_module,
                fusion_decoder=fusion_decoder,
                text_prototypes=eval_prototypes,
                hust_loader=hust_loader,
                device=device,
                enable_finetuning=enable_finetuning,  # 使用传入的参数
                finetune_samples=finetune_samples,
                finetune_lr=finetune_lr,
                finetune_steps=finetune_steps
            )
            
            # ===== 恢复原始权重（如果使用了 EMA）=====
            if use_ema and ema_manager is not None:
                ema_manager.restore_all({
                    'signal_encoder': signal_encoder,
                    'spec_encoder': spec_encoder,
                    'fusion_module': fusion_module,
                    'fusion_decoder': fusion_decoder
                })
            
            # 恢复训练模式（zero_shot_eval_hust 会将模型设置为 eval 模式）
            if signal_was_training:
                signal_encoder.train()
            if spec_was_training:
                spec_encoder.train()
            if fusion_was_training:
                fusion_module.train()
            if decoder_was_training:
                fusion_decoder.train()

            # ===== (Margin Router) Update target every zero-shot =====
            # Detect dominant predicted class on HUST and focus margins accordingly.
            try:
                enable_router = bool(getattr(cfg, "ENABLE_MARGIN_ROUTER", True))
            except Exception:
                enable_router = True
            if enable_router:
                preds = metrics.get("fusion_decoder_preds", None)
                if preds is not None and len(preds) > 0:
                    try:
                        import numpy as _np
                        preds_arr = _np.asarray(preds, dtype=_np.int64)
                        counts = _np.bincount(preds_arr, minlength=4)
                        top = int(counts.argmax())
                        frac = float(counts.max()) / float(max(1, int(counts.sum())))
                        try:
                            frac_th = float(getattr(cfg, "MARGIN_ROUTER_PRED_RATIO_TH", 0.70))
                        except Exception:
                            frac_th = 0.70
                        if frac >= float(frac_th):
                            if top == 1:
                                margin_router_target = "vs_inner"
                            elif top == 3:
                                margin_router_target = "vs_rolling"
                            elif top == 0:
                                margin_router_target = "vs_normal"
                            elif top == 2:
                                margin_router_target = "vs_outer"
                            else:
                                margin_router_target = "all"
                            print(f"[Margin Router] Set margin_router_target={margin_router_target} (top_pred={top}, frac={frac:.3f}).")
                        else:
                            margin_router_target = "all"
                            print(f"[Margin Router] Keep margin_router_target=all (top_pred_frac={frac:.3f} < {frac_th:.2f}).")
                    except Exception:
                        pass

            # ===== (S-C2) Training health gate: phase switch on collapse =====
            # If HUST zero-shot diagnosis collapses (min per-class recall too low),
            # temporarily disable prototype-head training for a few epochs to recover stability.
            try:
                enable_train_gate = bool(getattr(cfg, "ENABLE_TRAIN_HEALTH_GATE", False))
                cooldown_epochs = int(getattr(cfg, "TRAIN_HEALTH_GATE_COOLDOWN_EPOCHS", 0))
                # Use relaxed threshold for training gate if set (e.g. 0.02 so prototype-head can train earlier).
                recall_th = getattr(cfg, "HEALTH_GATE_MIN_FAULT_RECALL", None)
                if recall_th is None:
                    recall_th = float(getattr(cfg, "BEST_CKPT_MIN_CLASS_RECALL", 0.0))
                else:
                    recall_th = float(recall_th)
                fault_only = bool(getattr(cfg, "HEALTH_GATE_FAULT_ONLY", True))
            except Exception:
                enable_train_gate = False
                cooldown_epochs = 0
                recall_th = 0.0
                fault_only = True

            if enable_train_gate and cooldown_epochs > 0 and recall_th > 0:
                per_class = metrics.get("fusion_decoder_acc_per_class_raw", None)
                if per_class is None:
                    per_class = metrics.get("fusion_decoder_acc_per_class", None)
                if isinstance(per_class, dict) and len(per_class) > 0:
                    try:
                        if fault_only:
                            # Only consider fault classes (1/2/3). This avoids false triggers when Normal recall=0 early.
                            keys = [k for k in per_class.keys() if int(k) in (1, 2, 3)]
                            vals = [float(per_class[k]) for k in keys] if len(keys) > 0 else [float(v) for v in per_class.values()]
                            min_recall = min(vals) if len(vals) > 0 else min(float(v) for v in per_class.values())
                            metric_name = "min_fault_recall"
                        else:
                            min_recall = min(float(v) for v in per_class.values())
                            metric_name = "min_class_recall"
                        # Avoid boundary triggers caused by float rounding in logs (e.g., "0.0500 < 0.0500").
                        try:
                            gate_eps = float(getattr(cfg, "HEALTH_GATE_EPS", 1e-6))
                        except Exception:
                            gate_eps = 1e-6
                        # Additional safeguard: only trigger cooldown when predictions are also highly collapsed
                        # (a single-class absorber). This prevents false triggers during "partial recovery"
                        # phases where one class recall is temporarily low but the distribution is not collapsed.
                        top_frac = 0.0
                        try:
                            preds = metrics.get("fusion_decoder_preds", None)
                            if preds is not None and len(preds) > 0:
                                import numpy as _np
                                preds_arr = _np.asarray(preds, dtype=_np.int64)
                                counts = _np.bincount(preds_arr, minlength=4)
                                top_frac = float(counts.max()) / float(max(1, int(counts.sum())))
                        except Exception:
                            top_frac = 0.0
                        try:
                            frac_th = float(getattr(cfg, "MARGIN_ROUTER_PRED_RATIO_TH", 0.70))
                        except Exception:
                            frac_th = 0.70

                        if (min_recall < float(recall_th) - float(gate_eps)) and (top_frac >= float(frac_th)):
                            proto_train_cooldown_until = max(int(proto_train_cooldown_until), int(current_epoch) + int(cooldown_epochs))
                            print(f"[Train Health Gate] Collapse detected on HUST ({metric_name}={min_recall:.4f} < {recall_th:.4f}). "
                                  f"Disable prototype-head training until epoch {proto_train_cooldown_until}.")
                    except Exception:
                        pass

            # ===== 优化后的输出格式：明确区分异常检测和故障诊断 =====
            print(f"\n{'='*80}")
            print(f"[Zero-shot Evaluation Epoch {current_epoch}]")
            print(f"{'='*80}")
            
            # ===== 第一部分：异常检测性能（Normal vs Fault，二分类）=====
            print("\n【异常检测性能 (Anomaly Detection: Normal vs Fault)】")
            print("-" * 80)
            auc_nf = metrics.get('auc_nf', float('nan'))
            if not (isinstance(auc_nf, float) and (auc_nf != auc_nf)):  # 不是 NaN
                print(f"  AUC_nf (主要指标): {auc_nf:.4f}")
                print(f"    说明: 使用分层NF头的全局二分类性能（Normal vs Any Fault）")
            else:
                print(f"  AUC_nf: N/A (未计算)")
            print()
            
            # ===== 第二部分：故障诊断性能（4类分类）=====
            print("【故障诊断性能 (Fault Diagnosis: 4-Class Classification)】")
            print("-" * 80)
            print(f"  整体准确率 (Overall Accuracy):")
            print(f"    Signal Encoder:     {metrics['signal_zero_shot_acc']:.4f} ({metrics['signal_zero_shot_acc']*100:.2f}%)")
            print(f"    Spec Encoder:       {metrics['spec_zero_shot_acc']:.4f} ({metrics['spec_zero_shot_acc']*100:.2f}%)")
            print(f"    Fusion (Zero-shot): {metrics['fusion_zero_shot_acc']:.4f} ({metrics['fusion_zero_shot_acc']*100:.2f}%)")
            print(f"    Fusion Decoder:     {metrics['fusion_decoder_acc']:.4f} ({metrics['fusion_decoder_acc']*100:.2f}%) ⭐")
            if 'fusion_decoder_acc_raw' in metrics:
                print(f"    Fusion Decoder (raw argmax): {metrics['fusion_decoder_acc_raw']:.4f} ({metrics['fusion_decoder_acc_raw']*100:.2f}%)")
            # ATT110: report spec_head direct 4-class accuracy alongside fusion decoder
            if metrics.get('spec_head_diag_acc') is not None:
                print(f"    SpecHead Direct:    {metrics['spec_head_diag_acc']:.4f} ({metrics['spec_head_diag_acc']*100:.2f}%) [ATT110 spec_head bypass]")
            print(f"    说明: Fusion Decoder 是主要诊断方法，其他为对比基准")
            print()
            
            # 类别级性能监控
            if enable_class_monitoring:
                # 获取预测结果（如果可用）
                if 'signal_preds' in metrics and 'spec_preds' in metrics:
                    # 计算类别级指标
                    num_classes = len(metrics.get('signal_acc_per_class', {}))
                    if num_classes > 0:
                        # 获取预测结果和标签
                        labels = metrics.get('labels', [])
                        signal_preds = metrics.get('signal_preds', [])
                        spec_preds = metrics.get('spec_preds', [])
                        fusion_zs_preds = metrics.get('fusion_zero_shot_preds', [])
                        fusion_dec_preds = metrics.get('fusion_decoder_preds', [])
                        
                        signal_metrics = compute_class_metrics(
                            labels, signal_preds, num_classes
                        )
                        spec_metrics = compute_class_metrics(
                            labels, spec_preds, num_classes
                        )
                        fusion_metrics = compute_class_metrics(
                            labels, fusion_zs_preds, num_classes
                        ) if len(fusion_zs_preds) > 0 else None
                        fusion_dec_metrics = compute_class_metrics(
                            labels, fusion_dec_preds, num_classes
                        ) if len(fusion_dec_preds) > 0 else None
                        
                        # 类别级详细指标（仅显示 Fusion Decoder，主要方法）
                        print(f"  类别级详细指标 (Per-Class Metrics - Fusion Decoder):")
                        if fusion_dec_metrics:
                            # 定义类别名称
                            class_names = {0: "Normal", 1: "Inner Race", 2: "Outer Race", 3: "Rolling Element"}
                            print(f"    {'Class':<6} {'Name':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                            print(f"    {'-'*6} {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
                            for c in range(num_classes):
                                class_name = class_names.get(c, f"Class{c}")
                                print(f"    {c:<6} {class_name:<15} "
                                      f"{fusion_dec_metrics[c]['precision']:<10.4f} "
                                      f"{fusion_dec_metrics[c]['recall']:<10.4f} "
                                      f"{fusion_dec_metrics[c]['f1']:<10.4f} "
                                      f"{fusion_dec_metrics[c]['support']:<8}")
                            print()
                            
                            # 汇总指标
                            print(f"  汇总指标 (Summary Metrics - Fusion Decoder):")
                            print(f"    宏平均 (Macro Avg):   P={fusion_dec_metrics['macro']['precision']:.4f}, "
                                  f"R={fusion_dec_metrics['macro']['recall']:.4f}, "
                                  f"F1={fusion_dec_metrics['macro']['f1']:.4f}")
                            print(f"    加权平均 (Weighted):   P={fusion_dec_metrics['weighted']['precision']:.4f}, "
                                  f"R={fusion_dec_metrics['weighted']['recall']:.4f}, "
                                  f"F1={fusion_dec_metrics['weighted']['f1']:.4f}")
                            # Dual-level logging: concise natural-language summary for readability.
                            try:
                                nf_text = f"AUC_nf={auc_nf:.4f}" if not (isinstance(auc_nf, float) and (auc_nf != auc_nf)) else "AUC_nf=N/A"
                                diag_acc = float(metrics.get("fusion_decoder_acc", 0.0))
                                normal_r = float(fusion_dec_metrics[0]['recall']) if 0 in fusion_dec_metrics else float("nan")
                                inner_r = float(fusion_dec_metrics[1]['recall']) if 1 in fusion_dec_metrics else float("nan")
                                outer_r = float(fusion_dec_metrics[2]['recall']) if 2 in fusion_dec_metrics else float("nan")
                                rolling_r = float(fusion_dec_metrics[3]['recall']) if 3 in fusion_dec_metrics else float("nan")
                                print("  自然语言摘要 (Readable Summary):")
                                print(
                                    f"    本轮 Normal-vs-Fault 指标 {nf_text}；"
                                    f"4类诊断 Acc={diag_acc:.4f}, Macro-F1={fusion_dec_metrics['macro']['f1']:.4f}。"
                                )
                                print(
                                    f"    召回分布: Normal={normal_r:.3f}, Inner={inner_r:.3f}, "
                                    f"Outer={outer_r:.3f}, Rolling={rolling_r:.3f}。"
                                )
                            except Exception:
                                pass
                        print()
                else:
                    # 如果没有预测结果，至少打印per-class accuracy
                    print("\n[Per-class Accuracy]")
                    if 'signal_acc_per_class' in metrics:
                        print(f"Signal: {metrics['signal_acc_per_class']}")
                    if 'spec_acc_per_class' in metrics:
                        print(f"Spec: {metrics['spec_acc_per_class']}")
                    if 'fusion_zero_shot_acc_per_class' in metrics:
                        print(f"Fusion: {metrics['fusion_zero_shot_acc_per_class']}")
                    if 'fusion_decoder_acc_per_class' in metrics:
                        print(f"FusionDec: {metrics['fusion_decoder_acc_per_class']}")
            
            # ===== 混淆矩阵和主要混淆对（故障诊断详细分析）=====
            if enable_class_monitoring and 'fusion_decoder_confusion_matrix' in metrics:
                cm = metrics['fusion_decoder_confusion_matrix']
                class_names = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                
                print("【故障诊断详细分析 (Diagnosis Details)】")
                print("-" * 80)
                print("  混淆矩阵 (Confusion Matrix - Fusion Decoder):")
                # Print header with class names
                header = "True\\Pred |" + "".join(f" {class_names.get(i, str(i)):>6} |" for i in range(cm.size(0)))
                print(f"    {header}")
                print(f"    {'-'*len(header)}")
                # Print rows
                for i in range(cm.size(0)):
                    row = f"    {class_names.get(i, str(i)):>6} |" + "".join(f" {cm[i, j]:>6} |" for j in range(cm.size(1)))
                    print(row)
                print()
                
                # Print main confusion pairs
                if 'fusion_decoder_main_confusions' in metrics:
                    main_confusions = metrics['fusion_decoder_main_confusions']
                    if main_confusions:
                        print("  主要混淆对 (Top Confusion Pairs):")
                        for idx, (true_label, pred_label, count) in enumerate(main_confusions, 1):
                            true_name = class_names.get(true_label, f"Class{true_label}")
                            pred_name = class_names.get(pred_label, f"Class{pred_label}")
                            print(f"    {idx}. {true_name} → {pred_name}: {count} samples")
                print()

                # ===== (Probe) Split-head validation signals =====
                # These help diagnose "roll absorber" vs "NF decision harming acc" without extra runs.
                try:
                    probe = metrics.get("split_head_probe", None)
                except Exception:
                    probe = None
                if isinstance(probe, dict) and len(probe) > 0:
                    print("【验证探针 (Validation Probes)】")
                    print("-" * 80)
                    try:
                        if "decision_override_frac" in probe:
                            print(f"  Decision override frac (raw_argmax → decision_mode): {float(probe['decision_override_frac']):.4f}")
                        if "decision_override_frac_by_true" in probe and isinstance(probe["decision_override_frac_by_true"], dict):
                            by_true = probe["decision_override_frac_by_true"]
                            # print in class order
                            line = "  Override frac by TRUE class: "
                            parts_line = []
                            for c in [0, 1, 2, 3]:
                                if c in by_true:
                                    parts_line.append(f"{class_names.get(c,str(c))}={float(by_true[c]):.3f}")
                            if parts_line:
                                print(line + ", ".join(parts_line))
                    except Exception:
                        pass
                    try:
                        if "io_acc" in probe:
                            print(f"  IO-head acc on TRUE {{Inner,Outer}}: {float(probe['io_acc']):.4f}")
                        if "io_confusion" in probe and torch.is_tensor(probe["io_confusion"]):
                            cmio = probe["io_confusion"]
                            if cmio.numel() == 4:
                                # rows: true (inner,outer), cols: pred (inner,outer)
                                print("  IO-head confusion (TRUE\\PRED | inner | outer):")
                                print(f"    inner | {int(cmio[0,0].item()):5d} | {int(cmio[0,1].item()):5d} |")
                                print(f"    outer | {int(cmio[1,0].item()):5d} | {int(cmio[1,1].item()):5d} |")
                    except Exception:
                        pass
                    try:
                        if "rolling" in probe and isinstance(probe["rolling"], dict):
                            print("  Rolling head (sigmoid) by TRUE class:")
                            for c in [0, 1, 2, 3]:
                                if c not in probe["rolling"]:
                                    continue
                                s = probe["rolling"][c]
                                print(
                                    f"    {class_names.get(c,str(c)):<7} "
                                    f"p_mean={float(s.get('p_mean',0.0)):.3f}, "
                                    f"p>0.5={float(s.get('p_gt_0.5',0.0)):.3f}, "
                                    f"logit_mean={float(s.get('logit_mean',0.0)):.3f}"
                                )
                    except Exception:
                        pass
                    print()
            
            print(f"{'='*80}\n")

            # 保存到 history
            history["signal_zero_shot_acc"].append(metrics["signal_zero_shot_acc"])
            history["spec_zero_shot_acc"].append(metrics["spec_zero_shot_acc"])
            history["fusion_zero_shot_acc"].append(metrics["fusion_zero_shot_acc"])
            history["fusion_decoder_acc"].append(metrics["fusion_decoder_acc"])

            # ===== (Attempt 40) Save HUST-best checkpoint (analysis only) =====
            # NOTE: this is only enabled when TRAIN_ENABLE_TARGET_MONITOR=True.
            try:
                current_hust = float(metrics.get("fusion_decoder_acc", 0.0))
            except Exception:
                current_hust = 0.0
            if current_hust > float(hust_best_fusion_decoder_acc) + 1e-12:
                hust_best_fusion_decoder_acc = float(current_hust)
                hust_best_epoch = int(current_epoch)
                _prev_hust_ckpt = hust_best_checkpoint_path
                hust_best_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{current_epoch}_hust_best.pth")
                if _prev_hust_ckpt is not None and os.path.exists(_prev_hust_ckpt) and _prev_hust_ckpt != hust_best_checkpoint_path:
                    try:
                        os.remove(_prev_hust_ckpt)
                    except Exception:
                        pass
                try:
                    ckpt_text_prototypes = eval_prototypes if eval_prototypes is not None else text_prototypes
                    # If evaluation used EMA weights, persist the EMA weights into the saved checkpoint too.
                    applied_ema_for_save = False
                    if use_ema and ema_manager is not None:
                        try:
                            ema_manager.apply_all({
                                'signal_encoder': signal_encoder,
                                'spec_encoder': spec_encoder,
                                'fusion_module': fusion_module,
                                'fusion_decoder': fusion_decoder
                            })
                            applied_ema_for_save = True
                        except Exception:
                            applied_ema_for_save = False

                    save_checkpoint(
                        hust_best_checkpoint_path,
                        signal_encoder,
                        spec_encoder,
                        text_encoder,
                        fusion_module,
                        fusion_decoder,
                        optimizer,
                        current_epoch,
                        avg_total,
                        text_prototypes=ckpt_text_prototypes,
                        ema_manager=ema_manager if use_ema else None,
                        proto_adapter=proto_adapter,
                        learnable_proto_module=learnable_proto_module,
                        decouple_spec=decouple_spec,
                        decouple_signal=decouple_signal,
                        condition_classifier=condition_classifier,
                    )
                    print(
                        f"[HUST Best Checkpoint] Saved new HUST-best checkpoint: {hust_best_checkpoint_path} "
                        f"(fusion_decoder_acc={hust_best_fusion_decoder_acc:.4f})"
                    )
                    if applied_ema_for_save and use_ema and ema_manager is not None:
                        try:
                            ema_manager.restore_all({
                                'signal_encoder': signal_encoder,
                                'spec_encoder': spec_encoder,
                                'fusion_module': fusion_module,
                                'fusion_decoder': fusion_decoder
                            })
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[HUST Best Checkpoint] Warning: failed to save HUST-best checkpoint: {e}")
        
        # ---- 验证集评估（每个epoch都做，用于early-stop） ----
        if val_loader is not None and use_early_stopping:
            val_metrics = evaluate_on_val_loader(
                val_loader,
                signal_encoder,
                spec_encoder,
                fusion_module,
                fusion_decoder,
                text_encoder,
                device,
                precomputed_class_weights=precomputed_class_weights,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                label_smoothing=label_smoothing,
                use_moe=use_moe,
                text_prototypes=text_prototypes,
                wst_encoder=wst_encoder,
                _wst_bypass_active=_wst_bypass_active,
                env_encoder=env_encoder,
                _env_bypass_active=_env_bypass_active,
                scd_encoder=scd_encoder,
                _scd_bypass_active=_scd_bypass_active,
            )
            
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])
            # ATT132 P0: track val_spec_head_acc per epoch
            _vsha = val_metrics.get("val_spec_head_acc", float('nan'))
            if not (isinstance(_vsha, float) and _vsha != _vsha):  # not NaN
                history["val_spec_head_acc"].append(_vsha)
            
            _spec_acc_str = f", SpecHead Acc={_vsha:.4f}" if not (isinstance(_vsha, float) and _vsha != _vsha) else ""
            print(f"[Validation] Epoch {current_epoch}: Val Loss={val_metrics['val_loss']:.4f}, Val Acc={val_metrics['val_acc']:.4f}{_spec_acc_str}")

            if bool(getattr(cfg, "DUAL_CONTRACT_ENABLE", False)):
                _vf_contract = float(val_metrics.get("val_fusion_diag_acc", val_metrics["val_acc"]))
                _spec_contract_str = (
                    f"{_vsha:.4f}" if not (isinstance(_vsha, float) and _vsha != _vsha) else "nan"
                )
                print(
                    f"[ATT475-ValContracts] Epoch {current_epoch}: "
                    f"spec_contract(val_spec_head_acc)={_spec_contract_str} "
                    f"fusion_contract(val_fusion_diag_acc)={_vf_contract:.4f}"
                )

            if _att475_dual_ckpt:
                try:
                    ckpt_tp_dc = text_prototypes
                    if use_bayesian_prototypes:
                        ckpt_tp_dc = text_prototypes if text_prototypes is not None else initial_text_prototypes
                    _fus_val = float(val_metrics["val_acc"])
                    _spec_val = val_metrics.get("val_spec_head_acc", float("nan"))
                    _tag_prefix = f"{checkpoint_tag}_" if checkpoint_tag else ""

                    if not (isinstance(_spec_val, float) and _spec_val != _spec_val) and _spec_val > _att475_best_spec_v:
                        _new_sp = os.path.join(
                            checkpoint_dir,
                            f"{_tag_prefix}epoch_{current_epoch}_ATT475_best_spec.pth",
                        )
                        if _att475_best_spec_path is not None and os.path.exists(_att475_best_spec_path) and _att475_best_spec_path != _new_sp:
                            try:
                                os.remove(_att475_best_spec_path)
                            except Exception:
                                pass
                        save_checkpoint(
                            _new_sp,
                            signal_encoder,
                            spec_encoder,
                            text_encoder,
                            fusion_module,
                            fusion_decoder,
                            optimizer,
                            current_epoch,
                            float(_spec_val),
                            text_prototypes=ckpt_tp_dc,
                            ema_manager=ema_manager if use_ema else None,
                            proto_adapter=proto_adapter,
                            learnable_proto_module=learnable_proto_module,
                            decouple_spec=decouple_spec,
                            decouple_signal=decouple_signal,
                            condition_classifier=condition_classifier,
                        )
                        _att475_best_spec_v = float(_spec_val)
                        _att475_best_spec_path = _new_sp
                        print(f"[ATT475-DualCkpt] Saved spec-contract best: {_new_sp} (val_spec_head_acc={_spec_val:.4f})")

                    if _fus_val > _att475_best_fus_v:
                        _new_fus = os.path.join(
                            checkpoint_dir,
                            f"{_tag_prefix}epoch_{current_epoch}_ATT475_best_fusion.pth",
                        )
                        if _att475_best_fus_path is not None and os.path.exists(_att475_best_fus_path) and _att475_best_fus_path != _new_fus:
                            try:
                                os.remove(_att475_best_fus_path)
                            except Exception:
                                pass
                        save_checkpoint(
                            _new_fus,
                            signal_encoder,
                            spec_encoder,
                            text_encoder,
                            fusion_module,
                            fusion_decoder,
                            optimizer,
                            current_epoch,
                            _fus_val,
                            text_prototypes=ckpt_tp_dc,
                            ema_manager=ema_manager if use_ema else None,
                            proto_adapter=proto_adapter,
                            learnable_proto_module=learnable_proto_module,
                            decouple_spec=decouple_spec,
                            decouple_signal=decouple_signal,
                            condition_classifier=condition_classifier,
                        )
                        _att475_best_fus_v = _fus_val
                        _att475_best_fus_path = _new_fus
                        print(f"[ATT475-DualCkpt] Saved fusion-contract best: {_new_fus} (val_fusion_diag_acc={_fus_val:.4f})")
                except Exception as _att475_ck_e:
                    print(f"[ATT475-DualCkpt] Warning: {_att475_ck_e}")

            # ATT453: z_spec class centroid cosine diagnostics (every epoch)
            try:
                _diag_z_specs = val_metrics.get("_diag_z_spec_by_class", None)
                if _diag_z_specs is not None and len(_diag_z_specs) > 0:
                    _cls_ids = sorted(_diag_z_specs.keys())
                    _centroids = {}
                    for _cid in _cls_ids:
                        _feats = _diag_z_specs[_cid]
                        if len(_feats) > 0:
                            _stack = torch.stack(_feats)
                            _centroids[_cid] = F.normalize(_stack.mean(dim=0), dim=0)
                    if len(_centroids) >= 2:
                        _cos_parts = []
                        _cids = sorted(_centroids.keys())
                        for _i, _ci in enumerate(_cids):
                            for _cj in _cids[_i + 1:]:
                                _cos_val = float(torch.dot(_centroids[_ci], _centroids[_cj]))
                                _cos_parts.append(f"({_ci},{_cj})={_cos_val:.4f}")
                        print(f"[ATT453-zSpec] E{current_epoch} centroid_cos=[{', '.join(_cos_parts)}]")
            except Exception:
                pass

            # ===== Early stopping 检查（使用验证集指标，不使用目标域） =====
            improved = False
            # Smooth the early-stop metric to reduce noise sensitivity (helps when runs/epochs are limited)
            try:
                smooth_window = int(getattr(cfg, "EARLY_STOP_SMOOTH_WINDOW", 1))
            except Exception:
                smooth_window = 1
            smooth_window = max(1, smooth_window)

            # Only update best checkpoint after min_epochs (ensures major structures are enabled)
            if current_epoch < early_stop_min_epochs:
                # ATT130 A: During pre-window, track true best val_loss and save a checkpoint
                # whenever a new minimum is found — this allows the post-window baseline to start
                # from the *actual* best pre-window value rather than the last epoch's value.
                # Bug history: ATT125 and ATT129 both suffered from resetting baseline to history[-1]
                # (e.g., E9=0.4916) while a superior checkpoint existed at E8=0.3436.
                pre_window_improved = False
                if early_stop_metric == "val_loss":
                    tail = history["val_loss"][-smooth_window:] if len(history["val_loss"]) > 0 else [val_metrics["val_loss"]]
                    current_pre_val = float(sum(tail) / max(1, len(tail)))
                    if current_pre_val < best_pre_window_value:
                        best_pre_window_value = current_pre_val
                        best_pre_window_epoch = current_epoch
                        pre_window_improved = True
                elif early_stop_metric == "val_acc":
                    tail = history["val_acc"][-smooth_window:] if len(history["val_acc"]) > 0 else [val_metrics["val_acc"]]
                    current_pre_val = float(sum(tail) / max(1, len(tail)))
                    if current_pre_val > best_pre_window_value:
                        best_pre_window_value = current_pre_val
                        best_pre_window_epoch = current_epoch
                        pre_window_improved = True
                elif early_stop_metric == "val_spec_head_acc":
                    # ATT132 P0: spec_head MLP 独立路径准确率（越大越好）
                    # 数学理据：与 fusion decoder 路径解耦，直接优化 spec_head 判别能力
                    # 解决 ATT131 Epoch8(val_loss最优) vs Epoch9(spec_head更优)的路径不匹配
                    _sha_hist = history["val_spec_head_acc"]
                    tail = _sha_hist[-smooth_window:] if len(_sha_hist) > 0 else [val_metrics.get("val_spec_head_acc", -1.0)]
                    current_pre_val = float(sum(tail) / max(1, len(tail)))
                    if current_pre_val > best_pre_window_value:
                        best_pre_window_value = current_pre_val
                        best_pre_window_epoch = current_epoch
                        pre_window_improved = True

                if pre_window_improved:
                    try:
                        pre_ckpt_path = os.path.join(
                            checkpoint_dir,
                            _prebest_checkpoint_filename(current_epoch, checkpoint_tag)
                        )
                        ckpt_text_prototypes = text_prototypes
                        if use_bayesian_prototypes:
                            ckpt_text_prototypes = text_prototypes if text_prototypes is not None else initial_text_prototypes
                        save_checkpoint(
                            pre_ckpt_path,
                            signal_encoder,
                            spec_encoder,
                            text_encoder,
                            fusion_module,
                            fusion_decoder,
                            optimizer,
                            current_epoch,
                            best_pre_window_value,
                            text_prototypes=ckpt_text_prototypes,
                            ema_manager=ema_manager if use_ema else None,
                            proto_adapter=proto_adapter,
                            learnable_proto_module=learnable_proto_module,
                            decouple_spec=decouple_spec,
                            decouple_signal=decouple_signal,
                            condition_classifier=condition_classifier,
                        )
                        if best_pre_window_ckpt_path is not None and os.path.exists(best_pre_window_ckpt_path) and best_pre_window_ckpt_path != pre_ckpt_path:
                            try:
                                os.remove(best_pre_window_ckpt_path)
                            except Exception:
                                pass
                        best_pre_window_ckpt_path = pre_ckpt_path
                        print(f"[Early Stop] Pre-window best {early_stop_metric}={best_pre_window_value:.4f} at epoch {current_epoch}; saved {pre_ckpt_path}")
                    except Exception as e:
                        print(f"[Warning] Failed to save pre-window best checkpoint: {e}")
                else:
                    print(f"[Early Stop] Epoch {current_epoch} < min_epochs ({early_stop_min_epochs}), "
                          f"tracking pre-window best={best_pre_window_value:.4f} at epoch {best_pre_window_epoch}.")

                # ATT130 A Fix: at the boundary epoch, reset baseline to the TRUE pre-window minimum,
                # not just the last-epoch value. Also promote the best pre-window checkpoint.
                # ATT168: when early_stop_skip_prewindow_baseline=True, discard pre-window best entirely;
                # best_metric_value stays at float('inf'), so tracking starts fresh from min_epochs.
                # This prevents noisy early-epoch troughs (e.g. Task2 E2 val_loss=0.19) from setting
                # an unreachable baseline that causes premature ES.
                if current_epoch == early_stop_min_epochs - 1:
                    if early_stop_skip_prewindow_baseline:
                        print(f"[Early Stop] skip_prewindow_baseline=True: discarding pre-window best "
                              f"({early_stop_metric}={best_pre_window_value:.4f} at epoch {best_pre_window_epoch}); "
                              f"tracking starts fresh from epoch {early_stop_min_epochs}.")
                        # best_metric_value remains float('inf')/-1.0, best_metric_epoch remains -1
                    else:
                        best_metric_value = best_pre_window_value
                        best_metric_epoch = best_pre_window_epoch
                        print(f"[Early Stop] Baseline reset to true pre-window best: "
                              f"{early_stop_metric}={best_metric_value:.4f} at epoch {best_metric_epoch} "
                              f"(will start tracking from epoch {early_stop_min_epochs}).")

                    # Promote the best pre-window checkpoint as the starting best checkpoint
                    # (only when NOT skipping pre-window baseline)
                    if not early_stop_skip_prewindow_baseline and best_pre_window_ckpt_path is not None:
                        try:
                            promoted_path = os.path.join(
                                checkpoint_dir,
                                _best_checkpoint_filename(best_metric_epoch, checkpoint_tag)
                            )
                            import shutil
                            shutil.copy2(best_pre_window_ckpt_path, promoted_path)
                            best_checkpoint_path = promoted_path
                            print(f"[Best Checkpoint] Promoted pre-window best checkpoint to: {promoted_path}")
                        except Exception as e:
                            print(f"[Warning] Failed to promote pre-window best checkpoint: {e}")
                            # Fallback: save current model state at min_epochs-1
                            try:
                                reset_ckpt_path = os.path.join(
                                    checkpoint_dir,
                                    _best_checkpoint_filename(current_epoch, checkpoint_tag)
                                )
                                ckpt_text_prototypes = text_prototypes
                                if use_bayesian_prototypes:
                                    ckpt_text_prototypes = text_prototypes if text_prototypes is not None else initial_text_prototypes
                                save_checkpoint(
                                    reset_ckpt_path,
                                    signal_encoder,
                                    spec_encoder,
                                    text_encoder,
                                    fusion_module,
                                    fusion_decoder,
                                    optimizer,
                                    current_epoch,
                                    best_metric_value,
                                    text_prototypes=ckpt_text_prototypes,
                                    ema_manager=ema_manager if use_ema else None,
                                    proto_adapter=proto_adapter,
                                    learnable_proto_module=learnable_proto_module,
                                    decouple_spec=decouple_spec,
                                    decouple_signal=decouple_signal,
                                    condition_classifier=condition_classifier,
                                )
                                print(f"[Best Checkpoint] Fallback: saved baseline checkpoint at reset epoch: {reset_ckpt_path}")
                            except Exception as e2:
                                print(f"[Warning] Fallback checkpoint save also failed: {e2}")
                    elif not early_stop_skip_prewindow_baseline:
                        # No pre-window best found yet — save current state as fallback
                        # (only when NOT skipping pre-window baseline)
                        try:
                            reset_ckpt_path = os.path.join(
                                checkpoint_dir,
                                _best_checkpoint_filename(current_epoch, checkpoint_tag)
                            )
                            ckpt_text_prototypes = text_prototypes
                            if use_bayesian_prototypes:
                                ckpt_text_prototypes = text_prototypes if text_prototypes is not None else initial_text_prototypes
                            save_checkpoint(
                                reset_ckpt_path,
                                signal_encoder,
                                spec_encoder,
                                text_encoder,
                                fusion_module,
                                fusion_decoder,
                                optimizer,
                                current_epoch,
                                best_metric_value,
                                text_prototypes=ckpt_text_prototypes,
                                ema_manager=ema_manager if use_ema else None,
                                proto_adapter=proto_adapter,
                                learnable_proto_module=learnable_proto_module,
                                decouple_spec=decouple_spec,
                                decouple_signal=decouple_signal,
                                condition_classifier=condition_classifier,
                            )
                            print(f"[Best Checkpoint] Saved baseline best checkpoint at reset epoch: {reset_ckpt_path}")
                        except Exception as e:
                            print(f"[Warning] Failed to save baseline best checkpoint at reset epoch: {e}")
            else:
                # After min_epochs, normal best-checkpoint tracking
                if early_stop_metric == "val_loss":
                    # smoothed val_loss (lower is better)
                    tail = history["val_loss"][-smooth_window:] if len(history["val_loss"]) > 0 else [val_metrics["val_loss"]]
                    current_metric_value = float(sum(tail) / max(1, len(tail)))
                    # 损失越小越好
                    if current_metric_value < best_metric_value:
                        best_metric_value = current_metric_value
                        best_metric_epoch = current_epoch
                        no_improve_count = 0
                        improved = True
                        print(f"[Early Stop] New best {early_stop_metric} (smooth@{smooth_window}): {best_metric_value:.4f} at epoch {best_metric_epoch}")
                    else:
                        no_improve_count += 1
                        print(f"[Early Stop] No improvement for {no_improve_count} epoch(s), best {early_stop_metric} (smooth@{smooth_window}): {best_metric_value:.4f} at epoch {best_metric_epoch}")
                elif early_stop_metric == "val_acc":
                    # smoothed val_acc (higher is better)
                    tail = history["val_acc"][-smooth_window:] if len(history["val_acc"]) > 0 else [val_metrics["val_acc"]]
                    current_metric_value = float(sum(tail) / max(1, len(tail)))
                    # 准确率越大越好
                    if current_metric_value > best_metric_value:
                        best_metric_value = current_metric_value
                        best_metric_epoch = current_epoch
                        no_improve_count = 0
                        improved = True
                        print(f"[Early Stop] New best {early_stop_metric} (smooth@{smooth_window}): {best_metric_value:.4f} at epoch {best_metric_epoch}")
                    else:
                        no_improve_count += 1
                        print(f"[Early Stop] No improvement for {no_improve_count} epoch(s), best {early_stop_metric} (smooth@{smooth_window}): {best_metric_value:.4f} at epoch {best_metric_epoch}")
                elif early_stop_metric == "val_spec_head_acc":
                    # ATT132 P0: smoothed val_spec_head_acc (higher is better)
                    # spec_head MLP 独立路径准确率，直接对齐 spec_head_diag_acc 目标
                    _sha_hist = history["val_spec_head_acc"]
                    tail = _sha_hist[-smooth_window:] if len(_sha_hist) > 0 else [val_metrics.get("val_spec_head_acc", -1.0)]
                    current_metric_value = float(sum(tail) / max(1, len(tail)))
                    if current_metric_value > best_metric_value:
                        best_metric_value = current_metric_value
                        best_metric_epoch = current_epoch
                        no_improve_count = 0
                        improved = True
                        print(f"[Early Stop] New best {early_stop_metric} (smooth@{smooth_window}): {best_metric_value:.4f} at epoch {best_metric_epoch}")
                    else:
                        no_improve_count += 1
                        print(f"[Early Stop] No improvement for {no_improve_count} epoch(s), best {early_stop_metric} (smooth@{smooth_window}): {best_metric_value:.4f} at epoch {best_metric_epoch}")
                else:
                    # 如果仍使用旧指标（融合zero-shot/decoder准确率），保持向后兼容
                    # 但这些指标来自HUST，会造成泄露，应避免使用
                    print(f"[Warning] early_stop_metric='{early_stop_metric}' is deprecated. Use 'val_loss' or 'val_acc' to avoid target-domain leakage.")
                    if hust_loader is not None and (current_epoch % test_interval == 0 or current_epoch == num_epochs):
                        # 仅在测试间隔时才有这些指标
                        if early_stop_metric == "fusion_zero_shot_acc":
                            current_metric_value = metrics.get("fusion_zero_shot_acc", 0.0)
                        elif early_stop_metric == "fusion_decoder_acc":
                            current_metric_value = metrics.get("fusion_decoder_acc", 0.0)
                        else:
                            current_metric_value = metrics.get("fusion_zero_shot_acc", 0.0)

                        if current_metric_value > best_metric_value:
                            best_metric_value = current_metric_value
                            best_metric_epoch = current_epoch
                            no_improve_count = 0
                            improved = True
                        else:
                            no_improve_count += 1

            # ===== 保存 best checkpoint（当指标改善时立即保存） =====
            # ATT133 P0: do NOT hardcode metric names here.
            # Any early_stop_metric branch that sets improved=True must persist a best checkpoint.
            if improved:
                try:
                    # ===== (S-C) Health gate =====
                    # Strict no-leakage default: do not use target-domain labels/metrics for checkpoint gating.
                    try:
                        enable_health_gate = bool(getattr(cfg, "ENABLE_BEST_CKPT_HEALTH_GATE", False))
                        use_target_health_gate = bool(getattr(cfg, "USE_TARGET_DOMAIN_HEALTH_GATE_FOR_BEST_CKPT", False))
                        min_recall_th = float(getattr(cfg, "BEST_CKPT_MIN_CLASS_RECALL", 0.0))
                    except Exception:
                        enable_health_gate = False
                        use_target_health_gate = False
                        min_recall_th = 0.0

                    allow_save = True
                    if enable_health_gate and use_target_health_gate and hust_loader is not None and min_recall_th > 0:
                        try:
                            # ATT375: use ETF prototypes when available
                            if use_etf and etf_prototypes is not None:
                                hp = etf_prototypes
                            else:
                                hp = text_prototypes if text_prototypes is not None else initial_text_prototypes
                            health_metrics = zero_shot_eval_hust(
                                signal_encoder=signal_encoder,
                                spec_encoder=spec_encoder,
                                fusion_module=fusion_module,
                                fusion_decoder=fusion_decoder,
                                text_prototypes=hp,
                                hust_loader=hust_loader,
                                device=device,
                                enable_finetuning=False,
                                finetune_samples=0,
                                finetune_lr=1e-4,
                                finetune_steps=0
                            )
                            per_class = health_metrics.get("fusion_decoder_acc_per_class_raw", None)
                            if per_class is None:
                                per_class = health_metrics.get("fusion_decoder_acc_per_class", None)
                            if isinstance(per_class, dict) and len(per_class) > 0:
                                try:
                                    fault_only = bool(getattr(cfg, "HEALTH_GATE_FAULT_ONLY", True))
                                except Exception:
                                    fault_only = True
                                if fault_only:
                                    keys = [k for k in per_class.keys() if int(k) in (1, 2, 3)]
                                    items = [(k, float(per_class[k])) for k in keys] if len(keys) > 0 else [(k, float(v)) for k, v in per_class.items()]
                                    if len(items) > 0:
                                        min_k, min_recall = min(items, key=lambda kv: kv[1])
                                    else:
                                        min_k, min_recall = None, 0.0
                                    metric_name = "min_fault_recall"
                                else:
                                    items = [(k, float(v)) for k, v in per_class.items()]
                                    min_k, min_recall = min(items, key=lambda kv: kv[1]) if len(items) > 0 else (None, 0.0)
                                    metric_name = "min_class_recall"
                                # Avoid boundary triggers caused by float rounding in logs (e.g., "0.0500 < 0.0500").
                                try:
                                    gate_eps = float(getattr(cfg, "HEALTH_GATE_EPS", 1e-6))
                                except Exception:
                                    gate_eps = 1e-6
                                # Attempt 39: relax one-vote veto. If configured, require at least K fault classes
                                # to exceed recall threshold. This allows saving E10-like peaks where Rolling may be 0.
                                try:
                                    min_fault_classes = getattr(cfg, "HEALTH_GATE_MIN_FAULT_CLASSES", None)
                                    if min_fault_classes is not None:
                                        min_fault_classes = int(min_fault_classes)
                                except Exception:
                                    min_fault_classes = None

                                if fault_only and (min_fault_classes is not None) and (min_fault_classes > 0):
                                    fault_items = [(k, float(v)) for k, v in items]
                                    ok = [(k, v) for k, v in fault_items if v >= float(min_recall_th) - float(gate_eps)]
                                    num_ok = int(len(ok))
                                    if num_ok < int(min_fault_classes):
                                        print(
                                            f"[Best Checkpoint] Health gate: skip saving at epoch {current_epoch} "
                                            f"(fault_ok={num_ok}/{len(fault_items)}, need>={min_fault_classes}, th={min_recall_th:.4f})"
                                        )
                                        allow_save = False
                                else:
                                    if min_recall < float(min_recall_th) - float(gate_eps):
                                        print(f"[Best Checkpoint] Health gate: skip saving at epoch {current_epoch} "
                                              f"({metric_name}={min_recall:.4f} < {min_recall_th:.4f}, min_class={min_k})")
                                        allow_save = False
                        except Exception as e:
                            print(f"[Best Checkpoint] Health gate warning: {e} (will proceed without gating)")

                    if allow_save:
                        _prev_best_ckpt = best_checkpoint_path
                        best_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            _best_checkpoint_filename(current_epoch, checkpoint_tag)
                        )
                        if _prev_best_ckpt is not None and os.path.exists(_prev_best_ckpt) and _prev_best_ckpt != best_checkpoint_path:
                            try:
                                os.remove(_prev_best_ckpt)
                            except Exception:
                                pass

                        # store training-time prototypes into best checkpoint
                        ckpt_text_prototypes = text_prototypes
                        if use_bayesian_prototypes:
                            # if still None (edge case), fall back to prior
                            ckpt_text_prototypes = text_prototypes if text_prototypes is not None else initial_text_prototypes
                        else:
                            ckpt_text_prototypes = text_prototypes

                        save_checkpoint(
                            best_checkpoint_path,
                            signal_encoder,
                            spec_encoder,
                            text_encoder,
                            fusion_module,
                            fusion_decoder,
                            optimizer,
                            current_epoch,
                            best_metric_value,
                            text_prototypes=ckpt_text_prototypes,
                            ema_manager=ema_manager if use_ema else None,
                            proto_adapter=proto_adapter,
                            learnable_proto_module=learnable_proto_module,
                            decouple_spec=decouple_spec,
                            decouple_signal=decouple_signal,
                            condition_classifier=condition_classifier,
                        )

                        print(f"[Best Checkpoint] Saved best checkpoint: {best_checkpoint_path}")
                        print(f"[Best Checkpoint] Best {early_stop_metric}: {best_metric_value:.4f} at epoch {current_epoch}")

                except Exception as e:
                    print(f"[Warning] Failed to save best checkpoint: {e}")
                    try:
                        if best_checkpoint_path and os.path.isfile(best_checkpoint_path):
                            from utils.checkpoint import is_checkpoint_readable
                            if not is_checkpoint_readable(best_checkpoint_path):
                                os.remove(best_checkpoint_path)
                                print(
                                    f"[Checkpoint] Removed unreadable checkpoint after failed save: "
                                    f"{best_checkpoint_path}"
                                )
                    except Exception:
                        pass
            
            # Only allow early stopping after min_epochs (ensures major structures are enabled)
            if current_epoch >= early_stop_min_epochs:
                if no_improve_count >= early_stop_patience:
                    print(f"[Early Stop] Stopping training: {early_stop_patience} consecutive evaluations without improvement.")
                    print(f"[Early Stop] Best checkpoint: Epoch {best_metric_epoch} with {early_stop_metric} {best_metric_value:.4f}")
                    break  # 退出训练循环
            # Before min_epochs, no_improve_count is not incremented (handled above in the metric comparison)
        
        # ---- 保存 checkpoint（默认每 10 轮）----
        _ckpt_interval = int(getattr(cfg, "CHECKPOINT_SAVE_INTERVAL", 10))
        if _ckpt_interval > 0 and current_epoch % _ckpt_interval == 0:
            try:
                # store training-time prototypes into checkpoint for later evaluation
                ckpt_text_prototypes = text_prototypes
                if use_bayesian_prototypes:
                    # if still None (edge case), fall back to prior
                    ckpt_text_prototypes = text_prototypes if text_prototypes is not None else initial_text_prototypes
                else:
                    ckpt_text_prototypes = text_prototypes

                save_checkpoint(
                    os.path.join(checkpoint_dir, f"epoch_{current_epoch}_{checkpoint_tag}.pth"),
                    signal_encoder,
                    spec_encoder,
                    text_encoder,
                    fusion_module,
                    fusion_decoder,
                    optimizer,
                    current_epoch,
                    avg_total,
                    text_prototypes=ckpt_text_prototypes,
                    ema_manager=ema_manager if use_ema else None,
                    proto_adapter=proto_adapter,
                    learnable_proto_module=learnable_proto_module,
                    decouple_spec=decouple_spec,
                    decouple_signal=decouple_signal,
                    condition_classifier=condition_classifier,
                )
            except Exception as e:
                print(f"[Checkpoint] Failed to save: {e}")

    # ===== 训练结束：写回 best checkpoint 信息并返回 history =====
    # 说明：
    # - 之前函数 docstring 声称会 return history，但实际没有 return，导致上层无法获知 early-stop 的 best epoch。
    # - 这里把 best epoch / metric / best checkpoint path 写到 history，供 main.py 评估阶段选择正确 checkpoint。
    best_checkpoint_path = None
    if use_early_stopping and best_metric_epoch is not None and best_metric_epoch != -1:
        # 优先 phase-safe best；若 torch.save 曾失败会留下截断 .pth，此处用可读性检测并回退 pre_best。
        best_cand = os.path.join(checkpoint_dir, _best_checkpoint_filename(best_metric_epoch, checkpoint_tag))
        best_cand_legacy = os.path.join(checkpoint_dir, _best_checkpoint_filename(best_metric_epoch, None))
        primary = best_cand if os.path.exists(best_cand) else (
            best_cand_legacy if os.path.exists(best_cand_legacy) else best_cand
        )
        best_checkpoint_path = resolve_readable_checkpoint(
            primary,
            checkpoint_dir=checkpoint_dir,
            checkpoint_tag=checkpoint_tag,
            best_epoch=int(best_metric_epoch),
            extra_candidates=[
                best_pre_window_ckpt_path,
                os.path.join(checkpoint_dir, f"epoch_{best_metric_epoch}_{checkpoint_tag}.pth"),
            ],
        )

    # ===== Attempt 72: SWAD-like checkpoint averaging around best epoch =====
    swad_checkpoint_path = None
    swad_used_epochs = []
    try:
        use_swad = bool(getattr(cfg, "USE_SWAD", False))
        swad_window = int(getattr(cfg, "SWAD_WINDOW", 2))
        swad_min_models = int(getattr(cfg, "SWAD_MIN_MODELS", 3))
    except Exception:
        use_swad = False
        swad_window = 2
        swad_min_models = 3

    if use_swad and best_checkpoint_path is not None and best_metric_epoch is not None and int(best_metric_epoch) != -1:
        try:
            swad_checkpoint_path, swad_used_epochs = build_swad_checkpoint(
                checkpoint_dir=checkpoint_dir,
                best_epoch=int(best_metric_epoch),
                checkpoint_tag=checkpoint_tag,
                swad_window=swad_window,
                swad_min_models=swad_min_models
            )
            if swad_checkpoint_path is not None and os.path.exists(swad_checkpoint_path):
                print(
                    f"[SWAD] Built averaged checkpoint: {swad_checkpoint_path} "
                    f"(epochs={swad_used_epochs})"
                )
            else:
                print(
                    f"[SWAD] Skipped: insufficient checkpoints in window "
                    f"(window={swad_window}, min_models={swad_min_models}, found={len(swad_used_epochs)})."
                )
        except Exception as e:
            print(f"[SWAD] Warning: failed to build averaged checkpoint: {e}")

    history["best_metric"] = early_stop_metric if use_early_stopping else None
    history["best_metric_value"] = best_metric_value
    history["best_metric_epoch"] = best_metric_epoch
    history["best_checkpoint_path"] = best_checkpoint_path
    history["swad_checkpoint_path"] = swad_checkpoint_path
    history["swad_used_epochs"] = swad_used_epochs

    # Attempt 40: record HUST-best checkpoint info for downstream evaluation scripts.
    history["hust_best_fusion_decoder_acc"] = float(hust_best_fusion_decoder_acc) if hust_best_fusion_decoder_acc is not None else None
    history["hust_best_epoch"] = int(hust_best_epoch) if hust_best_epoch is not None else None
    history["hust_best_checkpoint_path"] = hust_best_checkpoint_path

    return history
