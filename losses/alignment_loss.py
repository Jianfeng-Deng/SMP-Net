import torch
import torch.nn.functional as F

from losses.consistency_loss import fusion_consistency_loss


# --------------------------------------------------
# Focal Loss for imbalanced classification
# --------------------------------------------------

# --------------------------------------------------
# LDAM Loss (Label-Distribution-Aware Margin Loss)
# NeurIPS 2019: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
# --------------------------------------------------

class LDAMLoss(torch.nn.Module):
    """
    LDAM Loss with class-dependent margins based on sample counts.
    
    The margin for class j is: margin_j = C * n_j^(-1/4)
    where C is a hyperparameter (max_margin) and n_j is the sample count for class j.
    
    During forward pass, the true-class logit is reduced by its margin before softmax,
    effectively creating a larger decision boundary for minority classes.
    
    Args:
        class_counts: list or tensor of sample counts per class [C]
        max_margin: float, the C parameter controlling maximum margin
        scale: float, the s parameter for logit scaling
        weight: tensor [C] or None, optional class weights for CE
    """
    
    def __init__(self, class_counts, max_margin=0.5, scale=30.0, weight=None, normalize_margins=True):
        super().__init__()
        # Compute class-dependent margins: margin_j = C * n_j^(-1/4)
        counts = torch.tensor(class_counts, dtype=torch.float32)
        margins = max_margin * (counts ** (-0.25))
        # Optional normalization: keeps the largest margin at max_margin.
        # Disable to preserve relative margin gaps for extreme minority classes.
        if normalize_margins:
            margins = margins / margins.max() * max_margin
        self.register_buffer('margins', margins)
        self.scale = scale
        self.weight = weight
        self.normalize_margins = bool(normalize_margins)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] logits before softmax
            targets: [B] target class indices
        
        Returns:
            loss: scalar LDAM loss
        """
        device = logits.device
        margins = self.margins.to(device)
        
        # Get margins for each sample based on their true class
        # margin_i = margins[targets[i]]
        batch_margins = margins[targets]  # [B]
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()  # [B, C]
        
        # Subtract margin from true-class logit: logit_y -= margin_y
        # For other classes, logit stays the same
        adjusted_logits = logits - batch_margins.unsqueeze(1) * targets_one_hot
        
        # Scale logits
        adjusted_logits = adjusted_logits * self.scale
        
        # Compute cross-entropy
        if self.weight is not None:
            weight = self.weight.to(device)
            loss = F.cross_entropy(adjusted_logits, targets, weight=weight)
        else:
            loss = F.cross_entropy(adjusted_logits, targets)
        
        return loss


# --------------------------------------------------
# Class Prototype Separation Loss
# Enforces minimum distance between class centroids in feature space
# --------------------------------------------------

class PrototypeSeparationLoss(torch.nn.Module):
    """
    Maintains EMA class centroids and penalizes when any two centroids are too close.
    
    Loss = sum_{i<j} max(0, delta - ||mu_i - mu_j||_2)
    
    where delta is the desired minimum distance between class centroids.
    
    Args:
        num_classes: int, number of classes
        feature_dim: int, dimension of feature vectors
        delta: float, minimum desired distance between class centroids
        momentum: float, EMA momentum for centroid updates
    """
    
    def __init__(self, num_classes, feature_dim, delta=0.5, momentum=0.9, use_log_ratio=False, log_eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.delta = delta
        self.momentum = momentum
        self.use_log_ratio = bool(use_log_ratio)
        self.log_eps = float(log_eps)
        
        # Initialize centroids with zeros (will be properly initialized on first batch)
        self.register_buffer('centroids', torch.zeros(num_classes, feature_dim))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))
    
    def update_centroids(self, features, labels):
        """
        Update EMA centroids based on current batch.
        
        Args:
            features: [B, D] feature vectors (should be normalized or from consistent space)
            labels: [B] class labels
        """
        device = features.device
        centroids = self.centroids.to(device)
        initialized = self.initialized.to(device)
        
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features = features[mask]
                    batch_centroid = class_features.mean(dim=0)
                    
                    if not initialized[c]:
                        # First time seeing this class: initialize directly
                        centroids[c] = batch_centroid
                        initialized[c] = True
                    else:
                        # EMA update
                        centroids[c] = self.momentum * centroids[c] + (1 - self.momentum) * batch_centroid
            
            # Copy back
            self.centroids = centroids
            self.initialized = initialized
    
    def forward(self, features=None, labels=None):
        """
        Compute separation loss based on current centroids.
        
        If features and labels are provided, updates centroids first.
        
        Returns:
            loss: scalar separation loss
            distances: dict of pairwise distances for logging
        """
        if features is not None and labels is not None:
            self.update_centroids(features, labels)
        
        device = self.centroids.device
        
        # Compute pairwise distances between all class centroids
        loss = torch.tensor(0.0, device=device)
        distances = {}
        num_pairs = 0
        
        # Only compute loss for classes that have been initialized
        initialized_mask = self.initialized
        
        for i in range(self.num_classes):
            if not initialized_mask[i]:
                continue
            for j in range(i + 1, self.num_classes):
                if not initialized_mask[j]:
                    continue
                
                # L2 distance
                dist = torch.norm(self.centroids[i] - self.centroids[j], p=2)
                distances[f"dist_{i}_{j}"] = float(dist.item())
                
                if self.use_log_ratio:
                    # Continuous separation loss: -log(||mu_i - mu_j|| / delta)
                    # This keeps a non-zero gradient even beyond delta.
                    denom = max(float(self.delta), self.log_eps)
                    ratio = (dist + float(self.log_eps)) / float(denom)
                    pair_loss = -torch.log(ratio)
                else:
                    # Hinge loss: penalize if distance < delta
                    pair_loss = F.relu(self.delta - dist)
                loss = loss + pair_loss
                num_pairs += 1
        
        # Normalize by number of pairs
        if num_pairs > 0:
            loss = loss / num_pairs
        
        return loss, distances
    
    def get_centroid_distances(self):
        """Return current pairwise centroid distances for logging."""
        distances = {}
        for i in range(self.num_classes):
            if not self.initialized[i]:
                continue
            for j in range(i + 1, self.num_classes):
                if not self.initialized[j]:
                    continue
                dist = torch.norm(self.centroids[i] - self.centroids[j], p=2)
                distances[f"dist_{i}_{j}"] = float(dist.item())
        return distances


def focal_loss(logits, targets, alpha=None, gamma=2.0, class_weights=None, label_smoothing=0.0):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        logits: [B, C] logits before softmax
        targets: [B] target class indices
        alpha: [C] or float, class weights for focal loss (None means use class_weights or uniform)
        gamma: float, focusing parameter (default 2.0)
        class_weights: [C] or None, class weights for reweighting (used if alpha is None)
        label_smoothing: float, label smoothing coefficient (0.0 means no smoothing)
    
    Returns:
        loss: scalar focal loss
    """
    # Compute cross entropy with label smoothing if needed
    if label_smoothing > 0.0:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        ce_loss = -(true_dist * log_probs).sum(dim=1)
    else:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Compute p_t (probability of true class)
    probs = F.softmax(logits, dim=1)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1.0 - p_t) ** gamma
    
    # Apply alpha weights if provided
    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_t = torch.ones_like(p_t) * alpha
        else:
            # alpha is a tensor [C]
            alpha_t = alpha[targets].to(logits.device)
        focal_weight = alpha_t * focal_weight
    elif class_weights is not None:
        # Use class_weights as alpha
        alpha_t = class_weights[targets].to(logits.device)
        focal_weight = alpha_t * focal_weight
    
    # Compute focal loss
    focal_loss_val = focal_weight * ce_loss
    
    return focal_loss_val.mean()


# --------------------------------------------------
# Sample-level semantic alignment
# --------------------------------------------------

def sample_semantic_ce_loss(z_x, labels, text_prototypes, temperature=0.07, class_weights=None, 
                           use_focal_loss=False, focal_gamma=2.0, focal_alpha=None, 
                           label_smoothing=0.0):
    """
    z_x: [B, D]
    labels: [B] with labels matching text_prototypes keys (e.g., 0..C-1)
    text_prototypes: dict {label: tensor[D]}
    class_weights: torch.tensor [C] or None
    use_focal_loss: bool, whether to use focal loss instead of standard CE
    focal_gamma: float, focusing parameter for focal loss
    focal_alpha: [C] or float or None, alpha parameter for focal loss
    label_smoothing: float, label smoothing coefficient
    """
    z_x = F.normalize(z_x, dim=-1)
    # build proto matrix sorted by label keys
    keys = sorted(text_prototypes.keys())
    proto_mat = torch.stack([text_prototypes[k] for k in keys], dim=0).to(z_x.device)  # [C, D]
    proto_mat = F.normalize(proto_mat, dim=-1)

    logits = z_x @ proto_mat.T / temperature  # [B, C]

    # map labels to indices (in case labels are non-contiguous)
    label2idx = {k: i for i, k in enumerate(keys)}
    idx_labels = torch.tensor([label2idx[int(l.item())] for l in labels], device=z_x.device)

    if use_focal_loss:
        loss = focal_loss(logits, idx_labels, alpha=focal_alpha, gamma=focal_gamma, 
                         class_weights=class_weights, label_smoothing=label_smoothing)
    else:
        # Standard cross entropy with optional class weights and label smoothing
        if label_smoothing > 0.0:
            loss = F.cross_entropy(logits, idx_labels, 
                                  weight=class_weights.to(z_x.device) if class_weights is not None else None,
                                  label_smoothing=label_smoothing)
        else:
            if class_weights is not None:
                loss = F.cross_entropy(logits, idx_labels, weight=class_weights.to(z_x.device))
            else:
                loss = F.cross_entropy(logits, idx_labels)
    return loss, logits



# --------------------------------------------------
# Weak inter-class margin regularization
# --------------------------------------------------
def semantic_margin_regularizer(
    z_x,
    labels,
    text_prototypes,
    margin=0.2
):
    z_x = F.normalize(z_x, dim=-1)

    all_labels = sorted(text_prototypes.keys())
    proto_mat = torch.stack(
        [text_prototypes[l] for l in all_labels],
        dim=0
    ).to(z_x.device)

    proto_mat = F.normalize(proto_mat, dim=-1)
    sim = z_x @ proto_mat.T

    pos_idx = torch.tensor(
        [all_labels.index(l.item()) for l in labels],
        device=z_x.device
    )

    pos_sim = sim[torch.arange(sim.size(0)), pos_idx]

    # 修复FP16溢出问题：使用更小的值，并在FP32下执行masked_fill
    # FP16的范围大约是[-65504, 65504]，-1e9会溢出，改用-1e4
    mask_value = -1e4 if sim.dtype == torch.float16 else -1e9
    
    # 确保masked_fill在合适的精度下执行
    one_hot_mask = F.one_hot(pos_idx, sim.size(1)).bool()
    neg_sim, _ = torch.max(
        sim.masked_fill(one_hot_mask, mask_value),
        dim=1
    )

    return F.relu(margin + neg_sim - pos_sim).mean()


def fusion_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    *,
    allowed_mask: torch.Tensor | None = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha=None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    use_balanced_softmax: bool = False,
    class_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy (or focal) loss for 4-way diagnosis logits, with optional per-sample allowed-class masking.

    This implements A2 ("partial-label / masked CE") for heterogeneous label spaces:
      - We do NOT hard-mask logits inside the model.
      - We mask/renormalize at the loss level, and ignore samples whose true label is disallowed.

    Args:
      logits: [B,C]
      labels: [B]
      class_weights: [C] or None
      allowed_mask: [B,C] bool mask, True means class is allowed for that sample.
    """
    if logits.ndim != 2:
        raise ValueError("fusion_ce_loss expects logits [B,C]")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("fusion_ce_loss expects labels [B]")

    device = logits.device
    B, C = logits.size(0), logits.size(1)
    labels = labels.to(device)

    # Build sample-valid mask: ignore out-of-range labels and disallowed true labels
    valid = (labels >= 0) & (labels < C)
    if allowed_mask is not None:
        if allowed_mask.shape != (B, C):
            raise ValueError(f"allowed_mask must be [B,C]={B,C}, got {tuple(allowed_mask.shape)}")
        allowed_mask = allowed_mask.to(device=device)
        idx = torch.arange(B, device=device)
        valid = valid & allowed_mask[idx, labels.clamp(min=0, max=C - 1)]

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=logits.dtype)

    # Filter to valid samples
    logits_v = logits[valid]
    labels_v = labels[valid]
    w = class_weights.to(device) if class_weights is not None else None
    allowed_v = allowed_mask[valid] if allowed_mask is not None else None

    # Balanced Softmax (logit adjustment by class frequency)
    if use_balanced_softmax and class_counts is not None:
        try:
            counts = class_counts.to(device=device, dtype=logits_v.dtype).view(1, -1)
            if counts.size(1) == logits_v.size(1):
                log_prior = torch.log(counts.clamp(min=1.0))
                logits_v = logits_v + log_prior
                # Avoid double compensation with class weights
                w = None
        except Exception:
            pass

    # Mask disallowed classes by a large negative constant (safe in fp16)
    if allowed_v is not None:
        mask_value = -1e4 if logits_v.dtype in (torch.float16, torch.bfloat16) else -1e9
        logits_v = logits_v.masked_fill(~allowed_v, mask_value)

    # Label smoothing with masking: smooth only over allowed classes.
    if label_smoothing and float(label_smoothing) > 0.0:
        ls = float(label_smoothing)
        logp = F.log_softmax(logits_v, dim=1)
        with torch.no_grad():
            if allowed_v is not None:
                allowed_cnt = allowed_v.float().sum(dim=1).clamp(min=1.0)  # [N]
                # Distribute smoothing mass over allowed classes excluding the true label.
                true = F.one_hot(labels_v, num_classes=C).to(dtype=logp.dtype)
                # allowed without true label
                allowed_wo_true = (allowed_v.float() - (allowed_v.float() * true)).clamp(min=0.0)
                denom = allowed_wo_true.sum(dim=1, keepdim=True).clamp(min=1.0)
                smooth_dist = allowed_wo_true / denom
                target = (1.0 - ls) * true + ls * smooth_dist
            else:
                true = F.one_hot(labels_v, num_classes=C).to(dtype=logp.dtype)
                smooth = torch.full_like(true, fill_value=ls / float(max(C - 1, 1)))
                target = smooth.scatter(1, labels_v.unsqueeze(1), 1.0 - ls)
        ce = -(target * logp).sum(dim=1)
    else:
        ce = F.cross_entropy(logits_v, labels_v, reduction="none", weight=w)

    if use_focal_loss:
        # Focal weight from masked softmax probs
        probs = F.softmax(logits_v, dim=1)
        pt = probs.gather(1, labels_v.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_w = (1.0 - pt) ** float(focal_gamma)

        alpha_t = None
        if focal_alpha is not None:
            if isinstance(focal_alpha, (list, tuple, torch.Tensor)):
                alpha_tensor = torch.tensor(focal_alpha, device=device, dtype=ce.dtype)
                alpha_t = alpha_tensor[labels_v]
            else:
                alpha_t = float(focal_alpha)
        elif w is not None:
            alpha_t = w[labels_v].to(dtype=ce.dtype)
        if alpha_t is not None and torch.is_tensor(alpha_t):
            focal_w = focal_w * alpha_t
        elif alpha_t is not None:
            focal_w = focal_w * float(alpha_t)
        return (focal_w * ce).mean()

    return ce.mean()


# --------------------------------------------------
# (3a) NF contrastive / margin for Normal vs Fault on z_gate
# --------------------------------------------------

def nf_contrastive_loss(z_gate: torch.Tensor, labels_nf: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised contrastive on z_gate: pull same class (normal/fault) together, push different apart.
    labels_nf: [B] 0=normal, 1=fault.
    """
    z_gate = F.normalize(z_gate.float(), dim=-1)
    B = z_gate.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z_gate.device, dtype=z_gate.dtype)
    sim = (z_gate @ z_gate.T) / max(temperature, 1e-6)  # [B,B]
    labels_nf = labels_nf.long().view(-1, 1)
    mask_pos = (labels_nf == labels_nf.T).float()
    mask_pos = mask_pos * (1.0 - torch.eye(B, device=z_gate.device))
    n_pos = mask_pos.sum(dim=1).clamp(min=1e-6)
    exp_sim = torch.exp(sim.clamp(max=50.0))
    log_sum_exp_neg = torch.log((exp_sim * (1.0 - mask_pos)).sum(dim=1).clamp(min=1e-6))
    loss_pos = -((sim * mask_pos).sum(dim=1) / n_pos - log_sum_exp_neg)
    return loss_pos.mean()


def nf_margin_loss(
    logits_nf_global: torch.Tensor,
    labels_nf: torch.Tensor,
    margin: float = 0.25,
) -> torch.Tensor:
    """
    Margin on NF logits: for normal (0) want logit_normal > logit_fault + margin;
    for fault (1) want logit_fault > logit_normal + margin.
    logits_nf_global: [B, 2] (normal, fault).
    """
    if logits_nf_global is None or logits_nf_global.dim() != 2 or logits_nf_global.size(1) != 2:
        return torch.tensor(0.0, device=logits_nf_global.device if logits_nf_global is not None else torch.device("cpu"))
    logit_n = logits_nf_global[:, 0]
    logit_f = logits_nf_global[:, 1]
    diff = logit_f - logit_n  # positive => fault preferred
    # normal (0): want diff <= -margin  =>  loss = max(0, diff + margin)
    # fault (1):  want diff >= +margin  =>  loss = max(0, -diff + margin)
    labels_nf = labels_nf.long().to(logits_nf_global.device)
    loss = torch.where(
        labels_nf == 0,
        F.relu(diff + margin),
        F.relu(-diff + margin),
    )
    return loss.mean()


def normal_reference_loss(
    z_gate: torch.Tensor,
    labels_nf: torch.Tensor,
    normal_ref: torch.Tensor,
    weight_fault_push: float = 1.0,
) -> torch.Tensor:
    """
    (3b) Pull normal samples to normal_ref; push fault samples away from normal_ref.
    z_gate: [B, D], normal_ref: [D]. Cosine similarity in [−1,1].
    """
    z_gate = F.normalize(z_gate.float(), dim=-1)
    normal_ref = F.normalize(normal_ref.float().to(z_gate.device).view(1, -1), dim=-1)
    cos = (z_gate @ normal_ref.T).squeeze(1)  # [B]
    labels_nf = labels_nf.long().to(z_gate.device)
    # normal (0): maximize cos => loss = 1 - cos
    # fault (1):  minimize cos => loss = 1 + cos (or max(0, cos - margin))
    loss = torch.where(
        labels_nf == 0,
        1.0 - cos,
        weight_fault_push * (1.0 + cos),
    )
    return loss.mean()


def loss_fusion_cls(fusion_logits, labels, weights, use_focal_loss=False, focal_gamma=2.0, 
                    focal_alpha=None, label_smoothing=0.0, hierarchical_parts=None, 
                    adaptive_pos_weight_global=None, adaptive_pos_weight_multi=None,
                    confusion_penalty_weights=None, use_balanced_softmax=False, class_counts=None):
    """
    Fusion decoder classification loss
    
    Args:
        fusion_logits: [B, C] logits from fusion decoder (4-class, for backward compatibility)
        labels: [B] target labels (0=normal, 1/2/3=fault)
        weights: [C] class weights
        use_focal_loss: bool, whether to use focal loss
        focal_gamma: float, focusing parameter
        focal_alpha: [C] or float or None, alpha parameter for focal loss
        label_smoothing: float, label smoothing coefficient
        hierarchical_parts: dict with keys "logits_nf" [B,2] and "logits_fault" [B,3] if hierarchical head
        adaptive_pos_weight_global: scalar or None, adaptive pos_weight for global nf BCE (overrides default)
        adaptive_pos_weight_multi: [3] tensor or None, adaptive pos_weight for per-type nf BCE (overrides default)
    
    Returns:
        loss: scalar loss
    """
    # If hierarchical parts provided, use two-stage training.
    #
    # IMPORTANT: Hybrid nf training (global 2-way + per-type nf_k)
    # - global nf: normal vs any-fault (稳定“放行故障”，默认AUC_nf口径)
    # - per-type nf_k: normal vs fault_k (给类3独立边界；必须配pos_weight做正负平衡)
    if hierarchical_parts is not None:
        # ===== (Stage) Optional override: force flat 4-way CE/Focal even if hierarchical parts exist =====
        # Used by two-stage training to "strip prototype diagnosis from the main trunk" while still
        # allowing auxiliary losses to access intermediate logits (e.g., outer margin logits).
        try:
            if bool(hierarchical_parts.get("force_flat_ce", False)):
                allowed_mask = None
                try:
                    allowed_mask = hierarchical_parts.get("allowed_diag_mask", None)
                except Exception:
                    allowed_mask = None
                return fusion_ce_loss(
                    fusion_logits,
                    labels,
                    weights,
                    allowed_mask=allowed_mask,
                    use_focal_loss=use_focal_loss,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                    label_smoothing=label_smoothing,
                    use_balanced_softmax=use_balanced_softmax,
                    class_counts=class_counts
                )
        except Exception:
            pass

        logits_fault = hierarchical_parts.get("logits_fault", None)              # [B,3]
        logits_nf_multi = hierarchical_parts.get("logits_nf_multi", None)        # [B,3]
        logits_nf_global = hierarchical_parts.get("logits_nf_global", None)      # [B,2]
        nf_pos_weight_multi = hierarchical_parts.get("nf_pos_weight_multi", None)  # [3] (neg/pos)
        nf_pos_weight_global = hierarchical_parts.get("nf_pos_weight_global", None)  # [1] (neg/pos)
        domain_ids = hierarchical_parts.get("domain_ids", None)                  # [B] or None
        current_epoch = hierarchical_parts.get("current_epoch", None)
        fault_type_weights = hierarchical_parts.get("fault_type_weights", None)  # [3] or None (inner/outer/rolling)

        # Loss weights / nf_multi gate (configurable)
        try:
            import config as cfg
            w_nf_global = float(getattr(cfg, "FUSIONDEC_W_NF_GLOBAL", 1.1))
            w_nf_multi_cfg = float(getattr(cfg, "FUSIONDEC_W_NF_MULTI", 1.0))
            w_fault = float(getattr(cfg, "FUSIONDEC_W_FAULT_TYPE", 2.0))
            use_nf_multi_loss = bool(getattr(cfg, "USE_NF_MULTI_LOSS", True))
            nf_multi_start = int(getattr(cfg, "NF_MULTI_START_EPOCH", 0))
            pw_multi_max = float(getattr(cfg, "NF_MULTI_POS_WEIGHT_MAX", 0.0))
        except Exception:
            w_nf_global = 1.1
            w_nf_multi_cfg = 1.0
            w_fault = 2.0
            use_nf_multi_loss = True
            nf_multi_start = 0
            pw_multi_max = 0.0

        # If disabled or before start epoch: effectively remove nf_multi from training
        enable_nf_multi = bool(use_nf_multi_loss)
        if current_epoch is not None and int(current_epoch) < int(nf_multi_start):
            enable_nf_multi = False

        if logits_nf_multi is None or logits_fault is None or logits_nf_global is None:
            # Backward compatibility: fall back to old keys if present
            logits_nf = hierarchical_parts.get("logits_nf", None)            # [B,2]
            if logits_nf is None:
                raise KeyError("hierarchical_parts missing nf logits (expected logits_nf_multi or logits_nf).")

            labels_nf = (labels > 0).long()
            labels_fault = labels - 1
            fault_mask = (labels > 0)

            if label_smoothing > 0.0:
                loss_nf = F.cross_entropy(logits_nf, labels_nf, label_smoothing=label_smoothing)
            else:
                loss_nf = F.cross_entropy(logits_nf, labels_nf)

            if fault_mask.sum() > 0 and logits_fault is not None:
                loss_fault = F.cross_entropy(logits_fault[fault_mask], labels_fault[fault_mask])
            else:
                loss_fault = torch.tensor(0.0, device=logits_nf.device)

            return 1.2 * loss_nf + 0.8 * loss_fault

        # ===== Stage A-a: global nf (normal vs any-fault) =====
        # Use calibrated anomaly_logit with BCEWithLogits + pos_weight (neg/pos):
        # This directly targets the "gate collapse to normal" failure mode you observed on HUST.
        labels_nf = (labels > 0).float()  # [B] 0/1
        anomaly_logit_global = logits_nf_global[:, 1] - logits_nf_global[:, 0]  # [B]
        # Clamp logits to prevent BCE explosion dominating training
        try:
            import config as cfg
            logit_clamp = float(getattr(cfg, "NF_LOGIT_CLAMP", 0.0))
        except Exception:
            logit_clamp = 0.0
        if logit_clamp and logit_clamp > 0:
            anomaly_logit_global = torch.clamp(anomaly_logit_global, min=-float(logit_clamp), max=float(logit_clamp))
        
        # Use adaptive pos_weight if provided (from adaptive class weights), otherwise fall back to decoder default
        if adaptive_pos_weight_global is not None:
            pwg = adaptive_pos_weight_global.to(device=anomaly_logit_global.device, dtype=anomaly_logit_global.dtype)
            loss_nf_global = F.binary_cross_entropy_with_logits(anomaly_logit_global, labels_nf, pos_weight=pwg)
        elif nf_pos_weight_global is not None:
            pwg = nf_pos_weight_global.to(device=anomaly_logit_global.device, dtype=anomaly_logit_global.dtype)
            loss_nf_global = F.binary_cross_entropy_with_logits(anomaly_logit_global, labels_nf, pos_weight=pwg)
        else:
            loss_nf_global = F.binary_cross_entropy_with_logits(anomaly_logit_global, labels_nf)

        device = labels.device

        # ===== Stage A-b: per-type nf_k (multi-label BCE with pos_weight) =====
        # targets: normal -> [0,0,0]; fault k -> one-hot at (k-1)
        # IMPORTANT (stability): this term is optional and can be disabled when Rolling is extremely rare.
        if enable_nf_multi:
            B = labels.size(0)
            targets_nf = torch.zeros(B, 3, device=device, dtype=logits_nf_multi.dtype)
            fault_mask = (labels > 0)
            if fault_mask.any():
                idx = (labels[fault_mask] - 1).clamp(min=0, max=2)
                targets_nf[fault_mask, idx] = 1.0

            # Use adaptive pos_weight if provided (from adaptive class weights), otherwise fall back to decoder default
            if adaptive_pos_weight_multi is not None:
                pw = adaptive_pos_weight_multi.to(device=device, dtype=logits_nf_multi.dtype)
            elif nf_pos_weight_multi is not None:
                pw = nf_pos_weight_multi.to(device=device, dtype=logits_nf_multi.dtype)
            else:
                pw = None
            if pw is not None and pw_multi_max and pw_multi_max > 0:
                pw = torch.clamp(pw, max=float(pw_multi_max))

            logits_nf_multi_use = logits_nf_multi
            if logit_clamp and logit_clamp > 0:
                logits_nf_multi_use = torch.clamp(logits_nf_multi, min=-float(logit_clamp), max=float(logit_clamp))

            if pw is None:
                loss_nf_multi = F.binary_cross_entropy_with_logits(logits_nf_multi_use, targets_nf)
            else:
                loss_nf_multi = F.binary_cross_entropy_with_logits(logits_nf_multi_use, targets_nf, pos_weight=pw)
        else:
            loss_nf_multi = torch.tensor(0.0, device=labels.device, dtype=logits_nf_global.dtype)
            fault_mask = (labels > 0)

        # ===== Stage B: fault type among {1,2,3} =====
        labels_fault = labels - 1  # {0,1,2} for {inner,outer,rolling}
        if fault_mask.sum() > 0:
            # ===== (Structural) Split-head fault-type loss =====
            use_split = False
            logits_io_all = hierarchical_parts.get("logits_fault_io", None)          # [B,2] or None
            logits_roll_raw_all = hierarchical_parts.get("logits_fault_roll_raw", None)  # [B,2] (non_roll, roll) preferred; legacy may be [B,1]
            try:
                import config as cfg
                use_split = bool(getattr(cfg, "USE_SPLIT_FAULT_HEADS", True))
            except Exception:
                use_split = False
            use_split = bool(use_split and (logits_io_all is not None) and (logits_roll_raw_all is not None))

            if use_split:
                try:
                    import config as cfg
                    w_io = float(getattr(cfg, "SPLIT_FAULT_W_IO", 1.0))
                    w_roll = float(getattr(cfg, "SPLIT_FAULT_W_ROLL", 1.0))
                    # Legacy config keys (kept for backward compatibility)
                    pw_roll_max = float(getattr(cfg, "SPLIT_FAULT_ROLL_POS_WEIGHT_MAX", 20.0))
                except Exception:
                    w_io, w_roll, pw_roll_max = 1.0, 1.0, 20.0

                # (1) Rolling vs Non-rolling (conditional on fault only)
                # We explicitly avoid using Normal samples here, otherwise the rolling head becomes a second NF head.
                try:
                    roll_fault_only = bool(getattr(cfg, "ROLLING_TRAIN_ON_FAULT_ONLY", True))
                except Exception:
                    roll_fault_only = True
                roll_mask = fault_mask.clone() if roll_fault_only else torch.ones_like(labels, dtype=torch.bool)
                roll_logits_all = logits_roll_raw_all
                loss_roll = torch.tensor(0.0, device=device, dtype=logits_nf_multi.dtype)
                if roll_mask.any() and roll_logits_all is not None:
                    if roll_logits_all.dim() == 2 and roll_logits_all.size(1) == 2:
                        # 2-logit softmax (preferred): target 1 if rolling else 0
                        roll_logits = roll_logits_all[roll_mask].to(device=device, dtype=logits_nf_multi.dtype)  # [N,2]
                        roll_targets = (labels[roll_mask] == 3).long()  # [N] in {0,1}
                        # Optional class weight: up-weight rolling within faults
                        w_roll_ce = None
                        try:
                            if fault_type_weights is not None and isinstance(fault_type_weights, torch.Tensor) and fault_type_weights.numel() >= 3:
                                wr = float(fault_type_weights[2].detach().float().cpu().item())
                                wr = max(1.0, float(wr))
                                if pw_roll_max and pw_roll_max > 0:
                                    wr = min(float(wr), float(pw_roll_max))
                                w_roll_ce = torch.tensor([1.0, wr], device=device, dtype=logits_nf_multi.dtype)
                        except Exception:
                            w_roll_ce = None
                        try:
                            use_roll_focal = bool(getattr(cfg, "USE_ROLLING_FOCAL_LOSS", False))
                            roll_gamma = float(getattr(cfg, "ROLLING_FOCAL_GAMMA", 2.0))
                            roll_alpha = float(getattr(cfg, "ROLLING_FOCAL_ALPHA", 0.75))
                        except Exception:
                            use_roll_focal, roll_gamma, roll_alpha = False, 2.0, 0.75
                        if use_roll_focal:
                            roll_logit_diff = roll_logits[:, 1] - roll_logits[:, 0]
                            roll_prob = torch.sigmoid(roll_logit_diff)
                            p_t = torch.where(roll_targets == 1, roll_prob, 1.0 - roll_prob)
                            alpha_t = torch.where(
                                roll_targets == 1,
                                torch.full_like(p_t, roll_alpha),
                                torch.full_like(p_t, 1.0 - roll_alpha),
                            )
                            loss_roll = -alpha_t * torch.pow(1.0 - p_t, roll_gamma) * torch.log(p_t + 1e-8)
                            loss_roll = loss_roll.mean()
                        elif use_focal_loss:
                            loss_roll = focal_loss(
                                roll_logits,
                                roll_targets,
                                alpha=focal_alpha,
                                gamma=focal_gamma,
                                class_weights=w_roll_ce,
                                label_smoothing=label_smoothing
                            )
                        else:
                            if label_smoothing > 0.0:
                                loss_roll = F.cross_entropy(roll_logits, roll_targets, weight=w_roll_ce, label_smoothing=label_smoothing)
                            else:
                                loss_roll = F.cross_entropy(roll_logits, roll_targets, weight=w_roll_ce)
                    else:
                        # Legacy 1-logit path: treat as BCE, still fault-only
                        roll_logit = roll_logits_all
                        if roll_logit.ndim == 2 and roll_logit.size(1) == 1:
                            roll_logit = roll_logit[:, 0]
                        roll_logit = roll_logit.to(device=device, dtype=logits_nf_multi.dtype)
                        roll_targets = (labels == 3).to(device=device, dtype=logits_nf_multi.dtype)
                        roll_logit = roll_logit[roll_mask]
                        roll_targets = roll_targets[roll_mask]
                        loss_roll = F.binary_cross_entropy_with_logits(roll_logit, roll_targets)

                # (2) Inner vs Outer (2-way), only on {1,2}
                io_mask = (labels == 1) | (labels == 2)
                if io_mask.any():
                    logits_io = logits_io_all[io_mask].to(device=device, dtype=logits_nf_multi.dtype)  # [N,2]
                    targets_io = (labels[io_mask] - 1).clamp(0, 1).long()
                    w_io_ce = None
                    if fault_type_weights is not None and isinstance(fault_type_weights, torch.Tensor) and fault_type_weights.numel() >= 2:
                        try:
                            w_io_ce = fault_type_weights[:2].to(device=device, dtype=logits_nf_multi.dtype)
                        except Exception:
                            w_io_ce = None
                    if use_focal_loss:
                        loss_io = focal_loss(
                            logits_io,
                            targets_io,
                            alpha=focal_alpha,
                            gamma=focal_gamma,
                            class_weights=w_io_ce,
                            label_smoothing=label_smoothing
                        )
                    else:
                        if label_smoothing > 0.0:
                            loss_io = F.cross_entropy(logits_io, targets_io, weight=w_io_ce, label_smoothing=label_smoothing)
                        else:
                            loss_io = F.cross_entropy(logits_io, targets_io, weight=w_io_ce)
                else:
                    loss_io = torch.tensor(0.0, device=device, dtype=logits_nf_multi.dtype)

                loss_fault = float(w_io) * loss_io + float(w_roll) * loss_roll
            else:
                # ===== Legacy 3-way fault-type loss (domain-aware 2-way on PU/HIT) =====
                # Optional: AM-Softmax (additive margin) on fault-type logits
                use_am = False
                am_m = 0.20
                am_s = 1.0
                am_start_epoch = 0
                am_inner_outer_only = False
                am_warmup_epochs = 0
                try:
                    import config as cfg
                    use_am = bool(getattr(cfg, "USE_AM_SOFTMAX_FAULT", False))
                    am_m = float(getattr(cfg, "AM_SOFTMAX_M", am_m))
                    am_s = float(getattr(cfg, "AM_SOFTMAX_S", am_s))
                    am_start_epoch = int(getattr(cfg, "AM_SOFTMAX_START_EPOCH", am_start_epoch))
                    am_inner_outer_only = bool(getattr(cfg, "AM_SOFTMAX_INNER_OUTER_ONLY", am_inner_outer_only))
                    am_warmup_epochs = int(getattr(cfg, "AM_SOFTMAX_WARMUP_EPOCHS", am_warmup_epochs))
                except Exception:
                    pass

                logits_fault_train = logits_fault[fault_mask]
                targets_fault_train = labels_fault[fault_mask]

                # Determine AM-Softmax effective margin (if enabled)
                enable_am = False
                eff_m = float(am_m)
                if use_am:
                    cur_epoch = hierarchical_parts.get("current_epoch", None)
                    enable_am = True
                    if cur_epoch is not None and am_start_epoch > 0 and int(cur_epoch) < int(am_start_epoch):
                        enable_am = False
                    if enable_am and cur_epoch is not None and am_warmup_epochs and int(am_warmup_epochs) > 0:
                        t = (int(cur_epoch) - int(am_start_epoch) + 1) / float(int(am_warmup_epochs))
                        t = max(0.0, min(1.0, t))
                        eff_m = float(am_m) * float(t)

                def _apply_am_softmax(logits_x, targets_x):
                    if not (enable_am and logits_x is not None and targets_x is not None):
                        return logits_x
                    lx = logits_x.clone()
                    idx = torch.arange(lx.size(0), device=lx.device)
                    tgt = targets_x.clamp(0, lx.size(1) - 1)
                    lx[idx, tgt] -= float(eff_m)
                    lx = lx * float(am_s)
                    return lx

                def _loss_with_optional_penalty(logits_x, targets_x, ce_weight=None):
                    if logits_x.size(0) == 0:
                        return torch.tensor(0.0, device=device, dtype=logits_nf_multi.dtype)
                    if confusion_penalty_weights is not None and len(confusion_penalty_weights) >= 3:
                        individual_losses = F.cross_entropy(
                            logits_x,
                            targets_x,
                            reduction='none',
                            weight=ce_weight,
                            label_smoothing=label_smoothing if label_smoothing > 0.0 else 0.0
                        )
                        penalty_weights = torch.ones_like(individual_losses)
                        for ii, lab in enumerate(targets_x):
                            lv = int(lab.item())
                            if 0 <= lv < 3:
                                penalty_weights[ii] = confusion_penalty_weights[lv]
                        return (individual_losses * penalty_weights).mean()
                    if use_focal_loss:
                        return focal_loss(
                            logits_x,
                            targets_x,
                            alpha=focal_alpha,
                            gamma=focal_gamma,
                            class_weights=ce_weight,
                            label_smoothing=label_smoothing
                        )
                    if label_smoothing > 0.0:
                        return F.cross_entropy(logits_x, targets_x, weight=ce_weight, label_smoothing=label_smoothing)
                    return F.cross_entropy(logits_x, targets_x, weight=ce_weight)

                if domain_ids is not None:
                    try:
                        import config as cfg
                        hit_id = int(getattr(cfg, "HIT_DOMAIN_ID", 4))
                        pu_id = int(getattr(cfg, "PU_DOMAIN_ID", 2))
                    except Exception:
                        hit_id = 3
                        pu_id = 1
                    dom_fault = domain_ids[fault_mask]
                    no_roll = (dom_fault == hit_id) | (dom_fault == pu_id)
                    has_roll = ~no_roll

                    # Safety: if a sample is labeled Rolling (target==2) but domain does not support it, skip it.
                    bad = (targets_fault_train == 2) & no_roll
                    if bad.any():
                        keep = ~bad
                        logits_fault_train = logits_fault_train[keep]
                        targets_fault_train = targets_fault_train[keep]
                        dom_fault = dom_fault[keep]
                        no_roll = (dom_fault == hit_id) | (dom_fault == pu_id)
                        has_roll = ~no_roll

                    logits_nr = logits_fault_train[no_roll, :2]
                    targets_nr = targets_fault_train[no_roll].clamp(0, 1)
                    logits_r = logits_fault_train[has_roll, :3]
                    targets_r = targets_fault_train[has_roll].clamp(0, 2)

                    logits_nr = _apply_am_softmax(logits_nr, targets_nr)
                    logits_r = _apply_am_softmax(logits_r, targets_r)

                    w_nr = None
                    w_r = None
                    if fault_type_weights is not None and isinstance(fault_type_weights, torch.Tensor):
                        try:
                            fw = fault_type_weights.to(device=device, dtype=logits_nf_multi.dtype)
                            if fw.numel() >= 2:
                                w_nr = fw[:2]
                            if fw.numel() >= 3:
                                w_r = fw[:3]
                        except Exception:
                            w_nr, w_r = None, None

                    loss_nr = _loss_with_optional_penalty(logits_nr, targets_nr, ce_weight=w_nr)
                    loss_r = _loss_with_optional_penalty(logits_r, targets_r, ce_weight=w_r)

                    n_nr = int(logits_nr.size(0))
                    n_r = int(logits_r.size(0))
                    n_tot = max(1, n_nr + n_r)
                    loss_fault = (loss_nr * float(n_nr) + loss_r * float(n_r)) / float(n_tot)
                else:
                    # Backward compatibility: single 3-way fault-type loss
                    if logits_fault_train.size(0) == 0:
                        loss_fault = torch.tensor(0.0, device=device, dtype=logits_nf_multi.dtype)
                    else:
                        logits_fault_train = _apply_am_softmax(
                            logits_fault_train,
                            targets_fault_train.clamp(0, logits_fault_train.size(1) - 1)
                        )
                        w_all = None
                        if fault_type_weights is not None and isinstance(fault_type_weights, torch.Tensor):
                            try:
                                w_all = fault_type_weights.to(device=device, dtype=logits_nf_multi.dtype)[:logits_fault_train.size(1)]
                            except Exception:
                                w_all = None
                        loss_fault = _loss_with_optional_penalty(
                            logits_fault_train,
                            targets_fault_train.clamp(0, logits_fault_train.size(1) - 1),
                            ce_weight=w_all
                        )
        else:
            loss_fault = torch.tensor(0.0, device=device, dtype=logits_nf_multi.dtype)

        # ===== Stage C (Optional): Residual Proxy Supervision =====
        # If residual proxies are enabled, add supervision for fault-discriminative residual features
        loss_residual = torch.tensor(0.0, device=device, dtype=logits_nf_multi.dtype)
        logits_residual = hierarchical_parts.get("logits_residual", None)
        residual_mask = hierarchical_parts.get("residual_mask", None)
        
        if logits_residual is not None:
            # Fault samples: maximize agreement with true fault type (masked CE)
            if fault_mask.sum() > 0:
                # logits_residual: [B, 3] (inner/outer/rolling), but HIT only has 2 faults
                # residual_mask: [B, 3] bool mask (True=available, False=unavailable for this domain)
                logits_residual_fault = logits_residual[fault_mask]  # [N_fault, 3]
                labels_fault_idx = labels_fault[fault_mask]  # [N_fault] in {0, 1, 2}
                
                if residual_mask is not None:
                    mask_fault = residual_mask[fault_mask]  # [N_fault, 3]
                    # Apply mask: use a large FINITE negative value (avoid -inf triggering non-finite/AMP issues)
                    logits_residual_fault_masked = logits_residual_fault.clone()
                    mask_value = -1e4
                    logits_residual_fault_masked[~mask_fault] = mask_value
                    loss_residual_fault = F.cross_entropy(
                        logits_residual_fault_masked,
                        labels_fault_idx,
                        reduction='mean'
                    )
                else:
                    # No mask: standard CE
                    loss_residual_fault = F.cross_entropy(
                        logits_residual_fault,
                        labels_fault_idx,
                        reduction='mean'
                    )
                
                loss_residual = loss_residual + loss_residual_fault
            
            # Normal samples: suppress all residual evidence (push logits toward negative)
            if (~fault_mask).sum() > 0:
                logits_residual_normal = logits_residual[~fault_mask]  # [N_normal, 3]
                # Use max-pooled logit as the "strongest fault evidence" and suppress it
                max_residual_logit = torch.max(logits_residual_normal, dim=1)[0]  # [N_normal]
                # BCE loss: treat max_residual_logit as "fault probability logit", target=0
                loss_residual_normal = F.binary_cross_entropy_with_logits(
                    max_residual_logit,
                    torch.zeros_like(max_residual_logit),
                    reduction='mean'
                )
                loss_residual = loss_residual + 0.5 * loss_residual_normal  # weight 0.5 for normal suppression
        
        # Combine: global gating + per-type boundary + fault type + residual
        # 调整策略：
        # - 1.1 * loss_nf_global: 略微调低 nf_global 压力，减少对嵌入空间的过度扭曲
        # - 1.0 * loss_nf_multi: 保持 per-type 边界压力
        # - 2.0 * loss_fault: 大幅提升具体故障子类压力，解决 Class 2/3 → Class 1 混淆
        # - 0.2 * loss_residual: residual proxy 分支（辅助，不应过强）
        import config as cfg
        base_w_residual = getattr(cfg, 'RESIDUAL_LOSS_WEIGHT', 0.2)
        warmup_epochs = getattr(cfg, 'RESIDUAL_WARMUP_EPOCHS', 0)
        ramp_epochs = getattr(cfg, 'RESIDUAL_RAMP_EPOCHS', 0)
        current_epoch = hierarchical_parts.get("current_epoch", None)

        # Linear ramp: 0 -> base_w_residual after warmup
        if current_epoch is None:
            w_residual = base_w_residual
        else:
            if current_epoch <= warmup_epochs:
                w_residual = 0.0
            elif ramp_epochs <= 0:
                w_residual = base_w_residual
            else:
                t = (current_epoch - warmup_epochs) / float(ramp_epochs)
                t = max(0.0, min(1.0, t))
                w_residual = base_w_residual * t

        return float(w_nf_global) * loss_nf_global + float(w_nf_multi_cfg) * loss_nf_multi + float(w_fault) * loss_fault + w_residual * loss_residual
    
    # Original single-stage loss (backward compatibility)
    if use_focal_loss:
        loss = focal_loss(fusion_logits, labels, alpha=focal_alpha, gamma=focal_gamma,
                         class_weights=weights, label_smoothing=label_smoothing)
    else:
        if label_smoothing > 0.0:
            loss = F.cross_entropy(fusion_logits, labels, 
                                  weight=weights.to(fusion_logits.device) if weights is not None else None,
                                  label_smoothing=label_smoothing)
        else:
            loss = F.cross_entropy(fusion_logits, labels, 
                                  weight=weights.to(fusion_logits.device) if weights is not None else None)
    return loss


def fusion_batch_soft_silhouette_loss(features, labels, eps=1e-6):
    """
    Batch soft silhouette on L2-normalized features using cosine distance 1 - cos.

    For each sample i: a_i = mean distance to same-class others (0 if alone in batch);
    b_i = min over present classes c != y_i of (mean distance from i to samples in c).
    Silhouette s_i = (b_i - a_i) / max(a_i, b_i, eps). Loss = mean(1 - s_i) over valid rows.

    ATT485 (S1): class-aggregated geometry signal; less batch-pair starved than SupCon at small B.
    """
    if features is None or labels is None or not torch.is_tensor(features):
        return None
    if features.dim() != 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    B = features.size(0)
    if B < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    device = features.device
    dtype = features.dtype
    y = labels.long().view(-1)
    if y.numel() != B:
        return torch.tensor(0.0, device=device, dtype=dtype)
    present = torch.unique(y)
    if present.numel() < 2:
        return torch.tensor(0.0, device=device, dtype=dtype)
    feats = F.normalize(features, dim=-1)
    dist = 1.0 - torch.matmul(feats, feats.t())
    dist = dist.clamp(min=0.0, max=2.0)
    eye = torch.eye(B, device=device, dtype=torch.bool)
    M_same = (y.unsqueeze(1) == y.unsqueeze(0)) & (~eye)
    same_cnt = M_same.float().sum(dim=1)
    a = torch.zeros(B, device=device, dtype=dtype)
    mask_intra = same_cnt > 0
    a[mask_intra] = (dist * M_same.float()).sum(dim=1)[mask_intra] / same_cnt[mask_intra].clamp_min(1.0)

    n_pres = int(present.numel())
    D_pres = torch.zeros(B, n_pres, device=device, dtype=dtype)
    for j in range(n_pres):
        m = y == present[j]
        if m.sum() == 0:
            continue
        D_pres[:, j] = dist[:, m].mean(dim=1)

    mask_other = present.view(1, -1) != y.view(-1, 1)
    large = torch.as_tensor(1.0e4, device=device, dtype=dtype)
    D_eff = D_pres.masked_fill(~mask_other, large)
    b = D_eff.min(dim=1).values

    denom = torch.maximum(a, b).clamp_min(eps)
    s = (b - a) / denom
    loss_vec = 1.0 - s
    return loss_vec.mean()


def fusion_metric_learnable_proto_ce_loss(
    z_metric,
    learnable_prototypes,
    labels,
    temperature,
    class_weights=None,
):
    """
    ATT486 (S1): auxiliary softmax CE on scaled cosine logits between z_metric and
    row-wise L2-normalized learnable prototypes (typically `LearnablePrototypes.forward()`).

    Gradients flow to z_metric (decoder/fusion path) and to learnable prototype parameters.
    This is independent of `USE_PROTO_CLASSIFIER` (fixed-text proto CE on the same head slot).
    """
    if not torch.is_tensor(z_metric) or not torch.is_tensor(learnable_prototypes):
        return None
    if z_metric.dim() != 2 or learnable_prototypes.dim() != 2:
        return None
    if labels is None or not torch.is_tensor(labels):
        return None
    C = int(learnable_prototypes.size(0))
    if C < 2:
        return None
    y = labels.long().view(-1)
    if y.numel() != z_metric.size(0):
        return None
    if int(y.max().item()) >= C or int(y.min().item()) < 0:
        return None
    zm = F.normalize(z_metric.float(), dim=-1, eps=1e-6)
    P = F.normalize(learnable_prototypes.float(), dim=-1, eps=1e-6)
    if P.size(-1) != zm.size(-1):
        return None
    t = max(float(temperature), 1e-8)
    logits = torch.matmul(zm, P.t()) / t
    cw = None
    if class_weights is not None and torch.is_tensor(class_weights):
        if class_weights.numel() == C:
            cw = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, y, weight=cw)


def cross_domain_metric_centroid_loss(
    z_metric,
    domain_ids,
    labels,
    sep_margin=0.10,
    min_samples_per_centroid=2,
):
    """
    Cross-domain centroid objective on metric branch features.
    - Align same-class centroids across domains.
    - Separate different-class centroids across domains.
    """
    if (
        (not torch.is_tensor(z_metric))
        or (not torch.is_tensor(domain_ids))
        or (not torch.is_tensor(labels))
        or z_metric.dim() != 2
    ):
        return None
    if z_metric.size(0) < 2:
        return None
    d = domain_ids.long().view(-1)
    y = labels.long().view(-1)
    if d.numel() != z_metric.size(0) or y.numel() != z_metric.size(0):
        return None

    z = F.normalize(z_metric.float(), dim=-1, eps=1e-6)
    unique_domains = torch.unique(d)
    unique_classes = torch.unique(y)
    if unique_domains.numel() < 2 or unique_classes.numel() < 2:
        return None

    centroids = {}
    for dom in unique_domains.tolist():
        for cls in unique_classes.tolist():
            mask = (d == int(dom)) & (y == int(cls))
            if int(mask.sum().item()) >= int(min_samples_per_centroid):
                centroids[(int(dom), int(cls))] = z[mask].mean(dim=0)
    if len(centroids) < 2:
        return None

    align_losses = []
    sep_losses = []
    for cls in unique_classes.tolist():
        cls_centroids = [
            (int(dom), centroids[(int(dom), int(cls))])
            for dom in unique_domains.tolist()
            if (int(dom), int(cls)) in centroids
        ]
        for i in range(len(cls_centroids)):
            for j in range(i + 1, len(cls_centroids)):
                ci = cls_centroids[i][1]
                cj = cls_centroids[j][1]
                align_losses.append(
                    1.0 - F.cosine_similarity(ci.unsqueeze(0), cj.unsqueeze(0), dim=-1).mean()
                )
        for _, c_i in cls_centroids:
            for cls_other in unique_classes.tolist():
                if int(cls_other) == int(cls):
                    continue
                for dom_j in unique_domains.tolist():
                    key_other = (int(dom_j), int(cls_other))
                    if key_other not in centroids:
                        continue
                    c_j = centroids[key_other]
                    cos_ij = F.cosine_similarity(c_i.unsqueeze(0), c_j.unsqueeze(0), dim=-1).mean()
                    sep_losses.append(F.relu(cos_ij - float(sep_margin)))

    if len(align_losses) == 0 and len(sep_losses) == 0:
        return None
    total = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    if len(align_losses) > 0:
        total = total + torch.stack(align_losses).mean()
    if len(sep_losses) > 0:
        total = total + torch.stack(sep_losses).mean()
    return total


# --------------------------------------------------
# Final multimodal loss with fusion
# --------------------------------------------------
def multimodal_alignment_loss(
    z_signal,
    z_spec,
    z_fusion,
    fusion_logits,
    labels,
    text_prototypes,
    hierarchical_parts=None,
    z_fusion_raw=None,
    temperature=0.07,
    w_fusion=1.5,  # 提升语义对齐权重，恢复 HUST_AUC
    w_signal=0.3,
    w_spec=0.3,
    w_margin=0.2,  # 提升 margin 正则项系数，配合 W_MARGIN=0.3 使用
    w_fusion_cls=0.5,
    class_weights=None,
    z_fusion_aug=None,  # 增强版本的fusion embedding（用于一致性正则）
    augment_indices=None,  # 增强样本的索引
    w_consistency=0.0,   # 一致性损失权重
    consistency_temperature=0.07,  # 一致性损失温度参数
    # Focal Loss and Label Smoothing parameters
    use_focal_loss=False,  # 是否使用Focal Loss
    focal_gamma=2.0,       # Focal Loss的gamma参数
    focal_alpha=None,      # Focal Loss的alpha参数（None表示使用class_weights）
    label_smoothing=0.0,   # Label Smoothing系数
    # Adaptive pos_weight for hierarchical BCE loss
    adaptive_pos_weight_global=None,  # 自适应global nf pos_weight (scalar)
    adaptive_pos_weight_multi=None,   # 自适应per-type nf pos_weight [3]
    confusion_penalty_weights=None,   # Confusion-aware 惩罚权重 [3]，分别对应 Class 1/2/3
    use_balanced_softmax=False,
    class_counts=None,
    # ===== (A) SupCon + Prototype separation =====
    use_supcon_loss=False,
    supcon_weight=0.0,
    supcon_temperature=0.07,
    supcon_fault_only=True,
    use_proto_sep_loss=False,
    proto_sep_weight=0.0
    ,
    # ATT479: SupCon on fusion pre-L2 (z_fusion_supcon); decoder path may use detach(z_fusion).
    use_fusion_supcon_loss=False,
    fusion_supcon_weight=0.0,
    fusion_supcon_temperature=0.07,
    fusion_supcon_fault_only=False,
    z_fusion_supcon=None,
    # ATT485: "supcon" (pairwise SupCon) vs "silhouette" (batch soft silhouette on same tensor path).
    fusion_geometry_loss_mode="silhouette",
    fusion_silhouette_eps=1e-6,
    ablation_gate_bank=None,
    # ===== (B) Outer margin (anti-collapse for Outer class) =====
    use_outer_margin_loss=True,
    outer_margin_weight=0.05,
    outer_margin=0.2,
    outer_margin_start_epoch=5,
    # ===== (B2) Inner margin (anti-collapse for Inner class) =====
    use_inner_margin_loss=True,
    inner_margin_weight=0.05,
    inner_margin=0.2,
    inner_margin_start_epoch=5,
    decoupled_stage2_margin_override=False
):
    """
    核心思想：
    - fusion 是主语义空间
    - 单模态仅作正则
    
    Args:
        use_focal_loss: 是否使用Focal Loss替代标准交叉熵
        focal_gamma: Focal Loss的聚焦参数，越大越关注难样本
        focal_alpha: Focal Loss的类别权重，None表示使用class_weights
        label_smoothing: 标签平滑系数，减少过拟合
        decoupled_stage2_margin_override: ATT379c - when True, bypass _core_loss_only
            margin disabling to selectively re-enable margins during Stage 2.
    """

    # ATT378b: read CORE_LOSS_ONLY_MODE once; used to force-disable all non-core margins
    # that are controlled by independent USE_* flags inside this function.
    try:
        import config as _cfg_check
        _core_loss_only = bool(getattr(_cfg_check, "CORE_LOSS_ONLY_MODE", False))
    except Exception:
        _core_loss_only = False
    if decoupled_stage2_margin_override:
        _core_loss_only = False

    loss_fusion, logit_fusion = sample_semantic_ce_loss(
        z_fusion, labels, text_prototypes, temperature, class_weights,
        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        label_smoothing=label_smoothing
    )

    loss_signal, _ = sample_semantic_ce_loss(
        z_signal, labels, text_prototypes, temperature, class_weights,
        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        label_smoothing=label_smoothing
    )

    loss_spec, _ = sample_semantic_ce_loss(
        z_spec, labels, text_prototypes, temperature, class_weights,
        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        label_smoothing=label_smoothing
    )

    loss_margin = semantic_margin_regularizer(
        z_fusion, labels, text_prototypes
    )

    # Prefer explicitly passed hierarchical_parts (stable under AMP/autocast).
    # Fallback to legacy tensor attribute for backward compatibility.
    if not isinstance(hierarchical_parts, dict):
        hierarchical_parts = None
        if hasattr(fusion_logits, 'hierarchical_parts'):
            try:
                _hp = fusion_logits.hierarchical_parts
                if isinstance(_hp, dict):
                    hierarchical_parts = _hp
            except Exception:
                hierarchical_parts = None

    # ===== (Attempt 64) Semantic anchor against Normal absorber =====
    # For fault samples, enforce:
    #   sim(z, proto_fault) >= sim(z, proto_normal) + margin
    # This directly counters the observed collapse "fault -> normal".
    loss_sem_fault_vs_normal = torch.tensor(0.0, device=z_fusion.device)
    sem_fvn_weight = 0.0
    sem_fvn_margin = 0.15
    sem_fvn_start_epoch = 1
    use_sem_fvn = False
    try:
        import config as cfg
        use_sem_fvn = bool(getattr(cfg, "USE_SEMANTIC_FAULT_VS_NORMAL_ANCHOR", False))
        sem_fvn_weight = float(getattr(cfg, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_WEIGHT", 0.0))
        sem_fvn_margin = float(getattr(cfg, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_MARGIN", 0.15))
        sem_fvn_start_epoch = int(getattr(cfg, "SEMANTIC_FAULT_VS_NORMAL_ANCHOR_START_EPOCH", 1))
    except Exception:
        use_sem_fvn = False
        sem_fvn_weight = 0.0

    if use_sem_fvn and sem_fvn_weight > 0 and isinstance(text_prototypes, dict) and (0 in text_prototypes):
        cur_epoch = 0
        try:
            if isinstance(hierarchical_parts, dict):
                cur_epoch = int(hierarchical_parts.get("current_epoch", 0) or 0)
        except Exception:
            cur_epoch = 0
        if cur_epoch >= int(sem_fvn_start_epoch):
            z_n = F.normalize(z_fusion, dim=-1, eps=1e-6)
            p_normal = F.normalize(text_prototypes[0].to(z_n.device), dim=-1, eps=1e-6)
            per_class_losses = []
            for c in (1, 2, 3):
                if c not in text_prototypes:
                    continue
                m_c = (labels == c)
                if not m_c.any():
                    continue
                p_c = F.normalize(text_prototypes[c].to(z_n.device), dim=-1, eps=1e-6)
                z_c = z_n[m_c]
                sim_pos = torch.matmul(z_c, p_c)
                sim_neg_n = torch.matmul(z_c, p_normal)
                per_class_losses.append(F.relu(float(sem_fvn_margin) - (sim_pos - sim_neg_n)).mean())
            if len(per_class_losses) > 0:
                loss_sem_fault_vs_normal = torch.stack(per_class_losses).mean()

    # ===== (C0) Align raw fusion embedding to text prototypes =====
    # 关键动机：当前 z_fusion 往往先经语义适配再优化，导致原始融合空间在评估的 FusionProto 路径上失配。
    loss_fusion_proto_align = torch.tensor(0.0, device=z_fusion.device)
    use_fusion_proto_align = False
    fusion_proto_align_weight = 0.0
    fusion_proto_align_temp = 0.07
    fusion_proto_align_start_epoch = 0
    fusion_proto_min_samples = 2
    fusion_proto_use_class_weights = True
    try:
        import config as cfg
        use_fusion_proto_align = bool(getattr(cfg, "USE_FUSION_PROTO_ALIGN_LOSS", False))
        fusion_proto_align_weight = float(getattr(cfg, "FUSION_PROTO_ALIGN_WEIGHT", 0.0))
        fusion_proto_align_temp = float(getattr(cfg, "FUSION_PROTO_ALIGN_TEMPERATURE", 0.07))
        fusion_proto_align_start_epoch = int(getattr(cfg, "FUSION_PROTO_ALIGN_START_EPOCH", 0))
        fusion_proto_min_samples = int(getattr(cfg, "FUSION_PROTO_ALIGN_MIN_SAMPLES_PER_CLASS", 2))
        fusion_proto_use_class_weights = bool(getattr(cfg, "FUSION_PROTO_ALIGN_USE_CLASS_WEIGHTS", True))
    except Exception:
        use_fusion_proto_align = False
        fusion_proto_align_weight = 0.0

    if z_fusion_raw is None:
        z_fusion_raw = z_fusion

    if use_fusion_proto_align and fusion_proto_align_weight > 0 and isinstance(text_prototypes, dict):
        cur_epoch = 0
        try:
            if isinstance(hierarchical_parts, dict):
                cur_epoch = int(hierarchical_parts.get("current_epoch", 0) or 0)
        except Exception:
            cur_epoch = 0
        if cur_epoch >= int(fusion_proto_align_start_epoch):
            valid_classes = []
            proto_rows = []
            class_keep_mask = torch.zeros_like(labels, dtype=torch.bool)
            for c in sorted(text_prototypes.keys()):
                if int(c) < 0:
                    continue
                p = text_prototypes[c]
                if not torch.is_tensor(p) or p.dim() != 1:
                    continue
                cls_mask = (labels == int(c))
                if int(cls_mask.sum().item()) < int(fusion_proto_min_samples):
                    continue
                valid_classes.append(int(c))
                proto_rows.append(F.normalize(p.to(z_fusion_raw.device), dim=-1, eps=1e-6))
                class_keep_mask = class_keep_mask | cls_mask

            if len(valid_classes) >= 2 and class_keep_mask.any():
                z_raw = F.normalize(z_fusion_raw[class_keep_mask], dim=-1, eps=1e-6).float()
                proto_mat = torch.stack(proto_rows, dim=0).float()  # [C,D]
                logits_raw = torch.matmul(z_raw, proto_mat.t()) / float(max(1e-6, fusion_proto_align_temp))
                cls_to_local = {c: i for i, c in enumerate(valid_classes)}
                targets_global = labels[class_keep_mask].long()
                targets_local = torch.tensor(
                    [cls_to_local[int(t.item())] for t in targets_global],
                    device=logits_raw.device,
                    dtype=torch.long,
                )
                ce_weight = None
                if fusion_proto_use_class_weights and class_weights is not None and torch.is_tensor(class_weights):
                    try:
                        selected = [float(class_weights[int(c)].item()) for c in valid_classes]
                        ce_weight = torch.tensor(selected, device=logits_raw.device, dtype=logits_raw.dtype)
                    except Exception:
                        ce_weight = None
                loss_fusion_proto_align = F.cross_entropy(logits_raw, targets_local, weight=ce_weight)

    # ===== (C0b) Normal↔Rolling inter-class overlap constraint on z_spec =====
    loss_nr_inter_cos_margin = torch.tensor(0.0, device=z_fusion.device)
    use_nr_inter_cos_margin = False
    nr_inter_cos_weight = 0.0
    nr_inter_cos_target_max = 0.15
    nr_inter_cos_start_epoch = 0
    nr_inter_cos_min_samples = 2
    nr_inter_cos_hard = torch.tensor(0.0, device=z_fusion.device)
    try:
        import config as cfg
        use_nr_inter_cos_margin = bool(getattr(cfg, "USE_NR_INTER_COS_MARGIN_LOSS", False))
        nr_inter_cos_weight = float(getattr(cfg, "NR_INTER_COS_MARGIN_WEIGHT", 0.0))
        nr_inter_cos_target_max = float(getattr(cfg, "NR_INTER_COS_MARGIN_TARGET_MAX", 0.15))
        nr_inter_cos_start_epoch = int(getattr(cfg, "NR_INTER_COS_MARGIN_START_EPOCH", 0))
        nr_inter_cos_min_samples = int(getattr(cfg, "NR_INTER_COS_MIN_SAMPLES", 2))
    except Exception:
        use_nr_inter_cos_margin = False
        nr_inter_cos_weight = 0.0

    if use_nr_inter_cos_margin and nr_inter_cos_weight > 0:
        cur_epoch = 0
        try:
            if isinstance(hierarchical_parts, dict):
                cur_epoch = int(hierarchical_parts.get("current_epoch", 0) or 0)
        except Exception:
            cur_epoch = 0
        if cur_epoch >= int(nr_inter_cos_start_epoch):
            mask_n = (labels == 0)
            mask_r = (labels == 3)
            if int(mask_n.sum().item()) >= int(nr_inter_cos_min_samples) and int(mask_r.sum().item()) >= int(nr_inter_cos_min_samples):
                z_spec_n = F.normalize(z_spec[mask_n], dim=-1, eps=1e-6).float()
                z_spec_r = F.normalize(z_spec[mask_r], dim=-1, eps=1e-6).float()
                sim_nr = torch.matmul(z_spec_n, z_spec_r.t())  # [Nn, Nr]
                if sim_nr.numel() > 0:
                    k = max(1, min(4, sim_nr.size(1)))
                    nr_inter_cos_hard = sim_nr.topk(k, dim=1).values.mean()
                    loss_nr_inter_cos_margin = F.relu(nr_inter_cos_hard - float(nr_inter_cos_target_max))

    # ===== Prompt-tuned text prototype separation (Inner vs Outer) =====
    loss_text_prompt_sep = torch.tensor(0.0, device=z_fusion.device)
    text_prompt_sep_weight = 0.0
    text_prompt_sep_target_max = 0.50
    text_prompt_sep_start_epoch = 1
    text_prompt_inner_outer_cos = torch.tensor(float("nan"), device=z_fusion.device)
    use_text_prompt_sep = False
    try:
        import config as cfg
        use_text_prompt_sep = bool(getattr(cfg, "USE_TEXT_PROMPT_SEPARATION_LOSS", False))
        text_prompt_sep_weight = float(getattr(cfg, "TEXT_PROMPT_SEPARATION_WEIGHT", 0.0))
        text_prompt_sep_target_max = float(getattr(cfg, "TEXT_PROMPT_INNER_OUTER_TARGET_MAX", 0.50))
        text_prompt_sep_start_epoch = int(getattr(cfg, "TEXT_PROMPT_SEPARATION_START_EPOCH", 1))
    except Exception:
        use_text_prompt_sep = False
        text_prompt_sep_weight = 0.0

    if use_text_prompt_sep and text_prompt_sep_weight > 0 and isinstance(text_prototypes, dict):
        cur_epoch = 0
        try:
            if isinstance(hierarchical_parts, dict):
                cur_epoch = int(hierarchical_parts.get("current_epoch", 0) or 0)
        except Exception:
            cur_epoch = 0
        if cur_epoch >= int(text_prompt_sep_start_epoch) and 1 in text_prototypes and 2 in text_prototypes:
            p_inner = F.normalize(text_prototypes[1].to(z_fusion.device).float(), dim=-1, eps=1e-6)
            p_outer = F.normalize(text_prototypes[2].to(z_fusion.device).float(), dim=-1, eps=1e-6)
            text_prompt_inner_outer_cos = torch.sum(p_inner * p_outer)
            loss_text_prompt_sep = F.relu(text_prompt_inner_outer_cos - float(text_prompt_sep_target_max))

    # ===== (C-NF) NF-gate margin loss on global NF logits =====
    # Directly targets "Fault → Normal" gate errors under domain shift (dominant after nf_gate enabled).
    loss_nf_gate_margin = torch.tensor(0.0, device=z_fusion.device)
    use_nf_gate_margin = False
    nf_gate_margin_w = 0.0
    nf_gate_margin_m = 0.2
    nf_gate_margin_start = 0
    try:
        import config as cfg
        use_nf_gate_margin = bool(getattr(cfg, "USE_NF_GATE_MARGIN_LOSS", False))
        nf_gate_margin_w = float(getattr(cfg, "NF_GATE_MARGIN_WEIGHT", 0.0))
        nf_gate_margin_m = float(getattr(cfg, "NF_GATE_MARGIN", 0.2))
        nf_gate_margin_start = int(getattr(cfg, "NF_GATE_MARGIN_START_EPOCH", 0))
    except Exception:
        use_nf_gate_margin = False
        nf_gate_margin_w = 0.0
    if use_nf_gate_margin and nf_gate_margin_w > 0 and isinstance(hierarchical_parts, dict):
        cur_epoch = hierarchical_parts.get("current_epoch", 0)
        if cur_epoch is None:
            cur_epoch = 0
        if int(cur_epoch) >= int(nf_gate_margin_start):
            logits_nf_global = hierarchical_parts.get("logits_nf_global", None)
            if logits_nf_global is not None and logits_nf_global.dim() == 2 and logits_nf_global.size(1) == 2:
                anomaly_logit = (logits_nf_global[:, 1] - logits_nf_global[:, 0]).float()  # [B]
                m = float(nf_gate_margin_m)
                # normals: want anomaly_logit <= -m ; faults: want anomaly_logit >= +m
                m_norm = (labels == 0)
                m_fault = (labels != 0)
                loss_n = F.relu(anomaly_logit[m_norm] + m).mean() if m_norm.any() else torch.tensor(0.0, device=z_fusion.device)
                loss_f = F.relu(m - anomaly_logit[m_fault]).mean() if m_fault.any() else torch.tensor(0.0, device=z_fusion.device)
                loss_nf_gate_margin = loss_n + loss_f

    # ===== (Margin Router) Optionally focus margins on current absorber =====
    # The router target is set by training loop based on HUST zero-shot collapse:
    #   "all" | "vs_inner" | "vs_rolling" | "vs_normal"
    router_enabled = False
    router_target = "all"
    try:
        import config as cfg
        router_enabled = bool(getattr(cfg, "ENABLE_MARGIN_ROUTER", False))
    except Exception:
        router_enabled = False
    if router_enabled and isinstance(hierarchical_parts, dict):
        try:
            router_target = str(hierarchical_parts.get("margin_router_target", "all")).lower()
        except Exception:
            router_target = "all"
    if router_target not in {"all", "vs_inner", "vs_rolling", "vs_normal"}:
        router_target = "all"

    # ===== Supervised Contrastive (fault-only by default) =====
    def _supcon(features, y, temperature=0.07):
        # features: [N,D] normalized; y: [N] long
        if features is None or y is None:
            return torch.tensor(0.0, device=z_fusion.device)
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        feats = F.normalize(features, dim=-1)
        logits = torch.matmul(feats, feats.t()) / float(temperature)  # [N,N]
        # mask self
        self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(self_mask, -1e4 if logits.dtype == torch.float16 else -1e9)
        # positive mask
        y = y.view(-1, 1)
        pos_mask = (y == y.t()) & (~self_mask)
        # if no positives for all, return 0
        pos_counts = pos_mask.sum(dim=1)
        if (pos_counts > 0).sum() == 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_counts.clamp_min(1).float()
        loss = -mean_log_prob_pos[pos_counts > 0].mean()
        return loss
    
    try:
        import config as _cfg_proto_cls
        _use_proto_cls = bool(getattr(_cfg_proto_cls, "USE_PROTO_CLASSIFIER", False))
        _proto_cls_T = float(getattr(_cfg_proto_cls, "PROTO_CLS_TEMPERATURE", 0.10))
        _use_proto_cos_align = bool(getattr(_cfg_proto_cls, "USE_PROTO_COSINE_ALIGN", False))
        _proto_cos_align_w = float(getattr(_cfg_proto_cls, "PROTO_COSINE_ALIGN_WEIGHT", 0.10))
    except Exception:
        _use_proto_cls = False
        _proto_cls_T = 0.10
        _use_proto_cos_align = False
        _proto_cos_align_w = 0.10

    if _use_proto_cls and text_prototypes is not None and len(text_prototypes) > 0:
        _z_for_proto_cls = None
        if isinstance(hierarchical_parts, dict):
            _z_metric_candidate = hierarchical_parts.get("z_metric", None)
            if _z_metric_candidate is not None and torch.is_tensor(_z_metric_candidate):
                _z_for_proto_cls = _z_metric_candidate
        if _z_for_proto_cls is None:
            _z_for_proto_cls = z_fusion_raw if z_fusion_raw is not None else z_fusion
        loss_fusion_decoder, _logits_proto_cls = sample_semantic_ce_loss(
            _z_for_proto_cls, labels, text_prototypes, _proto_cls_T, class_weights,
            use_focal_loss=use_focal_loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            label_smoothing=label_smoothing
        )
        if isinstance(hierarchical_parts, dict):
            hierarchical_parts["logits_diag"] = _logits_proto_cls
            hierarchical_parts["logits_diag_supervised"] = _logits_proto_cls
            _ln = _logits_proto_cls[:, 0:1]
            _lf = torch.logsumexp(_logits_proto_cls[:, 1:], dim=1, keepdim=True)
            hierarchical_parts["logits_nf_global"] = torch.cat([_ln, _lf], dim=1)
            hierarchical_parts["logits_nf"] = hierarchical_parts["logits_nf_global"]
    else:
        loss_fusion_decoder = loss_fusion_cls(
            fusion_logits, labels, class_weights,
            use_focal_loss=use_focal_loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            label_smoothing=label_smoothing,
            hierarchical_parts=hierarchical_parts,
            adaptive_pos_weight_global=adaptive_pos_weight_global,
            adaptive_pos_weight_multi=adaptive_pos_weight_multi,
            confusion_penalty_weights=confusion_penalty_weights,
            use_balanced_softmax=use_balanced_softmax,
            class_counts=class_counts
        )

    # ===== (A2) ATT436-β: Cosine alignment loss (cooperative with linear head CE) =====
    loss_proto_cos_align = torch.tensor(0.0, device=z_fusion.device)
    if _use_proto_cos_align and text_prototypes is not None and len(text_prototypes) > 0:
        _z_for_align = None
        if isinstance(hierarchical_parts, dict):
            _z_cand = hierarchical_parts.get("z_metric", None)
            if _z_cand is not None and torch.is_tensor(_z_cand):
                _z_for_align = _z_cand
        if _z_for_align is None:
            _z_for_align = z_fusion

        _proto_dict = text_prototypes
        if isinstance(_proto_dict, dict):
            _unique_cls = sorted(_proto_dict.keys())
            _proto_list = [_proto_dict[c] for c in _unique_cls]
            _proto_mat = torch.stack(_proto_list, dim=0)
        elif torch.is_tensor(_proto_dict):
            _proto_mat = _proto_dict
            _unique_cls = list(range(_proto_mat.size(0)))
        else:
            _proto_mat = None

        if _proto_mat is not None and labels is not None:
            _proto_mat_n = F.normalize(_proto_mat.to(_z_for_align.device, dtype=_z_for_align.dtype), dim=-1)
            _z_align_n = F.normalize(_z_for_align, dim=-1)
            _valid = labels < _proto_mat_n.size(0)
            if _valid.any():
                _sel_z = _z_align_n[_valid]
                _sel_labels = labels[_valid]
                _sel_protos = _proto_mat_n[_sel_labels]
                _cos_sim = (_sel_z * _sel_protos).sum(dim=-1)
                loss_proto_cos_align = (1.0 - _cos_sim).mean() * _proto_cos_align_w
        if isinstance(hierarchical_parts, dict):
            hierarchical_parts["_info_proto_cos_align"] = loss_proto_cos_align.item()

    # ===== (B) Outer/Inner margin loss =====
    # For samples with true label==2 (Outer) / label==1 (Inner), enforce:
    #   logit_target >= max(logits_other) + margin
    # using a dedicated projection (logits_diag_outer) to avoid harming other classes.
    loss_outer_margin = torch.tensor(0.0, device=z_fusion.device)
    loss_inner_margin = torch.tensor(0.0, device=z_fusion.device)
    if use_outer_margin_loss and hierarchical_parts is not None:
        cur_epoch = hierarchical_parts.get("current_epoch", None)
        if cur_epoch is None or int(cur_epoch) >= int(outer_margin_start_epoch):
            # IMPORTANT: apply margins on the same logits used for diagnosis decisions.
            # Prefer `logits_diag` (decision logits). Fallback to `logits_diag_outer` only if missing.
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None:
                m_outer = (labels == 2)
                if m_outer.any():
                    lo = logits_decision[m_outer]  # [N,4]
                    # Outer class is index 2
                    logit_outer = lo[:, 2]
                    # max other over {0,1,3}
                    other = torch.stack([lo[:, 0], lo[:, 1], lo[:, 3]], dim=1).max(dim=1)[0]
                    loss_outer_margin = F.relu(float(outer_margin) - (logit_outer - other)).mean()

    if use_inner_margin_loss and hierarchical_parts is not None:
        cur_epoch = hierarchical_parts.get("current_epoch", None)
        if cur_epoch is None or int(cur_epoch) >= int(inner_margin_start_epoch):
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None:
                m_inner = (labels == 1)
                if m_inner.any():
                    li = logits_decision[m_inner]  # [N,4]
                    # Inner class is index 1
                    logit_inner = li[:, 1]
                    # max other over {0,2,3}
                    other = torch.stack([li[:, 0], li[:, 2], li[:, 3]], dim=1).max(dim=1)[0]
                    loss_inner_margin = F.relu(float(inner_margin) - (logit_inner - other)).mean()

    # ===== (B3) Normal margin loss (anti-collapse for Normal) =====
    # For true Normal samples, enforce:
    #   logit_normal >= max(logits_faults) + margin
    # This prevents "Normal absorbed into Rolling/other fault" collapse.
    loss_normal_margin = torch.tensor(0.0, device=z_fusion.device)
    try:
        import config as cfg
        use_normal_margin_loss = bool(getattr(cfg, "USE_NORMAL_MARGIN_LOSS", False))
        normal_margin_weight = float(getattr(cfg, "NORMAL_MARGIN_WEIGHT", 0.0))
        normal_margin = float(getattr(cfg, "NORMAL_MARGIN", 0.2))
        normal_margin_start_epoch = int(getattr(cfg, "NORMAL_MARGIN_START_EPOCH", 0))
    except Exception:
        use_normal_margin_loss = False
        normal_margin_weight = 0.0
        normal_margin = 0.2
        normal_margin_start_epoch = 0
    if _core_loss_only:
        use_normal_margin_loss = False
        normal_margin_weight = 0.0

    if use_normal_margin_loss and normal_margin_weight > 0 and hierarchical_parts is not None:
        cur_epoch = hierarchical_parts.get("current_epoch", None)
        if cur_epoch is None or int(cur_epoch) >= int(normal_margin_start_epoch):
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None:
                m_normal = (labels == 0)
                if m_normal.any():
                    ln = logits_decision[m_normal]  # [N,4]
                    logit_n = ln[:, 0]
                    other = ln[:, 1:].max(dim=1)[0]
                    loss_normal_margin = F.relu(float(normal_margin) - (logit_n - other)).mean()

    # ===== (B3b) Inner/Outer vs Normal margins (directly targets Inner/Outer→Normal collapse) =====
    loss_inner_vs_normal = torch.tensor(0.0, device=z_fusion.device)
    loss_outer_vs_normal = torch.tensor(0.0, device=z_fusion.device)
    loss_rolling_vs_normal = torch.tensor(0.0, device=z_fusion.device)
    try:
        import config as cfg
        use_in_vn = bool(getattr(cfg, "USE_INNER_VS_NORMAL_MARGIN_LOSS", False))
        in_vn_w = float(getattr(cfg, "INNER_VS_NORMAL_MARGIN_WEIGHT", 0.0))
        in_vn_m = float(getattr(cfg, "INNER_VS_NORMAL_MARGIN", 0.2))
        in_vn_start = int(getattr(cfg, "INNER_VS_NORMAL_MARGIN_START_EPOCH", 0))

        use_out_vn = bool(getattr(cfg, "USE_OUTER_VS_NORMAL_MARGIN_LOSS", False))
        out_vn_w = float(getattr(cfg, "OUTER_VS_NORMAL_MARGIN_WEIGHT", 0.0))
        out_vn_m = float(getattr(cfg, "OUTER_VS_NORMAL_MARGIN", 0.2))
        out_vn_start = int(getattr(cfg, "OUTER_VS_NORMAL_MARGIN_START_EPOCH", 0))

        use_roll_vn = bool(getattr(cfg, "USE_ROLLING_VS_NORMAL_MARGIN_LOSS", False))
        roll_vn_w = float(getattr(cfg, "ROLLING_VS_NORMAL_MARGIN_WEIGHT", 0.0))
        roll_vn_m = float(getattr(cfg, "ROLLING_VS_NORMAL_MARGIN", 0.2))
        roll_vn_start = int(getattr(cfg, "ROLLING_VS_NORMAL_MARGIN_START_EPOCH", 0))
    except Exception:
        use_in_vn = False
        in_vn_w = 0.0
        in_vn_m = 0.2
        in_vn_start = 0
        use_out_vn = False
        out_vn_w = 0.0
        out_vn_m = 0.2
        out_vn_start = 0
        use_roll_vn = False
        roll_vn_w = 0.0
        roll_vn_m = 0.2
        roll_vn_start = 0
    if _core_loss_only:
        use_in_vn = False; in_vn_w = 0.0
        use_out_vn = False; out_vn_w = 0.0
        use_roll_vn = False; roll_vn_w = 0.0

    if (use_in_vn and in_vn_w > 0) or (use_out_vn and out_vn_w > 0) or (use_roll_vn and roll_vn_w > 0):
        if hierarchical_parts is not None:
            cur_epoch = hierarchical_parts.get("current_epoch", None)
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None and logits_decision.dim() == 2 and logits_decision.size(1) >= 4:
                if use_in_vn and in_vn_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(in_vn_start)):
                    m_inner = (labels == 1)
                    if m_inner.any():
                        li = logits_decision[m_inner]
                        loss_inner_vs_normal = F.relu(float(in_vn_m) - (li[:, 1] - li[:, 0])).mean()
                if use_out_vn and out_vn_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(out_vn_start)):
                    m_outer = (labels == 2)
                    if m_outer.any():
                        lo = logits_decision[m_outer]
                        loss_outer_vs_normal = F.relu(float(out_vn_m) - (lo[:, 2] - lo[:, 0])).mean()
                if use_roll_vn and roll_vn_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(roll_vn_start)):
                    m_roll = (labels == 3)
                    if m_roll.any():
                        lr = logits_decision[m_roll]
                        loss_rolling_vs_normal = F.relu(float(roll_vn_m) - (lr[:, 3] - lr[:, 0])).mean()

    # ===== (B4) Anti-rolling-absorption margin (directly targets all→rolling) =====
    # For true label in {0,1,2}, enforce:
    #   logit_true >= logit_rolling + margin
    loss_rolling_absorb = torch.tensor(0.0, device=z_fusion.device)
    try:
        import config as cfg
        use_rolling_absorb = bool(getattr(cfg, "USE_ROLLING_ABSORB_MARGIN_LOSS", False))
        rolling_absorb_weight = float(getattr(cfg, "ROLLING_ABSORB_MARGIN_WEIGHT", 0.0))
        rolling_absorb_margin = float(getattr(cfg, "ROLLING_ABSORB_MARGIN", 0.2))
        rolling_absorb_start_epoch = int(getattr(cfg, "ROLLING_ABSORB_MARGIN_START_EPOCH", 0))
    except Exception:
        use_rolling_absorb = False
        rolling_absorb_weight = 0.0
        rolling_absorb_margin = 0.2
        rolling_absorb_start_epoch = 0
    if _core_loss_only:
        use_rolling_absorb = False; rolling_absorb_weight = 0.0

    if use_rolling_absorb and rolling_absorb_weight > 0 and hierarchical_parts is not None:
        cur_epoch = hierarchical_parts.get("current_epoch", None)
        if cur_epoch is None or int(cur_epoch) >= int(rolling_absorb_start_epoch):
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None and logits_decision.dim() == 2 and logits_decision.size(1) >= 4:
                # exclude rolling samples; only apply on labels 0/1/2
                m_nr = (labels >= 0) & (labels <= 2)
                if m_nr.any():
                    lnr = logits_decision[m_nr]  # [N,4]
                    y_nr = labels[m_nr].long().clamp(0, 3)
                    idx = torch.arange(lnr.size(0), device=lnr.device)
                    logit_true = lnr[idx, y_nr]
                    logit_roll = lnr[:, 3]
                    loss_rolling_absorb = F.relu(float(rolling_absorb_margin) - (logit_true - logit_roll)).mean()

    # ===== (B4b) Inner/Outer vs Rolling margins (directly targets Inner→Rolling / Outer→Rolling) =====
    # For true Inner samples, enforce: logit_inner >= logit_rolling + margin
    # For true Outer samples, enforce: logit_outer >= logit_rolling + margin
    loss_inner_vs_roll = torch.tensor(0.0, device=z_fusion.device)
    loss_outer_vs_roll = torch.tensor(0.0, device=z_fusion.device)
    try:
        import config as cfg
        use_in_vr = bool(getattr(cfg, "USE_INNER_VS_ROLLING_MARGIN_LOSS", False))
        in_vr_w = float(getattr(cfg, "INNER_VS_ROLLING_MARGIN_WEIGHT", 0.0))
        in_vr_m = float(getattr(cfg, "INNER_VS_ROLLING_MARGIN", 0.2))
        in_vr_start = int(getattr(cfg, "INNER_VS_ROLLING_MARGIN_START_EPOCH", 0))

        use_out_vr = bool(getattr(cfg, "USE_OUTER_VS_ROLLING_MARGIN_LOSS", False))
        out_vr_w = float(getattr(cfg, "OUTER_VS_ROLLING_MARGIN_WEIGHT", 0.0))
        out_vr_m = float(getattr(cfg, "OUTER_VS_ROLLING_MARGIN", 0.2))
        out_vr_start = int(getattr(cfg, "OUTER_VS_ROLLING_MARGIN_START_EPOCH", 0))
    except Exception:
        use_in_vr = False
        in_vr_w = 0.0
        in_vr_m = 0.2
        in_vr_start = 0
        use_out_vr = False
        out_vr_w = 0.0
        out_vr_m = 0.2
        out_vr_start = 0
    if _core_loss_only:
        use_in_vr = False; in_vr_w = 0.0
        use_out_vr = False; out_vr_w = 0.0

    if (use_in_vr and in_vr_w > 0) or (use_out_vr and out_vr_w > 0):
        if hierarchical_parts is not None:
            cur_epoch = hierarchical_parts.get("current_epoch", None)
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None and logits_decision.dim() == 2 and logits_decision.size(1) >= 4:
                logit_roll = logits_decision[:, 3]
                if use_in_vr and in_vr_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(in_vr_start)):
                    m_inner = (labels == 1)
                    if m_inner.any():
                        li = logits_decision[m_inner]
                        loss_inner_vs_roll = F.relu(float(in_vr_m) - (li[:, 1] - li[:, 3])).mean()
                if use_out_vr and out_vr_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(out_vr_start)):
                    m_outer = (labels == 2)
                    if m_outer.any():
                        lo = logits_decision[m_outer]
                        loss_outer_vs_roll = F.relu(float(out_vr_m) - (lo[:, 2] - lo[:, 3])).mean()

    # ===== (B4c) Anti-inner-absorption margins (Normal/Outer/Rolling vs Inner) =====
    # Directly targets collapse mode "all → Inner".
    loss_normal_vs_inner = torch.tensor(0.0, device=z_fusion.device)
    loss_outer_vs_inner = torch.tensor(0.0, device=z_fusion.device)
    loss_rolling_vs_inner = torch.tensor(0.0, device=z_fusion.device)
    try:
        import config as cfg
        use_n_vi = bool(getattr(cfg, "USE_NORMAL_VS_INNER_MARGIN_LOSS", False))
        n_vi_w = float(getattr(cfg, "NORMAL_VS_INNER_MARGIN_WEIGHT", 0.0))
        n_vi_m = float(getattr(cfg, "NORMAL_VS_INNER_MARGIN", 0.2))
        n_vi_start = int(getattr(cfg, "NORMAL_VS_INNER_MARGIN_START_EPOCH", 0))

        use_o_vi = bool(getattr(cfg, "USE_OUTER_VS_INNER_MARGIN_LOSS", False))
        o_vi_w = float(getattr(cfg, "OUTER_VS_INNER_MARGIN_WEIGHT", 0.0))
        o_vi_m = float(getattr(cfg, "OUTER_VS_INNER_MARGIN", 0.2))
        o_vi_start = int(getattr(cfg, "OUTER_VS_INNER_MARGIN_START_EPOCH", 0))

        use_r_vi = bool(getattr(cfg, "USE_ROLLING_VS_INNER_MARGIN_LOSS", False))
        r_vi_w = float(getattr(cfg, "ROLLING_VS_INNER_MARGIN_WEIGHT", 0.0))
        r_vi_m = float(getattr(cfg, "ROLLING_VS_INNER_MARGIN", 0.2))
        r_vi_start = int(getattr(cfg, "ROLLING_VS_INNER_MARGIN_START_EPOCH", 0))
    except Exception:
        use_n_vi = False
        n_vi_w = 0.0
        n_vi_m = 0.2
        n_vi_start = 0
        use_o_vi = False
        o_vi_w = 0.0
        o_vi_m = 0.2
        o_vi_start = 0
        use_r_vi = False
        r_vi_w = 0.0
        r_vi_m = 0.2
        r_vi_start = 0
    if _core_loss_only:
        use_n_vi = False; n_vi_w = 0.0
        use_o_vi = False; o_vi_w = 0.0
        use_r_vi = False; r_vi_w = 0.0

    if (use_n_vi and n_vi_w > 0) or (use_o_vi and o_vi_w > 0) or (use_r_vi and r_vi_w > 0):
        if hierarchical_parts is not None:
            cur_epoch = hierarchical_parts.get("current_epoch", None)
            logits_decision = hierarchical_parts.get("logits_diag", None)
            if logits_decision is None:
                logits_decision = hierarchical_parts.get("logits_diag_outer", None)
            if logits_decision is not None and logits_decision.dim() == 2 and logits_decision.size(1) >= 4:
                if use_n_vi and n_vi_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(n_vi_start)):
                    m_normal = (labels == 0)
                    if m_normal.any():
                        ln = logits_decision[m_normal]
                        loss_normal_vs_inner = F.relu(float(n_vi_m) - (ln[:, 0] - ln[:, 1])).mean()
                if use_o_vi and o_vi_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(o_vi_start)):
                    m_outer = (labels == 2)
                    if m_outer.any():
                        lo = logits_decision[m_outer]
                        loss_outer_vs_inner = F.relu(float(o_vi_m) - (lo[:, 2] - lo[:, 1])).mean()
                if use_r_vi and r_vi_w > 0 and (cur_epoch is None or int(cur_epoch) >= int(r_vi_start)):
                    m_roll = (labels == 3)
                    if m_roll.any():
                        lr = logits_decision[m_roll]
                        loss_rolling_vs_inner = F.relu(float(r_vi_m) - (lr[:, 3] - lr[:, 1])).mean()

    # ===== (S-SEM) Semantic distillation: prototype teacher -> learnable linear diagnosis head =====
    # Train the linear diagnosis head (student) to match semantic teacher logits (pre-linear, pre-bias).
    loss_sem_distill = torch.tensor(0.0, device=z_fusion.device)
    sem_distill_weight = 0.0
    sem_distill_temp = 2.0
    sem_distill_start = 0
    teacher_maxconf = 0.98  # if teacher is too peaky (likely collapsed), skip distillation
    use_sem_distill = False
    try:
        import config as cfg
        use_sem_distill = bool(getattr(cfg, "USE_SEMANTIC_DISTILL", False))
        sem_distill_weight = float(getattr(cfg, "SEMANTIC_DISTILL_WEIGHT", 0.0))
        sem_distill_temp = float(getattr(cfg, "SEMANTIC_DISTILL_T", 2.0))
        sem_distill_start = int(getattr(cfg, "SEMANTIC_DISTILL_START_EPOCH", 0))
        teacher_maxconf = float(getattr(cfg, "SEMANTIC_DISTILL_TEACHER_MAXCONF", 0.98))
    except Exception:
        use_sem_distill = False
        sem_distill_weight = 0.0

    if use_sem_distill and sem_distill_weight > 0 and isinstance(hierarchical_parts, dict):
        cur_epoch = hierarchical_parts.get("current_epoch", 0)
        if cur_epoch is None:
            cur_epoch = 0
        if int(cur_epoch) >= int(sem_distill_start):
            teacher = hierarchical_parts.get("logits_diag_teacher", None)
            student = hierarchical_parts.get("logits_diag_linear", None)
            if teacher is not None and student is not None and teacher.dim() == 2 and student.dim() == 2:
                t = float(sem_distill_temp)
                t = 1.0 if (t != t or t <= 0) else t
                teacher_f = torch.nan_to_num(teacher.float(), nan=0.0, posinf=1e4, neginf=-1e4)
                student_f = torch.nan_to_num(student.float(), nan=0.0, posinf=1e4, neginf=-1e4)
                # Skip distillation if teacher is collapsed (too confident everywhere).
                with torch.no_grad():
                    p_t = F.softmax(teacher_f / t, dim=1)
                    maxconf = float(p_t.max(dim=1).values.mean().detach().cpu().item())
                if maxconf <= float(teacher_maxconf):
                    # Optional: fault-only distillation (recommended when using nf_gate)
                    try:
                        import config as cfg
                        fault_only = bool(getattr(cfg, "SEMANTIC_DISTILL_FAULT_ONLY", True))
                    except Exception:
                        fault_only = True
                    if fault_only:
                        m = (labels > 0)
                        if m.any():
                            loss_sem_distill = F.kl_div(
                                F.log_softmax(student_f[m] / t, dim=1),
                                F.softmax(teacher_f[m] / t, dim=1),
                                reduction="batchmean"
                            ) * (t * t)
                    else:
                        loss_sem_distill = F.kl_div(
                            F.log_softmax(student_f / t, dim=1),
                            F.softmax(teacher_f / t, dim=1),
                            reduction="batchmean"
                        ) * (t * t)

    # SupCon on z_res (fault-only by default)
    # ATT379(S3): SupCLIP — include text prototypes as positives in SupCon.
    loss_supcon = torch.tensor(0.0, device=z_fusion.device)
    _use_supclip = False
    try:
        import config as _cfg_sc
        _use_supclip = bool(getattr(_cfg_sc, "USE_SUPCLIP", False))
    except Exception:
        pass
    if use_supcon_loss and hierarchical_parts is not None and "z_res" in hierarchical_parts:
        feats = hierarchical_parts["z_res"]
        y = labels

        # SupCLIP: append text prototypes to features so they act as anchors
        _text_protos_for_supclip = None
        if _use_supclip and text_prototypes is not None:
            try:
                _tp_keys = sorted(text_prototypes.keys())
                _tp_list = [F.normalize(text_prototypes[k].float(), dim=-1) for k in _tp_keys]
                _text_protos_for_supclip = torch.stack(_tp_list, dim=0).to(feats.device)  # [C, D]
            except Exception:
                _text_protos_for_supclip = None

        if supcon_fault_only:
            m = (y > 0)
            if m.sum() >= 2:
                feats_f = F.normalize(feats[m], dim=-1)
                y_f = y[m].long()

                # SupCLIP: append fault-class text prototypes
                if _text_protos_for_supclip is not None:
                    _fault_ids = [k for k in sorted(text_prototypes.keys()) if int(k) > 0]
                    if _fault_ids:
                        _tp_fault = torch.stack([F.normalize(text_prototypes[k].float(), dim=-1) for k in _fault_ids], dim=0).to(feats_f.device)
                        _tp_labels = torch.tensor(_fault_ids, device=feats_f.device, dtype=y_f.dtype)
                        feats_f = torch.cat([feats_f, _tp_fault], dim=0)
                        y_f = torch.cat([y_f, _tp_labels], dim=0)

                N = feats_f.size(0)
                logits = torch.matmul(feats_f, feats_f.t()) / float(supcon_temperature)  # [N,N]
                self_mask = torch.eye(N, dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(self_mask, -1e4 if logits.dtype == torch.float16 else -1e9)

                y_col = y_f.view(-1, 1)
                pos_mask = (y_col == y_col.t()) & (~self_mask)
                pos_counts = pos_mask.sum(dim=1)  # [N]

                valid = pos_counts > 0
                if valid.any():
                    uniq, cnt = torch.unique(y_f, return_counts=True)
                    cnt_map = {int(k.item()): int(v.item()) for k, v in zip(uniq, cnt)}
                    w = torch.tensor(
                        [1.0 / float(cnt_map[int(t.item())]) for t in y_f],
                        device=logits.device,
                        dtype=logits.dtype
                    )

                    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_counts.clamp_min(1).float()

                    loss_supcon = -(w[valid] * mean_log_prob_pos[valid]).sum() / w[valid].sum().clamp_min(1e-12)
        else:
            # Non-fault-only SupCon with optional SupCLIP
            _feats_sc = feats
            _y_sc = y
            if _text_protos_for_supclip is not None:
                _tp_labels = torch.tensor(sorted(text_prototypes.keys()), device=feats.device, dtype=y.dtype)
                _feats_sc = torch.cat([F.normalize(feats, dim=-1), _text_protos_for_supclip], dim=0)
                _y_sc = torch.cat([y, _tp_labels], dim=0)
            loss_supcon = _supcon(_feats_sc, _y_sc, temperature=supcon_temperature)

    # ATT479: supervised contrastive on fusion pre-L2 (pairwise). ATT485: optional batch soft silhouette.
    loss_fusion_supcon = torch.tensor(0.0, device=z_fusion.device)
    _geom_mode = str(fusion_geometry_loss_mode or "silhouette").strip().lower()
    if _geom_mode not in ("supcon", "silhouette"):
        _geom_mode = "silhouette"
    if (
        use_fusion_supcon_loss
        and fusion_supcon_weight > 0.0
        and z_fusion_supcon is not None
        and torch.is_tensor(z_fusion_supcon)
    ):
        if _geom_mode == "silhouette":
            _eps = float(fusion_silhouette_eps)
            if fusion_supcon_fault_only:
                m = labels > 0
                if m.sum() >= 2:
                    _sl = fusion_batch_soft_silhouette_loss(
                        z_fusion_supcon[m], labels[m], eps=_eps
                    )
                    loss_fusion_supcon = _sl if _sl is not None else loss_fusion_supcon
            else:
                _sl = fusion_batch_soft_silhouette_loss(
                    z_fusion_supcon, labels, eps=_eps
                )
                loss_fusion_supcon = _sl if _sl is not None else loss_fusion_supcon
        elif fusion_supcon_fault_only:
            m = labels > 0
            if m.sum() >= 2:
                loss_fusion_supcon = _supcon(
                    z_fusion_supcon[m], labels[m], temperature=fusion_supcon_temperature
                )
        else:
            loss_fusion_supcon = _supcon(
                z_fusion_supcon, labels, temperature=fusion_supcon_temperature
            )

    # Prototype separation (comes from decoder forward for correct gradients)
    loss_proto_sep = torch.tensor(0.0, device=z_fusion.device)
    if use_proto_sep_loss and hierarchical_parts is not None:
        v = hierarchical_parts.get("proto_sep_loss", None)
        if v is not None:
            loss_proto_sep = v

    # ===== (C3) Diagnosis anti-collapse regularization (batch-level KL) =====
    # Prevent "all → one class" collapse by matching batch-mean predicted distribution to batch label histogram.
    loss_diag_anticollapse = torch.tensor(0.0, device=z_fusion.device)
    use_diag_anticollapse = False
    diag_anticollapse_weight = 0.0
    diag_anticollapse_eps = 0.01
    diag_anticollapse_temp = 1.0
    try:
        import config as cfg
        use_diag_anticollapse = bool(getattr(cfg, "USE_DIAG_ANTICOLLAPSE_REG", False))
        diag_anticollapse_weight = float(getattr(cfg, "DIAG_ANTICOLLAPSE_WEIGHT", 0.0))
        diag_anticollapse_eps = float(getattr(cfg, "DIAG_ANTICOLLAPSE_EPS", 0.01))
        diag_anticollapse_temp = float(getattr(cfg, "DIAG_ANTICOLLAPSE_TEMPERATURE", 1.0))
    except Exception:
        use_diag_anticollapse = False
        diag_anticollapse_weight = 0.0

    if use_diag_anticollapse and diag_anticollapse_weight > 0 and hierarchical_parts is not None:
        logits_diag = hierarchical_parts.get("logits_diag", None)  # [B,4]
        if logits_diag is not None and logits_diag.dim() == 2 and logits_diag.size(1) >= 2:
            # Predicted distribution (batch-mean)
            p = F.softmax(logits_diag / float(diag_anticollapse_temp), dim=1)  # [B,C]
            p_bar = p.mean(dim=0)  # [C]
            num_classes = logits_diag.size(1)

            # Target distribution q
            target = "batch"
            prior = None
            try:
                import config as cfg
                target = str(getattr(cfg, "DIAG_ANTICOLLAPSE_TARGET", "batch")).lower()
                prior = getattr(cfg, "DIAG_ANTICOLLAPSE_PRIOR", None)
            except Exception:
                target = "batch"
                prior = None

            if target == "uniform":
                q = torch.full((num_classes,), 1.0 / float(num_classes), device=p_bar.device, dtype=p_bar.dtype)
            elif target == "fixed" and prior is not None:
                try:
                    q = torch.tensor(prior, device=p_bar.device, dtype=p_bar.dtype)
                    if q.numel() != num_classes:
                        q = torch.full((num_classes,), 1.0 / float(num_classes), device=p_bar.device, dtype=p_bar.dtype)
                    q = torch.clamp(q, min=1e-8)
                    q = q / q.sum()
                except Exception:
                    q = torch.full((num_classes,), 1.0 / float(num_classes), device=p_bar.device, dtype=p_bar.dtype)
            else:
                # "batch" (legacy): label histogram (smoothed)
                y = labels.clamp(min=0, max=num_classes - 1).long()
                counts = torch.bincount(y, minlength=num_classes).to(dtype=p_bar.dtype, device=p_bar.device)
                q = (counts + float(diag_anticollapse_eps)) / (counts.sum() + float(diag_anticollapse_eps) * float(num_classes))

            # KL(p_bar || q)
            p_bar = torch.clamp(p_bar, min=1e-8, max=1.0)
            q = torch.clamp(q, min=1e-8, max=1.0)
            loss_diag_anticollapse = (p_bar * (torch.log(p_bar) - torch.log(q))).sum()

    # 一致性损失（如果提供了增强版本的fusion embedding）
    loss_consistency = torch.tensor(0.0, device=z_fusion.device)
    if z_fusion_aug is not None and w_consistency > 0:
        # z_fusion_aug可能只包含部分样本（优化后只增强部分样本）
        if augment_indices is not None:
            # 使用augment_indices来选择对应的原始样本
            z_fusion_subset = z_fusion[augment_indices]
            loss_consistency = fusion_consistency_loss(
                z_fusion_subset, z_fusion_aug, temperature=consistency_temperature
            )
        else:
            # 全部样本都增强（旧版本兼容）
            loss_consistency = fusion_consistency_loss(
                z_fusion, z_fusion_aug, temperature=consistency_temperature
            )

    # ===== Apply margin router (reduce conflicting margins) =====
    # In router mode, keep only the margin group that targets the current absorber.
    if router_enabled and router_target != "all":
        if router_target == "vs_inner":
            # Keep only {normal/outer/rolling vs inner}
            outer_margin_weight = 0.0
            inner_margin_weight = 0.0
            normal_margin_weight = 0.0
            in_vn_w = 0.0
            out_vn_w = 0.0
            roll_vn_w = 0.0
            rolling_absorb_weight = 0.0
            in_vr_w = 0.0
            out_vr_w = 0.0
        elif router_target == "vs_rolling":
            # Keep only {anti-rolling-absorb + inner/outer vs rolling}
            outer_margin_weight = 0.0
            inner_margin_weight = 0.0
            normal_margin_weight = 0.0
            in_vn_w = 0.0
            out_vn_w = 0.0
            roll_vn_w = 0.0
            n_vi_w = 0.0
            o_vi_w = 0.0
            r_vi_w = 0.0
        elif router_target == "vs_normal":
            # Keep only {normal margin + (inner/outer/rolling vs normal)}
            outer_margin_weight = 0.0
            inner_margin_weight = 0.0
            rolling_absorb_weight = 0.0
            in_vr_w = 0.0
            out_vr_w = 0.0
            n_vi_w = 0.0
            o_vi_w = 0.0
            r_vi_w = 0.0

    if ablation_gate_bank is not None:
        loss_fusion_proto_align = ablation_gate_bank.scale_loss(
            "gcsm_fusion_align", loss_fusion_proto_align
        )
        _gcsm_geom_key = (
            "gcsm_silhouette"
            if str(fusion_geometry_loss_mode or "silhouette").strip().lower() == "silhouette"
            else "gcsm_metric_geometry"
        )
        loss_fusion_supcon = ablation_gate_bank.scale_loss(_gcsm_geom_key, loss_fusion_supcon)

    total_loss = (
        w_fusion * loss_fusion +
        w_signal * loss_signal +
        w_spec * loss_spec +
        w_margin * loss_margin +
        w_fusion_cls * loss_fusion_decoder +
        w_consistency * loss_consistency +
        supcon_weight * loss_supcon +
        fusion_supcon_weight * loss_fusion_supcon +
        proto_sep_weight * loss_proto_sep +
        sem_distill_weight * loss_sem_distill +
        nf_gate_margin_w * loss_nf_gate_margin +
        fusion_proto_align_weight * loss_fusion_proto_align +
        nr_inter_cos_weight * loss_nr_inter_cos_margin +
        outer_margin_weight * loss_outer_margin +
        inner_margin_weight * loss_inner_margin +
        normal_margin_weight * loss_normal_margin +
        in_vn_w * loss_inner_vs_normal +
        out_vn_w * loss_outer_vs_normal +
        roll_vn_w * loss_rolling_vs_normal +
        rolling_absorb_weight * loss_rolling_absorb +
        in_vr_w * loss_inner_vs_roll +
        out_vr_w * loss_outer_vs_roll +
        n_vi_w * loss_normal_vs_inner +
        o_vi_w * loss_outer_vs_inner +
        r_vi_w * loss_rolling_vs_inner +
        text_prompt_sep_weight * loss_text_prompt_sep +
        diag_anticollapse_weight * loss_diag_anticollapse +
        sem_fvn_weight * loss_sem_fault_vs_normal +
        loss_proto_cos_align
    )

    loss_dict = {
        "fusion": loss_fusion.item(),
        "signal": loss_signal.item(),
        "spec": loss_spec.item(),
        "margin": loss_margin.item(),
        "fusion_cls": loss_fusion_decoder.item(),
        "proto_cos_align": loss_proto_cos_align.item(),
        "supcon": float(loss_supcon.detach().cpu().item()) if use_supcon_loss else 0.0,
        "fusion_supcon": float(loss_fusion_supcon.detach().cpu().item())
        if (use_fusion_supcon_loss and fusion_supcon_weight > 0.0)
        else 0.0,
        "proto_sep": float(loss_proto_sep.detach().cpu().item()) if use_proto_sep_loss else 0.0,
        "sem_distill": float(loss_sem_distill.detach().cpu().item()) if (use_sem_distill and sem_distill_weight > 0) else 0.0,
        "nf_gate_margin": float(loss_nf_gate_margin.detach().cpu().item()) if (use_nf_gate_margin and nf_gate_margin_w > 0) else 0.0,
        "fusion_proto_align": float(loss_fusion_proto_align.detach().cpu().item()) if (use_fusion_proto_align and fusion_proto_align_weight > 0) else 0.0,
        "nr_inter_cos_margin": float(loss_nr_inter_cos_margin.detach().cpu().item()) if (use_nr_inter_cos_margin and nr_inter_cos_weight > 0) else 0.0,
        "nr_inter_cos_hard": float(nr_inter_cos_hard.detach().cpu().item()) if (use_nr_inter_cos_margin and nr_inter_cos_weight > 0) else 0.0,
        "text_prompt_sep": float(loss_text_prompt_sep.detach().cpu().item()) if (use_text_prompt_sep and text_prompt_sep_weight > 0) else 0.0,
        "_info_prompt_inner_outer_cos": float(text_prompt_inner_outer_cos.detach().cpu().item()) if torch.is_tensor(text_prompt_inner_outer_cos) else float("nan"),
        "outer_margin": float(loss_outer_margin.detach().cpu().item()) if (use_outer_margin_loss and outer_margin_weight > 0) else 0.0,
        "inner_margin": float(loss_inner_margin.detach().cpu().item()) if (use_inner_margin_loss and inner_margin_weight > 0) else 0.0,
        "normal_margin": float(loss_normal_margin.detach().cpu().item()) if (use_normal_margin_loss and normal_margin_weight > 0) else 0.0,
        "inner_vs_normal_margin": float(loss_inner_vs_normal.detach().cpu().item()) if (use_in_vn and in_vn_w > 0) else 0.0,
        "outer_vs_normal_margin": float(loss_outer_vs_normal.detach().cpu().item()) if (use_out_vn and out_vn_w > 0) else 0.0,
        "rolling_vs_normal_margin": float(loss_rolling_vs_normal.detach().cpu().item()) if (use_roll_vn and roll_vn_w > 0) else 0.0,
        "rolling_absorb_margin": float(loss_rolling_absorb.detach().cpu().item()) if (use_rolling_absorb and rolling_absorb_weight > 0) else 0.0,
        "inner_vs_rolling_margin": float(loss_inner_vs_roll.detach().cpu().item()) if (use_in_vr and in_vr_w > 0) else 0.0,
        "outer_vs_rolling_margin": float(loss_outer_vs_roll.detach().cpu().item()) if (use_out_vr and out_vr_w > 0) else 0.0,
        "normal_vs_inner_margin": float(loss_normal_vs_inner.detach().cpu().item()) if (use_n_vi and n_vi_w > 0) else 0.0,
        "outer_vs_inner_margin": float(loss_outer_vs_inner.detach().cpu().item()) if (use_o_vi and o_vi_w > 0) else 0.0,
        "rolling_vs_inner_margin": float(loss_rolling_vs_inner.detach().cpu().item()) if (use_r_vi and r_vi_w > 0) else 0.0,
        "diag_anticollapse": float(loss_diag_anticollapse.detach().cpu().item()) if (use_diag_anticollapse and diag_anticollapse_weight > 0) else 0.0,
        "sem_fault_vs_normal_anchor": float(loss_sem_fault_vs_normal.detach().cpu().item()) if (use_sem_fvn and sem_fvn_weight > 0) else 0.0,
        "margin_router_target": router_target,
        "total": total_loss.item()
    }
    loss_dict["_tensor_terms"] = {
        "fusion_cls_weighted": (w_fusion_cls * loss_fusion_decoder),
        "supcon_weighted": (supcon_weight * loss_supcon),
        "fusion_supcon_weighted": (fusion_supcon_weight * loss_fusion_supcon),
        "fusion_proto_align_weighted": (fusion_proto_align_weight * loss_fusion_proto_align),
        "nr_inter_cos_margin_weighted": (nr_inter_cos_weight * loss_nr_inter_cos_margin),
        "text_prompt_sep_weighted": (text_prompt_sep_weight * loss_text_prompt_sep),
        "signal_ce_weighted": (w_signal * loss_signal),
        "spec_ce_weighted": (w_spec * loss_spec),
        "fusion_alignment_weighted": (w_fusion * loss_fusion),
    }
    
    if w_consistency > 0:
        loss_dict["consistency"] = loss_consistency.item()

    return total_loss, loss_dict
