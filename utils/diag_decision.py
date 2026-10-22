import torch
import torch.nn.functional as F


def decide_diagnosis_from_logits(
    *,
    logits_diag: torch.Tensor,
    logits_nf_global: torch.Tensor,
    diag_num_classes: int = 4,
    decision_mode: str = "nf_protect",
    nf_mode: str = "threshold",
    nf_threshold: float | None = None,
    platt: tuple[float, float] | None = None,
    # nf_gate gray-zone fallback
    gray_low: float = 0.35,
    gray_high: float = 0.65,
    gray_fallback: str = "raw",
    # nf_protect settings
    use_nf_protected: bool = True,
    pfault_max: float = 0.01,
    protect_mode: str = "soft_bias",
    protect_bias: float = 4.0,
    protect_diag_margin: float = 0.5,
    # soft_hierarchical settings
    soft_alpha: float = 1.0,
) -> dict:
    """
    Unified diagnosis decision logic used by:
      - training-time zero-shot eval
      - evaluation-time target-domain evaluation

    Returns dict with:
      - diag_preds, diag_preds_raw
      - nf_preds (0=normal, 1=fault), p_fault
      - nf_gate_gray_frac
    """
    if logits_diag.ndim != 2:
        raise ValueError("logits_diag must be [B,C]")
    if logits_nf_global.ndim != 2 or logits_nf_global.size(1) != 2:
        raise ValueError("logits_nf_global must be [B,2]")

    decision_mode = str(decision_mode).lower()
    nf_mode = str(nf_mode).lower()
    gray_fallback = str(gray_fallback).lower()
    protect_mode = str(protect_mode).lower()

    # Use the requested number of classes for diagnosis.
    logits_use_raw = logits_diag[:, : int(diag_num_classes)]
    diag_preds_raw = logits_use_raw.argmax(dim=1)

    # Post-hoc logit adjustment (inference-time prior correction):
    # z'_k = z_k - tau * log(n_k), where n_k is source class count.
    # This is applied only to decision logits, never to training loss.
    logits_use = logits_use_raw
    try:
        import config as cfg
        use_posthoc = bool(getattr(cfg, "USE_POSTHOC_LOGIT_ADJUST", False))
        tau = float(getattr(cfg, "POSTHOC_LOGIT_ADJUST_TAU", 1.0))
        counts_cfg = getattr(cfg, "POSTHOC_LOGIT_CLASS_COUNTS", None)
        if use_posthoc and counts_cfg is not None and abs(tau) > 1e-12:
            counts = torch.as_tensor(
                counts_cfg,
                device=logits_use_raw.device,
                dtype=logits_use_raw.dtype,
            ).view(-1)
            c = int(logits_use_raw.size(1))
            if int(counts.numel()) >= c:
                counts = counts[:c].clamp(min=1.0)
                logits_use = logits_use_raw - float(tau) * torch.log(counts).view(1, -1)
    except Exception:
        logits_use = logits_use_raw

    # NF head: anomaly_logit and calibrated p_fault (for gray-zone and optional thresholding).
    anomaly_logit = (logits_nf_global[:, 1] - logits_nf_global[:, 0]).float()
    if platt is not None:
        s, b = float(platt[0]), float(platt[1])
        anomaly_logit = (s * anomaly_logit) + b
    p_fault = torch.sigmoid(anomaly_logit)

    # NF prediction
    if nf_mode == "threshold" and nf_threshold is not None:
        nf_preds = (anomaly_logit > float(nf_threshold)).long()
    else:
        nf_preds = logits_nf_global.argmax(dim=1).long()

    nf_gate_gray_frac = 0.0

    # ===== Open-set rejection wrapper =====
    # decision_mode examples:
    # - "open_set_soft_hierarchical"
    # - "open_set_argmax"
    # - "open_set_nf_protect"
    if decision_mode.startswith("open_set"):
        base = "soft_hierarchical"
        if "_" in decision_mode:
            try:
                base = decision_mode.split("_", 2)[-1]
            except Exception:
                base = "soft_hierarchical"
        # Compute base prediction (recursive call with base mode)
        out_base = decide_diagnosis_from_logits(
            logits_diag=logits_diag,
            logits_nf_global=logits_nf_global,
            diag_num_classes=diag_num_classes,
            decision_mode=base,
            nf_mode=nf_mode,
            nf_threshold=nf_threshold,
            platt=platt,
            gray_low=gray_low,
            gray_high=gray_high,
            gray_fallback=gray_fallback,
            use_nf_protected=use_nf_protected,
            pfault_max=pfault_max,
            protect_mode=protect_mode,
            protect_bias=protect_bias,
            protect_diag_margin=protect_diag_margin,
            soft_alpha=soft_alpha,
        )

        # Rejection criteria: low confidence OR NF uncertainty gray-zone
        try:
            import config as cfg
            pmax_th = float(getattr(cfg, "OPEN_SET_PMAX_TH", 0.55))
            gL = float(getattr(cfg, "OPEN_SET_PFAULT_GRAY_LOW", gray_low))
            gH = float(getattr(cfg, "OPEN_SET_PFAULT_GRAY_HIGH", gray_high))
            reject_label = int(getattr(cfg, "OPEN_SET_REJECT_LABEL", -1))
        except Exception:
            pmax_th, gL, gH, reject_label = 0.55, gray_low, gray_high, -1

        probs = torch.softmax(logits_use.float(), dim=1)
        pmax = probs.max(dim=1).values
        gray = (p_fault >= float(gL)) & (p_fault <= float(gH))
        unknown = (pmax < float(pmax_th)) | gray

        diag_preds = out_base["diag_preds"].clone()
        diag_preds[unknown] = int(reject_label)
        out_base["diag_preds"] = diag_preds
        out_base["open_set_unknown_frac"] = float(unknown.float().mean().detach().cpu().item()) if unknown.numel() > 0 else 0.0
        out_base["open_set_pmax_mean"] = float(pmax.mean().detach().cpu().item()) if pmax.numel() > 0 else 0.0
        return out_base
    # (New scheme 1) two_stage: if NF says Normal -> 0; else argmax(fault 3-way)+1
    if decision_mode == "two_stage":
        fault_pred = logits_use[:, 1:4].argmax(dim=1) + 1  # 1/2/3
        diag_preds = torch.where(nf_preds == 0, torch.zeros_like(fault_pred, device=fault_pred.device), fault_pred)
        return {
            "diag_preds": diag_preds,
            "diag_preds_raw": diag_preds_raw,
            "nf_preds": nf_preds,
            "p_fault": p_fault,
            "nf_gate_gray_frac": nf_gate_gray_frac,
        }

    # Attempt 9/10: sequential decision (Train soft, infer hard)
    # Rule: if NF indicates fault (default: anomaly_logit > 0 i.e. P(fault) > P(normal)),
    # then ignore normal class and predict among fault classes only; else predict normal.
    # NOTE: This MUST be used as a *decision rule* only; training loss must use unmasked soft logits.
    if decision_mode == "sequential":
        thr = 0.0
        if nf_mode == "threshold" and nf_threshold is not None:
            thr = float(nf_threshold)
        is_fault = (anomaly_logit > float(thr))
        fault_pred = logits_use[:, 1:].argmax(dim=1) + 1  # 1/2/3
        diag_preds = torch.where(is_fault, fault_pred, torch.zeros_like(fault_pred))
        return {
            "diag_preds": diag_preds,
            "diag_preds_raw": diag_preds_raw,
            "nf_preds": nf_preds,
            "p_fault": p_fault,
            "nf_gate_gray_frac": nf_gate_gray_frac,
        }

    if decision_mode == "nf_gate":
        fault_pred = logits_use[:, 1:].argmax(dim=1) + 1
        diag_preds = torch.where(nf_preds == 0, torch.zeros_like(fault_pred), fault_pred)

        # Gray-zone fallback: when NF is uncertain, avoid hard gate mistakes.
        gray = (p_fault >= float(gray_low)) & (p_fault <= float(gray_high))
        if gray.numel() > 0:
            nf_gate_gray_frac = float(gray.float().mean().detach().cpu().item())
        if gray.any() and gray_fallback == "raw":
            diag_preds = diag_preds.clone()
            diag_preds[gray] = diag_preds_raw[gray]

        return {
            "diag_preds": diag_preds,
            "diag_preds_raw": diag_preds_raw,
            "nf_preds": nf_preds,
            "p_fault": p_fault,
            "nf_gate_gray_frac": nf_gate_gray_frac,
        }

    # soft_hierarchical: bias fault-type logits by NF "faultness" logit (continuous, no hard gate).
    # This is a conservative alternative to nf_gate when hard gating collapses fault types under shift.
    if decision_mode == "soft_hierarchical":
        logits_soft = logits_use.clone()
        a = anomaly_logit.to(dtype=logits_soft.dtype)
        logits_soft[:, 1:] = logits_soft[:, 1:] + float(soft_alpha) * a.unsqueeze(1)
        diag_preds = logits_soft.argmax(dim=1)
        return {
            "diag_preds": diag_preds,
            "diag_preds_raw": diag_preds_raw,
            "nf_preds": nf_preds,
            "p_fault": p_fault,
            "nf_gate_gray_frac": nf_gate_gray_frac,
        }

    if decision_mode == "argmax":
        diag_preds = logits_use.argmax(dim=1)
        return {
            "diag_preds": diag_preds,
            "diag_preds_raw": diag_preds_raw,
            "nf_preds": nf_preds,
            "p_fault": p_fault,
            "nf_gate_gray_frac": nf_gate_gray_frac,
        }

    # nf_protect: only intervene when NF is very confident "normal"
    diag_preds = diag_preds_raw
    if use_nf_protected:
        # Use *raw* NF softmax here (not Platt), because this is a conservative protection heuristic.
        p_fault_raw = torch.softmax(logits_nf_global.float(), dim=1)[:, 1]
        nf_is_normal = p_fault_raw < float(pfault_max)
        if nf_is_normal.any():
            if protect_mode == "hard":
                diag_preds = diag_preds.clone()
                diag_preds[nf_is_normal] = 0
            else:
                logits_adj = logits_use.clone()
                bias = (
                    float(protect_bias)
                    * (float(pfault_max) - p_fault_raw).clamp(min=0.0)
                    / float(pfault_max)
                ).to(dtype=logits_adj.dtype)
                best_fault = logits_adj[:, 1:].max(dim=1).values
                margin = best_fault - logits_adj[:, 0]
                allow = nf_is_normal & (margin < float(protect_diag_margin))
                if allow.any():
                    logits_adj = logits_adj.clone()
                    logits_adj[allow, 0] = logits_adj[allow, 0] + bias[allow]
                diag_preds = logits_adj.argmax(dim=1)

    return {
        "diag_preds": diag_preds,
        "diag_preds_raw": diag_preds_raw,
        "nf_preds": nf_preds,
        "p_fault": p_fault,
        "nf_gate_gray_frac": nf_gate_gray_frac,
    }

