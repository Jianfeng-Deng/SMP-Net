"""
Module Portfolio + unsupervised test-time expert selection.

Each ablation case (full_model, backbone_only, wo_domain_cond, ...) is an expert.
At test time we score experts without labels and pick the best (or soft-weight vote).
"""
from __future__ import annotations

import importlib
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.ablation_gating import load_spec_from_batch

DEFAULT_PORTFOLIO_EXPERTS: List[str] = [
    "full_model",
    "backbone_only",
    "wo_domain_cond",
    "wo_pidr",
    "wo_gcsm",
]

# Keys cleared before applying a case (mirrors experiments/run_ablation.sh).
_ABLATION_ENV_KEYS = [
    "ABLATION_GATING_ENABLE",
    "MULTISCALE_STFT_ENABLE",
    "USE_PHYSICAL_FREQ_POSITION_ENCODING",
    "WST_ENABLE",
    "ENV_BYPASS_INJECT",
    "USE_PHYS_FEATURE_PROTO",
    "SPEC_FS_COND_FILM_ENABLE",
    "USE_STAT_FILM",
    "USE_MIXSTYLE_SIGNAL",
    "USE_MIXSTYLE_SPEC",
    "USE_DOMAIN_LORA",
    "DUAL_CONTRACT_ENABLE",
    "SPLIT_ARCH_ENABLE",
    "FUSION_MEAN_RESIDUAL_ENABLE",
    "FUSION_CROSS_MODAL_ALIGN_WEIGHT",
    "FUSION_DECODER_INPUT_GRAD_LIVE_RATIO",
    "FUSION_ALIGN_GROUP_ENABLE",
    "METRIC_GEOMETRY_GROUP_ENABLE",
    "DOMAIN_ADAPT_GROUP_ENABLE",
    "FUSION_GEOMETRY_LOSS_MODE",
]

_BASE_ENV = {
    "ENABLE_INTRA_CONDITION_TASK": "1",
    "TASK1_RUN_ALL_PHASES_IN_ONE_SCRIPT": "1",
    "TASK1_TRAIN_PER_DOMAIN_ISOLATED": "1",
    "LOO_ENABLE": "1",
    "LOO_DOMAINS": "CWRU,PU,HUST,SEU,HIT",
    "STRICT_INTRA_CONDITION_SPLIT": "1",
    "LOO_STRICT_MODE": "1",
    "SKIP_TASK2": "1",
    "FUSION_ALIGN_GROUP_ENABLE": "1",
    "METRIC_GEOMETRY_GROUP_ENABLE": "1",
    "DOMAIN_ADAPT_GROUP_ENABLE": "1",
    "TASK1_ENABLE_PRETEST_FINETUNE": "0",
}

_CASE_ENV_PATCHES: Dict[str, Dict[str, str]] = {
    "full_model": {},
    "full_model_gated": {"ABLATION_GATING_ENABLE": "1"},
    "full_model_adaptive": {
        "ABLATION_GATING_ENABLE": "1",
        "ABLATION_GATE_INIT_DEFAULT": "0.90",
        "ABLATION_GATE_LR_MULT": "5.0",
    },
    "backbone_only": {
        "MULTISCALE_STFT_ENABLE": "0",
        "USE_PHYSICAL_FREQ_POSITION_ENCODING": "0",
        "WST_ENABLE": "0",
        "ENV_BYPASS_INJECT": "0",
        "USE_PHYS_FEATURE_PROTO": "0",
        "SPEC_FS_COND_FILM_ENABLE": "0",
        "USE_STAT_FILM": "0",
        "USE_MIXSTYLE_SIGNAL": "0",
        "USE_MIXSTYLE_SPEC": "0",
        "USE_DOMAIN_LORA": "0",
        "DUAL_CONTRACT_ENABLE": "0",
        "SPLIT_ARCH_ENABLE": "0",
        "FUSION_MEAN_RESIDUAL_ENABLE": "0",
        "FUSION_CROSS_MODAL_ALIGN_WEIGHT": "0.0",
        "FUSION_ALIGN_GROUP_ENABLE": "0",
        "METRIC_GEOMETRY_GROUP_ENABLE": "0",
    },
    "wo_pidr": {
        "MULTISCALE_STFT_ENABLE": "0",
        "USE_PHYSICAL_FREQ_POSITION_ENCODING": "0",
        "WST_ENABLE": "0",
        "ENV_BYPASS_INJECT": "0",
        "USE_PHYS_FEATURE_PROTO": "0",
        "SPEC_FS_COND_FILM_ENABLE": "0",
        "USE_STAT_FILM": "0",
        "USE_MIXSTYLE_SIGNAL": "0",
        "USE_MIXSTYLE_SPEC": "0",
        "USE_DOMAIN_LORA": "0",
    },
    "wo_gidc": {
        "DUAL_CONTRACT_ENABLE": "0",
        "SPLIT_ARCH_ENABLE": "0",
        "FUSION_MEAN_RESIDUAL_ENABLE": "0",
        "FUSION_CROSS_MODAL_ALIGN_WEIGHT": "0.0",
    },
    "wo_gcsm": {
        "FUSION_ALIGN_GROUP_ENABLE": "0",
        "METRIC_GEOMETRY_GROUP_ENABLE": "0",
    },
    "wo_domain_cond": {
        "SPEC_FS_COND_FILM_ENABLE": "0",
        "USE_STAT_FILM": "0",
        "USE_MIXSTYLE_SIGNAL": "0",
        "USE_MIXSTYLE_SPEC": "0",
        "USE_DOMAIN_LORA": "0",
    },
    "wo_mixstyle": {
        "USE_MIXSTYLE_SIGNAL": "0",
        "USE_MIXSTYLE_SPEC": "0",
    },
    "wo_fs_cond_film": {
        "SPEC_FS_COND_FILM_ENABLE": "0",
    },
    "wo_stat_film": {
        "USE_STAT_FILM": "0",
    },
    "wo_domain_lora": {
        "USE_DOMAIN_LORA": "0",
    },
    "wo_wst": {
        "WST_ENABLE": "0",
    },
    "wo_env_bypass": {
        "ENV_BYPASS_INJECT": "0",
    },
    "wo_phys_feature_proto": {
        "USE_PHYS_FEATURE_PROTO": "0",
    },
}


def clear_ablation_env() -> None:
    for key in _ABLATION_ENV_KEYS:
        os.environ.pop(key, None)


def apply_portfolio_case_env(case_name: str) -> Any:
    """Apply ablation-case env vars and reload config module."""
    if case_name not in _CASE_ENV_PATCHES:
        raise ValueError(f"Unknown portfolio expert case: {case_name}")

    clear_ablation_env()
    for key, val in _BASE_ENV.items():
        os.environ[key] = str(val)
    os.environ.pop("FUSION_DECODER_INPUT_GRAD_LIVE_RATIO", None)
    for key, val in _CASE_ENV_PATCHES[case_name].items():
        os.environ[key] = str(val)

    import config as config_module

    importlib.reload(config_module)
    return config_module.Config()


def _encode_domain_ids_batch(batch: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    domains = batch.get("domain", None)
    if domains is None:
        return None
    try:
        from utils.evaluation_utils import _encode_domain_ids

        return _encode_domain_ids(domains, batch.get("condition", None), device)
    except Exception:
        domain_map = {"CWRU": 0, "PU": 2, "HUST": 3, "HIT": 4, "SEU": 5}
        return torch.tensor([domain_map.get(str(d), 0) for d in domains], dtype=torch.long, device=device)


def collect_unsupervised_signals(
    detector: Any,
    loader: Any,
    device: torch.device,
    diag_num_classes: int = 4,
) -> Dict[str, Any]:
    """
    Forward-only unsupervised signals for test-time expert selection (no labels).
    Returns logits/probs for optional soft voting plus scalar diagnostics.
    """
    detector.signal_encoder.eval()
    detector.spec_encoder.eval()
    detector.fusion_module.eval()
    if detector.fusion_decoder is not None:
        detector.fusion_decoder.eval()

    all_fusion_probs: List[torch.Tensor] = []
    all_spec_probs: List[torch.Tensor] = []
    fusion_feats: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            spec = load_spec_from_batch(batch, device, detector)
            signal_mask = batch["signal_mask"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            domain_ids = _encode_domain_ids_batch(batch, device)
            stat_vec = batch.get("stat_vec", None)
            if stat_vec is not None:
                stat_vec = stat_vec.to(device)

            try:
                z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
            except TypeError:
                z_sig = detector.signal_encoder(signal, signal_mask)

            spec_kwargs = {
                "spec": spec,
                "spec_mask": spec_mask,
                "domain_ids": domain_ids,
                "stat_vec": stat_vec,
            }
            wst, env, ratio, scd = None, None, None, None
            if getattr(detector, "_wst_bypass_active", False) and getattr(detector, "wst_encoder", None) is not None:
                wst_raw = batch.get("wst_spec", None)
                if wst_raw is not None:
                    wst = detector.wst_encoder(wst_raw.to(device))
            if getattr(detector, "_env_bypass_active", False) and getattr(detector, "env_encoder", None) is not None:
                env_raw = batch.get("env_spec", None)
                if env_raw is not None:
                    env = detector.env_encoder(env_raw.to(device))
                ratio_raw = batch.get("bpfi_bpfo_ratio", None)
                if ratio_raw is not None:
                    ratio = ratio_raw.to(device)
            if getattr(detector, "_scd_bypass_active", False) and getattr(detector, "scd_encoder", None) is not None:
                scd_raw = batch.get("scd_spec", None)
                if scd_raw is not None:
                    scd = detector.scd_encoder(scd_raw.to(device))
            if wst is not None:
                spec_kwargs["wst_aux"] = wst
            if env is not None:
                spec_kwargs["env_aux"] = env
            if ratio is not None:
                spec_kwargs["ratio_aux"] = ratio
            if scd is not None:
                spec_kwargs["scd_aux"] = scd

            use_spec_head = getattr(detector.spec_encoder, "use_classification_head", False)
            if use_spec_head:
                z_spec, spec_logits = detector.spec_encoder(**spec_kwargs, return_logits=True)
            else:
                z_spec = detector.spec_encoder(**spec_kwargs)
                spec_logits = None

            if hasattr(detector.fusion_module, "domain_embedding"):
                z_fusion = detector.fusion_module(
                    z_sig,
                    z_spec,
                    domain_ids=domain_ids,
                    text_prototypes=detector._active_text_prototypes(),
                )
            else:
                z_fusion = detector.fusion_module(z_sig, z_spec)

            fusion_feats.append(z_fusion.detach().float().cpu())

            if spec_logits is not None and spec_logits.dim() == 2:
                _c = min(int(diag_num_classes), int(spec_logits.size(1)))
                all_spec_probs.append(
                    torch.softmax(spec_logits[:, :_c].float(), dim=1).cpu()
                )

            if detector.fusion_decoder is not None:
                protos = detector._active_text_prototypes()
                try:
                    fusion_logits = detector.fusion_decoder(
                        z_fusion,
                        domain_ids=domain_ids,
                        text_prototypes=protos,
                    )
                except TypeError:
                    fusion_logits = detector.fusion_decoder(z_fusion, text_prototypes=protos)
                if isinstance(fusion_logits, tuple):
                    fusion_logits = fusion_logits[0]
                if fusion_logits is not None and fusion_logits.dim() == 2:
                    _c = min(int(diag_num_classes), int(fusion_logits.size(1)))
                    all_fusion_probs.append(
                        torch.softmax(fusion_logits[:, :_c].float(), dim=1).cpu()
                    )

    if len(all_fusion_probs) == 0:
        return {
            "score": float("-inf"),
            "fusion_var": 0.0,
            "mean_entropy": 1.0,
            "mean_margin": 0.0,
            "spec_fusion_agreement": 0.0,
            "collapsed": True,
            "n_samples": 0,
            "fusion_probs": None,
            "pred_labels": None,
        }

    fusion_probs = torch.cat(all_fusion_probs, dim=0)
    n_samples = int(fusion_probs.size(0))
    num_classes = int(fusion_probs.size(1))

    if len(fusion_feats) > 0:
        z_all = torch.cat(fusion_feats, dim=0)
        fusion_var = float(z_all.var(dim=0).mean().item()) if z_all.numel() > 0 else 0.0
    else:
        fusion_var = 0.0

    entropy = -(fusion_probs * torch.log(fusion_probs.clamp(min=1e-8))).sum(dim=1)
    mean_entropy = float(entropy.mean().item())
    max_entropy = math.log(max(2, num_classes))
    norm_entropy = mean_entropy / max(max_entropy, 1e-8)

    top2 = torch.topk(fusion_probs, k=min(2, num_classes), dim=1).values
    if top2.size(1) >= 2:
        margins = top2[:, 0] - top2[:, 1]
    else:
        margins = top2[:, 0]
    mean_margin = float(margins.mean().item())

    spec_fusion_agreement = 0.0
    if len(all_spec_probs) == len(all_fusion_probs) and len(all_spec_probs) > 0:
        spec_probs = torch.cat(all_spec_probs, dim=0)
        agree = (spec_probs.argmax(dim=1) == fusion_probs.argmax(dim=1)).float().mean()
        spec_fusion_agreement = float(agree.item())

    collapsed = fusion_var < 1e-4 or (norm_entropy < 0.05 and mean_margin > 0.95)

    # Higher is better. Penalize collapse / over-confidence; reward stable structure.
    score = (
        2.0 * math.log(fusion_var + 1e-6)
        + 1.5 * mean_margin
        + 1.0 * spec_fusion_agreement
        - 1.2 * norm_entropy
    )
    if collapsed:
        score -= 5.0

    pred_labels = fusion_probs.argmax(dim=1).numpy()

    return {
        "score": float(score),
        "fusion_var": fusion_var,
        "mean_entropy": mean_entropy,
        "norm_entropy": norm_entropy,
        "mean_margin": mean_margin,
        "spec_fusion_agreement": spec_fusion_agreement,
        "collapsed": collapsed,
        "n_samples": n_samples,
        "fusion_probs": fusion_probs.numpy(),
        "pred_labels": pred_labels,
    }


def select_expert_by_score(expert_signals: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
    best_name = ""
    best_score = float("-inf")
    for name, sig in expert_signals.items():
        sc = float(sig.get("score", float("-inf")))
        if sc > best_score:
            best_score = sc
            best_name = name
    return best_name, best_score


def soft_weighted_vote(
    expert_signals: Dict[str, Dict[str, Any]],
    temperature: float = 1.0,
) -> Optional[np.ndarray]:
    """Soft-weighted probability vote across experts (unsupervised weights from scores)."""
    probs_list = []
    scores = []
    for name, sig in expert_signals.items():
        p = sig.get("fusion_probs", None)
        if p is None:
            continue
        probs_list.append(np.asarray(p, dtype=np.float64))
        scores.append(float(sig.get("score", 0.0)))
    if not probs_list:
        return None

    min_len = min(p.shape[0] for p in probs_list)
    min_c = min(p.shape[1] for p in probs_list)
    stacked = np.stack([p[:min_len, :min_c] for p in probs_list], axis=0)

    s = np.asarray(scores, dtype=np.float64)
    s = s - s.max()
    w = np.exp(s / max(temperature, 1e-6))
    w = w / w.sum()
    mix = np.tensordot(w, stacked, axes=(0, 0))
    mix = mix / np.clip(mix.sum(axis=1, keepdims=True), 1e-8, None)
    return mix.argmax(axis=1)


def compute_diag_acc(labels: np.ndarray, preds: np.ndarray, diag_num_classes: int) -> float:
    if labels is None or preds is None or len(labels) == 0:
        return float("nan")
    lbl = np.asarray(labels, dtype=np.int64)
    prd = np.asarray(preds, dtype=np.int64)
    if int(diag_num_classes) == 3:
        lbl = np.clip(lbl, 0, 2)
        prd = np.clip(prd, 0, 2)
    return float(np.mean(lbl == prd))
