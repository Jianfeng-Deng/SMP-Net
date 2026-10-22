"""
Helpers for learnable ablation gating: input spec blending and gate-bank attachment.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from models.ablation_gates import AblationGateBank, apply_residual_gate


def is_ablation_gating_enabled(cfg: Any = None) -> bool:
    if cfg is None:
        try:
            import config as cfg
        except Exception:
            return False
    return bool(getattr(cfg, "ABLATION_GATING_ENABLE", False))


def create_ablation_gate_bank(cfg: Any, device: torch.device) -> Optional[AblationGateBank]:
    if not is_ablation_gating_enabled(cfg):
        return None
    init_default = float(getattr(cfg, "ABLATION_GATE_INIT_DEFAULT", 0.5))
    overrides = getattr(cfg, "ABLATION_GATE_INIT_OVERRIDES", None) or {}
    bank = AblationGateBank(init_default=init_default, init_overrides=overrides)
    return bank.to(device)


def attach_gate_bank(
    gate_bank: Optional[AblationGateBank],
    detector: Any = None,
    signal_encoder: Any = None,
    spec_encoder: Any = None,
    fusion_module: Any = None,
) -> None:
    """Attach the same gate bank reference to models that participate in gating."""
    for obj in (detector, signal_encoder, spec_encoder, fusion_module):
        if obj is not None:
            setattr(obj, "_ablation_gate_bank", gate_bank)
    if detector is not None:
        detector.ablation_gate_bank = gate_bank


def get_gate_bank(module: Any) -> Optional[AblationGateBank]:
    if module is None:
        return None
    bank = getattr(module, "_ablation_gate_bank", None)
    if bank is not None:
        return bank
    if hasattr(module, "ablation_gate_bank"):
        return getattr(module, "ablation_gate_bank", None)
    return None


def _align_spec_shapes(spec_a: torch.Tensor, spec_b: torch.Tensor) -> torch.Tensor:
    """Resize spec_b to match spec_a [B,C,F,T] if needed."""
    if spec_a.shape == spec_b.shape:
        return spec_b
    if spec_a.dim() != 4 or spec_b.dim() != 4:
        return spec_b
    if spec_a.shape[0] != spec_b.shape[0] or spec_a.shape[1] != spec_b.shape[1]:
        return spec_b
    return F.interpolate(
        spec_b,
        size=(spec_a.shape[2], spec_a.shape[3]),
        mode="bilinear",
        align_corners=False,
    )


def _expected_spec_channels(gate_ref: Any = None) -> Optional[int]:
    """Infer the channel count expected by the active SpecEncoder stem."""
    if gate_ref is None:
        return None
    spec_encoder = getattr(gate_ref, "spec_encoder", gate_ref)
    for attr in ("_stem_conv", "conv1"):
        conv = getattr(spec_encoder, attr, None)
        in_ch = getattr(conv, "in_channels", None)
        if isinstance(in_ch, int) and in_ch > 0:
            return in_ch
    stem = getattr(spec_encoder, "stem", None)
    if stem is not None:
        try:
            conv = stem[0]
            in_ch = getattr(conv, "in_channels", None)
            if isinstance(in_ch, int) and in_ch > 0:
                return in_ch
        except Exception:
            pass
    return None


def _match_expected_spec_channels(
    spec: torch.Tensor,
    batch: Optional[Dict[str, Any]],
    gate_ref: Any = None,
    cfg: Any = None,
) -> torch.Tensor:
    """Keep evaluation robust when a shared dataset is reused across ablation cases."""
    expected = _expected_spec_channels(gate_ref)
    if expected is None or spec.dim() != 4:
        return spec
    current = int(spec.shape[1])
    if current == expected:
        return spec

    if current > expected:
        # Most ablation mismatches are Full 7ch specs evaluated by a 6ch expert;
        # the appended physical-frequency axis is the last channel.
        adapted = spec[:, :expected, :, :]
    else:
        adapted = spec
        freq_pos = batch.get("spec_freq_pos", None) if batch is not None else None
        use_freq = bool(getattr(cfg, "USE_PHYSICAL_FREQ_POSITION_ENCODING", False)) if cfg is not None else False
        mode = str(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_MODE", "concat")).lower() if cfg is not None else "concat"
        if use_freq and mode == "concat" and freq_pos is not None and expected == current + 1:
            fp = freq_pos.to(device=spec.device, dtype=spec.dtype)
            if fp.dim() == 3:
                fp = fp.unsqueeze(0).expand(spec.shape[0], -1, -1, -1)
            if fp.dim() == 4 and fp.shape[0] == spec.shape[0]:
                if fp.shape[2:] != spec.shape[2:]:
                    fp = F.interpolate(fp, size=spec.shape[2:], mode="bilinear", align_corners=False)
                scale = float(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_SCALE", 2.0)) if cfg is not None else 2.0
                adapted = torch.cat([spec, scale * fp[:, :1, :, :]], dim=1)

    if int(adapted.shape[1]) != expected:
        return spec

    log_ref = gate_ref if gate_ref is not None else spec
    if not getattr(log_ref, "_spec_channel_adapt_logged", False):
        print(f"[SpecInputAdapt] channels {current} -> {expected} for active SpecEncoder")
        try:
            setattr(log_ref, "_spec_channel_adapt_logged", True)
        except Exception:
            pass
    return adapted


def apply_spec_input_gates(
    spec: torch.Tensor,
    batch: Optional[Dict[str, Any]],
    gate_bank: Optional[AblationGateBank],
    cfg: Any = None,
) -> torch.Tensor:
    """
    Blend optional baseline tensors from the batch (MS-STFT / freq encoding).
    Expects batch keys when gating is active:
      - spec_legacy: baseline STFT without MS-STFT (same C as spec when possible)
      - spec_freq_pos: [B,1,F,T] or [1,F,T] physical-frequency axis (concat/add modes)
    """
    if gate_bank is None or batch is None:
        return spec

    spec_out = spec

    spec_legacy = batch.get("spec_legacy", None)
    if spec_legacy is not None:
        leg = spec_legacy.to(device=spec.device, dtype=spec.dtype)
        if leg.dim() == 3:
            leg = leg.unsqueeze(0).expand(spec.shape[0], -1, -1, -1)
        leg = _align_spec_shapes(spec_out, leg)
        if leg.shape == spec_out.shape:
            spec_out = apply_residual_gate(gate_bank, "msstft", spec_out, leg)

    spec_freq_pos = batch.get("spec_freq_pos", None)
    if spec_freq_pos is not None:
        fp = spec_freq_pos.to(device=spec.device, dtype=spec.dtype)
        if fp.dim() == 3:
            fp = fp.unsqueeze(0).expand(spec_out.shape[0], -1, -1, -1)
        if fp.dim() == 4 and fp.shape[0] == spec_out.shape[0]:
            mode = "concat"
            scale = 2.0
            if cfg is not None:
                mode = str(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_MODE", "concat")).lower()
                scale = float(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_SCALE", 2.0))
            if mode == "concat":
                g = gate_bank.scalar("freq_encoding")
                gated_freq = g * (scale * fp)
                spec_out = torch.cat([spec_out, gated_freq], dim=1)
            elif mode == "add":
                if fp.shape[1] == 1 and fp.shape[2:] == spec_out.shape[2:]:
                    delta = scale * fp
                    spec_with = spec_out + delta
                    spec_out = apply_residual_gate(gate_bank, "freq_encoding", spec_with, spec_out)
            else:
                if fp.shape[1] == 1 and fp.shape[2:] == spec_out.shape[2:]:
                    spec_with = spec_out + scale * fp
                    spec_out = apply_residual_gate(gate_bank, "freq_encoding", spec_with, spec_out)

    return spec_out


def prepare_batch_spec(
    batch: Dict[str, Any],
    device: torch.device,
    gate_ref: Any = None,
    cfg: Any = None,
) -> torch.Tensor:
    """Load spec from batch and apply input-level ablation gates (MS-STFT / freq encoding)."""
    spec = batch["spec"].to(device)
    bank = get_gate_bank(gate_ref) if gate_ref is not None else None
    if bank is None and gate_ref is not None:
        bank = getattr(gate_ref, "ablation_gate_bank", None)
    spec = apply_spec_input_gates(spec, batch, bank, cfg)
    return _match_expected_spec_channels(spec, batch, gate_ref, cfg)


def load_spec_from_batch(
    batch: Dict[str, Any],
    device: torch.device,
    gate_ref: Any = None,
    cfg: Any = None,
) -> torch.Tensor:
    """Alias for evaluation/inference paths that previously used batch['spec'].to(device)."""
    if cfg is None:
        try:
            import config as cfg
        except Exception:
            cfg = None
    return prepare_batch_spec(batch, device, gate_ref, cfg)


def gate_bypass_vector(
    gate_bank: Optional[AblationGateBank],
    key: str,
    bypass_vec: Optional[torch.Tensor],
    zero_vec: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Gate an auxiliary bypass embedding: out = g * bypass (identity = 0)."""
    if bypass_vec is None:
        return None
    if gate_bank is None:
        return bypass_vec
    if zero_vec is None:
        zero_vec = torch.zeros_like(bypass_vec)
    return apply_residual_gate(gate_bank, key, bypass_vec, zero_vec)
