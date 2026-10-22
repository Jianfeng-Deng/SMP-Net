from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.signal_to_text import extract_diagnostic_features


def _safe_l2(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=-1)


def _cosine_kmeans(
    z: torch.Tensor,
    k: int,
    max_iters: int = 25,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cosine k-means on normalized embeddings.
    Returns:
      centers: [k, d]
      assign:  [n]
      conf:    [n], cosine to assigned center
    """
    z = _safe_l2(z)
    n = int(z.size(0))
    d = int(z.size(1))
    k_eff = max(1, min(int(k), n))

    g = torch.Generator(device=z.device)
    g.manual_seed(int(seed))
    init_ids = torch.randperm(n, generator=g, device=z.device)[:k_eff]
    centers = z[init_ids].clone()

    for _ in range(max(1, int(max_iters))):
        sims = torch.matmul(z, centers.t())  # [n, k]
        assign = sims.argmax(dim=1)
        new_centers = []
        for ci in range(k_eff):
            m = assign == ci
            if int(m.sum().item()) <= 0:
                # Reuse farthest sample from current centers as fallback.
                hard_idx = int(torch.argmin(sims.max(dim=1).values).item())
                new_centers.append(z[hard_idx])
                continue
            c = z[m].mean(dim=0, keepdim=False)
            new_centers.append(_safe_l2(c.unsqueeze(0)).squeeze(0))
        stacked = torch.stack(new_centers, dim=0)
        if torch.allclose(stacked, centers, atol=1e-4, rtol=1e-4):
            centers = stacked
            break
        centers = stacked

    sims = torch.matmul(z, centers.t())
    assign = sims.argmax(dim=1)
    conf = sims.gather(1, assign.view(-1, 1)).squeeze(1)
    if int(k_eff) < int(k):
        pad = torch.zeros(int(k) - int(k_eff), d, device=z.device, dtype=z.dtype)
        centers = torch.cat([centers, pad], dim=0)
    return centers, assign, conf


def _median_or_default(vals: Sequence[float], default: float = 0.0) -> float:
    if vals is None or len(vals) == 0:
        return float(default)
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def _empty_evidence() -> Dict[str, float]:
    return {
        "rms": 0.0,
        "peak_to_peak": 0.0,
        "kurtosis": 0.0,
        "crest_factor": 0.0,
        "dominant_freq": 0.0,
        "spectral_entropy": 0.0,
        "spectral_centroid": 0.0,
        "wavelet_entropy": 0.0,
        "impulse_factor": 0.0,
        "tf_concentration": 0.0,
        "tf_entropy": 0.0,
        "ridge_ratio": 0.0,
    }


def _fault_name(label_id: int) -> str:
    mapping = {
        0: "healthy bearing",
        1: "inner race fault",
        2: "outer race fault",
        3: "rolling element fault",
    }
    return mapping.get(int(label_id), f"fault class {int(label_id)}")


def _fault_location(label_id: int) -> str:
    mapping = {
        0: "global bearing path",
        1: "inner race contact zone",
        2: "fixed outer race contact",
        3: "rolling element and cage coupling",
    }
    return mapping.get(int(label_id), "unknown location")


def _severity_from_evidence(kurtosis: float, crest: float) -> str:
    if kurtosis >= 6.0 or crest >= 5.5:
        return "severe"
    if kurtosis >= 4.0 or crest >= 4.0:
        return "moderate"
    return "mild"


def _load_condition_from_rms(rms: float) -> str:
    if rms >= 1.00:
        return "high load / high energy"
    if rms >= 0.35:
        return "medium load"
    return "light load / weak excitation"


def _time_pattern(kurtosis: float, impulse_factor: float, p2p: float) -> str:
    if kurtosis > 6.0 and impulse_factor > 3.0:
        return "high-kurtosis strong impact train"
    if kurtosis > 4.0:
        return "moderate impulsive-modulated mixture"
    if p2p > 1.0:
        return "low-kurtosis but elevated peak-to-peak"
    return "near-stationary weak-impact background"


def _fault_specific_frequency_hint(label_id: int, dom_freq: float) -> str:
    if dom_freq <= 1e-6:
        if int(label_id) == 0:
            return "no stable BPFI/BPFO/BSF-like line"
        return "fault-frequency line is weak or diffuse"
    if int(label_id) == 1:
        return f"BPFI-like component around {dom_freq:.1f}Hz with shaft-speed sidebands"
    if int(label_id) == 2:
        return f"BPFO-like narrow-band line around {dom_freq:.1f}Hz and harmonics"
    if int(label_id) == 3:
        return f"BSF-like dispersed components around {dom_freq:.1f}Hz with cage modulation"
    return f"dominant component around {dom_freq:.1f}Hz"


def _freq_pattern(label_id: int, dom_freq: float, spec_entropy: float, spec_centroid: float) -> str:
    _hint = _fault_specific_frequency_hint(label_id, dom_freq)
    if dom_freq <= 1e-6:
        return _hint
    if spec_entropy >= 4.0:
        return f"{_hint}; broad-band energy spreading is obvious"
    if spec_centroid > dom_freq * 1.5:
        return f"{_hint}; with high-frequency bias in spectral centroid"
    return f"{_hint}; narrow-band stability is preserved"


def _tf_pattern(wavelet_entropy: float, tf_concentration: float, ridge_ratio: float) -> str:
    if tf_concentration >= 0.22 and ridge_ratio >= 2.0:
        return "highly localized burst ridges with concentrated energy"
    if wavelet_entropy >= 1.5:
        return "fragmented high-energy patches in time-frequency map"
    if wavelet_entropy >= 0.8:
        return "intermittent ridge continuity with moderate localization"
    return "relatively smooth time-frequency texture"


def _build_slot(
    label_id: int,
    evidence: Dict[str, float],
    confidence: float,
) -> Dict[str, Any]:
    rms = float(evidence.get("rms", 0.0))
    p2p = float(evidence.get("peak_to_peak", 0.0))
    kurt = float(evidence.get("kurtosis", 0.0))
    crest = float(evidence.get("crest_factor", 0.0))
    dom_f = float(evidence.get("dominant_freq", 0.0))
    spec_en = float(evidence.get("spectral_entropy", 0.0))
    spec_centroid = float(evidence.get("spectral_centroid", 0.0))
    wav_en = float(evidence.get("wavelet_entropy", 0.0))
    tf_conc = float(evidence.get("tf_concentration", 0.0))
    ridge_ratio = float(evidence.get("ridge_ratio", 0.0))
    impulse_factor = float(evidence.get("impulse_factor", 0.0))
    if impulse_factor <= 1e-8:
        impulse_factor = float(np.abs(evidence.get("max", 0.0)) / (np.mean(np.abs([rms])) + 1e-6))

    return {
        "fault_type": _fault_name(label_id),
        "location": _fault_location(label_id),
        "severity": _severity_from_evidence(kurt, crest),
        "load_condition": _load_condition_from_rms(rms),
        "time_domain_pattern": _time_pattern(kurt, impulse_factor, p2p),
        "frequency_pattern": _freq_pattern(label_id, dom_f, spec_en, spec_centroid),
        "time_frequency_pattern": _tf_pattern(wav_en, tf_conc, ridge_ratio),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "dominant_freq_hz": float(max(0.0, dom_f)),
        # Numeric evidence anchors (improve text diversity and reduce template homogeneity).
        "rms": float(rms),
        "kurtosis": float(kurt),
        "crest_factor": float(crest),
        "spectral_entropy": float(spec_en),
        "spectral_centroid_hz": float(spec_centroid),
        "tf_concentration": float(tf_conc),
        "ridge_ratio": float(ridge_ratio),
    }


def _render_template_variants(slot: Dict[str, Any]) -> List[str]:
    s = slot
    variants = []
    variants.append(
        f"This cluster indicates {s['fault_type']} located at {s['location']} with {s['severity']} severity. "
        f"Under {s['load_condition']}, time-domain behavior is {s['time_domain_pattern']}. "
        f"Frequency-domain evidence: {s['frequency_pattern']}. "
        f"Time-frequency evidence: {s['time_frequency_pattern']}. "
        f"Confidence={s['confidence']:.2f}."
    )
    variants.append(
        f"[fault_type]={s['fault_type']}; [location]={s['location']}; [severity]={s['severity']}; "
        f"[load_condition]={s['load_condition']}; [time_domain_pattern]={s['time_domain_pattern']}; "
        f"[frequency_pattern]={s['frequency_pattern']}; [time_frequency_pattern]={s['time_frequency_pattern']}; "
        f"[dominant_freq_hz]={float(s.get('dominant_freq_hz', 0.0)):.2f}; [confidence]={s['confidence']:.2f}; "
        f"[rms]={float(s.get('rms', 0.0)):.4f}; [kurtosis]={float(s.get('kurtosis', 0.0)):.3f}; "
        f"[crest_factor]={float(s.get('crest_factor', 0.0)):.3f}; [spec_entropy]={float(s.get('spectral_entropy', 0.0)):.3f}; "
        f"[spec_centroid_hz]={float(s.get('spectral_centroid_hz', 0.0)):.2f}; [tf_concentration]={float(s.get('tf_concentration', 0.0)):.3f}; "
        f"[ridge_ratio]={float(s.get('ridge_ratio', 0.0)):.3f}."
    )
    variants.append(
        f"Structured diagnosis: fault={s['fault_type']}, region={s['location']}, severity={s['severity']}, "
        f"load={s['load_condition']}, time={s['time_domain_pattern']}, "
        f"freq={s['frequency_pattern']}, tf={s['time_frequency_pattern']}, conf={s['confidence']:.2f}; "
        f"stats(rms={float(s.get('rms', 0.0)):.4f}, kurt={float(s.get('kurtosis', 0.0)):.3f}, "
        f"crest={float(s.get('crest_factor', 0.0)):.3f}, entropy={float(s.get('spectral_entropy', 0.0)):.3f}, "
        f"centroid={float(s.get('spectral_centroid_hz', 0.0)):.1f}Hz, tfc={float(s.get('tf_concentration', 0.0)):.3f}, "
        f"ridge={float(s.get('ridge_ratio', 0.0)):.3f})."
    )
    return variants


def _physics_plausibility(slot: Dict[str, Any], label_id: int) -> float:
    txt = (
        f"{slot.get('fault_type','')} {slot.get('time_domain_pattern','')} "
        f"{slot.get('frequency_pattern','')} {slot.get('time_frequency_pattern','')}"
    ).lower()
    score = 0.5
    if int(label_id) == 0:
        if ("stationary" in txt) or ("smooth" in txt) or ("no stable" in txt):
            score += 0.2
    elif int(label_id) == 1:
        if ("bpfi" in txt) or ("sideband" in txt) or ("modulated" in txt):
            score += 0.2
    elif int(label_id) == 2:
        if ("bpfo" in txt) or ("narrow-band" in txt) or ("stable" in txt):
            score += 0.2
    elif int(label_id) == 3:
        if ("bsf" in txt) or ("intermittent" in txt) or ("fragmented" in txt):
            score += 0.2
    if ("unknown" in txt) or ("nan" in txt):
        score -= 0.2
    return float(max(0.0, min(1.0, score)))


def _spec_freq_axis_hz(
    spec_freq_pos: Optional[np.ndarray],
    f_bins: int,
    fs: float,
    freq_reference_hz: float,
) -> np.ndarray:
    if spec_freq_pos is not None:
        pos = np.asarray(spec_freq_pos, dtype=np.float64)
        if pos.ndim == 3:
            pos = pos[0]
        if pos.ndim == 2 and pos.shape[0] == f_bins:
            axis_norm = np.nanmedian(pos, axis=1)
            axis_norm = np.nan_to_num(axis_norm, nan=0.0, posinf=0.0, neginf=0.0)
            axis_norm = np.clip(axis_norm, 0.0, 1.0)
            return axis_norm * float(max(1.0, freq_reference_hz))
    nyq = float(max(1.0, fs) * 0.5)
    return np.linspace(0.0, nyq, num=f_bins, dtype=np.float64)


def _extract_spec_evidence(
    specs: Optional[List[np.ndarray]],
    spec_freq_pos_list: Optional[List[np.ndarray]],
    fs_values: Optional[List[float]],
    freq_reference_hz: float = 32000.0,
    spec_posenc_scale: float = 0.0,
) -> Dict[str, float]:
    if specs is None or len(specs) == 0:
        return _empty_evidence()

    dom_vals: List[float] = []
    ent_vals: List[float] = []
    cent_vals: List[float] = []
    tf_wav_vals: List[float] = []
    tf_conc_vals: List[float] = []
    tf_entropy_vals: List[float] = []
    ridge_vals: List[float] = []

    for i, spec in enumerate(specs):
        arr = np.asarray(spec, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[1] <= 1 or arr.shape[2] <= 1:
            continue

        pos_i = None
        if spec_freq_pos_list is not None and i < len(spec_freq_pos_list):
            pos_i = np.asarray(spec_freq_pos_list[i], dtype=np.float64)
            if pos_i.ndim == 2:
                pos_i = pos_i[None, :, :]
            if pos_i.ndim != 3:
                pos_i = None

        # Recover pre-posenc spectrum for evidence to avoid leakage of injected position channel value.
        if pos_i is not None and arr.shape[-2:] == pos_i.shape[-2:]:
            arr = arr - float(spec_posenc_scale) * pos_i

        mag = np.abs(arr)
        # [F]
        freq_energy = np.nanmean(mag, axis=(0, 2))
        freq_energy = np.nan_to_num(freq_energy, nan=0.0, posinf=0.0, neginf=0.0)
        if float(freq_energy.sum()) <= 1e-12:
            continue

        fs_i = float(fs_values[i]) if (fs_values is not None and i < len(fs_values)) else 1.0
        freq_axis_hz = _spec_freq_axis_hz(
            spec_freq_pos=pos_i,
            f_bins=int(freq_energy.shape[0]),
            fs=fs_i,
            freq_reference_hz=float(freq_reference_hz),
        )

        p = freq_energy / (freq_energy.sum() + 1e-12)
        dom_idx = int(np.argmax(freq_energy))
        dom_vals.append(float(freq_axis_hz[min(dom_idx, len(freq_axis_hz) - 1)]))
        ent_vals.append(float(-np.sum(p * np.log(p + 1e-12))))
        cent_vals.append(float(np.sum(freq_axis_hz * p)))

        tf_energy = np.nanmean(mag, axis=0)  # [F, T]
        tf_energy = np.nan_to_num(tf_energy, nan=0.0, posinf=0.0, neginf=0.0)
        tf_flat = tf_energy.reshape(-1)
        if tf_flat.size > 8 and float(tf_flat.sum()) > 1e-12:
            q90 = float(np.quantile(tf_flat, 0.90))
            tf_conc_vals.append(float(np.mean(tf_flat >= q90)))
            tf_prob = tf_flat / (tf_flat.sum() + 1e-12)
            tf_entropy_vals.append(float(-np.sum(tf_prob * np.log(tf_prob + 1e-12))))
        else:
            tf_conc_vals.append(0.0)
            tf_entropy_vals.append(0.0)

        # ridge ratio: average per-time max over global mean (higher -> sharper ridges)
        max_per_t = np.max(tf_energy, axis=0)
        ridge_vals.append(float(np.mean(max_per_t) / (np.mean(tf_energy) + 1e-8)))

        # reuse wavelet_entropy slot with a robust tf entropy proxy
        tf_wav_vals.append(float(tf_entropy_vals[-1]))

    out = _empty_evidence()
    out["dominant_freq"] = _median_or_default(dom_vals, 0.0)
    out["spectral_entropy"] = _median_or_default(ent_vals, 0.0)
    out["spectral_centroid"] = _median_or_default(cent_vals, 0.0)
    out["wavelet_entropy"] = _median_or_default(tf_wav_vals, 0.0)
    out["tf_concentration"] = _median_or_default(tf_conc_vals, 0.0)
    out["tf_entropy"] = _median_or_default(tf_entropy_vals, 0.0)
    out["ridge_ratio"] = _median_or_default(ridge_vals, 0.0)
    return out


def _aggregate_cluster_evidence(
    signals: Optional[List[np.ndarray]],
    fs_values: Optional[List[float]],
    specs: Optional[List[np.ndarray]] = None,
    spec_freq_pos_list: Optional[List[np.ndarray]] = None,
    freq_reference_hz: float = 32000.0,
    spec_posenc_scale: float = 0.0,
    use_spec_evidence: bool = True,
) -> Dict[str, float]:
    out = _empty_evidence()

    rms_vals: List[float] = []
    p2p_vals: List[float] = []
    kurt_vals: List[float] = []
    crest_vals: List[float] = []
    dom_vals_signal: List[float] = []
    ent_vals_signal: List[float] = []
    cent_vals_signal: List[float] = []
    wav_vals_signal: List[float] = []
    imp_vals: List[float] = []

    if signals is not None and len(signals) > 0:
        for i, sig in enumerate(signals):
            fs_i = float(fs_values[i]) if (fs_values is not None and i < len(fs_values)) else 1.0
            try:
                feats = extract_diagnostic_features(np.asarray(sig, dtype=np.float32), max(1.0, fs_i))
            except Exception:
                continue
            rms_vals.append(float(feats.get("rms", 0.0)))
            p2p_vals.append(float(feats.get("peak_to_peak", 0.0)))
            kurt_vals.append(float(feats.get("kurtosis", 0.0)))
            crest_vals.append(float(feats.get("crest_factor", 0.0)))
            dom_vals_signal.append(float(feats.get("dominant_freq", 0.0)))
            ent_vals_signal.append(float(feats.get("spectral_entropy", 0.0)))
            cent_vals_signal.append(float(feats.get("spectral_centroid", 0.0)))
            wav_vals_signal.append(float(feats.get("wavelet_entropy", 0.0)))
            mean_abs = float(np.mean(np.abs(np.asarray(sig, dtype=np.float64))) + 1e-8)
            imp_vals.append(float(np.max(np.abs(np.asarray(sig, dtype=np.float64))) / mean_abs))

    out["rms"] = _median_or_default(rms_vals, 0.0)
    out["peak_to_peak"] = _median_or_default(p2p_vals, 0.0)
    out["kurtosis"] = _median_or_default(kurt_vals, 0.0)
    out["crest_factor"] = _median_or_default(crest_vals, 0.0)
    out["impulse_factor"] = _median_or_default(imp_vals, 0.0)

    spec_evd = _extract_spec_evidence(
        specs=specs,
        spec_freq_pos_list=spec_freq_pos_list,
        fs_values=fs_values,
        freq_reference_hz=float(freq_reference_hz),
        spec_posenc_scale=float(spec_posenc_scale),
    )

    if use_spec_evidence:
        out["dominant_freq"] = float(spec_evd.get("dominant_freq", 0.0))
        out["spectral_entropy"] = float(spec_evd.get("spectral_entropy", 0.0))
        out["spectral_centroid"] = float(spec_evd.get("spectral_centroid", 0.0))
        out["wavelet_entropy"] = float(spec_evd.get("wavelet_entropy", 0.0))
        out["tf_concentration"] = float(spec_evd.get("tf_concentration", 0.0))
        out["tf_entropy"] = float(spec_evd.get("tf_entropy", 0.0))
        out["ridge_ratio"] = float(spec_evd.get("ridge_ratio", 0.0))
    else:
        out["dominant_freq"] = _median_or_default(dom_vals_signal, 0.0)
        out["spectral_entropy"] = _median_or_default(ent_vals_signal, 0.0)
        out["spectral_centroid"] = _median_or_default(cent_vals_signal, 0.0)
        out["wavelet_entropy"] = _median_or_default(wav_vals_signal, 0.0)

    return out


def build_dynamic_semantic_cluster_bank(
    support_emb: torch.Tensor,
    support_labels: torch.Tensor,
    support_signals: Optional[List[np.ndarray]] = None,
    support_fs: Optional[List[float]] = None,
    support_specs: Optional[List[np.ndarray]] = None,
    support_spec_freq_pos: Optional[List[np.ndarray]] = None,
    num_classes: int = 4,
    k_per_class: int = 4,
    max_iters: int = 25,
    min_cluster_samples: int = 8,
    max_signal_samples_per_cluster: int = 48,
    freq_reference_hz: float = 32000.0,
    spec_posenc_scale: float = 0.0,
    use_spec_evidence: bool = True,
    text_embedder: Optional[Callable[[List[str]], torch.Tensor]] = None,
    text_mix_alpha: float = 0.35,
    text_mix_min_cos: float = -1.0,
    text_mix_fallback_alpha: float = 0.0,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    if support_emb is None or support_labels is None:
        return None
    if support_emb.dim() != 2 or support_emb.size(0) < 2:
        return None

    z = _safe_l2(support_emb)
    y = support_labels.long().view(-1)
    if int(y.numel()) != int(z.size(0)):
        return None

    entries: List[Dict[str, Any]] = []
    all_centers: List[torch.Tensor] = []
    all_labels: List[int] = []

    rng = np.random.default_rng(int(seed))

    for cls in range(int(num_classes)):
        cls_idx = torch.where(y == int(cls))[0]
        if int(cls_idx.numel()) <= 0:
            continue
        z_cls = z[cls_idx]
        n_cls = int(z_cls.size(0))
        k_eff = max(1, min(int(k_per_class), n_cls))

        centers, assign, conf = _cosine_kmeans(
            z_cls,
            k=k_eff,
            max_iters=max_iters,
            seed=seed + int(cls),
        )

        for ci in range(k_eff):
            m = assign == int(ci)
            n_ci = int(m.sum().item())
            if n_ci <= 0:
                continue
            # Merge too-small clusters into larger geometry by skipping tiny unstable clusters.
            if n_ci < int(min_cluster_samples) and n_cls >= int(min_cluster_samples):
                continue

            member_local = torch.where(m)[0]
            member_global = cls_idx[member_local]
            center = _safe_l2(centers[ci].view(1, -1)).squeeze(0)
            cons_val = float(conf[m].mean().detach().cpu().item())
            cov_ratio = float(n_ci / max(1, n_cls))

            pick_global = member_global.detach().cpu().numpy().tolist()
            if len(pick_global) > int(max_signal_samples_per_cluster):
                pick_global = rng.choice(
                    pick_global,
                    size=int(max_signal_samples_per_cluster),
                    replace=False,
                ).tolist()

            sig_subset = None
            fs_subset = None
            spec_subset = None
            spec_pos_subset = None
            if support_signals is not None and len(support_signals) == int(z.size(0)):
                sig_subset = [np.asarray(support_signals[int(ix)], dtype=np.float32) for ix in pick_global]
            if support_fs is not None and len(support_fs) == int(z.size(0)):
                fs_subset = [float(support_fs[int(ix)]) for ix in pick_global]
            if support_specs is not None and len(support_specs) == int(z.size(0)):
                spec_subset = [np.asarray(support_specs[int(ix)], dtype=np.float32) for ix in pick_global]
            if support_spec_freq_pos is not None and len(support_spec_freq_pos) == int(z.size(0)):
                spec_pos_subset = [np.asarray(support_spec_freq_pos[int(ix)], dtype=np.float32) for ix in pick_global]

            evidence = _aggregate_cluster_evidence(
                signals=sig_subset,
                fs_values=fs_subset,
                specs=spec_subset,
                spec_freq_pos_list=spec_pos_subset,
                freq_reference_hz=float(freq_reference_hz),
                spec_posenc_scale=float(spec_posenc_scale),
                use_spec_evidence=bool(use_spec_evidence),
            )
            slot = _build_slot(int(cls), evidence, confidence=max(0.0, min(1.0, (cons_val + 1.0) * 0.5)))
            candidates = _render_template_variants(slot)

            entries.append(
                {
                    "class_id": int(cls),
                    "cluster_id": int(ci),
                    "center": center.detach().cpu(),
                    "size": int(n_ci),
                    "class_size": int(n_cls),
                    "coverage": float(cov_ratio),
                    "consistency_cos": float(cons_val),
                    "slot": slot,
                    "evidence": evidence,
                    "candidates": candidates,
                }
            )
            all_centers.append(center.detach().cpu())
            all_labels.append(int(cls))

    if len(entries) == 0:
        return None

    center_visual = _safe_l2(torch.stack(all_centers, dim=0))
    label_vec = torch.tensor(all_labels, dtype=torch.long)

    proto_weights = []
    selected_texts = []
    selected_stats = []
    for i, entry in enumerate(entries):
        cls_i = int(entry["class_id"])
        c_i = center_visual[i : i + 1]
        mask_other = label_vec != cls_i
        if int(mask_other.sum().item()) > 0:
            max_cos_other = float(torch.matmul(c_i, center_visual[mask_other].t()).max().cpu().item())
        else:
            max_cos_other = 0.0
        sep_score = float(max(0.0, min(1.0, (1.0 - max_cos_other) * 0.5)))
        cons_score = float(max(0.0, min(1.0, (float(entry["consistency_cos"]) + 1.0) * 0.5)))
        phy_score = _physics_plausibility(entry["slot"], cls_i)
        cov_score = float(max(0.0, min(1.0, entry["coverage"])))

        best_text = ""
        best_score = -1.0
        for ci, cand in enumerate(entry["candidates"]):
            # Small tie-break bonus to preserve deterministic but varied templates.
            score = 0.45 * cons_score + 0.30 * sep_score + 0.15 * phy_score + 0.10 * cov_score + 1e-4 * ci
            if score > best_score:
                best_score = score
                best_text = cand

        weight = max(1e-4, cov_score * max(1e-4, cons_score))
        proto_weights.append(weight)
        selected_texts.append(best_text)
        selected_stats.append(
            {
                "class_id": cls_i,
                "cluster_id": int(entry["cluster_id"]),
                "size": int(entry["size"]),
                "class_size": int(entry["class_size"]),
                "coverage": float(cov_score),
                "consistency": float(cons_score),
                "separability": float(sep_score),
                "physics": float(phy_score),
                "template_score": float(best_score),
                "text": best_text,
                "slot": entry["slot"],
                "evidence": entry.get("evidence", {}),
                "dominant_freq_hz": float(entry.get("evidence", {}).get("dominant_freq", 0.0)),
            }
        )

    w = torch.tensor(proto_weights, dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-8)

    center_text = None
    center_final = center_visual
    text_mix_alpha = float(max(0.0, min(1.0, text_mix_alpha)))
    text_mix_min_cos = float(text_mix_min_cos)
    text_mix_fallback_alpha = float(max(0.0, min(1.0, text_mix_fallback_alpha)))
    text_mix_effective_alpha = float(text_mix_alpha)
    text_mix_fallback_triggered = False
    visual_text_cos_mean = float("nan")
    if callable(text_embedder) and len(selected_texts) == int(center_visual.size(0)):
        try:
            _txt = text_embedder(selected_texts)
            if isinstance(_txt, np.ndarray):
                _txt = torch.from_numpy(_txt)
            if torch.is_tensor(_txt):
                _txt = _txt.detach().cpu().float()
                if _txt.dim() == 2 and int(_txt.size(0)) == int(center_visual.size(0)):
                    center_text = _safe_l2(_txt)
                    if int(center_text.size(1)) == int(center_visual.size(1)):
                        center_final = _safe_l2(
                            (1.0 - text_mix_alpha) * center_visual + text_mix_alpha * center_text
                        )
        except Exception:
            center_text = None
            center_final = center_visual

    if center_text is not None and center_text.shape == center_visual.shape:
        tv_cos = torch.sum(center_visual * center_text, dim=1)
        visual_text_cos_mean = float(tv_cos.mean().item())
        if np.isfinite(text_mix_min_cos) and text_mix_min_cos > -1.0 and visual_text_cos_mean < text_mix_min_cos:
            text_mix_fallback_triggered = True
            text_mix_effective_alpha = float(text_mix_fallback_alpha)
            center_final = _safe_l2(
                (1.0 - text_mix_effective_alpha) * center_visual
                + text_mix_effective_alpha * center_text
            )
        for i in range(len(selected_stats)):
            selected_stats[i]["text_visual_cos"] = float(tv_cos[i].cpu().item())
            selected_stats[i]["text_mix_alpha_effective"] = float(text_mix_effective_alpha)
            selected_stats[i]["text_mix_fallback"] = int(text_mix_fallback_triggered)
    else:
        for i in range(len(selected_stats)):
            selected_stats[i]["text_visual_cos"] = float("nan")
            selected_stats[i]["text_mix_alpha_effective"] = float(0.0)
            selected_stats[i]["text_mix_fallback"] = 0

    return {
        "centers": center_final.detach().cpu(),       # [M, D], used by predictor
        "centers_visual": center_visual.detach().cpu(),
        "centers_text": center_text.detach().cpu() if torch.is_tensor(center_text) else None,
        "labels": label_vec.detach().cpu(),           # [M]
        "weights": w.detach().cpu(),                  # [M]
        "texts": selected_texts,                      # len M
        "stats": selected_stats,                      # len M
        "num_clusters": int(center_final.size(0)),
        "text_mix_alpha": float(text_mix_alpha),
        "text_mix_alpha_effective": float(text_mix_effective_alpha),
        "text_mix_fallback_triggered": bool(text_mix_fallback_triggered),
        "visual_text_cos_mean": float(visual_text_cos_mean),
    }


def predict_with_dynamic_cluster_bank(
    query_emb: torch.Tensor,
    cluster_bank: Dict[str, Any],
    num_classes: int = 4,
    topk_vote: int = 7,
    temperature: float = 0.07,
) -> Optional[Dict[str, Any]]:
    if query_emb is None or cluster_bank is None:
        return None
    if query_emb.dim() != 2 or query_emb.size(0) <= 0:
        return None

    centers = cluster_bank.get("centers", None)
    labels = cluster_bank.get("labels", None)
    weights = cluster_bank.get("weights", None)
    if not torch.is_tensor(centers) or not torch.is_tensor(labels) or not torch.is_tensor(weights):
        return None
    if centers.dim() != 2 or labels.dim() != 1 or int(centers.size(0)) <= 0:
        return None

    device = query_emb.device
    q = _safe_l2(query_emb.float())
    c = _safe_l2(centers.to(device).float())
    lbl = labels.to(device).long().view(-1)
    w = weights.to(device).float().view(-1)

    sims = torch.matmul(q, c.t()) / max(1e-6, float(temperature))  # [N, M]
    k = max(1, min(int(topk_vote), int(c.size(0))))
    topv, topi = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
    attn = torch.softmax(topv, dim=1)  # [N, k]
    pick_cls = lbl[topi]               # [N, k]
    pick_w = w[topi]                   # [N, k]
    contrib = attn * pick_w

    cls_prob = torch.zeros(q.size(0), int(num_classes), device=device, dtype=q.dtype)
    cls_prob.scatter_add_(1, pick_cls, contrib)
    cls_prob = cls_prob / cls_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    preds = cls_prob.argmax(dim=1)

    return {
        "probs": cls_prob,
        "preds": preds,
        "topk_indices": topi,
        "topk_scores": topv,
    }


def summarize_cluster_bank(cluster_bank: Dict[str, Any], max_clusters: int = 10) -> List[str]:
    lines: List[str] = []
    if cluster_bank is None:
        return lines
    stats = cluster_bank.get("stats", [])
    if not isinstance(stats, list):
        return lines
    max_show = max(1, int(max_clusters))
    for row in stats[:max_show]:
        _dom_f = float(row.get("dominant_freq_hz", 0.0))
        _tv_cos = row.get("text_visual_cos", float("nan"))
        lines.append(
            "[DynMultiProtoBank] "
            f"class={int(row.get('class_id', -1))} "
            f"cluster={int(row.get('cluster_id', -1))} "
            f"size={int(row.get('size', 0))}/{int(row.get('class_size', 0))} "
            f"cov={float(row.get('coverage', 0.0)):.3f} "
            f"cons={float(row.get('consistency', 0.0)):.3f} "
            f"sep={float(row.get('separability', 0.0)):.3f} "
            f"phys={float(row.get('physics', 0.0)):.3f} "
            f"score={float(row.get('template_score', 0.0)):.3f} "
            f"domf={_dom_f:.2f}Hz "
            f"tv_cos={float(_tv_cos):.3f} "
            f"text={str(row.get('text', ''))}"
        )
    return lines
