"""
T2MFDF-inspired sample-level signal feature → structured text generation.

Reference: T2MFDF: A LLM-Enhanced Multimodal Fault Diagnosis Framework
           Integrating Time-Series and Textual Data (IEEE TIM 2025, Zhou et al.)

Converts raw vibration signals into structured diagnostic text descriptions
that can be encoded by the text encoder for sample-level semantic features.
"""

import numpy as np
import warnings
from typing import Dict, Optional


def extract_diagnostic_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract time-domain, frequency-domain, envelope spectrum, and time-frequency
    diagnostic features from a raw vibration signal.

    ATT431: Enhanced with physics-informed envelope spectrum analysis for
    inner/outer/rolling fault discrimination (T2MFDF-inspired).

    Args:
        signal: 1D numpy array of vibration signal values
        fs: sampling frequency in Hz

    Returns:
        dict with named diagnostic features
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    if n < 4:
        return _empty_features()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # ---- Time-domain ----
        rms = float(np.sqrt(np.mean(signal ** 2)))
        std_val = float(np.std(signal))
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        median_val = float(np.median(signal))
        peak_to_peak = max_val - min_val
        crest_factor = float(max_val / (rms + 1e-12))

        from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
        kurt = float(sp_kurtosis(signal, fisher=True))
        skewness = float(sp_skew(signal))

        trend = "upward" if signal[-1] > signal[0] else "downward"

        # ---- Frequency-domain ----
        from scipy.fft import rfft, rfftfreq
        fft_vals = rfft(signal)
        fft_mag = np.abs(fft_vals)
        freqs = rfftfreq(n, 1.0 / fs)

        fft_mag_pos = fft_mag[1:]
        freqs_pos = freqs[1:]
        if len(fft_mag_pos) > 0 and np.sum(fft_mag_pos) > 1e-12:
            dominant_freq_idx = np.argmax(fft_mag_pos)
            dominant_freq = float(freqs_pos[dominant_freq_idx])

            spec_norm = fft_mag_pos / (np.sum(fft_mag_pos) + 1e-12)
            spectral_entropy = float(-np.sum(spec_norm * np.log(spec_norm + 1e-12)))
            spectral_centroid = float(np.sum(freqs_pos * spec_norm))

            sq_dev = (freqs_pos - spectral_centroid) ** 2
            spectral_bandwidth = float(np.sqrt(np.sum(sq_dev * spec_norm)))
        else:
            dominant_freq = 0.0
            spectral_entropy = 0.0
            spectral_centroid = 0.0
            spectral_bandwidth = 0.0

        # ---- Frequency band energy distribution ----
        band_ratios = _compute_band_energy_ratios(fft_mag_pos, freqs_pos, fs)

        # ---- Envelope spectrum analysis (physics-informed) ----
        env_features = _extract_envelope_spectrum_features(signal, fs)

        # ---- Time-frequency (wavelet entropy) ----
        wavelet_entropy = _compute_wavelet_entropy(signal)

    features = {
        "rms": _safe_float(rms),
        "std": _safe_float(std_val),
        "min": _safe_float(min_val),
        "max": _safe_float(max_val),
        "median": _safe_float(median_val),
        "peak_to_peak": _safe_float(peak_to_peak),
        "crest_factor": _safe_float(crest_factor),
        "kurtosis": _safe_float(kurt),
        "skewness": _safe_float(skewness),
        "trend": trend,
        "dominant_freq": _safe_float(dominant_freq),
        "spectral_entropy": _safe_float(spectral_entropy),
        "spectral_centroid": _safe_float(spectral_centroid),
        "spectral_bandwidth": _safe_float(spectral_bandwidth),
        "wavelet_entropy": _safe_float(wavelet_entropy),
        **band_ratios,
        **env_features,
    }
    return features


def features_to_diagnostic_text(features: Dict, fs: Optional[float] = None) -> str:
    """
    Convert extracted diagnostic features into a structured text description
    suitable for LLM/text encoder input.

    ATT431: Enhanced with physics-informed envelope spectrum and band energy
    descriptions for discriminative fault-type separation.
    """
    parts = [
        f"Bearing vibration signal statistics:",
        f"RMS amplitude {features['rms']:.4f},",
        f"kurtosis {features['kurtosis']:.2f},",
        f"crest factor {features['crest_factor']:.2f},",
        f"skewness {features['skewness']:.2f},",
        f"peak-to-peak {features['peak_to_peak']:.4f}.",
    ]

    bw = features.get("spectral_bandwidth", 0.0)
    parts.append(
        f"Frequency analysis:"
        f" dominant frequency {features['dominant_freq']:.1f} Hz,"
        f" spectral entropy {features['spectral_entropy']:.3f},"
        f" spectral centroid {features['spectral_centroid']:.1f} Hz,"
        f" bandwidth {bw:.1f} Hz."
    )

    # Band energy ratios
    low_r = features.get("band_low_ratio", 0.0)
    mid_r = features.get("band_mid_ratio", 0.0)
    high_r = features.get("band_high_ratio", 0.0)
    if isinstance(low_r, (int, float)):
        parts.append(
            f"Band energy: low {low_r:.2f}, mid {mid_r:.2f}, high {high_r:.2f}."
        )

    # Envelope spectrum (physics-informed fault indicators)
    env_dom = features.get("env_dominant_freq", 0.0)
    env_kurt = features.get("env_kurtosis", 0.0)
    env_sb = features.get("env_sideband_ratio", 0.0)
    env_harm = features.get("env_harmonic_strength", 0.0)
    env_mod = features.get("env_modulation_index", 0.0)
    parts.append(
        f"Envelope spectrum: dominant {env_dom:.1f} Hz,"
        f" kurtosis {env_kurt:.2f},"
        f" sideband ratio {env_sb:.3f},"
        f" harmonic strength {env_harm:.3f},"
        f" modulation index {env_mod:.3f}."
    )

    parts.append(
        f"Time-frequency: wavelet entropy {features['wavelet_entropy']:.3f}."
    )

    # Physics-informed diagnostic hints
    kurt_val = features.get("kurtosis", 0.0)
    se_val = features.get("spectral_entropy", 0.0)

    if isinstance(kurt_val, (int, float)) and kurt_val > 5.0:
        parts.append("High kurtosis indicates impulsive fault-like transients.")
    elif isinstance(kurt_val, (int, float)) and kurt_val < 2.0:
        parts.append("Low kurtosis suggests smooth, non-impulsive vibration.")

    if isinstance(se_val, (int, float)) and se_val > 4.0:
        parts.append("High spectral entropy indicates broadband energy distribution.")
    elif isinstance(se_val, (int, float)) and se_val < 2.0:
        parts.append("Low spectral entropy indicates concentrated narrowband energy.")

    if isinstance(env_kurt, (int, float)) and isinstance(env_sb, (int, float)):
        if env_kurt > 4.0 and env_sb > 0.3:
            parts.append(
                "High envelope kurtosis with prominent sidebands suggests "
                "rotating-zone defect with amplitude modulation (inner race characteristic)."
            )
        elif env_kurt < 2.0 and env_sb < 0.15:
            parts.append(
                "Low envelope kurtosis without sidebands suggests "
                "fixed-zone impact or healthy bearing."
            )

    if isinstance(env_harm, (int, float)) and env_harm > 0.5:
        parts.append(
            "Strong harmonic peaks in envelope spectrum indicate "
            "periodic fault impulses at characteristic frequency."
        )

    return " ".join(parts)


def signal_to_text(signal: np.ndarray, fs: float) -> str:
    """
    End-to-end: raw signal → diagnostic text description.
    """
    features = extract_diagnostic_features(signal, fs)
    return features_to_diagnostic_text(features, fs)


def batch_signals_to_texts(signals, fs_values) -> list:
    """
    Batch conversion of signals to diagnostic texts.

    Args:
        signals: list of 1D numpy arrays
        fs_values: single float or list of floats (per-sample sampling rates)

    Returns:
        list of diagnostic text strings
    """
    if isinstance(fs_values, (int, float)):
        fs_values = [float(fs_values)] * len(signals)

    texts = []
    for sig, fs in zip(signals, fs_values):
        try:
            texts.append(signal_to_text(np.asarray(sig), float(fs)))
        except Exception:
            texts.append("Bearing vibration signal: feature extraction failed.")
    return texts


def _compute_wavelet_entropy(signal: np.ndarray) -> float:
    """Compute wavelet entropy using discrete wavelet decomposition."""
    try:
        import pywt
        max_level = pywt.dwt_max_level(len(signal), "db4")
        level = min(6, max(1, max_level))
        coeffs = pywt.wavedec(signal, "db4", level=level)
        energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
        total_e = np.sum(energies)
        if total_e < 1e-12:
            return 0.0
        probs = energies / total_e
        probs = probs[probs > 1e-12]
        return float(-np.sum(probs * np.log(probs)))
    except Exception:
        return 0.0


def _safe_float(v) -> float:
    """Convert to float, replacing NaN/Inf with 0."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _extract_envelope_features(signal: np.ndarray, fs: float) -> tuple:
    """
    ATT379b-S2: Physics-informed envelope analysis features for Inner vs Outer discrimination.

    Inner race: impact zone rotates with shaft → amplitude modulation → high envelope kurtosis + sidebands.
    Outer race: impact zone fixed → no AM → lower envelope kurtosis, no sidebands.

    Returns: (envelope_kurtosis, envelope_sc_norm, envelope_sideband_ratio)
    """
    try:
        from scipy.signal import hilbert
        from scipy.stats import kurtosis as sp_kurtosis

        analytic = hilbert(signal)
        envelope = np.abs(analytic)

        env_kurt = float(sp_kurtosis(envelope, fisher=True))

        from scipy.fft import rfft, rfftfreq
        env_fft = np.abs(rfft(envelope - np.mean(envelope)))
        env_freqs = rfftfreq(len(envelope), 1.0 / fs)

        env_pos = env_fft[1:]
        env_freq_pos = env_freqs[1:]
        env_total = float(np.sum(env_pos))

        if env_total > 1e-12 and len(env_pos) > 1:
            env_norm = env_pos / (env_total + 1e-12)
            env_sc = float(np.sum(env_freq_pos * env_norm))
            env_sc_norm = env_sc / max(1.0, float(fs))

            sorted_idx = np.argsort(env_pos)[::-1]
            peak1 = float(env_pos[sorted_idx[0]])
            peak2 = float(env_pos[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            sideband_ratio = peak2 / (peak1 + 1e-12)
        else:
            env_sc_norm = 0.0
            sideband_ratio = 0.0

        return _safe_float(env_kurt), _safe_float(env_sc_norm), _safe_float(sideband_ratio)
    except Exception:
        return 0.0, 0.0, 0.0


def extract_stat_vector(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    ATT377-StatFiLM + ATT379b-S2: extract 11-dim domain-invariant stat vector.
    Returns: np.float32 array of shape [11]:
      [rms, kurtosis, crest_factor, skewness,
       spectral_entropy, spectral_centroid_norm, dominant_freq_norm, wavelet_entropy,
       envelope_kurtosis, envelope_sc_norm, envelope_sideband_ratio]
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    zero_vec = np.zeros(11, dtype=np.float32)
    if n < 4:
        return zero_vec
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rms = float(np.sqrt(np.mean(signal ** 2)))
            from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
            kurt = float(sp_kurtosis(signal, fisher=True))
            crest = float(np.max(np.abs(signal)) / (rms + 1e-12))
            skew = float(sp_skew(signal))

            from scipy.fft import rfft, rfftfreq
            fft_mag = np.abs(rfft(signal))
            freqs = rfftfreq(n, 1.0 / fs)
            fft_pos = fft_mag[1:]
            freq_pos = freqs[1:]
            total_e = float(np.sum(fft_pos))
            if total_e > 1e-12 and len(fft_pos) > 0:
                spec_norm = fft_pos / (total_e + 1e-12)
                se = float(-np.sum(spec_norm * np.log(spec_norm + 1e-12)))
                sc = float(np.sum(freq_pos * spec_norm))
                df = float(freq_pos[np.argmax(fft_pos)])
            else:
                se, sc, df = 0.0, 0.0, 0.0
            fs_f = max(1.0, float(fs))
            sc_norm = sc / fs_f
            df_norm = df / fs_f
            we = _compute_wavelet_entropy(signal)

            env_kurt, env_sc_norm, env_sb_ratio = _extract_envelope_features(signal, fs)

        vec = np.array([
            _safe_float(rms), _safe_float(kurt), _safe_float(crest), _safe_float(skew),
            _safe_float(se), _safe_float(sc_norm), _safe_float(df_norm), _safe_float(we),
            _safe_float(env_kurt), _safe_float(env_sc_norm), _safe_float(env_sb_ratio),
        ], dtype=np.float32)
        return vec
    except Exception:
        return zero_vec


def _compute_band_energy_ratios(
    fft_mag_pos: np.ndarray, freqs_pos: np.ndarray, fs: float
) -> Dict[str, float]:
    """Compute low/mid/high band energy ratios (normalised by Nyquist)."""
    nyquist = fs / 2.0
    if len(fft_mag_pos) == 0 or nyquist < 1.0:
        return {"band_low_ratio": 0.0, "band_mid_ratio": 0.0, "band_high_ratio": 0.0}
    energy = fft_mag_pos ** 2
    total_e = float(np.sum(energy))
    if total_e < 1e-12:
        return {"band_low_ratio": 0.0, "band_mid_ratio": 0.0, "band_high_ratio": 0.0}
    low_mask = freqs_pos < 0.2 * nyquist
    mid_mask = (freqs_pos >= 0.2 * nyquist) & (freqs_pos < 0.6 * nyquist)
    high_mask = freqs_pos >= 0.6 * nyquist
    return {
        "band_low_ratio": _safe_float(np.sum(energy[low_mask]) / total_e),
        "band_mid_ratio": _safe_float(np.sum(energy[mid_mask]) / total_e),
        "band_high_ratio": _safe_float(np.sum(energy[high_mask]) / total_e),
    }


def _extract_envelope_spectrum_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    ATT431: Physics-informed envelope spectrum features for fault-type discrimination.
    
    Inner race: rotating impact zone → amplitude modulation → high env_kurtosis,
                prominent sidebands around characteristic frequency.
    Outer race: fixed impact zone → no AM → low env_kurtosis, no sidebands.
    Rolling element: double-impact per rotation → characteristic at 2× cage frequency.
    """
    default = {
        "env_kurtosis": 0.0,
        "env_dominant_freq": 0.0,
        "env_sideband_ratio": 0.0,
        "env_harmonic_strength": 0.0,
        "env_modulation_index": 0.0,
    }
    try:
        from scipy.signal import hilbert
        from scipy.stats import kurtosis as sp_kurtosis
        from scipy.fft import rfft, rfftfreq

        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        env_mean = float(np.mean(envelope))
        env_std = float(np.std(envelope))

        env_kurt = float(sp_kurtosis(envelope, fisher=True))
        modulation_index = env_std / (env_mean + 1e-12)

        env_ac = envelope - env_mean
        env_fft = np.abs(rfft(env_ac))
        env_freqs = rfftfreq(len(envelope), 1.0 / fs)

        env_pos = env_fft[1:]
        env_freq_pos = env_freqs[1:]
        env_total = float(np.sum(env_pos))

        if env_total < 1e-12 or len(env_pos) < 3:
            return default

        peak_idx = np.argmax(env_pos)
        env_dominant_freq = float(env_freq_pos[peak_idx])
        peak_val = float(env_pos[peak_idx])

        sorted_idx = np.argsort(env_pos)[::-1]
        second_peak = float(env_pos[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        sideband_ratio = second_peak / (peak_val + 1e-12)

        fund = env_dominant_freq
        harm_strength = 0.0
        if fund > 1.0:
            freq_res = float(env_freq_pos[1] - env_freq_pos[0]) if len(env_freq_pos) > 1 else 1.0
            tol = max(2.0, 1.5 * freq_res)
            n_harmonics = 0
            harm_energy = 0.0
            for h in range(2, 5):
                target = fund * h
                if target > env_freq_pos[-1]:
                    break
                mask = np.abs(env_freq_pos - target) < tol
                if np.any(mask):
                    harm_energy += float(np.max(env_pos[mask]))
                    n_harmonics += 1
            if n_harmonics > 0:
                harm_strength = harm_energy / (peak_val * n_harmonics + 1e-12)

        return {
            "env_kurtosis": _safe_float(env_kurt),
            "env_dominant_freq": _safe_float(env_dominant_freq),
            "env_sideband_ratio": _safe_float(sideband_ratio),
            "env_harmonic_strength": _safe_float(harm_strength),
            "env_modulation_index": _safe_float(modulation_index),
        }
    except Exception:
        return default


def _empty_features() -> Dict[str, float]:
    """Return default features for too-short signals."""
    return {
        "rms": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
        "median": 0.0, "peak_to_peak": 0.0, "crest_factor": 0.0,
        "kurtosis": 0.0, "skewness": 0.0, "trend": "unknown",
        "dominant_freq": 0.0, "spectral_entropy": 0.0,
        "spectral_centroid": 0.0, "spectral_bandwidth": 0.0,
        "wavelet_entropy": 0.0,
        "band_low_ratio": 0.0, "band_mid_ratio": 0.0, "band_high_ratio": 0.0,
        "env_kurtosis": 0.0, "env_dominant_freq": 0.0,
        "env_sideband_ratio": 0.0, "env_harmonic_strength": 0.0,
        "env_modulation_index": 0.0,
    }
