# utils/stft_dataset.py (replace/append these functions)
import numpy as np
from scipy.signal import stft, sosfiltfilt, butter, hilbert, welch
import torch
import torch.nn.functional as torch_fn

# Optional dependency (recommended for SWT-based spec)
try:
    import pywt  # type: ignore
except Exception:
    pywt = None

# -------------------------
# Helper: add AWGN to signal
# -------------------------
def add_awgn(signal: np.ndarray, snr_db: float):
    """
    Additive White Gaussian Noise to achieve target SNR (dB).
    signal: 1D numpy array
    snr_db: desired SNR in dB (signal_power / noise_power)
    """
    # signal power
    sig_power = np.mean(signal.astype(np.float64) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    # required noise power
    noise_power = sig_power / (snr_linear + 1e-12)
    # gaussian noise
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return (signal + noise).astype(signal.dtype)


# -------------------------
# SpecAugment-like masks
# -------------------------
def freq_mask(spec: np.ndarray, F_mask_max: int = 15, num_masks: int = 1):
    """
    spec: [F, T]
    Masks up to num_masks frequency bands (set to 0)
    """
    F, T = spec.shape
    for _ in range(num_masks):
        f = np.random.randint(0, F_mask_max + 1)
        if f == 0:
            continue
        f0 = np.random.randint(0, max(1, F - f) + 1)
        spec[f0:f0 + f, :] = 0.0
    return spec

def time_mask(spec: np.ndarray, T_mask_max: int = 25, num_masks: int = 1):
    """
    spec: [F, T]
    Masks up to num_masks time bands (set to 0)
    """
    F, T = spec.shape
    for _ in range(num_masks):
        t = np.random.randint(0, T_mask_max + 1)
        if t == 0:
            continue
        t0 = np.random.randint(0, max(1, T - t) + 1)
        spec[:, t0:t0 + t] = 0.0
    return spec

def spec_augment(spec: np.ndarray,
                 freq_mask_param: int = 12,
                 time_mask_param: int = 20,
                 num_freq_masks: int = 1,
                 num_time_masks: int = 1):
    """
    Apply simple SpecAugment (in-place) on a single [F, T] spec numpy array.
    """
    spec = spec.copy()
    if num_freq_masks > 0:
        spec = freq_mask(spec, F_mask_max=freq_mask_param, num_masks=num_freq_masks)
    if num_time_masks > 0:
        spec = time_mask(spec, T_mask_max=time_mask_param, num_masks=num_time_masks)
    return spec


# -------------------------
# Existing utilities (kept, with safety fixes)
# -------------------------
def spectral_energy_distribution(
    signal,
    fs,
    nperseg=1024,
    normalize=True
):
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        scaling="density"
    )

    energy = psd * np.diff(freqs, prepend=0)

    if normalize:
        energy = energy / (energy.sum() + 1e-12)

    return freqs, energy


def energy_based_bands(
    freqs,
    energy,
    num_bands=3,
    f_min=1.0,
    min_bandwidth=5.0
):
    energy = np.maximum(energy, 1e-12)
    cum_energy = np.cumsum(energy)
    cum_energy = cum_energy / cum_energy[-1]

    targets = np.linspace(0.0, 1.0, num_bands + 1)

    bands = []
    F = len(freqs)
    nyquist = freqs[-1] if len(freqs) > 0 else 0.0

    for i in range(num_bands):
        idx_low = np.searchsorted(cum_energy, targets[i])
        idx_high = np.searchsorted(cum_energy, targets[i + 1])

        idx_low = min(idx_low, F - 1)
        idx_high = min(idx_high, F - 1)

        if idx_high <= idx_low:
            idx_high = min(idx_low + 1, F - 1)

        f_low = max(freqs[idx_low], f_min)
        f_high = freqs[idx_high]

        if f_high - f_low < min_bandwidth:
            f_high = min(f_low + min_bandwidth, nyquist)

        if f_high <= f_low:
            # skip invalid band
            continue

        bands.append((float(f_low), float(f_high)))

    # ensure we have at least one band
    if len(bands) == 0 and F > 0:
        bands = [(float(freqs[0]), float(freqs[-1]))]

    return bands


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs

    lowcut = max(lowcut, 1e-3)
    highcut = min(highcut, nyq - 1e-3)

    if highcut <= lowcut:
        raise ValueError(f"Invalid band: lowcut={lowcut}, highcut={highcut}")

    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(
        order,
        [low, high],
        btype='band',
        output='sos'
    )
    return sosfiltfilt(sos, signal)


# -------------------------
# Modified band_hilbert_stft:
#   supports adding AWGN before filtering and SpecAugment afterwards
# -------------------------
def band_hilbert_stft(signal, fs, band, n_fft,
                      add_noise=False,
                      snr_db_range=(20.0, 40.0),
                      apply_specaug=False,
                      specaug_params=None):
    """
    signal: 1D numpy array
    band: (lowcut, highcut)
    n_fft: int
    add_noise: whether to add AWGN (random SNR in snr_db_range)
    apply_specaug: whether to apply spec_augment on resulting [F, T]
    specaug_params: dict with keys freq_mask_param, time_mask_param, etc.
    """
    if add_noise:
        snr_db = float(np.random.uniform(snr_db_range[0], snr_db_range[1]))
        signal = add_awgn(signal, snr_db)

    lowcut, highcut = band
    filtered = bandpass_filter(signal, fs, lowcut, highcut)

    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    f, t, Zxx = stft(
        envelope,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft // 2,
        boundary=None
    )

    spec = np.log1p(np.abs(Zxx)).astype(np.float32)  # [F, T]

    if apply_specaug:
        params = specaug_params or {}
        spec = spec_augment(
            spec,
            freq_mask_param=params.get("freq_mask_param", 12),
            time_mask_param=params.get("time_mask_param", 20),
            num_freq_masks=params.get("num_freq_masks", 1),
            num_time_masks=params.get("num_time_masks", 1)
        ).astype(np.float32)

    return spec


# -------------------------
# Pad specs to same shape (keeps shape consistent for stacking)
# -------------------------
def pad_specs_to_max(specs):
    """
    specs: list of np.ndarray, each [F_i, T_i]
    return: np.ndarray [C, F_max, T_max]
    优化版本：预分配数组，避免多次stack
    """
    if len(specs) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)

    # 一次性获取所有shape，避免多次遍历
    shapes = [s.shape for s in specs]
    F_max = max(shape[0] for shape in shapes)
    T_max = max(shape[1] for shape in shapes)

    # 预分配输出数组
    C = len(specs)
    padded = np.zeros((C, F_max, T_max), dtype=np.float32)
    
    # 直接赋值，避免多次pad和stack
    for i, s in enumerate(specs):
        F, T = s.shape
        padded[i, :F, :T] = s

    return padded


# -------------------------
# Modified multi-channel spec generator (with augmentation options)
# -------------------------
def multi_channel_spec_energy_adaptive(signal, fs, bands=None,
                                       num_bands=3,
                                       n_ffts=(256, 512),
                                       augment=False,
                                       noise_prob=0.5,
                                       snr_db_range=(20.0, 34.0),
                                       specaug_prob=0.5,
                                       specaug_params=None):
    """
    Generate multiple channel spectrograms per sample.
    - signal: 1D numpy array
    - fs: sample rate
    - bands: optional precomputed bands list
    - num_bands: if bands is None, compute this many bands
    - n_ffts: iterable of n_fft sizes to compute per band
    - augment: whether to perform augmentations (probabilistic)
    - noise_prob: probability to add AWGN to the raw signal (per band call)
    - snr_db_range: SNR range used when adding AWGN
    - specaug_prob: probability to apply SpecAugment to each produced spec
    - specaug_params: dict forwarded to spec_augment
    """
    if bands is None:
        freqs, energy = spectral_energy_distribution(signal, fs)
        bands = energy_based_bands(freqs, energy, num_bands=num_bands)

    specs = []
    for band in bands:
        for n_fft in n_ffts:
            do_noise = augment and (np.random.rand() < noise_prob)
            do_specaug = augment and (np.random.rand() < specaug_prob)
            spec = band_hilbert_stft(
                signal,
                fs,
                band,
                n_fft,
                add_noise=do_noise,
                snr_db_range=snr_db_range,
                apply_specaug=do_specaug,
                specaug_params=specaug_params
            )
            specs.append(spec)

    # pad to same shapes and stack: [C, F_max, T_max]
    specs = pad_specs_to_max(specs)
    return specs


# ============================================================================
# SWT-based multi-channel spec (band-hilbert-SWT)
# Keeps MoE layout compatibility: 3 bands × 2 channels = 6 channels.
# Output: [C=6, F=levels, T=time_bins]
# ============================================================================
def _downsample_time(mat: np.ndarray, t_bins: int) -> np.ndarray:
    """
    mat: [F, T] -> [F, t_bins] via linear interpolation on time axis.
    """
    F, T = mat.shape
    if t_bins is None or int(t_bins) <= 0 or T == int(t_bins):
        return mat.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, int(t_bins), dtype=np.float64)
    out = np.empty((F, int(t_bins)), dtype=np.float32)
    for f in range(F):
        out[f] = np.interp(x_new, x_old, mat[f].astype(np.float64)).astype(np.float32)
    return out


def _swt_stack(x: np.ndarray, wavelet: str = "db4", levels: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (cA_stack, cD_stack) as [levels, T].
    Pads x so len(x) is divisible by 2**levels (required by SWT).
    """
    if pywt is None:
        raise ImportError("pywt (PyWavelets) is required for SWT spec. Please install 'PyWavelets'.")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if int(levels) <= 0:
        raise ValueError("levels must be positive")

    m = 2 ** int(levels)
    n = x.shape[0]
    n_pad = int(np.ceil(n / m) * m)
    if n_pad != n:
        x = np.pad(x, (0, n_pad - n), mode="reflect")

    coeffs = pywt.swt(x, wavelet=str(wavelet), level=int(levels), trim_approx=False, norm=True)
    cA = np.stack([c[0] for c in coeffs], axis=0)  # [levels, T_pad]
    cD = np.stack([c[1] for c in coeffs], axis=0)  # [levels, T_pad]
    cA = cA[:, :n]
    cD = cD[:, :n]
    return cA.astype(np.float32), cD.astype(np.float32)


def band_hilbert_swt(
    signal,
    fs,
    band,
    wavelet="db4",
    levels=6,
    time_bins=128,
    add_noise=False,
    snr_db_range=(20.0, 40.0),
    apply_specaug=False,
    specaug_params=None,
):
    """
    Returns two channels (approx/detail) for the given band:
      - cA: [levels, time_bins]
      - cD: [levels, time_bins]
    """
    if add_noise:
        snr_db = float(np.random.uniform(snr_db_range[0], snr_db_range[1]))
        signal = add_awgn(signal, snr_db)

    lowcut, highcut = band
    filtered = bandpass_filter(signal, fs, lowcut, highcut)
    envelope = np.abs(hilbert(filtered))

    cA, cD = _swt_stack(envelope, wavelet=str(wavelet), levels=int(levels))
    cA = np.log1p(np.abs(cA))
    cD = np.log1p(np.abs(cD))
    cA = _downsample_time(cA, int(time_bins))
    cD = _downsample_time(cD, int(time_bins))

    if apply_specaug:
        params = specaug_params or {}
        # SWT has small F; keep masks conservative by default
        cA = spec_augment(
            cA,
            freq_mask_param=params.get("freq_mask_param", 2),
            time_mask_param=params.get("time_mask_param", 12),
            num_freq_masks=params.get("num_freq_masks", 1),
            num_time_masks=params.get("num_time_masks", 1),
        ).astype(np.float32)
        cD = spec_augment(
            cD,
            freq_mask_param=params.get("freq_mask_param", 2),
            time_mask_param=params.get("time_mask_param", 12),
            num_freq_masks=params.get("num_freq_masks", 1),
            num_time_masks=params.get("num_time_masks", 1),
        ).astype(np.float32)

    return cA.astype(np.float32), cD.astype(np.float32)


def multi_channel_swt_torch_stft_fast(
    signal,
    fs,
    bands,
    num_bands=3,
    levels=6,
    time_bins=128,
    augment=False,
    noise_prob=0.5,
    snr_db_range=(20.0, 34.0),
    specaug_prob=0.5,
    specaug_params=None,
    norm_mode: str = "zscore",
    norm_clip: float = 5.0,
    norm_eps: float = 1e-6,
    stft_device: str = "cpu",
    stft_n_fft: int = 512,
    stft_hop_length: int = 128,
    stft_win_length: int = 512,
    stft_center: bool = True,
    stft_power: float = 1.0,
):
    """
    Torch-STFT fast path used to replace scipy/pywt-heavy SWT extraction.
    Output layout is kept compatible with SWT path: [C=num_bands*2, F=levels, T=time_bins].
    """
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if augment and (np.random.rand() < float(noise_prob)):
        snr_db = float(np.random.uniform(snr_db_range[0], snr_db_range[1]))
        x = add_awgn(x, snr_db).astype(np.float32)

    _device = str(stft_device).lower()
    if _device.startswith("cuda") and (not torch.cuda.is_available()):
        _device = "cpu"
    if int(stft_hop_length) <= 0:
        stft_hop_length = max(1, int(stft_n_fft) // 4)
    if int(stft_win_length) <= 0:
        stft_win_length = int(stft_n_fft)

    x_t = torch.tensor(x, dtype=torch.float32, device=_device)
    win = torch.hann_window(int(stft_win_length), device=_device, dtype=torch.float32)
    stft_complex = torch.stft(
        x_t,
        n_fft=int(stft_n_fft),
        hop_length=int(stft_hop_length),
        win_length=int(stft_win_length),
        window=win,
        center=bool(stft_center),
        return_complex=True,
    )
    mag = torch.abs(stft_complex).clamp_min(1e-8).pow(float(stft_power))
    freqs = torch.linspace(0.0, float(fs) * 0.5, steps=int(mag.size(0)), device=mag.device, dtype=mag.dtype)

    specs = []
    for band in list(bands)[: int(num_bands)]:
        lowcut, highcut = float(band[0]), float(band[1])
        band_mask = (freqs >= lowcut) & (freqs < highcut)
        if int(band_mask.sum().item()) <= 0:
            band_mag = torch.zeros((1, int(mag.size(1))), device=mag.device, dtype=mag.dtype)
        else:
            band_mag = mag[band_mask]
            if band_mag.dim() != 2:
                band_mag = band_mag.reshape(int(band_mag.size(0)), -1)
        band_mag_log = torch.log1p(band_mag)
        cA = torch_fn.adaptive_avg_pool2d(
            band_mag_log.unsqueeze(0).unsqueeze(0),
            (int(levels), int(time_bins)),
        ).squeeze(0).squeeze(0)
        band_delta = torch.diff(band_mag, dim=1, prepend=band_mag[:, :1]).abs()
        cD = torch_fn.adaptive_avg_pool2d(
            torch.log1p(band_delta).unsqueeze(0).unsqueeze(0),
            (int(levels), int(time_bins)),
        ).squeeze(0).squeeze(0)

        if augment and (np.random.rand() < float(specaug_prob)):
            params = specaug_params or {}
            cA_np = spec_augment(
                cA.detach().cpu().numpy().astype(np.float32),
                freq_mask_param=params.get("freq_mask_param", 2),
                time_mask_param=params.get("time_mask_param", 12),
                num_freq_masks=params.get("num_freq_masks", 1),
                num_time_masks=params.get("num_time_masks", 1),
            )
            cD_np = spec_augment(
                cD.detach().cpu().numpy().astype(np.float32),
                freq_mask_param=params.get("freq_mask_param", 2),
                time_mask_param=params.get("time_mask_param", 12),
                num_freq_masks=params.get("num_freq_masks", 1),
                num_time_masks=params.get("num_time_masks", 1),
            )
            cA = torch.tensor(cA_np, device=mag.device, dtype=mag.dtype)
            cD = torch.tensor(cD_np, device=mag.device, dtype=mag.dtype)

        specs.append(cA)
        specs.append(cD)

    c_target = int(num_bands) * 2
    if len(specs) < c_target:
        while len(specs) < c_target:
            specs.append(torch.zeros((int(levels), int(time_bins)), device=mag.device, dtype=mag.dtype))

    out = torch.stack(specs[:c_target], dim=0).to(dtype=torch.float32)

    nm = str(norm_mode).lower() if norm_mode is not None else "none"
    if nm == "zscore":
        _eps = float(norm_eps) if norm_eps is not None else 1e-6
        mu = out.mean(dim=(1, 2), keepdim=True)
        std = out.std(dim=(1, 2), keepdim=True, unbiased=False)
        out = (out - mu) / (std + _eps)
        if norm_clip is not None:
            out = torch.clamp(out, min=-float(norm_clip), max=float(norm_clip))

    return out.detach().cpu().numpy().astype(np.float32)


def multi_channel_swt_energy_adaptive(
    signal,
    fs,
    bands=None,
    num_bands=3,
    wavelet="db4",
    levels=6,
    time_bins=128,
    augment=False,
    noise_prob=0.5,
    snr_db_range=(20.0, 34.0),
    specaug_prob=0.5,
    specaug_params=None,
    norm_mode: str = "zscore",
    norm_clip: float = 5.0,
    norm_eps: float = 1e-6,
    use_torch_stft_fastpath: bool = False,
    stft_device: str = "cpu",
    stft_n_fft: int = 512,
    stft_hop_length: int = 128,
    stft_win_length: int = 512,
    stft_center: bool = True,
    stft_power: float = 1.0,
):
    """
    SWT alternative to multi_channel_spec_energy_adaptive.
    Output: [C=num_bands*2, F=levels, T=time_bins]
    """
    if bands is None:
        freqs, energy = spectral_energy_distribution(signal, fs)
        bands = energy_based_bands(freqs, energy, num_bands=num_bands)
    if bool(use_torch_stft_fastpath):
        return multi_channel_swt_torch_stft_fast(
            signal=signal,
            fs=fs,
            bands=bands,
            num_bands=num_bands,
            levels=levels,
            time_bins=time_bins,
            augment=augment,
            noise_prob=noise_prob,
            snr_db_range=snr_db_range,
            specaug_prob=specaug_prob,
            specaug_params=specaug_params,
            norm_mode=norm_mode,
            norm_clip=norm_clip,
            norm_eps=norm_eps,
            stft_device=stft_device,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            stft_center=stft_center,
            stft_power=stft_power,
        )

    do_noise = augment and (np.random.rand() < float(noise_prob))
    do_specaug = augment and (np.random.rand() < float(specaug_prob))

    specs = []
    for band in bands[:num_bands]:
        cA, cD = band_hilbert_swt(
            signal,
            fs=fs,
            band=band,
            wavelet=wavelet,
            levels=levels,
            time_bins=time_bins,
            add_noise=do_noise,
            snr_db_range=snr_db_range,
            apply_specaug=do_specaug,
            specaug_params=specaug_params,
        )
        specs.append(cA)
        specs.append(cD)

    # Ensure fixed channel count
    C_target = int(num_bands) * 2
    if len(specs) < C_target:
        while len(specs) < C_target:
            specs.append(np.zeros((int(levels), int(time_bins)), dtype=np.float32))

    out = np.stack(specs[:C_target], axis=0).astype(np.float32)  # [C,F,T]

    # Per-channel normalization for stability (SWT cA/cD can have very different scales)
    nm = str(norm_mode).lower() if norm_mode is not None else "none"
    if nm == "zscore":
        eps = float(norm_eps) if norm_eps is not None else 1e-6
        for c in range(out.shape[0]):
            x = out[c]
            mu = float(np.mean(x, dtype=np.float64))
            sigma = float(np.std(x, dtype=np.float64))
            if not np.isfinite(mu) or not np.isfinite(sigma) or sigma < eps:
                continue
            x = (x - mu) / (sigma + eps)
            if norm_clip is not None:
                x = np.clip(x, -float(norm_clip), float(norm_clip))
            out[c] = x.astype(np.float32)

    return out


# ============================================================================
# ATT418-MS: Multi-Scale STFT
# Overcomes the Heisenberg uncertainty trade-off by computing STFT at multiple
# window sizes on the same signal. Small windows capture transients (bearing
# impacts), large windows resolve narrowband harmonics (characteristic freqs).
# Output: [C=len(n_ffts), target_f, target_t] with all scales aligned to a
# common (target_f, target_t) grid via bilinear interpolation.
# ============================================================================

def _freq_align_resize(
    mag: np.ndarray,
    fs: int,
    target_f: int,
    target_t: int,
    freq_align_max_hz: float,
) -> np.ndarray:
    """
    ATT423-P1: Physical frequency alignment resize.

    Ensures bin index N in the output always represents the same physical frequency
    regardless of the source domain's sampling rate.

    For fs/2 <= max_hz: STFT covers a subset of the target range.
      → resize frequency axis to proportional bins, zero-pad the rest.
    For fs/2 > max_hz: STFT covers more than the target range.
      → crop frequency axis to max_hz, then resize.

    mag: [n_freq, n_frames] raw STFT magnitude
    """
    nyquist = float(fs) / 2.0
    n_freq_raw = mag.shape[0]

    if nyquist <= freq_align_max_hz:
        n_effective = max(1, int(round(target_f * nyquist / freq_align_max_hz)))
        mag_t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)
        mag_t = torch_fn.interpolate(
            mag_t, size=(n_effective, target_t), mode="bilinear", align_corners=False
        )
        if n_effective < target_f:
            pad = torch.zeros(1, 1, target_f - n_effective, target_t)
            mag_t = torch.cat([mag_t, pad], dim=2)
        return mag_t.squeeze(0).squeeze(0).numpy()
    else:
        crop_bin = max(1, int(round(n_freq_raw * freq_align_max_hz / nyquist)))
        mag_cropped = mag[:crop_bin, :]
        mag_t = torch.from_numpy(mag_cropped).unsqueeze(0).unsqueeze(0)
        mag_t = torch_fn.interpolate(
            mag_t, size=(target_f, target_t), mode="bilinear", align_corners=False
        )
        return mag_t.squeeze(0).squeeze(0).numpy()


def _order_norm_resize(
    mag: np.ndarray,
    fs: int,
    target_f: int,
    target_t: int,
    rpm: float,
    max_orders: float = 200.0,
) -> np.ndarray:
    """
    ATT457: Order-spectrum normalization resize.

    Converts STFT frequency axis from Hz to orders (multiples of shaft frequency).
    Bin index N in the output represents order = N * max_orders / target_f,
    ensuring domain-invariant fault-characteristic alignment: BPFI/BPFO appear
    at approximately the same bin positions regardless of bearing geometry or speed.

    mag: [n_freq, n_frames] raw STFT magnitude
    rpm: shaft speed in RPM (must be > 0)
    max_orders: maximum order number to retain (default 200 covers all typical
                bearing fault harmonics up to ~20× BPFI)
    """
    shaft_freq = max(rpm, 1.0) / 60.0
    nyquist = float(fs) / 2.0
    n_freq_raw = mag.shape[0]
    max_hz = shaft_freq * max_orders

    if nyquist <= max_hz:
        n_effective = max(1, int(round(target_f * nyquist / max_hz)))
        mag_t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)
        mag_t = torch_fn.interpolate(
            mag_t, size=(n_effective, target_t), mode="bilinear", align_corners=False
        )
        if n_effective < target_f:
            pad = torch.zeros(1, 1, target_f - n_effective, target_t)
            mag_t = torch.cat([mag_t, pad], dim=2)
        return mag_t.squeeze(0).squeeze(0).numpy()
    else:
        crop_bin = max(1, int(round(n_freq_raw * max_hz / nyquist)))
        mag_cropped = mag[:crop_bin, :]
        mag_t = torch.from_numpy(mag_cropped).unsqueeze(0).unsqueeze(0)
        mag_t = torch_fn.interpolate(
            mag_t, size=(target_f, target_t), mode="bilinear", align_corners=False
        )
        return mag_t.squeeze(0).squeeze(0).numpy()


def multi_scale_stft(
    signal: np.ndarray,
    fs: int,
    n_ffts: tuple = (64, 128, 256, 512, 1024, 2048),
    target_f: int = 128,
    target_t: int = 128,
    hop_divisor: int = 4,
    norm_mode: str = "zscore",
    norm_clip: float = 5.0,
    norm_eps: float = 1e-6,
    augment: bool = False,
    noise_prob: float = 0.5,
    snr_db_range: tuple = (20.0, 34.0),
    specaug_prob: float = 0.5,
    specaug_params: dict = None,
    freq_align_max_hz: float = 0.0,
    order_norm_rpm: float = 0.0,
    order_norm_max_orders: float = 200.0,
) -> np.ndarray:
    """
    Multi-Scale STFT: compute magnitude spectrograms at multiple window sizes
    and resize to a common grid.

    Parameters
    ----------
    signal : 1D numpy array of the vibration signal.
    fs : sampling frequency (Hz).
    n_ffts : tuple of int, STFT window sizes (ascending recommended).
    target_f : int, output frequency bins per channel.
    target_t : int, output time bins per channel.
    hop_divisor : int, hop_length = n_fft // hop_divisor (larger → more overlap).
    norm_mode : 'zscore' | 'log' | 'none', per-channel normalization.
    norm_clip : float, clip range for zscore normalization.
    augment : bool, whether to apply data augmentation.
    freq_align_max_hz : float, >0 enables physical frequency alignment to [0, max_hz].

    Returns
    -------
    np.ndarray of shape [C, target_f, target_t] where C = len(n_ffts).
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    sig_len = len(signal)
    use_freq_align = freq_align_max_hz > 0
    use_order_norm = order_norm_rpm > 0

    if augment and np.random.rand() < noise_prob:
        snr_db = float(np.random.uniform(snr_db_range[0], snr_db_range[1]))
        signal = add_awgn(signal, snr_db).astype(np.float64)

    specs = []
    for n_fft in n_ffts:
        actual_n_fft = min(int(n_fft), sig_len)
        if actual_n_fft < 4:
            specs.append(np.zeros((target_f, target_t), dtype=np.float32))
            continue

        hop_length = max(1, actual_n_fft // hop_divisor)
        noverlap = actual_n_fft - hop_length

        _, _, Zxx = stft(
            signal,
            fs=fs,
            window="hann",
            nperseg=actual_n_fft,
            noverlap=noverlap,
            boundary=None,
        )
        mag = np.abs(Zxx).astype(np.float32)

        nm = str(norm_mode).lower()
        if nm == "zscore":
            mu = float(np.mean(mag, dtype=np.float64))
            sigma = float(np.std(mag, dtype=np.float64))
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > norm_eps:
                mag = np.clip((mag - mu) / (sigma + norm_eps), -norm_clip, norm_clip)
        elif nm == "log":
            mag = np.log1p(mag)

        if use_order_norm:
            mag = _order_norm_resize(mag, fs, target_f, target_t,
                                     order_norm_rpm, order_norm_max_orders)
        elif use_freq_align:
            mag = _freq_align_resize(mag, fs, target_f, target_t, freq_align_max_hz)
        elif mag.shape[0] != target_f or mag.shape[1] != target_t:
            mag_t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)
            mag_t = torch_fn.interpolate(
                mag_t, size=(target_f, target_t), mode="bilinear", align_corners=False
            )
            mag = mag_t.squeeze(0).squeeze(0).numpy()

        if augment and np.random.rand() < specaug_prob:
            mag = spec_augment(mag, **(specaug_params or {}))

        specs.append(mag.astype(np.float32))

    return np.stack(specs, axis=0)  # [C, target_f, target_t]


# ---------------------------------------------------------------------------
# Wavelet Scattering Transform (WST)
# ---------------------------------------------------------------------------
# kymatio is optional; fallback computes a manual 2nd-order scattering via
# scipy/pywt so the pipeline never hard-fails.
try:
    from kymatio.numpy import Scattering1D as _KymatioScattering1D
    _HAS_KYMATIO = True
except Exception:
    _KymatioScattering1D = None
    _HAS_KYMATIO = False

_wst_logged = False
_wst_scattering_cache = {}  # (T, J, Q, order) → Scattering1D instance


def wavelet_scattering_features(
    signal: np.ndarray,
    fs: int,
    target_f: int = 128,
    target_t: int = 128,
    J: int = 8,
    Q: int = 8,
    order: int = 2,
    norm_mode: str = "zscore",
    norm_clip: float = 5.0,
    norm_eps: float = 1e-6,
) -> np.ndarray:
    """Compute Wavelet Scattering Transform features and reshape to 2-D images.

    Returns
    -------
    np.ndarray of shape ``[C_wst, target_f, target_t]`` where ``C_wst`` is
    determined by the scattering coefficient count and the reshaping strategy.

    The 2nd-order scattering coefficients explicitly capture amplitude
    modulation structure — the physical basis for distinguishing Ball defects
    (BSF modulated by cage frequency FTF) from Outer race defects (no
    modulation).
    """
    global _wst_logged

    signal = np.asarray(signal, dtype=np.float32).ravel()
    sig_len = len(signal)

    # Pad/truncate to power-of-2 for efficient scattering
    T = 1
    while T < sig_len:
        T *= 2
    if sig_len < T:
        padded = np.zeros(T, dtype=np.float32)
        padded[:sig_len] = signal
        signal = padded

    if _HAS_KYMATIO:
        _J = min(J, int(np.log2(T)) - 1)
        _cache_key = (T, _J, Q, order)
        if _cache_key not in _wst_scattering_cache:
            _wst_scattering_cache[_cache_key] = _KymatioScattering1D(
                J=_J, shape=(T,), Q=Q, max_order=order
            )
        scattering = _wst_scattering_cache[_cache_key]
        Sx = scattering(signal)  # [n_coeffs, n_time]
        Sx = np.asarray(Sx, dtype=np.float32)
    else:
        # Manual fallback: 1st-order = CWT magnitudes, 2nd-order = CWT of envelopes
        Sx = _manual_scattering(signal, T, J, Q, order)

    if Sx.ndim == 1:
        Sx = Sx.reshape(1, -1)

    n_coeffs, n_time = Sx.shape

    # Normalize each coefficient path independently
    nm = str(norm_mode).lower()
    for ci in range(n_coeffs):
        row = Sx[ci]
        if nm == "zscore":
            mu = float(np.mean(row))
            sigma = float(np.std(row))
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > norm_eps:
                Sx[ci] = np.clip((row - mu) / (sigma + norm_eps), -norm_clip, norm_clip)
        elif nm == "log":
            Sx[ci] = np.log1p(np.abs(row))

    # ATT445: single-channel output — all scattering orders mixed into one 2D image.
    # F_WST_3CH_WORSE_THAN_1CH: 3ch split (ATT443/444) degraded Ball 0.89→0.00.
    # 1ch lets SpecEncoder learn to extract modulation features from the mixed representation.
    if n_coeffs >= target_f:
        img = Sx[:target_f, :]
    else:
        repeats = (target_f + n_coeffs - 1) // n_coeffs
        img = np.tile(Sx, (repeats, 1))[:target_f, :]

    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    img_t = torch_fn.interpolate(img_t, size=(target_f, target_t), mode="bilinear", align_corners=False)
    img = img_t.squeeze(0).numpy()  # [1, target_f, target_t]

    _log_wst = False
    try:
        import config as _cfg_log
        _log_wst = bool(getattr(_cfg_log, "DATASET_LOG_FEATURE_BUILD", False))
    except Exception:
        _log_wst = False
    if _log_wst and not _wst_logged:
        backend = "kymatio" if _HAS_KYMATIO else "manual_fallback"
        print(f"[ATT445-WST] backend={backend}, J={J}, Q={Q}, order={order}, "
              f"sig_len={sig_len}, T_pad={T}, n_coeffs={n_coeffs}, n_time={n_time}, "
              f"output=({img.shape[0]},{img.shape[1]},{img.shape[2]})")
        _wst_logged = True

    return img.astype(np.float32)


def _morlet_wavelet(M: int, w: float = 5.0) -> np.ndarray:
    """Generate a Morlet wavelet of length M with center frequency w."""
    t = np.linspace(-M / 2.0, M / 2.0, M, endpoint=False, dtype=np.float64)
    sigma = M / (2.0 * np.pi * w)
    oscillation = np.exp(1j * w * t / (M / (2.0 * np.pi)))
    gaussian = np.exp(-0.5 * (t / (sigma + 1e-12)) ** 2)
    return np.real(oscillation * gaussian).astype(np.float32)


def _manual_scattering(
    signal: np.ndarray, T: int, J: int, Q: int, order: int
) -> np.ndarray:
    """Fallback scattering when kymatio is unavailable.

    Uses Morlet-like bandpass via scipy for 1st order, then computes envelopes
    and re-filters for 2nd order.
    """
    from scipy.signal import fftconvolve

    n_wavelets = J * Q
    center_freqs = np.logspace(np.log10(2.0), np.log10(max(4.0, T / 4.0)), num=n_wavelets)

    # 1st order: |signal * psi_lambda1|
    order1 = []
    for cf in center_freqs:
        width = max(8, int(cf))
        width = min(width, T)
        w = _morlet_wavelet(width, w=5.0)
        conv = fftconvolve(signal, w, mode="same")
        order1.append(np.abs(conv).astype(np.float32))

    coeffs = [np.array([float(np.mean(signal))], dtype=np.float32)]  # 0th order

    # Downsample 1st order for tractability
    ds = max(1, T // 256)
    for c1 in order1:
        coeffs.append(c1[::ds])

    if order >= 2:
        # 2nd order: |U1_lambda1 * psi_lambda2|, lambda2 < lambda1
        n2 = min(n_wavelets, 8)
        sub_cfs = center_freqs[:n2]
        for i, c1 in enumerate(order1[:n2]):
            for cf2 in sub_cfs[:max(1, i)]:
                width2 = max(8, int(cf2))
                width2 = min(width2, len(c1))
                w2 = _morlet_wavelet(width2, w=5.0)
                conv2 = fftconvolve(c1, w2, mode="same")
                coeffs.append(np.abs(conv2[::ds]).astype(np.float32))

    # Pad all to same length
    max_len = max(len(c) for c in coeffs)
    padded = []
    for c in coeffs:
        if len(c) < max_len:
            p = np.zeros(max_len, dtype=np.float32)
            p[:len(c)] = c
            padded.append(p)
        else:
            padded.append(c[:max_len])

    return np.stack(padded, axis=0)  # [n_coeffs, time]


# ============================================================================
# ATT449-S3: Envelope Spectrum for BPFO/BPFI discrimination
# ============================================================================

_BEARING_PARAMS = {
    "CWRU": {"n_balls": 9, "ball_diameter": 7.94e-3, "pitch_diameter": 39.04e-3,
             "contact_angle": 0.0, "default_rpm": 1772},
    "HUST": {"n_balls": 9, "ball_diameter": 7.94e-3, "pitch_diameter": 38.52e-3,
             "contact_angle": 0.0, "default_rpm": 1800},
    "PU":   {"n_balls": 16, "ball_diameter": 6.75e-3, "pitch_diameter": 45.0e-3,
             "contact_angle": 0.0, "default_rpm": 1500},
    "HIT":  {"n_balls": 8, "ball_diameter": 7.94e-3, "pitch_diameter": 34.55e-3,
             "contact_angle": 0.0, "default_rpm": 1800},
}

# ATT450-S2: Condition-key → RPM mapping for accurate characteristic-frequency computation.
# CWRU: 0HP=1797RPM, 1HP=1772, 2HP=1750, 3HP=1730.
CONDITION_RPM_MAP = {
    "CWRU": {
        "gk0": 1797, "gk_0": 1797, "hp0": 1797, "speed_1797": 1797,
        "gk1": 1772, "gk_1": 1772, "hp1": 1772, "speed_1772": 1772,
        "gk2": 1750, "gk_2": 1750, "hp2": 1750, "speed_1750": 1750,
        "gk3": 1730, "gk_3": 1730, "hp3": 1730, "speed_1730": 1730,
    },
}


def condition_to_rpm(domain: str, condition_key: str) -> float:
    """Resolve actual RPM from domain + condition key.  Falls back to default_rpm."""
    domain_map = CONDITION_RPM_MAP.get(domain, {})
    rpm = domain_map.get(str(condition_key).lower().strip(), None)
    if rpm is not None:
        return float(rpm)
    return float(_BEARING_PARAMS.get(domain, _BEARING_PARAMS["CWRU"])["default_rpm"])


def _characteristic_freqs(domain: str, rpm: float = None):
    """Compute BPFI, BPFO, BSF for a given bearing/domain."""
    params = _BEARING_PARAMS.get(domain, _BEARING_PARAMS["CWRU"])
    if rpm is None:
        rpm = params["default_rpm"]
    f_r = rpm / 60.0
    N = params["n_balls"]
    d = params["ball_diameter"]
    D = params["pitch_diameter"]
    ratio = d / D * np.cos(params["contact_angle"])
    bpfi = N / 2 * f_r * (1 + ratio)
    bpfo = N / 2 * f_r * (1 - ratio)
    bsf = D / (2 * d) * f_r * (1 - ratio ** 2)
    return {"bpfi": bpfi, "bpfo": bpfo, "bsf": bsf, "f_r": f_r}


def compute_envelope_stft(
    signal: np.ndarray,
    fs: float,
    target_f: int = 128,
    target_t: int = 128,
    n_fft: int = 256,
) -> np.ndarray:
    """Full-band envelope STFT (no band-pass filtering).

    Returns: [1, target_f, target_t] float32 log-magnitude envelope spectrogram.
    """
    from scipy.signal import hilbert as _hilbert
    from scipy.signal import stft as _stft

    sig = np.asarray(signal, dtype=np.float64).ravel()
    if len(sig) < n_fft:
        sig = np.pad(sig, (0, n_fft - len(sig)))

    analytic = _hilbert(sig)
    envelope = np.abs(analytic).astype(np.float64)

    _, _, Zxx = _stft(
        envelope,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft // 2,
        boundary=None,
    )
    mag = np.abs(Zxx).astype(np.float32)
    mag = np.log1p(mag)

    if mag.shape[0] != target_f or mag.shape[1] != target_t:
        t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)
        t = torch_fn.interpolate(t, size=(target_f, target_t), mode="bilinear", align_corners=False)
        mag = t.squeeze().numpy()

    mu = mag.mean()
    std = mag.std()
    if std > 1e-8:
        mag = (mag - mu) / std

    return mag.reshape(1, target_f, target_t).astype(np.float32)


def compute_bpfi_bpfo_ratio(
    signal: np.ndarray,
    fs: float,
    domain: str = "CWRU",
    rpm: float = None,
    tol_hz: float = 5.0,
) -> float:
    """Extract BPFI/BPFO amplitude ratio from envelope spectrum.

    > 1 → inner race fault dominant
    < 1 → outer race fault dominant
    ≈ 1 → normal or ball fault
    """
    from scipy.signal import hilbert as _hilbert
    from scipy.fft import rfft, rfftfreq

    sig = np.asarray(signal, dtype=np.float64).ravel()
    if len(sig) < 64:
        return 1.0

    cfreqs = _characteristic_freqs(domain, rpm)
    bpfi = cfreqs["bpfi"]
    bpfo = cfreqs["bpfo"]

    analytic = _hilbert(sig)
    envelope = np.abs(analytic)
    env_ac = envelope - np.mean(envelope)

    env_fft = np.abs(rfft(env_ac))
    env_freqs = rfftfreq(len(env_ac), 1.0 / fs)

    def _peak_amp(target_freq):
        mask = np.abs(env_freqs - target_freq) <= tol_hz
        if not np.any(mask):
            mask = np.abs(env_freqs - target_freq) <= 2 * tol_hz
        if not np.any(mask):
            return 1e-12
        return float(np.max(env_fft[mask]))

    amp_bpfi = _peak_amp(bpfi)
    amp_bpfo = _peak_amp(bpfo)

    ratio = amp_bpfi / (amp_bpfo + 1e-12)
    ratio = float(np.clip(ratio, 0.01, 100.0))
    return float(np.log(ratio))


# ---------------------------------------------------------------------------
# ATT464: Spectral Correlation Density (SCD) — Cyclic Modulation Spectrum
# ---------------------------------------------------------------------------

def compute_scd_channel(
    signal: np.ndarray,
    fs: int,
    n_fft: int = 256,
    hop_divisor: int = 4,
    target_f: int = 128,
    target_alpha: int = 128,
    norm_mode: str = "zscore",
    norm_clip: float = 5.0,
    norm_eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute a single-channel Cyclic Modulation Spectrum (CMS) as a lightweight
    approximation of Spectral Correlation Density for bearing fault diagnosis.

    The CMS captures second-order cyclostationarity by computing the spectral
    content of the instantaneous power envelope at each STFT frequency bin.
    Fault-induced periodic impulses at BPFO/BPFI/BSF create distinct cyclic
    frequency signatures that are more stable across domains than absolute
    frequency patterns captured by STFT alone.

    Theoretical basis:
      - Antoni (2007) "Cyclic spectral analysis in practice" — cyclostationary
        analysis as the gold standard for rotating machinery diagnostics.
      - Randall & Antoni (2011) "Rolling element bearing diagnostics — A tutorial"
        — cyclic modulation spectrum reveals fault-induced modulation patterns.

    Parameters
    ----------
    signal : 1D numpy array of vibration signal.
    fs : sampling rate (Hz).
    n_fft : FFT window size for the initial STFT decomposition.
    hop_divisor : hop_length = n_fft // hop_divisor.
    target_f : output spectral frequency bins.
    target_alpha : output cyclic frequency bins.
    norm_mode : 'zscore' | 'log' | 'none'.
    norm_clip : clip range for zscore.
    norm_eps : epsilon for numerical stability.

    Returns
    -------
    cms : np.ndarray of shape [1, target_f, target_alpha].
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    sig_len = len(signal)

    if sig_len < 16:
        return np.zeros((1, target_f, target_alpha), dtype=np.float32)

    actual_n_fft = min(n_fft, sig_len)
    if actual_n_fft < 4:
        return np.zeros((1, target_f, target_alpha), dtype=np.float32)

    hop_length = max(1, actual_n_fft // hop_divisor)
    noverlap = actual_n_fft - hop_length

    _, _, Zxx = stft(
        signal, fs=fs, window="hann",
        nperseg=actual_n_fft, noverlap=noverlap, boundary=None,
    )
    # Zxx: [nfreqs, nframes] complex

    power = np.abs(Zxx).astype(np.float64) ** 2  # [nfreqs, nframes]

    nframes = power.shape[1]
    if nframes < 2:
        return np.zeros((1, target_f, target_alpha), dtype=np.float32)

    power -= np.mean(power, axis=1, keepdims=True)

    cms = np.abs(np.fft.rfft(power, axis=1))  # [nfreqs, n_alpha]

    if cms.shape[0] != target_f or cms.shape[1] != target_alpha:
        cms_t = torch.from_numpy(cms.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        cms_t = torch_fn.interpolate(
            cms_t, size=(target_f, target_alpha),
            mode="bilinear", align_corners=False,
        )
        cms = cms_t.squeeze(0).squeeze(0).numpy()

    nm = str(norm_mode).lower()
    if nm == "zscore":
        mu = float(np.mean(cms, dtype=np.float64))
        sigma = float(np.std(cms, dtype=np.float64))
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > norm_eps:
            cms = np.clip((cms - mu) / (sigma + norm_eps), -norm_clip, norm_clip)
    elif nm == "log":
        cms = np.log1p(np.abs(cms))

    return cms.astype(np.float32).reshape(1, target_f, target_alpha)