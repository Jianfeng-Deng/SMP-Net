import numpy as np
import torch
import os
import hashlib
from torch.utils.data import Dataset
from fractions import Fraction
from scipy.signal import resample_poly

from utils.stft_dataset import (
    multi_channel_spec_energy_adaptive,
    multi_channel_swt_energy_adaptive,
    multi_scale_stft,
    spectral_energy_distribution,
    energy_based_bands,
    wavelet_scattering_features,
    _characteristic_freqs,
    compute_scd_channel,
)
import config as cfg


class MultimodalBearingDataset(Dataset):
    _ms_stft_class_logged = False

    """
    Multimodal dataset for bearing fault diagnosis.

    Each sample contains:
        - signal: processed 1D vibration signal (variable length, e.g., LDA)
        - spec:   2D spectrogram (from original signal)
        - text:   unified text description (string)
        - label:  integer fault label (for evaluation only)
        - domain: dataset source (CWRU / PU / HUST)
    """

    def __init__(
        self,
        X_origin,
        signals,
        texts,
        labels,
        fs,
        n_fft,
        domain,
        spec_augment=False,
        condition_keys=None,
        is_synthetic=None,
    ):
        """
        Args:
            X_origin (list or np.ndarray): original 1D signals
            signals  (list or np.ndarray): processed 1D signals (LDA)
            texts    (list of str): text descriptions
            labels   (list or np.ndarray): integer labels
            fs       (int): sampling frequency
            n_fft    (int): FFT window size
            domain   (str): dataset name, e.g. "CWRU", "PU", "HUST"
            is_synthetic (list/array): boolean mask, True for synthetic samples
        """
        assert len(X_origin) == len(signals) == len(texts) == len(labels), \
            "All input lists must have the same length."

        # 转换为列表或numpy数组以提高访问效率
        # 注意：避免使用dtype=object的numpy数组，因为它们在multiprocessing中可能无法正确序列化
        if isinstance(X_origin, list):
            self.X_origin = X_origin  # 保持为列表，支持不同长度的信号
        elif isinstance(X_origin, np.ndarray) and X_origin.dtype == object:
            # 如果已经是object类型的数组，转换为列表
            self.X_origin = X_origin.tolist()
        else:
            self.X_origin = X_origin
            
        if isinstance(signals, list):
            self.signals = signals  # 保持为列表
        elif isinstance(signals, np.ndarray) and signals.dtype == object:
            # 如果已经是object类型的数组，转换为列表
            self.signals = signals.tolist()
        else:
            self.signals = signals
            
        self.texts = texts
        self.labels = np.asarray(labels, dtype=np.int32)  # 确保是numpy数组
        self.domain = domain
        _domain_map = {"CWRU": 0, "PU": 2, "HUST": 3, "HIT": 4, "SEU": 5}
        self.domain_id = int(_domain_map.get(str(domain).upper(), 0))
        
        # Handle synthetic flag
        if is_synthetic is None:
            self.is_synthetic = np.zeros(len(self.labels), dtype=bool)
        else:
            if len(is_synthetic) != len(self.labels):
                raise ValueError("is_synthetic length must match labels length")
            self.is_synthetic = np.asarray(is_synthetic, dtype=bool)
        if condition_keys is None:
            self.condition_keys = np.asarray(["unknown"] * len(self.labels), dtype=object)
        else:
            if len(condition_keys) != len(self.labels):
                raise ValueError("condition_keys length must match labels length")
            self.condition_keys = np.asarray(condition_keys, dtype=object)
        # Deterministic per-sample integer condition ids for split utilities.
        uniq_keys = sorted(set([str(k) for k in self.condition_keys.tolist()]))
        key_to_id = {k: i for i, k in enumerate(uniq_keys)}
        self.condition_ids = np.asarray([key_to_id[str(k)] for k in self.condition_keys.tolist()], dtype=np.int32)
        self.condition_id_to_key = {v: k for k, v in key_to_id.items()}
        self.fs = fs
        self.n_fft = n_fft
        self.band_cache = {}
        self.swt_band_cache = {}
        self.spec_cache = {}  # 缓存频谱图（仅在spec_augment=False时使用）
        self.spec_augment = spec_augment
        self._spec_type = str(getattr(cfg, "SPEC_FEATURE_TYPE", "stft")).lower()
        self._swt_resample_before_extract = bool(
            getattr(cfg, "SWT_RESAMPLE_BEFORE_EXTRACT", False)
        )
        self._swt_target_fs = int(getattr(cfg, "SWT_TARGET_FS", self.fs))
        self._swt_use_dynamic_bands = bool(getattr(cfg, "SWT_USE_DYNAMIC_BANDS", False))
        self._swt_log_verbose = bool(getattr(cfg, "SWT_LOG_VERBOSE", False))
        self._swt_resample_logged = False
        self._swt_bands_logged = False
        self._swt_use_torch_stft_fastpath = bool(getattr(cfg, "SWT_USE_TORCH_STFT_FASTPATH", False))
        self._swt_torch_stft_device = str(getattr(cfg, "SWT_TORCH_STFT_DEVICE", "cpu"))
        self._swt_torch_stft_n_fft = int(getattr(cfg, "SWT_TORCH_STFT_N_FFT", 512))
        self._swt_torch_stft_hop_length = int(getattr(cfg, "SWT_TORCH_STFT_HOP_LENGTH", 128))
        self._swt_torch_stft_win_length = int(getattr(cfg, "SWT_TORCH_STFT_WIN_LENGTH", 512))
        self._swt_torch_stft_center = bool(getattr(cfg, "SWT_TORCH_STFT_CENTER", True))
        self._swt_torch_stft_power = float(getattr(cfg, "SWT_TORCH_STFT_POWER", 1.0))
        # ATT418-MS: Multi-Scale STFT config
        self._ms_stft_enable = bool(getattr(cfg, "MULTISCALE_STFT_ENABLE", False))
        self._log_feature_build = bool(getattr(cfg, "DATASET_LOG_FEATURE_BUILD", False))
        self._ms_stft_n_ffts = tuple(getattr(cfg, "MULTISCALE_STFT_N_FFTS", (64, 128, 256, 512, 1024, 2048)))
        self._ms_stft_target_f = int(getattr(cfg, "MULTISCALE_STFT_TARGET_F", 128))
        self._ms_stft_target_t = int(getattr(cfg, "MULTISCALE_STFT_TARGET_T", 128))
        self._ms_stft_hop_div = int(getattr(cfg, "MULTISCALE_STFT_HOP_DIVISOR", 4))
        self._ms_stft_norm = str(getattr(cfg, "MULTISCALE_STFT_NORM", "zscore"))
        # ATT423: physical frequency alignment
        self._freq_align_enable = bool(getattr(cfg, "SPEC_FREQ_ALIGN_ENABLE", False))
        self._freq_align_max_hz = float(getattr(cfg, "SPEC_FREQ_ALIGN_MAX_HZ", 6000.0))
        self._freq_align_logged = False
        # ATT457: order-spectrum normalization
        self._order_norm_enable = bool(getattr(cfg, "SPEC_ORDER_NORM_ENABLE", False))
        self._order_norm_max_orders = float(getattr(cfg, "SPEC_ORDER_NORM_MAX_ORDERS", 200.0))
        self._order_norm_logged = False
        self._ms_stft_logged = False
        # ATT464: SCD channel config (concat or bypass)
        self._scd_enable = bool(getattr(cfg, "SCD_CHANNEL_ENABLE", False))
        self._scd_bypass = bool(getattr(cfg, "SCD_BYPASS_INJECT", False))
        self._scd_n_fft = int(getattr(cfg, "SCD_N_FFT", 256))
        self._scd_hop_divisor = int(getattr(cfg, "SCD_HOP_DIVISOR", 4))
        self._scd_norm_mode = str(getattr(cfg, "SCD_NORM_MODE", "zscore"))
        self._scd_logged = False
        self._cache_in_memory = bool(getattr(cfg, "SPEC_CACHE_IN_MEMORY", True))
        self._cache_to_disk = bool(getattr(cfg, "SPEC_CACHE_TO_DISK", False))
        self._cache_dir = str(getattr(cfg, "SPEC_CACHE_DIR", "/tmp/spec_cache"))
        self._cache_dtype = str(getattr(cfg, "SPEC_CACHE_DTYPE", "float16")).lower()
        self._cache_version = str(getattr(cfg, "SPEC_CACHE_VERSION", "v1"))
        if self._cache_to_disk:
            os.makedirs(self._cache_dir, exist_ok=True)
        # ATT150 P0: input-level spec amplitude jitter (train-time only via spec_augment=True).
        self._spec_amp_jitter_enable = bool(getattr(cfg, "SPEC_AMP_JITTER_ENABLE", False))
        self._spec_amp_jitter_prob = float(getattr(cfg, "SPEC_AMP_JITTER_PROB", 0.5))
        self._spec_amp_jitter_std = float(getattr(cfg, "SPEC_AMP_JITTER_STD", 0.1))
        self._spec_amp_jitter_min = float(getattr(cfg, "SPEC_AMP_JITTER_MIN", 0.5))
        self._spec_amp_jitter_max = float(getattr(cfg, "SPEC_AMP_JITTER_MAX", 1.5))
        # ATT372: physical-frequency axis as position encoding (keep original bin-view spectra).
        self._use_physical_freq_posenc = bool(getattr(cfg, "USE_PHYSICAL_FREQ_POSITION_ENCODING", False))
        self._physical_freq_posenc_mode = str(
            getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_MODE", "add")
        ).lower()
        self._physical_freq_posenc_scale = float(
            getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_SCALE", 0.08)
        )
        self._physical_freq_ref_hz = float(getattr(cfg, "PHYSICAL_FREQ_REFERENCE_HZ", 32000.0))
        self._physical_freq_log_stats = bool(getattr(cfg, "PHYSICAL_FREQ_LOG_STATS", True))
        self._physical_freq_logged = False
        self._ablation_gating = bool(getattr(cfg, "ABLATION_GATING_ENABLE", False))
        if self._ablation_gating and (self._cache_in_memory or self._cache_to_disk):
            print(
                "[AblationGating] Disabling spec cache (requires fresh spec_legacy baseline per sample)."
            )
            self._cache_in_memory = False
            self._cache_to_disk = False
        self._use_stat_film = bool(getattr(cfg, "USE_STAT_FILM", False))
        self._use_phys_feature_proto = bool(getattr(cfg, "USE_PHYS_FEATURE_PROTO", False))
        # ATT442-WST: Wavelet Scattering Transform — extra channels for modulation capture
        self._wst_enable = bool(getattr(cfg, "WST_ENABLE", False))
        self._wst_j = int(getattr(cfg, "WST_J", 8))
        self._wst_q = int(getattr(cfg, "WST_Q", 8))
        self._wst_order = int(getattr(cfg, "WST_ORDER", 2))
        self._wst_norm = str(getattr(cfg, "WST_NORM_MODE", "zscore"))
        self._wst_bypass = bool(getattr(cfg, "WST_BYPASS_INJECT", False))
        self._wst_feat_cache = None
        self._wst_logged = False
        # ATT449-S3: Envelope Spectrum Bypass + BPFI/BPFO ratio
        self._env_bypass = bool(getattr(cfg, "ENV_BYPASS_INJECT", False))
        # HUST condition names encode the held-out speed. The legacy ratio uses
        # one 1800 RPM fallback for every HUST condition. This ablation removes
        # that scalar while retaining the envelope spectrum.
        self._disable_condition_rpm_ratio = (
            str(self.domain).upper() == "HUST"
            and bool(int(os.environ.get("HUST_DISABLE_CONDITION_RPM_RATIO", "1")))
        )
        self._env_logged = False

    def _cache_file_path(self, idx: int) -> str:
        ms_tag = ""
        if self._ms_stft_enable:
            _fa = f"|fa={int(self._freq_align_max_hz)}" if self._freq_align_enable else ""
            _on = f"|on={int(self._order_norm_max_orders)}" if self._order_norm_enable else ""
            ms_tag = f"|ms={','.join(str(x) for x in self._ms_stft_n_ffts)}|{self._ms_stft_target_f}x{self._ms_stft_target_t}{_fa}{_on}"
        wst_tag = ""
        if self._wst_enable:
            wst_tag = f"|wst=J{self._wst_j}Q{self._wst_q}O{self._wst_order}"
        scd_tag = f"|scd={self._scd_n_fft}" if self._scd_enable else ""
        key = (
            f"{self._cache_version}|{self._spec_type}|{self.domain}|{self.fs}|{self.n_fft}|"
            f"{self._swt_target_fs}|{self._swt_torch_stft_n_fft}|{self._swt_torch_stft_hop_length}|"
            f"{self._swt_torch_stft_win_length}|{self._swt_use_torch_stft_fastpath}{ms_tag}{wst_tag}{scd_tag}"
        )
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self._cache_dir, f"{self.domain}_{int(idx)}_{digest}.pt")

    def _resample_signal_for_swt(self, signal: np.ndarray):
        """
        Resample to a unified target fs for SWT extraction.
        This aligns physical frequency bands across domains.
        """
        x = np.asarray(signal, dtype=np.float32).reshape(-1)
        src_fs = float(self.fs)
        tgt_fs = float(self._swt_target_fs)
        if x.size < 2 or src_fs <= 0.0 or tgt_fs <= 0.0:
            return x, self.fs
        if (not self._swt_resample_before_extract) or abs(src_fs - tgt_fs) < 1e-6:
            return x, self.fs
        ratio = Fraction(tgt_fs / src_fs).limit_denominator(1000)
        y = resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
        return y, int(round(tgt_fs))

    def _build_physical_frequency_axis(self, f_bins: int, t_bins: int, fs_for_spec: int,
                                        order_rpm: float = 0.0):
        """
        Build physical-frequency position axis [1, F, T].
        - OrderNorm: linear order bins in [0, max_orders] (ATT457).
        - FreqAlign: linear Hz bins in [0, min(nyquist, max_hz)] (ATT457-fix).
        - STFT:  linear Hz bins in [0, fs/2].
        - SWT:   dyadic center-frequency proxy per decomposition level.
        """
        f_bins = max(1, int(f_bins))
        t_bins = max(1, int(t_bins))
        fs_now = max(1.0, float(fs_for_spec))

        if self._order_norm_enable and order_rpm > 0:
            orders = np.linspace(0.0, float(self._order_norm_max_orders),
                                 num=f_bins, dtype=np.float32)
            ref_ord = max(1.0, float(self._order_norm_max_orders))
            val_norm = np.clip(orders / ref_ord, a_min=0.0, a_max=1.5).astype(np.float32)
        elif str(self._spec_type).lower() == "swt":
            levels = np.arange(1, f_bins + 1, dtype=np.float32)
            hz = fs_now / (2.0 ** (levels + 1.0))
            ref_hz = max(1.0, float(self._physical_freq_ref_hz))
            val_norm = np.clip(hz / ref_hz, a_min=0.0, a_max=1.5).astype(np.float32)
        else:
            eff_max_hz = 0.5 * fs_now
            if self._freq_align_enable:
                eff_max_hz = min(eff_max_hz, self._freq_align_max_hz)
            hz = np.linspace(0.0, eff_max_hz, num=f_bins, dtype=np.float32)
            ref_hz = max(1.0, float(self._physical_freq_ref_hz))
            val_norm = np.clip(hz / ref_hz, a_min=0.0, a_max=1.5).astype(np.float32)

        axis = np.repeat(val_norm.reshape(1, f_bins, 1), t_bins, axis=2).astype(np.float32)
        return axis

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 原始信号生成频谱图
        origin_signal = self.X_origin[idx]
        fs_for_spec = int(self.fs)
        # ATT457: precompute RPM for order normalization and FreqPosEnc
        _sample_rpm = 0.0
        if self._order_norm_enable:
            from utils.stft_dataset import condition_to_rpm as _cond2rpm
            _sample_rpm = _cond2rpm(str(self.domain), str(self.condition_keys[idx]))

        cacheable = (not self.spec_augment)
        spec = None
        spec_legacy = None
        cache_path = self._cache_file_path(idx) if (cacheable and self._cache_to_disk) else None
        # 一级缓存：内存
        if cacheable and self._cache_in_memory and idx in self.spec_cache:
            spec = self.spec_cache[idx]
        # 二级缓存：磁盘（跨进程/跨运行复用）
        if spec is None and cacheable and cache_path is not None and os.path.exists(cache_path):
            try:
                cached_obj = torch.load(cache_path, map_location="cpu")
                if torch.is_tensor(cached_obj):
                    spec = cached_obj.float().cpu().numpy().astype(np.float32)
                elif isinstance(cached_obj, np.ndarray):
                    spec = cached_obj.astype(np.float32)
                if spec is not None and self._cache_in_memory:
                    self.spec_cache[idx] = spec
            except Exception:
                spec = None
        if spec is None:
            # Choose STFT or SWT spec based on config (default: SWT).
            if self._spec_type == "swt":
                swt_signal, swt_fs = self._resample_signal_for_swt(origin_signal)
                fs_for_spec = int(swt_fs)
                if (
                    self._swt_resample_before_extract
                    and self._swt_log_verbose
                    and (not self._swt_resample_logged)
                    and int(swt_fs) != int(self.fs)
                ):
                    print(
                        f"[SWT-Resample] domain={self.domain}, fs={int(self.fs)} -> target_fs={int(swt_fs)}"
                    )
                    self._swt_resample_logged = True
                if idx in self.swt_band_cache:
                    bands = self.swt_band_cache[idx]
                else:
                    # Attempt 62: Dynamic bands based on (potentially resampled) swt_fs
                    if self._swt_use_dynamic_bands:
                        freqs, energy = spectral_energy_distribution(swt_signal, swt_fs)
                        bands = energy_based_bands(freqs, energy, num_bands=3)
                        if self._swt_log_verbose and (not self._swt_bands_logged):
                            print(f"[SWT-DynamicBands] domain={self.domain}, fs={swt_fs}, bands={bands}")
                            self._swt_bands_logged = True
                    else:
                        # Fallback to original fs bands (legacy behavior of Att 55)
                        freqs, energy = spectral_energy_distribution(origin_signal, self.fs)
                        bands = energy_based_bands(freqs, energy, num_bands=3)
                    self.swt_band_cache[idx] = bands
                spec = multi_channel_swt_energy_adaptive(
                    swt_signal,
                    fs=swt_fs,
                    bands=bands,
                    num_bands=3,
                    wavelet=getattr(cfg, "SWT_WAVELET", "db4"),
                    levels=int(getattr(cfg, "SWT_LEVELS", 6)),
                    time_bins=int(getattr(cfg, "SWT_TIME_BINS", 128)),
                    augment=self.spec_augment,
                    norm_mode=str(getattr(cfg, "SWT_NORM", "zscore")),
                    norm_clip=float(getattr(cfg, "SWT_NORM_CLIP", 5.0)),
                    norm_eps=float(getattr(cfg, "SWT_NORM_EPS", 1e-6)),
                    use_torch_stft_fastpath=self._swt_use_torch_stft_fastpath,
                    stft_device=self._swt_torch_stft_device,
                    stft_n_fft=self._swt_torch_stft_n_fft,
                    stft_hop_length=self._swt_torch_stft_hop_length,
                    stft_win_length=self._swt_torch_stft_win_length,
                    stft_center=self._swt_torch_stft_center,
                    stft_power=self._swt_torch_stft_power,
                )
            elif self._ms_stft_enable:
                _align_hz = self._freq_align_max_hz if self._freq_align_enable else 0.0
                _order_rpm = _sample_rpm
                _order_max = self._order_norm_max_orders
                spec_ms = multi_scale_stft(
                    origin_signal,
                    fs=self.fs,
                    n_ffts=self._ms_stft_n_ffts,
                    target_f=self._ms_stft_target_f,
                    target_t=self._ms_stft_target_t,
                    hop_divisor=self._ms_stft_hop_div,
                    norm_mode=self._ms_stft_norm,
                    augment=self.spec_augment,
                    freq_align_max_hz=_align_hz,
                    order_norm_rpm=_order_rpm,
                    order_norm_max_orders=_order_max,
                )
                spec = spec_ms
                if self._ablation_gating:
                    if idx in self.band_cache:
                        _leg_bands = self.band_cache[idx]
                    else:
                        _leg_freqs, _leg_energy = spectral_energy_distribution(origin_signal, self.fs)
                        _leg_bands = energy_based_bands(_leg_freqs, _leg_energy, num_bands=3)
                        self.band_cache[idx] = _leg_bands
                    spec_legacy = multi_channel_spec_energy_adaptive(
                        origin_signal,
                        fs=self.fs,
                        bands=_leg_bands,
                        augment=self.spec_augment,
                    )
                if self._log_feature_build and not MultimodalBearingDataset._ms_stft_class_logged:
                    _mode_str = (f"OrderNorm rpm={_order_rpm:.0f} max_ord={_order_max:.0f}"
                                 if _order_rpm > 0
                                 else f"FreqAlign max_hz={int(_align_hz)}" if _align_hz > 0 else "OFF")
                    print(
                        f"[ATT418-MS] domain={self.domain}, fs={self.fs}, "
                        f"n_ffts={self._ms_stft_n_ffts}, "
                        f"output=({spec.shape[0]},{spec.shape[1]},{spec.shape[2]}), "
                        f"freq_mode={_mode_str}"
                    )
                    MultimodalBearingDataset._ms_stft_class_logged = True
                if self._log_feature_build and self._order_norm_enable and not self._order_norm_logged:
                    _shaft = _order_rpm / 60.0
                    _max_hz = _shaft * _order_max if _shaft > 0 else 0
                    _nyq = self.fs / 2.0
                    _coverage = min(1.0, _nyq / _max_hz) if _max_hz > 0 else 0
                    try:
                        _cf = _characteristic_freqs(str(self.domain), _order_rpm)
                        _bpfi_ord = _cf['bpfi'] / _shaft if _shaft > 0 else 0
                        _bpfo_ord = _cf['bpfo'] / _shaft if _shaft > 0 else 0
                    except Exception:
                        _bpfi_ord, _bpfo_ord = 0, 0
                    print(
                        f"[ATT457-OrderNorm] domain={self.domain}, fs={self.fs}, "
                        f"rpm={_order_rpm:.0f}, shaft_freq={_shaft:.2f}Hz, "
                        f"max_orders={_order_max:.0f}, eff_max_hz={_max_hz:.0f}Hz, "
                        f"coverage={_coverage:.4f}, "
                        f"BPFI_order={_bpfi_ord:.2f}, BPFO_order={_bpfo_ord:.2f}"
                    )
                    self._order_norm_logged = True
                elif self._log_feature_build and self._freq_align_enable and not self._freq_align_logged and not self._order_norm_enable:
                    _nyq = self.fs / 2.0
                    _ratio = min(1.0, _nyq / self._freq_align_max_hz)
                    print(
                        f"[ATT423-FreqAlign] domain={self.domain}, fs={self.fs}, "
                        f"nyquist={_nyq:.0f}Hz, max_hz={self._freq_align_max_hz:.0f}Hz, "
                        f"coverage_ratio={_ratio:.4f}"
                    )
                    self._freq_align_logged = True
                # ATT446: WST non-bypass (concat mode) — only when spec freshly computed
                if self._wst_enable and not self._wst_bypass:
                    _tf = int(spec.shape[1]) if spec.ndim == 3 else self._ms_stft_target_f
                    _tt = int(spec.shape[2]) if spec.ndim == 3 else self._ms_stft_target_t
                    wst_feat = wavelet_scattering_features(
                        origin_signal,
                        fs=self.fs,
                        target_f=_tf,
                        target_t=_tt,
                        J=self._wst_j,
                        Q=self._wst_q,
                        order=self._wst_order,
                        norm_mode=self._wst_norm,
                    )
                    spec = np.concatenate([spec, wst_feat], axis=0)
                    if self._log_feature_build and not self._wst_logged:
                        print(
                            f"[ATT446-WST] domain={self.domain}, fs={self.fs}, "
                            f"mode=concat, wst_shape={wst_feat.shape}, "
                            f"spec_shape=({spec.shape[0]},{spec.shape[1]},{spec.shape[2]})"
                        )
                        self._wst_logged = True
            else:
                # Legacy STFT path: band-based decomposition with energy-adaptive bands.
                if idx in self.band_cache:
                    bands = self.band_cache[idx]
                else:
                    freqs, energy = spectral_energy_distribution(origin_signal, self.fs)
                    bands = energy_based_bands(freqs, energy, num_bands=3)
                    self.band_cache[idx] = bands
                spec = multi_channel_spec_energy_adaptive(
                    origin_signal,
                    fs=self.fs,
                    bands=bands,
                    augment=self.spec_augment
                )
                # ATT405: cap STFT spectrogram size to prevent OOM for high-fs domains
                _max_f = int(getattr(cfg, "STFT_MAX_FREQ_BINS", 0))
                _max_t = int(getattr(cfg, "STFT_MAX_TIME_BINS", 0))
                if (_max_f > 0 or _max_t > 0) and isinstance(spec, np.ndarray) and spec.ndim == 3:
                    _c, _f, _t = spec.shape
                    _need_resize = False
                    _tgt_f, _tgt_t = _f, _t
                    if _max_f > 0 and _f > _max_f:
                        _tgt_f = _max_f
                        _need_resize = True
                    if _max_t > 0 and _t > _max_t:
                        _tgt_t = _max_t
                        _need_resize = True
                    if _need_resize:
                        import torch.nn.functional as _F_interp
                        _spec_t = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0)
                        _spec_t = _F_interp.interpolate(
                            _spec_t, size=(_tgt_f, _tgt_t), mode="bilinear", align_corners=False
                        )
                        spec = _spec_t.squeeze(0).numpy()
                        if not hasattr(self, "_stft_resize_logged"):
                            print(
                                f"[ATT405-STFTResize] domain={self.domain}, "
                                f"orig=({_c},{_f},{_t}) -> ({_c},{_tgt_f},{_tgt_t})"
                            )
                            self._stft_resize_logged = True
            # ATT464: SCD channel concatenation (after all spec paths, before caching)
            if self._scd_enable and isinstance(spec, np.ndarray) and spec.ndim == 3:
                _tf = int(spec.shape[1])
                _tt = int(spec.shape[2])
                scd_ch = compute_scd_channel(
                    origin_signal,
                    fs=self.fs,
                    n_fft=self._scd_n_fft,
                    hop_divisor=self._scd_hop_divisor,
                    target_f=_tf,
                    target_alpha=_tt,
                    norm_mode=self._scd_norm_mode,
                )
                spec = np.concatenate([spec, scd_ch], axis=0)
                if not self._scd_logged:
                    print(
                        f"[ATT464-SCD] domain={self.domain}, fs={self.fs}, "
                        f"n_fft={self._scd_n_fft}, scd_shape={scd_ch.shape}, "
                        f"spec_shape=({spec.shape[0]},{spec.shape[1]},{spec.shape[2]})"
                    )
                    self._scd_logged = True
            # 只在非augment模式下缓存
            if cacheable:
                if self._cache_in_memory:
                    self.spec_cache[idx] = spec
                if cache_path is not None:
                    try:
                        _spec_to_save = np.asarray(spec, dtype=np.float32)
                        if self._cache_dtype == "float16":
                            _spec_to_save = _spec_to_save.astype(np.float16)
                        _tmp_path = cache_path + ".tmp"
                        torch.save(torch.from_numpy(_spec_to_save), _tmp_path)
                        os.replace(_tmp_path, cache_path)
                    except Exception:
                        pass

        if self.spec_augment and self._spec_amp_jitter_enable and (np.random.rand() < self._spec_amp_jitter_prob):
            spec_np = np.asarray(spec, dtype=np.float32)
            if spec_np.ndim == 3 and spec_np.shape[0] > 0:
                c = int(spec_np.shape[0])
                scales = np.random.normal(
                    loc=1.0,
                    scale=self._spec_amp_jitter_std,
                    size=(c, 1, 1),
                ).astype(np.float32)
                scales = np.clip(scales, self._spec_amp_jitter_min, self._spec_amp_jitter_max)
                spec = (spec_np * scales).astype(np.float32)

        # ATT372: add physical-frequency position encoding before SpecEncoder.
        spec_freq_pos = None
        if self._use_physical_freq_posenc:
            spec_np = np.asarray(spec, dtype=np.float32)
            if spec_np.ndim == 3 and spec_np.shape[1] > 0 and spec_np.shape[2] > 0:
                _f, _t = int(spec_np.shape[1]), int(spec_np.shape[2])
                spec_freq_pos = self._build_physical_frequency_axis(_f, _t, fs_for_spec,
                                                                      order_rpm=_sample_rpm)
                if self._ablation_gating:
                    spec = spec_np
                elif self._physical_freq_posenc_mode == "add":
                    spec = (spec_np + float(self._physical_freq_posenc_scale) * spec_freq_pos).astype(np.float32)
                elif self._physical_freq_posenc_mode == "concat":
                    spec = np.concatenate([spec_np, spec_freq_pos], axis=0).astype(np.float32)
                else:
                    spec = spec_np
                if self._log_feature_build and self._physical_freq_log_stats and (not self._physical_freq_logged):
                    try:
                        print(
                            f"[FreqPosEnc] domain={self.domain}, spec_type={self._spec_type}, fs={int(fs_for_spec)}, "
                            f"mode={self._physical_freq_posenc_mode}, scale={float(self._physical_freq_posenc_scale):.4f}, "
                            f"axis_minmax=({float(spec_freq_pos.min()):.4f},{float(spec_freq_pos.max()):.4f})"
                        )
                        self._physical_freq_logged = True
                    except Exception:
                        pass

        # 处理后的信号（LDA等）作为 signal 模态
        signal = self.signals[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        # 转为 torch.Tensor（使用from_numpy避免复制，如果已经是numpy数组）
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
        else:
            signal = torch.tensor(signal, dtype=torch.float32)
        
        if isinstance(spec, np.ndarray):
            spec = torch.from_numpy(spec).float()
        else:
            spec = torch.tensor(spec, dtype=torch.float32)

        spec_freq_pos_tensor = None
        if spec_freq_pos is not None:
            spec_freq_pos_tensor = torch.from_numpy(spec_freq_pos).float()

        # ATT377-StatFiLM: extract 8-dim stat vector from raw signal
        stat_vec = None
        if self._use_stat_film:
            try:
                from utils.signal_to_text import extract_stat_vector
                _raw = np.asarray(origin_signal, dtype=np.float64).ravel()
                stat_vec = extract_stat_vector(_raw, float(self.fs))
            except Exception:
                stat_vec = np.zeros(11, dtype=np.float32)

        # ATT432-δ: extract physical diagnostic features for PhysFeatureProtoModule
        phys_feat_tensor = None
        if self._use_phys_feature_proto:
            try:
                from utils.signal_to_text import extract_diagnostic_features
                from models.phys_feature_proto import features_dict_to_tensor
                _raw_phys = np.asarray(origin_signal, dtype=np.float64).ravel()
                _feat_dict = extract_diagnostic_features(_raw_phys, float(self.fs))
                phys_feat_tensor = features_dict_to_tensor(_feat_dict)
            except Exception:
                from models.phys_feature_proto import PHYS_FEATURE_DIM
                phys_feat_tensor = torch.zeros(PHYS_FEATURE_DIM, dtype=torch.float32)

        out = {
            "signal": signal,    # shape: [T]
            "spec": spec,        # shape: [C, H, W]
            "text": text,        # str
            "label": int(label),
            "domain": self.domain,
            "domain_id": int(self.domain_id),
            "condition": str(self.condition_keys[idx]),
        }
        if spec_legacy is not None:
            if isinstance(spec_legacy, np.ndarray):
                out["spec_legacy"] = torch.from_numpy(spec_legacy).float()
            else:
                out["spec_legacy"] = torch.tensor(spec_legacy, dtype=torch.float32)
        if spec_freq_pos_tensor is not None:
            out["spec_freq_pos"] = spec_freq_pos_tensor
        # ATT446/ATT456-fix: WST bypass — always compute from origin_signal
        if self._wst_bypass and self._wst_enable:
            _spec_np = spec.numpy() if torch.is_tensor(spec) else np.asarray(spec)
            _tf = int(_spec_np.shape[1]) if _spec_np.ndim == 3 else self._ms_stft_target_f
            _tt = int(_spec_np.shape[2]) if _spec_np.ndim == 3 else self._ms_stft_target_t
            try:
                _wst_feat = wavelet_scattering_features(
                    origin_signal,
                    fs=self.fs,
                    target_f=_tf,
                    target_t=_tt,
                    J=self._wst_j,
                    Q=self._wst_q,
                    order=self._wst_order,
                    norm_mode=self._wst_norm,
                )
                if isinstance(_wst_feat, np.ndarray):
                    out["wst_spec"] = torch.from_numpy(_wst_feat).float()
                else:
                    out["wst_spec"] = torch.tensor(_wst_feat, dtype=torch.float32)
                if self._log_feature_build and not self._wst_logged:
                    print(
                        f"[ATT446-WST] domain={self.domain}, fs={self.fs}, "
                        f"mode=bypass, wst_shape={_wst_feat.shape}"
                    )
                    self._wst_logged = True
            except Exception:
                out["wst_spec"] = torch.zeros(1, _tf, _tt, dtype=torch.float32)
        # ATT449-S3 + ATT450-S2: Envelope Spectrum Bypass + RPM-aware BPFI/BPFO ratio
        if self._env_bypass:
            try:
                from utils.stft_dataset import compute_envelope_stft, compute_bpfi_bpfo_ratio, condition_to_rpm
                _spec_for_shape = spec.numpy() if torch.is_tensor(spec) else np.asarray(spec)
                _tf = int(_spec_for_shape.shape[1]) if _spec_for_shape.ndim == 3 else 128
                _tt = int(_spec_for_shape.shape[2]) if _spec_for_shape.ndim == 3 else 128
                _raw = np.asarray(origin_signal, dtype=np.float64).ravel()
                _env_stft = compute_envelope_stft(_raw, float(self.fs), target_f=_tf, target_t=_tt)
                out["env_spec"] = torch.from_numpy(_env_stft).float()
                _cond_key = str(self.condition_keys[idx])
                if self._disable_condition_rpm_ratio:
                    _rpm = 0.0
                    _ratio = 0.0
                else:
                    _rpm = condition_to_rpm(str(self.domain), _cond_key)
                    _ratio = compute_bpfi_bpfo_ratio(
                        _raw, float(self.fs), domain=str(self.domain), rpm=_rpm,
                    )
                out["bpfi_bpfo_ratio"] = torch.tensor([_ratio], dtype=torch.float32)
                if self._log_feature_build and not self._env_logged:
                    print(f"[ATT450-Env] domain={self.domain}, fs={self.fs}, "
                          f"condition={_cond_key}, rpm={_rpm:.0f}, "
                          f"env_stft_shape={_env_stft.shape}, ratio={_ratio:.4f}")
                    self._env_logged = True
            except Exception as _env_e:
                _spec_err = spec.numpy() if torch.is_tensor(spec) else np.asarray(spec)
                _tf = int(_spec_err.shape[1]) if _spec_err.ndim == 3 else 128
                _tt = int(_spec_err.shape[2]) if _spec_err.ndim == 3 else 128
                out["env_spec"] = torch.zeros(1, _tf, _tt, dtype=torch.float32)
                out["bpfi_bpfo_ratio"] = torch.zeros(1, dtype=torch.float32)
                if self._log_feature_build and not self._env_logged:
                    print(f"[ATT450-Env] Error computing envelope: {_env_e}")
                    self._env_logged = True
        # ATT464b: SCD bypass — compute CMS separately for TinySCDEncoder
        if self._scd_bypass:
            try:
                _spec_np = spec.numpy() if torch.is_tensor(spec) else np.asarray(spec)
                _tf = int(_spec_np.shape[1]) if _spec_np.ndim == 3 else 128
                _tt = int(_spec_np.shape[2]) if _spec_np.ndim == 3 else 128
                _scd_feat = compute_scd_channel(
                    origin_signal, fs=self.fs,
                    n_fft=self._scd_n_fft, hop_divisor=self._scd_hop_divisor,
                    target_f=_tf, target_alpha=_tt, norm_mode=self._scd_norm_mode,
                )
                out["scd_spec"] = torch.from_numpy(_scd_feat).float()
                if not self._scd_logged:
                    print(f"[ATT464b-SCD] domain={self.domain}, fs={self.fs}, "
                          f"mode=bypass, n_fft={self._scd_n_fft}, scd_shape={_scd_feat.shape}")
                    self._scd_logged = True
            except Exception as _scd_e:
                _spec_fb = spec.numpy() if torch.is_tensor(spec) else np.asarray(spec)
                _tf = int(_spec_fb.shape[1]) if _spec_fb.ndim == 3 else 128
                _tt = int(_spec_fb.shape[2]) if _spec_fb.ndim == 3 else 128
                out["scd_spec"] = torch.zeros(1, _tf, _tt, dtype=torch.float32)
                if not self._scd_logged:
                    print(f"[ATT464b-SCD] Error computing SCD: {_scd_e}")
                    self._scd_logged = True
        if spec_freq_pos_tensor is not None:
            out["spec_freq_pos"] = spec_freq_pos_tensor  # shape: [1, F, T]
        if stat_vec is not None:
            out["stat_vec"] = torch.from_numpy(stat_vec).float()  # shape: [11] (ATT379b-S2: 8→11)
        if phys_feat_tensor is not None:
            out["phys_features"] = phys_feat_tensor  # shape: [22]
        return out
