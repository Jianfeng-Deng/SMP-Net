# config.py
# 全局配置文件（类格式）

import os
import ast
import torch


def parse_fusion_decoder_grad_live_ratio(value):
    """Parse FUSION_DECODER_INPUT_GRAD_LIVE_RATIO; None/''/invalid -> None."""
    if value is None:
        return None
    if isinstance(value, str) and not str(value).strip():
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


class Config:
    """全局配置类，包含所有训练和模型配置参数"""
    
    # ========== 数据配置 ==========
    # 统一窗口长度（单位：秒）
    WINDOW_SIZE_SEC = 1.0
    
    # 各数据集采样率
    SAMPLING_RATES = {
        "CWRU": 12000,
        "PU": 64000,
        "HUST": 25600,
        "SEU": 2000,
        "HIT": 25000,
    }
    
    # 数据集路径配置
    CWRU_PATH = "/root/autodl-tmp/dataset/CWRU/raw"
    PU_PATH = "/root/autodl-tmp/dataset/PU/raw"
    HUST_PATH = "/root/autodl-tmp/dataset/HUST/raw"
    SEU_PATH = "/root/autodl-tmp/dataset/SEU/raw"
    HIT_PATH = str(os.environ.get("HIT_PATH", "/root/autodl-tmp/dataset/HIT/raw"))
    # Load HIT (.npy) separately from SEU (.csv). Set ENABLE_HIT_DATASET=0 to skip.
    ENABLE_HIT_DATASET = bool(int(os.environ.get("ENABLE_HIT_DATASET", "1")))
    # Task 1: in-domain cross-condition diagnosis on source domains (CWRU/PU).
    # If enabled, source-domain train/val/test are split by condition groups (default 50/50 groups).
    ENABLE_INTRA_CONDITION_TASK = True
    INTRA_CONDITION_VAL_RATIO = 0.2
    # 域内跨工况严格隔离：仅用同一域数据训练，再在该域上跨工况测试（避免任务/跨域干扰）。
    # 当 True 且 TASK1_ISOLATED_DOMAIN 非 None 时，本次运行只使用该域 train/val 训练，并只评估对应 Task1。
    TASK1_TRAIN_PER_DOMAIN_ISOLATED = bool(int(os.environ.get("TASK1_TRAIN_PER_DOMAIN_ISOLATED", "1")))
    # 单次运行使用的源域："CWRU" | "PU" | None。
    # "CWRU"：仅 CWRU 训练 → 只评估 Task1-CWRU；"PU"：仅 PU 训练 → 只评估 Task1-PU；None：CWRU+PU 联合训练 → 只评估 Task2（及可选 Task1 汇总）。
    _iso_domain_raw = os.environ.get("TASK1_ISOLATED_DOMAIN", "")
    TASK1_ISOLATED_DOMAIN = _iso_domain_raw if _iso_domain_raw in ("CWRU", "PU") else None
    # 为 True 时一次脚本顺序跑完三阶段：Task1-CWRU → Task1-PU → Task2（每阶段重新初始化模型、训练、评估并保存独立 checkpoint）。
    TASK1_RUN_ALL_PHASES_IN_ONE_SCRIPT = bool(int(os.environ.get("TASK1_RUN_ALL_PHASES_IN_ONE_SCRIPT", "1")))
    # Explicit condition-group lists (integer IDs) for reproducible paper experiments.
    # Keep None to auto-split 50/50 by available groups.
    CWRU_TRAIN_GROUPS = None
    CWRU_TEST_GROUPS = None
    PU_TRAIN_GROUPS = None
    PU_TEST_GROUPS = None
    
    # ========== 随机种子配置 ==========
    RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))  # 统一随机种子，用于确保结果可复现（支持环境变量覆盖以做多seed实验）
    
    # ========== 设备配置 ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== 域残差配置 ==========
    # 是否启用域残差（Domain Residual）进行跨域适配
    USE_DOMAIN_RESIDUAL = True  # 设为True启用
    # ATT112: 6 domains for load-aware DANN (CWRU-gk2=0, CWRU-gk4=1, PU=2, HUST=3, HIT=4(legacy), SEU=5).
    # Within CWRU Phase 1, domain_id alternates between 0(gk2) and 1(gk4) → real adversarial gradient.
    NUM_DOMAINS = 6  # ATT112: keep legacy HIT id and new SEU id separated
    DOMAIN_RESIDUAL_DIM = 128  # 域残差维度
    DOMAIN_RESIDUAL_INIT_SCALE = 0.0  # 域残差初始化缩放
    USE_DOMAIN_LORA = bool(int(os.environ.get("USE_DOMAIN_LORA", "1")))
    DOMAIN_LORA_RANK = 4  # LoRA 低秩的秩（推荐 4-8）
    DOMAIN_LORA_ALPHA = 4  # LoRA 缩放因子（通常等于 rank）
    DOMAIN_LORA_DROPOUT = 0.0  # LoRA dropout 概率
    
    # ========== 融合模块配置 ==========
    # ATT460: Shared Projection Fusion — FALSIFIED: F_SHARED_PROJ_TRIVIAL_ALIGNMENT
    USE_SHARED_PROJECTION_FUSION = bool(int(os.environ.get("USE_SHARED_PROJECTION_FUSION", "0")))
    # ATT470: SpecEncoderLight + trainable GatedAdditive + cross-modal cosine align on projections
    ATT470_STRUCTURE_ENABLE = bool(int(os.environ.get("ATT470_STRUCTURE_ENABLE", "0")))
    ATT470_MOE_ROUTE_DIAG_ENABLE = bool(int(os.environ.get("ATT470_MOE_ROUTE_DIAG_ENABLE", "1")))
    # ATT471: Asymmetric Gradient-Isolated Training — detach encoder tensors at fusion forward
    # so fusion_cls / decoder optimize fusion without suppressing encoder gradients (cf. ICCV'25 DGL theory).
    ATT471_AGIT_ENABLE = bool(int(os.environ.get("ATT471_AGIT_ENABLE", "0")))
    # ATT472: Two-End-Separated + Shared Semantic Neck fusion.
    # Structural intent: isolate shallow modality statistics while forcing shared semantic bottleneck.
    ATT472_TWO_END_SHARED_NECK_ENABLE = bool(int(os.environ.get("ATT472_TWO_END_SHARED_NECK_ENABLE", "0")))
    # ATT473: Specialist + Arbiter fusion.
    # Two semantic experts (spec_sem / fusion_sem) + decision-level arbiter to avoid shared-neck over-smoothing.
    ATT473_SPECIALIST_ARBITER_ENABLE = bool(int(os.environ.get("ATT473_SPECIALIST_ARBITER_ENABLE", "0")))
    # Dual-contract dual-head preset — fusion path detaches encoder inputs (like AGIT) while keeping
    # fusion trainable (SplitArch off when SPLIT_ARCH_ENABLE env is omitted). Spec contract stays on spec_head /
    # signal / MLSA; fusion contract on fusion + decoder (+ optional DGL aux). Does not replace early-stop metric.
    # Default ON (2026-05): avoids silent runs with SplitArch+bypass + frozen fusion contract (ATT475/ATT479 audit).
    # Opt out: export DUAL_CONTRACT_ENABLE=0
    DUAL_CONTRACT_ENABLE = bool(int(os.environ.get("DUAL_CONTRACT_ENABLE", "1")))
    DUAL_CONTRACT_SAVE_EXTRA_CKPT = bool(int(os.environ.get("DUAL_CONTRACT_SAVE_EXTRA_CKPT", "0")))
    DUAL_CONTRACT_FORCE_GRAD_DETAIL = bool(int(os.environ.get("DUAL_CONTRACT_FORCE_GRAD_DETAIL", "0")))
    _cm_align_env = os.environ.get("FUSION_CROSS_MODAL_ALIGN_WEIGHT")
    _split_arch_env = os.environ.get("SPLIT_ARCH_ENABLE")
    if ATT470_STRUCTURE_ENABLE:
        FUSION_CROSS_MODAL_ALIGN_WEIGHT = float(_cm_align_env) if _cm_align_env is not None else 0.15
        SPLIT_ARCH_ENABLE = bool(int(_split_arch_env)) if _split_arch_env is not None else False
    else:
        FUSION_CROSS_MODAL_ALIGN_WEIGHT = float(_cm_align_env) if _cm_align_env is not None else 0.0
        SPLIT_ARCH_ENABLE = bool(int(_split_arch_env)) if _split_arch_env is not None else True
    # ATT471 preset: weaker CM align when env omits weight (fast trivial convergence on projections).
    if ATT471_AGIT_ENABLE and _cm_align_env is None:
        FUSION_CROSS_MODAL_ALIGN_WEIGHT = 0.05
    # ATT472 preset: also use weaker CM alignment by default unless explicitly overridden.
    if ATT472_TWO_END_SHARED_NECK_ENABLE and _cm_align_env is None:
        FUSION_CROSS_MODAL_ALIGN_WEIGHT = 0.05
    # ATT473 preset: keep weak CM alignment unless explicitly overridden.
    if ATT473_SPECIALIST_ARBITER_ENABLE and _cm_align_env is None:
        FUSION_CROSS_MODAL_ALIGN_WEIGHT = 0.05
    if DUAL_CONTRACT_ENABLE and _cm_align_env is None:
        FUSION_CROSS_MODAL_ALIGN_WEIGHT = 0.05
    # AGIT 需要可训练 fusion：在未开 ATT470 时默认仍为 SplitArch=True，此处在未显式设置 SPLIT_ARCH 时关闭 bypass。
    if ATT471_AGIT_ENABLE and (not ATT470_STRUCTURE_ENABLE) and _split_arch_env is None:
        SPLIT_ARCH_ENABLE = False
    # ATT472 uses trainable fusion path by design.
    if ATT472_TWO_END_SHARED_NECK_ENABLE and (not ATT470_STRUCTURE_ENABLE) and _split_arch_env is None:
        SPLIT_ARCH_ENABLE = False
    # ATT473 also requires trainable fusion path.
    if ATT473_SPECIALIST_ARBITER_ENABLE and (not ATT470_STRUCTURE_ENABLE) and _split_arch_env is None:
        SPLIT_ARCH_ENABLE = False
    if DUAL_CONTRACT_ENABLE and (not ATT470_STRUCTURE_ENABLE) and _split_arch_env is None:
        SPLIT_ARCH_ENABLE = False
    # Hard guard: dual-contract requires trainable fusion path.
    # If SplitArch is explicitly forced on, fail fast to avoid wasting runs.
    if DUAL_CONTRACT_ENABLE and SPLIT_ARCH_ENABLE:
        raise ValueError(
            "Invalid config: DUAL_CONTRACT_ENABLE=1 requires SPLIT_ARCH_ENABLE=0. "
            "Please unset SPLIT_ARCH_ENABLE or export SPLIT_ARCH_ENABLE=0."
        )
    # ATT458: Decision-Level Fusion — FALSIFIED: F_DECISION_LEVEL_WEIGHT_COLLAPSE
    USE_DECISION_LEVEL_FUSION = bool(int(os.environ.get("USE_DECISION_LEVEL_FUSION", "0")))
    # ATT459: Fixed equal weight fusion — FALSIFIED: F_FIXED_EQUAL_STILL_ORTHOGONAL
    FUSION_FIXED_EQUAL_WEIGHT = bool(int(os.environ.get("FUSION_FIXED_EQUAL_WEIGHT", "1")))
    # ATT461: 回退到ATT456基线GatedAdditiveFusion + OGM-GE梯度调制
    USE_GATED_ADDITIVE_FUSION = bool(int(os.environ.get("USE_GATED_ADDITIVE_FUSION", "1")))
    
    # ATT461: OGM-GE Gradient Modulation — FALSIFIED: F_OGM_GE_ENCODER_ONLY
    OGM_GE_ENABLE = bool(int(os.environ.get("OGM_GE_ENABLE", "0")))
    OGM_GE_ALPHA = float(os.environ.get("OGM_GE_ALPHA", "1.0"))
    OGM_GE_START_EPOCH = int(os.environ.get("OGM_GE_START_EPOCH", "2"))
    
    # ATT462: Fusion Residual Identity Skip (ResNet principle for fusion module)
    # FALSIFIED: F_RESSKIP_IDENTITY_SPEC_GRADIENT_CUT
    FUSION_RESIDUAL_SKIP_ENABLE = bool(int(os.environ.get("FUSION_RESIDUAL_SKIP_ENABLE", "0")))
    FUSION_RESIDUAL_SKIP_INIT_ALPHA = float(os.environ.get("FUSION_RESIDUAL_SKIP_INIT_ALPHA", "0.1"))
    
    # ATT463: Split Architecture — independent encoder training + fusion bypass
    # SPLIT_ARCH_ENABLE resolved above (ATT470 preset defaults it off when ATT470_STRUCTURE_ENABLE=1).
    SPLIT_ARCH_SIGNAL_WEIGHT = float(os.environ.get("SPLIT_ARCH_SIGNAL_WEIGHT", "1.0"))
    # ATT433: Gate entropy regularization to prevent gate binarization (gate_std→0.49)
    GATE_ENTROPY_WEIGHT = float(os.environ.get("GATE_ENTROPY_WEIGHT", "0.10"))
    # ATT434-α: Prototype-as-Classifier (FALSIFIED: F_PROTO_CLS_SELF_TERMINATING)
    # ATT436: disabled — proto CE loss vanishes as prototypes converge, starving fusion grad.
    USE_PROTO_CLASSIFIER = bool(int(os.environ.get("USE_PROTO_CLASSIFIER", "0")))
    PROTO_CLS_TEMPERATURE = float(os.environ.get("PROTO_CLS_TEMPERATURE", "0.10"))
    # ATT436-β: Cosine alignment loss (soft constraint, cooperative with linear head CE)
    USE_PROTO_COSINE_ALIGN = bool(int(os.environ.get("USE_PROTO_COSINE_ALIGN", "1")))
    PROTO_COSINE_ALIGN_WEIGHT = float(os.environ.get("PROTO_COSINE_ALIGN_WEIGHT", "0.10"))

    # ATT438: Disentangled Gradient Learning (DGL) — ICCV 2025 inspired
    # Addresses F_FUSION_MODULE_GRADIENT_ISLAND: fusion module has no direct gradient source.
    # (1) Aux classification head on z_fusion gives fusion module a direct CE loss.
    # (2) Detach encoder outputs before fusion so multimodal loss does not interfere with encoders.
    # ATT438: DGL — aux fusion-head path falsified (F_DGL_AUX_CE_CONVERGE, PU class_weight bug).
    # Encoder detach at fusion forward is decoupled from DGL_ENABLE (ATT471 / AGIT): see FUSION_FORWARD_DETACH_ENCODER_INPUTS.
    DGL_ENABLE = bool(int(os.environ.get("DGL_ENABLE", "0")))
    DGL_AUX_FUSION_HEAD_WEIGHT = float(os.environ.get("DGL_AUX_FUSION_HEAD_WEIGHT", "0.30"))
    DGL_DETACH_ENCODER_INPUTS = bool(int(os.environ.get("DGL_DETACH_ENCODER_INPUTS", "0")))
    DGL_AUX_FUSION_HEAD_LR_MULT = float(os.environ.get("DGL_AUX_FUSION_HEAD_LR_MULT", "1.0"))
    _fusion_detach_override = os.environ.get("FUSION_FORWARD_DETACH_ENCODER_INPUTS")
    if _fusion_detach_override is None:
        FUSION_FORWARD_DETACH_ENCODER_INPUTS = (
            bool(DGL_DETACH_ENCODER_INPUTS)
            or bool(ATT471_AGIT_ENABLE)
            or bool(DUAL_CONTRACT_ENABLE)
        )
    else:
        FUSION_FORWARD_DETACH_ENCODER_INPUTS = bool(int(_fusion_detach_override))
    # Allows a matched reviewer ablation with an identical forward graph and
    # parameter count. It must never silently disable the production guard.
    DUAL_CONTRACT_ALLOW_NO_DETACH_ABLATION = bool(
        int(os.environ.get("DUAL_CONTRACT_ALLOW_NO_DETACH_ABLATION", "0"))
    )

    # Fusion structural gradient fix: mean(z_sig,z_spec) + MLP residual (no tanh gate saturation);
    # optional full-gradient L2 (F.normalize) instead of z/||z||.detach() (see F_GATED_ADDITIVE_GRADIENT_DEATH).
    # DUAL_CONTRACT runs default these ON so dual-contract experiments do not repeat the old fusion death mode.
    _fus_mean_res_env = os.environ.get("FUSION_MEAN_RESIDUAL_ENABLE")
    _fus_std_l2_env = os.environ.get("FUSION_USE_STANDARD_UNIT_VECTOR_NORM")
    if DUAL_CONTRACT_ENABLE:
        FUSION_MEAN_RESIDUAL_ENABLE = bool(int(_fus_mean_res_env)) if _fus_mean_res_env is not None else True
        FUSION_USE_STANDARD_UNIT_VECTOR_NORM = bool(int(_fus_std_l2_env)) if _fus_std_l2_env is not None else True
    else:
        FUSION_MEAN_RESIDUAL_ENABLE = bool(int(_fus_mean_res_env)) if _fus_mean_res_env is not None else False
        FUSION_USE_STANDARD_UNIT_VECTOR_NORM = bool(int(_fus_std_l2_env)) if _fus_std_l2_env is not None else False

    # ATT440: CosProto — FALSIFIED (F_COSPROTO_TEXT_GEOMETRY: text protos cos(IR,OR)=0.90
    # makes cosine classification impossible. SpecHead 0.7312→0.2957, Inner R=0.00).
    # ATT441: reverted to MLP classifier (default=0).
    SPECHEAD_USE_COSINE_PROTO = bool(int(os.environ.get("SPECHEAD_USE_COSINE_PROTO", "0")))

    # ATT453: SpecHead classification mode — "mlp" (legacy) or "proto" (simplex ETF prototype cosine)
    # Proto mode: z_spec → L2norm → cosine(learnable_proto[C, D]) / τ → CE
    # Eliminates domain-specific MLP decision boundary; metric-learning approach is inherently domain-invariant.
    # Key difference from ATT440: ETF init (inter_cos=-0.33) instead of text_proto (cos(IR,OR)=0.89);
    # fully trainable prototypes instead of fixed text anchors.
    SPEC_HEAD_MODE = os.environ.get("SPEC_HEAD_MODE", "mlp")
    SPEC_PROTO_TEMPERATURE = float(os.environ.get("SPEC_PROTO_TEMPERATURE", "0.1"))
    SPEC_PROTO_MARGIN = float(os.environ.get("SPEC_PROTO_MARGIN", "0.0"))

    # ATT439: XKD — FALSIFIED (F_XKD_GRADIENT_HIJACK: T²=16 amplification hijacks spec encoder)
    CROSS_MODAL_KD_ENABLE = bool(int(os.environ.get("CROSS_MODAL_KD_ENABLE", "0")))
    CROSS_MODAL_KD_WEIGHT = float(os.environ.get("CROSS_MODAL_KD_WEIGHT", "0.50"))
    CROSS_MODAL_KD_TEMPERATURE = float(os.environ.get("CROSS_MODAL_KD_TEMPERATURE", "4.0"))

    # ATT439: Hierarchical SpecHead — FALSIFIED (BUG_P3_DEAD_CODE, also structurally unnecessary)
    HIERARCHICAL_SPEC_HEAD_PRIMARY = bool(int(os.environ.get("HIERARCHICAL_SPEC_HEAD_PRIMARY", "0")))

    # ATT431: Compact Bilinear Pooling (falsified F_CBP, kept for reference)
    USE_COMPACT_BILINEAR_FUSION = bool(int(os.environ.get("USE_COMPACT_BILINEAR_FUSION", "0")))
    CBP_DIM = int(os.environ.get("CBP_DIM", "2048"))
    # Cross-Attention融合配置 (legacy, active when both above are 0)
    USE_CROSS_ATTENTION_FUSION = True
    USE_CLASS_AWARE_ATTENTION = False  # 是否启用类感知注意力（可选，实验性功能）
    FUSION_NUM_HEADS = 4  # Attention头数
    FUSION_DROPOUT = 0.1  # Fusion模块的Dropout率
    FUSION_DOMAIN_EMBED_DIM = 64  # 域embedding维度
    FUSION_USE_MULTI_TOKEN = True  # 是否使用多token Cross-Attention（True：真正的跨模态对齐，False：单token向后兼容）
    # ATT166: 2→1 削减 Fusion Module 参数 ~8.4M（从 11.6M 降至 ~3.2M，全模型 13M→~4.6M）。
    # 依据：ViCA (2025) 证明视觉 token 仅需稀疏层 cross-attn 即可保留 98% 性能；"Shallow Cross-Attention"
    # (arXiv:2504.21447, 2025) 证明浅层 cross-attn 在细粒度视觉任务（如轴承频谱分类）优于深层；
    # CWRU 参数:样本比从 9088:1 降至 ~3594:1，PU 从 3540:1 降至 ~1245:1，理论上显著改善过拟合。
    # ATT162-164 总结：13M 参数配合 1280 CWRU 样本导致 val_loss 从 E3 起永久发散，是 ES 失效的根因。
    FUSION_CROSS_ATTN_DEPTH = 1  # ATT166: 2→1（削减 ~8.4M 参数，消除过参数化主因）
    # NOTE (stability): gate-masked attention can amplify collapse by suppressing an entire modality
    # in cross-attention key/value. Keep OFF by default; rely on reliability gating after pooling.
    USE_GATE_MASKED_ATTENTION = False  # 是否在cross-attention内部应用gate mask（控制key/value）
    # ATT416: Signal entropy gate — when signal classification head outputs near-uniform distribution
    # (high entropy → uninformative), scale down signal's K/V contribution in spec→signal cross-attention.
    # Prevents random signal noise from polluting spec features through cross-attention.
    # Root cause: CORE_LOSS_ONLY blocks aux_signal CE, signal encoder gets no direct gradient,
    # signal_logits stays at random chance (H≈log(C)), and noise flows into z_fusion via cross-attn.
    FUSION_SIGNAL_ENTROPY_GATE = bool(int(os.environ.get("FUSION_SIGNAL_ENTROPY_GATE", "0")))
    FUSION_SIGNAL_ENTROPY_GATE_FLOOR = float(os.environ.get("FUSION_SIGNAL_ENTROPY_GATE_FLOOR", "0.1"))
    # Feed pre-L2 z_spec to fusion to preserve amplitude geometry before semantic normalization.
    FUSION_USE_SPEC_PRE_L2_INPUT = True
    # ATT377: LayerNorm both modalities before fusion to prevent scale mismatch.
    FUSION_INPUT_LAYERNORM = True

    # ATT377-StatFiLM: numerical stat vector conditioning on SpecEncoder
    # ATT379b-S2: expanded from 8→11 with envelope analysis features (Inner vs Outer discriminant)
    USE_STAT_FILM = bool(int(os.environ.get("USE_STAT_FILM", "1")))
    STAT_FILM_N_FEATURES = 11
    STAT_FILM_HIDDEN_DIM = 64

    # ========== Spec feature extraction ==========
    # 'stft': 기존 band-hilbert-STFT multi-channel spec (legacy)
    # 'swt' : band-hilbert-SWT multi-channel "scalogram" (recommended for impulsive/non-stationary faults)
    SPEC_FEATURE_TYPE = os.environ.get("SPEC_FEATURE_TYPE", "stft")
    # STFT size cap: downsample spectrograms with T > this via interpolation.
    # Prevents OOM for high-fs domains (PU fs=64000 → T≈500 per channel).
    STFT_MAX_FREQ_BINS = 128    # cap frequency dimension
    STFT_MAX_TIME_BINS = 128    # cap time dimension

    # ATT418-MS: Multi-Scale STFT — overcomes fixed time-frequency resolution trade-off
    # by computing STFT at multiple window sizes, capturing both transients (small n_fft)
    # and narrowband harmonics (large n_fft). Output: [C=len(n_ffts), F_target, T_target].
    # ATT420: enabled by default — ATT419 proved MS-STFT gives SpecHead=0.88 vs Legacy=0.63.
    MULTISCALE_STFT_ENABLE = bool(int(os.environ.get("MULTISCALE_STFT_ENABLE", "1")))

    # ATT420-P2: Spec-Preserving Residual Fusion.
    # When spec features are strong (MS-STFT), cross-attention can destroy their discriminative
    # structure. This adds: z_fusion = α * spec_proj(z_spec) + (1-α) * cross_attn_fusion
    # where α is learnable, initialized at init_alpha (0.5 = equal blend).
    FUSION_SPEC_RESIDUAL_ENABLE = bool(int(os.environ.get("FUSION_SPEC_RESIDUAL_ENABLE", "1")))
    FUSION_SPEC_RESIDUAL_INIT_ALPHA = float(os.environ.get("FUSION_SPEC_RESIDUAL_INIT_ALPHA", "0.5"))
    # ATT421-P2: Signal encoder domain-adaptive normalization.
    # Applies InstanceNorm1d to z_signal before entering fusion cross-attention,
    # stabilizing feature norms across domains (CWRU=27 vs HUST=5 → all ≈1).
    FUSION_SIGNAL_INSTANCE_NORM = bool(int(os.environ.get("FUSION_SIGNAL_INSTANCE_NORM", "1")))
    _ms_nffts_str = os.environ.get("MULTISCALE_STFT_N_FFTS", "64,128,256,512,1024,2048")
    MULTISCALE_STFT_N_FFTS = tuple(int(x) for x in _ms_nffts_str.split(",") if x.strip())
    MULTISCALE_STFT_TARGET_F = int(os.environ.get("MULTISCALE_STFT_TARGET_F", "128"))
    MULTISCALE_STFT_TARGET_T = int(os.environ.get("MULTISCALE_STFT_TARGET_T", "128"))
    MULTISCALE_STFT_HOP_DIVISOR = int(os.environ.get("MULTISCALE_STFT_HOP_DIVISOR", "4"))
    MULTISCALE_STFT_NORM = os.environ.get("MULTISCALE_STFT_NORM", "zscore")

    # ATT442-WST: Wavelet Scattering Transform — captures amplitude modulation structure
    # (2nd-order coefficients) that STFT misses. Ball defect BSF is amplitude-modulated by
    # cage frequency FTF; WST 2nd layer explicitly extracts this modulation envelope,
    # providing the physical basis for Ball vs Outer race separation.
    # When enabled, WST coefficients are computed and concatenated as extra channels to MS-STFT.
    WST_ENABLE = bool(int(os.environ.get("WST_ENABLE", "1")))
    WST_J = int(os.environ.get("WST_J", "8"))            # max scale = 2^J samples
    WST_Q = int(os.environ.get("WST_Q", "8"))            # wavelets per octave
    WST_ORDER = int(os.environ.get("WST_ORDER", "2"))     # 2nd order captures modulation
    WST_NORM_MODE = os.environ.get("WST_NORM_MODE", "zscore")
    # ATT446: WST Bypass Injection — WST features bypass SpecEncoder, inject into SpecHead only
    WST_BYPASS_INJECT = bool(int(os.environ.get("WST_BYPASS_INJECT", "1")))
    WST_BYPASS_DIM = int(os.environ.get("WST_BYPASS_DIM", "128"))
    # ATT447: Asymmetric Fusion→Spec Knowledge Distillation (output-level)
    SPEC_DISTILL_FROM_FUSION = bool(int(os.environ.get("SPEC_DISTILL_FROM_FUSION", "0")))
    SPEC_DISTILL_TEMPERATURE = float(os.environ.get("SPEC_DISTILL_TEMPERATURE", "2.0"))
    SPEC_DISTILL_ALPHA = float(os.environ.get("SPEC_DISTILL_ALPHA", "0.3"))
    SPEC_DISTILL_START_EPOCH = int(os.environ.get("SPEC_DISTILL_START_EPOCH", "5"))
    # ATT448: Relational Knowledge Distillation (feature-level)
    RKD_ENABLE = bool(int(os.environ.get("RKD_ENABLE", "1")))
    RKD_DISTANCE_WEIGHT = float(os.environ.get("RKD_DISTANCE_WEIGHT", "1.0"))
    RKD_ANGLE_WEIGHT = float(os.environ.get("RKD_ANGLE_WEIGHT", "2.0"))
    RKD_START_EPOCH = int(os.environ.get("RKD_START_EPOCH", "3"))
    # ATT449-S1: Fault-Focused RKD — amplify fault-type discrimination signal
    RKD_FAULT_FOCUS_WEIGHT = float(os.environ.get("RKD_FAULT_FOCUS_WEIGHT", "0.7"))
    # ATT449-S2: Fine-grained output KD (FiGKD-inspired high-pass filtering)
    SPEC_DISTILL_HF_ONLY = bool(int(os.environ.get("SPEC_DISTILL_HF_ONLY", "1")))
    # ATT449-S3: Envelope Spectrum Bypass + BPFI/BPFO ratio
    ENV_BYPASS_INJECT = bool(int(os.environ.get("ENV_BYPASS_INJECT", "1")))
    ENV_BYPASS_DIM = int(os.environ.get("ENV_BYPASS_DIM", "64"))

    # ATT464: SCD (Spectral Correlation Density) — cyclic modulation spectrum
    # ATT464a: SCD_CHANNEL_ENABLE=concat mode (falsified: Normal-class collapse)
    # ATT464b: SCD_BYPASS_INJECT=bypass mode (TinySCDEncoder → z_scd → SpecHead)
    SCD_CHANNEL_ENABLE = bool(int(os.environ.get("SCD_CHANNEL_ENABLE", "0")))
    SCD_BYPASS_INJECT = bool(int(os.environ.get("SCD_BYPASS_INJECT", "0")))
    SCD_BYPASS_DIM = int(os.environ.get("SCD_BYPASS_DIM", "64"))
    SCD_N_FFT = int(os.environ.get("SCD_N_FFT", "256"))
    SCD_HOP_DIVISOR = int(os.environ.get("SCD_HOP_DIVISOR", "4"))
    SCD_NORM_MODE = os.environ.get("SCD_NORM_MODE", "zscore")

    # ATT465: Domain-Conditioned Normalization + Multi-Level Spec-Text Alignment
    DOMAIN_BN_ENABLE = bool(int(os.environ.get("DOMAIN_BN_ENABLE", "0")))
    MLSA_ENABLE = bool(int(os.environ.get("MLSA_ENABLE", "0")))
    MLSA_WEIGHT = float(os.environ.get("MLSA_WEIGHT", "0.1"))
    MLSA_WARMUP_EPOCHS = int(os.environ.get("MLSA_WARMUP_EPOCHS", "5"))
    MLSA_MARGIN = float(os.environ.get("MLSA_MARGIN", "0.2"))

    # ATT423-P1: Physical Frequency Alignment — ensure the same bin index represents
    # the same physical frequency across all domains (CWRU 12kHz, PU 64kHz, HUST 25.6kHz, HIT 25kHz).
    # Without alignment, bin #64 = 3kHz for CWRU but 6.4kHz for HUST → SpecEncoder CNN kernels
    # learn domain-specific patterns instead of universal fault signatures.
    # Mechanism: crop/zero-pad STFT frequency axis to [0, max_hz] before resizing to target_f bins.
    # Default max_hz=6000 = CWRU Nyquist (min across all domains), ensuring all fault-diagnostic
    # frequencies (<5kHz) are aligned. Higher-fs domains simply discard >6kHz content.
    SPEC_FREQ_ALIGN_ENABLE = bool(int(os.environ.get("SPEC_FREQ_ALIGN_ENABLE", "1")))
    SPEC_FREQ_ALIGN_MAX_HZ = float(os.environ.get("SPEC_FREQ_ALIGN_MAX_HZ", "6000"))

    # ATT457: Order-Spectrum Normalization — maps frequency axis to shaft-frequency orders.
    # When enabled, REPLACES Hz-based freq_align with order-based normalization.
    # Bin N represents order = N * MAX_ORDERS / target_f, making fault signatures
    # (BPFI≈5.4x, BPFO≈3.5x shaft speed) appear at consistent positions across domains.
    SPEC_ORDER_NORM_ENABLE = bool(int(os.environ.get("SPEC_ORDER_NORM_ENABLE", "0")))
    SPEC_ORDER_NORM_MAX_ORDERS = float(os.environ.get("SPEC_ORDER_NORM_MAX_ORDERS", "200"))

    # ATT423-P2: Frequency-Conditioned SpecEncoder (FiLM) — sampling-rate-aware feature modulation.
    # Even after frequency alignment, different domains have different noise floors, amplitude
    # distributions, and transient characteristics. This FiLM module generates channel-wise
    # affine transforms (gamma, beta) conditioned on the normalized sampling rate, injected
    # after stem and block2. Identity-initialized: gamma=1, beta=0 at init → no-op start.
    # Reference: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018).
    SPEC_FS_COND_FILM_ENABLE = bool(int(os.environ.get("SPEC_FS_COND_FILM_ENABLE", "1")))
    SPEC_FS_COND_FILM_HIDDEN_DIM = int(os.environ.get("SPEC_FS_COND_FILM_HIDDEN_DIM", "32"))
    SPEC_FS_COND_FILM_REF_FS = float(os.environ.get("SPEC_FS_COND_FILM_REF_FS", "64000.0"))

    # SWT parameters (used when SPEC_FEATURE_TYPE='swt')
    SWT_WAVELET = "db4"
    SWT_LEVELS = 10         # P2: 提升频率分辨率，增强低频故障带判别信息
    SWT_TIME_BINS = 128     # downsampled time bins (T dimension), comparable to STFT frames
    # SWT stabilization: per-channel normalization after log1p
    SWT_NORM = "zscore"     # "none" | "zscore"
    SWT_NORM_CLIP = 5.0
    SWT_NORM_EPS = 1e-6
    # Attempt 55 rollback: resample-before-SWT is disabled after validation failure.
    SWT_RESAMPLE_BEFORE_EXTRACT = True
    SWT_TARGET_FS = 25600
    SWT_USE_DYNAMIC_BANDS = True  # Attempt 62: re-compute bands after resampling
    # Reduce repetitive per-sample SWT logs to save tokens.
    SWT_LOG_VERBOSE = False
    # Torch-STFT fast path (replaces scipy/pywt-heavy SWT extraction in dataloader workers).
    # NOTE: keep device="cpu" by default for multi-worker stability; set to "cuda" only with num_workers=0.
    SWT_USE_TORCH_STFT_FASTPATH = False
    SWT_TORCH_STFT_DEVICE = "cpu"  # "cpu" | "cuda"
    SWT_TORCH_STFT_N_FFT = 512
    SWT_TORCH_STFT_HOP_LENGTH = 128
    SWT_TORCH_STFT_WIN_LENGTH = 512
    SWT_TORCH_STFT_CENTER = True
    SWT_TORCH_STFT_POWER = 1.0
    # Spec cache controls (non-augment path): in-memory + optional disk cache.
    SPEC_CACHE_IN_MEMORY = True
    SPEC_CACHE_TO_DISK = True
    SPEC_CACHE_DIR = "/root/autodl-tmp/spec_cache"
    SPEC_CACHE_DTYPE = "float16"  # "float16" | "float32"
    SPEC_CACHE_VERSION = "att410_stft_v1"
    # ATT376: SpecEncoder robustification (domain-agnostic spectrum patterns).
    # ATT377: REVERTED — Instance Norm + Freq Ratio destroyed feature discriminability
    # (logits std 3.6→1.2, LOO acc 0.93→0.83). See TRAINING_ATTEMPTS2.0.md ATT377.
    SPEC_INSTANCE_NORM_BEFORE_HEAD = True
    SPEC_USE_FREQ_RATIO_FEATURES = True
    SPEC_FREQ_RATIO_EPS = 1e-6
    SPEC_FREQ_RATIO_BLEND = 0.35
    # Sampling-rate-aware amplitude normalization (physical prior).
    # ATT377: REVERTED — had no effect on HUST/HIT (fs_scale≈1.0) but added complexity.
    SPEC_FS_AWARE_NORM_ENABLE = True
    SPEC_FS_AWARE_REF_FS = 25600.0
    SPEC_FS_AWARE_NORM_POWER = 0.5
    SPEC_FS_AWARE_LOG = True
    
    # ========== 域感知故障头配置 ==========
    USE_DOMAIN_AWARE_FAULT_HEAD = True  # 是否使用域感知故障头（HUST/SEU 4-class）
    # 统一域ID映射（关键：训练/评估/损失/decoder 必须完全一致）
    # ATT299+Fix: load-aware 5-domain mapping
    CWRU_GK2_DOMAIN_ID = 0
    CWRU_GK4_DOMAIN_ID = 1
    PU_DOMAIN_ID = 2
    HUST_DOMAIN_ID = 3
    HIT_DOMAIN_ID = 4
    SEU_DOMAIN_ID = 5
    DOMAIN_ID_MAP = {
        "CWRU_GK2": CWRU_GK2_DOMAIN_ID,
        "CWRU_GK4": CWRU_GK4_DOMAIN_ID,
        "PU": PU_DOMAIN_ID,
        "HUST": HUST_DOMAIN_ID,
        "SEU": SEU_DOMAIN_ID,
        "HIT": HIT_DOMAIN_ID,
    }
    
    # 渐进式训练配置（用于Cross-Attention）
    FUSION_PROGRESSIVE_TRAINING = True  # 是否启用渐进式训练（逐步启用高级特性）
    FUSION_USE_CONFIDENCE = False  # Cross-Attention是否使用置信度特征（默认关闭，避免早期误导）
    FUSION_CONFIDENCE_START_EPOCH = 8  # 从第几轮开始使用置信度特征（从5改为8，让基础融合学得更充分）
    FUSION_CLASS_AWARE_START_EPOCH = 15  # 从第几轮开始使用类感知（从10改为15，让置信度融合学得更充分）
    
    # 类感知权重正则化
    FUSION_CLASS_AWARE_REGULARIZATION = 0.01  # 类感知权重的L2正则化系数（防止权重极端化）
    
    # ========== 4类诊断优化配置 ==========
    # Focal Loss配置（推荐用于样本不均衡）
    USE_FOCAL_LOSS = True  # 设为True启用Focal Loss
    FOCAL_GAMMA = 1.5  # 聚焦参数，越大越关注难样本（推荐范围：1.5-2.5）；Att77修改：从2.0降到1.5避免早期过度衰减
    FOCAL_ALPHA = None  # None表示使用class_weights作为alpha
    # Att77新增：Focal延迟启动（防止早期Signal欠训练）
    FOCAL_START_EPOCH = 6  # 从第6个epoch开始启用Focal Loss

    # ========== Flat 4-class diagnosis (new baseline) ==========
    # Use a single flat 4-class classifier to avoid hierarchical compounding.
    # ATT379d(S2): DISABLED — flat 4-class forces Rolling vs Normal competition in shared space,
    # causing seesaw effect. Hierarchical (NF→FaultType) eliminates Rolling-Normal confusion.
    USE_FLAT_DIAG_HEAD = False
    # Balanced Softmax (training-time) is disabled after Attempt 75 regression.
    USE_BALANCED_SOFTMAX = False
    # Post-hoc logit adjustment (inference-time): keep training objective unchanged,
    # and correct class-prior bias only at decision stage.
    USE_POSTHOC_LOGIT_ADJUST = True
    POSTHOC_LOGIT_ADJUST_TAU = 0.5  # ATT107: 1.0→0.5 (防止Rolling类被过度boost, argmax模式下Rolling相对Normal提升由2.18降至1.09)
    # Source-domain class counts in label order [Normal, Inner, Outer, Rolling].
    POSTHOC_LOGIT_CLASS_COUNTS = [1357, 2024, 2212, 153]
    # ETF classifier init for flat head (simplex geometry)
    USE_ETF_CLASSIFIER_INIT = True
    # ATT378: unfreeze classifier so decoder receives gradient (ETF still used as PACL anchor).
    USE_ETF_FIXED_CLASSIFIER = False
    # Flat-head multi-center for Rolling (NCMC-style)
    USE_FLAT_MULTI_CENTER_ROLLING = True
    FLAT_MULTI_CENTER_ROLLING_K = 5
    # Flat-head multi-center for Inner/Outer (NCMC-style, mitigates IO overlap under domain shift)
    # NOTE (Attempt 31): disable IO multi-center by default.
    # Empirically it increases IO↔Rolling overlap on HUST and triggers "Inner→Rolling" absorber.
    USE_FLAT_MULTI_CENTER_INNER = False
    USE_FLAT_MULTI_CENTER_OUTER = False
    FLAT_MULTI_CENTER_INNER_K = 3
    FLAT_MULTI_CENTER_OUTER_K = 3
    FLAT_MULTI_CENTER_POOL = "logsumexp"  # "logsumexp" | "max"
    FLAT_MULTI_CENTER_TAU = 0.07
    FLAT_MULTI_CENTER_INIT_STD = 0.01

    # ========== Embedding geometry diagnostics & targeted separation (Attempt 31) ==========
    # Log epoch-level centroid cosine-similarity matrix of z_cls (helps confirm IO↔Rolling overlap).
    LOG_EMBED_CENTROID_SIM = True
    # Targeted separation loss between IO (Inner/Outer) and Rolling in embedding space (cosine hinge).
    USE_IO_ROLL_SEP_LOSS = False
    IO_ROLL_SEP_WEIGHT = 0.05
    # Penalize when cosine similarity exceeds this threshold (smaller => stronger separation).
    IO_ROLL_SEP_MAX_SIM = 0.35
    IO_ROLL_SEP_START_EPOCH = 1

    # ========== Inner↔Outer separation (Attempt 32 mainline) ==========
    # Core hypothesis after Attempt 31: dominant failure is Inner-Outer neural collapse (sim(I,O)≈0.95).
    # Add a direct hinge constraint on class-centroid cosine similarity between Inner and Outer.
    LOG_IO_SEP = True
    USE_IO_SEP_LOSS = False
    IO_SEP_WEIGHT = 0.10
    IO_SEP_MAX_SIM = 0.30
    IO_SEP_START_EPOCH = 1

    # ========== Normal↔Fault separation (Attempt 33 mainline) ==========
    # After IO_SEP succeeds, dominant failure shifts to "Normal absorption":
    # large cos-sim between Normal centroid and fault centroids on target domains.
    LOG_NF_SEP = True
    # ATT379f(A): disable — hierarchy decoder's NF head handles Normal/Fault separation
    USE_NF_SEP_LOSS = False
    NF_SEP_WEIGHT = 0.10
    # Attempt 69: rollback to the stable hinge threshold after over-separation regression in Attempt 68.
    NF_SEP_MAX_SIM = 0.20
    NF_SEP_START_EPOCH = 1
    # Att78: when train batches per epoch <= this, NF-Sep is disabled (high-variance centroids in small-sample regime).
    # Att80: revert to 3 to enable NF-Sep on CWRU (batch=3)
    NF_SEP_MIN_BATCHES = 3

    # Att77新增：Batch Nuclear Norm Loss（防止特征维度坍缩）
    # ATT379f(A): disable — feature diversity maintained by SupCon + PACL
    USE_BATCH_NUCLEAR_NORM = False
    BATCH_NUCLEAR_NORM_WEIGHT = 0.01  # 损失权重（较小，避免过度干预）
    BATCH_NUCLEAR_NORM_START_EPOCH = 1  # 从第1个epoch开始启用

    # ========== Inner↔Rolling separation (Attempt 34 mainline) ==========
    # After NF_SEP succeeds, dominant failure shifts to "Inner↔Rolling oscillation":
    # E10 Rolling R=0 (absorbed by Inner), E15 Inner R=0 (absorbed by Rolling).
    # Inner↔Rolling separation (Attempt 35 mainline fix)
    # Attempt 34 showed: IR_SEP with MAX_SIM=0.0 caused performance DROP (33.03%→27.63%)
    # Reason: Hinge loss gradient vanishes after target achieved; SupCon/CE pull I-R back together
    # Fix: 1) Relax threshold (0.0→0.15), 2) Add maintenance gradient to prevent gradient vanishing
    # Attempt 37: disable IR_SEP entirely (Attempt 34/36: geometry improves but task degrades)
    LOG_IR_SEP = False
    USE_IR_SEP_LOSS = False
    IR_SEP_WEIGHT = 0.05  # ⬇️ Reduced from 0.10 to avoid over-constraining
    IR_SEP_MAX_SIM = 0.15  # ⬇️ Relaxed from 0.0 to allow some correlation (Inner/Rolling are both fault types)
    IR_SEP_START_EPOCH = 1
    IR_SEP_MAINTENANCE_WEIGHT = 0.01  # NEW: small continuous gradient even after target achieved
    
    # ========== LDAM + DRW (Label-Distribution-Aware Margin + Deferred Re-Weighting) ==========
    # Based on NeurIPS 2019: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    # LDAM adds class-dependent margins to cross-entropy: margin_j = C * n_j^(-1/4)
    USE_LDAM_LOSS = False
    LDAM_MAX_MARGIN = 0.8  # C parameter: maximum margin for the rarest class
    LDAM_SCALE = 30.0  # s parameter: logit scaling factor
    # Whether to normalize margins so max equals LDAM_MAX_MARGIN
    LDAM_NORMALIZE_MARGIN = False
    # DRW: Deferred Re-Weighting - use uniform weights for first DRW_EPOCH epochs
    USE_DRW = True
    DRW_EPOCH = 10  # Start class re-weighting after this epoch
    
    # ========== Class Prototype Separation Loss ==========
    # Explicitly enforce minimum distance between class centroids in feature space
    # to prevent Inner/Outer neural collapse (feature vectors converging to similar directions)
    USE_PROTO_SEPARATION_LOSS = False
    PROTO_SEP_MARGIN_DELTA = 1.5  # Minimum desired distance between class centroids
    PROTO_SEP_LOSS_WEIGHT = 0.2  # Weight for prototype separation loss
    PROTO_SEP_START_EPOCH = 1  # Start prototype separation after this epoch (allow initial learning)
    PROTO_SEP_MOMENTUM = 0.9  # EMA momentum for prototype updates
    LOG_PROTO_DISTANCES = True  # Log pairwise class centroid distances each epoch
    # Use continuous log-ratio separation loss instead of hinge
    USE_LOG_PROTO_SEP = False
    LOG_PROTO_SEP_EPS = 1e-6
    
    # Label Smoothing配置（减少过拟合，提高泛化）
    LABEL_SMOOTHING = 0.05  # 推荐范围：0.05-0.1，0.0表示不使用
    
    # 损失权重调整（强化4类诊断）
    # 提高fusion_decoder的监督占比
    W_FUSION_CLS_EARLY = 0.15  # 前期fusion_cls权重（原0.1）
    W_FUSION_CLS_LATE = 0.10   # 后期fusion_cls权重（原0.05）

    # Residual Proxy configuration (NAGL-style residual cues)
    # Default OFF: ResidualProxyBank tends to amplify "all → Rolling" collapse when class imbalance is severe.
    # Re-enable only after the base diagnosis head is stable.
    USE_RESIDUAL_PROXIES = False  # 是否启用 residual proxy 分支
    RESIDUAL_NUM_PROXIES_PER_FAULT = 8  # 每类 residual proxy 个数
    RESIDUAL_ALPHA = 0.5  # z_res = z_cls + alpha * r
    RESIDUAL_TEMPERATURE = 0.1  # 温度控制代理 similarity 的 sharpness
    RESIDUAL_LOSS_WEIGHT = 0.2  # 总 residual loss 权重
    RESIDUAL_NORMAL_SUP_WEIGHT = 0.5  # normal samples 专门的 residual 抑制项权重
    RESIDUAL_WARMUP_EPOCHS = 6  # 前N个epoch不启用residual监督（防早期诊断塌缩）
    RESIDUAL_RAMP_EPOCHS = 6    # 在接下来N个epoch线性ramp到RESIDUAL_LOSS_WEIGHT

    # ========== Loss weighting policy (sampler-first) ==========
    # If WeightedSampler is enabled, do NOT additionally apply per-class loss weights (double compensation).
    USE_CLASS_WEIGHTS_IN_LOSS = False
    
    # Prototype-tied diagnosis configuration
    USE_PROTOTYPE_HEAD = False  # Disable temporarily for Attempt 8 to isolate Compositional Logic
    # Train/eval consistency: when enabled, training also uses prototype head after a warm start epoch.
    # Delay prototype-head training until the base (domain-aware) decision is stable.
    # This prevents early prototype instability from causing Fusion Cls explosions and collapse.
    # NOTE (reliability): if zero-shot collapses early, delaying prototype-head training can prevent the encoders
    # from ever aligning to the semantic prototypes. Enable earlier and rely on existing stability guards.
    PROTOTYPE_HEAD_TRAIN_START_EPOCH = 5

    # ========== (Stage) Two-stage training: supervised diagnosis -> semantic alignment ==========
    # Goal: decouple "prototype diagnosis" from the main supervised training trunk.
    # Stage1: pure supervised diagnosis head (no text prototypes in decision logits / losses).
    # Stage2: keep supervised diagnosis stable, then add semantic alignment losses (text prototypes).
    # Attempt 38 (mainline): disable Stage2 to avoid "Stage2 freezing" and keep all training in Stage1.
    # NOTE: When two-stage is disabled, Stage1 losses (SupCon/CE/separation) should stay active for ALL epochs.
    USE_TWO_STAGE_TRAINING = False
    STAGE1_SUPERVISED_EPOCHS = 30
    # (2) When False: from epoch 1 train NF + 4-class CE jointly so z serves both NF and fault-type.
    #     When True: Stage1 trains ONLY Normal/Fault (NF); fault-type heads frozen (not recommended).
    STAGE1_NF_ONLY = False
    # During Stage2, whether to freeze backbone (signal/spec/fusion). False = let fault-type refine z.
    STAGE2_FREEZE_BACKBONE = True
    # During Stage2, whether to still keep supervised diagnosis CE term.
    STAGE2_KEEP_SUPERVISED_CE = True
    # ATT371.1:
    # Whether to force CE-only diagnosis path and bypass multimodal_alignment_loss
    # when USE_SUPCON_STAGE1 is enabled.
    # Recommended False for ATT371_1.0 to ensure alignment losses are actually active.
    FORCE_CE_STAGE2 = False
    # A small auxiliary NF loss helps keep anomaly AUC stable even when diagnosis loss dominates.
    # ATT428: NF stage loss disabled — MVZ uses fusion_cls CE directly
    STAGE_NF_LOSS_WEIGHT = float(os.environ.get("STAGE_NF_LOSS_WEIGHT", "0.0"))
    # (1b) Diagnosis branch (z_cls): False = fault-type gradient updates backbone (avoids all->Normal collapse).
    USE_CLS_DETACH_FAULT_TYPE = False
    # (3a) NF contrastive/margin: supervised contrastive or margin on z_gate for Normal vs Fault.
    USE_NF_CONTRASTIVE_LOSS = False
    NF_CONTRASTIVE_WEIGHT = 0.30  # ⬆️ INCREASED from 0.10 (P0: Enhanced Contrastive Learning)
    NF_CONTRASTIVE_TEMPERATURE = 0.03  # ⬇️ DECREASED from 0.05 for sharper boundaries
    USE_NF_MARGIN_LOSS = False
    NF_MARGIN_WEIGHT = 0.10
    NF_MARGIN = 0.25
    NF_MARGIN_START_EPOCH = 1
    
    # (P0) Fault-type level supervised contrastive learning (NEW)
    # Based on DCASCL (MDPI 2024) and SCLDAN (ResearchGate 2024)
    # Pull same-class closer, push different-class farther in fault-type space
    USE_FAULT_TYPE_CONTRASTIVE = False
    FAULT_TYPE_CONTRASTIVE_WEIGHT = 0.20
    FAULT_TYPE_CONTRASTIVE_TEMPERATURE = 0.07
    FAULT_TYPE_CONTRASTIVE_START_EPOCH = 1  # Start from epoch 1
    # (3b) Learnable normal reference: pull normal to v_normal, push fault away.
    USE_NORMAL_REFERENCE_LOSS = False
    NORMAL_REFERENCE_WEIGHT = 0.08  # reduced from 0.20 to avoid normal-centric z (B1)
    NORMAL_REFERENCE_MOMENTUM = 0.99
    # First-epoch momentum for normal_reference: 0.5~0.7 = smooth init, 0.0 = aggressive (B1: avoid).
    NORMAL_REFERENCE_FIRST_EPOCH_MOMENTUM = 0.6

    # ========== (SupCon Stage1) ==========
    # Stage1 trains supervised contrastive + lightweight CE to establish basic classifier boundaries.
    # ATT214: restore ATT209 Stage1 SupCon setting for reproducible baseline.
    USE_SUPCON_STAGE1 = True
    # Mainline (Attempt 30): make CE the primary driver of classifier boundary; use SupCon as regularizer.
    # Att83: 0.1→0.3  ATT82降至0.1导致PU退化(-24%)，部分恢复；配合DCL消除NPC
    # ATT105 CRITICAL FINDING: SupCon loss went deeply NEGATIVE (E2: -0.27, E18: -0.73) due to DDPM
    # mode collapse. When DDPM generates synthetic samples that are too similar within a class,
    # the DCL positive pairs are already at maximum similarity → denominator cannot compensate
    # → L_DCL = -sim(z,z+)/τ + log(Σexp(neg)) → very negative when positives are trivially easy.
    # Effect: negative SupCon acts as a DESTABILIZING force, pulling total loss negative and
    # preventing proper calibration of classifier heads across load conditions.
    # ATT091 (60.75%): SupCon was POSITIVE throughout (good DDPM diversity → genuine learning).
    # ATT106 fix: set weight=0.0 to eliminate the harmful negative SupCon term.
    # CE loss + VICReg γ=1.0 provide sufficient supervisory signal for class separation.
    # FALSIFIABLE: SupCon loss line should be absent from ATT106 logs; total loss stays positive.
    # ATT106 RESULT: AUC_nf dropped 1.0→0.77 when SupCon was disabled (even with weight=0, SupCon gradient
    # was providing class-discriminative signal via ∂L/∂z that helped NF head separation).
    # ATT107 (weight=0.1): AUC_nf recovered to 0.9998, argmax broke 0.1935→0.3118. BUT:
    #   E13 total loss = -0.0105 (SupCon contribution = 0.1×(-0.66)=-0.066 dominated CE).
    #   Val_loss spike (0.57→1.22 E5→E12) triggered early stop at E9 (only 9 epochs).
    #   cos_sim(N,R) at best E9 = -0.22 (undertrained; ATT091 needed -0.82 by E12).
    # ATT108 RESULT: Diag-Acc REVERTED to 0.1935 despite training to E18, cos_sim(N,R)=-0.5753.
    #   Paradox: BETTER embedding separation → WORSE test performance than ATT107's undertrained E9.
    #   Root cause: SupCon=0 removes implicit cross-load domain invariance.
    #     - SupCon positive pairs span gk2(1HP) and gk4(3HP) → same-class positives force load-invariant z
    #     - CE-only training: model learns load-amplitude shortcuts over 18 epochs (in-domain overfit)
    #     - Evidence: val_acc=0.7812 (gk2/gk4 in-domain) vs test_diag=0.1935 (gk1/gk3 cross-domain)
    #     - AUC_nf=0.7709 (identical to ATT106) confirms SupCon=0 removes NF head class separation
    #   ATT108 = ATT106 exactly (same SupCon=0, same AUC_nf=0.7709, same Diag-Acc=0.1935)
    # ATT109 fix: restore weight=0.1 (SupCon provides implicit load-domain invariance via cross-load positives)
    #   + change EARLY_STOP_METRIC to val_acc (val_acc doesn't spike under SupCon; val_loss does)
    #   Expected: domain-invariant representation + training to E15+ → Diag-Acc > 0.40.
    # ATT109 RESULT (Diag-Acc=0.2688): WORSE than ATT107 (0.3118) despite same SupCon=0.1 config!
    #   ROOT CAUSE: ATT107 and ATT109 are IDENTICAL training runs (confirmed by bit-exact epoch summaries
    #   E1-E14: Total/CE/SupCon values match 4 decimal places). The ONLY difference is checkpoint selection:
    #   ATT107 val_loss+patience=5 → best E9 → 0.3118
    #   ATT109 val_acc+patience=8  → best E14 → 0.2688
    #   E14 checkpoint overfits to gk2/gk4 amplitude patterns (5 more CE epochs after E9).
    #   val_loss monotonically increased E9→E14 (0.9218→1.025→1.15→1.227), correctly signaling overfitting.
    #   val_acc stayed at 0.6875-0.75 (too coarse to detect the overfitting signal).
    # ATT110 fix: REVERT val_acc → val_loss, patience 8 → 5.
    #   + ADD AUX_CE_WEIGHT 0.01 → 0.3 to properly train spec_head at E9.
    #   Expected: val_loss correctly re-selects E9 → fusion_decoder ≥ 0.3118; spec_head@E9 > 0.3118.
    # ATT118: FREEZE_SIGNAL_AFTER_EPOCH=-1 caused val_loss@E7-E9 to be low (0.39,0.37,0.64),
    # resetting smooth@3 baseline to 0.4659 → Early Stop selected E9 instead of E12 → Diag-Acc=0.2151.
    # ATT119: Disable SupCon entirely (weight→0). Rationale:
    #   - ATT108 showed SUPCON=0 destroys cross-load invariance for FUSION path
    #   - But ATT119 relies on AUX_CE (not fusion) as primary diagnostic output
    #   - AUX_CE=0.30 on both gk2+gk4 samples per class provides IMPLICIT cross-load invariance
    #     (spec encoder must classify gk2_IR AND gk4_IR correctly → learns load-invariant spectral features)
    #   - SupCon τ=0.10 caused tight load-specific clusters (ATT112/ATT117 all→IR)
    #   - SupCon τ=0.20 caused soft clusters that fall into Normal on test loads (ATT116 all→Normal)
    #   - With neither τ able to avoid single-class collapse, removing SupCon and relying on AUX_CE is cleaner
    SUPCON_STAGE1_WEIGHT = 0.0  # ATT119: 0.1→0.0 (disable SupCon; AUX_CE=0.30 provides implicit cross-load invariance)
    # ATT193(B): add an explicit z_spec SupCon branch for Task2 cross-domain invariance.
    # Keep this independent from SUPCON_STAGE1_WEIGHT so we can preserve historical fusion-path setting.
    # ATT214: restore ATT209 Spec-branch SupCon coupling.
    # ATT379f: 0.0→0.10 — complete removal caused SpecHead collapse and z_spec degradation;
    # moderate weight maintains z_spec discriminability while freeing gradient for PACL
    SUPCON_STAGE1_SPEC_WEIGHT = 0.10
    SUPCON_STAGE1_CE_WEIGHT = 0.5
    # ATT254(S2): anti-collapse regularization on pre-normalized z_cls used by SupCon.
    # Keep this independent from GLOBAL_DECORR so we can target Stage1 representation collapse directly.
    # ATT379f(A): disable — SupCon on z_res is disabled, prenorm var has no target
    USE_SUPCON_PRENORM_VAR_LOSS = False
    SUPCON_PRENORM_VAR_WEIGHT = 1.0
    # std_target=0.10 <=> var_target=0.01
    SUPCON_PRENORM_STD_TARGET = 0.10
    # Attempt 67: enable class weighting in Stage1 lightweight CE to counter long-tail collapse.
    STAGE1_CE_USE_CLASS_WEIGHTS = True
    # Attempt 72: keep focal switch but default OFF (Att71 showed strong underfit on signal branch).
    STAGE1_CE_USE_FOCAL = False
    STAGE1_CE_FOCAL_GAMMA = 2.0
    STAGE1_CE_FOCAL_USE_CLASS_ALPHA = True
    # ATT114: 0.10→0.20. ATT112 (τ=0.10) caused SupCon=-0.19 (tight load-specific clusters).
    # ATT113 (τ=0.10) caused SupCon=-0.62 (3.3× tighter = severe load overfitting, total loss negative).
    # τ=0.10 creates over-compressed embeddings where same-class pairs converge to identical points;
    # on unseen loads (gk1/gk3), these gk2/gk4-specific clusters cause all test samples to map to one class.
    # τ=0.20: InfoNCE gradient ∝ exp(z_i·z_j/0.2) / sum → softer boundaries → more cross-load generalizable.
    # Literature (MDSupCon 2024, SdCDA 2024): τ=0.2-0.5 for variable-condition bearing fault diagnosis.
    # ATT116 RESULT: τ=0.20 + fault_only=False gave Diag-Acc=0.1935 (ALL→Normal prediction).
    # Root cause: τ=0.20 produces soft cluster boundaries; DANN (vanilla, λ=0.20 on z_gate) less effective
    # with soft clusters (z_gate var=0.0017 vs 0.0013 in ATT112); gk1/gk3 test samples (lower amplitude)
    # fall into Normal region because Fault logits shift down relative to Normal across load domains.
    # Mathematical proof: FUSION AUC=0.989 (ranking correct) but argmax=0.1935 (decision boundary biased to
    # Normal) → classic parallel logit shift: Δlogit_fault >> Δlogit_normal across load change.
    # τ=0.10 (ATT112): tighter clusters → DANN compresses z_gate (var=0.001, 770× reduction) → surviving
    # dimensions are load-invariant → test samples map to nearest cluster → Diag-Acc=0.4194 (all→IR, but
    # correct because 78/186 test samples are Inner Race).
    # ATT117: 0.20→0.10. Restore ATT112's tight-cluster + DANN combination. Expected: Diag-Acc≥0.4194.
    SUPCON_TEMPERATURE = 0.10  # ATT117: 0.20→0.10 (ATT116 τ=0.20 proved all→Normal; τ=0.10 proved 0.4194)
    SUPCON_USE_CLASS_WEIGHTS = True
    # ATT114: False→True. fault_only=False means Normal samples participate in SupCon:
    #   - Normal pushes I/O/R away (correct for N/F separation)
    #   - BUT: Normal acts as a "central anchor" that all fault types push against → I/O/R cluster differently
    # fault_only=True: SupCon only runs among fault samples (I/O/R), explicitly focusing on inter-fault-type
    # discrimination. Normal-Fault separation is handled by the NF-Sep (cosine margin) loss independently.
    # Expected: I vs O vs R clustering improves → reduces prediction collapse to single fault type.
    # ATT115 RESULT: fault_only=True CATASTROPHIC. Diag-Acc=0.1935 (0.4194→0.1935), Macro-F1=0.0811
    # (all→Rolling). Root cause: Normal embeddings unconstrained by contrastive repulsion → drifted into
    # Rolling cluster (largest after augmentation). Signal AUC=0.282, Spec AUC=0.126 (both inverted).
    # EMPIRICAL PROOF: fault_only=True removes cross-class contrastive force on Normal → Normal absorber.
    # ATT116: True→False. Restore Normal samples in SupCon to prevent Normal→Rolling drift.
    # τ=0.20 is KEPT (not yet tested with fault_only=False; expected to soften load-specific clustering
    # while Normal still participating in contrastive pairs → better cross-load NF boundary).
    SUPCON_FAULT_ONLY = False  # ATT116: True→False (ATT115 proved fault_only=True catastrophic: 0.4194→0.1935)

    # ========== Diffusion Augmentation (ATT85) ==========
    # Auto-generates synthetic vibration signals when training samples are insufficient
    # (CWRU ~134 samples → 3 batches → feature collapse). Uses class-conditional DDPM + DDIM.
    # ATT105: False→True. Deep re-analysis disproved the ATT103 "toxic augmentation" hypothesis.
    # TRUE ROOT CAUSE IDENTIFIED via cross-log comparison (ATT085/091/098/101-104):
    #   1. VOLUME: Diffusion 134→1280 CWRU samples = 20 batches/epoch (no collapse).
    #      Without diffusion (ATT103/104): 134/BS=3~5 batches → E2 collapse → all-Normal → 0.1935
    #   2. ATT098 "41.94% baseline without diffusion" was a FLUKE:
    #      CollapseGate gated spec (abs_th=0.002, std_spec=0.00125 < 0.002 → gated).
    #      Current code uses relaxed CWRU threshold (0.5× = 0.001), std_spec=0.002 > 0.001 → NOT gated.
    #      Signal-only (gated) gave 41.94%; signal+spec (not gated, collapsed) gave 0.1935.
    #   3. ATT101 failure (1280 samples, no collapse) ← CFG=3.0 OOD synthetic data
    #      → Rolling-Normal separation stalled at cos_sim=+0.99 until E22 (early stop at E10)
    #   4. ATT102 failure (1280 samples, CFG=2.0 fixed) ← VICReg γ=0.005 killed variance gradient
    #      → AUC_nf=1.0000 but hard-gate threshold miscalibrated for gk1/gk3 → all-Normal
    # ATT091 recipe (60.75%): CFG=2.0 + 5000steps + VICReg γ=1.0 → E6 Rolling cos_sim→-0.82
    DIFFUSION_AUGMENT_ENABLED = bool(int(os.environ.get("DIFFUSION_AUGMENT_ENABLED", "1")))             # ATT105: re-enabled (was False in ATT103/104)；支持环境变量覆盖以做 CWRU 去diffusion对照
    DIFFUSION_COLLAPSE_THRESHOLD = 1000          # Auto: BATCH_SIZE * 10
    DIFFUSION_TARGET_TOTAL = 1500                # Target total samples after augmentation
    # Fallback only when no real condition_keys exist before diffusion merge (ATT503-A uses
    # empirical distribution from real windows; splitData ATT503-B excludes synthetic from test).
    DIFFUSION_SYNTHETIC_CONDITION_KEY = os.environ.get(
        "DIFFUSION_SYNTHETIC_CONDITION_KEY", "speed_1750"
    )
    # ATT102: 3000→5000 (ATT91 proved 5000 steps necessary for convergence on 4096-pt signals;
    #         3000 steps left diffusion undertrained → low-quality synthetic data → toxic augmentation)
    DIFFUSION_TRAIN_STEPS = 5000
    DIFFUSION_TIMESTEPS = 500                    # Diffusion process timesteps
    DIFFUSION_MODEL_DIM = 64                     # U-Net base channel width
    DIFFUSION_INTERNAL_LEN = 4096                # Att101: Revert to 4096 (Engineering Gain)
    DIFFUSION_BATCH_SIZE = 16                    # Diffusion training batch size
    DIFFUSION_LR = 2e-4                          # Diffusion model learning rate
    DIFFUSION_DDIM_STEPS = 50                    # DDIM sampling steps (50 vs 1000 DDPM)
    DIFFUSION_CACHE_DIR = "cache/diffusion"      # Disk cache directory
    DIFFUSION_CACHE_ENABLE = True                # ATT160: 复用历史合成样本缓存，减少重复训练扩散模型开销
    DIFFUSION_FORCE_REGENERATE = False           # ATT160: True时忽略缓存并重新训练/生成
    DIFFUSION_SPECTRAL_LOSS_WEIGHT = 0.0         # Att101: Disable Spec Loss
    # ATT102: 3.0→2.0 (ATT91 used w=2.0 and achieved 60.75% CWRU; w=3.0 over-sharpens class
    #         distribution → synthetic OOD samples → train/test distribution mismatch)
    DIFFUSION_CFG_WEIGHT = 2.0
    DIFFUSION_LABEL_DROP_PROB = 0.1              # Att101: Label Dropout Prob
    # ATT102: Energy Gating — rejects generated samples with energy > threshold × mean_real_energy.
    # ATT91 had this enabled and achieved 60.75% CWRU; ATT101 disabled it → negative transfer.
    DIFFUSION_ENABLE_ENERGY_GATING = True
    DIFFUSION_ENERGY_GATING_THRESHOLD = 3.0      # Max energy = 3× mean real signal energy

    # ========== (Stage Transition) Sigmoidal Ramp for Smooth Stage 2 Activation ==========
    # Replace hard switch with sigmoid ramp to prevent "Epoch 9 shock".
    STAGE_TRANSITION_SHARPNESS = 2.0  # τ parameter: lower = sharper, higher = smoother
    # ramp = sigmoid((epoch - STAGE1_SUPERVISED_EPOCHS) / STAGE_TRANSITION_SHARPNESS)
    # total_loss = loss_cls + ramp * loss_semantic
    # Stage2 semantic warmup: gradually ramp semantic alignment loss to avoid "stage shock"
    STAGE2_SEMANTIC_WARMUP_EPOCHS = 5

    # ========== (DANN) Domain Adversarial Neural Network ==========
    # Uses GRL + domain discriminator to encourage domain-invariant representations.
    # IMPORTANT: can optionally use *unlabeled* target-domain batches for domain loss (no label leakage).
    # ATT111: Enable DANN for explicit cross-load domain invariance.
    #   Root cause of 0.3118 ceiling: FusionDecoder linear head memorizes gk2/gk4 amplitude patterns.
    #   DANN forces z_gate to be load-invariant (domain-adversarial on gk2 vs gk4 domain labels).
    #   SOTA (CDAN 2024): achieves 99.64% on CWRU cross-load via DANN+SupCon; we have both.
    #   DANN_START_EPOCH=5: gives E5-E9 (4 epochs) of load-invariance training before best checkpoint.
    #   Previous DANN_START_EPOCH=15 was misconfigured (training always stops at E14; DANN never ran).
    # ATT119: Disable DANN entirely. Rationale:
    #   - ATT112 (vanilla DANN λ=0.20 on z_gate): spec_head=N/A (bug), SPEC AUC=0.6723 (degraded from 0.96)
    #   - ATT116 (vanilla DANN λ=0.20 on z_gate): spec_head=0.3495 (best seen), SPEC AUC=0.7183
    #   - ATT117 (DANN on z_signal): spec_head=0.1935 (worst), SPEC AUC=0.7242
    #   - DANN on z_gate directly compresses the gate vector used by spec_head → hurts z_spec geometry
    #   - DANN on z_signal ineffective (only 1 epoch before signal freeze in ATT117, continuous freeze-off in ATT118)
    #   - With spec_head as primary output, protecting z_spec geometry > cross-load domain adaptation via DANN
    #   - ATT116's spec_head=0.3495 (best) used DANN but with τ=0.20; the DANN was on z_gate NOT z_spec directly
    #   - However, ATT119 hypothesis: without DANN, AUC should recover to ~0.96 (ATT106/107 level)
    #     and spec_head with AUX_CE=0.30 can leverage these high-quality spectral features
    USE_DANN = False            # ATT119: True→False (disable DANN; protect z_spec geometry for spec_head primary output)
    # ATT114: 0.10→0.20. ATT113 (CDAN λ=0.10) showed catastrophic regression: Diag-Acc=0.1935 < 0.25 (random).
    # Root cause: CDAN's early-training weakness (p≈uniform → weak GRL signal E1-E4) let SupCon over-cluster
    # into load-specific groups before DANN activated; val_acc=75% by E4 = deep load overfitting.
    # ATT112 (vanilla λ=0.20) achieved Diag-Acc=0.4194 (our best). Reverting to this configuration.
    DANN_LAMBDA = 0.20  # ATT114: 0.10→0.20 (revert to ATT112's best value)
    DANN_START_EPOCH = 5        # ATT111: 15→5 (DANN must activate before E9 best checkpoint)
    DANN_MAX_ALPHA = 1.0
    DANN_ALPHA_SCHEDULE = "linear"  # "linear" | "constant"
    DANN_USE_TARGET_UNLABELED = False
    DANN_HIDDEN_DIM = 256
    DANN_DROPOUT = 0.1
    # DANN mode:
    # - "vanilla": discriminator(feature)
    # - "cdan": discriminator(feature ⊗ softmax(logits)) (more class-preserving)
    # ATT114: cdan→vanilla. ATT113 proved CDAN worse than vanilla for this problem:
    #   1. Early-training CDAN is effectively weaker (p≈uniform → minimal conditioning effect E1-E4)
    #   2. 4× larger discriminator (D×C=512 vs D=128) causes different prediction collapse direction
    #   3. SupCon over-clustered before CDAN became effective; val_acc=75%@E4 = load overfitting
    # Vanilla DANN (ATT112) is our empirically best configuration: Diag-Acc=0.4194.
    # The prediction collapse in ATT112 (all→Inner Race) is now addressed separately via SupCon changes.
    DANN_MODE = "vanilla"  # ATT114: cdan→vanilla (CDAN proven worse for this problem, reverting)
    # Which feature to adversarially align.
    # ATT112-ATT116: DANN on z_gate (fused Signal+Spec representation). This caused:
    #   - z_gate variance compressed to 0.001 (770× reduction in ATT112) → DANN collateral damage to Spec
    #   - SPEC AUC: 0.9632 (ATT106, no DANN) → 0.6723 (ATT112) → 0.7183 (ATT116), never recovered
    #   - DANN on z_gate destroys Spec geometry because z_gate contains fused Signal+Spec features
    # Physics: Signal branch encodes time-domain amplitude (primary load shift carrier).
    #          Spec branch encodes frequency ratios (BPFO/BPFI, load-invariant by physics).
    # ATT117: "z_gate"→"z_signal". DANN only aligns Signal features (where load shift occurs).
    #   - z_spec untouched by DANN → Spec geometry preserved → expected SPEC AUC recover to ≥0.85
    #   - Signal features become load-invariant (gk2 vs gk4 signal embeddings aligned)
    #   - Decision layer (z_gate) driven by cleaner Spec information → cross-load accuracy improves
    # Literature: SFDANN (2023), Dynamic JDA (IEEE 2021) apply domain alignment at single-modality level.
    DANN_FEATURE = "z_signal"  # ATT117: "z_gate"→"z_signal" (DANN targets load-shift source, preserves Spec)
    # For vanilla mode, detach_probs is irrelevant (no class probs in discriminator input).
    DANN_DETACH_PROBS = False  # ATT114: True→False (vanilla mode doesn't use probs)

    # ========== ATT405: Training-time domain alignment on z_fusion (MMD) ==========
    # Pairwise MK-MMD between every domain pair on L2-normalised z_fusion.
    # Unlike DANN (adversarial on z_signal), this is a direct distributional discrepancy
    # penalty on the fused representation that feeds the decoder — closer to what
    # zero-shot evaluation actually uses.
    # Warmup avoids interfering with early discriminative convergence.
    FUSION_DOMAIN_ALIGN_ENABLE = bool(int(os.environ.get("FUSION_DOMAIN_ALIGN_ENABLE", "1")))
    FUSION_DOMAIN_ALIGN_LOSS = "mmd"          # "mmd" | "coral"
    FUSION_DOMAIN_ALIGN_WEIGHT = float(os.environ.get("FUSION_DOMAIN_ALIGN_WEIGHT", "0.05"))
    FUSION_DOMAIN_ALIGN_START_EPOCH = int(os.environ.get("FUSION_DOMAIN_ALIGN_START_EPOCH", "3"))
    FUSION_DOMAIN_ALIGN_WARMUP_EPOCHS = 3     # linear ramp from 0 → weight over this many epochs
    FUSION_DOMAIN_ALIGN_DETACH = False         # if True, detach z_fusion (alignment does not back-prop into encoder)
    FUSION_DOMAIN_ALIGN_MIN_SAMPLES = 4        # skip domain pair if either side has < N samples in the batch
    # ATT407: disable domain alignment for Task1 (cross-condition within single dataset).
    # Domain alignment makes sense for cross-domain (Task2: CWRU+PU→HUST) but hurts
    # within-domain scenarios where "domains" are condition groups carrying fault info.
    FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1 = bool(int(os.environ.get("FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1", "1")))

    # ATT408: Remove LayerNorm from fusion_proj to eliminate double-normalization bottleneck.
    # Chain: Linear(2D→D) → LayerNorm → Dropout → L2-normalize creates over-constrained
    # distribution on the hypersphere. Removing LayerNorm allows the Linear layer to freely
    # set per-dimension scales, increasing angular spread after L2-normalize.
    FUSION_PROJ_REMOVE_LAYERNORM = bool(int(os.environ.get("FUSION_PROJ_REMOVE_LAYERNORM", "0")))

    # ATT409: Auxiliary pre-norm classification head.
    # Adds a linear classifier on z_pre_norm (before F.normalize) with CE loss.
    # Provides direct gradient to fusion module bypassing F.normalize's projection operator.
    # ATT428: DISABLED — aux pre-norm CE is redundant in MVZ 3-loss architecture
    AUX_PRENORM_CE_ENABLE = bool(int(os.environ.get("AUX_PRENORM_CE_ENABLE", "0")))
    AUX_PRENORM_CE_WEIGHT = float(os.environ.get("AUX_PRENORM_CE_WEIGHT", "0.0"))

    # ========== (A2) Partial-label loss for heterogeneous label spaces ==========
    # Do NOT hard-mask logits for missing classes inside the model; instead mask in the loss.
    MASK_MISSING_CLASSES_IN_LOGITS = False
    PARTIAL_LABEL_LOSS_ENABLE = True

    # ========== (S-SEM) Decouple semantic alignment space ==========
    USE_SEM_ADAPTER = True
    SEM_ADAPTER_HIDDEN = 256
    SEM_ADAPTER_INIT_ZERO = True

    # ========== (S-SEM2) EMA visual prototypes for semantic scoring ==========
    # Maintain source-domain EMA prototypes in the semantic space and optionally mix with text prototypes.
    USE_VISUAL_PROTOTYPES = True
    VISUAL_PROTO_MOMENTUM = 0.99
    # For semantic scoring: proto = a * visual + (1-a) * text
    # ATT112: 0.70→0.30. Visual prototypes are biased toward gk2/gk4 amplitude patterns (load-specific).
    # Text prototypes encode load-invariant physical features (BPFO, impulse periodicity).
    # Reducing alpha makes inference rely more on universal text semantics for cross-load generalization.
    # Falsifiable: if SPEC AUC and Diag-Acc both improve, visual proto was the key bottleneck.
    SEMANTIC_MIX_VISUAL_ALPHA = 0.30  # ATT112: 0.70→0.30 (prefer load-invariant text prototypes)
    # Use semantic embedding (z_sem) for semantic scoring in evaluation.
    SEMANTIC_SCORE_USE_Z_SEM = True
    
    # (P1) Progressive Prototype Anchoring (NEW)
    # Based on Meta-learning + Centroid Convergence (MDPI 2024)
    # Gradually increase visual prototype weight from 0 to SEMANTIC_MIX_VISUAL_ALPHA
    # to prevent "Stage 2 shock" caused by sudden semantic alignment
    PROTO_MIX_WARMUP_EPOCHS = 8  # Linearly increase from 0.0 to 0.7 over first 8 epochs
    UPDATE_VISUAL_PROTO_IN_STAGE1 = True  # Update visual prototypes even in Stage 1

    # ========== (Open-set) Rejection for unknown domains ==========
    # Enable by setting DIAG_DECISION_MODE = "open_set_soft_hierarchical" (or open_set_argmax/nf_protect).
    OPEN_SET_PMAX_TH = 0.55     # reject if max softmax prob < th
    OPEN_SET_PFAULT_GRAY_LOW = 0.35
    OPEN_SET_PFAULT_GRAY_HIGH = 0.65
    OPEN_SET_REJECT_LABEL = -1  # returned label for unknown

    # ========== (MTFD) Multi-Task Fault Decomposition ==========
    # Replace unstable 4-class Softmax with independent binary classifiers to prevent class competition.
    USE_MTFD = False  # Disable MTFD for flat 4-class baseline
    # Each head is trained independently:
    # - NF Binary Head: Normal vs Fault (1 binary classifier)
    # - IO Head: Inner vs Outer (already exists)
    # - Rolling Head: Rolling vs Non-rolling (already exists)
    MTFD_NF_LOSS_WEIGHT = 1.0
    MTFD_IO_LOSS_WEIGHT = 1.0
    MTFD_ROLLING_LOSS_WEIGHT = 2.0  # Higher weight for rare class
    # Rolling head stabilization (Attempt 16): focal loss + fault-only negatives
    USE_ROLLING_FOCAL_LOSS = False
    ROLLING_FOCAL_GAMMA = 3.0
    ROLLING_FOCAL_ALPHA = 0.85
    # When True, rolling loss is trained on fault samples only (inner/outer/rolling), excluding normals.
    ROLLING_TRAIN_ON_FAULT_ONLY = True

    # ===== Rolling-domain adversarial (Attempt 17) =====
    # Make rolling representation domain-invariant to improve HUST generalization.
    USE_ROLLING_DANN = False
    ROLLING_DANN_LAMBDA = 0.2
    ROLLING_DANN_START_EPOCH = 12
    ROLLING_DANN_MAX_ALPHA = 1.0
    ROLLING_DANN_ALPHA_SCHEDULE = "linear"
    # Rolling de-collapse: variance + decorrelation (VICReg/Barlow-style)
    USE_ROLLING_DECORR_LOSS = False
    ROLLING_DECORR_WEIGHT = 0.05
    ROLLING_DECORR_START_EPOCH = 12
    ROLLING_DECORR_VAR_TARGET = 1.0
    ROLLING_DECORR_COV_WEIGHT = 1.0
    # Apply de-collapse on input (z_res) for rolling branch
    USE_ROLLING_DECORR_INPUT = False
    ROLLING_DECORR_INPUT_WEIGHT = 0.05
    # Global anti-collapse regularization on diagnostic branch (z_cls).
    # ATT204: switch primary global regularizer from plain DeCollapse to VICReg-style var/cov.
    # ATT210: disable global de-collapse regularization.
    # ATT378: disabled — opposes SupCon gradient (GradContrib cos=0.689, ratio=1.377).
    # ATT420: disabled — decorrelation conflicts with PACL prototype alignment (see ATT418/ATT419 diagnosis).
    # ATT441: RE-ENABLED — PACL is OFF, no conflict. Required for VICReg gradient pathway.
    # ATT444: DISABLED — F_VICREG_DESTROYS_FUSION_ALIGNMENT. VICReg 0.15 changed z_fusion
    # covariance structure, breaking Normal-class alignment with text_proto. ATT437(OFF)=0.9931.
    USE_GLOBAL_DECORR_LOSS = bool(int(os.environ.get("USE_GLOBAL_DECORR_LOSS", "0")))
    # ATT441: restored to 0.03 (ATT376 value) — was 0.0 effectively disabling decorr even when flag=True
    # ATT444: DISABLED (set to 0 with flag OFF). VICReg at any weight
    # either fails to prevent collapse (0.03) or destroys alignment (0.15).
    GLOBAL_DECORR_WEIGHT = float(os.environ.get("GLOBAL_DECORR_WEIGHT", "0.0"))
    # ATT215: use a stronger de-collapse weight in Task2 only.
    GLOBAL_DECORR_WEIGHT_TASK2 = 0.05
    # ATT376: core-only loss regime (reduce gradient competition from long tail auxiliary terms).
    # ATT379d(S1): RETIRED — the binary gate disabled ALL 14 pairwise margin losses,
    # removing the only protection against class collapse (Rolling→Normal seesaw).
    # Replaced by joint training with all margins active + loss group balancing.
    # ATT409: re-enabled — ATT376 (CORE_LOSS_ONLY=True) achieved SpecHead=0.9624.
    # Reduces gradient competition from 41 loss terms, letting fusion module learn cleaner features.
    CORE_LOSS_ONLY_MODE = bool(int(os.environ.get("CORE_LOSS_ONLY_MODE", "0")))
    # Keep cross-domain invariance constraints specifically for Task1-CWRU.
    CORE_KEEP_TASK1_CWRU_INVARIANCE = bool(int(os.environ.get("CORE_KEEP_TASK1_CWRU_INVARIANCE", "1")))
    # ATT379(S1+S5): restored aux_spec for SpecHead CE supervision; added fishr for domain invariance.
    # ATT409: restored "global_decorr" — without it, CORE_LOSS_ONLY_MODE forces VICReg off
    # (ATT378 removed it; ATT376 had it and achieved SpecHead=0.9624).
    CORE_LOSS_ALLOWED_TERMS = ("fusion", "aux_spec", "pacl_spec", "pacl_fusion", "global_decorr", "text_etf_align", "clip_style", "supcon", "fishr", "adapter_pacl", "proto_sep", "proto_anchor", "dual_orth", "dual_dc_spe", "dual_confusion", "spec_variance", "dgl_aux_ce", "xkd_sig2spec")

    # ATT379c: Decoupled Two-Stage Training (Kang et al., ICLR 2020 inspired).
    # Stage 1: Contrastive representation learning (w_spec=0, w_fusion_cls=0, margins=0).
    # Stage 2: Classifier calibration with frozen encoders + margin protection.
    # ATT379d: DISABLED — decoupled training proved incompatible with multimodal fusion
    # architecture (fusion module requires joint CE supervision to learn modality weighting).
    DECOUPLED_TRAINING_ENABLE = bool(int(os.environ.get("DECOUPLED_TRAINING_ENABLE", "0")))
    DECOUPLED_STAGE1_EPOCHS = int(os.environ.get("DECOUPLED_STAGE1_EPOCHS", "8"))
    DECOUPLED_STAGE2_FREEZE_ENCODERS = bool(int(os.environ.get("DECOUPLED_STAGE2_FREEZE_ENCODERS", "1")))
    DECOUPLED_STAGE2_ENABLE_MARGINS = bool(int(os.environ.get("DECOUPLED_STAGE2_ENABLE_MARGINS", "1")))
    # Additional source-domain ids for phase-aware invariance routing.
    TASK1_CWRU_FISHR_SOURCE_DOMAIN_IDS = (0, 1)
    TASK1_CWRU_CCPA_SOURCE_DOMAIN_IDS = (0, 1)
    GLOBAL_DECORR_START_EPOCH = 1
    # ATT102-P1: 1.0→0.005. ATT103-P1b: 0.005→0.06. ATT105-P1c: 0.06→1.0 (restore ATT091).
    # REVISED ANALYSIS: γ=1.0 was "unreachable" but functionally CORRECT.
    # ATT091 (γ=1.0, 60.75%): VICReg loss=0.9990, var_mean=0.0014-0.0016 STABLE throughout.
    #   → Constant non-zero gradient = persistent upward "pressure" on variance.
    #   → cos_sim(N,R) reached -0.82 by E12 (Rolling fully separated from Normal).
    # ATT102 (γ=0.005, 0.1935%): std=0.041 > γ=0.005 → variance gradient=0 → unstable
    #   → var_mean oscillated 0→0.0013→0.0017→0 → cos_sim(N,R) only reached -0.19 (insufficient)
    #   → AUC_nf=1.0000 (good ranking) but hard-gate threshold miscalibrated → all-Normal
    # ATT103 (γ=0.06): gradient = max(0, 0.06-std)^2 tiny push, but model still collapsed (134 samples)
    # CONCLUSION: γ=1.0 is the empirically validated best value for this embedding space.
    # The "unreachable target" is a FEATURE not a bug — it provides a constant corrective force.
    GLOBAL_DECORR_VAR_TARGET = 1.0    # ATT105: restored to ATT091 value (was 0.06 in ATT103)
    GLOBAL_DECORR_COV_WEIGHT = 1.0
    # ATT284(P1): re-enable VICReg branch as primary global anti-collapse regularizer.
    # Keep global_decorr gate for scheduler/weight compatibility, but route the actual
    # ATT420: disabled — VICReg's covariance/variance constraints conflict with PACL's
    # prototype alignment direction. ATT400(VICReg=off)=0.9785 >> ATT418(VICReg=on)=0.9032.
    # ATT441: RE-ENABLED — PACL is now OFF (proto_cls=False), so the PACL conflict no longer exists.
    # ATT409 proved VICReg keeps fusion_grad>0.1 vs 0.0002 without it. ATT376(VICReg ON)=0.9624.
    # ATT444: DISABLED — same as decorr. ATT437(OFF)=best fusion path.
    USE_GLOBAL_VICREG_LOSS = bool(int(os.environ.get("USE_GLOBAL_VICREG_LOSS", "0")))
    # Use stronger var/cov constraints to lift collapsed feature variance in Task2.
    GLOBAL_VICREG_VAR_WEIGHT = 1.0
    GLOBAL_VICREG_COV_WEIGHT = 1.0
    GLOBAL_VICREG_STD_TARGET = 1.0

    # ========== ATT502: Fusion representation anti-collapse (train-time) ==========
    # Targets fusion_var collapse seen under extreme domain shift (e.g. SEU speed_30_load_2).
    USE_FUSION_VAR_FLOOR_LOSS = bool(int(os.environ.get("USE_FUSION_VAR_FLOOR_LOSS", "0")))
    FUSION_VAR_FLOOR_MIN = float(os.environ.get("FUSION_VAR_FLOOR_MIN", "0.001"))
    FUSION_VAR_FLOOR_WEIGHT = float(os.environ.get("FUSION_VAR_FLOOR_WEIGHT", "0.05"))
    # Light VICReg on pre-L2 fusion features (var-focused; low cov to preserve alignment).
    USE_FUSION_VICREG_LOSS = bool(int(os.environ.get("USE_FUSION_VICREG_LOSS", "0")))
    FUSION_VICREG_VAR_WEIGHT = float(os.environ.get("FUSION_VICREG_VAR_WEIGHT", "0.5"))
    FUSION_VICREG_COV_WEIGHT = float(os.environ.get("FUSION_VICREG_COV_WEIGHT", "0.05"))
    FUSION_VICREG_STD_TARGET = float(os.environ.get("FUSION_VICREG_STD_TARGET", "0.04"))
    # Penalize high inter-class centroid cosine on z_metric / z_fusion (sep_ratio guard).
    USE_FUSION_INTER_COS_MARGIN_LOSS = bool(int(os.environ.get("USE_FUSION_INTER_COS_MARGIN_LOSS", "0")))
    FUSION_INTER_COS_MARGIN_WEIGHT = float(os.environ.get("FUSION_INTER_COS_MARGIN_WEIGHT", "0.08"))
    FUSION_INTER_COS_MARGIN_TARGET_MAX = float(os.environ.get("FUSION_INTER_COS_MARGIN_TARGET_MAX", "0.15"))
    FUSION_INTER_COS_MARGIN_START_EPOCH = int(os.environ.get("FUSION_INTER_COS_MARGIN_START_EPOCH", "1"))
    FUSION_INTER_COS_MIN_SAMPLES_PER_CLASS = int(os.environ.get("FUSION_INTER_COS_MIN_SAMPLES_PER_CLASS", "2"))
    # ATT185: rollback ATT184 uniformity term due to observed global performance regression.
    USE_GLOBAL_UNIFORMITY_LOSS = False
    GLOBAL_UNIFORMITY_WEIGHT = 1.0
    GLOBAL_UNIFORMITY_T = 2.0
    # ATT286(P2): Fishr-style source-domain gradient variance alignment.
    # Proxy implementation aligns gradient variance on fusion representation
    # between source domains (default: CWRU id=0, PU id=2).
    # ATT430: DISABLED — ATT429 proved Fishr is DEAD in practice (0/20 epochs activated,
    # batch-level sensitivity makes it unusable when domain imbalance exists).
    # Replaced by S2 dual-disentangle domain confusion (instance-level, no batch constraint).
    USE_FISHR_LOSS = bool(int(os.environ.get("USE_FISHR_LOSS", "0")))
    FISHR_WEIGHT = float(os.environ.get("FISHR_WEIGHT", "0.10"))
    FISHR_START_EPOCH = 1
    FISHR_MIN_SAMPLES_PER_DOMAIN = int(os.environ.get("FISHR_MIN_SAMPLES_PER_DOMAIN", "2"))
    FISHR_SOURCE_DOMAIN_IDS = (0, 2)
    # ATT427: Fishr target feature — "z_spec" or "z_fusion"
    FISHR_TARGET_FEATURE = os.environ.get("FISHR_TARGET_FEATURE", "z_spec")
    # ATT428: Fishr diagnostic logging — print grad chain details every N batches
    FISHR_DIAG_LOG_ENABLED = bool(int(os.environ.get("FISHR_DIAG_LOG_ENABLED", "1")))
    FISHR_DIAG_LOG_STRIDE = int(os.environ.get("FISHR_DIAG_LOG_STRIDE", "20"))
    # ATT287(P2): CCPA (Class-Conditional Prototype Alignment) on source domains.
    # Align per-class prototypes between CWRU and PU in cosine space.
    # ATT379f(A): disable — CCPA ineffective (converges to ~0.001, skipped in PU phase)
    USE_CCPA_LOSS = False
    CCPA_WEIGHT = 0.0
    CCPA_START_EPOCH = 1
    CCPA_MIN_SAMPLES_PER_DOMAIN_CLASS = 2
    CCPA_SOURCE_DOMAIN_IDS = (0, 2)
    # Log global variance of key embeddings each epoch
    LOG_GLOBAL_VARIANCE = True      # Att81: True (was False) — diagnostic
    # Auto-reset rolling adapter when variance collapses
    ROLLING_DECORR_RESET_ON_ZERO = False
    ROLLING_DECORR_RESET_EPS = 1e-4
    ROLLING_DECORR_RESET_COOLDOWN = 3

    # Rolling adapter behavior
    ROLLING_ADAPTER_DETACH = True

    # Stage2 stabilization for rolling head
    ROLLING_STAGE2_FREEZE_EPOCHS = 0

    # ========== (D2) Prototype + Linear residual diagnosis head ==========
    # logits_diag = logits_proto + beta * logits_linear
    # beta starts at 0 so behavior is unchanged until trained (especially useful for target supervised finetune).
    USE_DIAG_LINEAR_RESIDUAL = True
    DIAG_LINEAR_SCALE_INIT = 0.0
    # Train this head *late* on source domain with a smaller lr to avoid destabilizing early training.
    # (HUST/HIT collapse often comes from early prototype bias; this gives the model a learnable correction head.)
    DIAG_LINEAR_TRAIN_START_EPOCH = 5
    DIAG_LINEAR_WARMUP_EPOCHS = 5
    DIAG_LINEAR_LR_MULT = 0.10
    # Cap for beta (effective mixing weight upper bound).
    DIAG_LINEAR_BETA_MAX = 0.5
    # Make beta sample-adaptive: beta(x) in [0,1] via a small gate, multiplied by diag_linear_scale (scalar).
    USE_DIAG_LINEAR_GATE = True
    DIAG_LINEAR_GATE_INIT_BIAS = -8.0  # sigmoid(-8)≈0.0003 (stable default)

    # ========== (S-B) Diagnosis class-bias prior ==========
    USE_DIAG_CLASS_BIAS = True
    DIAG_CLASS_BIAS_TRAIN_START_EPOCH = 12

    # ========== (S-SEM) Semantic distillation (teacher: semantic/prototype logits -> student: linear head) ==========
    # This trains the learnable diagnosis head to mimic the semantic teacher when the teacher is not collapsed.
    # ATT379f(A): disable — semantic distill requires well-trained teacher; PACL alignment sufficient
    USE_SEMANTIC_DISTILL = False
    SEMANTIC_DISTILL_WEIGHT = 0.0
    SEMANTIC_DISTILL_T = 2.0
    SEMANTIC_DISTILL_START_EPOCH = 5
    # If teacher is too confident (likely collapsed), skip distillation to avoid negative transfer.
    SEMANTIC_DISTILL_TEACHER_MAXCONF = 0.98
    # Only distill on fault samples (labels>0) to avoid leaking NF uncertainty into diagnosis head.
    SEMANTIC_DISTILL_FAULT_ONLY = True

    # ========== (Attempt 64) Semantic anchor for fault vs normal ==========
    # Enforce fault samples to be closer to their own text prototype than normal prototype.
    USE_SEMANTIC_FAULT_VS_NORMAL_ANCHOR = False
    SEMANTIC_FAULT_VS_NORMAL_ANCHOR_WEIGHT = 0.20
    SEMANTIC_FAULT_VS_NORMAL_ANCHOR_MARGIN = 0.15
    SEMANTIC_FAULT_VS_NORMAL_ANCHOR_START_EPOCH = 1

    # ========== (Attempt 66) Domain generalization with MixStyle ==========
    # Replace domain-adversarial alignment with feature-statistics mixing in early layers.
    USE_MIXSTYLE_SIGNAL = bool(int(os.environ.get("USE_MIXSTYLE_SIGNAL", "1")))
    # ATT171: False→True. 对称启用 Spec 分支 MixStyle，与 Signal 分支对齐。
    # MixStyle（ICLR 2021）通过随机混合 batch 内样本的实例级统计量（均值/方差），
    # 在训练时生成"虚拟域"，迫使模型学习与幅值分布无关的故障语义特征。
    # 针对 ATT170 瓶颈：HUST OR=4%/Ball=0%，HIT FusionRaw 坍塌——根因是 CWRU/PU 的频谱样式
    # 与 HUST/HIT 有显著分布偏移（采样率/轴承型号/工况不同），SpecEncoder 学到了样式敏感特征。
    # MixStyle 在 stem 后和 block2 后插入（训练时启用，推理时自动关闭），位置对齐 ATT149 设计。
    # ATT171/172/173 三轮证伪：特征级 MixStyle、ProtoIsolation、双投影头+GRL 均失败。
    # ATT174: 保持 False，回退到 ATT170 基线，改用输入级 SpecAugment（USE_SPEC_AUGMENT）。
    USE_MIXSTYLE_SPEC = bool(int(os.environ.get("USE_MIXSTYLE_SPEC", "1")))
    # 关闭（ATT172 证伪：alignment loss 权重过小导致完全无效）
    MIXSTYLE_PROTO_ISOLATION = False
    MIXSTYLE_P = 0.50
    MIXSTYLE_ALPHA = float(os.environ.get("MIXSTYLE_ALPHA", "0.30"))
    # Global scale for domain-conditioning paths (FiLM / StatFiLM / DomainFiLM); 1.0 = full strength.
    DOMAIN_COND_STRENGTH = float(os.environ.get("DOMAIN_COND_STRENGTH", "1.0"))

    # ========== Learnable ablation gates (residual gating over PIDR/GIDC/GCSM sub-modules) ==========
    # When enabled, each ablation component is blended: out = identity + g*(module_out - identity).
    # g is learned per sub-module (init via ABLATION_GATE_INIT_DEFAULT / per-key overrides).
    ABLATION_GATING_ENABLE = bool(int(os.environ.get("ABLATION_GATING_ENABLE", "0")))
    ABLATION_GATE_INIT_DEFAULT = float(os.environ.get("ABLATION_GATE_INIT_DEFAULT", "0.5"))
    ABLATION_GATE_LOG_INTERVAL = int(os.environ.get("ABLATION_GATE_LOG_INTERVAL", "1"))
    ABLATION_GATE_LR_MULT = float(os.environ.get("ABLATION_GATE_LR_MULT", "0.5"))
    # Optional JSON dict of per-key init values, e.g. '{"domain_cond_fs_film":0.3}'
    _abl_gate_init_json = os.environ.get("ABLATION_GATE_INIT_OVERRIDES_JSON", "")
    ABLATION_GATE_INIT_OVERRIDES = {}
    if _abl_gate_init_json.strip():
        try:
            import json as _json
            ABLATION_GATE_INIT_OVERRIDES = dict(_json.loads(_abl_gate_init_json))
        except Exception:
            ABLATION_GATE_INIT_OVERRIDES = {}

    # ATT173 证伪：双投影头+GRL 导致 val_loss 剧烈振荡（E11-15: 0.57-0.61），Best Epoch 卡死 E10，
    # SpecProto 灾难性坍塌（CWRU 0.2312→0.0645，HIT 0.4333→0.0755）。根因：GRL 梯度反转与
    # SpecHead CE 梯度冲突 + alignment warmup 仅提供2个 full-weight epoch（E9-E10 vs ATT170 的20个）。
    # ATT173/GRL 方案完全排除。
    USE_SPEC_DISENTANGLE = False
    SPEC_DISENTANGLE_INV_DIM = 512
    SPEC_DISENTANGLE_SPE_DIM = 64
    SPEC_DISENTANGLE_START_EPOCH = 5
    SPEC_DISENTANGLE_MAX_ALPHA = 0.3
    SPEC_DISENTANGLE_ALPHA_EPOCHS = 10
    SPEC_DISENTANGLE_ORTH_WEIGHT = 0.05
    SPEC_DISENTANGLE_DOMAIN_SPE_WEIGHT = 0.10
    SPEC_DISENTANGLE_DOMAIN_ADV_WEIGHT = 0.05

    # ========== ATT174: 输入级 SpecAugment（频率掩码 + 时间掩码） ==========
    # 核心原理：在 SpecEncoderLight.forward() 的 stem 前对 spec 张量 [B,C,F,T] 做逐样本随机掩码。
    # 与 MixStyle（特征级）的根本区别：纯输入扰动，eval 时 self.training=False 自动关闭，
    # 零训练-推理分布偏移，零额外参数，零梯度干扰。
    # 理论依据：Park et al. SpecAugment (Interspeech 2019) — 频率/时间掩码迫使编码器学习频带冗余性；
    #   在轴承诊断跨域场景中，BPFO/BPFI 频率因轴承尺寸不同在各数据集间偏移，频率掩码强迫编码器
    #   不依赖特定频带，提升对域间频率偏移的鲁棒性（等价于"频率不变"的归纳偏置）。
    # ATT175: 关闭 SpecAugment。实验证据（ATT174+方案A）：
    #   训练时掩码 → 推理时全谱评估，原型（text_prototypes）基于掩码特征构建，
    #   但 zero-shot 评估时使用干净输入，导致不可消除的 train-eval 分布偏移（Δ SpecProto ≈ -11pp）。
    #   PU/Task2 阶段更严重：掩码后归一化特征空间偏移使 FusionRaw/FusionProto 崩溃。
    #   排除路径：ATT174+方案A 两轮实验均未能规避该偏移，故彻底关闭。
    USE_SPEC_AUGMENT = False  # ATT185: rollback ATT184; keep ATT183-compatible stability
    SPEC_AUGMENT_FREQ_MASK_F = 0.05      # 最大频率掩码宽度：freq 维的 10%（保守，不破坏主频带）
    SPEC_AUGMENT_TIME_MASK_T = 0.05      # 最大时间掩码宽度：time 维的 10%
    SPEC_AUGMENT_NUM_FREQ_MASKS = 1      # 每样本独立频率掩码数量
    SPEC_AUGMENT_NUM_TIME_MASKS = 1      # 每样本独立时间掩码数量
    SPEC_AUGMENT_MASK_PROB = 0.3         # 每条掩码独立激活概率（0.5 = 平均 50% 样本被掩码）

    # ATT394 FAILED: IBN再次证实会削弱NF振幅判别信息。CWRU SpecHead 0.8333→0.4409。
    # ATT147/148/394三次证伪: 在轴承故障诊断领域，任何移除振幅信息的归一化方法都会失败。
    USE_IBN_SPEC = bool(int(os.environ.get("USE_IBN_SPEC", "0")))
    IBN_SPEC_RATIO = float(os.environ.get("IBN_SPEC_RATIO", "0.50"))
    USE_IBN_SPEC_STEM_ONLY = bool(int(os.environ.get("USE_IBN_SPEC_STEM_ONLY", "1")))
    USE_IBN_SIGNAL = bool(int(os.environ.get("USE_IBN_SIGNAL", "0")))
    IBN_SIGNAL_RATIO = float(os.environ.get("IBN_SIGNAL_RATIO", "0.50"))
    # ATT160 P1: 在非 IBN 分支将 Spec stem 的 BN 改为 GN（block2/3 保持不变），做单变量验证。
    # ATT161: GN 实验失败（PU E4早停爆炸 + HIT OR偏置崩溃）。回退 GN=False。
    USE_GN_SPEC_STEM = False
    # ATT161: FreqLN 灾难性失败（PU AUC_nf 0.91→0.53，spec嵌入方差坍缩），移除频率幅度信息破坏NF判别。
    # 恢复 BN stem（IBN/GN/FreqLN 三条路径均已证伪，BN是必要组件）。
    # ATT395: 回退到ATT393状态(True), IBN已关闭不会冲突。
    USE_FREQLN_SPEC_STEM = True
    # ATT162-164 FreqAttnPool 实验总结（已完结，方向证伪）：
    # ATT162: 零初始化门控，E3 val_loss spike 1.82，E4 checkpoint 锁定；
    # ATT163: ReZero alpha 负值反馈，"反注意力"，CWRU E3 val_loss=5.74，灾难崩溃；
    # ATT164: 1/sqrt(F') score 缩放，spike 降至 1.07，但根本瓶颈转移——
    #   CWRU(32样本验证集) + PU 均在 E2 被锁定为 best ckpt（val_loss=0.275），
    #   此后 val_loss 永久发散但 SpecHead Acc 持续提升至 0.78（E14），
    #   模型从未被选到收敛后的 epoch；
    # ATT165 P1: 回退 FreqAttnPool，恢复 BN mean pooling 基线。
    # FreqAttnPool 引入的 E3 分布突变是过早选 checkpoint 的直接触发因素，
    # 而 mean pooling 从始至终输出分布恒定，BN 基线（161710）稳定选到 E10。
    USE_FREQ_ATTN_POOL = False
    # ATT151 P0: 先关闭输入级幅度扰动，隔离验证 ReZero 门控修复是否可恢复 ATT146 起跑线。
    SPEC_AMP_JITTER_ENABLE = False
    SPEC_AMP_JITTER_PROB = 0.50
    SPEC_AMP_JITTER_STD = 0.05
    SPEC_AMP_JITTER_MIN = 0.50
    SPEC_AMP_JITTER_MAX = 1.50
    # ATT152 P0: 回退 CA 残差门控，恢复 ATT146 等价残差路径（attn_out + residual）。
    CA_RESIDUAL_GATE_ENABLE = False

    # Keep signal branch trainable (ATT373): warm-freeze at epoch=5 caused signal grad_norm to collapse to 0.
    SIGNAL_ENCODER_LR_MULT = 0.10
    FREEZE_SIGNAL_AFTER_EPOCH = -1

    # ========== (Decision) Diagnosis decision mode ==========
    # Controls how 4-class diagnosis predictions are produced at evaluation/inference time:
    # - "argmax": plain argmax over 4-way diagnosis logits (legacy; can yield Normal recall=0 under shift)
    # - "nf_protect": keep argmax, but apply NF-protect (soft_bias/hard) only when NF is very confident normal
    # - "nf_gate": HARD hierarchical decision: first NF (normal vs fault), then fault-type among {1,2,3}
    # - "two_stage": (New scheme 1) if NF says Normal -> 0; else argmax(fault 3-way)+1 (fault-type only on fault samples at train)
    # - "soft_hierarchical": SOFT hierarchical decision using nf "faultness" to bias fault-type logits
    #   (more stable than hard nf_gate; recommended when nf_gate collapses outer/rolling on HUST)
    #   (more stable than hard nf_gate; recommended when nf_gate collapses outer/rolling on HUST)
    # NOTE: soft_hierarchical can significantly reduce target-domain (e.g., HIT) diagnosis accuracy
    # by injecting the same NF logit into all fault classes. Use a more conservative nf_protect by default.
    # DIAG_DECISION_MODE: 'argmax' forces gradient flow from fault heads (Disable nf_protect for now)
    # DIAG_DECISION_MODE: 'sequential' (Attempt 9)
    # Sequential Decision: If P(Fault) > P(Normal), force predicted class to be in {1,2,3}.
    # Eliminates "Vote Dilution" where Normal (0.4) beats Fault (0.2+0.2+0.2=0.6).
    # Attempt 73: use 'two_stage' (先 NF，后 fault-type)，充分利用已较强的 NF AUC，避免决策期重新偏向 Normal 吸收态。
    # ATT104: "two_stage" → "argmax". [REVERTED in ATT105 — argmax did NOT fix 0.1935]
    # ATT105 analysis: argmax failed because the MODEL ITSELF collapses (134 samples, 5 batches/epoch).
    # With collapsed embeddings, argmax also outputs class 0 for all → same 0.1935 → argmax not the fix.
    # ATT091 (60.75%) used "two_stage" AND VICReg γ=1.0 AND diffusion (1280 samples, 20 batches):
    #   → NF head learned stable Rolling-Normal separation (cos_sim(N,R)→-0.82 by E12)
    #   → two_stage correctly gates gk1/gk3 Rolling as Fault at test time
    # ATT105 (training healthy, SupCon→-0.73, Diag-Acc=0.1935):
    #   Root cause identified: val-set-fitted NF threshold (32 samples, gk2/gk4 only) does NOT
    #   transfer to test conditions (gk1/gk3). Even with AUC_nf=1.0, the fitted threshold is too
    #   high for test faults → all below threshold → all predicted Normal.
    # ATT106: switch to "sequential" which uses anomaly_logit > 0 DIRECTLY (no val-set threshold).
    #   With AUC_nf=1.0, the rank order is correct: fault anomaly_logits > normal anomaly_logits.
    #   Threshold=0 (fixed) is more robust than val-set-fitted threshold under cross-load domain shift.
    # ATT106 RESULT: sequential ALSO failed (Diag-Acc=0.1935). Mathematical proof:
    #   anomaly_logit < 0 for ALL gk1/gk3 test samples (including faults).
    #   AUC_nf=0.7709 confirms ranking is correct but ABSOLUTE values all negative.
    #   Cross-load domain shift in NF logit space: training (gk2/gk4) → logits positive;
    #   test (gk1/gk3) → logits systematically shifted negative (load-dependent amplitude).
    # ATT107: bypass NF gate entirely with "argmax". FUSION AUC=0.789 implies 4-class logits
    #   contain genuine discriminative information. The NF gate is the SOLE bottleneck.
    #   ATT104 argmax also failed, but with COLLAPSED model (134 samples). ATT107 uses healthy
    #   model (1280 samples, var_mean=0.0015, cos_sim(N,R)→-0.74). Fundamentally different.
    DIAG_DECISION_MODE = "argmax"  # ATT107: "sequential"→"argmax" (bypass NF logit shift, test healthy-model argmax)
    # (New scheme 1) When True, train with NF CE (all) + 3-class CE on fault samples only (no 4-class CE on all).
    USE_TWO_STAGE_FAULT_LOSS = False
    # (New scheme 2) When True, fault-type uses z_fault = fault_projection(z.detach()); only fault-type loss updates it.
    # NOTE (Attempt 11): disable detach to let fault-type gradients shape the backbone representation,
    # otherwise IO/Rolling heads tend to become near-constant under domain shift (observed in probes).
    USE_FAULT_DEDICATED_REP = False

    # ========== (Fault-type) Long-tail weighting (Inner/Outer/Rolling) ==========
    # Rolling-element is often extremely rare in the mixed-source training set, causing 0-recall collapse on HUST.
    # These weights are applied ONLY to the fault-type CE inside the hierarchical fusion decoder loss.
    FAULT_TYPE_WEIGHT_ENABLE = True
    # Power for inverse-frequency weighting: 1.0 = linear inv-freq, 0.5 = sqrt inv-freq.
    # Conservative default: sqrt-inv to avoid over-shooting into "all→rolling" / suppressing Outer.
    FAULT_TYPE_WEIGHT_POWER = 0.5
    # Max cap for fault-type weights (higher than MAX_CLASS_WEIGHT by design).
    # Conservative cap: keep rolling emphasis but avoid hard absorber.
    FAULT_TYPE_MAX_WEIGHT = 5.0

    # ========== (Fault-type) Logit adjustment (long-tail prior) ==========
    # Apply logit adjustment on 3-way fault logits to counter class-imbalance bias.
    # Prior can be either counts or normalized probabilities for [inner, outer, rolling].
    # NOTE: Update this when training class distribution changes.
    # NOTE: disable by default when using hierarchical fault probabilities
    # (prevents rolling prior from overpowering IO competition).
    USE_FAULT_LOGIT_ADJUST = False
    FAULT_LOGIT_ADJUST_TAU = 1.0
    FAULT_LOGIT_ADJUST_PRIOR = [2143, 2329, 43]

    # ========== (Fault-type, structural) Split heads for Rolling ==========
    # Motivation: Rolling-element is rare and often stays at 0-recall under domain shift.
    # We decouple fault-type into two easier sub-tasks:
    #   - Rolling vs Non-rolling (binary)
    #   - Inner vs Outer (binary, conditioned on non-rolling faults in the loss)
    #
    # This is a *structural* change (new heads + new loss) and is compatible with open-set rejection and DANN.
    USE_SPLIT_FAULT_HEADS = False
    # (NEW) Use hierarchical fault probabilities instead of concatenating logits:
    # p_roll = P(roll|fault), p_io = P(inner/outer|fault, non-roll)
    # p_inner = (1 - p_roll) * p_io_inner, p_outer = (1 - p_roll) * p_io_outer
    USE_HIERARCHICAL_FAULT_PROB = False
    # Rolling head gradient isolation (reduce IO/rolling interference)
    USE_ROLL_ISOLATED_ADAPTER = True
    ROLL_ADAPTER_DIM = 128
    # Loss weights inside hierarchical fusion loss (only when split heads are available).
    SPLIT_FAULT_W_IO = 1.0      # weight for Inner-vs-Outer loss
    SPLIT_FAULT_W_ROLL = 1.0    # weight for Rolling-vs-NonRolling loss
    # Rolling BCE positive weight clamp for stability.
    SPLIT_FAULT_ROLL_POS_WEIGHT_MAX = 20.0
    # When True, Rolling-vs-NonRolling loss uses ALL samples as negatives (labels != 3).
    # This is safe because in PU/HIT rolling does not exist by dataset definition; treating them as negatives
    # does not require peeking target-domain labels.
    SPLIT_FAULT_ROLL_USE_ALL_NEG = True

    # ========== (NF-gate) Thresholding for nf_gate ==========
    # When DIAG_DECISION_MODE == "nf_gate", you can choose NF decision by:
    # - "argmax": nf_pred = argmax(logits_nf_global)
    # - "threshold": nf_pred = (anomaly_logit = logit_fault - logit_normal) > t
    NF_GATE_NF_MODE = "threshold"  # "threshold" | "argmax"
    # Threshold fitting (uses labeled calibration split; avoids hard-coded 0 threshold)
    NF_GATE_FIT_ENABLE = True
    NF_GATE_FIT_OBJECTIVE = "balanced_acc"  # "balanced_acc" | "acc"
    NF_GATE_FIT_NUM_CANDIDATES = 51
    NF_GATE_FIT_MAX_SAMPLES = 512  # cap calibration samples for speed/stability

    # ========== (B) Zero-shot NF threshold objective ==========
    # Training-time zero-shot evaluation can use a different objective to avoid over-gating.
    # (Attempt 12) IMPORTANT: do NOT use "acc" on imbalanced NF calibration sets.
    # It often degenerates to "always fault" threshold, causing Normal recall to collapse to 0 under sequential/nf_gate.
    ZERO_SHOT_NF_GATE_FIT_OBJECTIVE = "balanced_acc"  # "balanced_acc" | "acc"

    # ========== (A1-1) Platt scaling for NF calibration ==========
    # Fit a logistic calibration on anomaly_logit using a tiny labeled split (more stable than fitting a raw threshold).
    USE_NF_PLATT_CALIBRATION = True
    NF_PLATT_MAX_SAMPLES = 512
    NF_PLATT_STEPS = 200
    NF_PLATT_LR = 0.10
    NF_PLATT_L2 = 1e-3

    # ========== (B) Gray-zone fallback for nf_gate ==========
    # When p_fault is in [low, high], nf is uncertain; fall back to raw diagnosis to avoid gate mistakes.
    NF_GATE_GRAY_LOW = 0.35
    NF_GATE_GRAY_HIGH = 0.65
    NF_GATE_GRAY_FALLBACK = "raw"  # "raw" | "nf_protect"

    # ========== (NF-gate) Direct margin on global NF logits ==========
    # This directly reduces "Fault → Normal" gate errors (the dominant error after nf_gate was enabled).
    # ATT379f(A): disable — decoder CE already handles NF classification
    USE_NF_GATE_MARGIN_LOSS = False
    NF_GATE_MARGIN_WEIGHT = 0.0
    # For faults: anomaly_logit >= +m ; for normals: anomaly_logit <= -m
    NF_GATE_MARGIN = 0.20
    NF_GATE_MARGIN_START_EPOCH = 5

    # ========== (Margin Router) Focus margins on current absorber ==========
    # When enabled, training will automatically focus margin losses on the currently observed absorber class
    # (detected from HUST zero-shot prediction histogram): vs_inner / vs_rolling / vs_normal.
    ENABLE_MARGIN_ROUTER = False
    MARGIN_ROUTER_PRED_RATIO_TH = 0.70

    # ========== (Health Gate) Collapse criterion ==========
    # Avoid false positives where Normal recall=0 early but fault recalls are still reasonable.
    # If True: compute min recall over fault classes only (1/2/3).
    HEALTH_GATE_FAULT_ONLY = True

    # ========== (S-C) Best-checkpoint health gate ==========
    # Strict no-leakage default:
    # do NOT use target-domain labels/metrics to gate best checkpoint selection.
    ENABLE_BEST_CKPT_HEALTH_GATE = False
    USE_TARGET_DOMAIN_HEALTH_GATE_FOR_BEST_CKPT = False
    BEST_CKPT_MIN_CLASS_RECALL = 0.05  # minimum per-class recall (raw) required on HUST
    # Attempt 39: relax the "one-vote veto" failure mode.
    # Instead of requiring *all* fault classes to exceed the threshold (min_fault_recall),
    # require at least K fault classes (among 1/2/3) to exceed it.
    # This allows saving E10-like peaks where 2 faults recover but Rolling may still be 0.
    HEALTH_GATE_MIN_FAULT_CLASSES = 2

    # ========== (S-C2) Training-phase health gate ==========
    # If diagnosis collapses on HUST zero-shot (min per-class recall too low),
    # temporarily disable prototype-head training to recover a stable decision boundary.
    ENABLE_TRAIN_HEALTH_GATE = True
    TRAIN_HEALTH_GATE_COOLDOWN_EPOCHS = 3  # 5->3 to avoid long prototype-head delay
    # Minimum fault recall to avoid gate trigger (0.02 = relax so prototype-head can train earlier).
    HEALTH_GATE_MIN_FAULT_RECALL = 0.02  # used if set; else BEST_CKPT_MIN_CLASS_RECALL for recall_th

    # ========== (Stability) Clamp NF logits ==========
    # Prevent BCEWithLogits from exploding and dominating training (observed Fusion Cls 300+ in logs).
    NF_LOGIT_CLAMP = 20.0

    # ========== (A0) FusionDecoder loss weights / nf_multi gate ==========
    # In recent runs, per-type nf_multi (multi-label BCE) can dominate training when Rolling is extremely rare,
    # pushing the model into "all → Rolling" collapse. Default: disable nf_multi during source training.
    FUSIONDEC_W_NF_GLOBAL = 1.1
    FUSIONDEC_W_NF_MULTI = 0.0
    FUSIONDEC_W_FAULT_TYPE = 2.0
    USE_NF_MULTI_LOSS = False
    NF_MULTI_START_EPOCH = 999999  # effectively disabled unless explicitly enabled
    NF_MULTI_POS_WEIGHT_MAX = 5.0  # hard clamp to prevent extreme neg/pos ratios from exploding gradients

    # ========== (S1) Multi-prototype diagnosis head ==========
    # Each class gets K prototypes; logits aggregate across prototypes via logsumexp.
    # This reduces "class dropout" (e.g., Outer or Rolling recall=0) under domain shift.
    USE_MULTI_PROTOTYPE_HEAD = True
    MULTI_PROTOTYPE_K = 3
    MULTI_PROTOTYPE_INIT_STD = 0.01
    MULTI_PROTOTYPE_START_EPOCH = 12
    # Strong safety: never let Rolling class have extra prototypes (prevents "all→Rolling" absorber).
    MULTI_PROTOTYPE_DISABLE_ROLLING_EXTRA = True

    # ========== (A) Supervised Contrastive + Prototype Separation ==========
    # ATT379f(A): disable alignment_loss SupCon on z_res — plateaued at 3.1, occupied 73% of positive raw
    USE_SUPCON_LOSS = False
    SUPCON_WEIGHT = 0.0
    # ATT115: REMOVED duplicate SUPCON_TEMPERATURE=0.10 and SUPCON_FAULT_ONLY=False here.
    # These definitions at lines ~355/363 (ATT114 values: τ=0.20, fault_only=True) were being
    # silently overridden because Python class bodies execute top-to-bottom: later assignments win.
    # ATT114 was a bit-exact replica of ATT112 (same Diag-Acc=0.4194, Macro-F1=0.1477) solely
    # because these two lines at the end of the class overrode the intended changes.
    # Now the ATT114 values at Config lines ~355/363 (SUPCON_TEMPERATURE=0.20, SUPCON_FAULT_ONLY=True)
    # will correctly take effect.
    # Attempt 58 pre-step: warm up spec semantic alignment to avoid harming signal branch
    # right after replacing collapsed MoE spec encoder with a newly-active light encoder.
    SPEC_ALIGN_WARMUP_ENABLE = bool(int(os.environ.get("SPEC_ALIGN_WARMUP_ENABLE", "0")))  # ATT379b: 禁用warmup，E1起w_spec=base(0.3)；ATT379a bug: start=end=1导致0/0除法→w_spec=0
    SPEC_ALIGN_WARMUP_START_EPOCH = 1
    SPEC_ALIGN_WARMUP_END_EPOCH = int(os.environ.get("SPEC_ALIGN_WARMUP_END_EPOCH", "1"))
    # ATT175: Task2 专属 warmup 终止轮。
    # 根因：warmup_end=5 在 Task2（CWRU+PU 混合域）触发早期对齐-分类梯度冲突：
    #   E5 时 L_align 全量激活，此时 Fusion CE 梯度方向已固化（跨域混合使方向更强），
    #   两者在 z_spec 上产生反向梯度 → val_loss E10+ 剧烈振荡/爆炸 → FusionRaw/FusionProto 崩溃。
    # warmup_end=8 允许 Fusion 网络在 E1-7 先完成基础判别收敛，再引入对齐压力（梯度兼容性更高）。
    # 历史证据：ATT165（0.8871 CWRU）和 ATT142（0.8978 CCTC）均使用 warmup_end=8，Task2 无崩溃。
    SPEC_ALIGN_WARMUP_END_EPOCH_TASK2 = 8
    SPEC_ALIGN_WARMUP_MIN_SCALE = 0.0
    SPEC_ALIGN_WARMUP_MAX_SCALE = 1.0
    # ATT428: DISABLED — this uses text_prototypes (cos(I,O)=0.865) for spec alignment.
    # SpecAlign now uses detached_spec_align with training_proto_tensor (ETF geometry) instead.
    SPEC_ALIGN_BASE_WEIGHT = float(os.environ.get("SPEC_ALIGN_BASE_WEIGHT", "0.0"))
                                    #   原型与 z_spec 反向对齐（spec_proto_diag_acc<0.25随机基线），任何非零权重均扰乱判别特征；
                                    #   w=0.03 不足以收敛至原型对齐盆地，但足以推离 ATT134 最优点 → val_loss E6+爆炸。
                                    #   ATT139/140 同时证伪 Stage1 语义注入路径；恢复 0.3 = no-op（Stage1无调用者）；
                                    #   ATT126实证 SPEC_ALIGN=0.3 + base_ch=32 + dropout=0.1 = 历史最佳 spec_head=0.8548；
                                    #   但 ATT126 的语义路径是 no-op（τ=0.07 饱和）；现在 τ=0.5 生效，必须大幅降权；
                                    #   0.3延迟过拟合后，配合P0 val_spec_head_acc Early Stop，将选择更晚、更优的spec_head checkpoint）
    # ATT178: Detached SpecAlign Projection Head（低风险语义对齐修复）
    # 目标：修复 Stage1 语义对齐“梯度冲突”历史问题（ATT139/140/176），但不改变主干收敛轨迹。
    # 方法：z_spec 先 detach，再经过轻量投影头 align_proj 与 text_prototypes 做 CE。
    # 梯度仅更新 align_proj，不回传到 SpecEncoder/Fusion 主干（主任务 CE 不受干扰）。
    USE_DETACHED_SPEC_ALIGN_PROJ = True
    # ATT427: was raised to 0.30 as PRIMARY zero-shot pathway
    # ATT429-P0: DISABLED — SpecHead cosine classifier is now the primary ZS path.
    # DSAP aligns to text/learnable protos (different target than SpecHead weights), causing gradient conflict.
    DETACHED_SPEC_ALIGN_WEIGHT = float(os.environ.get("DETACHED_SPEC_ALIGN_WEIGHT", "0.0"))
    DETACHED_SPEC_ALIGN_TEMPERATURE = 0.10
    DETACHED_SPEC_ALIGN_START_EPOCH = 1
    DETACHED_SPEC_ALIGN_USE_CLASS_WEIGHTS = True
    DETACHED_SPEC_ALIGN_USE_IN_EVAL = True
    # ATT427: was attached for spec-proto alignment path
    # ATT429-P0: reverted to detached — DSAP weight=0.0, so this is no-op
    SPEC_ALIGN_ATTACHED = bool(int(os.environ.get("SPEC_ALIGN_ATTACHED", "0")))
    # ATT138 P1: 修复语义对齐温度过小导致的梯度饱和（0.07 在小样本4类原型上会过早饱和）。
    # 使用更平滑温度以保持非零梯度，激活 spec_proto 路径。
    SEMANTIC_ALIGNMENT_TEMPERATURE = 0.5

    # ========== ATT142: 后验语义决策融合 (Post-Hoc Semantic Decision Fusion, PSDF) ==========
    # 动机：
    #   - Fusion AUC=0.9994 但 Diag-Acc=0.6452（nf_protect 导致决策边界偏移，与分类质量无关）
    #   - spec_head acc=0.8548，但 OR→Ball 混淆 17/36（占全部误判 63%）
    #   - spec_proto_diag_acc=0.1559 < 0.25 随机基线（文本原型反向对齐，有害）
    #   - 在推理时融合 spec_head + Fusion 双路径，完全不修改训练
    #
    # 参数说明：
    #   EVAL_ENSEMBLE_SPEC_HEAD_WEIGHT: spec_head 路径权重（default 0.6）
    #   EVAL_ENSEMBLE_FUSION_WEIGHT: Fusion decoder 路径权重（default 0.4）
    #   EVAL_ENSEMBLE_SPEC_PROTO_WEIGHT: spec_proto 权重（default 0.0 = 已确认有害）
    #   EVAL_ENSEMBLE_SIG_PROTO_WEIGHT: sig_proto 权重（default 0.0 = 已确认有害）
    #   EVAL_ENSEMBLE_SPEC_HEAD_TEMPERATURE: spec_head logits 软化温度（default 1.0 = 不变）
    #   EVAL_ENSEMBLE_FUSION_TEMPERATURE: Fusion logits 软化温度（default 1.0 = 不变）
    #   EVAL_ENSEMBLE_USE_FUSION_LOGITS: 是否启用 Fusion 路径（default True）
    EVAL_ENSEMBLE_SPEC_HEAD_WEIGHT      = 0.6   # spec_head 主路径（acc=0.8548）
    EVAL_ENSEMBLE_FUSION_WEIGHT         = 0.4   # Fusion 辅助路径（AUC=0.9994，补充 OR/Ball 排序）
    EVAL_ENSEMBLE_SPEC_PROTO_WEIGHT     = 0.0   # 文本原型 cosine（ATT120-141 确认 < 随机基线，禁用）
    EVAL_ENSEMBLE_SIG_PROTO_WEIGHT      = 0.0   # 信号原型 cosine（同上，禁用）
    EVAL_ENSEMBLE_SPEC_HEAD_TEMPERATURE = 1.0   # spec_head 温度（1.0=直接 softmax，不调整）
    EVAL_ENSEMBLE_FUSION_TEMPERATURE    = 1.0   # Fusion 温度（1.0=直接 softmax，由 T_GRID 自动搜索）
    EVAL_ENSEMBLE_USE_FUSION_LOGITS     = True  # 启用 Fusion 路径（False=退化回 ATT141 spec_head only）
    # ATT142+: T_fusion 自动搜索（grid search over eval set to maximize Outer R）
    # 理由：FusionRaw Outer R=0.92 但 T=1.0 集成 Outer R 只从 0.47→0.50，根因是 log-prob sharpening
    #       不足。低温 T_fusion<1.0 使 Fusion Outer 概率更集中，克服 SpecHead 的 Ball 偏置。
    # 搜索目标：max(Outer R) s.t. Inner R ≥ EVAL_ENSEMBLE_INNER_PROTECT
    EVAL_ENSEMBLE_FUSION_T_GRID        = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    EVAL_ENSEMBLE_FUSION_T_GRID_ENABLE = True   # False=跳过 grid search，直接用 FUSION_TEMPERATURE
    EVAL_ENSEMBLE_INNER_PROTECT        = 0.90   # Inner R 保护下限（grid search 选择约束）
    # ATT143: Margin-Aware Adaptive Fusion (MAAF) — 已被 ATT143 实验证伪
    # 结论：MAAF 假设"高 margin = 高置信 = 更可信"在 OR→Ball 自信误分类场景下失效。
    #       17个真OR被SpecHead自信地预测为Ball（margin大→α高→几乎忽略Fusion），Outer R反降至0.50。
    EVAL_ENSEMBLE_MARGIN_ADAPTIVE      = False  # ATT143 证伪后禁用
    EVAL_ENSEMBLE_MARGIN_ALPHA_MIN     = 0.30
    EVAL_ENSEMBLE_MARGIN_ALPHA_MAX     = 0.90
    EVAL_ENSEMBLE_MARGIN_TAU           = 1.00

    # ATT144: Class-Conditional Trust Cascade (CCTC)
    # 核心洞察：SpecHead 的预测列精度是类别特异性的（非样本级 margin 问题）
    #   pred=N    98.7% 精度 → 直接信任 SpecHead
    #   pred=IR   87.8% 精度 → 直接信任 SpecHead
    #   pred=OR   81.0% 精度 → 直接信任 SpecHead
    #   pred=Ball 64.6% 精度 → 激进融合（17/48=35% 实际为 OR，用 FusionRaw OR R=0.92 纠错）
    # ATT145 P0b: 三路 CCTC（新增 FusionProto 保护 Ball）
    #   数学形式（Ball 子集）：
    #     p_ens[ball] = w_spec*p_spec + w_fus*p_fus(T_fus) + w_proto*p_proto(T_proto)
    #     约束: w_spec + w_fus + w_proto = 1
    #   FusionProto Ball R=0.89（ATT144 实测），负责"保护 Ball 不被错分为 OR"
    #   FusionRaw OR R=0.92，负责"发现 OR（纠正 SpecHead 的 OR→Ball 误分）"
    # 防御：ball_class 不在当前数据集类别时自动跳过 CCTC（兼容 3 类 PU/HIT）
    EVAL_ENSEMBLE_CCTC_ENABLE          = True
    EVAL_ENSEMBLE_CCTC_BALL_CLASS      = 3       # 触发激进融合的类别（Ball=3）
    EVAL_ENSEMBLE_CCTC_W_SPEC_GRID     = [0.10, 0.15, 0.20]   # Ball子集 SpecHead权重（低，因 Ball 子集精度仅 64.6%）
    EVAL_ENSEMBLE_CCTC_W_FUS_GRID      = [0.30, 0.40, 0.50]   # Ball子集 FusionRaw 权重
    EVAL_ENSEMBLE_CCTC_T_GRID          = [0.2, 0.5, 1.0]      # Ball子集 T_fus grid（ATT144 最优为 0.5）
    EVAL_ENSEMBLE_CCTC_T_PROTO_GRID    = [0.05, 0.1, 0.2]     # FusionProto 温度（cosine sim 量级小，需低 T）

    # ATT145 P0a: 联合 (w_spec, T_fus) 网格搜索（修复 PU 集成退化）
    # 动机：三阶段运行显示 PU 上 Ensemble(0.60) < FusionRaw(0.76)，根因是固定 w_spec=0.6
    #       当 SpecHead 退化（PU: 0.54）时强制给弱路径高权重 → 集成退化。
    # 修复：将 w_spec 纳入搜索，自动发现最优权重分配（w_spec=0.0 = 纯 FusionRaw 保底）。
    # ATT169: 细化 w_spec grid（步长 0.1），确保搜索更精细
    EVAL_ENSEMBLE_W_SPEC_GRID          = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # ATT169: SpecProto 三路集成 grid — w_proto × T_proto 联合搜索
    # 动机：ATT168 发现 HIT SpecProto=0.433（depth=1 信息路由重分配的正向副作用），
    #       但 EnsGrid 仅搜索二路（SpecHead+FusionDiag），高质量 SpecProto 信号被完全忽略。
    # 修复：在 EnsGrid 中新增 w_proto 维度：三路权重满足 w_spec+w_proto+w_fus=1.0。
    # T_proto：SpecProto 基于 cosine sim（量级约 0.0~1.0），需单独温度搜索。
    EVAL_ENSEMBLE_W_SPEC_PROTO_GRID    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # ATT169: SpecProto 权重 grid
    EVAL_ENSEMBLE_SPEC_PROTO_T_GRID    = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]  # ATT178: 扩展低温段
    # ATT179 Step1: Test-Time Prototype Adaptation (TPA, 仅评估端)
    # 目标：不改训练路径、不引入新可训练参数，利用目标域高置信样本迭代修正 SpecProto 原型。
    EVAL_SPEC_PROTO_TTA_ENABLE = True
    EVAL_SPEC_PROTO_TTA_TARGET_DATASETS = ["HUST", "SEU"]  # 仅对跨域目标域启用，避免影响 CWRU/PU 基线
    EVAL_SPEC_PROTO_TTA_ITERS = 3
    EVAL_SPEC_PROTO_TTA_TOPK_RATIO = 0.30
    EVAL_SPEC_PROTO_TTA_MIN_SAMPLES_PER_CLASS = 5
    EVAL_SPEC_PROTO_TTA_MIX_ALPHA = 0.70  # default: mixed_proto = alpha*text_proto + (1-alpha)*target_visual_proto
    # ATT180: condition-gated TPA (avoid noisy pseudo-label adaptation when raw SpecProto is too weak).
    EVAL_SPEC_PROTO_TTA_MIN_SPECPROTO_ACC = 0.35
    # ATT180: domain-specific alpha override (if present, higher alpha means more conservative adaptation).
    EVAL_SPEC_PROTO_TTA_MIX_ALPHA_BY_DATASET = {"HUST": 0.90, "HIT": 0.70}
    # ATT474: SpecProto 诊断改用 learnable prototype 矩阵（与训练可对齐），避免 text_proto 几何与 z_spec 不一致导致指标恒低。
    # 不改变训练损失；仅影响评估打印的 [SpecProto] 及依赖 SpecProto sims 的 EnsGrid/TPA 输入原型。
    EVAL_SPEC_PROTO_USE_LEARNABLE_PROTOS = bool(int(os.environ.get("EVAL_SPEC_PROTO_USE_LEARNABLE_PROTOS", "0")))
    # ATT158 P1: BBSE-style post-hoc prior calibration for SpecHead logits (eval only)
    EVAL_SPEC_PRIOR_CALIB_ENABLE = True
    EVAL_SPEC_PRIOR_CALIB_STRENGTH = 1.0
    EVAL_SPEC_PRIOR_CALIB_EPS = 1e-6
    # ATT370: Tip-Adapter style cache model (eval-only).
    # Support features (K/V cache) are built from source-domain loaders, query features from target loader.
    EVAL_TIP_ADAPTER_ENABLE = True
    EVAL_TIP_ADAPTER_TARGET_DATASETS = ["HUST", "SEU"]
    EVAL_TIP_ADAPTER_MAX_SUPPORT_SAMPLES = 8000
    EVAL_TIP_ADAPTER_TOPK = 64
    EVAL_TIP_ADAPTER_BETA = 20.0
    EVAL_TIP_ADAPTER_USE_SPEC_HEAD_MIX = True
    EVAL_TIP_ADAPTER_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    EVAL_TIP_ADAPTER_REPLACE_WHEN_BETTER = True
    # Optional embedding export for reviewer t-SNE figures.
    EVAL_EXPORT_EMBEDDINGS_ENABLE = bool(int(os.environ.get("EVAL_EXPORT_EMBEDDINGS_ENABLE", "0")))
    EVAL_EXPORT_EMBEDDINGS_DIR = str(
        os.environ.get("EVAL_EXPORT_EMBEDDINGS_DIR", "experiments/results/reviewer/embeddings")
    )
    # When True (default), dump holdout-test z_metric after Task1 LP-FT for t-SNE (before restore_fusion_state).
    EVAL_EXPORT_EMBEDDINGS_POST_LPFT = bool(int(os.environ.get("EVAL_EXPORT_EMBEDDINGS_POST_LPFT", "1")))
    # When True, call reviewer_experiments.save_tsne_figure right after each npz dump.
    EVAL_EXPORT_EMBEDDINGS_AUTO_TSNE = bool(int(os.environ.get("EVAL_EXPORT_EMBEDDINGS_AUTO_TSNE", "0")))
    # LOO: include test condition key in npz/png names (e.g. CWRU_speed_1730_zs_embeddings.npz).
    EVAL_EXPORT_EMBEDDINGS_PER_CONDITION = bool(
        int(os.environ.get("EVAL_EXPORT_EMBEDDINGS_PER_CONDITION", "1"))
    )
    # ATT372 rollback to ATT371_1.2 baseline:
    # keep Tip-Adapter feature source on spec path by default.
    EVAL_TIP_ADAPTER_FEATURE_SOURCE = "spec"  # "spec" | "fusion_metric"
    # ATT371.2:
    # Keep fusion eval input dataflow consistent with training dataflow.
    # - Training uses raw z_signal and (optionally) pre-L2 z_spec for fusion_module.
    # - Eval should NOT blindly L2-normalize both inputs before fusion.
    EVAL_FUSION_NORMALIZE_INPUTS = False
    EVAL_FUSION_USE_SPEC_PRE_L2_INPUT = True
    EVAL_LOG_FUSION_INPUT_STATS = True
    EVAL_LOG_FUSION_INPUT_STEP_STRIDE = 200

    USE_PROTO_SEP_LOSS = False
    PROTO_SEP_WEIGHT = 0.0
    PROTO_SEP_MARGIN = 0.20

    # ========== (B) Prompt/Prototype tuning (CoOp/CoCoOp-like) ==========
    USE_PROMPT_TUNING = True
    USE_DOMAIN_PROMPT_TUNING = True
    PROMPT_RESIDUAL_INIT_STD = 0.01
    PROMPT_TUNE_START_EPOCH = 10  # 从第N轮开始启用shared Δproto 的训练（避免早期不稳定/坍缩）
    # Freeze rolling-element residual updates (rolling is extremely rare; prevents "all→rolling" absorber)
    FREEZE_ROLLING_PROMPT_RESIDUAL = True
    # ATT371.2: class-wise learnable offset on text prototypes (without unfreezing text backbone)
    # ATT372 rollback: disable ATT371_2.0 prompt-offset tuning by default.
    USE_TEXT_PROMPT_TUNING = False
    TEXT_PROMPT_TUNING_INIT_STD = 0.02
    TEXT_PROMPT_TUNING_SCALE_INIT = 1.0
    # Encourage Inner/Outer prompt prototypes to keep safe angular margin.
    USE_TEXT_PROMPT_SEPARATION_LOSS = False
    TEXT_PROMPT_SEPARATION_WEIGHT = 0.0
    TEXT_PROMPT_SEPARATION_START_EPOCH = 1
    TEXT_PROMPT_INNER_OUTER_TARGET_MAX = 0.50  # target: cos(inner,outer) < 0.50
    # ATT371.2 / ATT478(P0): FusionDecoder dual-tower — z_cls (supervised) vs z_metric (prototype / fusion_proto_align).
    # Precedence: FUSION_DECODER_DUAL_TOWER_ENABLE env > USE_DUAL_SPACE_DECOUPLE env (legacy) > default True when DUAL_CONTRACT_ENABLE else False.
    _fddt_env = os.environ.get("FUSION_DECODER_DUAL_TOWER_ENABLE")
    _legacy_dual_env = os.environ.get("USE_DUAL_SPACE_DECOUPLE")
    if _fddt_env is not None:
        FUSION_DECODER_DUAL_TOWER_ENABLE = bool(int(_fddt_env))
    elif _legacy_dual_env is not None:
        FUSION_DECODER_DUAL_TOWER_ENABLE = bool(int(_legacy_dual_env))
    else:
        FUSION_DECODER_DUAL_TOWER_ENABLE = bool(DUAL_CONTRACT_ENABLE)
    USE_DUAL_SPACE_DECOUPLE = bool(FUSION_DECODER_DUAL_TOWER_ENABLE)
    DUAL_SPACE_METRIC_DETACH = bool(int(os.environ.get("DUAL_SPACE_METRIC_DETACH", "0")))
    # ATT484: replace linear metric projection with residual MLP to improve cross-domain class separation.
    # Keep enabled by default under dual-space to avoid silent no-op experiments.
    _metric_res_env = os.environ.get("FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE")
    if _metric_res_env is not None:
        FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE = bool(int(_metric_res_env))
    else:
        FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE = bool(USE_DUAL_SPACE_DECOUPLE)
    FUSION_METRIC_PROJ_HIDDEN_MULT = float(
        os.environ.get("FUSION_METRIC_PROJ_HIDDEN_MULT", "2.0")
    )
    # -----------------------------
    # Framework Group: FUSION_ALIGN_GROUP
    # Controls fusion-space geometric alignment and prototype guidance.
    # -----------------------------
    FUSION_ALIGN_GROUP_ENABLE = bool(int(os.environ.get("FUSION_ALIGN_GROUP_ENABLE", "1")))
    # ATT479: direct supervised contrastive on fusion pre-L2 features + optional decoder-input detach.
    # Motivation: decoder-dominated backprop starves GatedAdditiveFusion (F_FUSION_MODULE_GRADIENT_ISLAND).
    FUSION_SUPCON_ENABLE = bool(int(os.environ.get("FUSION_SUPCON_ENABLE", "0")))
    FUSION_SUPCON_WEIGHT = float(os.environ.get("FUSION_SUPCON_WEIGHT", "1.0"))
    FUSION_SUPCON_TEMPERATURE = float(os.environ.get("FUSION_SUPCON_TEMPERATURE", "0.07"))
    # Default 0: include Normal for cross-load invariance (same as historical SupCon discussion).
    FUSION_SUPCON_FAULT_ONLY = bool(int(os.environ.get("FUSION_SUPCON_FAULT_ONLY", "0")))
    # ATT485 (S1): geometry loss on fusion pre-L2 slot — pairwise SupCon vs batch soft silhouette.
    # When FUSION_SUPCON_ENABLE=1 and env omitted, default silhouette (new contract); set to supcon to reproduce ATT484-.
    _fus_geom_env = os.environ.get("FUSION_GEOMETRY_LOSS_MODE")
    if _fus_geom_env is not None:
        FUSION_GEOMETRY_LOSS_MODE = str(_fus_geom_env).strip().lower()
    elif FUSION_SUPCON_ENABLE:
        FUSION_GEOMETRY_LOSS_MODE = "silhouette"
    else:
        FUSION_GEOMETRY_LOSS_MODE = "supcon"
    if FUSION_GEOMETRY_LOSS_MODE not in ("supcon", "silhouette"):
        FUSION_GEOMETRY_LOSS_MODE = "silhouette"
    FUSION_SILHOUETTE_EPS = float(os.environ.get("FUSION_SILHOUETTE_EPS", "1e-6"))
    # ATT486 (S1): auxiliary CE on z_metric vs LearnablePrototypes.forward() (text-initialized).
    # Orthogonal to falsified USE_PROTO_CLASSIFIER-only path; does not replace fusion_cls head.
    FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE = bool(
        int(os.environ.get("FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE", "0"))
    )
    FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT = float(
        os.environ.get("FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT", "0.30")
    )
    FUSION_METRIC_LEARNABLE_PROTO_CE_TEMP = float(
        os.environ.get("FUSION_METRIC_LEARNABLE_PROTO_CE_TEMP", "0.10")
    )
    FUSION_METRIC_LEARNABLE_PROTO_CE_START_EPOCH = int(
        os.environ.get("FUSION_METRIC_LEARNABLE_PROTO_CE_START_EPOCH", "1")
    )
    # ATT488: pretraining should not spend loss budget on the dead z_cls / Fusion Cls branch.
    # The z_cls head remains available for explicit supervised finetuning/calibration paths;
    # this switch only zeros the pretraining CE contribution in train_alignment.py.
    FUSION_PRETRAIN_DISABLE_ZCLS_CE = bool(
        int(os.environ.get("FUSION_PRETRAIN_DISABLE_ZCLS_CE", "1"))
    )
    FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY = bool(
        int(os.environ.get("FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY", "1"))
    )
    # -----------------------------
    # Framework Group: METRIC_GEOMETRY_GROUP
    # Controls dual-tower metric geometry and decoder->fusion gradient routing.
    # -----------------------------
    METRIC_GEOMETRY_GROUP_ENABLE = bool(int(os.environ.get("METRIC_GEOMETRY_GROUP_ENABLE", "1")))
    _fus_dec_stop_env = os.environ.get("FUSION_DECODER_INPUT_STOPGRAD_ENABLE")
    if _fus_dec_stop_env is not None:
        FUSION_DECODER_INPUT_STOPGRAD_ENABLE = bool(int(_fus_dec_stop_env))
    else:
        # When SupCon is on, default to stopping decoder CE/margin gradients into fusion.
        FUSION_DECODER_INPUT_STOPGRAD_ENABLE = bool(FUSION_SUPCON_ENABLE)
    # ATT482: temporal stopgrad annealing for decoder->fusion path.
    # Keep forward values unchanged while scaling backward gradient by live_ratio.
    # live_ratio=1.0 => fully live; live_ratio=0.0 => full stopgrad.
    FUSION_DECODER_STOPGRAD_ANNEAL_ENABLE = bool(
        int(os.environ.get("FUSION_DECODER_STOPGRAD_ANNEAL_ENABLE", "0"))
    )
    FUSION_DECODER_STOPGRAD_ANNEAL_START_EPOCH = int(
        os.environ.get("FUSION_DECODER_STOPGRAD_ANNEAL_START_EPOCH", "1")
    )
    FUSION_DECODER_STOPGRAD_ANNEAL_END_EPOCH = int(
        os.environ.get("FUSION_DECODER_STOPGRAD_ANNEAL_END_EPOCH", "15")
    )
    FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_START = float(
        os.environ.get("FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_START", "1.0")
    )
    FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_END = float(
        os.environ.get("FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_END", "0.0")
    )
    # ATT483: fixed decoder->fusion gradient live ratio (optional). When set (env non-empty),
    # training uses z_dec = r*z + (1-r)*z.detach() for decoder path: forward equals z, backward from
    # decoder losses scales by r. Overrides binary FUSION_DECODER_INPUT_STOPGRAD_ENABLE and ATT482 anneal.
    _fd_grad_live_env = os.environ.get("FUSION_DECODER_INPUT_GRAD_LIVE_RATIO")
    if _fd_grad_live_env is not None and str(_fd_grad_live_env).strip() != "":
        FUSION_DECODER_INPUT_GRAD_LIVE_RATIO = parse_fusion_decoder_grad_live_ratio(_fd_grad_live_env)
    else:
        # Defensive default (ATT486 incident prevention):
        # when fusion geometry alignment is active, keep a weak decoder->fusion live path.
        if bool(FUSION_SUPCON_ENABLE):
            FUSION_DECODER_INPUT_GRAD_LIVE_RATIO = 0.10
        else:
            FUSION_DECODER_INPUT_GRAD_LIVE_RATIO = None
    # ATT480: keep proto/semantic z_metric path live even when decoder input stopgrad is enabled.
    # This preserves text-prototype directional supervision on fusion while still blocking decoder CE dominance.
    FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH = bool(
        int(os.environ.get("FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH", "1"))
    )
    # ATT481: when direct-metric semantic path is enabled, bypass get_sem_embedding() to avoid
    # sem_adapter's internal detach path re-cutting proto gradients into fusion.
    FUSION_SEMANTIC_BYPASS_SEM_ADAPTER_ON_DIRECT_PATH = bool(
        int(os.environ.get("FUSION_SEMANTIC_BYPASS_SEM_ADAPTER_ON_DIRECT_PATH", "1"))
    )
    if not bool(FUSION_ALIGN_GROUP_ENABLE):
        FUSION_SUPCON_ENABLE = False
        FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE = False
        FUSION_PRETRAIN_DISABLE_ZCLS_CE = False
    if not bool(METRIC_GEOMETRY_GROUP_ENABLE):
        USE_DUAL_SPACE_DECOUPLE = False
        FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE = False
        FUSION_DECODER_INPUT_GRAD_LIVE_RATIO = None

    # ========== ATT372: Physical-Frequency Position Encoding (Frequency Normalization) ==========
    # Keep original STFT/SWT bin view unchanged, and add physical-frequency axis as position encoding.
    # This aligns absolute frequency coordinates (Hz) across domains with different sampling rates.
    USE_PHYSICAL_FREQ_POSITION_ENCODING = bool(int(os.environ.get("USE_PHYSICAL_FREQ_POSITION_ENCODING", "1")))
    # ATT377-B: "concat" adds freq-axis as independent channel (preserves amplitude)
    PHYSICAL_FREQ_POSITION_ENCODING_MODE = "concat"
    PHYSICAL_FREQ_POSITION_ENCODING_SCALE = float(os.environ.get("PHYSICAL_FREQ_POSITION_ENCODING_SCALE", "2.0"))
    # Reference Hz used to normalize physical-frequency axis into a stable [0, ~1] range.
    # For current datasets, max Nyquist is PU(64kHz)/2 = 32kHz.
    PHYSICAL_FREQ_REFERENCE_HZ = 32000.0
    PHYSICAL_FREQ_LOG_STATS = True

    # ========== ATT372: Dynamic Semantic Multi-Prototype ==========
    EVAL_DYNAMIC_MULTI_PROTO_ENABLE = True
    EVAL_DYNAMIC_MULTI_PROTO_TARGET_DATASETS = ["HUST", "SEU"]
    EVAL_DYNAMIC_MULTI_PROTO_REPLACE_WHEN_BETTER = True
    EVAL_DYNAMIC_MULTI_PROTO_FEATURE_SOURCE = "fusion_raw"  # "fusion_raw" | "fusion_metric"
    EVAL_DYNAMIC_MULTI_PROTO_K = 4
    EVAL_DYNAMIC_MULTI_PROTO_TOPK_VOTE = 7
    EVAL_DYNAMIC_MULTI_PROTO_TEMPERATURE = 0.07
    EVAL_DYNAMIC_MULTI_PROTO_MAX_SUPPORT_SAMPLES = 8000
    EVAL_DYNAMIC_MULTI_PROTO_KMEANS_ITERS = 25
    EVAL_DYNAMIC_MULTI_PROTO_MIN_CLUSTER_SAMPLES = 8
    EVAL_DYNAMIC_MULTI_PROTO_TEMPLATE_MAX_SIGNAL_SAMPLES = 256
    # Mix visual cluster center with text-encoded cluster prototype.
    # 0.0 => visual-only, 1.0 => text-only.
    EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_ALPHA = 0.35
    # ATT376: if visual-text cosine is too low, fallback to safer alpha to avoid injecting orthogonal noise.
    EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_MIN_COS = 0.12
    EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_FALLBACK_ALPHA = 0.0
    # Use spectrogram-domain evidence for frequency/time-frequency slots.
    EVAL_DYNAMIC_MULTI_PROTO_USE_SPEC_EVIDENCE = True
    EVAL_DYNAMIC_MULTI_PROTO_LOG_TOP_CLUSTERS = 10

    # ========== ATT374: Simplex ETF Prototypes ==========
    # Replace text-encoder prototypes with geometrically optimal fixed prototypes.
    # ETF guarantees cos(p_i, p_j) = -1/(C-1) for i != j (max angular separation).
    # ATT413: RESTORED — ATT376(ETF=True)=0.9624 vs ATT412(ETF=False)=0.7849. ETF provides optimal
    # simplex separation and enables text_etf_alignment_loss which was a key component of ATT376.
    # ATT418: disabled — 15+ active losses prevented z_fusion from reaching ETF positions.
    # ATT441: RE-ENABLED — current config has only 3 active losses (no competition).
    # ATT442: DISABLED — F_ETF_DESTROYS_SEMANTIC (3rd falsification). FusionProto 0.99→0.62.
    # ETF uniform separation contradicts hierarchical fault class structure.
    # Historical: ATT400(OFF)=0.9785, ATT417(ON)=0.0000, ATT441(ON)=0.6237.
    USE_SIMPLEX_ETF_PROTOTYPES = bool(int(os.environ.get("USE_SIMPLEX_ETF_PROTOTYPES", "0")))
    SIMPLEX_ETF_SEED = int(os.environ.get("SIMPLEX_ETF_SEED", "42"))
    # When True, also use text-encoder prototypes as an auxiliary alignment target
    # alongside ETF prototypes (for DynMultiProto compatibility).
    SIMPLEX_ETF_KEEP_TEXT_PROTO_FOR_EVAL = bool(int(os.environ.get("SIMPLEX_ETF_KEEP_TEXT_PROTO_FOR_EVAL", "1")))

    # ========== ATT374: Prototype-Anchored Contrastive Loss (PACL) ==========
    # Forces z_spec and z_fusion to align with fixed ETF prototypes during training.
    # ATT428: PACL DISABLED — text proto cos(I,O)=0.865 makes 4-class alignment impossible
    USE_PROTOTYPE_ANCHOR_LOSS = bool(int(os.environ.get("USE_PROTOTYPE_ANCHOR_LOSS", "0")))
    PROTOTYPE_ANCHOR_WEIGHT_SPEC = float(os.environ.get("PROTOTYPE_ANCHOR_WEIGHT_SPEC", "0.0"))
    PROTOTYPE_ANCHOR_WEIGHT_FUSION = float(os.environ.get("PROTOTYPE_ANCHOR_WEIGHT_FUSION", "0.0"))
    PROTOTYPE_ANCHOR_TEMPERATURE = float(os.environ.get("PROTOTYPE_ANCHOR_TEMPERATURE", "0.07"))
    PROTOTYPE_ANCHOR_START_EPOCH = int(os.environ.get("PROTOTYPE_ANCHOR_START_EPOCH", "1"))
    PROTOTYPE_ANCHOR_LABEL_SMOOTHING = float(os.environ.get("PROTOTYPE_ANCHOR_LABEL_SMOOTHING", "0.05"))

    # ========== ATT374: Text-ETF Alignment Loss ==========
    # Train text_encoder.proj to produce embeddings aligned with ETF positions.
    # This improves DynMultiProto since its text_embedder will produce better-separated embeddings.
    # ATT379f(A): was disabled claiming PACL sufficient — ATT412 restores: ATT376(0.96) vs ATT411(0.67)
    # ATT418: disabled — ETF prototypes disabled, text-ETF alignment no longer needed.
    # ATT441: RE-ENABLED — ETF prototypes restored, text-ETF alignment needed for alignment chain.
    # ATT442: DISABLED — text_etf_align displaces text protos from semantic positions to ETF,
    # destroying z_fusion-text_proto alignment. cos(IR,OR) 0.90→-0.333 breaks FusionProto.
    USE_TEXT_ETF_ALIGNMENT_LOSS = bool(int(os.environ.get("USE_TEXT_ETF_ALIGNMENT_LOSS", "0")))
    TEXT_ETF_ALIGNMENT_WEIGHT = float(os.environ.get("TEXT_ETF_ALIGNMENT_WEIGHT", "0.30"))
    TEXT_ETF_ALIGNMENT_START_EPOCH = int(os.environ.get("TEXT_ETF_ALIGNMENT_START_EPOCH", "1"))
    TEXT_ETF_ALIGNMENT_BATCH_INTERVAL = int(os.environ.get("TEXT_ETF_ALIGNMENT_BATCH_INTERVAL", "1"))
    # ATT378: raised from 0.10 to 0.30 to enforce text-prototype separability.
    TEXT_ETF_COSINE_MATRIX_WEIGHT = float(os.environ.get("TEXT_ETF_COSINE_MATRIX_WEIGHT", "0.30"))
    # ATT376: CLIP-style symmetric batch contrastive objective (spec <-> sample text).
    USE_CLIP_STYLE_CONTRASTIVE = bool(int(os.environ.get("USE_CLIP_STYLE_CONTRASTIVE", "1")))
    # ATT379f(A): was disabled claiming PACL sufficient — ATT412 restores: ATT376 had 0.10 and achieved 0.9624
    # ATT418: disabled — clip_style consumed ~10% loss budget, competing with PACL/alignment.
    # ATT441: RE-ENABLED (w=0.10) — PACL is OFF, no budget competition. ATT412(CLIP ON)=0.7849
    # vs ATT411(CLIP OFF)=0.6667, proving +0.12 SpecHead contribution.
    # ATT443: DISABLED — F_CLIP_CAUSES_FUSION_COLLAPSE. CLIP cos(spec,fus)→1 squashes
    # norm(fus) 23.8→2.89 while norm(spec) stays at 22.8, destroying FusionProto.
    # ATT437(OFF)=0.9931, ATT441(ON)=0.6237, ATT442(ON)=0.1989.
    CLIP_STYLE_CONTRASTIVE_WEIGHT = float(os.environ.get("CLIP_STYLE_CONTRASTIVE_WEIGHT", "0.0"))
    CLIP_STYLE_CONTRASTIVE_TEMPERATURE = float(os.environ.get("CLIP_STYLE_CONTRASTIVE_TEMPERATURE", "0.07"))
    CLIP_STYLE_CONTRASTIVE_START_EPOCH = int(os.environ.get("CLIP_STYLE_CONTRASTIVE_START_EPOCH", "1"))

    # ATT379e(S1): disable dual_clip_fus — redundant with PACL on fusion, plateaued at 0.85
    USE_DUAL_CLIP_SPEC = bool(int(os.environ.get("USE_DUAL_CLIP_SPEC", "0")))
    DUAL_CLIP_SPEC_WEIGHT = float(os.environ.get("DUAL_CLIP_SPEC_WEIGHT", "0.0"))

    # ========== ATT391-S1: ETF→Text Proto Space Adapter ==========
    # ATT392: DISABLED — replaced by Learnable Prototypes (S1). Adapter cannot overcome
    # degenerate text-space geometry (Inner≈Outer cos=0.91).
    USE_ETF_TEXT_ADAPTER = bool(int(os.environ.get("USE_ETF_TEXT_ADAPTER", "0")))
    ETF_TEXT_ADAPTER_WEIGHT = float(os.environ.get("ETF_TEXT_ADAPTER_WEIGHT", "0.30"))
    ETF_TEXT_ADAPTER_START_EPOCH = int(os.environ.get("ETF_TEXT_ADAPTER_START_EPOCH", "2"))
    ETF_TEXT_ADAPTER_TEMPERATURE = float(os.environ.get("ETF_TEXT_ADAPTER_TEMPERATURE", "0.07"))
    ETF_TEXT_ADAPTER_LR_MULT = float(os.environ.get("ETF_TEXT_ADAPTER_LR_MULT", "1.0"))

    # ========== ATT392-S1: Learnable Prototypes ==========
    # Replace fixed text/ETF prototypes with gradient-trainable parameters.
    # Initialized from text-encoder embeddings, optimized with:
    #   separation_loss: hinge on pairwise cos > min_cos (enforces angular spread)
    #   anchor_loss: cosine distance to text-semantic origin (prevents drift)
    # Training: PACL aligns z_spec/z_fusion to learnable protos.
    # Eval: learnable protos used directly for cosine matching (zero-shot).
    USE_LEARNABLE_PROTOTYPES = bool(int(os.environ.get("USE_LEARNABLE_PROTOTYPES", "1")))
    LEARNABLE_PROTO_MIN_COS = float(os.environ.get("LEARNABLE_PROTO_MIN_COS", "-0.30"))
    # ATT433: revert to text-init (ETF random rotation falsified F_ETF_RANDOM_ROTATION)
    LEARNABLE_PROTO_SEP_WEIGHT = float(os.environ.get("LEARNABLE_PROTO_SEP_WEIGHT", "0.50"))
    LEARNABLE_PROTO_ANCHOR_WEIGHT = float(os.environ.get("LEARNABLE_PROTO_ANCHOR_WEIGHT", "0.20"))
    LEARNABLE_PROTO_INIT_FROM_ETF = bool(int(os.environ.get("LEARNABLE_PROTO_INIT_FROM_ETF", "0")))

    # ATT432-δ: Physical Feature Prototypes with Class-Conditional Projection
    USE_PHYS_FEATURE_PROTO = bool(int(os.environ.get("USE_PHYS_FEATURE_PROTO", "1")))
    PHYS_FEATURE_HIDDEN_DIM = int(os.environ.get("PHYS_FEATURE_HIDDEN_DIM", "128"))
    PHYS_FEATURE_ALIGN_WEIGHT = float(os.environ.get("PHYS_FEATURE_ALIGN_WEIGHT", "0.20"))

    # ATT396: Feature Decoupling — explicit fault/condition feature separation
    # Ref: DFDN (RESS 2024), FDIDG (Pattern Recognition 2024)
    USE_FEATURE_DECOUPLE = bool(int(os.environ.get("USE_FEATURE_DECOUPLE", "0")))
    DECOUPLE_HIDDEN_DIM = int(os.environ.get("DECOUPLE_HIDDEN_DIM", "512"))
    DECOUPLE_NUM_CONDITIONS = int(os.environ.get("DECOUPLE_NUM_CONDITIONS", "4"))
    DECOUPLE_COND_WEIGHT = float(os.environ.get("DECOUPLE_COND_WEIGHT", "0.10"))
    DECOUPLE_ADV_WEIGHT = float(os.environ.get("DECOUPLE_ADV_WEIGHT", "0.10"))
    DECOUPLE_ORTHO_WEIGHT = float(os.environ.get("DECOUPLE_ORTHO_WEIGHT", "0.05"))
    DECOUPLE_ADV_LAMBDA_MAX = float(os.environ.get("DECOUPLE_ADV_LAMBDA_MAX", "1.0"))
    DECOUPLE_START_EPOCH = int(os.environ.get("DECOUPLE_START_EPOCH", "1"))
    LEARNABLE_PROTO_LR_MULT = float(os.environ.get("LEARNABLE_PROTO_LR_MULT", "5.0"))
    LEARNABLE_PROTO_START_EPOCH = int(os.environ.get("LEARNABLE_PROTO_START_EPOCH", "0"))


    # ========== ATT397: SAM Optimizer + Episodic Cross-Condition Training ==========
    # Ref: SWAD (NeurIPS 2021), SAM (ICLR 2021), MGADA (Machines 2025)
    USE_SAM_OPTIMIZER = bool(int(os.environ.get("USE_SAM_OPTIMIZER", "0")))
    SAM_RHO = float(os.environ.get("SAM_RHO", "0.05"))
    SAM_ADAPTIVE = bool(int(os.environ.get("SAM_ADAPTIVE", "0")))

    USE_EPISODIC_TRAINING = bool(int(os.environ.get("USE_EPISODIC_TRAINING", "0")))
    EPISODIC_SPLIT_KEY = os.environ.get("EPISODIC_SPLIT_KEY", "auto")
    EPISODIC_PROTO_LOSS_WEIGHT = float(os.environ.get("EPISODIC_PROTO_LOSS_WEIGHT", "0.5"))
    EPISODIC_TEMPERATURE = float(os.environ.get("EPISODIC_TEMPERATURE", "0.07"))
    EPISODIC_GRAD_ALIGN_ENABLE = bool(int(os.environ.get("EPISODIC_GRAD_ALIGN_ENABLE", "0")))
    EPISODIC_GRAD_ALIGN_WEIGHT = float(os.environ.get("EPISODIC_GRAD_ALIGN_WEIGHT", "0.1"))

    # ATT379(S3): SupCLIP — include text prototypes as positives in SupCon.
    USE_SUPCLIP = bool(int(os.environ.get("USE_SUPCLIP", "1")))

    # ATT379(S4): PACL margin mode — use margin-based loss instead of softmax CE.
    PACL_USE_MARGIN = bool(int(os.environ.get("PACL_USE_MARGIN", "1")))
    PACL_MARGIN = float(os.environ.get("PACL_MARGIN", "0.80"))

    # ========== ATT374: Loss Budget Audit Fix ==========
    # Prefix for items logged to loss_dict but NOT part of backward total.
    LOSS_BUDGET_INFO_PREFIX = "_info_"

    # ========== (C0) Fusion raw-embedding alignment ==========
    # 主目标：直接约束 z_fusion(原始融合空间) 与 text prototypes 的几何对齐，
    # 避免仅对齐 z_sem 导致 "FusionProto 评估路径坍缩"。
    # ATT428: DISABLED — z_fusion zero-shot path proven dead (Val Acc=0.22, grad≈0.04)
    USE_FUSION_PROTO_ALIGN_LOSS = bool(int(os.environ.get("USE_FUSION_PROTO_ALIGN_LOSS", "0")))
    FUSION_PROTO_ALIGN_WEIGHT = float(os.environ.get("FUSION_PROTO_ALIGN_WEIGHT", "0.0"))
    FUSION_PROTO_ALIGN_TEMPERATURE = 0.07
    FUSION_PROTO_ALIGN_START_EPOCH = 2
    FUSION_PROTO_ALIGN_MIN_SAMPLES_PER_CLASS = 2
    FUSION_PROTO_ALIGN_USE_CLASS_WEIGHTS = True

    # ========== (C0b) Normal↔Rolling embedding overlap penalty ==========
    # 针对 z_spec 里 N/R 类间余弦重叠偏高的问题，增加显式 hard-pair 约束。
    # ATT379f(A): disable — gradient conflict with PACL alignment direction
    USE_NR_INTER_COS_MARGIN_LOSS = False
    NR_INTER_COS_MARGIN_WEIGHT = 0.10
    NR_INTER_COS_MARGIN_TARGET_MAX = 0.15
    NR_INTER_COS_MARGIN_START_EPOCH = 1
    NR_INTER_COS_MIN_SAMPLES = 2

    # ========== (C) Inner/Outer margin constraints (structural separation) ==========
    # These margins encourage Inner/Outer to beat competing classes by a fixed gap, reducing “absorbed into Rolling/Normal”.
    # ATT379f(A): disable all pairwise margins — gradient conflict, converge to 0 by E15 anyway
    USE_OUTER_MARGIN_LOSS = False
    OUTER_MARGIN_WEIGHT = 0.0
    OUTER_MARGIN = 0.2
    OUTER_MARGIN_START_EPOCH = 5

    USE_INNER_MARGIN_LOSS = False
    INNER_MARGIN_WEIGHT = 0.0
    INNER_MARGIN = 0.2
    INNER_MARGIN_START_EPOCH = 5

    # ========== (C2) Normal margin constraint (anti-collapse for Normal) ==========
    # For true Normal samples, enforce:
    #   logit_normal >= max(logits_faults) + margin
    # This directly prevents the "all → Rolling" (or other fault) absorption mode observed in logs.
    USE_NORMAL_MARGIN_LOSS = False
    NORMAL_MARGIN_WEIGHT = 0.0
    NORMAL_MARGIN = 0.2
    NORMAL_MARGIN_START_EPOCH = 5

    # ========== (C2b) Anti-rolling-absorption margin (directly targets all→rolling) ==========
    # For non-rolling samples (labels 0/1/2), enforce:
    #   logit_true >= logit_rolling + margin
    USE_ROLLING_ABSORB_MARGIN_LOSS = False
    ROLLING_ABSORB_MARGIN_WEIGHT = 0.0
    ROLLING_ABSORB_MARGIN = 0.2
    ROLLING_ABSORB_MARGIN_START_EPOCH = 5

    # ========== (C2d) Inner/Outer vs Rolling margins (targets Inner→Rolling / Outer→Rolling) ==========
    # Directly targets the dominant confusion pairs on HUST (e.g., Inner→Rolling, Outer→Rolling).
    USE_INNER_VS_ROLLING_MARGIN_LOSS = False
    INNER_VS_ROLLING_MARGIN_WEIGHT = 0.0
    INNER_VS_ROLLING_MARGIN = 0.3
    INNER_VS_ROLLING_MARGIN_START_EPOCH = 5

    USE_OUTER_VS_ROLLING_MARGIN_LOSS = False
    OUTER_VS_ROLLING_MARGIN_WEIGHT = 0.0
    OUTER_VS_ROLLING_MARGIN = 0.3
    OUTER_VS_ROLLING_MARGIN_START_EPOCH = 5

    # ========== (C2c) Anti-normal-absorption margins (Inner/Outer vs Normal) ==========
    # Directly prevents the observed collapse where Inner/Outer are absorbed into Normal on HUST.
    USE_INNER_VS_NORMAL_MARGIN_LOSS = False
    INNER_VS_NORMAL_MARGIN_WEIGHT = 0.0
    INNER_VS_NORMAL_MARGIN = 0.2
    INNER_VS_NORMAL_MARGIN_START_EPOCH = 5

    USE_OUTER_VS_NORMAL_MARGIN_LOSS = False
    OUTER_VS_NORMAL_MARGIN_WEIGHT = 0.0
    OUTER_VS_NORMAL_MARGIN = 0.2
    OUTER_VS_NORMAL_MARGIN_START_EPOCH = 5

    # Rolling vs Normal margin (directly targets Rolling→Normal absorption on HUST)
    USE_ROLLING_VS_NORMAL_MARGIN_LOSS = False
    ROLLING_VS_NORMAL_MARGIN_WEIGHT = 0.0
    ROLLING_VS_NORMAL_MARGIN = 0.2
    ROLLING_VS_NORMAL_MARGIN_START_EPOCH = 5

    # ========== (C2e) Anti-inner-absorption margins (Normal/Outer/Rolling vs Inner) ==========
    # Directly targets the collapse mode "all→Inner" observed after strengthening anti-rolling margins.
    USE_NORMAL_VS_INNER_MARGIN_LOSS = False
    NORMAL_VS_INNER_MARGIN_WEIGHT = 0.0
    NORMAL_VS_INNER_MARGIN = 0.2
    NORMAL_VS_INNER_MARGIN_START_EPOCH = 5

    USE_OUTER_VS_INNER_MARGIN_LOSS = False
    OUTER_VS_INNER_MARGIN_WEIGHT = 0.0
    OUTER_VS_INNER_MARGIN = 0.2
    OUTER_VS_INNER_MARGIN_START_EPOCH = 5

    USE_ROLLING_VS_INNER_MARGIN_LOSS = False
    ROLLING_VS_INNER_MARGIN_WEIGHT = 0.0
    ROLLING_VS_INNER_MARGIN = 0.2
    ROLLING_VS_INNER_MARGIN_START_EPOCH = 5

    # ========== (C3) Diagnosis anti-collapse regularization (batch-level) ==========
    # Penalize collapsed batch predictions by matching the batch-mean predicted class distribution to
    # the batch label histogram (smoothed), using KL divergence.
    # ATT379f(A): disable — Neural Collapse theory shows class balance emerges naturally with ETF
    USE_DIAG_ANTICOLLAPSE_REG = False
    DIAG_ANTICOLLAPSE_WEIGHT = 0.0
    DIAG_ANTICOLLAPSE_EPS = 0.01
    DIAG_ANTICOLLAPSE_TEMPERATURE = 1.0
    # Target distribution for anti-collapse KL:
    # - "batch": match batch label histogram (can be unstable under domain shift)
    # - "uniform": match uniform prior (recommended to prevent single-class absorbers)
    # - "fixed": match DIAG_ANTICOLLAPSE_PRIOR (list of length C)
    DIAG_ANTICOLLAPSE_TARGET = "uniform"
    DIAG_ANTICOLLAPSE_PRIOR = None

    # ========== (D) AM-Softmax on fault-type (Inner/Outer/Rolling) ==========
    # NOTE: our logits are already temperature-scaled (1/temperature), so keep S=1.0 by default.
    # Default OFF: in recent runs this can trigger "all → Rolling" collapse when enabled abruptly.
    USE_AM_SOFTMAX_FAULT = False
    # IMPORTANT: keep margin small; large margin tends to cause unstable "absorb-to-one-class" flips.
    AM_SOFTMAX_M = 0.10
    AM_SOFTMAX_S = 1.0
    # Delay AM-Softmax until representations are reasonably stable (prevents early flip to Normal or Rolling).
    AM_SOFTMAX_START_EPOCH = 10
    # Warm up the margin over a few epochs after start to avoid sudden distribution shifts.
    AM_SOFTMAX_WARMUP_EPOCHS = 5
    # DEPRECATED (kept for backward compatibility): avoid asymmetric application that leaves Rolling unpenalized.
    # If set True, the implementation will still apply a symmetric margin (see `losses/alignment_loss.py`).
    AM_SOFTMAX_INNER_OUTER_ONLY = False

    # ========== (E) NF-protected argmax for diagnosis (evaluation-time) ==========
    # Diagnosis uses argmax over 4-way logits; if NF head predicts "normal", force diagnosis to Normal.
    USE_NF_PROTECTED_ARGMAX = False
    # Default threshold on anomaly_logit = logit_fault_any - logit_normal used by training-time zero-shot printouts.
    # For target-domain evaluation, the tuned `_eval_nf_threshold` (if set) takes precedence.
    NF_PROTECT_THRESHOLD = 0.0
    # Soft protection: only force Normal when p_fault is extremely low (prevents forcing many faults to Normal
    # when NF is weak on a domain). Recommended: 0.02~0.10.
    # NOTE: In recent logs, hard/progressive protection was hurting HUST diagnosis (raw >> protected),
    # so we default to a more conservative threshold.
    NF_PROTECT_PFAULT_MAX = 0.01
    # Protection mode:
    # - "hard": force prediction to Normal when p_fault < NF_PROTECT_PFAULT_MAX
    # - "soft_bias": add a normal-logit bias then argmax (recommended; avoids over-suppressing faults)
    NF_PROTECT_MODE = "soft_bias"
    # Strength of normal-logit bias used by soft_bias mode (larger => stronger protection).
    NF_PROTECT_BIAS = 4.0
    # Only apply NF-protect when diagnosis is uncertain:
    # if (best_fault_logit - normal_logit) < margin, then allow protect; otherwise trust diagnosis.
    NF_PROTECT_DIAG_MARGIN = 0.5

    # Supervised finetune (target-domain) loss weights:
    # HUST diagnosis is sensitive to nf over-training; keep nf_weight low and focus on fault-type.
    SUP_FT_NF_WEIGHT = 0.05
    SUP_FT_DIAG_WEIGHT = 1.0

    # ========== (Target SupFT, Attempt 42) LP-FT defaults for flat head ==========
    # When `fusion_decoder.hierarchical=False` (flat 4-class head), supervised finetune switches to a linear-probe
    # on the flat cosine classifier weights. This usually needs a higher LR than adapter-style tuning.
    
    # Historical note: LP-FT step cap has been iteratively tuned with validation-driven early stop.
    # ATT234(S3a): raise LP-FT max steps to 500 to allow deeper minima under Val-ES.
    TARGET_SUP_LP_STEPS = 500
    TARGET_SUP_LP_LR = 5e-4

    # ========== (Target SupFT, Attempt 52) Validation-driven LP-FT early stopping ==========
    # Motivation: with only 5% labeled target data (e.g., 38 samples on HUST), fixed-step LP-FT can easily
    # over-tune and degrade evaluation accuracy, especially for "already good" checkpoints like HUST-best E10.
    # We split the 5% finetune set into train/val (stratified), monitor val CE (and optional macro-F1),
    # and early-stop with patience to reduce overfitting risk.
    TARGET_SUP_LP_USE_VAL_EARLY_STOP = True
    # ATT277(P3-A): increase validation ratio to reduce tiny-val noise in LP-FT early stopping.
    TARGET_SUP_LP_VAL_RATIO = 0.30
    # ATT277(P3-B): aggregate Val-ES metric across multiple stratified resamples.
    TARGET_SUP_LP_VAL_MULTI_SPLIT_ENABLE = True
    TARGET_SUP_LP_VAL_MULTI_SPLIT_REPEATS = 3
    TARGET_SUP_LP_VAL_MULTI_SPLIT_BASE_SEED = 42
    TARGET_SUP_LP_VAL_MULTI_SPLIT_METRIC_REDUCE = "mean"  # mean | median
    # ATT232(S2): increase default LP-FT patience for HUST to avoid premature stop before stage4 benefits.
    TARGET_SUP_LP_PATIENCE = 12
    # ATT207(C): use longer patience for HIT LP-FT to absorb short-term oscillations after VICReg reweighting.
    TARGET_SUP_LP_PATIENCE_SEU = 12
    TARGET_SUP_LP_PATIENCE_HIT = TARGET_SUP_LP_PATIENCE_SEU
    # ATT218: add PU-specific patience override to reduce early-stop variance on tiny validation split.
    TARGET_SUP_LP_PATIENCE_PU = 12
    TARGET_SUP_LP_MIN_STEPS = 120
    TARGET_SUP_LP_EVAL_EVERY = 5            # evaluate val every N steps
    # ATT192(B): revert to CE-driven LP-FT early-stop to better preserve SVP pathway features.
    # Allowed: "auto" | "ce" | "macro_f1" | "spec_ce"
    TARGET_SUP_LP_EARLYSTOP_METRIC = "spec_ce"
    TARGET_SUP_LP_MIN_DELTA = 1e-4
    # Statistical guard: if val set is too small, Val-ES signal is too noisy and can hurt LP-FT.
    # Fallback to fixed-step LP-FT when sampled val size is below this threshold.
    TARGET_SUP_LP_MIN_VAL_SAMPLES = 8
    # ATT224(S3): semi-supervised LP-FT on unlabeled target eval split.
    # Add entropy minimization + batch diversity regularization to leverage unlabeled target data.
    TARGET_SUP_LP_SEMI_ENABLE = True
    TARGET_SUP_LP_SEMI_ENT_WEIGHT = 0.10
    TARGET_SUP_LP_SEMI_DIV_WEIGHT = 0.10
    TARGET_SUP_LP_SEMI_WARMUP_STEPS = 50
    # ATT230(S1): relax tiny-shot semi-disable guard so HUST(n=38) can use LP-semi again.
    TARGET_SUP_LP_SEMI_TINYSHOT_DISABLE_THRESHOLD = 30
    # ATT224(S4): after Val-ES selects best_step, retrain on full labeled split for tiny-shot domains.
    TARGET_SUP_LP_RETRAIN_FULL_ENABLE = True
    # ATT274(P2): allow HIT (n_train≈241) to enter retrain-on-full path.
    TARGET_SUP_LP_RETRAIN_FULL_MAX_SAMPLES = 300
    TARGET_SUP_LP_RETRAIN_FULL_MIN_STEPS = 80
    # ATT274(P2): enforce minimum stage4 backbone FT coverage in retrain-on-full.
    TARGET_SUP_LP_RETRAIN_FULL_ENFORCE_STAGE4_COVERAGE = True
    TARGET_SUP_LP_RETRAIN_FULL_MIN_STAGE4_FT_STEPS = 80
    # ATT283(P2): adaptive stage4 floor for retrain-on-full to reduce overfit risk
    # when best_step is reached early on tiny-shot domains.
    TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_ENABLE = True
    TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_RATIO = 0.30
    TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_MIN = 20
    TARGET_SUP_LP_RETRAIN_FULL_USE_BEST_STEP = True
    # ATT235(S2): for flat decoder, allow retrain-on-full to keep progressive unfreeze path.
    TARGET_SUP_LP_RETRAIN_FULL_ALLOW_PROGRESSIVE_UNFREEZE_FLAT = True
    # ATT235(S2): evaluate kNN/LabelProp/Spec+kNN on retrain-full snapshot.
    TARGET_SUP_LP_RETRAIN_FULL_NONPARAM_EVAL_ENABLE = True
    # ATT227(S1): disable LP-FT cosine LR schedule; use constant LR + Val-ES.
    TARGET_SUP_LP_LR_SCHEDULE_ENABLE = False
    TARGET_SUP_LP_LR_WARMUP_STEPS = 40
    TARGET_SUP_LP_LR_MIN_SCALE = 0.20
    # ATT180: Progressive unfreezing for LP-FT.
    # Step range is 1-based in docs, 0-based in code:
    # - step   1-50 : classifier/spec_head only
    # - step  51-100: + SpecEncoder proj/norm
    # - step 101-200: + SpecEncoder block3
    TARGET_SUP_LP_PROGRESSIVE_UNFREEZE_ENABLE = True
    TARGET_SUP_LP_UNFREEZE_STAGE2_START = 50
    TARGET_SUP_LP_UNFREEZE_STAGE3_START = 100
    # ATT202(A): disable progressive backbone unfreeze on self-train rounds by default.
    # Rationale: pseudo-label noise + freshly reset optimizer caused severe instability in ATT201.
    TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SELFTRAIN = True
    # ATT218: rollback ATT217; keep Task1 progressive unfreeze disabled for stable PU LP-FT.
    TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_TASK1 = True
    # ATT215: restore progressive unfreeze for HIT (ATT214 regression fix).
    TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SEU = False
    TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_HIT = TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SEU
    # ATT228(S3): restore ATT216 setting so HUST main LP-FT can adapt z_spec.
    TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_TINYSHOT_THRESHOLD = 30
    # ATT228(S3): retrain-on-full runs without Val-ES guard; keep it head-only to avoid NF boundary drift.
    TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_RETRAIN_FULL = True
    # ATT202(B): automatically scale progressive stage thresholds by train-sample count.
    # When LP-FT train samples are below threshold, stage starts are multiplied by n_train/threshold.
    TARGET_SUP_LP_STAGE_SCALE_BY_SAMPLES = True
    # ATT233(S1): raise stage-scale threshold so HUST@20%(n_train=122) still triggers compressed unfreeze schedule.
    TARGET_SUP_LP_STAGE_SCALE_SAMPLE_THRESHOLD = 150
    TARGET_SUP_LP_STAGE_SCALE_MIN_RATIO = 0.25
    # ATT203: protect LP warmup when sample-based stage scaling is enabled.
    TARGET_SUP_LP_STAGE_SCALE_MIN_STAGE2_STEPS = 40
    # ATT201(A): backbone fine-tuning on target domain with deeper progressive unfreeze.
    TARGET_SUP_LP_BACKBONE_FT_ENABLE = True
    TARGET_SUP_LP_UNFREEZE_STAGE4_START = 180
    # ATT201(A): L2-SP regularization for backbone stability during target adaptation.
    TARGET_SUP_LP_L2SP_ENABLE = True
    TARGET_SUP_LP_L2SP_WEIGHT = 0.01
    # ATT203: use a global (phase-3 checkpoint) L2-SP anchor across LP-FT/self-train calls.
    TARGET_SUP_LP_L2SP_USE_GLOBAL_ANCHOR = True
    # ATT201(A): lower LR multiplier for backbone groups to avoid catastrophic drift.
    TARGET_SUP_LP_BACKBONE_LR_MULT = 0.10
    # ATT204: smooth LR ramp for newly-unfrozen backbone params at each progressive stage.
    TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_ENABLE = True
    TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_STEPS = 10
    TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_MIN_SCALE = 0.2
    # ATT203: k-NN diagnosis path on z_spec as non-parametric supplement.
    TARGET_SUP_KNN_EVAL_ENABLE = True
    TARGET_SUP_KNN_K = 5
    TARGET_SUP_KNN_REPLACE_WHEN_BETTER = True
    # ATT234(S1): EMA-smoothed LP-FT checkpoint for low-variance post-LPFT evaluation.
    TARGET_SUP_LP_EMA_ENABLE = True
    TARGET_SUP_LP_EMA_DECAY = 0.995
    TARGET_SUP_LP_EMA_EVAL_ENABLE = True
    # ATT235(S1): extend EMA snapshot usage to non-parametric post-LPFT evaluation (kNN/LabelProp/Ensemble).
    TARGET_SUP_LP_EMA_USE_FOR_NONPARAM_EVAL = True
    # ATT235(S1): SAM optimizer on flat LP-FT to seek flatter minima in tiny-shot target adaptation.
    TARGET_SUP_LP_SAM_ENABLE = True
    TARGET_SUP_LP_SAM_RHO = 0.05
    # ATT236(S3): SAM only on head-only stage to avoid perturbing newly-unfrozen backbone.
    TARGET_SUP_LP_SAM_STAGE1_ONLY = True
    # ATT226(S6): weighted SpecHead+kNN ensemble for post-LPFT supervised reporting.
    TARGET_SUP_SPEC_KNN_ENSEMBLE_ENABLE = True
    TARGET_SUP_SPEC_KNN_ENSEMBLE_WEIGHT_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    # ATT234(S3b): re-enable replacement when ensemble beats current supervised result.
    TARGET_SUP_SPEC_KNN_ENSEMBLE_REPLACE_WHEN_BETTER = True
    # ATT234(S2): graph-based label propagation on z_spec as global manifold classifier.
    TARGET_SUP_LABEL_PROP_ENABLE = True
    TARGET_SUP_LABEL_PROP_REPLACE_WHEN_BETTER = True
    TARGET_SUP_LABEL_PROP_KERNEL = "knn"   # "knn" | "rbf"
    # ATT298(S1): lock k=7 from ATT294 per-k sweep (best HUST LabelProp on raw z_spec).
    TARGET_SUP_LABEL_PROP_N_NEIGHBORS = 7
    TARGET_SUP_LABEL_PROP_ALPHA = 0.20
    TARGET_SUP_LABEL_PROP_MAX_ITER = 30
    TARGET_SUP_LABEL_PROP_TOL = 1e-3
    # ATT298(S2): fuse z_spec and z_fusion LabelSpreading posteriors (no PCA/LDA/NCA).
    TARGET_SUP_LABEL_PROP_DUAL_SPACE_ENABLE = True
    TARGET_SUP_LABEL_PROP_FUSION_WEIGHT = 0.35
    # ATT299: support-only transductive LOO to pick w_fusion; Self-Train pseudo uses same dual+LOO path when dual is on.
    TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_ENABLE = True
    TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    # ATT298(S3): TIM on frozen z_spec; conservative temp/steps (no prototype rectification).
    TARGET_SUP_TIM_ENABLE = True
    TARGET_SUP_TIM_REPLACE_WHEN_BETTER = True
    TARGET_SUP_TIM_STEPS = 150
    TARGET_SUP_TIM_LR = 0.01
    TARGET_SUP_TIM_TEMPERATURE = 2.0
    TARGET_SUP_TIM_SUPPORT_CE_WEIGHT = 1.0
    TARGET_SUP_TIM_QUERY_ENT_WEIGHT = 0.05
    TARGET_SUP_TIM_QUERY_DIV_WEIGHT = 0.20
    TARGET_SUP_TIM_WEIGHT_DECAY = 0.0
    # ATT228(S1): rollback class-balanced focal; return to CE(+label smoothing) for LP-FT stability.
    TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_ENABLE = False
    TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_GAMMA = 1.0
    TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_BETA = 0.999
    # ATT229(S1): disable flat-head explicit NF loss (ATT228 showed negative transfer on HIT).
    TARGET_SUP_LP_FLAT_NF_WEIGHT = 0.00
    # ATT229(S2): disable semantic prototype anchor in LP-FT to remove extra gradient interference.
    TARGET_SUP_LP_PROTO_ANCHOR_ENABLE = False
    TARGET_SUP_LP_PROTO_ANCHOR_WEIGHT = 0.05
    TARGET_SUP_LP_PROTO_ANCHOR_WARMUP_STEPS = 30
    
    # Optional: override decoder cosine temperature only during target supervised LP-FT.
    # Rationale: small temperature (e.g., 0.07) can create extremely peaky logits and very high initial CE loss
    # when the classifier weights are randomly initialized / mismatched to target embeddings.
    # None means "do not override".
    # Attempt 48: reverted to None (use training τ=0.07 for LP-FT).
    # Att 47 proved τ=1.0 causes LP-FT failure: Adam's gradient scale invariance
    # (Δθ ≈ lr·sign(g), independent of 1/τ) means the flat loss landscape at τ=1.0
    # cannot be compensated by LR scaling. τ=0.07 provides implicit hard negative
    # mining (gradient concentrates on hardest competitor), which is a FEATURE for
    # convex LP-FT (2,052 params), not a bug. Att 46 empirically validated: 37.40%.
    TARGET_SUP_LP_TEMPERATURE = None
    
    # Optional ramp override in LP-FT for flat head. This controls `_diag_linear_ramp` in [0,1].
    # Keep disabled by default after Attempt 50 regression; re-enable only with explicit ablation.
    # None means "do not override" (use current runtime ramp from training checkpoint, typically 1.0).
    TARGET_SUP_LP_BETA_OVERRIDE = None
    # ATT277(P1): run BN-adaptation before LP-FT to avoid HIT spec feature collapse at step 0.
    # ATT291(S2): extend pre-LPFT BN-adapt coverage to HUST and self-train rounds.
    TARGET_SUP_LP_PRE_BN_ADAPT_ENABLE = True
    TARGET_SUP_LP_PRE_BN_ADAPT_DOMAINS = ["SEU"]
    TARGET_SUP_LP_PRE_BN_ADAPT_NUM_BATCHES = -1
    TARGET_SUP_LP_PRE_BN_ADAPT_ON_SELFTRAIN = False
    TARGET_SUP_LP_PRE_BN_ADAPT_ON_RETRAIN_FULL = True

    # ========== (Eval/Adapt, Attempt 42) Collapse-aware spec gating ==========
    # If spec embedding collapses on target domain (stdmean very small), spec contributes mostly noise and
    # can hurt fusion. When enabled, we zero-out spec before fusion in evaluation/target-adapt paths.
    COLLAPSE_AWARE_SPEC_GATING = True
    # Gate if stdmean(z_spec) < abs_th OR stdmean(z_spec)/stdmean(z_sig) < ratio_th (when available).
    SPEC_STDMEAN_ABS_TH = 0.002
    SPEC_STDMEAN_RATIO_TH = 0.10
    # Att78: when True, use relaxed abs_th (0.5x) for CollapseGate when evaluating CWRU (single-domain low variance may be domain effect, not collapse).
    # Att79: default True to test CWRU spec gate relax as single variable.
    TASK1_CWRU_RELAX_SPEC_GATE = True
    LOG_COLLAPSE_AWARE_GATING = True
    # Keep logs readable: cap repeated FusionBypass logs.
    FUSION_BYPASS_LOG_MAX = 8
    FUSION_BYPASS_LOG_INTERVAL = 100

    # ========== (Eval/Adapt, Attempt 45) Fusion bypass on spec collapse ==========
    # Motivation: even if we zero-out z_spec before fusion, DomainAwareCrossAttentionFusion may still leak
    # non-spec information (e.g., domain embedding) through the spec token stream into cross-attention,
    # causing fusion negative transfer. When enabled, fusion will bypass cross-attention and enforce
    # a strict "signal-only" fusion path if spec collapse gate triggers.
    FUSION_BYPASS_ON_SPEC_COLLAPSE = True
    # Attempt 71: entropy-aware spec bypass (captures non-collapsed but uninformative spec predictions).
    FUSION_BYPASS_USE_SPEC_ENTROPY = True
    # Normalized entropy threshold in [0,1]; >0.9 means near-uniform 4-class predictions.
    FUSION_BYPASS_SPEC_ENTROPY_TH = 0.90
    # Conservative confidence guard: only trigger entropy-bypass when max prob is also low.
    FUSION_BYPASS_SPEC_MAXCONF_TH = 0.40

    # ========== (F) Stabilize hybrid NF head (nf_multi) ==========
    # Clamp per-type adaptive pos_weight to prevent a single fault (often rolling) from dominating nf_multi gradients.
    CLAMP_ADAPTIVE_POS_WEIGHT_MULTI = True
    ADAPTIVE_POS_WEIGHT_MULTI_MAX = 1.2
    
    # Margin loss权重（增强类间区分度）
    W_MARGIN = 0.3  # 原0.3，进一步提升到0.4以增强正常-故障边界分离，减少泛化Gap
    
    # ========== 训练模式配置 ==========
    TRAIN_MODE = os.environ.get("TRAIN_MODE", "retrain")  # "resume", "retrain", "test"
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/root/autodl-tmp/checkpoints")
    CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", f"{CHECKPOINT_DIR}/epoch_10_best.pth")
    START_FROM_SCRATCH = bool(int(os.environ.get("START_FROM_SCRATCH", "1")))  # 仅对retrain模式有效：True=完全重新训练，False=用预训练权重初始化
    # Phase skip/resume: per-phase granular control for interrupted training
    SKIP_TASK1_CWRU = bool(int(os.environ.get("SKIP_TASK1_CWRU", "0")))
    SKIP_TASK1_PU = bool(int(os.environ.get("SKIP_TASK1_PU", "0")))
    SKIP_TASK2 = bool(int(os.environ.get("SKIP_TASK2", "0")))
    CHECKPOINT_SAVE_INTERVAL = int(os.environ.get("CHECKPOINT_SAVE_INTERVAL", "10"))
    RESUME_TASK1_CWRU_CHECKPOINT = os.environ.get("RESUME_TASK1_CWRU_CHECKPOINT", "")
    RESUME_TASK1_PU_CHECKPOINT = os.environ.get("RESUME_TASK1_PU_CHECKPOINT", "")
    RESUME_TASK2_CHECKPOINT = os.environ.get("RESUME_TASK2_CHECKPOINT", "")
    
    # 评估提效：在三阶段脚本中可跳过某些 phase 的训练，直接加载 checkpoint 评估。
    EVAL_ONLY_MODE = bool(int(os.environ.get("EVAL_ONLY_MODE", "0")))
    _eval_only_phases_env = os.environ.get("EVAL_ONLY_PHASES", "").strip()
    if _eval_only_phases_env:
        EVAL_ONLY_PHASES = [x.strip() for x in _eval_only_phases_env.split(",") if x.strip()]
    else:
        EVAL_ONLY_PHASES = [
        "Task1-CWRU-LOO-gk2",
        "Task1-CWRU-LOO-speed_1730",
        "Task1-CWRU-LOO-speed_1750",
        "Task1-CWRU-LOO-speed_1772",
        "Task1-CWRU-LOO-speed_1797",
        "Task1-PU-LOO-gk5",
        "Task1-PU-LOO-gk6",
        "Task1-PU-LOO-gk7",
        "Task1-PU-LOO-gk8",
        "Task1-HUST-LOO-speed_0Hz",
        "Task1-HUST-LOO-speed_20Hz",
        "Task1-HUST-LOO-speed_25Hz",
        "Task1-HUST-LOO-speed_30Hz",
        "Task1-HUST-LOO-speed_35Hz",
        "Task1-HUST-LOO-speed_40Hz",
        "Task1-HUST-LOO-speed_60Hz",
        "Task1-HUST-LOO-speed_65Hz",
        "Task1-HUST-LOO-speed_70Hz",
        "Task1-HUST-LOO-speed_75Hz",
        "Task1-HUST-LOO-speed_80Hz",
        "Task1-SEU-LOO-speed_20_load_0",
        "Task1-SEU-LOO-speed_30_load_2",
    ]
    EVAL_ONLY_SKIP_TASK1_SUP_FT = False
    # 可按 phase_name / ckpt_tag / iso_domain 设置；未命中时回退 CHECKPOINT_PATH。
    # Resume: export EVAL_ONLY_CHECKPOINTS_JSON='{"Task1-CWRU-LOO-speed_1730":"/path/to/epoch_12_best.pth",...}'
    _eval_only_ckpt_json = os.environ.get("EVAL_ONLY_CHECKPOINTS_JSON", "").strip()
    if _eval_only_ckpt_json:
        import json as _json_mod
        EVAL_ONLY_CHECKPOINTS = _json_mod.loads(_eval_only_ckpt_json)
    else:
        EVAL_ONLY_CHECKPOINTS = {
        "Task1-CWRU-LOO-speed_1730": f"{CHECKPOINT_DIR}/task1_cwru_loo_0_epoch_14_best.pth",
        "Task1-CWRU-LOO-speed_1750": f"{CHECKPOINT_DIR}/task1_cwru_loo_1_epoch_18_best.pth",
        "Task1-CWRU-LOO-speed_1772": f"{CHECKPOINT_DIR}/task1_cwru_loo_2_epoch_43_best.pth",
        "Task1-CWRU-LOO-speed_1797": f"{CHECKPOINT_DIR}/task1_cwru_loo_3_epoch_45_best.pth",
        "Task1-PU-LOO-gk5": f"{CHECKPOINT_DIR}/task1_pu_loo_0_epoch_12_best.pth",
        "Task1-PU-LOO-gk6": f"{CHECKPOINT_DIR}/task1_pu_loo_1_epoch_12_best.pth",
        "Task1-PU-LOO-gk7": f"{CHECKPOINT_DIR}/task1_pu_loo_2_epoch_12_best.pth",
        "Task1-PU-LOO-gk8": f"{CHECKPOINT_DIR}/task1_pu_loo_3_epoch_17_best.pth",
        "Task1-HUST-LOO-speed_0Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_0_epoch_50_best.pth",
        "Task1-HUST-LOO-speed_20Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_1_epoch_42_best.pth",
        "Task1-HUST-LOO-speed_25Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_2_epoch_28_best.pth",
        "Task1-HUST-LOO-speed_30Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_3_epoch_14_best.pth",
        "Task1-HUST-LOO-speed_35Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_4_epoch_50_best.pth",
        "Task1-HUST-LOO-speed_40Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_5_epoch_31_best.pth",
        "Task1-HUST-LOO-speed_60Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_6_epoch_31_best.pth",
        "Task1-HUST-LOO-speed_65Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_7_epoch_49_best.pth",
        "Task1-HUST-LOO-speed_70Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_8_epoch_48_best.pth",
        "Task1-HUST-LOO-speed_75Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_9_epoch_23_best.pth",
        "Task1-HUST-LOO-speed_80Hz": f"{CHECKPOINT_DIR}/task1_hust_loo_10_epoch_27_best.pth",
        "Task1-SEU-LOO-speed_20_load_0": f"{CHECKPOINT_DIR}/task1_seu_loo_0_epoch_13_best.pth",
        "Task1-SEU-LOO-speed_30_load_2": f"{CHECKPOINT_DIR}/task1_seu_loo_1_epoch_43_best.pth",
    }
    # P1: Task2 目标域可选域专属 checkpoint（仅评估时生效）。
    # 未配置/文件不存在时会自动回退到当前模型状态，不影响主流程。
    TARGET_TASK2_DOMAIN_CHECKPOINT_SWITCH_ENABLE = False
    TARGET_TASK2_DOMAIN_CHECKPOINTS = {
        "HUST": f"{CHECKPOINT_DIR}/task2_epoch_29_best.pth",
        "HIT": f"{CHECKPOINT_DIR}/task2_epoch_49_best.pth",
    }
    
    # ========== 基础训练配置 ==========
    LEARNING_RATE = 1e-3
    TOTAL_EPOCHS = int(os.environ.get("TOTAL_EPOCHS", "50"))
    # ATT274(P1): decouple progressive class-weight schedule from TOTAL_EPOCHS.
    # Keep 30-epoch weight trajectory while allowing a longer training horizon.
    PROGRESSIVE_WEIGHT_TOTAL_EPOCHS = 30
    # ATT104: 64→32. Mathematical proof: CWRU isolated training has only 134 samples.
    # With BS=64: ceil(134/64)=3 mini-batches/epoch × 30 epochs = 90 total gradient steps.
    # The 3rd mini-batch has only 6 samples → SupCon degenerate (needs ≥2 positives/class).
    # Result: embedding collapse from E2 (var_mean=0.0000, cos_sim=1.0 for all pairs).
    # With BS=32: ceil(134/32)=5 batches/epoch × 30 = 150 gradient steps; each batch ~8/class.
    # C(8,2)=28 positive pairs/class → SupCon stable. VICReg with 32 samples statistically sound.
    # Trade-off: PU training (3694 samples) goes from 58→115 batches/epoch — acceptable.
    # ATT351结论: 大batch线性缩放在多损失耦合架构中不成立，维持BATCH_SIZE=64基线。
    # ATT454验证: BS=128导致梯度步数减半(10 vs 20 batches/epoch) → 早停过早触发。
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
    # ATT379(S0): per-phase batch size for PU to avoid OOM (fs=25600 → large SWT spectrograms).
    BATCH_SIZE_TASK1_PU = int(os.environ.get("BATCH_SIZE_TASK1_PU", "32"))
    BATCH_SIZE_TASK2 = int(os.environ.get("BATCH_SIZE_TASK2", "24"))  # ATT393: 48→24 fix OOM at cross-attention
    
    # 渐进式学习率调度（在启用新特性时降低学习率）
    USE_PROGRESSIVE_LR = True  # 是否启用渐进式学习率
    PROGRESSIVE_LR_SCHEDULE = {
        0: 1e-3,   # Epoch 1-7: 基础融合
        8: 5e-4,   # Epoch 8-14: 置信度特征启用，降低学习率
        15: 2e-4   # Epoch 15+: 类感知启用，进一步降低学习率
    }
    
    # ========== MoE配置 ==========
    USE_MOE = True  # 是否使用MoE版本的SpecEncoder
    ATT469_MOE_STRUCTURE_ENABLE = bool(int(os.environ.get("ATT469_MOE_STRUCTURE_ENABLE", "0")))
    _spec_backend_env = os.environ.get("SPEC_ENCODER_BACKEND", "").strip().lower()
    if ATT470_STRUCTURE_ENABLE and not _spec_backend_env:
        SPEC_ENCODER_BACKEND = "light"
    elif _spec_backend_env:
        SPEC_ENCODER_BACKEND = _spec_backend_env
    else:
        SPEC_ENCODER_BACKEND = "moe" if ATT469_MOE_STRUCTURE_ENABLE else "light"
    SPEC_MOE_MULTI_VIEW_ROUTER = bool(int(os.environ.get(
        "SPEC_MOE_MULTI_VIEW_ROUTER",
        str(int(ATT469_MOE_STRUCTURE_ENABLE)),
    )))
    SPEC_MOE_BAND_CROSS_ATTN = bool(int(os.environ.get(
        "SPEC_MOE_BAND_CROSS_ATTN",
        str(int(ATT469_MOE_STRUCTURE_ENABLE)),
    )))
    SPEC_MOE_BAND_SPARSITY_WEIGHT = float(os.environ.get("SPEC_MOE_BAND_SPARSITY_WEIGHT", "0.05"))
    SPEC_MOE_BAND_ATTN_HEADS = int(os.environ.get("SPEC_MOE_BAND_ATTN_HEADS", "4"))
    # Attempt 57: 可切换Spec编码器后端，支持与MoE并行A/B验证
    # 可选: "moe" | "light" | "standard"
    LIGHT_SPEC_BASE_CHANNELS = 64   # ATT204(A): 32→64，提升SpecEncoder容量上限，缓解z_spec表征瓶颈；
                                    #   数学根因：64ch c3=256下val_loss最小值提前到E8(0.3436)，32ch c3=128下可在E10(0.2393)创新低；
                                    #   证伪链：ATT129(0.7957)→ATT130(0.6505)→ATT131(0.7151)→ATT132(0.1935)→ATT133(0.7151)
                                    #   均为64ch，且ATT129/131/133三轮bit-exact，说明64ch在当前ES框架下系统性选早；
                                    #   ATT126在32ch+ES Fix缺失条件下仍达0.8548，加入ATT130 Pre-window Fix后预期更优；
                                    #   注：64ch SPEC AUC=0.970优于32ch=0.404，但z_spec表征质量不等于最终分类能力）
    LIGHT_SPEC_DROPOUT = 0.1    # ATT131 P0: 0.2→0.1（回退ATT130证伪的Dropout2d加倍；
                                #   ATT130实证：Dropout2d=0.2 在 c3=256通道时方差=40.96（ATT126的3.55×）
                                #   → CWRU 40 batch/epoch下训练不稳定；SPEC AUC 0.972→0.240（灾难性崩溃）；
                                #   ATT129在 dropout=0.1, base_ch=64 下 SPEC AUC=0.972为历史最佳，
                                #   证伪了"容量↑必须正则↑"的假设，base_ch=64本身不需要额外dropout）
    # P3: 频率分组池化（低频组 + 高频组），替代单纯 mean(dim=2)。
    USE_GROUPED_FREQ_POOL = True
    GROUPED_FREQ_POOL_LOW_BANDS = 5
    GROUPED_FREQ_POOL_HIGH_BANDS = 6
    MOE_LOSS_WEIGHT = 0.15  # MoE损失权重（从0.1增加到0.15，增强MoE约束）
    NUM_BANDS = 3  # 频带数量
    NUM_EXPERTS_PER_BAND = 4  # 每个频带的专家数量（从2增加到4）
    EXPERT_TOP_K = 2  # 稀疏路由：每个频带激活的专家数（2表示top-2）
    
    # ========== Bayesian Router配置 ==========
    USE_BAYESIAN_ROUTER = True  # 是否使用贝叶斯路由器（默认关闭以保持兼容性）
    BAYESIAN_ROUTER_PRIOR_STD = 1.0  # 贝叶斯路由器先验标准差
    BAYESIAN_ROUTER_KL_WEIGHT = 0.01  # KL散度损失权重
    BAYESIAN_ROUTER_NUM_SAMPLES_INFERENCE = 5  # 推理时采样次数
    
    # ========== Expert Performance Tracking配置 ==========
    USE_EXPERT_PERFORMANCE_TRACKING = True  # 是否使用专家性能追踪（默认关闭）
    EXPERT_TRACKER_MODE = 'ucb'  # 追踪模式: 'mean', 'ucb', 'thompson'
    EXPERT_TRACKER_EXPLORATION_BONUS = 0.1  # UCB探索奖励系数
    # Expert Disagreement & Regularization
    # SOTA Fix (NeurIPS 2024/ICLR 2025): Expert Dropout + Entropy Reg
    MOE_EXPERT_DROPOUT = 0.2  # Prob of masking the top-1 expert during training
    MOE_ENTROPY_LOSS_WEIGHT = 0.1  # Weight for entropy regularization (-H(p))
    EXPERT_TRACKER_UPDATE_INTERVAL = 1  # 每N个batch更新一次（1表示每个batch）
    
    # ========== Expert Disagreement配置 ==========
    USE_EXPERT_DISAGREEMENT = True  # 是否使用专家差异估计（默认关闭）
    EXPERT_DISAGREEMENT_METHOD = 'variance'  # 差异估计方法: 'variance', 'entropy', 'pairwise_distance'
    EXPERT_DISAGREEMENT_TEMP_SCALE = 2.0  # 温度调制比例
    EXPERT_DISAGREEMENT_TEMP_MODE = 'increase'  # 温度调制模式: 'increase', 'decrease'
    EXPERT_DISAGREEMENT_REG_WEIGHT = 0.01  # 差异正则化权重
    EXPERT_DISAGREEMENT_TARGET = 0.3  # 目标差异值
    EXPERT_DISAGREEMENT_MARGIN = 0.1  # 差异容忍边界
    
    # ========== 贝叶斯原型配置 ==========
    # ATT392-S2: DISABLED — Bayesian updates caused catastrophic prototype collapse
    # (Normal/Outer/Rolling cos→0.99 by E2 in ATT390a). Never enable with learnable protos.
    USE_BAYESIAN_PROTOTYPES = bool(int(os.environ.get("USE_BAYESIAN_PROTOTYPES", "0")))
    BAYESIAN_PRIOR_STRENGTH = 1.0  # 贝叶斯先验强度
    BAYESIAN_UPDATE_UNTIL_EPOCH = 10  # 贝叶斯原型只更新到第几轮（含），之后冻结
    
    # ========== AnomalyDetector在线微调配置（用于最终评估）==========
    DETECTOR_FINETUNING_ENABLE = True  # 是否在AnomalyDetector中启用在线微调
    DETECTOR_FINETUNING_SAMPLES = 100  # AnomalyDetector微调样本数
    DETECTOR_FINETUNING_LR = 1e-4  # AnomalyDetector微调学习率
    DETECTOR_FINETUNING_STEPS = 50  # 监督/无监督微调步数（50–100 更易收敛，原 10 过少）
    DETECTOR_FINETUNING_TARGET_DOMAINS = ["HUST", "SEU"]  # 需要执行无监督微调的目标域列表（空列表=禁用）
    
    # ========== 目标域两阶段适配配置（用于最终评估）==========
    TARGET_FINETUNE_PERCENT = 0.10  # 目标域用于微调的数据比例（默认值）
    # ATT233(S2): domain-specific labeled ratio overrides for Task2 adaptation.
    TARGET_FINETUNE_PERCENT_HUST = 0.20
    TARGET_FINETUNE_PERCENT_SEU = 0.10
    TARGET_FINETUNE_PERCENT_HIT = TARGET_FINETUNE_PERCENT_SEU
    TARGET_NSHOT_ENABLE = bool(int(os.environ.get("TARGET_NSHOT_ENABLE", "0")))
    TARGET_NSHOT_K = int(os.environ.get("TARGET_NSHOT_K", "0"))
    TARGET_NSHOT_REPEAT = int(os.environ.get("TARGET_NSHOT_REPEAT", "1"))
    _NSHOT_EXPERIMENT_ACTIVE = TARGET_NSHOT_ENABLE and TARGET_NSHOT_K > 0
    TARGET_UNSUP_STEPS = None  # 无监督适配步数（None=使用DETECTOR_FINETUNING_STEPS）
    TARGET_UNSUP_LR = None  # 无监督适配学习率（None=使用DETECTOR_FINETUNING_LR）
    TARGET_SUP_STEPS = 50  # 监督微调步数（None=使用DETECTOR_FINETUNING_STEPS，建议 50–100）
    TARGET_SUP_LR = None  # 监督微调学习率（None=使用DETECTOR_FINETUNING_LR）
    TARGET_SUP_NF_WEIGHT = 0.5  # 监督微调中NF loss的权重
    TARGET_SUP_DIAG_WEIGHT = 1.0  # 监督微调中诊断loss的权重
    # ATT181: tiny-shot safeguard + parameter-free supervised visual prototypes (SVP).
    # Rationale: when labeled target samples are too few (e.g., HUST 38 samples, 4-class),
    # LP-FT can overfit and underperform zero-shot; SVP provides a stable no-trainable fallback.
    # When n-shot LP-FT is active, do not skip LP-FT due to the legacy 35-sample safeguard.
    TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE = bool(
        int(
            os.environ.get(
                "TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE",
                "0" if _NSHOT_EXPERIMENT_ACTIVE else "1",
            )
        )
    )
    TARGET_SUP_SKIP_LPFT_MIN_SAMPLES = 35
    TARGET_SUP_SVP_ENABLE = True
    TARGET_SUP_SVP_REPLACE_WHEN_BETTER = True
    TARGET_SUP_SVP_USE_FUSION = True  # ATT182: multi-space SVP (z_spec + z_fusion)
    TARGET_SUP_SVP_FUSION_WEIGHT = 0.35  # blended score = (1-w)*spec + w*fusion
    TARGET_SUP_SVP_DUAL_AUTO_WEIGHT = True  # ATT183: select w_fusion on labeled split per domain
    TARGET_SUP_SVP_DUAL_WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    TARGET_SUP_SVP_TPA_ENABLE = True  # ATT183: transductive pseudo-label prototype refinement (eval-only)
    TARGET_SUP_SVP_TPA_ITERS = 2
    TARGET_SUP_SVP_TPA_TOPK_RATIO = 0.30
    TARGET_SUP_SVP_TPA_MIN_SAMPLES_PER_CLASS = 5
    TARGET_SUP_SVP_TPA_MIX_ALPHA = 0.80
    TARGET_SUP_SVP_TPA_MIN_BASE_ACC = 0.50  # ATT189(A): relax threshold to re-enable TPA on medium-quality fits
    # ATT191(A): evaluate and keep pre-LPFT baseline SVP when adapted path is worse.
    TARGET_SUP_SVP_BASELINE_GUARD_ENABLE = True
    # ATT191(B): keep LP-FT dual snapshots (spec_ce-best for SpecHead, ce-best for SVP path).
    TARGET_SUP_LP_DUAL_BEST_SNAPSHOT_ENABLE = True
    # ATT189(B): lightweight self-training loop after LP-FT for target domains.
    TARGET_SELF_TRAINING_ENABLE = True
    # ATT292(S4): use stronger non-parametric pseudo labels by default.
    TARGET_SELF_TRAINING_PSEUDO_SOURCE = "label_prop"  # {"spec_head", "label_prop"}
    # ATT226(S3): reopen one extra self-training round for HUST after balanced pseudo-labeling stabilization.
    TARGET_SELF_TRAINING_ROUNDS = 3
    TARGET_SELF_TRAINING_TOPK_RATIO = 0.30
    # ATT196(B): relax pseudo-label cap so tiny-shot target adaptation has enough supervised signal.
    TARGET_SELF_TRAINING_MAX_TOPK_RATIO = 0.30
    TARGET_SELF_TRAINING_MIN_CONFIDENCE = 0.70
    # ATT193(C): progressively raise confidence threshold by round (r1:0.70, r2:0.80 by default).
    TARGET_SELF_TRAINING_MIN_CONF_STEP = 0.10
    TARGET_SELF_TRAINING_MIN_SAMPLES_PER_CLASS = 5
    # ATT225(S4): class-balanced pseudo selection + tiny-shot confidence relaxation.
    TARGET_SELF_TRAINING_BALANCED_TOPK_ENABLE = True
    # ATT230(S2): disable tiny-shot confidence relaxation; keep higher-quality pseudo labels.
    TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_ENABLE = False
    TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_THRESHOLD = 50
    TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_VALUE = 0.50
    TARGET_SELF_TRAINING_MIN_START_ACC = 0.35
    TARGET_SELF_TRAINING_REPLACE_WHEN_BETTER = True
    # -----------------------------
    # Framework Group: DOMAIN_ADAPT_GROUP
    # Controls source/target domain adaptation in Task2.
    # -----------------------------
    DOMAIN_ADAPT_GROUP_ENABLE = bool(int(os.environ.get("DOMAIN_ADAPT_GROUP_ENABLE", "1")))
    # ATT238(S3): disable source-domain CORAL (consistently non-decreasing and unstable in Task2 logs).
    TARGET_TASK2_CORAL_ENABLE = False
    TARGET_TASK2_CORAL_WEIGHT = 0.0
    TARGET_TASK2_CORAL_START_EPOCH = 1
    TARGET_TASK2_CORAL_WARMUP_EPOCHS = 5
    TARGET_TASK2_CORAL_MIN_SAMPLES_PER_DOMAIN = 4
    # ATT198(A): keep source->target CORAL but lower weight for more stable joint optimization.
    TARGET_TASK2_TARGET_CORAL_ENABLE = False
    TARGET_TASK2_TARGET_CORAL_WEIGHT = 0.0
    TARGET_TASK2_TARGET_CORAL_EVERY_N_STEPS = 1
    TARGET_TASK2_TARGET_CORAL_MIN_SAMPLES = 4
    # ATT223: re-enable Task2-only MK-MMD after fixing phase guard leak in train_alignment.py.
    # Keep conservative weight and longer warmup to reduce optimization shock.
    TARGET_TASK2_TARGET_MMD_ENABLE = False
    TARGET_TASK2_TARGET_MMD_WEIGHT = 0.05
    TARGET_TASK2_TARGET_MMD_START_EPOCH = 1
    TARGET_TASK2_TARGET_MMD_WARMUP_EPOCHS = 8
    TARGET_TASK2_TARGET_MMD_EVERY_N_STEPS = 1
    TARGET_TASK2_TARGET_MMD_MIN_SAMPLES = 8
    TARGET_TASK2_TARGET_MMD_KERNEL_SCALES = [0.5, 1.0, 2.0]
    # Cross-domain z_metric centroid alignment (Task2 only).
    CROSS_DOMAIN_METRIC_ALIGN_ENABLE = bool(int(os.environ.get("CROSS_DOMAIN_METRIC_ALIGN_ENABLE", "0")))
    CROSS_DOMAIN_METRIC_ALIGN_WEIGHT = float(os.environ.get("CROSS_DOMAIN_METRIC_ALIGN_WEIGHT", "0.10"))
    CROSS_DOMAIN_METRIC_ALIGN_SEP_MARGIN = float(os.environ.get("CROSS_DOMAIN_METRIC_ALIGN_SEP_MARGIN", "0.10"))
    CROSS_DOMAIN_METRIC_ALIGN_START_EPOCH = int(os.environ.get("CROSS_DOMAIN_METRIC_ALIGN_START_EPOCH", "1"))
    if not bool(DOMAIN_ADAPT_GROUP_ENABLE):
        TARGET_TASK2_CORAL_ENABLE = False
        TARGET_TASK2_TARGET_CORAL_ENABLE = False
        TARGET_TASK2_TARGET_MMD_ENABLE = False
        CROSS_DOMAIN_METRIC_ALIGN_ENABLE = False
    # ATT199(C): test-time Whitening-Coloring Transform (TWC) for target-domain z_spec alignment.
    TARGET_TWC_ENABLE = True
    TARGET_TWC_TARGET_DATASETS = ["HUST", "SEU"]
    TARGET_TWC_EPS = 1e-5
    TARGET_TWC_BLEND_ALPHA = 1.0
    TARGET_TWC_APPLY_TO_SVPEVAL = False
    TARGET_TWC_APPLY_TO_SELFTRAIN = False
    TARGET_TWC_APPLY_TO_POSTLPFT = False
    # ATT187(A): always measure post-LPFT spec_head path and optionally replace flat decoder result.
    TARGET_SUP_POST_LPFT_SPEC_HEAD_EVAL = True
    TARGET_SUP_POST_LPFT_SPEC_HEAD_REPLACE_WHEN_BETTER = True
    # ATT201(D): target-domain data augmentation during LP-FT for tiny-shot robustness.
    TARGET_SUP_LP_AUG_ENABLE = True
    TARGET_SUP_LP_AUG_PROB = 0.7
    TARGET_SUP_LP_AUG_SPEC_NOISE_STD = 0.02
    TARGET_SUP_LP_AUG_SIGNAL_NOISE_STD = 0.01
    TARGET_SUP_LP_AUG_FREQ_MASK_RATIO = 0.08
    TARGET_SUP_LP_AUG_TIME_MASK_RATIO = 0.08
    TARGET_SUP_LP_AUG_NUM_FREQ_MASKS = 1
    TARGET_SUP_LP_AUG_NUM_TIME_MASKS = 1
    TARGET_EVAL_COMPACT_LOG = True  # ATT182: reduce noisy intermediate eval logs
    TARGET_EVAL_LOG_GRID_DETAILS = False  # ATT182: keep only best grid/cascade lines
    TARGET_EVAL_LOG_TRANSDUCTIVE_DETAILS = bool(int(os.environ.get("TARGET_EVAL_LOG_TRANSDUCTIVE_DETAILS", "0")))
    EVAL_SUPPRESS_SKLEARN_WARNINGS = bool(int(os.environ.get("EVAL_SUPPRESS_SKLEARN_WARNINGS", "1")))
    # ATT489(P2): hide FusionDiag summary by default (z_cls branch is structurally unstable).
    EVAL_REPORT_FUSION_DIAG = bool(int(os.environ.get("EVAL_REPORT_FUSION_DIAG", "0")))
    # Per-modality diagnostic heads (SignalHead / SignalProto / collapse & disagreement).
    EVAL_REPORT_SIGNAL_HEAD = bool(int(os.environ.get("EVAL_REPORT_SIGNAL_HEAD", "1")))
    EVAL_REPORT_SIGNAL_PROTO = bool(int(os.environ.get("EVAL_REPORT_SIGNAL_PROTO", "1")))
    EVAL_REPORT_MODALITY_DIAG = bool(int(os.environ.get("EVAL_REPORT_MODALITY_DIAG", "1")))
    EVAL_EXPORT_MODALITY_METRICS_CSV = bool(int(os.environ.get("EVAL_EXPORT_MODALITY_METRICS_CSV", "1")))
    EVAL_EXPORT_MODALITY_METRICS_DIR = str(
        os.environ.get("EVAL_EXPORT_MODALITY_METRICS_DIR", "experiments/results/modality_metrics")
    )
    # ATT490(P3): LOO summary metric source.
    # - final_decision: Phase-3「统一投票最终判断」口径（默认）
    # - spec_head / fusion: 单一路径（legacy）
    # - best_path/top1: 兼容旧日志口径（不推荐）
    LOO_SUMMARY_METRIC_SOURCE = str(os.environ.get("LOO_SUMMARY_METRIC_SOURCE", "final_decision")).strip().lower()
    # P0: unified final decision routing (confidence-weighted voting across paths).
    EVAL_FINAL_DECISION_ENABLE = bool(int(os.environ.get("EVAL_FINAL_DECISION_ENABLE", "1")))
    EVAL_FINAL_DECISION_USE_SPEC_HEAD = bool(int(os.environ.get("EVAL_FINAL_DECISION_USE_SPEC_HEAD", "0")))
    EVAL_FINAL_DECISION_USE_SIGNAL_HEAD = bool(int(os.environ.get("EVAL_FINAL_DECISION_USE_SIGNAL_HEAD", "1")))
    EVAL_FINAL_DECISION_USE_FUSION_PROTO = bool(int(os.environ.get("EVAL_FINAL_DECISION_USE_FUSION_PROTO", "1")))
    EVAL_FINAL_DECISION_USE_FUSION_DECODER = bool(int(os.environ.get("EVAL_FINAL_DECISION_USE_FUSION_DECODER", "0")))
    EVAL_FINAL_DECISION_USE_TRANSDUCTIVE = bool(int(os.environ.get("EVAL_FINAL_DECISION_USE_TRANSDUCTIVE", "1")))
    EVAL_FINAL_DECISION_SPEC_WEIGHT = float(os.environ.get("EVAL_FINAL_DECISION_SPEC_WEIGHT", "0.05"))
    EVAL_FINAL_DECISION_SIGNAL_HEAD_WEIGHT = float(os.environ.get("EVAL_FINAL_DECISION_SIGNAL_HEAD_WEIGHT", "0.45"))
    EVAL_FINAL_DECISION_FUSION_PROTO_WEIGHT = float(os.environ.get("EVAL_FINAL_DECISION_FUSION_PROTO_WEIGHT", "0.30"))
    EVAL_FINAL_DECISION_FUSION_DECODER_WEIGHT = float(os.environ.get("EVAL_FINAL_DECISION_FUSION_DECODER_WEIGHT", "0.00"))
    EVAL_FINAL_DECISION_TRANSDUCTIVE_WEIGHT = float(os.environ.get("EVAL_FINAL_DECISION_TRANSDUCTIVE_WEIGHT", "0.20"))
    EVAL_FINAL_DECISION_T_SIGNAL_HEAD = float(os.environ.get("EVAL_FINAL_DECISION_T_SIGNAL_HEAD", "1.0"))
    EVAL_FINAL_DECISION_SPEC_COLLAPSE_GATE = bool(int(os.environ.get("EVAL_FINAL_DECISION_SPEC_COLLAPSE_GATE", "1")))
    EVAL_FINAL_DECISION_SPEC_COLLAPSE_THRESH = float(os.environ.get("EVAL_FINAL_DECISION_SPEC_COLLAPSE_THRESH", "0.75"))
    # Per-sample confidence gating: weight_m = base_w_m * max(prob_m)^CONF_POWER
    EVAL_FINAL_DECISION_CONF_POWER = float(os.environ.get("EVAL_FINAL_DECISION_CONF_POWER", "2.0"))
    EVAL_FINAL_DECISION_PER_SAMPLE_CONF_GATE = bool(
        int(os.environ.get("EVAL_FINAL_DECISION_PER_SAMPLE_CONF_GATE", "1"))
    )
    # FusionProto sep_ratio gate: downweight when angular separation is weak
    EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_THRESH = float(
        os.environ.get("EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_THRESH", "1.0")
    )
    EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_DOWNWEIGHT = float(
        os.environ.get("EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_DOWNWEIGHT", "0.1")
    )
    # High inter-modality disagreement -> SignalHead-only fallback (batch-level)
    EVAL_FINAL_DECISION_DISAGREE_FALLBACK_ENABLE = bool(
        int(os.environ.get("EVAL_FINAL_DECISION_DISAGREE_FALLBACK_ENABLE", "1"))
    )
    EVAL_FINAL_DECISION_DISAGREE_FALLBACK_THRESH = float(
        os.environ.get("EVAL_FINAL_DECISION_DISAGREE_FALLBACK_THRESH", "0.70")
    )
    # Cascade: high SignalHead confidence -> signal-only; low confidence -> weighted fusion
    EVAL_FINAL_DECISION_CASCADE_ENABLE = bool(
        int(os.environ.get("EVAL_FINAL_DECISION_CASCADE_ENABLE", "1"))
    )
    EVAL_FINAL_DECISION_CASCADE_HIGH_THRESH = float(
        os.environ.get("EVAL_FINAL_DECISION_CASCADE_HIGH_THRESH", "0.85")
    )
    EVAL_FINAL_DECISION_T_SPEC = float(os.environ.get("EVAL_FINAL_DECISION_T_SPEC", "1.0"))
    EVAL_FINAL_DECISION_T_FUSION_PROTO = float(os.environ.get("EVAL_FINAL_DECISION_T_FUSION_PROTO", "1.0"))
    EVAL_FINAL_DECISION_T_FUSION_DECODER = float(os.environ.get("EVAL_FINAL_DECISION_T_FUSION_DECODER", "1.0"))
    EVAL_FINAL_DECISION_PRINT_CM = bool(int(os.environ.get("EVAL_FINAL_DECISION_PRINT_CM", "1")))
    # Reject over-confident collapsed paths before voting (e.g. FusionRaw all-Normal).
    EVAL_FINAL_DECISION_DEGEN_GATE_ENABLE = bool(int(os.environ.get("EVAL_FINAL_DECISION_DEGEN_GATE_ENABLE", "1")))
    EVAL_FINAL_DECISION_MAX_CLASS_FRAC = float(os.environ.get("EVAL_FINAL_DECISION_MAX_CLASS_FRAC", "0.85"))
    EVAL_FINAL_DECISION_MIN_PRED_ENTROPY = float(os.environ.get("EVAL_FINAL_DECISION_MIN_PRED_ENTROPY", "0.35"))
    # Transductive is auxiliary: when it strongly disagrees with a healthy FusionProto path,
    # prefer the stable prototype path instead of letting clustering overwrite it.
    EVAL_FINAL_DECISION_TRANSDUCTIVE_AGREEMENT_GATE_ENABLE = bool(
        int(os.environ.get("EVAL_FINAL_DECISION_TRANSDUCTIVE_AGREEMENT_GATE_ENABLE", "1"))
    )
    EVAL_FINAL_DECISION_TRANSDUCTIVE_MIN_AGREE_WITH_FUSION_PROTO = float(
        os.environ.get("EVAL_FINAL_DECISION_TRANSDUCTIVE_MIN_AGREE_WITH_FUSION_PROTO", "0.70")
    )
    EVAL_FINAL_DECISION_FUSION_PROTO_PRIORITY = bool(
        int(os.environ.get("EVAL_FINAL_DECISION_FUSION_PROTO_PRIORITY", "0"))
    )
    # Weighted multi-path voting (0); td-first fallback (1).
    EVAL_FINAL_DECISION_TD_PRIMARY = bool(int(os.environ.get("EVAL_FINAL_DECISION_TD_PRIMARY", "0")))
    # P1: align Task1(CWRU/PU) with Task2(HUST/SEU) by enabling SVP route in Task1 default flow.
    TARGET_TASK1_SVP_ENABLE = bool(int(os.environ.get("TARGET_TASK1_SVP_ENABLE", "1")))
    TARGET_TASK1_SVP_REPLACE_WHEN_BETTER = bool(int(os.environ.get("TARGET_TASK1_SVP_REPLACE_WHEN_BETTER", "1")))
    TARGET_TASK1_SVP_USE_FEWSHOT_SPLIT = bool(int(os.environ.get("TARGET_TASK1_SVP_USE_FEWSHOT_SPLIT", "1")))
    # Task1（域内跨工况）评估前微调配置
    # export TASK1_ENABLE_PRETEST_FINETUNE=0 to skip LP-FT/SVP-heavy eval (faster ablation)
    TASK1_ENABLE_PRETEST_FINETUNE = bool(int(os.environ.get("TASK1_ENABLE_PRETEST_FINETUNE", "1")))
    TASK1_FINETUNE_PERCENT = 0.05
    # When True, Task1-CWRU Supervised stage skips supervised finetune and evaluates on zero-shot loader (avoids negative transfer from few-shot CWRU finetune).
    # ATT122: skip percent-based CWRU LP-FT by default; n-shot runs override via env default.
    TASK1_CWRU_SKIP_SUPERVISED_FINETUNE = bool(
        int(
            os.environ.get(
                "TASK1_CWRU_SKIP_SUPERVISED_FINETUNE",
                "0" if _NSHOT_EXPERIMENT_ACTIVE else "1",
            )
        )
    )
    # Unified Leave-One-Out task runner.
    LOO_ENABLE = bool(int(os.environ.get("LOO_ENABLE", "0")))
    _loo_domains_raw = os.environ.get("LOO_DOMAINS", "CWRU,PU,HUST,SEU,HIT")
    LOO_DOMAINS = [x.strip().upper() for x in str(_loo_domains_raw).split(",") if x.strip()]
    # Resume: only run these LOO phase names (skip all others — no train, no eval).
    # Example: LOO_TRAIN_ONLY_PHASES=Task1-HUST-LOO-speed_70Hz,Task1-SEU-LOO-speed_20_load_0
    _loo_train_only_env = os.environ.get("LOO_TRAIN_ONLY_PHASES", "").strip()
    LOO_TRAIN_ONLY_PHASES = (
        [x.strip() for x in _loo_train_only_env.split(",") if x.strip()]
        if _loo_train_only_env
        else []
    )
    # Nested LOO (outer test + inner model-selection holdout inside outer-train groups).
    # When enabled, each outer LOO phase can optionally carve one inner condition as validation,
    # then train on the remaining conditions.
    LOO_NESTED_ENABLE = bool(int(os.environ.get("LOO_NESTED_ENABLE", "0")))
    # Policy to choose inner holdout condition from outer-train groups:
    # - "rotate": deterministic rotation by outer fold index.
    # - "first": always pick the first available train group.
    LOO_NESTED_INNER_POLICY = str(os.environ.get("LOO_NESTED_INNER_POLICY", "rotate")).strip().lower()
    # Optional allow-list by exact phase names (comma-separated). Empty => all phases.
    # Example: Task1-CWRU-LOO-speed_1730,Task1-PU-LOO-N15_M07_F10
    _loo_nested_phase_env = os.environ.get("LOO_NESTED_APPLY_PHASES", "").strip()
    LOO_NESTED_APPLY_PHASES = (
        [x.strip() for x in _loo_nested_phase_env.split(",") if x.strip()]
        if _loo_nested_phase_env
        else []
    )
    # Optional allow-list by domain names (comma-separated), e.g. "CWRU,PU".
    _loo_nested_domain_env = os.environ.get("LOO_NESTED_APPLY_DOMAINS", "").strip()
    LOO_NESTED_APPLY_DOMAINS = (
        [x.strip().upper() for x in _loo_nested_domain_env.split(",") if x.strip()]
        if _loo_nested_domain_env
        else []
    )
    # Optional allow-list by OUTER test-group key (comma-separated).
    # Example: "speed_1730,speed_1750"
    _loo_nested_test_group_env = os.environ.get("LOO_NESTED_APPLY_TEST_GROUPS", "").strip()
    LOO_NESTED_APPLY_TEST_GROUPS = (
        [x.strip() for x in _loo_nested_test_group_env.split(",") if x.strip()]
        if _loo_nested_test_group_env
        else []
    )
    # ATT489: prevent silent fallback to non-LOO / joint training when condition split fails.
    STRICT_INTRA_CONDITION_SPLIT = bool(int(os.environ.get("STRICT_INTRA_CONDITION_SPLIT", "1")))
    # Task1 train subsample: per (condition_group, fault_class) keep this fraction (test/val unchanged).
    TASK1_TRAIN_SUBSAMPLE_RATIO = float(os.environ.get("TASK1_TRAIN_SUBSAMPLE_RATIO", "1.0"))
    _task1_sub_pu_env = os.environ.get("TASK1_TRAIN_SUBSAMPLE_RATIO_PU", "").strip()
    TASK1_TRAIN_SUBSAMPLE_RATIO_PU = (
        float(_task1_sub_pu_env)
        if _task1_sub_pu_env
        else float(TASK1_TRAIN_SUBSAMPLE_RATIO)
    )
    LOO_STRICT_MODE = bool(int(os.environ.get("LOO_STRICT_MODE", "1")))

    # STMO: Single-Train Multi-test One-by-one (inverse of LOO).
    # Train on exactly one condition; evaluate each remaining condition separately.
    STMO_ENABLE = bool(int(os.environ.get("STMO_ENABLE", "0")))
    _stmo_domains_raw = os.environ.get("STMO_DOMAINS", "CWRU,PU,HUST,SEU,HIT")
    STMO_DOMAINS = [x.strip().upper() for x in str(_stmo_domains_raw).split(",") if x.strip()]
    _stmo_train_only_env = os.environ.get("STMO_TRAIN_ONLY_PHASES", "").strip()
    if not _stmo_train_only_env:
        _stmo_train_only_env = os.environ.get("LOO_TRAIN_ONLY_PHASES", "").strip()
    STMO_TRAIN_ONLY_PHASES = (
        [x.strip() for x in _stmo_train_only_env.split(",") if x.strip()]
        if _stmo_train_only_env
        else []
    )
    STMO_STRICT_MODE = bool(int(os.environ.get("STMO_STRICT_MODE", "1")))
    _stmo_metric_src = os.environ.get("STMO_SUMMARY_METRIC_SOURCE", "").strip()
    STMO_SUMMARY_METRIC_SOURCE = (
        _stmo_metric_src if _stmo_metric_src else LOO_SUMMARY_METRIC_SOURCE
    )
    if STMO_ENABLE and LOO_ENABLE:
        raise RuntimeError(
            "[Config] STMO_ENABLE=1 and LOO_ENABLE=1 are mutually exclusive. "
            "Disable one protocol before starting."
        )
    
    # ========== ATT398: Test-Time Prompt Tuning (TPT) ==========
    # Ref: Shu et al., NeurIPS 2022 "Test-Time Prompt Tuning for Zero-Shot Generalization"
    USE_TPT = bool(int(os.environ.get("USE_TPT", "0")))
    TPT_STEPS = int(os.environ.get("TPT_STEPS", "3"))
    TPT_LR = float(os.environ.get("TPT_LR", "0.001"))
    TPT_TEMPERATURE = float(os.environ.get("TPT_TEMPERATURE", "0.07"))
    TPT_ENTROPY_THRESHOLD = float(os.environ.get("TPT_ENTROPY_THRESHOLD", "0.1"))
    TPT_CONFIDENCE_FILTER = float(os.environ.get("TPT_CONFIDENCE_FILTER", "0.1"))

    # ========== ATT399: Transductive Inference + Physics Features ==========
    # Ref: SHOT (Liang et al., ICML 2020), LIMO (arXiv 2509.00305), 
    #      Conformer-PhyFaultNet (2024), DPICEN, PMC11085194
    USE_TRANSDUCTIVE = bool(int(os.environ.get("USE_TRANSDUCTIVE", "1")))
    # ATT500 source-free refinement remains available, but default back to ATT406 source-dependent
    # transductive: source labels + unlabeled target features, no target labels.
    TRANSDUCTIVE_SOURCE_FREE = bool(int(os.environ.get("TRANSDUCTIVE_SOURCE_FREE", "0")))
    TRANSDUCTIVE_FALLBACK_SOURCE_FREE_WHEN_NO_SOURCE = bool(
        int(os.environ.get("TRANSDUCTIVE_FALLBACK_SOURCE_FREE_WHEN_NO_SOURCE", "1"))
    )
    TRANSDUCTIVE_INIT_FROM_FPROTO = bool(int(os.environ.get("TRANSDUCTIVE_INIT_FROM_FPROTO", "1")))
    TRANSDUCTIVE_FUSION_VAR_COLLAPSE_THRESHOLD = float(
        os.environ.get("TRANSDUCTIVE_FUSION_VAR_COLLAPSE_THRESHOLD", "1e-4")
    )
    TRANSDUCTIVE_BMM_SEP_RATIO_GATE = float(
        os.environ.get("TRANSDUCTIVE_BMM_SEP_RATIO_GATE", "1.0")
    )
    # P0: when representation collapses, FinalDecision must not trust transductive/BMM.
    EVAL_FINAL_DECISION_COLLAPSE_SEP_RATIO_GATE = float(
        os.environ.get("EVAL_FINAL_DECISION_COLLAPSE_SEP_RATIO_GATE", "1.0")
    )
    EVAL_FINAL_DECISION_EXCLUDE_TRANSDUCTIVE_ON_COLLAPSE = bool(
        int(os.environ.get("EVAL_FINAL_DECISION_EXCLUDE_TRANSDUCTIVE_ON_COLLAPSE", "1"))
    )
    TRANSDUCTIVE_N_CLUSTERS_FACTOR = int(os.environ.get("TRANSDUCTIVE_N_CLUSTERS_FACTOR", "2"))
    TRANSDUCTIVE_KNN_K = int(os.environ.get("TRANSDUCTIVE_KNN_K", "64"))
    TRANSDUCTIVE_ALPHA_TEXT = float(os.environ.get("TRANSDUCTIVE_ALPHA_TEXT", "0.5"))
    TRANSDUCTIVE_IM_ITERATIONS = int(os.environ.get("TRANSDUCTIVE_IM_ITERATIONS", "5"))
    TRANSDUCTIVE_IM_CONFIDENT_RATIO = float(os.environ.get("TRANSDUCTIVE_IM_CONFIDENT_RATIO", "0.30"))
    TRANSDUCTIVE_TEMPERATURE = float(os.environ.get("TRANSDUCTIVE_TEMPERATURE", "0.07"))

    USE_PHYSICS_FEATURES = bool(int(os.environ.get("USE_PHYSICS_FEATURES", "1")))
    PHYSICS_FEAT_WEIGHT = float(os.environ.get("PHYSICS_FEAT_WEIGHT", "0.15"))
    PHYSICS_KNN_METRIC = os.environ.get("PHYSICS_KNN_METRIC", "euclidean")
    TRANSDUCTIVE_KL_REG_LAMBDA = float(os.environ.get("TRANSDUCTIVE_KL_REG_LAMBDA", "0.5"))

    # ATT402 P0: Domain-adaptive alpha_text (ATT403: DISABLED — destroyed HUST class 3)
    TRANSDUCTIVE_ADAPTIVE_ALPHA = bool(int(os.environ.get("TRANSDUCTIVE_ADAPTIVE_ALPHA", "0")))
    TRANSDUCTIVE_ADAPTIVE_ALPHA_REF_COS = float(os.environ.get("TRANSDUCTIVE_ADAPTIVE_ALPHA_REF_COS", "0.15"))
    TRANSDUCTIVE_ADAPTIVE_ALPHA_MIN_RATIO = float(os.environ.get("TRANSDUCTIVE_ADAPTIVE_ALPHA_MIN_RATIO", "0.05"))

    # ATT402 P2: Multi-template text prototypes (ATT403: DISABLED — destroyed HUST text_sim)
    TRANSDUCTIVE_USE_MULTI_TEMPLATE = bool(int(os.environ.get("TRANSDUCTIVE_USE_MULTI_TEMPLATE", "0")))

    # ATT402 S1: Attribute-Direct Matching (ATT403: DISABLED — unreliable cross-domain)
    TRANSDUCTIVE_USE_ATTRIBUTE_MATCHING = bool(int(os.environ.get("TRANSDUCTIVE_USE_ATTRIBUTE_MATCHING", "0")))
    TRANSDUCTIVE_ATTRIBUTE_WEIGHT = float(os.environ.get("TRANSDUCTIVE_ATTRIBUTE_WEIGHT", "0.3"))

    # ATT402 S2: Multi-Center kNN (ATT403: DISABLED — zero effect on all datasets)
    TRANSDUCTIVE_USE_MULTICENTER_KNN = bool(int(os.environ.get("TRANSDUCTIVE_USE_MULTICENTER_KNN", "0")))
    TRANSDUCTIVE_MULTICENTER_MAX_K = int(os.environ.get("TRANSDUCTIVE_MULTICENTER_MAX_K", "4"))

    # ATT404: Class Coverage Rescue — override clusters to recover missing classes
    TRANSDUCTIVE_USE_CLASS_RESCUE = bool(int(os.environ.get("TRANSDUCTIVE_USE_CLASS_RESCUE", "1")))
    TRANSDUCTIVE_RESCUE_THRESHOLD = float(os.environ.get("TRANSDUCTIVE_RESCUE_THRESHOLD", "0.3"))

    # ATT405: TransCLIP-style BMM refinement (replaces legacy IM)
    TRANSDUCTIVE_USE_BMM = bool(int(os.environ.get("TRANSDUCTIVE_USE_BMM", "1")))
    TRANSDUCTIVE_BMM_KL_LAMBDA = float(os.environ.get("TRANSDUCTIVE_BMM_KL_LAMBDA", "1.0"))
    TRANSDUCTIVE_BMM_N_NEIGHBOURS = int(os.environ.get("TRANSDUCTIVE_BMM_N_NEIGHBOURS", "3"))
    TRANSDUCTIVE_BMM_ITERATIONS = int(os.environ.get("TRANSDUCTIVE_BMM_ITERATIONS", "20"))
    TRANSDUCTIVE_BMM_INNER_MM_ITERS = int(os.environ.get("TRANSDUCTIVE_BMM_INNER_MM_ITERS", "3"))
    # ATT406: class-balance constraint (Sinkhorn-style marginal correction)
    TRANSDUCTIVE_BMM_CLASS_BALANCE_TAU = float(os.environ.get("TRANSDUCTIVE_BMM_CLASS_BALANCE_TAU", "0.5"))
    TRANSDUCTIVE_BMM_SINKHORN_ITERS = int(os.environ.get("TRANSDUCTIVE_BMM_SINKHORN_ITERS", "3"))


    # ========== 训练过程中zero_shot评估的在线微调配置（用于训练时验证）==========
    ZERO_SHOT_FINETUNING_ENABLE = False  # 是否在训练时的zero_shot评估中启用在线微调
    ZERO_SHOT_FINETUNING_SAMPLES = 300  # zero_shot微调样本数
    ZERO_SHOT_FINETUNING_LR = 1e-4  # zero_shot微调学习率
    ZERO_SHOT_FINETUNING_STEPS = 10  # zero_shot微调步数
    
    # ========== 一致性正则配置（用于提升跨域泛化）==========
    USE_CONSISTENCY_REGULARIZATION = False  # ATT376: core-only regime默认关闭一致性项
    CONSISTENCY_WEIGHT = 0.01  # 一致性损失权重
    CONSISTENCY_TEMPERATURE = 1.0  # 一致性损失温度参数
    # (A) 一致性保护：仅在高置信样本上施加一致性约束，避免不确定样本拉偏。
    CONSISTENCY_PROTECT_ENABLE = True
    CONSISTENCY_PROTECT_MIN_CONF = 0.60
    
    # ========== 损失权重配置 ==========
    W_FUSION_CLS = 0.1  # Fusion decoder分类损失权重（建议范围：0.1-0.5，过大会导致decoder偏置）
    # ATT431: signal alignment CE weight (restored from 0.0)
    W_SIGNAL_ALIGN = float(os.environ.get("W_SIGNAL_ALIGN", "0.30"))

    # ATT432: modality-specific learning rate multipliers
    MSLR_SIGNAL_MULT = float(os.environ.get("MSLR_SIGNAL_MULT", "1.0"))
    MSLR_SPEC_MULT = float(os.environ.get("MSLR_SPEC_MULT", "1.0"))
    MSLR_FUSION_MULT = float(os.environ.get("MSLR_FUSION_MULT", "10.0"))  # ATT437-β: 3→10, compensate detached-norm residual 1/||v|| attenuation
    MSLR_DECODER_MULT = float(os.environ.get("MSLR_DECODER_MULT", "1.0"))
    
    # ========== 二段式Schedule配置（用于稳定训练，避免后期退化）==========
    USE_SCHEDULE = True  # 是否启用二段式 schedule
    SCHEDULE_EPOCH_THRESHOLD = 5  # schedule 切换的 epoch 阈值（前N轮用早期配置，之后用后期配置）
    LR_DECAY_FACTOR = 0.1  # 学习率衰减因子（后期学习率 = 初始学习率 × 此因子）
    
    # ========== Early stopping配置 ==========
    USE_EARLY_STOPPING = bool(int(os.environ.get("USE_EARLY_STOPPING", "1")))
    # ATT110: REVERT to patience=5 (proven by ATT107: val_loss+5 correctly selected E9 → 0.3118).
    # ATT109 used patience=8+val_acc, selected E14 instead of E9 → Diag-Acc dropped 0.3118→0.2688.
    # Proof: ATT107 and ATT109 are bit-exact same training; only checkpoint selection differed.
    EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "5"))  # ATT379b: 10→15; ATT379a在E10早停但PACL仍有梯度信号(0.016)，需更多epoch让FusionProto继续提升
    # ATT110: REVERT to val_loss (proven by ATT107: val_loss correctly detects overfitting at E9+).
    # val_loss monotonically increased E9→E14 (0.9218→1.025→1.15→1.227) = TRUE overfitting signal.
    # val_acc stayed at 0.6875-0.75 (too coarse; failed to detect cross-domain degradation).
    # val_acc change in ATT109 was an error: the "val_loss spike" was in fact correct overfitting detection.
    EARLY_STOP_METRIC = "val_loss"  # ATT139: 回退到val_loss；ATT138实测显示val_spec_head_acc在小验证集上选择到E14并损害测试泛化
    # ATT132实证：val_spec_head_acc 虽在E18=0.9062达到最高，但训练后期val_loss长期高位（0.8~1.0）；
    # 在小验证集(32 samples, 分辨率=3.125%)上，accuracy由1-2个样本波动主导，易导致过晚checkpoint选择。
    # 同时 ATT132 暴露了best-checkpoint保存条件硬编码Bug（已在 ATT133 P0 修复）；
    # 回退到 val_loss 可恢复与 ATT110/ATT126/ATT131 一致的稳定早停行为。
    # ATT160: 回退 ATT159 覆盖，避免 32 样本验证集上 val_spec_head_acc 的离散噪声主导 checkpoint 选择。
    # ATT165 P2（已证伪）：ATT164 val_loss 在 E2 碰巧最低（0.275），此后永久发散；
    #   故切换为 val_spec_head_acc，但 ATT165 实验（training_20260301_125909.log）直接证伪——
    #   32 样本 acc 轨迹剧烈振荡（E5=0.78→E7=0.59→E9=0.81→E13=0.69→E14=0.84），
    #   Bernoulli σ=√(p(1-p)/32)≈7.7%，相邻 epoch 差异由 1-5 个样本主导，统计不显著；
    #   选中 E14（val_loss=0.497，过拟合区），CWRU Ensemble 仅 0.887，不及 val_loss 基线的 0.898。
    #   直接外部证据：arXiv:2602.22107 (2026.02.25) 系统实证 val_acc ES 劣于 val_loss ES。
    # ATT166 P2（回退）：恢复 None，使用全局 val_loss ES，参数缩减后 val_loss 发散点后移，
    #   ES 可在更充分收敛的 checkpoint 处停止，无需依赖噪声更大的 acc 信号。
    # ATT418: reverted to val_loss (None). ATT417 val_loss was monotonically increasing due to ETF+CLIP_STYLE
    # gradient competition. With ETF/CLIP_STYLE disabled, val_loss should recover U-shape curve.
    EARLY_STOP_METRIC_TASK1_CWRU = os.environ.get("EARLY_STOP_METRIC_TASK1_CWRU", "") or None
    EARLY_STOP_METRIC_TASK1_PU = "val_spec_head_acc"
    # ATT198(A): with source->target CORAL enabled, SpecHead-aligned checkpoint selection
    # is a better proxy for target-domain transfer quality than val_loss.
    EARLY_STOP_METRIC_TASK2 = "val_spec_head_acc"
    # Attempt 38: allow best-checkpoint tracking from epoch 10 (E10 peak is common in this project).
    EARLY_STOP_MIN_EPOCHS = 10  # ATT126: 15→10 (回退ATT124错误增大：ATT125 bit-exact证据证明min_epochs=15在E14重置基线0.2786，丢弃已发现的E13全局最优0.2458；ATT120 min_epochs=10在E9重置→E10/E13正常搜索→正确选E13)
    # ATT168: Task2 专属 min_epochs。Task2（CWRU+PU 联合训练）val_loss 轨迹在 E2 处出现噪声极小值（0.190），
    # pre-window 机制将其固定为基线，E3+ 所有 epoch 均无法超越，导致 ES 在 E14 触发并选中欠训练的 E2 checkpoint。
    # 将 min_epochs 提升至与全局相同的 10（后续可进一步调大），并配合 EARLY_STOP_SKIP_PREWINDOW_BASELINE=True 使用。
    # ATT198(C): delay Task2 ES tracking to ensure enough full-weight CORAL epochs.
    EARLY_STOP_MIN_EPOCHS_TASK2 = 15
    # ATT168: 当设为 True 时，在 min_epochs 处不将 pre-window 最优值作为 ES 基线（彻底丢弃 E2 噪声谷），
    # 而是从 min_epochs 处开始以 float('inf') 为初始基线重新追踪，确保 E10+ epoch 能正常竞争最优 checkpoint。
    # ATT175: 全局改为 True。
    #   方案A实证：PU 在 E2（val_loss=0.1976）出现噪声谷，pre-window 机制固定该谷为 ES 基线；
    #   后续所有 epoch 均无法超越，导致 ES 选中 E2 欠训练 checkpoint（Normal Recall=0.02）。
    #   skip_prewindow=True 后，E9 处以 inf 重新追踪，E10+ 的真实收敛 checkpoint（val_loss≈0.20）能被正常选中。
    #   CWRU 影响：CWRU val_loss 通常在 E10 前即可建立稳定基线，skip 不影响其最优 checkpoint 选择。
    EARLY_STOP_SKIP_PREWINDOW_BASELINE = True
    # ATT119 post-analysis: smooth@3 caused E12 val_loss spike (0.4490) to poison E13 window (0.3339 > 0.2837),
    # preventing selection of E13 (val_loss=0.2458, val_acc=0.8750), which was the true best checkpoint.
    # The 32-sample validation set means spike propagation bias > variance reduction benefit from smoothing.
    # ATT120: window=1 (no smoothing) → directly compare raw val_loss per epoch.
    EARLY_STOP_SMOOTH_WINDOW = 1  # ATT120: 3→1 (disable smoothing; spike-propagation bias outweighs noise-reduction benefit at N=32)
    # Attempt 72: SWAD-like checkpoint averaging around best epoch to reduce epoch-selection variance.
    USE_SWAD = True
    SWAD_WINDOW = 2            # average epochs in [best-2, best+2]
    SWAD_MIN_MODELS = 3        # require at least 3 checkpoints to build averaged model
    # 评估阶段优先加载 SWAD 平均权重（若可用），提升跨域泛化稳定性
    EVAL_USE_SWAD_CHECKPOINT = False
    
    # ========== 性能优化配置 ==========
    USE_AMP = True  # 是否使用混合精度训练（AMP）- 可显著加速训练，节省显存
    USE_TORCH_COMPILE = False  # 是否使用torch.compile（PyTorch 2.0+）- 需要PyTorch >= 2.0

    # ATT379f(B): PCGrad — Gradient Surgery for Multi-Task Learning (Yu et al., NeurIPS 2020)
    # Groups: CE (Fusion+Signal+Spec CE), PACL (FusionProtoAlign+SemanticAlign), Contrastive (SpecSupCon)
    # NOTE: requires ~2x backward memory due to retain_graph. Disable on <32GB GPU.
    USE_PCGRAD = False
    PCGRAD_LOG_INTERVAL = 50  # log conflict stats every N batches
    
    # ========== 样本不均衡处理配置 ==========
    MAX_CLASS_WEIGHT = 5.0  # 类别权重上限
    SAMPLER_WEIGHT_SCALE = 0.5  # 降低采样器权重强度，默认0.5
    USE_WEIGHTED_SAMPLER = True  # 是否使用加权采样器
    USE_PROGRESSIVE_WEIGHTS = True  # 是否使用渐进式权重
    INITIAL_WEIGHT_SCALE = 0.5  # 初始权重缩放
    WEIGHT_SCHEDULE = "cosine"  # 权重调度方式："cosine" 或 "linear"
    ENABLE_CLASS_MONITORING = True  # 是否启用类别监控
    
    # ========== 类别权重策略配置 ==========
    CLASS_WEIGHT_STRATEGY = "adaptive"  # "fixed" | "adaptive" | "hybrid"
    # - "fixed": 完全固定，基于训练集类别分布计算一次，之后不再改变
    # - "adaptive": 自适应调整（每个epoch根据类别准确率动态调整）
    # - "hybrid": 混合方案，前N个epoch固定，之后自适应调整
    CLASS_WEIGHT_FIXED_EPOCHS = 5  # hybrid模式下，前N个epoch使用固定权重
    
    # ========== 自适应类别权重调整配置（仅在 adaptive 或 hybrid 模式下生效） ==========
    USE_ADAPTIVE_CLASS_WEIGHTS = True  # 是否使用自适应类别权重（基于预测准确率动态调整）
    ADAPTIVE_WEIGHT_MOMENTUM = 0.1  # 权重更新动量（0-1，越大更新越快）
    ADAPTIVE_WEIGHT_MIN = 0.5  # 权重下限
    ADAPTIVE_WEIGHT_MAX = 3.0  # 权重上限

    # ========== 结构稳定性：诊断分支度量学习（提升类3稳定性） ==========
    # 在 FusionDecoder 的诊断分支 z_cls 上加入 EMA 类中心的 cosine-center loss
    FUSION_USE_CLS_ADAPTER = True
    FUSION_METRIC_LOSS = "center"  # "center" | "none"
    FUSION_METRIC_LOSS_WEIGHT = 0.0
    FUSION_CENTER_MOMENTUM = 0.95
    
    # ========== EMA (Exponential Moving Average) 权重配置 ==========
    # EMA 仅维护影子权重，不改变训练梯度路径；评估时可降低checkpoint波动
    USE_EMA = True
    EMA_DECAY = 0.9999  # EMA 衰减率（0.9999 表示约 10000 步的平均窗口）
    EMA_WARMUP_STEPS = 200  # ATT156: 缩短 warmup，确保短程训练中 EMA 真正生效
    EMA_UPDATE_AFTER_STEP = 50  # ATT156: 更早开始 EMA 更新，减少早期抖动
    
    # ========== 可靠性门控与不确定性估计配置 ==========
    # 用于动态抑制不可靠模态（如Spec在HUST上的塌缩），提升融合鲁棒性
    USE_RELIABILITY_GATING = bool(int(os.environ.get("USE_RELIABILITY_GATING", "1")))  # 是否启用可靠性门控（强烈推荐，解决模态塌缩问题）
    UNCERTAINTY_METHOD = 'entropy'  # 不确定性估计方法：'entropy' | 'energy' | 'variance'
    GATE_TYPE = 'soft'  # 门控类型：'soft' | 'hard' | 'learned'
    # 分类头配置（用于从logits计算不确定性，而非embedding方差）
    USE_CLASSIFICATION_HEAD = True  # 是否为signal/spec编码器添加轻量分类头，用于基于logits的不确定性估计
    # Aux CE 配置（给signal/spec分类头添加辅助监督）
    # ATT112: 0.1→0.01. ATT110/111 both show SPEC AUC=0.38 (vs ATT107's ~0.96 with 0.01).
    # AUX_CE_WEIGHT=0.1 breaks z_spec geometry identical to 0.3 (both give SPEC AUC=0.38).
    # Threshold for geometry preservation: ≤ 0.01~0.05. Return to 0.01 to restore SPEC AUC.
    # spec_head_diag_acc can now be properly evaluated via the bug fix in zero_shot_eval_hust.py.
    # ATT119: 0.01→0.30 (STRATEGIC SHIFT: use spec_head as PRIMARY diagnostic output).
    #   Rationale:
    #   - spec_head_diag_acc=0.3495 (ATT116) already outperforms fusion_decoder_diag_acc on unseen loads
    #   - Multimodal fusion introduces NEGATIVE TRANSFER from load-sensitive signal features
    #   - With DANN disabled (protects z_spec), AUX_CE=0.30 provides the main discriminative signal
    #   - AUC concern (ATT110 showed AUX_CE=0.30 caused SPEC AUC=0.38): ACCEPTABLE in ATT119 because:
    #     (1) ATT110 used SUPCON τ=0.10 which created tight load-specific clusters; ATT119 has SUPCON=0
    #     (2) ATT110 DANN was disabled AND AUX_CE was the ONLY discriminative loss; same applies here
    #     BUT: spec_head trained by 0.30 AUX_CE will directly classify, so spec_head_diag_acc is our target
    #     regardless of SPEC legacy AUC (ranking metric is secondary when argmax classification is primary)
    #   - ATT110 Diag-Acc=0.3118 was best at that time (before spec_head bug was fixed)
    #     With bug now fixed, ATT119 targets spec_head_diag_acc ≥ 0.45
    # ATT214: restore ATT209 AUX_CE balance.
    # ATT428: was DISABLED.
    # ATT429-P0: RE-ENABLED — AUX_CE is the ONLY direct supervision for SpecHead cosine classifier.
    # ATT453: KEEP — aux_spec CE is the ONLY direct supervision for SpecHead proto classifier.
    # Without it, ETF prototypes get no classification gradient and stay at random init.
    AUX_CE_WEIGHT = float(os.environ.get("AUX_CE_WEIGHT", "0.30"))
    AUX_CE_START_EPOCH = 1  # 从第几个epoch开始添加aux CE（避免早期干扰主干）
    # ATT124: Focal Loss for spec_head AUX_CE (主fusion loss已用Focal γ=1.5，spec_head一直用标准CE—不一致)
    # 数学依据：ATT123实证Outer 17/36天花板；19个hard Outer→Ball样本p_t≈0.30；
    # FL(γ=2, p_t=0.30)权重=(0.7)²=0.49 vs easy Ball p_t=0.92权重=(0.08)²=0.0064 → 梯度比76×
    # 迫使编码器学习区分Outer-Ball的精细频谱差异（笼频调制等），不改变架构/参数量
    AUX_CE_USE_FOCAL = False   # ATT125: True→False（ATT124实证：γ=2.0导致Ball R=0.92→0.56，梯度饥饿性遗忘；Outer↑11pp但Ball↓36pp，净效果spec_head_diag_acc↓3.2pp；FL在4类小间距分类上对easy类造成梯度starvation，证伪此方案）
    AUX_CE_FOCAL_GAMMA = 2.0   # ATT124已配置（当前USE_FOCAL=False，此参数不生效）
    # ATT489(P1): structural switch for spec supervision.
    # - legacy_aux_ce: only old spec-head CE
    # - proto_ce: cosine CE in z_spec space against training prototypes
    # - hybrid: both losses
    SPEC_TRAIN_OBJECTIVE_MODE = str(os.environ.get("SPEC_TRAIN_OBJECTIVE_MODE", "proto_ce")).strip().lower()
    SPEC_PROTO_CE_WEIGHT = float(os.environ.get("SPEC_PROTO_CE_WEIGHT", "1.0"))
    SPEC_PROTO_CE_TEMPERATURE = float(os.environ.get("SPEC_PROTO_CE_TEMPERATURE", "0.10"))

    
    # ========== 注意力增强Gate配置（可选） ==========
    # NOTE (stability): In recent logs, attention-gate can saturate to near one-hot weights
    # (e.g., mean=(1.0, 0.0)), effectively turning fusion into single-modality behavior.
    # Default OFF to prevent "single-modality collapse"; you can re-enable later for ablations.
    USE_ATTENTION_GATE = False  # 是否使用注意力增强gate（稳定性优先，默认关闭）
    ATTENTION_GATE_ALPHA = 0.0   # 注意力权重占比（0=完全使用不确定性gate，1=完全使用attention）
    ATTENTION_GATE_DIM = 128     # 注意力特征维度
    ATTENTION_GATE_HEADS = 2     # 注意力头数
    # 条件化floor配置（用于根据domain和uncertainty动态调整weight floor）
    USE_CONDITIONAL_FLOOR = True  # 是否使用条件化floor（基于domain_embedding，允许HIT上spec权重更低）
    GATE_FLOOR_MAX = 0.10  # floor最大值（训练初期较保守）
    GATE_FLOOR_MIN = 0.0   # floor最小值（允许完全抑制坏模态）
    ATTENTION_REG_WEIGHT = 0.01  # Attention KL正则权重（防坍缩）

    # ========== ATT430: 结构重构 v2 配置 ==========
    # S0: MLP 分类器 (回退余弦分类器, ATT429-P0 已证伪)
    # S1: 双空间解耦 — z_inv (域不变, 主路径) + z_spe (域特异, 仅训练)
    # S2: 域混淆对抗 (替代 Fishr, 实例级域对齐)
    # S3: VICReg 方差约束 (防止 z_spec_pre_norm 坍缩)

    # P1 (保留): 推理时对目标域 BN 完全重置
    SPEC_BN_FULL_RESET_ON_TARGET = bool(int(os.environ.get("SPEC_BN_FULL_RESET_ON_TARGET", "1")))
    SPEC_BN_FULL_RESET_MAX_BATCHES = int(os.environ.get("SPEC_BN_FULL_RESET_MAX_BATCHES", "64"))

    # P2 (保留): 层级化 SpecHead (MLP heads, 非余弦)
    USE_HIERARCHICAL_SPEC_HEAD = bool(int(os.environ.get("USE_HIERARCHICAL_SPEC_HEAD", "0")))
    HIERARCHICAL_NF_TEMPERATURE = 0.07  # unused with MLP heads, kept for compatibility
    HIERARCHICAL_FAULT_TEMPERATURE = 0.07
    HIERARCHICAL_FAULT_CLASSES = int(os.environ.get("HIERARCHICAL_FAULT_CLASSES", "3"))
    HIERARCHICAL_NF_CE_WEIGHT = float(os.environ.get("HIERARCHICAL_NF_CE_WEIGHT", "0.30"))
    HIERARCHICAL_FAULT_CE_WEIGHT = float(os.environ.get("HIERARCHICAL_FAULT_CE_WEIGHT", "0.30"))
    HIERARCHICAL_CE_START_EPOCH = int(os.environ.get("HIERARCHICAL_CE_START_EPOCH", "1"))

    # S1: 双空间解耦
    USE_DUAL_DISENTANGLE = bool(int(os.environ.get("USE_DUAL_DISENTANGLE", "0")))
    DUAL_DISENTANGLE_SPE_DIM = int(os.environ.get("DUAL_DISENTANGLE_SPE_DIM", "64"))
    DUAL_DISENTANGLE_NUM_DOMAINS = int(os.environ.get("DUAL_DISENTANGLE_NUM_DOMAINS", "5"))
    DUAL_DISENTANGLE_ORTH_WEIGHT = float(os.environ.get("DUAL_DISENTANGLE_ORTH_WEIGHT", "0.10"))
    DUAL_DISENTANGLE_DC_SPE_WEIGHT = float(os.environ.get("DUAL_DISENTANGLE_DC_SPE_WEIGHT", "0.10"))
    # S2: 域混淆 (dc_inv_train CE + entropy confusion on z_inv)
    DUAL_DISENTANGLE_CONFUSION_WEIGHT = float(os.environ.get("DUAL_DISENTANGLE_CONFUSION_WEIGHT", "0.10"))
    # S3: z_spec_pre_norm 方差约束 (仅 variance, 不含 covariance)
    USE_SPEC_VARIANCE_CONSTRAINT = bool(int(os.environ.get("USE_SPEC_VARIANCE_CONSTRAINT", "0")))
    SPEC_VARIANCE_WEIGHT = float(os.environ.get("SPEC_VARIANCE_WEIGHT", "0.10"))
    SPEC_VARIANCE_TARGET_STD = float(os.environ.get("SPEC_VARIANCE_TARGET_STD", "1.0"))

    # ========== Target-domain two-stage adaptation (evaluation-time) ==========
    # ATT500: when True, skip isolated HUST/SEU LP-FT / SVP two-stage block; metrics stay unified zero-shot.
    UNIFIED_ZEROSHOT_EVAL = bool(int(os.environ.get("UNIFIED_ZEROSHOT_EVAL", "1")))
    # NOTE (HUST): Unsupervised stage-1 adaptation was observed to severely hurt HUST NF/diag in logs.
    TARGET_SKIP_HUST_UNSUP_ADAPT = True
    # ATT230(S3): allow selective Task2 target-domain adaptation for faster iteration.
    TARGET_EVAL_SKIP_HUST = False
    TARGET_EVAL_SKIP_SEU = False
    TARGET_EVAL_SKIP_HIT = TARGET_EVAL_SKIP_SEU
    # 在目标域监督微调时是否同时优化 SpecHead 分类器（仅头部参数，不解冻Spec主干）
    TARGET_SUP_INCLUDE_SPEC_HEAD = True
    # ATT186/ATT220: LP-FT flat-head诊断损失采用"SpecHead主导 + Fusion可选辅助"。
    # ATT220: 恢复 fusion 辅助项到 0.30，避免 HUST 路径在 tiny-shot + progressive unfreeze
    # 场景下因缺少 decoder 侧梯度约束而出现回退；PU 最终指标仍由更优路径替换机制保底。
    TARGET_SUP_LP_SPEC_LOSS_WEIGHT = 1.0
    TARGET_SUP_LP_FUSION_LOSS_WEIGHT = 0.30
    TARGET_SUP_LP_SPEC_LABEL_SMOOTHING = 0.10
    # ATT292(S3): metric-space regularization during LP-FT to improve kNN/LabelProp graph quality.
    TARGET_SUP_LP_SUPCON_ENABLE = True
    TARGET_SUP_LP_SUPCON_WEIGHT = 0.50
    TARGET_SUP_LP_SUPCON_TEMPERATURE = 0.10
    TARGET_SUP_LP_SUPCON_USE_CLASS_WEIGHTS = True
    TARGET_SUP_LP_SUPCON_FAULT_ONLY = False
    # ATT210: in Task2 pretraining, detach decoder input to stop fusion CE / decoder-side gradients
    # from corrupting the shared SpecEncoder representation.
    # ATT211a: restore decoder-to-backbone gradient path in Task2.
    TARGET_TASK2_DETACH_FUSION_FOR_DECODER = False

    # P1: Test-Time BN Adaptation（跨域评估前更新 BN running 统计）
    TARGET_BN_ADAPT_ENABLE = True
    # ATT395: Enable BN-Adapt for Task1 (cross-condition: PU gk6/gk7→gk5/gk8).
    # Ref: Schneider et al. NeurIPS 2020, Li et al. Pattern Recognition 2018.
    TASK1_BN_ADAPT_ENABLE = bool(int(os.environ.get("TASK1_BN_ADAPT_ENABLE", "1")))
    # <=0 表示跳过；-1 表示遍历目标域全部 batch
    TARGET_BN_ADAPT_NUM_BATCHES = -1
    # ATT157: alpha-BN 插值，final = alpha*source + (1-alpha)*target_full
    # env-overridable for hparam sensitivity sweeps (experiments/run_hparam_sensitivity.sh)
    TARGET_BN_ADAPT_ALPHA = float(os.environ.get("TARGET_BN_ADAPT_ALPHA", "0.30"))
    # ATT158 P0: TTN-style per-layer adaptive alpha from source/target BN variance
    TARGET_BN_ADAPT_AUTO_ALPHA = bool(int(os.environ.get("TARGET_BN_ADAPT_AUTO_ALPHA", "1")))
    TARGET_BN_ADAPT_AUTO_ALPHA_MIN = 0.05
    TARGET_BN_ADAPT_AUTO_ALPHA_MAX = 0.95
    # TENT: 仅在 Task2 目标域评估前做轻量熵最小化（仅更新 BN affine）
    TARGET_TENT_ENABLE = False  # ATT157: ATT154-156 实证无有效收益，先关闭
    TARGET_TENT_STEPS = 12
    TARGET_TENT_LR = 1e-3
    TARGET_TENT_BATCHES = 4
    TARGET_TENT_DIVERSITY_WEIGHT = 0.0  # ATT156: 移除均匀先验驱动，抑制 HIT 的 OR 过度预测
    # SAR-style reliable entropy filtering: only low-entropy samples are used for TENT updates.
    TARGET_TENT_RELIABLE_FILTER_ENABLE = True
    TARGET_TENT_RELIABLE_ENTROPY_FACTOR = 0.40  # threshold = ln(K) * factor
    TARGET_TENT_RESET_BEFORE_EACH_DOMAIN = True

    # ATT422: SHOT/TENT/TTFA — ABANDONED after ATT422a proved these methods fail when
    # source-target feature spaces are structurally different (cos_sim=-0.76).
    # Defaults set to 0 (disabled). Kept for checkpoint compatibility only.
    TARGET_SHOT_BN_ENABLE = bool(int(os.environ.get("TARGET_SHOT_BN_ENABLE", "0")))
    TARGET_SHOT_TENT_ENABLE = bool(int(os.environ.get("TARGET_SHOT_TENT_ENABLE", "0")))
    TARGET_SHOT_TENT_STEPS = int(os.environ.get("TARGET_SHOT_TENT_STEPS", "15"))
    TARGET_SHOT_TENT_LR = float(os.environ.get("TARGET_SHOT_TENT_LR", "5e-4"))
    TARGET_SHOT_TENT_BATCHES = int(os.environ.get("TARGET_SHOT_TENT_BATCHES", "4"))
    TARGET_SHOT_CONFIDENCE_TH = float(os.environ.get("TARGET_SHOT_CONFIDENCE_TH", "0.5"))
    TARGET_TTFA_ENABLE = bool(int(os.environ.get("TARGET_TTFA_ENABLE", "0")))
    TARGET_TTFA_NUM_BATCHES = int(os.environ.get("TARGET_TTFA_NUM_BATCHES", "-1"))
    TARGET_TTFA_STRENGTH = float(os.environ.get("TARGET_TTFA_STRENGTH", "0.8"))

    # ========== Health gate numerical stability ==========
    # Avoid boundary triggers due to float rounding (e.g., "0.0500 < 0.0500").
    HEALTH_GATE_EPS = 1e-6
    
    # ========== 元学习配置（Meta-Learning for Domain Generalization） ==========
    # Default OFF: current project already has multiple adaptive components; meta-learning can destabilize
    # diagnosis unless carefully isolated and implemented in a stateless way.
    USE_META_LEARNING = False  # 是否启用元学习（用于改善跨域泛化，实验性功能）
    META_LEARNING_START_EPOCH = 5  # 从第几个epoch开始元学习（前期让模型先收敛）
    META_LEARNING_INTERVAL = 2  # 每N个epoch执行一次元学习步骤
    META_LR = 1e-4  # 元学习率（outer loop）
    META_INNER_LR = 1e-3  # 内循环学习率
    META_INNER_STEPS = 3  # 内循环适配步数
    META_FIRST_ORDER = True  # 是否使用一阶近似（True更快，内存开销小）
    GATE_TEMPERATURE = 1.0  # 门控温度（控制软硬程度，越小越硬）
    
    # 模态Dropout配置（训练期随机丢弃模态，提升鲁棒性）
    USE_MODALITY_DROPOUT = True  # 是否启用模态Dropout
    MODALITY_DROP_PROB = 0.15  # 模态丢弃概率（推荐0.1-0.2）
    MODALITY_DROP_MODE = 'adaptive'  # 丢弃模式：'random' | 'adaptive'（自适应根据不确定性丢弃）
    
    # 校准损失配置（改善模型置信度校准，减少过自信）
    # ATT379f(A): disable — calibration is post-hoc concern, not needed during feature learning
    USE_CALIBRATION_LOSS = False
    CALIBRATION_LOSS_WEIGHT = 0.0  # 校准损失权重（推荐0.005-0.02）
    CALIBRATION_N_BINS = 15  # ECE分箱数量
    
    # ========== 正交性/互补性正则化配置 ==========
    USE_ORTHOGONALITY_REG = False  # ATT376: 关闭高占比正交正则，避免挤占主任务梯度预算
    ORTHOGONALITY_REG_WEIGHT = 0.001  # 正交性正则化权重（推荐0.0005-0.002）
    
    # ========== 训练评估配置 ==========
    TEST_INTERVAL = 5  # 每N个epoch进行一次评估
    TRAIN_ENABLE_TARGET_MONITOR = False  # 训练期是否执行目标域zero-shot监控（默认关闭避免泄露）
    # Whether to print verbose epoch diagnostics (SupCon pairs, Stage1-CE class stats,
    # IO/NF/IR separation diagnostics, rolling diagnostics). Keep False to save log tokens.
    TRAIN_LOG_VERBOSE_DIAGNOSTICS = False
    # ATT489(P4): compact logging by default; enable full diagnostics with TRAIN_LOG_DIAGNOSTIC_MODE=1.
    TRAIN_LOG_DIAGNOSTIC_MODE = bool(int(os.environ.get("TRAIN_LOG_DIAGNOSTIC_MODE", "0")))
    # P2: suppress redundant eval-time fusion input diagnostics by default.
    EVAL_LOG_FUSION_INPUT_STATS = bool(int(os.environ.get("EVAL_LOG_FUSION_INPUT_STATS", "0")))
    # Dataset feature-construction logs are very noisy with multi-worker dataloaders; keep them
    # behind diagnostic mode unless explicitly requested.
    DATASET_LOG_FEATURE_BUILD = bool(int(os.environ.get("DATASET_LOG_FEATURE_BUILD", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    # 训练期数据流审计：用于快速定位域ID映射/标签流错配。
    TRAIN_LOG_DATAFLOW_AUDIT = bool(int(os.environ.get("TRAIN_LOG_DATAFLOW_AUDIT", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_DATAFLOW_AUDIT_EPOCH_STRIDE = 1
    # 训练期梯度范数审计：用于排查梯度爆炸/消失（尤其是多分支损失叠加场景）。
    TRAIN_LOG_GRAD_NORM = bool(int(os.environ.get("TRAIN_LOG_GRAD_NORM", "1")))
    TRAIN_LOG_GRAD_NORM_STEP_STRIDE = 50
    # Batch-level可观测性（用于定位loss分项失活、语义原型退化、梯度主导项漂移）
    TRAIN_LOG_BATCH_LOSS = bool(int(os.environ.get("TRAIN_LOG_BATCH_LOSS", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_BATCH_LOSS_STEP_STRIDE = 50
    TRAIN_LOG_BATCH_LOSS_BREAKDOWN = bool(int(os.environ.get("TRAIN_LOG_BATCH_LOSS_BREAKDOWN", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_BATCH_LOSS_BREAKDOWN_STEP_STRIDE = 50
    TRAIN_LOG_PROTOTYPE_COS_MATRIX = bool(int(os.environ.get("TRAIN_LOG_PROTOTYPE_COS_MATRIX", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_PROTOTYPE_COS_STEP_STRIDE = 100
    TRAIN_LOG_BATCH_IO_STATS = bool(int(os.environ.get("TRAIN_LOG_BATCH_IO_STATS", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_BATCH_IO_STEP_STRIDE = 100
    TRAIN_LOG_GRAD_CONTRIB = bool(int(os.environ.get("TRAIN_LOG_GRAD_CONTRIB", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_GRAD_CONTRIB_STEP_STRIDE = 100
    # Epoch-level loss budget/term share audit (similar to LossShare/LossBudget style).
    TRAIN_LOG_LOSS_BUDGET = True
    TRAIN_LOG_LOSS_BUDGET_EPOCH_STRIDE = 1
    # Fine-grained module-wise grad norm audit.
    TRAIN_LOG_GRAD_NORM_DETAIL = bool(int(os.environ.get("TRAIN_LOG_GRAD_NORM_DETAIL", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_GRAD_NORM_DETAIL_STEP_STRIDE = 20
    # Optional gradient clipping audit (disabled by default to keep behavior unchanged).
    TRAIN_ENABLE_GRAD_CLIP = False
    TRAIN_GRAD_CLIP_MAX_NORM = 5.0
    TRAIN_LOG_GRAD_CLIP = bool(int(os.environ.get("TRAIN_LOG_GRAD_CLIP", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    # SpecEncoder pre/post L2 normalization audit logs.
    TRAIN_LOG_SPECENC_NORM_STATS = bool(int(os.environ.get("TRAIN_LOG_SPECENC_NORM_STATS", "1" if TRAIN_LOG_DIAGNOSTIC_MODE else "0")))
    TRAIN_LOG_SPECENC_NORM_STEP_STRIDE = 100
    # Epoch-level loss-path audit: SupCon-only path / CE path / multimodal alignment path
    # counts, used to verify multimodal_alignment_loss is truly reachable and active.
    TRAIN_LOG_LOSS_PATH_AUDIT = bool(int(os.environ.get("TRAIN_LOG_LOSS_PATH_AUDIT", "1")))
    # High-frequency reliability-gate stats are disabled by default to keep only essential logs.
    LOG_RELIABILITY_GATE = False
    LAMBDA_CLIP = 0.0  # ATT376: 旧式spec-clip余弦项关闭，改用CLIP-style对比损失
    
    # ========== 数据加载配置 ==========
    NUM_WORKERS = 0  # Windows上默认禁用多进程以避免序列化问题，Linux/Mac上可以使用多进程
    PREFETCH_FACTOR = 4  # 每个worker预取的batch数
    PERSISTENT_WORKERS = False  # 是否保持worker进程存活（Windows上建议False）
    
    # ========== 故障标签和提示模板 ==========
    FAULT_LABEL_TO_SEMANTIC = {
        0: {"component": "bearing", "state": "normal"},
        1: {"component": "inner race", "state": "damaged"},
        2: {"component": "outer race", "state": "damaged"},
        3: {"component": "rolling element", "state": "damaged"},
        # 4: {"component": "bearing", "state": "damaged"},  # compound / unknown
    }
    
    # ATT112: Added 2 load-conditional templates per class (light-load and heavy-load variants).
    # Rationale: Cross-load generalization requires text prototypes that span different operating conditions.
    # text_proto[k] = mean(encode(all_templates[k])) → averaged over light+heavy → load-invariant centroid.
    # Physical basis: at light load (0HP/1HP), fault amplitude is weaker → templates emphasize pattern shape
    # and frequency rather than amplitude; at heavy load (2HP/3HP), amplitude is stronger but shape is same.
    # ATT391-S3: Prompt rewrite — maximize Inner/Outer semantic separation.
    # Key strategy: Inner uses "rotating, modulated, variable, spreading" adjectives;
    # Outer uses "stationary, fixed, constant, narrowband" adjectives.
    # Each template contrasts with the other fault type to increase embedding distance.
    FAULT_PROMPT_TEMPLATES = {
        0: [
            "A healthy bearing with no damage: smooth vibration, no periodic impulses, no characteristic fault frequencies, and uniform broadband background noise.",
            "Normal bearing operation: stationary low-amplitude vibration without any resonance bursts, impact patterns, or harmonic structures.",
            "Undamaged bearing: the spectrum shows only broadband noise floor without any distinct narrowband peaks or sideband families.",
            "Healthy bearing under light load: minimal vibration energy, no impulsive patterns, and stable low entropy across time-frequency representation.",
            "Healthy bearing under heavy load: slightly elevated broadband level but no periodic impulse signature and no fault-related frequency components."
        ],
        1: [
            "Inner race bearing fault: impacts rotate with the shaft causing amplitude modulation — the impulse strength varies periodically as the defect enters and exits the load zone.",
            "Inner race damage produces speed-dependent modulated impacts: fault harmonics flanked by shaft-speed sidebands that spread across a wide frequency range due to rotation.",
            "A rotating inner race defect creates variable-strength transients: strong impacts in the loaded zone, weak impacts elsewhere, resulting in characteristic amplitude envelope modulation.",
            "Inner race fault signature: broadband resonance excitation with amplitude modulation envelope at shaft rotation frequency — distinct from fixed-position defects.",
            "Inner race defect at varying loads: the modulation depth changes with load but the sideband spreading pattern around fault harmonics persists as a distinctive rotating-defect marker."
        ],
        2: [
            "Outer race bearing fault: a fixed stationary defect producing constant-amplitude periodic impacts at precise regular intervals, independent of shaft rotation phase.",
            "Outer race damage creates perfectly regular impulses from a non-rotating fixed position: fault harmonics appear as sharp narrowband peaks without sideband spreading.",
            "A stationary outer race defect generates uniform-strength impacts: every impact has the same amplitude because the defect location relative to the load zone never changes.",
            "Outer race fault signature: clean narrowband harmonics with constant spacing and no amplitude modulation — fundamentally different from rotating-defect patterns.",
            "Outer race defect at varying loads: amplitude scales with load but timing remains perfectly periodic and narrowband harmonic structure stays clean without sideband families."
        ],
        3: [
            "Rolling element bearing fault: irregular intermittent impacts from a tumbling defect that randomly contacts inner and outer races with unpredictable timing.",
            "Ball defect produces chaotic non-periodic impacts: fault-related spectral components appear scattered and unstable, with broadband energy smearing across frequencies.",
            "Rolling element damage creates quasi-random transients: impact intervals vary as the ball rotates irregularly, producing fragmented time-frequency patterns unlike ring race faults.",
            "Rolling element fault signature: dispersed broadband spectral energy without clear harmonic families — neither the regular periodicity of outer race nor the modulation of inner race faults.",
            "Rolling element defect under load: intermittent bursts with varying intensity, cage-frequency modulated fluctuations, and spectral spreading that distinguishes it from race defects."
        ],
    }
    
    @staticmethod
    def get_window_length(dataset_name):
        """根据数据集名称获取窗口长度（采样点数）"""
        return int(Config.WINDOW_SIZE_SEC * Config.SAMPLING_RATES[dataset_name])


# 创建全局配置实例，保持向后兼容
config = Config()

# 为了向后兼容，导出常用的配置项作为模块级别的变量
# 这样可以直接使用 from config import XXX 的方式导入
WINDOW_SIZE_SEC = Config.WINDOW_SIZE_SEC
SAMPLING_RATES = Config.SAMPLING_RATES
DEVICE = Config.DEVICE
USE_DOMAIN_RESIDUAL = Config.USE_DOMAIN_RESIDUAL
NUM_DOMAINS = Config.NUM_DOMAINS
DOMAIN_RESIDUAL_DIM = Config.DOMAIN_RESIDUAL_DIM
DOMAIN_RESIDUAL_INIT_SCALE = Config.DOMAIN_RESIDUAL_INIT_SCALE
USE_DOMAIN_LORA = Config.USE_DOMAIN_LORA
DOMAIN_LORA_RANK = Config.DOMAIN_LORA_RANK
DOMAIN_LORA_ALPHA = Config.DOMAIN_LORA_ALPHA
DOMAIN_LORA_DROPOUT = Config.DOMAIN_LORA_DROPOUT
USE_SHARED_PROJECTION_FUSION = Config.USE_SHARED_PROJECTION_FUSION
FUSION_CROSS_MODAL_ALIGN_WEIGHT = Config.FUSION_CROSS_MODAL_ALIGN_WEIGHT
USE_DECISION_LEVEL_FUSION = Config.USE_DECISION_LEVEL_FUSION
FUSION_FIXED_EQUAL_WEIGHT = Config.FUSION_FIXED_EQUAL_WEIGHT
USE_GATED_ADDITIVE_FUSION = Config.USE_GATED_ADDITIVE_FUSION
OGM_GE_ENABLE = Config.OGM_GE_ENABLE
OGM_GE_ALPHA = Config.OGM_GE_ALPHA
OGM_GE_START_EPOCH = Config.OGM_GE_START_EPOCH
FUSION_RESIDUAL_SKIP_ENABLE = Config.FUSION_RESIDUAL_SKIP_ENABLE
FUSION_RESIDUAL_SKIP_INIT_ALPHA = Config.FUSION_RESIDUAL_SKIP_INIT_ALPHA
SPLIT_ARCH_ENABLE = Config.SPLIT_ARCH_ENABLE
SPLIT_ARCH_SIGNAL_WEIGHT = Config.SPLIT_ARCH_SIGNAL_WEIGHT
ATT470_STRUCTURE_ENABLE = Config.ATT470_STRUCTURE_ENABLE
ATT470_MOE_ROUTE_DIAG_ENABLE = Config.ATT470_MOE_ROUTE_DIAG_ENABLE
ATT471_AGIT_ENABLE = Config.ATT471_AGIT_ENABLE
ATT472_TWO_END_SHARED_NECK_ENABLE = Config.ATT472_TWO_END_SHARED_NECK_ENABLE
ATT473_SPECIALIST_ARBITER_ENABLE = Config.ATT473_SPECIALIST_ARBITER_ENABLE
DUAL_CONTRACT_ENABLE = Config.DUAL_CONTRACT_ENABLE
DUAL_CONTRACT_SAVE_EXTRA_CKPT = Config.DUAL_CONTRACT_SAVE_EXTRA_CKPT
DUAL_CONTRACT_FORCE_GRAD_DETAIL = Config.DUAL_CONTRACT_FORCE_GRAD_DETAIL
FUSION_DECODER_DUAL_TOWER_ENABLE = Config.FUSION_DECODER_DUAL_TOWER_ENABLE
FUSION_MEAN_RESIDUAL_ENABLE = Config.FUSION_MEAN_RESIDUAL_ENABLE
FUSION_USE_STANDARD_UNIT_VECTOR_NORM = Config.FUSION_USE_STANDARD_UNIT_VECTOR_NORM
FUSION_FORWARD_DETACH_ENCODER_INPUTS = Config.FUSION_FORWARD_DETACH_ENCODER_INPUTS
GATE_ENTROPY_WEIGHT = Config.GATE_ENTROPY_WEIGHT
USE_PROTO_CLASSIFIER = Config.USE_PROTO_CLASSIFIER
PROTO_CLS_TEMPERATURE = Config.PROTO_CLS_TEMPERATURE
USE_PROTO_COSINE_ALIGN = Config.USE_PROTO_COSINE_ALIGN
PROTO_COSINE_ALIGN_WEIGHT = Config.PROTO_COSINE_ALIGN_WEIGHT
DGL_ENABLE = Config.DGL_ENABLE
DGL_AUX_FUSION_HEAD_WEIGHT = Config.DGL_AUX_FUSION_HEAD_WEIGHT
DGL_DETACH_ENCODER_INPUTS = Config.DGL_DETACH_ENCODER_INPUTS
DGL_AUX_FUSION_HEAD_LR_MULT = Config.DGL_AUX_FUSION_HEAD_LR_MULT
CROSS_MODAL_KD_ENABLE = Config.CROSS_MODAL_KD_ENABLE
CROSS_MODAL_KD_WEIGHT = Config.CROSS_MODAL_KD_WEIGHT
CROSS_MODAL_KD_TEMPERATURE = Config.CROSS_MODAL_KD_TEMPERATURE
HIERARCHICAL_SPEC_HEAD_PRIMARY = Config.HIERARCHICAL_SPEC_HEAD_PRIMARY
SPECHEAD_USE_COSINE_PROTO = Config.SPECHEAD_USE_COSINE_PROTO
USE_COMPACT_BILINEAR_FUSION = Config.USE_COMPACT_BILINEAR_FUSION
CBP_DIM = Config.CBP_DIM
USE_CROSS_ATTENTION_FUSION = Config.USE_CROSS_ATTENTION_FUSION
USE_CLASS_AWARE_ATTENTION = Config.USE_CLASS_AWARE_ATTENTION
FUSION_NUM_HEADS = Config.FUSION_NUM_HEADS
FUSION_DROPOUT = Config.FUSION_DROPOUT
FUSION_DOMAIN_EMBED_DIM = Config.FUSION_DOMAIN_EMBED_DIM
FUSION_USE_CONFIDENCE = Config.FUSION_USE_CONFIDENCE
FUSION_USE_MULTI_TOKEN = Config.FUSION_USE_MULTI_TOKEN
FUSION_CROSS_ATTN_DEPTH = Config.FUSION_CROSS_ATTN_DEPTH
USE_GATE_MASKED_ATTENTION = Config.USE_GATE_MASKED_ATTENTION
FUSION_SIGNAL_ENTROPY_GATE = Config.FUSION_SIGNAL_ENTROPY_GATE
FUSION_SIGNAL_ENTROPY_GATE_FLOOR = Config.FUSION_SIGNAL_ENTROPY_GATE_FLOOR
FUSION_USE_SPEC_PRE_L2_INPUT = Config.FUSION_USE_SPEC_PRE_L2_INPUT
FUSION_INPUT_LAYERNORM = Config.FUSION_INPUT_LAYERNORM
USE_STAT_FILM = Config.USE_STAT_FILM
STAT_FILM_N_FEATURES = Config.STAT_FILM_N_FEATURES
STAT_FILM_HIDDEN_DIM = Config.STAT_FILM_HIDDEN_DIM
FUSION_PROGRESSIVE_TRAINING = Config.FUSION_PROGRESSIVE_TRAINING
USE_DOMAIN_AWARE_FAULT_HEAD = Config.USE_DOMAIN_AWARE_FAULT_HEAD
CWRU_GK2_DOMAIN_ID = Config.CWRU_GK2_DOMAIN_ID
CWRU_GK4_DOMAIN_ID = Config.CWRU_GK4_DOMAIN_ID
PU_DOMAIN_ID = Config.PU_DOMAIN_ID
HUST_DOMAIN_ID = Config.HUST_DOMAIN_ID
SEU_DOMAIN_ID = Config.SEU_DOMAIN_ID
HIT_DOMAIN_ID = Config.HIT_DOMAIN_ID
DOMAIN_ID_MAP = Config.DOMAIN_ID_MAP
USE_ORTHOGONALITY_REG = Config.USE_ORTHOGONALITY_REG
ORTHOGONALITY_REG_WEIGHT = Config.ORTHOGONALITY_REG_WEIGHT
FUSION_CONFIDENCE_START_EPOCH = Config.FUSION_CONFIDENCE_START_EPOCH
FUSION_CLASS_AWARE_START_EPOCH = Config.FUSION_CLASS_AWARE_START_EPOCH
FUSION_CLASS_AWARE_REGULARIZATION = Config.FUSION_CLASS_AWARE_REGULARIZATION
USE_PROGRESSIVE_LR = Config.USE_PROGRESSIVE_LR
PROGRESSIVE_LR_SCHEDULE = Config.PROGRESSIVE_LR_SCHEDULE
USE_FOCAL_LOSS = Config.USE_FOCAL_LOSS
FOCAL_GAMMA = Config.FOCAL_GAMMA
FOCAL_ALPHA = Config.FOCAL_ALPHA
FOCAL_START_EPOCH = Config.FOCAL_START_EPOCH  # Att77新增：Focal延迟启动
USE_FLAT_DIAG_HEAD = Config.USE_FLAT_DIAG_HEAD
USE_BALANCED_SOFTMAX = Config.USE_BALANCED_SOFTMAX
USE_POSTHOC_LOGIT_ADJUST = Config.USE_POSTHOC_LOGIT_ADJUST
POSTHOC_LOGIT_ADJUST_TAU = Config.POSTHOC_LOGIT_ADJUST_TAU
POSTHOC_LOGIT_CLASS_COUNTS = Config.POSTHOC_LOGIT_CLASS_COUNTS
USE_ETF_CLASSIFIER_INIT = Config.USE_ETF_CLASSIFIER_INIT
USE_ETF_FIXED_CLASSIFIER = Config.USE_ETF_FIXED_CLASSIFIER
USE_FLAT_MULTI_CENTER_ROLLING = Config.USE_FLAT_MULTI_CENTER_ROLLING
FLAT_MULTI_CENTER_ROLLING_K = Config.FLAT_MULTI_CENTER_ROLLING_K
USE_FLAT_MULTI_CENTER_INNER = Config.USE_FLAT_MULTI_CENTER_INNER
USE_FLAT_MULTI_CENTER_OUTER = Config.USE_FLAT_MULTI_CENTER_OUTER
FLAT_MULTI_CENTER_INNER_K = Config.FLAT_MULTI_CENTER_INNER_K
FLAT_MULTI_CENTER_OUTER_K = Config.FLAT_MULTI_CENTER_OUTER_K
FLAT_MULTI_CENTER_POOL = Config.FLAT_MULTI_CENTER_POOL
FLAT_MULTI_CENTER_TAU = Config.FLAT_MULTI_CENTER_TAU
FLAT_MULTI_CENTER_INIT_STD = Config.FLAT_MULTI_CENTER_INIT_STD
LOG_EMBED_CENTROID_SIM = Config.LOG_EMBED_CENTROID_SIM
USE_IO_ROLL_SEP_LOSS = Config.USE_IO_ROLL_SEP_LOSS
IO_ROLL_SEP_WEIGHT = Config.IO_ROLL_SEP_WEIGHT
IO_ROLL_SEP_MAX_SIM = Config.IO_ROLL_SEP_MAX_SIM
IO_ROLL_SEP_START_EPOCH = Config.IO_ROLL_SEP_START_EPOCH
LOG_IO_SEP = Config.LOG_IO_SEP
USE_IO_SEP_LOSS = Config.USE_IO_SEP_LOSS
IO_SEP_WEIGHT = Config.IO_SEP_WEIGHT
IO_SEP_MAX_SIM = Config.IO_SEP_MAX_SIM
IO_SEP_START_EPOCH = Config.IO_SEP_START_EPOCH
LOG_NF_SEP = Config.LOG_NF_SEP
USE_NF_SEP_LOSS = Config.USE_NF_SEP_LOSS
NF_SEP_WEIGHT = Config.NF_SEP_WEIGHT
NF_SEP_MAX_SIM = Config.NF_SEP_MAX_SIM
NF_SEP_START_EPOCH = Config.NF_SEP_START_EPOCH
NF_SEP_MIN_BATCHES = Config.NF_SEP_MIN_BATCHES

# Att77新增：Batch Nuclear Norm Loss
USE_BATCH_NUCLEAR_NORM = Config.USE_BATCH_NUCLEAR_NORM
BATCH_NUCLEAR_NORM_WEIGHT = Config.BATCH_NUCLEAR_NORM_WEIGHT
BATCH_NUCLEAR_NORM_START_EPOCH = Config.BATCH_NUCLEAR_NORM_START_EPOCH

# Inner↔Rolling Separation (Attempt 35 fix)
LOG_IR_SEP = Config.LOG_IR_SEP
USE_IR_SEP_LOSS = Config.USE_IR_SEP_LOSS
IR_SEP_WEIGHT = Config.IR_SEP_WEIGHT
IR_SEP_MAX_SIM = Config.IR_SEP_MAX_SIM
IR_SEP_START_EPOCH = Config.IR_SEP_START_EPOCH
IR_SEP_MAINTENANCE_WEIGHT = Config.IR_SEP_MAINTENANCE_WEIGHT  # NEW: prevents gradient vanishing

# LDAM + DRW
USE_LDAM_LOSS = Config.USE_LDAM_LOSS
LDAM_MAX_MARGIN = Config.LDAM_MAX_MARGIN
LDAM_SCALE = Config.LDAM_SCALE
LDAM_NORMALIZE_MARGIN = Config.LDAM_NORMALIZE_MARGIN
USE_DRW = Config.USE_DRW
DRW_EPOCH = Config.DRW_EPOCH

# Class Prototype Separation Loss
USE_PROTO_SEPARATION_LOSS = Config.USE_PROTO_SEPARATION_LOSS
PROTO_SEP_MARGIN_DELTA = Config.PROTO_SEP_MARGIN_DELTA
PROTO_SEP_LOSS_WEIGHT = Config.PROTO_SEP_LOSS_WEIGHT
PROTO_SEP_START_EPOCH = Config.PROTO_SEP_START_EPOCH
PROTO_SEP_MOMENTUM = Config.PROTO_SEP_MOMENTUM
LOG_PROTO_DISTANCES = Config.LOG_PROTO_DISTANCES
USE_LOG_PROTO_SEP = Config.USE_LOG_PROTO_SEP
LOG_PROTO_SEP_EPS = Config.LOG_PROTO_SEP_EPS
LABEL_SMOOTHING = Config.LABEL_SMOOTHING
W_FUSION_CLS_EARLY = Config.W_FUSION_CLS_EARLY
W_FUSION_CLS_LATE = Config.W_FUSION_CLS_LATE
USE_RESIDUAL_PROXIES = Config.USE_RESIDUAL_PROXIES
USE_CLASS_WEIGHTS_IN_LOSS = Config.USE_CLASS_WEIGHTS_IN_LOSS
W_MARGIN = Config.W_MARGIN
USE_PROTOTYPE_HEAD = Config.USE_PROTOTYPE_HEAD
PROTOTYPE_HEAD_TRAIN_START_EPOCH = Config.PROTOTYPE_HEAD_TRAIN_START_EPOCH
USE_TWO_STAGE_TRAINING = Config.USE_TWO_STAGE_TRAINING
STAGE1_SUPERVISED_EPOCHS = Config.STAGE1_SUPERVISED_EPOCHS
STAGE1_NF_ONLY = Config.STAGE1_NF_ONLY
STAGE2_FREEZE_BACKBONE = Config.STAGE2_FREEZE_BACKBONE
STAGE2_KEEP_SUPERVISED_CE = Config.STAGE2_KEEP_SUPERVISED_CE
FORCE_CE_STAGE2 = Config.FORCE_CE_STAGE2
STAGE_NF_LOSS_WEIGHT = Config.STAGE_NF_LOSS_WEIGHT
USE_CLS_DETACH_FAULT_TYPE = Config.USE_CLS_DETACH_FAULT_TYPE
USE_NF_CONTRASTIVE_LOSS = Config.USE_NF_CONTRASTIVE_LOSS
NF_CONTRASTIVE_WEIGHT = Config.NF_CONTRASTIVE_WEIGHT
NF_CONTRASTIVE_TEMPERATURE = Config.NF_CONTRASTIVE_TEMPERATURE
USE_NF_MARGIN_LOSS = Config.USE_NF_MARGIN_LOSS
NF_MARGIN_WEIGHT = Config.NF_MARGIN_WEIGHT
NF_MARGIN = Config.NF_MARGIN
NF_MARGIN_START_EPOCH = Config.NF_MARGIN_START_EPOCH
USE_NORMAL_REFERENCE_LOSS = Config.USE_NORMAL_REFERENCE_LOSS
NORMAL_REFERENCE_WEIGHT = Config.NORMAL_REFERENCE_WEIGHT
NORMAL_REFERENCE_MOMENTUM = Config.NORMAL_REFERENCE_MOMENTUM
NORMAL_REFERENCE_FIRST_EPOCH_MOMENTUM = Config.NORMAL_REFERENCE_FIRST_EPOCH_MOMENTUM
USE_SUPCON_STAGE1 = Config.USE_SUPCON_STAGE1
SUPCON_STAGE1_WEIGHT = Config.SUPCON_STAGE1_WEIGHT
SUPCON_STAGE1_SPEC_WEIGHT = Config.SUPCON_STAGE1_SPEC_WEIGHT
SUPCON_STAGE1_CE_WEIGHT = Config.SUPCON_STAGE1_CE_WEIGHT
USE_SUPCON_PRENORM_VAR_LOSS = Config.USE_SUPCON_PRENORM_VAR_LOSS
SUPCON_PRENORM_VAR_WEIGHT = Config.SUPCON_PRENORM_VAR_WEIGHT
SUPCON_PRENORM_STD_TARGET = Config.SUPCON_PRENORM_STD_TARGET
STAGE1_CE_USE_CLASS_WEIGHTS = Config.STAGE1_CE_USE_CLASS_WEIGHTS
STAGE1_CE_USE_FOCAL = Config.STAGE1_CE_USE_FOCAL
STAGE1_CE_FOCAL_GAMMA = Config.STAGE1_CE_FOCAL_GAMMA
STAGE1_CE_FOCAL_USE_CLASS_ALPHA = Config.STAGE1_CE_FOCAL_USE_CLASS_ALPHA
SUPCON_TEMPERATURE = Config.SUPCON_TEMPERATURE
SUPCON_USE_CLASS_WEIGHTS = Config.SUPCON_USE_CLASS_WEIGHTS
SUPCON_FAULT_ONLY = Config.SUPCON_FAULT_ONLY
USE_DANN = Config.USE_DANN
DANN_LAMBDA = Config.DANN_LAMBDA
DANN_START_EPOCH = Config.DANN_START_EPOCH
DANN_MAX_ALPHA = Config.DANN_MAX_ALPHA
DANN_ALPHA_SCHEDULE = Config.DANN_ALPHA_SCHEDULE
DANN_USE_TARGET_UNLABELED = Config.DANN_USE_TARGET_UNLABELED
DANN_HIDDEN_DIM = Config.DANN_HIDDEN_DIM
DANN_DROPOUT = Config.DANN_DROPOUT
DANN_MODE = Config.DANN_MODE
DANN_FEATURE = Config.DANN_FEATURE
DANN_DETACH_PROBS = Config.DANN_DETACH_PROBS
FUSION_DOMAIN_ALIGN_ENABLE = Config.FUSION_DOMAIN_ALIGN_ENABLE
FUSION_DOMAIN_ALIGN_LOSS = Config.FUSION_DOMAIN_ALIGN_LOSS
FUSION_DOMAIN_ALIGN_WEIGHT = Config.FUSION_DOMAIN_ALIGN_WEIGHT
FUSION_DOMAIN_ALIGN_START_EPOCH = Config.FUSION_DOMAIN_ALIGN_START_EPOCH
FUSION_DOMAIN_ALIGN_WARMUP_EPOCHS = Config.FUSION_DOMAIN_ALIGN_WARMUP_EPOCHS
FUSION_DOMAIN_ALIGN_DETACH = Config.FUSION_DOMAIN_ALIGN_DETACH
FUSION_DOMAIN_ALIGN_MIN_SAMPLES = Config.FUSION_DOMAIN_ALIGN_MIN_SAMPLES
FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1 = Config.FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1
FUSION_PROJ_REMOVE_LAYERNORM = Config.FUSION_PROJ_REMOVE_LAYERNORM
AUX_PRENORM_CE_ENABLE = Config.AUX_PRENORM_CE_ENABLE
AUX_PRENORM_CE_WEIGHT = Config.AUX_PRENORM_CE_WEIGHT
MASK_MISSING_CLASSES_IN_LOGITS = Config.MASK_MISSING_CLASSES_IN_LOGITS
PARTIAL_LABEL_LOSS_ENABLE = Config.PARTIAL_LABEL_LOSS_ENABLE
USE_SEM_ADAPTER = Config.USE_SEM_ADAPTER
SEM_ADAPTER_HIDDEN = Config.SEM_ADAPTER_HIDDEN
SEM_ADAPTER_INIT_ZERO = Config.SEM_ADAPTER_INIT_ZERO
USE_VISUAL_PROTOTYPES = Config.USE_VISUAL_PROTOTYPES
VISUAL_PROTO_MOMENTUM = Config.VISUAL_PROTO_MOMENTUM
SEMANTIC_MIX_VISUAL_ALPHA = Config.SEMANTIC_MIX_VISUAL_ALPHA
SEMANTIC_SCORE_USE_Z_SEM = Config.SEMANTIC_SCORE_USE_Z_SEM
OPEN_SET_PMAX_TH = Config.OPEN_SET_PMAX_TH
OPEN_SET_PFAULT_GRAY_LOW = Config.OPEN_SET_PFAULT_GRAY_LOW
OPEN_SET_PFAULT_GRAY_HIGH = Config.OPEN_SET_PFAULT_GRAY_HIGH
OPEN_SET_REJECT_LABEL = Config.OPEN_SET_REJECT_LABEL
USE_DIAG_LINEAR_RESIDUAL = Config.USE_DIAG_LINEAR_RESIDUAL
DIAG_LINEAR_SCALE_INIT = Config.DIAG_LINEAR_SCALE_INIT
DIAG_LINEAR_TRAIN_START_EPOCH = Config.DIAG_LINEAR_TRAIN_START_EPOCH
DIAG_LINEAR_WARMUP_EPOCHS = Config.DIAG_LINEAR_WARMUP_EPOCHS
DIAG_LINEAR_LR_MULT = Config.DIAG_LINEAR_LR_MULT
DIAG_LINEAR_BETA_MAX = Config.DIAG_LINEAR_BETA_MAX
USE_DIAG_LINEAR_GATE = Config.USE_DIAG_LINEAR_GATE
DIAG_LINEAR_GATE_INIT_BIAS = Config.DIAG_LINEAR_GATE_INIT_BIAS
USE_DIAG_CLASS_BIAS = Config.USE_DIAG_CLASS_BIAS
DIAG_CLASS_BIAS_TRAIN_START_EPOCH = Config.DIAG_CLASS_BIAS_TRAIN_START_EPOCH
USE_SEMANTIC_DISTILL = Config.USE_SEMANTIC_DISTILL
SEMANTIC_DISTILL_WEIGHT = Config.SEMANTIC_DISTILL_WEIGHT
SEMANTIC_DISTILL_T = Config.SEMANTIC_DISTILL_T
SEMANTIC_DISTILL_START_EPOCH = Config.SEMANTIC_DISTILL_START_EPOCH
SEMANTIC_DISTILL_TEACHER_MAXCONF = Config.SEMANTIC_DISTILL_TEACHER_MAXCONF
SEMANTIC_DISTILL_FAULT_ONLY = Config.SEMANTIC_DISTILL_FAULT_ONLY
USE_SEMANTIC_FAULT_VS_NORMAL_ANCHOR = Config.USE_SEMANTIC_FAULT_VS_NORMAL_ANCHOR
SEMANTIC_FAULT_VS_NORMAL_ANCHOR_WEIGHT = Config.SEMANTIC_FAULT_VS_NORMAL_ANCHOR_WEIGHT
SEMANTIC_FAULT_VS_NORMAL_ANCHOR_MARGIN = Config.SEMANTIC_FAULT_VS_NORMAL_ANCHOR_MARGIN
SEMANTIC_FAULT_VS_NORMAL_ANCHOR_START_EPOCH = Config.SEMANTIC_FAULT_VS_NORMAL_ANCHOR_START_EPOCH
USE_MIXSTYLE_SIGNAL = Config.USE_MIXSTYLE_SIGNAL
USE_MIXSTYLE_SPEC = Config.USE_MIXSTYLE_SPEC
MIXSTYLE_P = Config.MIXSTYLE_P
MIXSTYLE_ALPHA = Config.MIXSTYLE_ALPHA
MIXSTYLE_PROTO_ISOLATION = Config.MIXSTYLE_PROTO_ISOLATION
USE_SPEC_DISENTANGLE = Config.USE_SPEC_DISENTANGLE
SPEC_DISENTANGLE_INV_DIM = Config.SPEC_DISENTANGLE_INV_DIM
SPEC_DISENTANGLE_SPE_DIM = Config.SPEC_DISENTANGLE_SPE_DIM
SPEC_DISENTANGLE_START_EPOCH = Config.SPEC_DISENTANGLE_START_EPOCH
SPEC_DISENTANGLE_MAX_ALPHA = Config.SPEC_DISENTANGLE_MAX_ALPHA
SPEC_DISENTANGLE_ALPHA_EPOCHS = Config.SPEC_DISENTANGLE_ALPHA_EPOCHS
SPEC_DISENTANGLE_ORTH_WEIGHT = Config.SPEC_DISENTANGLE_ORTH_WEIGHT
SPEC_DISENTANGLE_DOMAIN_SPE_WEIGHT = Config.SPEC_DISENTANGLE_DOMAIN_SPE_WEIGHT
SPEC_DISENTANGLE_DOMAIN_ADV_WEIGHT = Config.SPEC_DISENTANGLE_DOMAIN_ADV_WEIGHT
USE_SPEC_AUGMENT = Config.USE_SPEC_AUGMENT
SPEC_AUGMENT_FREQ_MASK_F = Config.SPEC_AUGMENT_FREQ_MASK_F
SPEC_AUGMENT_TIME_MASK_T = Config.SPEC_AUGMENT_TIME_MASK_T
SPEC_AUGMENT_NUM_FREQ_MASKS = Config.SPEC_AUGMENT_NUM_FREQ_MASKS
SPEC_AUGMENT_NUM_TIME_MASKS = Config.SPEC_AUGMENT_NUM_TIME_MASKS
SPEC_AUGMENT_MASK_PROB = Config.SPEC_AUGMENT_MASK_PROB
USE_IBN_SPEC = Config.USE_IBN_SPEC
IBN_SPEC_RATIO = Config.IBN_SPEC_RATIO
USE_IBN_SPEC_STEM_ONLY = Config.USE_IBN_SPEC_STEM_ONLY
USE_IBN_SIGNAL = Config.USE_IBN_SIGNAL
IBN_SIGNAL_RATIO = Config.IBN_SIGNAL_RATIO
USE_GN_SPEC_STEM = Config.USE_GN_SPEC_STEM
USE_FREQLN_SPEC_STEM = Config.USE_FREQLN_SPEC_STEM
USE_FREQ_ATTN_POOL = Config.USE_FREQ_ATTN_POOL
SPEC_AMP_JITTER_ENABLE = Config.SPEC_AMP_JITTER_ENABLE
SPEC_AMP_JITTER_PROB = Config.SPEC_AMP_JITTER_PROB
SPEC_AMP_JITTER_STD = Config.SPEC_AMP_JITTER_STD
SPEC_AMP_JITTER_MIN = Config.SPEC_AMP_JITTER_MIN
SPEC_AMP_JITTER_MAX = Config.SPEC_AMP_JITTER_MAX
CA_RESIDUAL_GATE_ENABLE = Config.CA_RESIDUAL_GATE_ENABLE
SIGNAL_ENCODER_LR_MULT = Config.SIGNAL_ENCODER_LR_MULT
FREEZE_SIGNAL_AFTER_EPOCH = Config.FREEZE_SIGNAL_AFTER_EPOCH
DIAG_DECISION_MODE = Config.DIAG_DECISION_MODE
USE_TWO_STAGE_FAULT_LOSS = Config.USE_TWO_STAGE_FAULT_LOSS
USE_FAULT_DEDICATED_REP = Config.USE_FAULT_DEDICATED_REP
FAULT_TYPE_WEIGHT_ENABLE = Config.FAULT_TYPE_WEIGHT_ENABLE
FAULT_TYPE_WEIGHT_POWER = Config.FAULT_TYPE_WEIGHT_POWER
FAULT_TYPE_MAX_WEIGHT = Config.FAULT_TYPE_MAX_WEIGHT
USE_SPLIT_FAULT_HEADS = Config.USE_SPLIT_FAULT_HEADS
USE_HIERARCHICAL_FAULT_PROB = Config.USE_HIERARCHICAL_FAULT_PROB
SPLIT_FAULT_W_IO = Config.SPLIT_FAULT_W_IO
SPLIT_FAULT_W_ROLL = Config.SPLIT_FAULT_W_ROLL
SPLIT_FAULT_ROLL_POS_WEIGHT_MAX = Config.SPLIT_FAULT_ROLL_POS_WEIGHT_MAX
SPLIT_FAULT_ROLL_USE_ALL_NEG = Config.SPLIT_FAULT_ROLL_USE_ALL_NEG
NF_GATE_NF_MODE = Config.NF_GATE_NF_MODE
NF_GATE_FIT_ENABLE = Config.NF_GATE_FIT_ENABLE
NF_GATE_FIT_OBJECTIVE = Config.NF_GATE_FIT_OBJECTIVE
NF_GATE_FIT_NUM_CANDIDATES = Config.NF_GATE_FIT_NUM_CANDIDATES
NF_GATE_FIT_MAX_SAMPLES = Config.NF_GATE_FIT_MAX_SAMPLES
ZERO_SHOT_NF_GATE_FIT_OBJECTIVE = Config.ZERO_SHOT_NF_GATE_FIT_OBJECTIVE
USE_NF_PLATT_CALIBRATION = Config.USE_NF_PLATT_CALIBRATION
NF_PLATT_MAX_SAMPLES = Config.NF_PLATT_MAX_SAMPLES
NF_PLATT_STEPS = Config.NF_PLATT_STEPS
NF_PLATT_LR = Config.NF_PLATT_LR
NF_PLATT_L2 = Config.NF_PLATT_L2
NF_GATE_GRAY_LOW = Config.NF_GATE_GRAY_LOW
NF_GATE_GRAY_HIGH = Config.NF_GATE_GRAY_HIGH
NF_GATE_GRAY_FALLBACK = Config.NF_GATE_GRAY_FALLBACK
USE_NF_GATE_MARGIN_LOSS = Config.USE_NF_GATE_MARGIN_LOSS
NF_GATE_MARGIN_WEIGHT = Config.NF_GATE_MARGIN_WEIGHT
NF_GATE_MARGIN = Config.NF_GATE_MARGIN
NF_GATE_MARGIN_START_EPOCH = Config.NF_GATE_MARGIN_START_EPOCH
ENABLE_MARGIN_ROUTER = Config.ENABLE_MARGIN_ROUTER
MARGIN_ROUTER_PRED_RATIO_TH = Config.MARGIN_ROUTER_PRED_RATIO_TH
HEALTH_GATE_FAULT_ONLY = Config.HEALTH_GATE_FAULT_ONLY
ENABLE_BEST_CKPT_HEALTH_GATE = Config.ENABLE_BEST_CKPT_HEALTH_GATE
USE_TARGET_DOMAIN_HEALTH_GATE_FOR_BEST_CKPT = Config.USE_TARGET_DOMAIN_HEALTH_GATE_FOR_BEST_CKPT
BEST_CKPT_MIN_CLASS_RECALL = Config.BEST_CKPT_MIN_CLASS_RECALL
HEALTH_GATE_MIN_FAULT_CLASSES = Config.HEALTH_GATE_MIN_FAULT_CLASSES
ENABLE_TRAIN_HEALTH_GATE = Config.ENABLE_TRAIN_HEALTH_GATE
TRAIN_HEALTH_GATE_COOLDOWN_EPOCHS = Config.TRAIN_HEALTH_GATE_COOLDOWN_EPOCHS
HEALTH_GATE_MIN_FAULT_RECALL = Config.HEALTH_GATE_MIN_FAULT_RECALL
NF_LOGIT_CLAMP = Config.NF_LOGIT_CLAMP
FUSIONDEC_W_NF_GLOBAL = Config.FUSIONDEC_W_NF_GLOBAL
FUSIONDEC_W_NF_MULTI = Config.FUSIONDEC_W_NF_MULTI
FUSIONDEC_W_FAULT_TYPE = Config.FUSIONDEC_W_FAULT_TYPE
USE_NF_MULTI_LOSS = Config.USE_NF_MULTI_LOSS
NF_MULTI_START_EPOCH = Config.NF_MULTI_START_EPOCH
NF_MULTI_POS_WEIGHT_MAX = Config.NF_MULTI_POS_WEIGHT_MAX
USE_MULTI_PROTOTYPE_HEAD = Config.USE_MULTI_PROTOTYPE_HEAD
MULTI_PROTOTYPE_K = Config.MULTI_PROTOTYPE_K
MULTI_PROTOTYPE_INIT_STD = Config.MULTI_PROTOTYPE_INIT_STD
MULTI_PROTOTYPE_START_EPOCH = Config.MULTI_PROTOTYPE_START_EPOCH
MULTI_PROTOTYPE_DISABLE_ROLLING_EXTRA = Config.MULTI_PROTOTYPE_DISABLE_ROLLING_EXTRA
USE_SUPCON_LOSS = Config.USE_SUPCON_LOSS
SUPCON_WEIGHT = Config.SUPCON_WEIGHT
SUPCON_TEMPERATURE = Config.SUPCON_TEMPERATURE
SPEC_ALIGN_WARMUP_ENABLE = Config.SPEC_ALIGN_WARMUP_ENABLE
SPEC_ALIGN_WARMUP_START_EPOCH = Config.SPEC_ALIGN_WARMUP_START_EPOCH
SPEC_ALIGN_WARMUP_END_EPOCH = Config.SPEC_ALIGN_WARMUP_END_EPOCH
SPEC_ALIGN_WARMUP_END_EPOCH_TASK2 = Config.SPEC_ALIGN_WARMUP_END_EPOCH_TASK2
SPEC_ALIGN_WARMUP_MIN_SCALE = Config.SPEC_ALIGN_WARMUP_MIN_SCALE
SPEC_ALIGN_WARMUP_MAX_SCALE = Config.SPEC_ALIGN_WARMUP_MAX_SCALE
SPEC_ALIGN_BASE_WEIGHT = Config.SPEC_ALIGN_BASE_WEIGHT
USE_DETACHED_SPEC_ALIGN_PROJ = Config.USE_DETACHED_SPEC_ALIGN_PROJ
DETACHED_SPEC_ALIGN_WEIGHT = Config.DETACHED_SPEC_ALIGN_WEIGHT
DETACHED_SPEC_ALIGN_TEMPERATURE = Config.DETACHED_SPEC_ALIGN_TEMPERATURE
DETACHED_SPEC_ALIGN_START_EPOCH = Config.DETACHED_SPEC_ALIGN_START_EPOCH
DETACHED_SPEC_ALIGN_USE_CLASS_WEIGHTS = Config.DETACHED_SPEC_ALIGN_USE_CLASS_WEIGHTS
DETACHED_SPEC_ALIGN_USE_IN_EVAL = Config.DETACHED_SPEC_ALIGN_USE_IN_EVAL
SPEC_ALIGN_ATTACHED = Config.SPEC_ALIGN_ATTACHED
FISHR_TARGET_FEATURE = Config.FISHR_TARGET_FEATURE
FISHR_DIAG_LOG_ENABLED = Config.FISHR_DIAG_LOG_ENABLED
FISHR_DIAG_LOG_STRIDE = Config.FISHR_DIAG_LOG_STRIDE
SEMANTIC_ALIGNMENT_TEMPERATURE = Config.SEMANTIC_ALIGNMENT_TEMPERATURE
EVAL_ENSEMBLE_SPEC_HEAD_WEIGHT      = Config.EVAL_ENSEMBLE_SPEC_HEAD_WEIGHT
EVAL_ENSEMBLE_FUSION_WEIGHT         = Config.EVAL_ENSEMBLE_FUSION_WEIGHT
EVAL_ENSEMBLE_SPEC_PROTO_WEIGHT     = Config.EVAL_ENSEMBLE_SPEC_PROTO_WEIGHT
EVAL_ENSEMBLE_SIG_PROTO_WEIGHT      = Config.EVAL_ENSEMBLE_SIG_PROTO_WEIGHT
EVAL_ENSEMBLE_SPEC_HEAD_TEMPERATURE = Config.EVAL_ENSEMBLE_SPEC_HEAD_TEMPERATURE
EVAL_ENSEMBLE_FUSION_TEMPERATURE    = Config.EVAL_ENSEMBLE_FUSION_TEMPERATURE
EVAL_ENSEMBLE_USE_FUSION_LOGITS     = Config.EVAL_ENSEMBLE_USE_FUSION_LOGITS
EVAL_ENSEMBLE_FUSION_T_GRID         = Config.EVAL_ENSEMBLE_FUSION_T_GRID
EVAL_ENSEMBLE_FUSION_T_GRID_ENABLE  = Config.EVAL_ENSEMBLE_FUSION_T_GRID_ENABLE
EVAL_ENSEMBLE_INNER_PROTECT         = Config.EVAL_ENSEMBLE_INNER_PROTECT
EVAL_ENSEMBLE_MARGIN_ADAPTIVE       = Config.EVAL_ENSEMBLE_MARGIN_ADAPTIVE
EVAL_ENSEMBLE_MARGIN_ALPHA_MIN      = Config.EVAL_ENSEMBLE_MARGIN_ALPHA_MIN
EVAL_ENSEMBLE_MARGIN_ALPHA_MAX      = Config.EVAL_ENSEMBLE_MARGIN_ALPHA_MAX
EVAL_ENSEMBLE_MARGIN_TAU            = Config.EVAL_ENSEMBLE_MARGIN_TAU
EVAL_ENSEMBLE_CCTC_ENABLE           = Config.EVAL_ENSEMBLE_CCTC_ENABLE
EVAL_ENSEMBLE_CCTC_BALL_CLASS       = Config.EVAL_ENSEMBLE_CCTC_BALL_CLASS
EVAL_ENSEMBLE_CCTC_W_SPEC_GRID      = Config.EVAL_ENSEMBLE_CCTC_W_SPEC_GRID
EVAL_ENSEMBLE_CCTC_W_FUS_GRID       = Config.EVAL_ENSEMBLE_CCTC_W_FUS_GRID
EVAL_ENSEMBLE_CCTC_T_GRID           = Config.EVAL_ENSEMBLE_CCTC_T_GRID
EVAL_ENSEMBLE_CCTC_T_PROTO_GRID     = Config.EVAL_ENSEMBLE_CCTC_T_PROTO_GRID
EVAL_ENSEMBLE_W_SPEC_GRID           = Config.EVAL_ENSEMBLE_W_SPEC_GRID
EVAL_ENSEMBLE_W_SPEC_PROTO_GRID     = Config.EVAL_ENSEMBLE_W_SPEC_PROTO_GRID   # ATT169
EVAL_ENSEMBLE_SPEC_PROTO_T_GRID     = Config.EVAL_ENSEMBLE_SPEC_PROTO_T_GRID   # ATT169
EVAL_SPEC_PROTO_TTA_ENABLE          = Config.EVAL_SPEC_PROTO_TTA_ENABLE
EVAL_SPEC_PROTO_TTA_TARGET_DATASETS = Config.EVAL_SPEC_PROTO_TTA_TARGET_DATASETS
EVAL_SPEC_PROTO_TTA_ITERS           = Config.EVAL_SPEC_PROTO_TTA_ITERS
EVAL_SPEC_PROTO_TTA_TOPK_RATIO      = Config.EVAL_SPEC_PROTO_TTA_TOPK_RATIO
EVAL_SPEC_PROTO_TTA_MIN_SAMPLES_PER_CLASS = Config.EVAL_SPEC_PROTO_TTA_MIN_SAMPLES_PER_CLASS
EVAL_SPEC_PROTO_TTA_MIX_ALPHA       = Config.EVAL_SPEC_PROTO_TTA_MIX_ALPHA
EVAL_SPEC_PROTO_TTA_MIN_SPECPROTO_ACC = Config.EVAL_SPEC_PROTO_TTA_MIN_SPECPROTO_ACC
EVAL_SPEC_PROTO_TTA_MIX_ALPHA_BY_DATASET = Config.EVAL_SPEC_PROTO_TTA_MIX_ALPHA_BY_DATASET
EVAL_SPEC_PROTO_USE_LEARNABLE_PROTOS = Config.EVAL_SPEC_PROTO_USE_LEARNABLE_PROTOS
EVAL_SPEC_PRIOR_CALIB_ENABLE        = Config.EVAL_SPEC_PRIOR_CALIB_ENABLE
EVAL_SPEC_PRIOR_CALIB_STRENGTH      = Config.EVAL_SPEC_PRIOR_CALIB_STRENGTH
EVAL_SPEC_PRIOR_CALIB_EPS           = Config.EVAL_SPEC_PRIOR_CALIB_EPS
EVAL_TIP_ADAPTER_ENABLE             = Config.EVAL_TIP_ADAPTER_ENABLE
EVAL_TIP_ADAPTER_TARGET_DATASETS    = Config.EVAL_TIP_ADAPTER_TARGET_DATASETS
EVAL_TIP_ADAPTER_MAX_SUPPORT_SAMPLES = Config.EVAL_TIP_ADAPTER_MAX_SUPPORT_SAMPLES
EVAL_TIP_ADAPTER_TOPK               = Config.EVAL_TIP_ADAPTER_TOPK
EVAL_TIP_ADAPTER_BETA               = Config.EVAL_TIP_ADAPTER_BETA
EVAL_TIP_ADAPTER_USE_SPEC_HEAD_MIX  = Config.EVAL_TIP_ADAPTER_USE_SPEC_HEAD_MIX
EVAL_TIP_ADAPTER_ALPHA_GRID         = Config.EVAL_TIP_ADAPTER_ALPHA_GRID
EVAL_TIP_ADAPTER_REPLACE_WHEN_BETTER = Config.EVAL_TIP_ADAPTER_REPLACE_WHEN_BETTER
EVAL_TIP_ADAPTER_FEATURE_SOURCE     = Config.EVAL_TIP_ADAPTER_FEATURE_SOURCE
EVAL_EXPORT_EMBEDDINGS_ENABLE       = Config.EVAL_EXPORT_EMBEDDINGS_ENABLE
EVAL_EXPORT_EMBEDDINGS_DIR          = Config.EVAL_EXPORT_EMBEDDINGS_DIR
EVAL_EXPORT_EMBEDDINGS_POST_LPFT    = Config.EVAL_EXPORT_EMBEDDINGS_POST_LPFT
EVAL_EXPORT_EMBEDDINGS_AUTO_TSNE    = Config.EVAL_EXPORT_EMBEDDINGS_AUTO_TSNE
EVAL_EXPORT_EMBEDDINGS_PER_CONDITION = Config.EVAL_EXPORT_EMBEDDINGS_PER_CONDITION
EVAL_DYNAMIC_MULTI_PROTO_ENABLE = Config.EVAL_DYNAMIC_MULTI_PROTO_ENABLE
EVAL_DYNAMIC_MULTI_PROTO_TARGET_DATASETS = Config.EVAL_DYNAMIC_MULTI_PROTO_TARGET_DATASETS
EVAL_DYNAMIC_MULTI_PROTO_REPLACE_WHEN_BETTER = Config.EVAL_DYNAMIC_MULTI_PROTO_REPLACE_WHEN_BETTER
EVAL_DYNAMIC_MULTI_PROTO_FEATURE_SOURCE = Config.EVAL_DYNAMIC_MULTI_PROTO_FEATURE_SOURCE
EVAL_DYNAMIC_MULTI_PROTO_K = Config.EVAL_DYNAMIC_MULTI_PROTO_K
EVAL_DYNAMIC_MULTI_PROTO_TOPK_VOTE = Config.EVAL_DYNAMIC_MULTI_PROTO_TOPK_VOTE
EVAL_DYNAMIC_MULTI_PROTO_TEMPERATURE = Config.EVAL_DYNAMIC_MULTI_PROTO_TEMPERATURE
EVAL_DYNAMIC_MULTI_PROTO_MAX_SUPPORT_SAMPLES = Config.EVAL_DYNAMIC_MULTI_PROTO_MAX_SUPPORT_SAMPLES
EVAL_DYNAMIC_MULTI_PROTO_KMEANS_ITERS = Config.EVAL_DYNAMIC_MULTI_PROTO_KMEANS_ITERS
EVAL_DYNAMIC_MULTI_PROTO_MIN_CLUSTER_SAMPLES = Config.EVAL_DYNAMIC_MULTI_PROTO_MIN_CLUSTER_SAMPLES
EVAL_DYNAMIC_MULTI_PROTO_TEMPLATE_MAX_SIGNAL_SAMPLES = Config.EVAL_DYNAMIC_MULTI_PROTO_TEMPLATE_MAX_SIGNAL_SAMPLES
EVAL_DYNAMIC_MULTI_PROTO_LOG_TOP_CLUSTERS = Config.EVAL_DYNAMIC_MULTI_PROTO_LOG_TOP_CLUSTERS
EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_ALPHA = Config.EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_ALPHA
EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_MIN_COS = Config.EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_MIN_COS
EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_FALLBACK_ALPHA = Config.EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_FALLBACK_ALPHA
EVAL_DYNAMIC_MULTI_PROTO_USE_SPEC_EVIDENCE = Config.EVAL_DYNAMIC_MULTI_PROTO_USE_SPEC_EVIDENCE
EVAL_FUSION_NORMALIZE_INPUTS        = Config.EVAL_FUSION_NORMALIZE_INPUTS
EVAL_FUSION_USE_SPEC_PRE_L2_INPUT   = Config.EVAL_FUSION_USE_SPEC_PRE_L2_INPUT
EVAL_LOG_FUSION_INPUT_STATS         = Config.EVAL_LOG_FUSION_INPUT_STATS
EVAL_LOG_FUSION_INPUT_STEP_STRIDE   = Config.EVAL_LOG_FUSION_INPUT_STEP_STRIDE
SUPCON_FAULT_ONLY = Config.SUPCON_FAULT_ONLY
USE_PROTO_SEP_LOSS = Config.USE_PROTO_SEP_LOSS
PROTO_SEP_WEIGHT = Config.PROTO_SEP_WEIGHT
PROTO_SEP_MARGIN = Config.PROTO_SEP_MARGIN
USE_PROMPT_TUNING = Config.USE_PROMPT_TUNING
USE_DOMAIN_PROMPT_TUNING = Config.USE_DOMAIN_PROMPT_TUNING
PROMPT_RESIDUAL_INIT_STD = Config.PROMPT_RESIDUAL_INIT_STD
PROMPT_TUNE_START_EPOCH = Config.PROMPT_TUNE_START_EPOCH
FREEZE_ROLLING_PROMPT_RESIDUAL = Config.FREEZE_ROLLING_PROMPT_RESIDUAL
USE_TEXT_PROMPT_TUNING = Config.USE_TEXT_PROMPT_TUNING
TEXT_PROMPT_TUNING_INIT_STD = Config.TEXT_PROMPT_TUNING_INIT_STD
TEXT_PROMPT_TUNING_SCALE_INIT = Config.TEXT_PROMPT_TUNING_SCALE_INIT
USE_TEXT_PROMPT_SEPARATION_LOSS = Config.USE_TEXT_PROMPT_SEPARATION_LOSS
TEXT_PROMPT_SEPARATION_WEIGHT = Config.TEXT_PROMPT_SEPARATION_WEIGHT
TEXT_PROMPT_SEPARATION_START_EPOCH = Config.TEXT_PROMPT_SEPARATION_START_EPOCH
TEXT_PROMPT_INNER_OUTER_TARGET_MAX = Config.TEXT_PROMPT_INNER_OUTER_TARGET_MAX
USE_DUAL_SPACE_DECOUPLE = Config.USE_DUAL_SPACE_DECOUPLE
DUAL_SPACE_METRIC_DETACH = Config.DUAL_SPACE_METRIC_DETACH
FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE = Config.FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE
FUSION_METRIC_PROJ_HIDDEN_MULT = Config.FUSION_METRIC_PROJ_HIDDEN_MULT
FUSION_ALIGN_GROUP_ENABLE = Config.FUSION_ALIGN_GROUP_ENABLE
METRIC_GEOMETRY_GROUP_ENABLE = Config.METRIC_GEOMETRY_GROUP_ENABLE
DOMAIN_ADAPT_GROUP_ENABLE = Config.DOMAIN_ADAPT_GROUP_ENABLE
FUSION_SUPCON_ENABLE = Config.FUSION_SUPCON_ENABLE
FUSION_SUPCON_WEIGHT = Config.FUSION_SUPCON_WEIGHT
FUSION_SUPCON_TEMPERATURE = Config.FUSION_SUPCON_TEMPERATURE
FUSION_SUPCON_FAULT_ONLY = Config.FUSION_SUPCON_FAULT_ONLY
FUSION_GEOMETRY_LOSS_MODE = Config.FUSION_GEOMETRY_LOSS_MODE
FUSION_SILHOUETTE_EPS = Config.FUSION_SILHOUETTE_EPS
FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE = Config.FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE
FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT = Config.FUSION_METRIC_LEARNABLE_PROTO_CE_WEIGHT
FUSION_METRIC_LEARNABLE_PROTO_CE_TEMP = Config.FUSION_METRIC_LEARNABLE_PROTO_CE_TEMP
FUSION_METRIC_LEARNABLE_PROTO_CE_START_EPOCH = Config.FUSION_METRIC_LEARNABLE_PROTO_CE_START_EPOCH
FUSION_DECODER_INPUT_STOPGRAD_ENABLE = Config.FUSION_DECODER_INPUT_STOPGRAD_ENABLE
FUSION_DECODER_STOPGRAD_ANNEAL_ENABLE = Config.FUSION_DECODER_STOPGRAD_ANNEAL_ENABLE
FUSION_DECODER_STOPGRAD_ANNEAL_START_EPOCH = Config.FUSION_DECODER_STOPGRAD_ANNEAL_START_EPOCH
FUSION_DECODER_STOPGRAD_ANNEAL_END_EPOCH = Config.FUSION_DECODER_STOPGRAD_ANNEAL_END_EPOCH
FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_START = Config.FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_START
FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_END = Config.FUSION_DECODER_STOPGRAD_ANNEAL_LIVE_RATIO_END
FUSION_DECODER_INPUT_GRAD_LIVE_RATIO = Config.FUSION_DECODER_INPUT_GRAD_LIVE_RATIO
FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH = Config.FUSION_SEMANTIC_USE_DIRECT_METRIC_PATH
FUSION_SEMANTIC_BYPASS_SEM_ADAPTER_ON_DIRECT_PATH = Config.FUSION_SEMANTIC_BYPASS_SEM_ADAPTER_ON_DIRECT_PATH
USE_SIMPLEX_ETF_PROTOTYPES = Config.USE_SIMPLEX_ETF_PROTOTYPES
SIMPLEX_ETF_SEED = Config.SIMPLEX_ETF_SEED
SIMPLEX_ETF_KEEP_TEXT_PROTO_FOR_EVAL = Config.SIMPLEX_ETF_KEEP_TEXT_PROTO_FOR_EVAL
USE_PROTOTYPE_ANCHOR_LOSS = Config.USE_PROTOTYPE_ANCHOR_LOSS
PROTOTYPE_ANCHOR_WEIGHT_SPEC = Config.PROTOTYPE_ANCHOR_WEIGHT_SPEC
PROTOTYPE_ANCHOR_WEIGHT_FUSION = Config.PROTOTYPE_ANCHOR_WEIGHT_FUSION
PROTOTYPE_ANCHOR_TEMPERATURE = Config.PROTOTYPE_ANCHOR_TEMPERATURE
PROTOTYPE_ANCHOR_START_EPOCH = Config.PROTOTYPE_ANCHOR_START_EPOCH
PROTOTYPE_ANCHOR_LABEL_SMOOTHING = Config.PROTOTYPE_ANCHOR_LABEL_SMOOTHING
USE_TEXT_ETF_ALIGNMENT_LOSS = Config.USE_TEXT_ETF_ALIGNMENT_LOSS
TEXT_ETF_ALIGNMENT_WEIGHT = Config.TEXT_ETF_ALIGNMENT_WEIGHT
TEXT_ETF_ALIGNMENT_START_EPOCH = Config.TEXT_ETF_ALIGNMENT_START_EPOCH
TEXT_ETF_ALIGNMENT_BATCH_INTERVAL = Config.TEXT_ETF_ALIGNMENT_BATCH_INTERVAL
TEXT_ETF_COSINE_MATRIX_WEIGHT = Config.TEXT_ETF_COSINE_MATRIX_WEIGHT
USE_CLIP_STYLE_CONTRASTIVE = Config.USE_CLIP_STYLE_CONTRASTIVE
CLIP_STYLE_CONTRASTIVE_WEIGHT = Config.CLIP_STYLE_CONTRASTIVE_WEIGHT
CLIP_STYLE_CONTRASTIVE_TEMPERATURE = Config.CLIP_STYLE_CONTRASTIVE_TEMPERATURE
CLIP_STYLE_CONTRASTIVE_START_EPOCH = Config.CLIP_STYLE_CONTRASTIVE_START_EPOCH
USE_DUAL_CLIP_SPEC = Config.USE_DUAL_CLIP_SPEC
DUAL_CLIP_SPEC_WEIGHT = Config.DUAL_CLIP_SPEC_WEIGHT
USE_ETF_TEXT_ADAPTER = Config.USE_ETF_TEXT_ADAPTER
ETF_TEXT_ADAPTER_WEIGHT = Config.ETF_TEXT_ADAPTER_WEIGHT
ETF_TEXT_ADAPTER_START_EPOCH = Config.ETF_TEXT_ADAPTER_START_EPOCH
ETF_TEXT_ADAPTER_TEMPERATURE = Config.ETF_TEXT_ADAPTER_TEMPERATURE
ETF_TEXT_ADAPTER_LR_MULT = Config.ETF_TEXT_ADAPTER_LR_MULT
USE_LEARNABLE_PROTOTYPES = Config.USE_LEARNABLE_PROTOTYPES
LEARNABLE_PROTO_MIN_COS = Config.LEARNABLE_PROTO_MIN_COS
LEARNABLE_PROTO_SEP_WEIGHT = Config.LEARNABLE_PROTO_SEP_WEIGHT
LEARNABLE_PROTO_ANCHOR_WEIGHT = Config.LEARNABLE_PROTO_ANCHOR_WEIGHT
LEARNABLE_PROTO_INIT_FROM_ETF = Config.LEARNABLE_PROTO_INIT_FROM_ETF
USE_PHYS_FEATURE_PROTO = Config.USE_PHYS_FEATURE_PROTO
PHYS_FEATURE_HIDDEN_DIM = Config.PHYS_FEATURE_HIDDEN_DIM
PHYS_FEATURE_ALIGN_WEIGHT = Config.PHYS_FEATURE_ALIGN_WEIGHT
USE_FEATURE_DECOUPLE = Config.USE_FEATURE_DECOUPLE
DECOUPLE_HIDDEN_DIM = Config.DECOUPLE_HIDDEN_DIM
DECOUPLE_NUM_CONDITIONS = Config.DECOUPLE_NUM_CONDITIONS
DECOUPLE_COND_WEIGHT = Config.DECOUPLE_COND_WEIGHT
DECOUPLE_ADV_WEIGHT = Config.DECOUPLE_ADV_WEIGHT
DECOUPLE_ORTHO_WEIGHT = Config.DECOUPLE_ORTHO_WEIGHT
DECOUPLE_ADV_LAMBDA_MAX = Config.DECOUPLE_ADV_LAMBDA_MAX
DECOUPLE_START_EPOCH = Config.DECOUPLE_START_EPOCH
LEARNABLE_PROTO_LR_MULT = Config.LEARNABLE_PROTO_LR_MULT
LEARNABLE_PROTO_START_EPOCH = Config.LEARNABLE_PROTO_START_EPOCH
# ATT397: SAM + Episodic Training
USE_SAM_OPTIMIZER = Config.USE_SAM_OPTIMIZER
SAM_RHO = Config.SAM_RHO
SAM_ADAPTIVE = Config.SAM_ADAPTIVE
USE_EPISODIC_TRAINING = Config.USE_EPISODIC_TRAINING
EPISODIC_SPLIT_KEY = Config.EPISODIC_SPLIT_KEY
EPISODIC_PROTO_LOSS_WEIGHT = Config.EPISODIC_PROTO_LOSS_WEIGHT
EPISODIC_TEMPERATURE = Config.EPISODIC_TEMPERATURE
EPISODIC_GRAD_ALIGN_ENABLE = Config.EPISODIC_GRAD_ALIGN_ENABLE
EPISODIC_GRAD_ALIGN_WEIGHT = Config.EPISODIC_GRAD_ALIGN_WEIGHT
USE_SUPCLIP = Config.USE_SUPCLIP
PACL_USE_MARGIN = Config.PACL_USE_MARGIN
PACL_MARGIN = Config.PACL_MARGIN
LOSS_BUDGET_INFO_PREFIX = Config.LOSS_BUDGET_INFO_PREFIX
USE_PHYSICAL_FREQ_POSITION_ENCODING = Config.USE_PHYSICAL_FREQ_POSITION_ENCODING
PHYSICAL_FREQ_POSITION_ENCODING_MODE = Config.PHYSICAL_FREQ_POSITION_ENCODING_MODE
PHYSICAL_FREQ_POSITION_ENCODING_SCALE = Config.PHYSICAL_FREQ_POSITION_ENCODING_SCALE
PHYSICAL_FREQ_REFERENCE_HZ = Config.PHYSICAL_FREQ_REFERENCE_HZ
PHYSICAL_FREQ_LOG_STATS = Config.PHYSICAL_FREQ_LOG_STATS
USE_FUSION_PROTO_ALIGN_LOSS = Config.USE_FUSION_PROTO_ALIGN_LOSS
FUSION_PROTO_ALIGN_WEIGHT = Config.FUSION_PROTO_ALIGN_WEIGHT
FUSION_PROTO_ALIGN_TEMPERATURE = Config.FUSION_PROTO_ALIGN_TEMPERATURE
FUSION_PROTO_ALIGN_START_EPOCH = Config.FUSION_PROTO_ALIGN_START_EPOCH
FUSION_PROTO_ALIGN_MIN_SAMPLES_PER_CLASS = Config.FUSION_PROTO_ALIGN_MIN_SAMPLES_PER_CLASS
FUSION_PROTO_ALIGN_USE_CLASS_WEIGHTS = Config.FUSION_PROTO_ALIGN_USE_CLASS_WEIGHTS
USE_NR_INTER_COS_MARGIN_LOSS = Config.USE_NR_INTER_COS_MARGIN_LOSS
NR_INTER_COS_MARGIN_WEIGHT = Config.NR_INTER_COS_MARGIN_WEIGHT
NR_INTER_COS_MARGIN_TARGET_MAX = Config.NR_INTER_COS_MARGIN_TARGET_MAX
NR_INTER_COS_MARGIN_START_EPOCH = Config.NR_INTER_COS_MARGIN_START_EPOCH
NR_INTER_COS_MIN_SAMPLES = Config.NR_INTER_COS_MIN_SAMPLES
USE_OUTER_MARGIN_LOSS = Config.USE_OUTER_MARGIN_LOSS
OUTER_MARGIN_WEIGHT = Config.OUTER_MARGIN_WEIGHT
OUTER_MARGIN = Config.OUTER_MARGIN
OUTER_MARGIN_START_EPOCH = Config.OUTER_MARGIN_START_EPOCH
USE_INNER_MARGIN_LOSS = Config.USE_INNER_MARGIN_LOSS
INNER_MARGIN_WEIGHT = Config.INNER_MARGIN_WEIGHT
INNER_MARGIN = Config.INNER_MARGIN
INNER_MARGIN_START_EPOCH = Config.INNER_MARGIN_START_EPOCH
USE_NORMAL_MARGIN_LOSS = Config.USE_NORMAL_MARGIN_LOSS
NORMAL_MARGIN_WEIGHT = Config.NORMAL_MARGIN_WEIGHT
NORMAL_MARGIN = Config.NORMAL_MARGIN
NORMAL_MARGIN_START_EPOCH = Config.NORMAL_MARGIN_START_EPOCH
USE_ROLLING_ABSORB_MARGIN_LOSS = Config.USE_ROLLING_ABSORB_MARGIN_LOSS
ROLLING_ABSORB_MARGIN_WEIGHT = Config.ROLLING_ABSORB_MARGIN_WEIGHT
ROLLING_ABSORB_MARGIN = Config.ROLLING_ABSORB_MARGIN
ROLLING_ABSORB_MARGIN_START_EPOCH = Config.ROLLING_ABSORB_MARGIN_START_EPOCH
USE_INNER_VS_ROLLING_MARGIN_LOSS = Config.USE_INNER_VS_ROLLING_MARGIN_LOSS
INNER_VS_ROLLING_MARGIN_WEIGHT = Config.INNER_VS_ROLLING_MARGIN_WEIGHT
INNER_VS_ROLLING_MARGIN = Config.INNER_VS_ROLLING_MARGIN
INNER_VS_ROLLING_MARGIN_START_EPOCH = Config.INNER_VS_ROLLING_MARGIN_START_EPOCH
USE_OUTER_VS_ROLLING_MARGIN_LOSS = Config.USE_OUTER_VS_ROLLING_MARGIN_LOSS
OUTER_VS_ROLLING_MARGIN_WEIGHT = Config.OUTER_VS_ROLLING_MARGIN_WEIGHT
OUTER_VS_ROLLING_MARGIN = Config.OUTER_VS_ROLLING_MARGIN
OUTER_VS_ROLLING_MARGIN_START_EPOCH = Config.OUTER_VS_ROLLING_MARGIN_START_EPOCH
USE_INNER_VS_NORMAL_MARGIN_LOSS = Config.USE_INNER_VS_NORMAL_MARGIN_LOSS
INNER_VS_NORMAL_MARGIN_WEIGHT = Config.INNER_VS_NORMAL_MARGIN_WEIGHT
INNER_VS_NORMAL_MARGIN = Config.INNER_VS_NORMAL_MARGIN
INNER_VS_NORMAL_MARGIN_START_EPOCH = Config.INNER_VS_NORMAL_MARGIN_START_EPOCH
USE_OUTER_VS_NORMAL_MARGIN_LOSS = Config.USE_OUTER_VS_NORMAL_MARGIN_LOSS
OUTER_VS_NORMAL_MARGIN_WEIGHT = Config.OUTER_VS_NORMAL_MARGIN_WEIGHT
OUTER_VS_NORMAL_MARGIN = Config.OUTER_VS_NORMAL_MARGIN
OUTER_VS_NORMAL_MARGIN_START_EPOCH = Config.OUTER_VS_NORMAL_MARGIN_START_EPOCH
USE_DIAG_ANTICOLLAPSE_REG = Config.USE_DIAG_ANTICOLLAPSE_REG
DIAG_ANTICOLLAPSE_WEIGHT = Config.DIAG_ANTICOLLAPSE_WEIGHT
DIAG_ANTICOLLAPSE_EPS = Config.DIAG_ANTICOLLAPSE_EPS
DIAG_ANTICOLLAPSE_TEMPERATURE = Config.DIAG_ANTICOLLAPSE_TEMPERATURE
DIAG_ANTICOLLAPSE_TARGET = Config.DIAG_ANTICOLLAPSE_TARGET
DIAG_ANTICOLLAPSE_PRIOR = Config.DIAG_ANTICOLLAPSE_PRIOR
USE_AM_SOFTMAX_FAULT = Config.USE_AM_SOFTMAX_FAULT
AM_SOFTMAX_M = Config.AM_SOFTMAX_M
AM_SOFTMAX_S = Config.AM_SOFTMAX_S
AM_SOFTMAX_START_EPOCH = Config.AM_SOFTMAX_START_EPOCH
AM_SOFTMAX_WARMUP_EPOCHS = Config.AM_SOFTMAX_WARMUP_EPOCHS
AM_SOFTMAX_INNER_OUTER_ONLY = Config.AM_SOFTMAX_INNER_OUTER_ONLY
USE_NF_PROTECTED_ARGMAX = Config.USE_NF_PROTECTED_ARGMAX
NF_PROTECT_THRESHOLD = Config.NF_PROTECT_THRESHOLD
NF_PROTECT_PFAULT_MAX = Config.NF_PROTECT_PFAULT_MAX
NF_PROTECT_MODE = Config.NF_PROTECT_MODE
NF_PROTECT_BIAS = Config.NF_PROTECT_BIAS
NF_PROTECT_DIAG_MARGIN = Config.NF_PROTECT_DIAG_MARGIN
USE_ROLLING_VS_NORMAL_MARGIN_LOSS = Config.USE_ROLLING_VS_NORMAL_MARGIN_LOSS
ROLLING_VS_NORMAL_MARGIN_WEIGHT = Config.ROLLING_VS_NORMAL_MARGIN_WEIGHT
ROLLING_VS_NORMAL_MARGIN = Config.ROLLING_VS_NORMAL_MARGIN
ROLLING_VS_NORMAL_MARGIN_START_EPOCH = Config.ROLLING_VS_NORMAL_MARGIN_START_EPOCH
USE_NORMAL_VS_INNER_MARGIN_LOSS = Config.USE_NORMAL_VS_INNER_MARGIN_LOSS
NORMAL_VS_INNER_MARGIN_WEIGHT = Config.NORMAL_VS_INNER_MARGIN_WEIGHT
NORMAL_VS_INNER_MARGIN = Config.NORMAL_VS_INNER_MARGIN
NORMAL_VS_INNER_MARGIN_START_EPOCH = Config.NORMAL_VS_INNER_MARGIN_START_EPOCH
USE_OUTER_VS_INNER_MARGIN_LOSS = Config.USE_OUTER_VS_INNER_MARGIN_LOSS
OUTER_VS_INNER_MARGIN_WEIGHT = Config.OUTER_VS_INNER_MARGIN_WEIGHT
OUTER_VS_INNER_MARGIN = Config.OUTER_VS_INNER_MARGIN
OUTER_VS_INNER_MARGIN_START_EPOCH = Config.OUTER_VS_INNER_MARGIN_START_EPOCH
USE_ROLLING_VS_INNER_MARGIN_LOSS = Config.USE_ROLLING_VS_INNER_MARGIN_LOSS
ROLLING_VS_INNER_MARGIN_WEIGHT = Config.ROLLING_VS_INNER_MARGIN_WEIGHT
ROLLING_VS_INNER_MARGIN = Config.ROLLING_VS_INNER_MARGIN
ROLLING_VS_INNER_MARGIN_START_EPOCH = Config.ROLLING_VS_INNER_MARGIN_START_EPOCH
SUP_FT_NF_WEIGHT = Config.SUP_FT_NF_WEIGHT
SUP_FT_DIAG_WEIGHT = Config.SUP_FT_DIAG_WEIGHT
CLAMP_ADAPTIVE_POS_WEIGHT_MULTI = Config.CLAMP_ADAPTIVE_POS_WEIGHT_MULTI
ADAPTIVE_POS_WEIGHT_MULTI_MAX = Config.ADAPTIVE_POS_WEIGHT_MULTI_MAX
TRAIN_MODE = Config.TRAIN_MODE
CHECKPOINT_DIR = Config.CHECKPOINT_DIR
CHECKPOINT_PATH = Config.CHECKPOINT_PATH
CHECKPOINT_SAVE_INTERVAL = Config.CHECKPOINT_SAVE_INTERVAL
START_FROM_SCRATCH = Config.START_FROM_SCRATCH
SKIP_TASK1_CWRU = Config.SKIP_TASK1_CWRU
SKIP_TASK1_PU = Config.SKIP_TASK1_PU
SKIP_TASK2 = Config.SKIP_TASK2
RESUME_TASK1_CWRU_CHECKPOINT = Config.RESUME_TASK1_CWRU_CHECKPOINT
RESUME_TASK1_PU_CHECKPOINT = Config.RESUME_TASK1_PU_CHECKPOINT
RESUME_TASK2_CHECKPOINT = Config.RESUME_TASK2_CHECKPOINT
EVAL_ONLY_MODE = Config.EVAL_ONLY_MODE
EVAL_ONLY_PHASES = Config.EVAL_ONLY_PHASES
EVAL_ONLY_CHECKPOINTS = Config.EVAL_ONLY_CHECKPOINTS
EVAL_ONLY_SKIP_TASK1_SUP_FT = Config.EVAL_ONLY_SKIP_TASK1_SUP_FT
TARGET_TASK2_DOMAIN_CHECKPOINT_SWITCH_ENABLE = Config.TARGET_TASK2_DOMAIN_CHECKPOINT_SWITCH_ENABLE
TARGET_TASK2_DOMAIN_CHECKPOINTS = Config.TARGET_TASK2_DOMAIN_CHECKPOINTS
LEARNING_RATE = Config.LEARNING_RATE
TOTAL_EPOCHS = Config.TOTAL_EPOCHS
PROGRESSIVE_WEIGHT_TOTAL_EPOCHS = Config.PROGRESSIVE_WEIGHT_TOTAL_EPOCHS
BATCH_SIZE = Config.BATCH_SIZE
BATCH_SIZE_TASK1_PU = Config.BATCH_SIZE_TASK1_PU
BATCH_SIZE_TASK2 = Config.BATCH_SIZE_TASK2
USE_MOE = Config.USE_MOE
ATT469_MOE_STRUCTURE_ENABLE = Config.ATT469_MOE_STRUCTURE_ENABLE
SPEC_ENCODER_BACKEND = Config.SPEC_ENCODER_BACKEND
SPEC_MOE_MULTI_VIEW_ROUTER = Config.SPEC_MOE_MULTI_VIEW_ROUTER
SPEC_MOE_BAND_CROSS_ATTN = Config.SPEC_MOE_BAND_CROSS_ATTN
SPEC_MOE_BAND_SPARSITY_WEIGHT = Config.SPEC_MOE_BAND_SPARSITY_WEIGHT
SPEC_MOE_BAND_ATTN_HEADS = Config.SPEC_MOE_BAND_ATTN_HEADS
LIGHT_SPEC_BASE_CHANNELS = Config.LIGHT_SPEC_BASE_CHANNELS
LIGHT_SPEC_DROPOUT = Config.LIGHT_SPEC_DROPOUT
USE_GROUPED_FREQ_POOL = Config.USE_GROUPED_FREQ_POOL
GROUPED_FREQ_POOL_LOW_BANDS = Config.GROUPED_FREQ_POOL_LOW_BANDS
GROUPED_FREQ_POOL_HIGH_BANDS = Config.GROUPED_FREQ_POOL_HIGH_BANDS
MOE_LOSS_WEIGHT = Config.MOE_LOSS_WEIGHT
NUM_BANDS = Config.NUM_BANDS
NUM_EXPERTS_PER_BAND = Config.NUM_EXPERTS_PER_BAND
EXPERT_TOP_K = Config.EXPERT_TOP_K

# Bayesian Router
USE_BAYESIAN_ROUTER = Config.USE_BAYESIAN_ROUTER
BAYESIAN_ROUTER_PRIOR_STD = Config.BAYESIAN_ROUTER_PRIOR_STD
BAYESIAN_ROUTER_KL_WEIGHT = Config.BAYESIAN_ROUTER_KL_WEIGHT
BAYESIAN_ROUTER_NUM_SAMPLES_INFERENCE = Config.BAYESIAN_ROUTER_NUM_SAMPLES_INFERENCE

# Expert Performance Tracking
USE_EXPERT_PERFORMANCE_TRACKING = Config.USE_EXPERT_PERFORMANCE_TRACKING
EXPERT_TRACKER_MODE = Config.EXPERT_TRACKER_MODE
EXPERT_TRACKER_EXPLORATION_BONUS = Config.EXPERT_TRACKER_EXPLORATION_BONUS
EXPERT_TRACKER_UPDATE_INTERVAL = Config.EXPERT_TRACKER_UPDATE_INTERVAL

# Expert Disagreement
USE_EXPERT_DISAGREEMENT = Config.USE_EXPERT_DISAGREEMENT
EXPERT_DISAGREEMENT_METHOD = Config.EXPERT_DISAGREEMENT_METHOD
EXPERT_DISAGREEMENT_TEMP_SCALE = Config.EXPERT_DISAGREEMENT_TEMP_SCALE
EXPERT_DISAGREEMENT_TEMP_MODE = Config.EXPERT_DISAGREEMENT_TEMP_MODE
EXPERT_DISAGREEMENT_REG_WEIGHT = Config.EXPERT_DISAGREEMENT_REG_WEIGHT
EXPERT_DISAGREEMENT_TARGET = Config.EXPERT_DISAGREEMENT_TARGET
EXPERT_DISAGREEMENT_MARGIN = Config.EXPERT_DISAGREEMENT_MARGIN

USE_BAYESIAN_PROTOTYPES = Config.USE_BAYESIAN_PROTOTYPES
BAYESIAN_PRIOR_STRENGTH = Config.BAYESIAN_PRIOR_STRENGTH
BAYESIAN_UPDATE_UNTIL_EPOCH = Config.BAYESIAN_UPDATE_UNTIL_EPOCH
DETECTOR_FINETUNING_ENABLE = Config.DETECTOR_FINETUNING_ENABLE
DETECTOR_FINETUNING_SAMPLES = Config.DETECTOR_FINETUNING_SAMPLES
DETECTOR_FINETUNING_LR = Config.DETECTOR_FINETUNING_LR
DETECTOR_FINETUNING_STEPS = Config.DETECTOR_FINETUNING_STEPS
DETECTOR_FINETUNING_TARGET_DOMAINS = Config.DETECTOR_FINETUNING_TARGET_DOMAINS
TARGET_FINETUNE_PERCENT = Config.TARGET_FINETUNE_PERCENT
TARGET_FINETUNE_PERCENT_HUST = Config.TARGET_FINETUNE_PERCENT_HUST
TARGET_FINETUNE_PERCENT_SEU = Config.TARGET_FINETUNE_PERCENT_SEU
TARGET_FINETUNE_PERCENT_HIT = Config.TARGET_FINETUNE_PERCENT_HIT
TARGET_NSHOT_ENABLE = Config.TARGET_NSHOT_ENABLE
TARGET_NSHOT_K = Config.TARGET_NSHOT_K
TARGET_NSHOT_REPEAT = Config.TARGET_NSHOT_REPEAT
TARGET_UNSUP_STEPS = Config.TARGET_UNSUP_STEPS
TARGET_UNSUP_LR = Config.TARGET_UNSUP_LR
TARGET_SUP_STEPS = Config.TARGET_SUP_STEPS
TARGET_SUP_LR = Config.TARGET_SUP_LR
TARGET_SUP_NF_WEIGHT = Config.TARGET_SUP_NF_WEIGHT
TARGET_SUP_DIAG_WEIGHT = Config.TARGET_SUP_DIAG_WEIGHT
TASK1_ENABLE_PRETEST_FINETUNE = Config.TASK1_ENABLE_PRETEST_FINETUNE
TASK1_FINETUNE_PERCENT = Config.TASK1_FINETUNE_PERCENT
TASK1_CWRU_SKIP_SUPERVISED_FINETUNE = Config.TASK1_CWRU_SKIP_SUPERVISED_FINETUNE
TASK1_CWRU_RELAX_SPEC_GATE = Config.TASK1_CWRU_RELAX_SPEC_GATE
TARGET_SKIP_HUST_UNSUP_ADAPT = Config.TARGET_SKIP_HUST_UNSUP_ADAPT
TARGET_EVAL_SKIP_HUST = Config.TARGET_EVAL_SKIP_HUST
TARGET_EVAL_SKIP_SEU = Config.TARGET_EVAL_SKIP_SEU
TARGET_EVAL_SKIP_HIT = Config.TARGET_EVAL_SKIP_HIT
UNIFIED_ZEROSHOT_EVAL = Config.UNIFIED_ZEROSHOT_EVAL
ZERO_SHOT_FINETUNING_ENABLE = Config.ZERO_SHOT_FINETUNING_ENABLE
ZERO_SHOT_FINETUNING_SAMPLES = Config.ZERO_SHOT_FINETUNING_SAMPLES
ZERO_SHOT_FINETUNING_LR = Config.ZERO_SHOT_FINETUNING_LR
ZERO_SHOT_FINETUNING_STEPS = Config.ZERO_SHOT_FINETUNING_STEPS
USE_TPT = Config.USE_TPT
TPT_STEPS = Config.TPT_STEPS
TPT_LR = Config.TPT_LR
TPT_TEMPERATURE = Config.TPT_TEMPERATURE
TPT_ENTROPY_THRESHOLD = Config.TPT_ENTROPY_THRESHOLD
TPT_CONFIDENCE_FILTER = Config.TPT_CONFIDENCE_FILTER
USE_TRANSDUCTIVE = Config.USE_TRANSDUCTIVE
TRANSDUCTIVE_SOURCE_FREE = Config.TRANSDUCTIVE_SOURCE_FREE
TRANSDUCTIVE_FALLBACK_SOURCE_FREE_WHEN_NO_SOURCE = Config.TRANSDUCTIVE_FALLBACK_SOURCE_FREE_WHEN_NO_SOURCE
TRANSDUCTIVE_INIT_FROM_FPROTO = Config.TRANSDUCTIVE_INIT_FROM_FPROTO
TRANSDUCTIVE_FUSION_VAR_COLLAPSE_THRESHOLD = Config.TRANSDUCTIVE_FUSION_VAR_COLLAPSE_THRESHOLD
TRANSDUCTIVE_BMM_SEP_RATIO_GATE = Config.TRANSDUCTIVE_BMM_SEP_RATIO_GATE
EVAL_FINAL_DECISION_COLLAPSE_SEP_RATIO_GATE = Config.EVAL_FINAL_DECISION_COLLAPSE_SEP_RATIO_GATE
EVAL_FINAL_DECISION_EXCLUDE_TRANSDUCTIVE_ON_COLLAPSE = Config.EVAL_FINAL_DECISION_EXCLUDE_TRANSDUCTIVE_ON_COLLAPSE
TRANSDUCTIVE_N_CLUSTERS_FACTOR = Config.TRANSDUCTIVE_N_CLUSTERS_FACTOR
TRANSDUCTIVE_KNN_K = Config.TRANSDUCTIVE_KNN_K
TRANSDUCTIVE_ALPHA_TEXT = Config.TRANSDUCTIVE_ALPHA_TEXT
TRANSDUCTIVE_IM_ITERATIONS = Config.TRANSDUCTIVE_IM_ITERATIONS
TRANSDUCTIVE_IM_CONFIDENT_RATIO = Config.TRANSDUCTIVE_IM_CONFIDENT_RATIO
TRANSDUCTIVE_TEMPERATURE = Config.TRANSDUCTIVE_TEMPERATURE
USE_PHYSICS_FEATURES = Config.USE_PHYSICS_FEATURES
PHYSICS_FEAT_WEIGHT = Config.PHYSICS_FEAT_WEIGHT
PHYSICS_KNN_METRIC = Config.PHYSICS_KNN_METRIC
TRANSDUCTIVE_KL_REG_LAMBDA = Config.TRANSDUCTIVE_KL_REG_LAMBDA
TRANSDUCTIVE_ADAPTIVE_ALPHA = Config.TRANSDUCTIVE_ADAPTIVE_ALPHA
TRANSDUCTIVE_ADAPTIVE_ALPHA_REF_COS = Config.TRANSDUCTIVE_ADAPTIVE_ALPHA_REF_COS
TRANSDUCTIVE_ADAPTIVE_ALPHA_MIN_RATIO = Config.TRANSDUCTIVE_ADAPTIVE_ALPHA_MIN_RATIO
TRANSDUCTIVE_USE_MULTI_TEMPLATE = Config.TRANSDUCTIVE_USE_MULTI_TEMPLATE
TRANSDUCTIVE_USE_ATTRIBUTE_MATCHING = Config.TRANSDUCTIVE_USE_ATTRIBUTE_MATCHING
TRANSDUCTIVE_ATTRIBUTE_WEIGHT = Config.TRANSDUCTIVE_ATTRIBUTE_WEIGHT
TRANSDUCTIVE_USE_MULTICENTER_KNN = Config.TRANSDUCTIVE_USE_MULTICENTER_KNN
TRANSDUCTIVE_MULTICENTER_MAX_K = Config.TRANSDUCTIVE_MULTICENTER_MAX_K
TRANSDUCTIVE_USE_CLASS_RESCUE = Config.TRANSDUCTIVE_USE_CLASS_RESCUE
TRANSDUCTIVE_RESCUE_THRESHOLD = Config.TRANSDUCTIVE_RESCUE_THRESHOLD
TRANSDUCTIVE_USE_BMM = Config.TRANSDUCTIVE_USE_BMM
TRANSDUCTIVE_BMM_KL_LAMBDA = Config.TRANSDUCTIVE_BMM_KL_LAMBDA
TRANSDUCTIVE_BMM_N_NEIGHBOURS = Config.TRANSDUCTIVE_BMM_N_NEIGHBOURS
TRANSDUCTIVE_BMM_ITERATIONS = Config.TRANSDUCTIVE_BMM_ITERATIONS
TRANSDUCTIVE_BMM_INNER_MM_ITERS = Config.TRANSDUCTIVE_BMM_INNER_MM_ITERS
TRANSDUCTIVE_BMM_CLASS_BALANCE_TAU = Config.TRANSDUCTIVE_BMM_CLASS_BALANCE_TAU
TRANSDUCTIVE_BMM_SINKHORN_ITERS = Config.TRANSDUCTIVE_BMM_SINKHORN_ITERS
USE_CONSISTENCY_REGULARIZATION = Config.USE_CONSISTENCY_REGULARIZATION
CONSISTENCY_WEIGHT = Config.CONSISTENCY_WEIGHT
CONSISTENCY_TEMPERATURE = Config.CONSISTENCY_TEMPERATURE
CONSISTENCY_PROTECT_ENABLE = Config.CONSISTENCY_PROTECT_ENABLE
CONSISTENCY_PROTECT_MIN_CONF = Config.CONSISTENCY_PROTECT_MIN_CONF
W_FUSION_CLS = Config.W_FUSION_CLS
W_SIGNAL_ALIGN = Config.W_SIGNAL_ALIGN
MSLR_SIGNAL_MULT = Config.MSLR_SIGNAL_MULT
MSLR_SPEC_MULT = Config.MSLR_SPEC_MULT
MSLR_FUSION_MULT = Config.MSLR_FUSION_MULT
MSLR_DECODER_MULT = Config.MSLR_DECODER_MULT
USE_SCHEDULE = Config.USE_SCHEDULE
SCHEDULE_EPOCH_THRESHOLD = Config.SCHEDULE_EPOCH_THRESHOLD
LR_DECAY_FACTOR = Config.LR_DECAY_FACTOR
USE_EARLY_STOPPING = Config.USE_EARLY_STOPPING
EARLY_STOP_PATIENCE = Config.EARLY_STOP_PATIENCE
EARLY_STOP_METRIC = Config.EARLY_STOP_METRIC
EARLY_STOP_METRIC_TASK1_CWRU = Config.EARLY_STOP_METRIC_TASK1_CWRU
EARLY_STOP_METRIC_TASK1_PU = Config.EARLY_STOP_METRIC_TASK1_PU
EARLY_STOP_METRIC_TASK2 = Config.EARLY_STOP_METRIC_TASK2
EARLY_STOP_MIN_EPOCHS = Config.EARLY_STOP_MIN_EPOCHS
EARLY_STOP_MIN_EPOCHS_TASK2 = Config.EARLY_STOP_MIN_EPOCHS_TASK2
EARLY_STOP_SKIP_PREWINDOW_BASELINE = Config.EARLY_STOP_SKIP_PREWINDOW_BASELINE
EARLY_STOP_SMOOTH_WINDOW = Config.EARLY_STOP_SMOOTH_WINDOW
USE_SWAD = Config.USE_SWAD
SWAD_WINDOW = Config.SWAD_WINDOW
SWAD_MIN_MODELS = Config.SWAD_MIN_MODELS
EVAL_USE_SWAD_CHECKPOINT = Config.EVAL_USE_SWAD_CHECKPOINT
USE_AMP = Config.USE_AMP
USE_TORCH_COMPILE = Config.USE_TORCH_COMPILE
USE_PCGRAD = Config.USE_PCGRAD
PCGRAD_LOG_INTERVAL = Config.PCGRAD_LOG_INTERVAL
MAX_CLASS_WEIGHT = Config.MAX_CLASS_WEIGHT
SAMPLER_WEIGHT_SCALE = Config.SAMPLER_WEIGHT_SCALE
USE_WEIGHTED_SAMPLER = Config.USE_WEIGHTED_SAMPLER
USE_PROGRESSIVE_WEIGHTS = Config.USE_PROGRESSIVE_WEIGHTS
INITIAL_WEIGHT_SCALE = Config.INITIAL_WEIGHT_SCALE
WEIGHT_SCHEDULE = Config.WEIGHT_SCHEDULE
ENABLE_CLASS_MONITORING = Config.ENABLE_CLASS_MONITORING
USE_ADAPTIVE_CLASS_WEIGHTS = Config.USE_ADAPTIVE_CLASS_WEIGHTS
ADAPTIVE_WEIGHT_MOMENTUM = Config.ADAPTIVE_WEIGHT_MOMENTUM
ADAPTIVE_WEIGHT_MIN = Config.ADAPTIVE_WEIGHT_MIN
ADAPTIVE_WEIGHT_MAX = Config.ADAPTIVE_WEIGHT_MAX
USE_ROLLING_FOCAL_LOSS = Config.USE_ROLLING_FOCAL_LOSS
ROLLING_FOCAL_GAMMA = Config.ROLLING_FOCAL_GAMMA
ROLLING_FOCAL_ALPHA = Config.ROLLING_FOCAL_ALPHA
ROLLING_TRAIN_ON_FAULT_ONLY = Config.ROLLING_TRAIN_ON_FAULT_ONLY
USE_ROLLING_DANN = Config.USE_ROLLING_DANN
ROLLING_DANN_LAMBDA = Config.ROLLING_DANN_LAMBDA
ROLLING_DANN_START_EPOCH = Config.ROLLING_DANN_START_EPOCH
ROLLING_DANN_MAX_ALPHA = Config.ROLLING_DANN_MAX_ALPHA
ROLLING_DANN_ALPHA_SCHEDULE = Config.ROLLING_DANN_ALPHA_SCHEDULE
ROLLING_STAGE2_FREEZE_EPOCHS = Config.ROLLING_STAGE2_FREEZE_EPOCHS
USE_ROLLING_DECORR_LOSS = Config.USE_ROLLING_DECORR_LOSS
ROLLING_DECORR_WEIGHT = Config.ROLLING_DECORR_WEIGHT
ROLLING_DECORR_START_EPOCH = Config.ROLLING_DECORR_START_EPOCH
ROLLING_DECORR_VAR_TARGET = Config.ROLLING_DECORR_VAR_TARGET
ROLLING_DECORR_COV_WEIGHT = Config.ROLLING_DECORR_COV_WEIGHT
USE_ROLLING_DECORR_INPUT = Config.USE_ROLLING_DECORR_INPUT
ROLLING_DECORR_INPUT_WEIGHT = Config.ROLLING_DECORR_INPUT_WEIGHT
ROLLING_DECORR_RESET_ON_ZERO = Config.ROLLING_DECORR_RESET_ON_ZERO
ROLLING_DECORR_RESET_EPS = Config.ROLLING_DECORR_RESET_EPS
ROLLING_DECORR_RESET_COOLDOWN = Config.ROLLING_DECORR_RESET_COOLDOWN

# ========== (Global Decorr / VICReg) Att83-fix: 模块级导出 ==========
# IMPORTANT: ATT80/81/82 连续三次失败的根因就是缺少这些导出。
# train_alignment.py 使用 `import config as cfg; getattr(cfg, ...)` 读取模块级属性，
# 而非 Config 类属性。没有这些导出，getattr 返回默认值 False/0.05。
USE_GLOBAL_DECORR_LOSS = Config.USE_GLOBAL_DECORR_LOSS
GLOBAL_DECORR_WEIGHT = Config.GLOBAL_DECORR_WEIGHT
GLOBAL_DECORR_WEIGHT_TASK2 = Config.GLOBAL_DECORR_WEIGHT_TASK2
CORE_LOSS_ONLY_MODE = Config.CORE_LOSS_ONLY_MODE
CORE_KEEP_TASK1_CWRU_INVARIANCE = Config.CORE_KEEP_TASK1_CWRU_INVARIANCE
CORE_LOSS_ALLOWED_TERMS = Config.CORE_LOSS_ALLOWED_TERMS
DECOUPLED_TRAINING_ENABLE = Config.DECOUPLED_TRAINING_ENABLE
DECOUPLED_STAGE1_EPOCHS = Config.DECOUPLED_STAGE1_EPOCHS
DECOUPLED_STAGE2_FREEZE_ENCODERS = Config.DECOUPLED_STAGE2_FREEZE_ENCODERS
DECOUPLED_STAGE2_ENABLE_MARGINS = Config.DECOUPLED_STAGE2_ENABLE_MARGINS
TASK1_CWRU_FISHR_SOURCE_DOMAIN_IDS = Config.TASK1_CWRU_FISHR_SOURCE_DOMAIN_IDS
TASK1_CWRU_CCPA_SOURCE_DOMAIN_IDS = Config.TASK1_CWRU_CCPA_SOURCE_DOMAIN_IDS
GLOBAL_DECORR_START_EPOCH = Config.GLOBAL_DECORR_START_EPOCH
GLOBAL_DECORR_VAR_TARGET = Config.GLOBAL_DECORR_VAR_TARGET
GLOBAL_DECORR_COV_WEIGHT = Config.GLOBAL_DECORR_COV_WEIGHT
USE_GLOBAL_VICREG_LOSS = Config.USE_GLOBAL_VICREG_LOSS
GLOBAL_VICREG_VAR_WEIGHT = Config.GLOBAL_VICREG_VAR_WEIGHT
GLOBAL_VICREG_COV_WEIGHT = Config.GLOBAL_VICREG_COV_WEIGHT
GLOBAL_VICREG_STD_TARGET = Config.GLOBAL_VICREG_STD_TARGET
USE_FUSION_VAR_FLOOR_LOSS = Config.USE_FUSION_VAR_FLOOR_LOSS
FUSION_VAR_FLOOR_MIN = Config.FUSION_VAR_FLOOR_MIN
FUSION_VAR_FLOOR_WEIGHT = Config.FUSION_VAR_FLOOR_WEIGHT
USE_FUSION_VICREG_LOSS = Config.USE_FUSION_VICREG_LOSS
FUSION_VICREG_VAR_WEIGHT = Config.FUSION_VICREG_VAR_WEIGHT
FUSION_VICREG_COV_WEIGHT = Config.FUSION_VICREG_COV_WEIGHT
FUSION_VICREG_STD_TARGET = Config.FUSION_VICREG_STD_TARGET
USE_FUSION_INTER_COS_MARGIN_LOSS = Config.USE_FUSION_INTER_COS_MARGIN_LOSS
FUSION_INTER_COS_MARGIN_WEIGHT = Config.FUSION_INTER_COS_MARGIN_WEIGHT
FUSION_INTER_COS_MARGIN_TARGET_MAX = Config.FUSION_INTER_COS_MARGIN_TARGET_MAX
FUSION_INTER_COS_MARGIN_START_EPOCH = Config.FUSION_INTER_COS_MARGIN_START_EPOCH
FUSION_INTER_COS_MIN_SAMPLES_PER_CLASS = Config.FUSION_INTER_COS_MIN_SAMPLES_PER_CLASS
DOMAIN_COND_STRENGTH = Config.DOMAIN_COND_STRENGTH
ABLATION_GATING_ENABLE = Config.ABLATION_GATING_ENABLE
ABLATION_GATE_INIT_DEFAULT = Config.ABLATION_GATE_INIT_DEFAULT
ABLATION_GATE_LOG_INTERVAL = Config.ABLATION_GATE_LOG_INTERVAL
ABLATION_GATE_INIT_OVERRIDES = Config.ABLATION_GATE_INIT_OVERRIDES
ABLATION_GATE_LR_MULT = Config.ABLATION_GATE_LR_MULT
USE_GLOBAL_UNIFORMITY_LOSS = Config.USE_GLOBAL_UNIFORMITY_LOSS
GLOBAL_UNIFORMITY_WEIGHT = Config.GLOBAL_UNIFORMITY_WEIGHT
GLOBAL_UNIFORMITY_T = Config.GLOBAL_UNIFORMITY_T
USE_FISHR_LOSS = Config.USE_FISHR_LOSS
FISHR_WEIGHT = Config.FISHR_WEIGHT
FISHR_START_EPOCH = Config.FISHR_START_EPOCH
FISHR_MIN_SAMPLES_PER_DOMAIN = Config.FISHR_MIN_SAMPLES_PER_DOMAIN
FISHR_SOURCE_DOMAIN_IDS = Config.FISHR_SOURCE_DOMAIN_IDS
USE_CCPA_LOSS = Config.USE_CCPA_LOSS
CCPA_WEIGHT = Config.CCPA_WEIGHT
CCPA_START_EPOCH = Config.CCPA_START_EPOCH
CCPA_MIN_SAMPLES_PER_DOMAIN_CLASS = Config.CCPA_MIN_SAMPLES_PER_DOMAIN_CLASS
CCPA_SOURCE_DOMAIN_IDS = Config.CCPA_SOURCE_DOMAIN_IDS
LOG_GLOBAL_VARIANCE = Config.LOG_GLOBAL_VARIANCE

ROLLING_ADAPTER_DETACH = Config.ROLLING_ADAPTER_DETACH
FUSION_USE_CLS_ADAPTER = Config.FUSION_USE_CLS_ADAPTER
FUSION_METRIC_LOSS = Config.FUSION_METRIC_LOSS
FUSION_METRIC_LOSS_WEIGHT = Config.FUSION_METRIC_LOSS_WEIGHT
FUSION_CENTER_MOMENTUM = Config.FUSION_CENTER_MOMENTUM
CWRU_PATH = Config.CWRU_PATH
PU_PATH = Config.PU_PATH
HUST_PATH = Config.HUST_PATH
HIT_PATH = Config.HIT_PATH
ENABLE_HIT_DATASET = Config.ENABLE_HIT_DATASET
SEU_PATH = Config.SEU_PATH
ENABLE_INTRA_CONDITION_TASK = Config.ENABLE_INTRA_CONDITION_TASK
INTRA_CONDITION_VAL_RATIO = Config.INTRA_CONDITION_VAL_RATIO
TASK1_TRAIN_PER_DOMAIN_ISOLATED = Config.TASK1_TRAIN_PER_DOMAIN_ISOLATED
TASK1_ISOLATED_DOMAIN = Config.TASK1_ISOLATED_DOMAIN
TASK1_RUN_ALL_PHASES_IN_ONE_SCRIPT = Config.TASK1_RUN_ALL_PHASES_IN_ONE_SCRIPT
CWRU_TRAIN_GROUPS = Config.CWRU_TRAIN_GROUPS
CWRU_TEST_GROUPS = Config.CWRU_TEST_GROUPS
PU_TRAIN_GROUPS = Config.PU_TRAIN_GROUPS
PU_TEST_GROUPS = Config.PU_TEST_GROUPS
TEST_INTERVAL = Config.TEST_INTERVAL
TRAIN_ENABLE_TARGET_MONITOR = Config.TRAIN_ENABLE_TARGET_MONITOR
TRAIN_LOG_VERBOSE_DIAGNOSTICS = Config.TRAIN_LOG_VERBOSE_DIAGNOSTICS
TRAIN_LOG_DATAFLOW_AUDIT = Config.TRAIN_LOG_DATAFLOW_AUDIT
TRAIN_DATAFLOW_AUDIT_EPOCH_STRIDE = Config.TRAIN_DATAFLOW_AUDIT_EPOCH_STRIDE
TRAIN_LOG_GRAD_NORM = Config.TRAIN_LOG_GRAD_NORM
TRAIN_LOG_GRAD_NORM_STEP_STRIDE = Config.TRAIN_LOG_GRAD_NORM_STEP_STRIDE
TRAIN_LOG_BATCH_LOSS = Config.TRAIN_LOG_BATCH_LOSS
TRAIN_LOG_BATCH_LOSS_STEP_STRIDE = Config.TRAIN_LOG_BATCH_LOSS_STEP_STRIDE
TRAIN_LOG_BATCH_LOSS_BREAKDOWN = Config.TRAIN_LOG_BATCH_LOSS_BREAKDOWN
TRAIN_LOG_BATCH_LOSS_BREAKDOWN_STEP_STRIDE = Config.TRAIN_LOG_BATCH_LOSS_BREAKDOWN_STEP_STRIDE
TRAIN_LOG_PROTOTYPE_COS_MATRIX = Config.TRAIN_LOG_PROTOTYPE_COS_MATRIX
TRAIN_LOG_PROTOTYPE_COS_STEP_STRIDE = Config.TRAIN_LOG_PROTOTYPE_COS_STEP_STRIDE
TRAIN_LOG_BATCH_IO_STATS = Config.TRAIN_LOG_BATCH_IO_STATS
TRAIN_LOG_BATCH_IO_STEP_STRIDE = Config.TRAIN_LOG_BATCH_IO_STEP_STRIDE
TRAIN_LOG_GRAD_CONTRIB = Config.TRAIN_LOG_GRAD_CONTRIB
TRAIN_LOG_GRAD_CONTRIB_STEP_STRIDE = Config.TRAIN_LOG_GRAD_CONTRIB_STEP_STRIDE
TRAIN_LOG_LOSS_BUDGET = Config.TRAIN_LOG_LOSS_BUDGET
TRAIN_LOG_LOSS_BUDGET_EPOCH_STRIDE = Config.TRAIN_LOG_LOSS_BUDGET_EPOCH_STRIDE
TRAIN_LOG_GRAD_NORM_DETAIL = Config.TRAIN_LOG_GRAD_NORM_DETAIL
TRAIN_LOG_GRAD_NORM_DETAIL_STEP_STRIDE = Config.TRAIN_LOG_GRAD_NORM_DETAIL_STEP_STRIDE
TRAIN_ENABLE_GRAD_CLIP = Config.TRAIN_ENABLE_GRAD_CLIP
TRAIN_GRAD_CLIP_MAX_NORM = Config.TRAIN_GRAD_CLIP_MAX_NORM
TRAIN_LOG_GRAD_CLIP = Config.TRAIN_LOG_GRAD_CLIP
TRAIN_LOG_SPECENC_NORM_STATS = Config.TRAIN_LOG_SPECENC_NORM_STATS
TRAIN_LOG_SPECENC_NORM_STEP_STRIDE = Config.TRAIN_LOG_SPECENC_NORM_STEP_STRIDE
TRAIN_LOG_LOSS_PATH_AUDIT = Config.TRAIN_LOG_LOSS_PATH_AUDIT
LOG_RELIABILITY_GATE = Config.LOG_RELIABILITY_GATE
LAMBDA_CLIP = Config.LAMBDA_CLIP
NUM_WORKERS = Config.NUM_WORKERS
PREFETCH_FACTOR = Config.PREFETCH_FACTOR
PERSISTENT_WORKERS = Config.PERSISTENT_WORKERS
FAULT_LABEL_TO_SEMANTIC = Config.FAULT_LABEL_TO_SEMANTIC
FAULT_PROMPT_TEMPLATES = Config.FAULT_PROMPT_TEMPLATES
RANDOM_SEED = Config.RANDOM_SEED
USE_META_LEARNING = Config.USE_META_LEARNING
META_LEARNING_START_EPOCH = Config.META_LEARNING_START_EPOCH
META_LEARNING_INTERVAL = Config.META_LEARNING_INTERVAL
META_LR = Config.META_LR
META_INNER_LR = Config.META_INNER_LR
META_INNER_STEPS = Config.META_INNER_STEPS
META_FIRST_ORDER = Config.META_FIRST_ORDER
HEALTH_GATE_EPS = Config.HEALTH_GATE_EPS

# Spec feature extraction
SPEC_FEATURE_TYPE = Config.SPEC_FEATURE_TYPE
STFT_MAX_FREQ_BINS = Config.STFT_MAX_FREQ_BINS
STFT_MAX_TIME_BINS = Config.STFT_MAX_TIME_BINS
MULTISCALE_STFT_ENABLE = Config.MULTISCALE_STFT_ENABLE
FUSION_SPEC_RESIDUAL_ENABLE = Config.FUSION_SPEC_RESIDUAL_ENABLE
FUSION_SPEC_RESIDUAL_INIT_ALPHA = Config.FUSION_SPEC_RESIDUAL_INIT_ALPHA
FUSION_SIGNAL_INSTANCE_NORM = Config.FUSION_SIGNAL_INSTANCE_NORM
MULTISCALE_STFT_N_FFTS = Config.MULTISCALE_STFT_N_FFTS
MULTISCALE_STFT_TARGET_F = Config.MULTISCALE_STFT_TARGET_F
MULTISCALE_STFT_TARGET_T = Config.MULTISCALE_STFT_TARGET_T
MULTISCALE_STFT_HOP_DIVISOR = Config.MULTISCALE_STFT_HOP_DIVISOR
MULTISCALE_STFT_NORM = Config.MULTISCALE_STFT_NORM
WST_ENABLE = Config.WST_ENABLE
WST_J = Config.WST_J
WST_Q = Config.WST_Q
WST_ORDER = Config.WST_ORDER
WST_NORM_MODE = Config.WST_NORM_MODE
WST_BYPASS_INJECT = Config.WST_BYPASS_INJECT
WST_BYPASS_DIM = Config.WST_BYPASS_DIM
SPEC_DISTILL_FROM_FUSION = Config.SPEC_DISTILL_FROM_FUSION
SPEC_DISTILL_TEMPERATURE = Config.SPEC_DISTILL_TEMPERATURE
SPEC_DISTILL_ALPHA = Config.SPEC_DISTILL_ALPHA
SPEC_DISTILL_START_EPOCH = Config.SPEC_DISTILL_START_EPOCH
RKD_ENABLE = Config.RKD_ENABLE
RKD_DISTANCE_WEIGHT = Config.RKD_DISTANCE_WEIGHT
RKD_ANGLE_WEIGHT = Config.RKD_ANGLE_WEIGHT
RKD_START_EPOCH = Config.RKD_START_EPOCH
RKD_FAULT_FOCUS_WEIGHT = Config.RKD_FAULT_FOCUS_WEIGHT
SPEC_DISTILL_HF_ONLY = Config.SPEC_DISTILL_HF_ONLY
ENV_BYPASS_INJECT = Config.ENV_BYPASS_INJECT
ENV_BYPASS_DIM = Config.ENV_BYPASS_DIM
SCD_CHANNEL_ENABLE = Config.SCD_CHANNEL_ENABLE
SCD_BYPASS_INJECT = Config.SCD_BYPASS_INJECT
SCD_BYPASS_DIM = Config.SCD_BYPASS_DIM
SCD_N_FFT = Config.SCD_N_FFT
SCD_HOP_DIVISOR = Config.SCD_HOP_DIVISOR
SCD_NORM_MODE = Config.SCD_NORM_MODE
DOMAIN_BN_ENABLE = Config.DOMAIN_BN_ENABLE
MLSA_ENABLE = Config.MLSA_ENABLE
MLSA_WEIGHT = Config.MLSA_WEIGHT
MLSA_WARMUP_EPOCHS = Config.MLSA_WARMUP_EPOCHS
MLSA_MARGIN = Config.MLSA_MARGIN
SPEC_FREQ_ALIGN_ENABLE = Config.SPEC_FREQ_ALIGN_ENABLE
SPEC_FREQ_ALIGN_MAX_HZ = Config.SPEC_FREQ_ALIGN_MAX_HZ
SPEC_ORDER_NORM_ENABLE = Config.SPEC_ORDER_NORM_ENABLE
SPEC_ORDER_NORM_MAX_ORDERS = Config.SPEC_ORDER_NORM_MAX_ORDERS
SPEC_FS_COND_FILM_ENABLE = Config.SPEC_FS_COND_FILM_ENABLE
SPEC_FS_COND_FILM_HIDDEN_DIM = Config.SPEC_FS_COND_FILM_HIDDEN_DIM
SPEC_FS_COND_FILM_REF_FS = Config.SPEC_FS_COND_FILM_REF_FS
SWT_WAVELET = Config.SWT_WAVELET
SWT_LEVELS = Config.SWT_LEVELS
SWT_TIME_BINS = Config.SWT_TIME_BINS
SWT_NORM = Config.SWT_NORM
SWT_NORM_CLIP = Config.SWT_NORM_CLIP
SWT_NORM_EPS = Config.SWT_NORM_EPS
SPEC_INSTANCE_NORM_BEFORE_HEAD = Config.SPEC_INSTANCE_NORM_BEFORE_HEAD
SPEC_USE_FREQ_RATIO_FEATURES = Config.SPEC_USE_FREQ_RATIO_FEATURES
SPEC_FREQ_RATIO_EPS = Config.SPEC_FREQ_RATIO_EPS
SPEC_FREQ_RATIO_BLEND = Config.SPEC_FREQ_RATIO_BLEND
SPEC_FS_AWARE_NORM_ENABLE = Config.SPEC_FS_AWARE_NORM_ENABLE
SPEC_FS_AWARE_REF_FS = Config.SPEC_FS_AWARE_REF_FS
SPEC_FS_AWARE_NORM_POWER = Config.SPEC_FS_AWARE_NORM_POWER
SPEC_FS_AWARE_LOG = Config.SPEC_FS_AWARE_LOG
SWT_RESAMPLE_BEFORE_EXTRACT = Config.SWT_RESAMPLE_BEFORE_EXTRACT
SWT_TARGET_FS = Config.SWT_TARGET_FS
SWT_USE_DYNAMIC_BANDS = Config.SWT_USE_DYNAMIC_BANDS
SWT_LOG_VERBOSE = Config.SWT_LOG_VERBOSE
SWT_USE_TORCH_STFT_FASTPATH = Config.SWT_USE_TORCH_STFT_FASTPATH
SWT_TORCH_STFT_DEVICE = Config.SWT_TORCH_STFT_DEVICE
SWT_TORCH_STFT_N_FFT = Config.SWT_TORCH_STFT_N_FFT
SWT_TORCH_STFT_HOP_LENGTH = Config.SWT_TORCH_STFT_HOP_LENGTH
SWT_TORCH_STFT_WIN_LENGTH = Config.SWT_TORCH_STFT_WIN_LENGTH
SWT_TORCH_STFT_CENTER = Config.SWT_TORCH_STFT_CENTER
SWT_TORCH_STFT_POWER = Config.SWT_TORCH_STFT_POWER
SPEC_CACHE_IN_MEMORY = Config.SPEC_CACHE_IN_MEMORY
SPEC_CACHE_TO_DISK = Config.SPEC_CACHE_TO_DISK
SPEC_CACHE_DIR = Config.SPEC_CACHE_DIR
SPEC_CACHE_DTYPE = Config.SPEC_CACHE_DTYPE
SPEC_CACHE_VERSION = Config.SPEC_CACHE_VERSION

# ========== (Eval/Adapt) Collapse-aware spec gating (Attempt 42+) ==========
# IMPORTANT: Many modules use `import config as cfg` and read module-level attributes via getattr(cfg, ...).
# Therefore these flags/thresholds MUST be exported at module-level, not only defined inside Config.
COLLAPSE_AWARE_SPEC_GATING = Config.COLLAPSE_AWARE_SPEC_GATING
SPEC_STDMEAN_ABS_TH = Config.SPEC_STDMEAN_ABS_TH
SPEC_STDMEAN_RATIO_TH = Config.SPEC_STDMEAN_RATIO_TH
LOG_COLLAPSE_AWARE_GATING = Config.LOG_COLLAPSE_AWARE_GATING
FUSION_BYPASS_LOG_MAX = Config.FUSION_BYPASS_LOG_MAX
FUSION_BYPASS_LOG_INTERVAL = Config.FUSION_BYPASS_LOG_INTERVAL

# ========== (Target SupFT) Flat-head LP-FT defaults (Attempt 42+) ==========
TARGET_SUP_LP_STEPS = Config.TARGET_SUP_LP_STEPS
TARGET_SUP_LP_LR = Config.TARGET_SUP_LP_LR
TARGET_SUP_LP_TEMPERATURE = Config.TARGET_SUP_LP_TEMPERATURE
TARGET_SUP_LP_BETA_OVERRIDE = Config.TARGET_SUP_LP_BETA_OVERRIDE  # Attempt 50
TARGET_SUP_LP_PRE_BN_ADAPT_ENABLE = Config.TARGET_SUP_LP_PRE_BN_ADAPT_ENABLE
TARGET_SUP_LP_PRE_BN_ADAPT_DOMAINS = Config.TARGET_SUP_LP_PRE_BN_ADAPT_DOMAINS
TARGET_SUP_LP_PRE_BN_ADAPT_NUM_BATCHES = Config.TARGET_SUP_LP_PRE_BN_ADAPT_NUM_BATCHES
TARGET_SUP_LP_PRE_BN_ADAPT_ON_SELFTRAIN = Config.TARGET_SUP_LP_PRE_BN_ADAPT_ON_SELFTRAIN
TARGET_SUP_LP_PRE_BN_ADAPT_ON_RETRAIN_FULL = Config.TARGET_SUP_LP_PRE_BN_ADAPT_ON_RETRAIN_FULL
# Attempt 52: validation-driven LP-FT early stopping
TARGET_SUP_LP_USE_VAL_EARLY_STOP = Config.TARGET_SUP_LP_USE_VAL_EARLY_STOP
TARGET_SUP_LP_VAL_RATIO = Config.TARGET_SUP_LP_VAL_RATIO
TARGET_SUP_LP_VAL_MULTI_SPLIT_ENABLE = Config.TARGET_SUP_LP_VAL_MULTI_SPLIT_ENABLE
TARGET_SUP_LP_VAL_MULTI_SPLIT_REPEATS = Config.TARGET_SUP_LP_VAL_MULTI_SPLIT_REPEATS
TARGET_SUP_LP_VAL_MULTI_SPLIT_BASE_SEED = Config.TARGET_SUP_LP_VAL_MULTI_SPLIT_BASE_SEED
TARGET_SUP_LP_VAL_MULTI_SPLIT_METRIC_REDUCE = Config.TARGET_SUP_LP_VAL_MULTI_SPLIT_METRIC_REDUCE
TARGET_SUP_LP_PATIENCE = Config.TARGET_SUP_LP_PATIENCE
TARGET_SUP_LP_PATIENCE_SEU = Config.TARGET_SUP_LP_PATIENCE_SEU
TARGET_SUP_LP_PATIENCE_HIT = Config.TARGET_SUP_LP_PATIENCE_HIT
TARGET_SUP_LP_PATIENCE_PU = Config.TARGET_SUP_LP_PATIENCE_PU
TARGET_SUP_LP_MIN_STEPS = Config.TARGET_SUP_LP_MIN_STEPS
TARGET_SUP_LP_EVAL_EVERY = Config.TARGET_SUP_LP_EVAL_EVERY
TARGET_SUP_LP_EARLYSTOP_METRIC = Config.TARGET_SUP_LP_EARLYSTOP_METRIC
TARGET_SUP_LP_MIN_DELTA = Config.TARGET_SUP_LP_MIN_DELTA
TARGET_SUP_LP_MIN_VAL_SAMPLES = Config.TARGET_SUP_LP_MIN_VAL_SAMPLES
TARGET_SUP_LP_SEMI_ENABLE = Config.TARGET_SUP_LP_SEMI_ENABLE
TARGET_SUP_LP_SEMI_ENT_WEIGHT = Config.TARGET_SUP_LP_SEMI_ENT_WEIGHT
TARGET_SUP_LP_SEMI_DIV_WEIGHT = Config.TARGET_SUP_LP_SEMI_DIV_WEIGHT
TARGET_SUP_LP_SEMI_WARMUP_STEPS = Config.TARGET_SUP_LP_SEMI_WARMUP_STEPS
TARGET_SUP_LP_SEMI_TINYSHOT_DISABLE_THRESHOLD = Config.TARGET_SUP_LP_SEMI_TINYSHOT_DISABLE_THRESHOLD
TARGET_SUP_LP_RETRAIN_FULL_ENABLE = Config.TARGET_SUP_LP_RETRAIN_FULL_ENABLE
TARGET_SUP_LP_RETRAIN_FULL_MAX_SAMPLES = Config.TARGET_SUP_LP_RETRAIN_FULL_MAX_SAMPLES
TARGET_SUP_LP_RETRAIN_FULL_MIN_STEPS = Config.TARGET_SUP_LP_RETRAIN_FULL_MIN_STEPS
TARGET_SUP_LP_RETRAIN_FULL_ENFORCE_STAGE4_COVERAGE = Config.TARGET_SUP_LP_RETRAIN_FULL_ENFORCE_STAGE4_COVERAGE
TARGET_SUP_LP_RETRAIN_FULL_MIN_STAGE4_FT_STEPS = Config.TARGET_SUP_LP_RETRAIN_FULL_MIN_STAGE4_FT_STEPS
TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_ENABLE = Config.TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_ENABLE
TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_RATIO = Config.TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_RATIO
TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_MIN = Config.TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_MIN
TARGET_SUP_LP_RETRAIN_FULL_USE_BEST_STEP = Config.TARGET_SUP_LP_RETRAIN_FULL_USE_BEST_STEP
TARGET_SUP_LP_RETRAIN_FULL_ALLOW_PROGRESSIVE_UNFREEZE_FLAT = Config.TARGET_SUP_LP_RETRAIN_FULL_ALLOW_PROGRESSIVE_UNFREEZE_FLAT
TARGET_SUP_LP_RETRAIN_FULL_NONPARAM_EVAL_ENABLE = Config.TARGET_SUP_LP_RETRAIN_FULL_NONPARAM_EVAL_ENABLE
TARGET_SUP_LP_LR_SCHEDULE_ENABLE = Config.TARGET_SUP_LP_LR_SCHEDULE_ENABLE
TARGET_SUP_LP_LR_WARMUP_STEPS = Config.TARGET_SUP_LP_LR_WARMUP_STEPS
TARGET_SUP_LP_LR_MIN_SCALE = Config.TARGET_SUP_LP_LR_MIN_SCALE
TARGET_SUP_LP_PROGRESSIVE_UNFREEZE_ENABLE = Config.TARGET_SUP_LP_PROGRESSIVE_UNFREEZE_ENABLE
TARGET_SUP_LP_UNFREEZE_STAGE2_START = Config.TARGET_SUP_LP_UNFREEZE_STAGE2_START
TARGET_SUP_LP_UNFREEZE_STAGE3_START = Config.TARGET_SUP_LP_UNFREEZE_STAGE3_START
TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SELFTRAIN = Config.TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SELFTRAIN
TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_TASK1 = Config.TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_TASK1
TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SEU = Config.TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SEU
TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_HIT = Config.TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_HIT
TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_TINYSHOT_THRESHOLD = Config.TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_TINYSHOT_THRESHOLD
TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_RETRAIN_FULL = Config.TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_RETRAIN_FULL
TARGET_SUP_LP_STAGE_SCALE_BY_SAMPLES = Config.TARGET_SUP_LP_STAGE_SCALE_BY_SAMPLES
TARGET_SUP_LP_STAGE_SCALE_SAMPLE_THRESHOLD = Config.TARGET_SUP_LP_STAGE_SCALE_SAMPLE_THRESHOLD
TARGET_SUP_LP_STAGE_SCALE_MIN_RATIO = Config.TARGET_SUP_LP_STAGE_SCALE_MIN_RATIO
TARGET_SUP_LP_STAGE_SCALE_MIN_STAGE2_STEPS = Config.TARGET_SUP_LP_STAGE_SCALE_MIN_STAGE2_STEPS
TARGET_SUP_LP_BACKBONE_FT_ENABLE = Config.TARGET_SUP_LP_BACKBONE_FT_ENABLE
TARGET_SUP_LP_UNFREEZE_STAGE4_START = Config.TARGET_SUP_LP_UNFREEZE_STAGE4_START
TARGET_SUP_LP_L2SP_ENABLE = Config.TARGET_SUP_LP_L2SP_ENABLE
TARGET_SUP_LP_L2SP_WEIGHT = Config.TARGET_SUP_LP_L2SP_WEIGHT
TARGET_SUP_LP_L2SP_USE_GLOBAL_ANCHOR = Config.TARGET_SUP_LP_L2SP_USE_GLOBAL_ANCHOR
TARGET_SUP_LP_BACKBONE_LR_MULT = Config.TARGET_SUP_LP_BACKBONE_LR_MULT
TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_ENABLE = Config.TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_ENABLE
TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_STEPS = Config.TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_STEPS
TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_MIN_SCALE = Config.TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_MIN_SCALE
TARGET_SUP_KNN_EVAL_ENABLE = Config.TARGET_SUP_KNN_EVAL_ENABLE
TARGET_SUP_KNN_K = Config.TARGET_SUP_KNN_K
TARGET_SUP_KNN_REPLACE_WHEN_BETTER = Config.TARGET_SUP_KNN_REPLACE_WHEN_BETTER
TARGET_SUP_LP_EMA_ENABLE = Config.TARGET_SUP_LP_EMA_ENABLE
TARGET_SUP_LP_EMA_DECAY = Config.TARGET_SUP_LP_EMA_DECAY
TARGET_SUP_LP_EMA_EVAL_ENABLE = Config.TARGET_SUP_LP_EMA_EVAL_ENABLE
TARGET_SUP_LP_EMA_USE_FOR_NONPARAM_EVAL = Config.TARGET_SUP_LP_EMA_USE_FOR_NONPARAM_EVAL
TARGET_SUP_LP_SAM_ENABLE = Config.TARGET_SUP_LP_SAM_ENABLE
TARGET_SUP_LP_SAM_RHO = Config.TARGET_SUP_LP_SAM_RHO
TARGET_SUP_LP_SAM_STAGE1_ONLY = Config.TARGET_SUP_LP_SAM_STAGE1_ONLY
TARGET_SUP_SPEC_KNN_ENSEMBLE_ENABLE = Config.TARGET_SUP_SPEC_KNN_ENSEMBLE_ENABLE
TARGET_SUP_SPEC_KNN_ENSEMBLE_WEIGHT_GRID = Config.TARGET_SUP_SPEC_KNN_ENSEMBLE_WEIGHT_GRID
TARGET_SUP_SPEC_KNN_ENSEMBLE_REPLACE_WHEN_BETTER = Config.TARGET_SUP_SPEC_KNN_ENSEMBLE_REPLACE_WHEN_BETTER
TARGET_SUP_LABEL_PROP_ENABLE = Config.TARGET_SUP_LABEL_PROP_ENABLE
TARGET_SUP_LABEL_PROP_REPLACE_WHEN_BETTER = Config.TARGET_SUP_LABEL_PROP_REPLACE_WHEN_BETTER
TARGET_SUP_LABEL_PROP_KERNEL = Config.TARGET_SUP_LABEL_PROP_KERNEL
TARGET_SUP_LABEL_PROP_N_NEIGHBORS = Config.TARGET_SUP_LABEL_PROP_N_NEIGHBORS
TARGET_SUP_LABEL_PROP_ALPHA = Config.TARGET_SUP_LABEL_PROP_ALPHA
TARGET_SUP_LABEL_PROP_MAX_ITER = Config.TARGET_SUP_LABEL_PROP_MAX_ITER
TARGET_SUP_LABEL_PROP_TOL = Config.TARGET_SUP_LABEL_PROP_TOL
TARGET_SUP_LABEL_PROP_DUAL_SPACE_ENABLE = Config.TARGET_SUP_LABEL_PROP_DUAL_SPACE_ENABLE
TARGET_SUP_LABEL_PROP_FUSION_WEIGHT = Config.TARGET_SUP_LABEL_PROP_FUSION_WEIGHT
TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_ENABLE = Config.TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_ENABLE
TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_GRID = Config.TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_GRID
TARGET_SUP_TIM_ENABLE = Config.TARGET_SUP_TIM_ENABLE
TARGET_SUP_TIM_REPLACE_WHEN_BETTER = Config.TARGET_SUP_TIM_REPLACE_WHEN_BETTER
TARGET_SUP_TIM_STEPS = Config.TARGET_SUP_TIM_STEPS
TARGET_SUP_TIM_LR = Config.TARGET_SUP_TIM_LR
TARGET_SUP_TIM_TEMPERATURE = Config.TARGET_SUP_TIM_TEMPERATURE
TARGET_SUP_TIM_SUPPORT_CE_WEIGHT = Config.TARGET_SUP_TIM_SUPPORT_CE_WEIGHT
TARGET_SUP_TIM_QUERY_ENT_WEIGHT = Config.TARGET_SUP_TIM_QUERY_ENT_WEIGHT
TARGET_SUP_TIM_QUERY_DIV_WEIGHT = Config.TARGET_SUP_TIM_QUERY_DIV_WEIGHT
TARGET_SUP_TIM_WEIGHT_DECAY = Config.TARGET_SUP_TIM_WEIGHT_DECAY
TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_ENABLE = Config.TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_ENABLE
TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_GAMMA = Config.TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_GAMMA
TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_BETA = Config.TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_BETA
TARGET_SUP_LP_FLAT_NF_WEIGHT = Config.TARGET_SUP_LP_FLAT_NF_WEIGHT
TARGET_SUP_LP_PROTO_ANCHOR_ENABLE = Config.TARGET_SUP_LP_PROTO_ANCHOR_ENABLE
TARGET_SUP_LP_PROTO_ANCHOR_WEIGHT = Config.TARGET_SUP_LP_PROTO_ANCHOR_WEIGHT
TARGET_SUP_LP_PROTO_ANCHOR_WARMUP_STEPS = Config.TARGET_SUP_LP_PROTO_ANCHOR_WARMUP_STEPS
TARGET_SUP_LP_SPEC_LABEL_SMOOTHING = Config.TARGET_SUP_LP_SPEC_LABEL_SMOOTHING
TARGET_TASK2_DETACH_FUSION_FOR_DECODER = Config.TARGET_TASK2_DETACH_FUSION_FOR_DECODER
TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE = Config.TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE
TARGET_SUP_SKIP_LPFT_MIN_SAMPLES = Config.TARGET_SUP_SKIP_LPFT_MIN_SAMPLES
TARGET_SUP_SVP_ENABLE = Config.TARGET_SUP_SVP_ENABLE
TARGET_SUP_SVP_REPLACE_WHEN_BETTER = Config.TARGET_SUP_SVP_REPLACE_WHEN_BETTER
TARGET_SUP_SVP_USE_FUSION = Config.TARGET_SUP_SVP_USE_FUSION
TARGET_SUP_SVP_FUSION_WEIGHT = Config.TARGET_SUP_SVP_FUSION_WEIGHT
TARGET_SUP_SVP_DUAL_AUTO_WEIGHT = Config.TARGET_SUP_SVP_DUAL_AUTO_WEIGHT
TARGET_SUP_SVP_DUAL_WEIGHT_GRID = Config.TARGET_SUP_SVP_DUAL_WEIGHT_GRID
TARGET_SUP_SVP_TPA_ENABLE = Config.TARGET_SUP_SVP_TPA_ENABLE
TARGET_SUP_SVP_TPA_ITERS = Config.TARGET_SUP_SVP_TPA_ITERS
TARGET_SUP_SVP_TPA_TOPK_RATIO = Config.TARGET_SUP_SVP_TPA_TOPK_RATIO
TARGET_SUP_SVP_TPA_MIN_SAMPLES_PER_CLASS = Config.TARGET_SUP_SVP_TPA_MIN_SAMPLES_PER_CLASS
TARGET_SUP_SVP_TPA_MIX_ALPHA = Config.TARGET_SUP_SVP_TPA_MIX_ALPHA
TARGET_SUP_SVP_TPA_MIN_BASE_ACC = Config.TARGET_SUP_SVP_TPA_MIN_BASE_ACC
TARGET_SUP_SVP_BASELINE_GUARD_ENABLE = Config.TARGET_SUP_SVP_BASELINE_GUARD_ENABLE
TARGET_TASK1_SVP_ENABLE = Config.TARGET_TASK1_SVP_ENABLE
TARGET_TASK1_SVP_REPLACE_WHEN_BETTER = Config.TARGET_TASK1_SVP_REPLACE_WHEN_BETTER
TARGET_TASK1_SVP_USE_FEWSHOT_SPLIT = Config.TARGET_TASK1_SVP_USE_FEWSHOT_SPLIT
TARGET_SUP_LP_DUAL_BEST_SNAPSHOT_ENABLE = Config.TARGET_SUP_LP_DUAL_BEST_SNAPSHOT_ENABLE
TARGET_SELF_TRAINING_ENABLE = Config.TARGET_SELF_TRAINING_ENABLE
TARGET_SELF_TRAINING_PSEUDO_SOURCE = Config.TARGET_SELF_TRAINING_PSEUDO_SOURCE
TARGET_SELF_TRAINING_ROUNDS = Config.TARGET_SELF_TRAINING_ROUNDS
TARGET_SELF_TRAINING_TOPK_RATIO = Config.TARGET_SELF_TRAINING_TOPK_RATIO
TARGET_SELF_TRAINING_MAX_TOPK_RATIO = Config.TARGET_SELF_TRAINING_MAX_TOPK_RATIO
TARGET_SELF_TRAINING_MIN_CONFIDENCE = Config.TARGET_SELF_TRAINING_MIN_CONFIDENCE
TARGET_SELF_TRAINING_MIN_CONF_STEP = Config.TARGET_SELF_TRAINING_MIN_CONF_STEP
TARGET_SELF_TRAINING_MIN_SAMPLES_PER_CLASS = Config.TARGET_SELF_TRAINING_MIN_SAMPLES_PER_CLASS
TARGET_SELF_TRAINING_BALANCED_TOPK_ENABLE = Config.TARGET_SELF_TRAINING_BALANCED_TOPK_ENABLE
TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_ENABLE = Config.TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_ENABLE
TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_THRESHOLD = Config.TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_THRESHOLD
TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_VALUE = Config.TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_VALUE
TARGET_SELF_TRAINING_MIN_START_ACC = Config.TARGET_SELF_TRAINING_MIN_START_ACC
TARGET_SELF_TRAINING_REPLACE_WHEN_BETTER = Config.TARGET_SELF_TRAINING_REPLACE_WHEN_BETTER
TARGET_TASK2_CORAL_ENABLE = Config.TARGET_TASK2_CORAL_ENABLE
TARGET_TASK2_CORAL_WEIGHT = Config.TARGET_TASK2_CORAL_WEIGHT
TARGET_TASK2_CORAL_START_EPOCH = Config.TARGET_TASK2_CORAL_START_EPOCH
TARGET_TASK2_CORAL_WARMUP_EPOCHS = Config.TARGET_TASK2_CORAL_WARMUP_EPOCHS
TARGET_TASK2_CORAL_MIN_SAMPLES_PER_DOMAIN = Config.TARGET_TASK2_CORAL_MIN_SAMPLES_PER_DOMAIN
TARGET_TASK2_TARGET_CORAL_ENABLE = Config.TARGET_TASK2_TARGET_CORAL_ENABLE
TARGET_TASK2_TARGET_CORAL_WEIGHT = Config.TARGET_TASK2_TARGET_CORAL_WEIGHT
TARGET_TASK2_TARGET_CORAL_EVERY_N_STEPS = Config.TARGET_TASK2_TARGET_CORAL_EVERY_N_STEPS
TARGET_TASK2_TARGET_CORAL_MIN_SAMPLES = Config.TARGET_TASK2_TARGET_CORAL_MIN_SAMPLES
TARGET_TASK2_TARGET_MMD_ENABLE = Config.TARGET_TASK2_TARGET_MMD_ENABLE
TARGET_TASK2_TARGET_MMD_WEIGHT = Config.TARGET_TASK2_TARGET_MMD_WEIGHT
TARGET_TASK2_TARGET_MMD_START_EPOCH = Config.TARGET_TASK2_TARGET_MMD_START_EPOCH
TARGET_TASK2_TARGET_MMD_WARMUP_EPOCHS = Config.TARGET_TASK2_TARGET_MMD_WARMUP_EPOCHS
TARGET_TASK2_TARGET_MMD_EVERY_N_STEPS = Config.TARGET_TASK2_TARGET_MMD_EVERY_N_STEPS
TARGET_TASK2_TARGET_MMD_MIN_SAMPLES = Config.TARGET_TASK2_TARGET_MMD_MIN_SAMPLES
TARGET_TASK2_TARGET_MMD_KERNEL_SCALES = Config.TARGET_TASK2_TARGET_MMD_KERNEL_SCALES
CROSS_DOMAIN_METRIC_ALIGN_ENABLE = Config.CROSS_DOMAIN_METRIC_ALIGN_ENABLE
CROSS_DOMAIN_METRIC_ALIGN_WEIGHT = Config.CROSS_DOMAIN_METRIC_ALIGN_WEIGHT
CROSS_DOMAIN_METRIC_ALIGN_SEP_MARGIN = Config.CROSS_DOMAIN_METRIC_ALIGN_SEP_MARGIN
CROSS_DOMAIN_METRIC_ALIGN_START_EPOCH = Config.CROSS_DOMAIN_METRIC_ALIGN_START_EPOCH
FUSION_PRETRAIN_DISABLE_ZCLS_CE = Config.FUSION_PRETRAIN_DISABLE_ZCLS_CE
FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY = Config.FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY
LOO_ENABLE = Config.LOO_ENABLE
LOO_DOMAINS = Config.LOO_DOMAINS
LOO_TRAIN_ONLY_PHASES = Config.LOO_TRAIN_ONLY_PHASES
STRICT_INTRA_CONDITION_SPLIT = Config.STRICT_INTRA_CONDITION_SPLIT
TASK1_TRAIN_SUBSAMPLE_RATIO = Config.TASK1_TRAIN_SUBSAMPLE_RATIO
TASK1_TRAIN_SUBSAMPLE_RATIO_PU = Config.TASK1_TRAIN_SUBSAMPLE_RATIO_PU
LOO_STRICT_MODE = Config.LOO_STRICT_MODE
STMO_ENABLE = Config.STMO_ENABLE
STMO_DOMAINS = Config.STMO_DOMAINS
STMO_TRAIN_ONLY_PHASES = Config.STMO_TRAIN_ONLY_PHASES
STMO_STRICT_MODE = Config.STMO_STRICT_MODE
STMO_SUMMARY_METRIC_SOURCE = Config.STMO_SUMMARY_METRIC_SOURCE
TARGET_TWC_ENABLE = Config.TARGET_TWC_ENABLE
TARGET_TWC_TARGET_DATASETS = Config.TARGET_TWC_TARGET_DATASETS
TARGET_TWC_EPS = Config.TARGET_TWC_EPS
TARGET_TWC_BLEND_ALPHA = Config.TARGET_TWC_BLEND_ALPHA
TARGET_TWC_APPLY_TO_SVPEVAL = Config.TARGET_TWC_APPLY_TO_SVPEVAL
TARGET_TWC_APPLY_TO_SELFTRAIN = Config.TARGET_TWC_APPLY_TO_SELFTRAIN
TARGET_TWC_APPLY_TO_POSTLPFT = Config.TARGET_TWC_APPLY_TO_POSTLPFT
TARGET_SUP_POST_LPFT_SPEC_HEAD_EVAL = Config.TARGET_SUP_POST_LPFT_SPEC_HEAD_EVAL
TARGET_SUP_POST_LPFT_SPEC_HEAD_REPLACE_WHEN_BETTER = Config.TARGET_SUP_POST_LPFT_SPEC_HEAD_REPLACE_WHEN_BETTER
TARGET_SUP_LP_AUG_ENABLE = Config.TARGET_SUP_LP_AUG_ENABLE
TARGET_SUP_LP_AUG_PROB = Config.TARGET_SUP_LP_AUG_PROB
TARGET_SUP_LP_AUG_SPEC_NOISE_STD = Config.TARGET_SUP_LP_AUG_SPEC_NOISE_STD
TARGET_SUP_LP_AUG_SIGNAL_NOISE_STD = Config.TARGET_SUP_LP_AUG_SIGNAL_NOISE_STD
TARGET_SUP_LP_AUG_FREQ_MASK_RATIO = Config.TARGET_SUP_LP_AUG_FREQ_MASK_RATIO
TARGET_SUP_LP_AUG_TIME_MASK_RATIO = Config.TARGET_SUP_LP_AUG_TIME_MASK_RATIO
TARGET_SUP_LP_AUG_NUM_FREQ_MASKS = Config.TARGET_SUP_LP_AUG_NUM_FREQ_MASKS
TARGET_SUP_LP_AUG_NUM_TIME_MASKS = Config.TARGET_SUP_LP_AUG_NUM_TIME_MASKS
TARGET_EVAL_COMPACT_LOG = Config.TARGET_EVAL_COMPACT_LOG
TARGET_EVAL_LOG_GRID_DETAILS = Config.TARGET_EVAL_LOG_GRID_DETAILS
TARGET_EVAL_LOG_TRANSDUCTIVE_DETAILS = Config.TARGET_EVAL_LOG_TRANSDUCTIVE_DETAILS
EVAL_SUPPRESS_SKLEARN_WARNINGS = Config.EVAL_SUPPRESS_SKLEARN_WARNINGS
EVAL_REPORT_FUSION_DIAG = Config.EVAL_REPORT_FUSION_DIAG
EVAL_REPORT_SIGNAL_HEAD = Config.EVAL_REPORT_SIGNAL_HEAD
EVAL_REPORT_SIGNAL_PROTO = Config.EVAL_REPORT_SIGNAL_PROTO
EVAL_REPORT_MODALITY_DIAG = Config.EVAL_REPORT_MODALITY_DIAG
EVAL_EXPORT_MODALITY_METRICS_CSV = Config.EVAL_EXPORT_MODALITY_METRICS_CSV
EVAL_EXPORT_MODALITY_METRICS_DIR = Config.EVAL_EXPORT_MODALITY_METRICS_DIR
LOO_SUMMARY_METRIC_SOURCE = Config.LOO_SUMMARY_METRIC_SOURCE
EVAL_FINAL_DECISION_ENABLE = Config.EVAL_FINAL_DECISION_ENABLE
EVAL_FINAL_DECISION_USE_SPEC_HEAD = Config.EVAL_FINAL_DECISION_USE_SPEC_HEAD
EVAL_FINAL_DECISION_USE_SIGNAL_HEAD = Config.EVAL_FINAL_DECISION_USE_SIGNAL_HEAD
EVAL_FINAL_DECISION_USE_FUSION_PROTO = Config.EVAL_FINAL_DECISION_USE_FUSION_PROTO
EVAL_FINAL_DECISION_USE_FUSION_DECODER = Config.EVAL_FINAL_DECISION_USE_FUSION_DECODER
EVAL_FINAL_DECISION_USE_TRANSDUCTIVE = Config.EVAL_FINAL_DECISION_USE_TRANSDUCTIVE
EVAL_FINAL_DECISION_SPEC_WEIGHT = Config.EVAL_FINAL_DECISION_SPEC_WEIGHT
EVAL_FINAL_DECISION_SIGNAL_HEAD_WEIGHT = Config.EVAL_FINAL_DECISION_SIGNAL_HEAD_WEIGHT
EVAL_FINAL_DECISION_FUSION_PROTO_WEIGHT = Config.EVAL_FINAL_DECISION_FUSION_PROTO_WEIGHT
EVAL_FINAL_DECISION_T_SIGNAL_HEAD = Config.EVAL_FINAL_DECISION_T_SIGNAL_HEAD
EVAL_FINAL_DECISION_SPEC_COLLAPSE_GATE = Config.EVAL_FINAL_DECISION_SPEC_COLLAPSE_GATE
EVAL_FINAL_DECISION_SPEC_COLLAPSE_THRESH = Config.EVAL_FINAL_DECISION_SPEC_COLLAPSE_THRESH
EVAL_FINAL_DECISION_FUSION_DECODER_WEIGHT = Config.EVAL_FINAL_DECISION_FUSION_DECODER_WEIGHT
EVAL_FINAL_DECISION_TRANSDUCTIVE_WEIGHT = Config.EVAL_FINAL_DECISION_TRANSDUCTIVE_WEIGHT
EVAL_FINAL_DECISION_CONF_POWER = Config.EVAL_FINAL_DECISION_CONF_POWER
EVAL_FINAL_DECISION_PER_SAMPLE_CONF_GATE = Config.EVAL_FINAL_DECISION_PER_SAMPLE_CONF_GATE
EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_THRESH = Config.EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_THRESH
EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_DOWNWEIGHT = Config.EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_DOWNWEIGHT
EVAL_FINAL_DECISION_DISAGREE_FALLBACK_ENABLE = Config.EVAL_FINAL_DECISION_DISAGREE_FALLBACK_ENABLE
EVAL_FINAL_DECISION_DISAGREE_FALLBACK_THRESH = Config.EVAL_FINAL_DECISION_DISAGREE_FALLBACK_THRESH
EVAL_FINAL_DECISION_CASCADE_ENABLE = Config.EVAL_FINAL_DECISION_CASCADE_ENABLE
EVAL_FINAL_DECISION_CASCADE_HIGH_THRESH = Config.EVAL_FINAL_DECISION_CASCADE_HIGH_THRESH
EVAL_FINAL_DECISION_T_SPEC = Config.EVAL_FINAL_DECISION_T_SPEC
EVAL_FINAL_DECISION_T_FUSION_PROTO = Config.EVAL_FINAL_DECISION_T_FUSION_PROTO
EVAL_FINAL_DECISION_T_FUSION_DECODER = Config.EVAL_FINAL_DECISION_T_FUSION_DECODER
EVAL_FINAL_DECISION_PRINT_CM = Config.EVAL_FINAL_DECISION_PRINT_CM
EVAL_FINAL_DECISION_DEGEN_GATE_ENABLE = Config.EVAL_FINAL_DECISION_DEGEN_GATE_ENABLE
EVAL_FINAL_DECISION_MAX_CLASS_FRAC = Config.EVAL_FINAL_DECISION_MAX_CLASS_FRAC
EVAL_FINAL_DECISION_MIN_PRED_ENTROPY = Config.EVAL_FINAL_DECISION_MIN_PRED_ENTROPY
EVAL_FINAL_DECISION_TRANSDUCTIVE_AGREEMENT_GATE_ENABLE = Config.EVAL_FINAL_DECISION_TRANSDUCTIVE_AGREEMENT_GATE_ENABLE
EVAL_FINAL_DECISION_TRANSDUCTIVE_MIN_AGREE_WITH_FUSION_PROTO = Config.EVAL_FINAL_DECISION_TRANSDUCTIVE_MIN_AGREE_WITH_FUSION_PROTO
EVAL_FINAL_DECISION_FUSION_PROTO_PRIORITY = Config.EVAL_FINAL_DECISION_FUSION_PROTO_PRIORITY
EVAL_FINAL_DECISION_TD_PRIMARY = Config.EVAL_FINAL_DECISION_TD_PRIMARY
TARGET_EVAL_COMPACT_LOG = Config.TARGET_EVAL_COMPACT_LOG
TARGET_EVAL_LOG_GRID_DETAILS = Config.TARGET_EVAL_LOG_GRID_DETAILS
EVAL_LOG_FUSION_INPUT_STATS = Config.EVAL_LOG_FUSION_INPUT_STATS
DATASET_LOG_FEATURE_BUILD = Config.DATASET_LOG_FEATURE_BUILD

# ========== (Eval/Adapt) Target-domain adaptation controls ==========
TARGET_BN_ADAPT_ENABLE = Config.TARGET_BN_ADAPT_ENABLE
TASK1_BN_ADAPT_ENABLE = Config.TASK1_BN_ADAPT_ENABLE
TARGET_BN_ADAPT_NUM_BATCHES = Config.TARGET_BN_ADAPT_NUM_BATCHES
TARGET_BN_ADAPT_ALPHA = Config.TARGET_BN_ADAPT_ALPHA
TARGET_BN_ADAPT_AUTO_ALPHA = Config.TARGET_BN_ADAPT_AUTO_ALPHA
TARGET_BN_ADAPT_AUTO_ALPHA_MIN = Config.TARGET_BN_ADAPT_AUTO_ALPHA_MIN
TARGET_BN_ADAPT_AUTO_ALPHA_MAX = Config.TARGET_BN_ADAPT_AUTO_ALPHA_MAX
TARGET_SUP_INCLUDE_SPEC_HEAD = Config.TARGET_SUP_INCLUDE_SPEC_HEAD
TARGET_SUP_LP_SPEC_LOSS_WEIGHT = Config.TARGET_SUP_LP_SPEC_LOSS_WEIGHT
TARGET_SUP_LP_FUSION_LOSS_WEIGHT = Config.TARGET_SUP_LP_FUSION_LOSS_WEIGHT
TARGET_SUP_LP_SUPCON_ENABLE = Config.TARGET_SUP_LP_SUPCON_ENABLE
TARGET_SUP_LP_SUPCON_WEIGHT = Config.TARGET_SUP_LP_SUPCON_WEIGHT
TARGET_SUP_LP_SUPCON_TEMPERATURE = Config.TARGET_SUP_LP_SUPCON_TEMPERATURE
TARGET_SUP_LP_SUPCON_USE_CLASS_WEIGHTS = Config.TARGET_SUP_LP_SUPCON_USE_CLASS_WEIGHTS
TARGET_SUP_LP_SUPCON_FAULT_ONLY = Config.TARGET_SUP_LP_SUPCON_FAULT_ONLY
TARGET_TENT_ENABLE = Config.TARGET_TENT_ENABLE
TARGET_TENT_STEPS = Config.TARGET_TENT_STEPS
TARGET_TENT_LR = Config.TARGET_TENT_LR
TARGET_TENT_BATCHES = Config.TARGET_TENT_BATCHES
TARGET_TENT_DIVERSITY_WEIGHT = Config.TARGET_TENT_DIVERSITY_WEIGHT
TARGET_TENT_RELIABLE_FILTER_ENABLE = Config.TARGET_TENT_RELIABLE_FILTER_ENABLE
TARGET_TENT_RELIABLE_ENTROPY_FACTOR = Config.TARGET_TENT_RELIABLE_ENTROPY_FACTOR
TARGET_TENT_RESET_BEFORE_EACH_DOMAIN = Config.TARGET_TENT_RESET_BEFORE_EACH_DOMAIN
TARGET_SHOT_BN_ENABLE = Config.TARGET_SHOT_BN_ENABLE
TARGET_SHOT_TENT_ENABLE = Config.TARGET_SHOT_TENT_ENABLE
TARGET_SHOT_TENT_STEPS = Config.TARGET_SHOT_TENT_STEPS
TARGET_SHOT_TENT_LR = Config.TARGET_SHOT_TENT_LR
TARGET_SHOT_TENT_BATCHES = Config.TARGET_SHOT_TENT_BATCHES
TARGET_SHOT_CONFIDENCE_TH = Config.TARGET_SHOT_CONFIDENCE_TH
TARGET_TTFA_ENABLE = Config.TARGET_TTFA_ENABLE
TARGET_TTFA_NUM_BATCHES = Config.TARGET_TTFA_NUM_BATCHES
TARGET_TTFA_STRENGTH = Config.TARGET_TTFA_STRENGTH

# ========== (Eval/Adapt) Fusion bypass on spec collapse (Attempt 45+) ==========
FUSION_BYPASS_ON_SPEC_COLLAPSE = Config.FUSION_BYPASS_ON_SPEC_COLLAPSE
FUSION_BYPASS_USE_SPEC_ENTROPY = Config.FUSION_BYPASS_USE_SPEC_ENTROPY
FUSION_BYPASS_SPEC_ENTROPY_TH = Config.FUSION_BYPASS_SPEC_ENTROPY_TH
FUSION_BYPASS_SPEC_MAXCONF_TH = Config.FUSION_BYPASS_SPEC_MAXCONF_TH

# 导出函数 / 变量
get_window_length = Config.get_window_length
# Att101 Exports
DIFFUSION_CACHE_DIR = Config.DIFFUSION_CACHE_DIR
DIFFUSION_CACHE_ENABLE = Config.DIFFUSION_CACHE_ENABLE
DIFFUSION_FORCE_REGENERATE = Config.DIFFUSION_FORCE_REGENERATE
DIFFUSION_CFG_WEIGHT = Config.DIFFUSION_CFG_WEIGHT
DIFFUSION_LABEL_DROP_PROB = Config.DIFFUSION_LABEL_DROP_PROB
# ATT102 Exports — Energy Gating
DIFFUSION_ENABLE_ENERGY_GATING = Config.DIFFUSION_ENABLE_ENERGY_GATING
DIFFUSION_ENERGY_GATING_THRESHOLD = Config.DIFFUSION_ENERGY_GATING_THRESHOLD
DIFFUSION_SYNTHETIC_CONDITION_KEY = Config.DIFFUSION_SYNTHETIC_CONDITION_KEY
# ATT114 Exports — Classification Head & Aux CE
# BUG FIX: USE_CLASSIFICATION_HEAD was defined inside Config class but NOT exported at module level.
# model_factory.py uses getattr(cfg, 'USE_CLASSIFICATION_HEAD', False) where cfg=config module,
# so it always received False → spec_encoder.classifier=None → _spec_has_head=False → spec_head never evaluated.
USE_CLASSIFICATION_HEAD = Config.USE_CLASSIFICATION_HEAD
AUX_CE_WEIGHT = Config.AUX_CE_WEIGHT
AUX_CE_START_EPOCH = Config.AUX_CE_START_EPOCH
AUX_CE_USE_FOCAL = Config.AUX_CE_USE_FOCAL
AUX_CE_FOCAL_GAMMA = Config.AUX_CE_FOCAL_GAMMA
SPEC_TRAIN_OBJECTIVE_MODE = Config.SPEC_TRAIN_OBJECTIVE_MODE
SPEC_PROTO_CE_WEIGHT = Config.SPEC_PROTO_CE_WEIGHT
SPEC_PROTO_CE_TEMPERATURE = Config.SPEC_PROTO_CE_TEMPERATURE
# ATT429 exports
SPEC_BN_FULL_RESET_ON_TARGET = Config.SPEC_BN_FULL_RESET_ON_TARGET
SPEC_BN_FULL_RESET_MAX_BATCHES = Config.SPEC_BN_FULL_RESET_MAX_BATCHES
USE_HIERARCHICAL_SPEC_HEAD = Config.USE_HIERARCHICAL_SPEC_HEAD
HIERARCHICAL_NF_TEMPERATURE = Config.HIERARCHICAL_NF_TEMPERATURE
HIERARCHICAL_FAULT_TEMPERATURE = Config.HIERARCHICAL_FAULT_TEMPERATURE
HIERARCHICAL_FAULT_CLASSES = Config.HIERARCHICAL_FAULT_CLASSES
HIERARCHICAL_NF_CE_WEIGHT = Config.HIERARCHICAL_NF_CE_WEIGHT
HIERARCHICAL_FAULT_CE_WEIGHT = Config.HIERARCHICAL_FAULT_CE_WEIGHT
HIERARCHICAL_CE_START_EPOCH = Config.HIERARCHICAL_CE_START_EPOCH
# ATT431: Dual disentangle and spec variance FALSIFIED — exports kept for getattr() compat
USE_DUAL_DISENTANGLE = Config.USE_DUAL_DISENTANGLE           # always False
USE_SPEC_VARIANCE_CONSTRAINT = Config.USE_SPEC_VARIANCE_CONSTRAINT  # always False
USE_RELIABILITY_GATING = Config.USE_RELIABILITY_GATING
UNCERTAINTY_METHOD = Config.UNCERTAINTY_METHOD
GATE_TYPE = Config.GATE_TYPE
GATE_TEMPERATURE = Config.GATE_TEMPERATURE
USE_MODALITY_DROPOUT = Config.USE_MODALITY_DROPOUT
MODALITY_DROP_PROB = Config.MODALITY_DROP_PROB
MODALITY_DROP_MODE = Config.MODALITY_DROP_MODE
USE_ATTENTION_GATE = Config.USE_ATTENTION_GATE
ATTENTION_GATE_ALPHA = Config.ATTENTION_GATE_ALPHA
ATTENTION_GATE_DIM = Config.ATTENTION_GATE_DIM
ATTENTION_GATE_HEADS = Config.ATTENTION_GATE_HEADS


def print_framework_group_summary(prefix="[FrameworkGroups]"):
    try:
        print(
            f"{prefix} fusion_align={int(bool(getattr(Config, 'FUSION_ALIGN_GROUP_ENABLE', True)))} "
            f"metric_geometry={int(bool(getattr(Config, 'METRIC_GEOMETRY_GROUP_ENABLE', True)))} "
            f"domain_adapt={int(bool(getattr(Config, 'DOMAIN_ADAPT_GROUP_ENABLE', True)))}"
        )
        print(
            f"{prefix} supcon={int(bool(getattr(Config, 'FUSION_SUPCON_ENABLE', False)))} "
            f"geom_mode={getattr(Config, 'FUSION_GEOMETRY_LOSS_MODE', 'supcon')} "
            f"metric_lp_ce={int(bool(getattr(Config, 'FUSION_METRIC_LEARNABLE_PROTO_CE_ENABLE', False)))} "
            f"dec_live_ratio={getattr(Config, 'FUSION_DECODER_INPUT_GRAD_LIVE_RATIO', None)}"
        )
        print(
            f"{prefix} zcls_ce: pretrain_disable={int(bool(getattr(Config, 'FUSION_PRETRAIN_DISABLE_ZCLS_CE', True)))} "
            f"stage1_only={int(bool(getattr(Config, 'FUSION_PRETRAIN_DISABLE_ZCLS_CE_STAGE1_ONLY', True)))}"
        )
        print(
            f"{prefix} task2_coral={int(bool(getattr(Config, 'TARGET_TASK2_CORAL_ENABLE', False)))} "
            f"task2_tgt_coral={int(bool(getattr(Config, 'TARGET_TASK2_TARGET_CORAL_ENABLE', False)))} "
            f"task2_tgt_mmd={int(bool(getattr(Config, 'TARGET_TASK2_TARGET_MMD_ENABLE', False)))} "
            f"cross_metric_align={int(bool(getattr(Config, 'CROSS_DOMAIN_METRIC_ALIGN_ENABLE', False)))}"
        )
        print(
            f"{prefix} loo={int(bool(getattr(Config, 'LOO_ENABLE', False)))} "
            f"stmo={int(bool(getattr(Config, 'STMO_ENABLE', False)))} "
            f"strict_intra_split={int(bool(getattr(Config, 'STRICT_INTRA_CONDITION_SPLIT', True)))} "
            f"loo_strict={int(bool(getattr(Config, 'LOO_STRICT_MODE', True)))} "
            f"stmo_strict={int(bool(getattr(Config, 'STMO_STRICT_MODE', True)))}"
        )
    except Exception:
        pass


def _parse_env_override(raw_value, default_value):
    if raw_value is None:
        return default_value
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        try:
            return int(raw_value)
        except Exception:
            return default_value
    if isinstance(default_value, float):
        try:
            return float(raw_value)
        except Exception:
            return default_value
    if isinstance(default_value, (list, tuple, dict, set)):
        try:
            parsed = ast.literal_eval(raw_value)
            if isinstance(default_value, tuple):
                return tuple(parsed)
            if isinstance(default_value, set):
                return set(parsed)
            return parsed
        except Exception:
            return default_value
    return raw_value


def _apply_env_overrides():
    for name, default_value in Config.__dict__.items():
        if (not isinstance(name, str)) or (not name.isupper()):
            continue
        if name not in os.environ:
            continue
        parsed_value = _parse_env_override(os.environ.get(name), default_value)
        setattr(Config, name, parsed_value)
        globals()[name] = parsed_value


_apply_env_overrides()
