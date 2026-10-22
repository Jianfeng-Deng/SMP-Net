"""
模型初始化工厂函数
"""
import torch
import os

from anomalyDetection.anomalyDetector import AnomalyDetector
from models.SignalEncoder import SignalEncoder
from models.SpecEncoder import SpecEncoder
from models.SpecEncoderLight import SpecEncoderLight
from models.SpecEncoderMoE import SpecEncoderMoE
from models.TextEncoder import TextEncoder
from models.clip_wrapper import FrozenCLIP
from models.fusionDecoder import FusionDecoder
from models.fusionModule import GatedAdditiveFusion
from utils.ablation_gating import attach_gate_bank, create_ablation_gate_bank


def _fusion_cls(name: str):
    """Lazy import for optional / legacy fusion backends (may be absent from fusionModule.py)."""
    import models.fusionModule as _fm

    cls = getattr(_fm, name, None)
    if cls is None:
        raise ImportError(
            f"Fusion backend '{name}' is not defined in models.fusionModule. "
            "Current production path uses GatedAdditiveFusion — set USE_GATED_ADDITIVE_FUSION=1 "
            "and disable USE_CROSS_ATTENTION_FUSION / ATT472 / ATT473 / other fusion flags."
        )
    return cls


def create_models(cfg, device):
    """
    创建所有模型组件
    
    Returns:
        tuple: (signal_encoder, spec_encoder, text_encoder, clip_model, 
                fusion_module, fusion_decoder, detector)
    """
    # ATT418-MS: Multi-Scale STFT determines channel count from n_fft list
    _ms_enable = bool(getattr(cfg, "MULTISCALE_STFT_ENABLE", False))
    if _ms_enable:
        num_spec_channels = len(getattr(cfg, "MULTISCALE_STFT_N_FFTS", (64, 128, 256, 512, 1024, 2048)))
        print(f"[ATT418-MS] Multi-Scale STFT: spec channels = {num_spec_channels}")
    else:
        num_spec_channels = 6
    # ATT377-B: extra channel when physical-freq axis uses concat mode
    _pf_enabled = bool(getattr(cfg, "USE_PHYSICAL_FREQ_POSITION_ENCODING", False))
    _pf_mode = str(getattr(cfg, "PHYSICAL_FREQ_POSITION_ENCODING_MODE", "add")).lower()
    _ablation_gating = bool(getattr(cfg, "ABLATION_GATING_ENABLE", False))
    if _pf_enabled and _pf_mode == "concat":
        num_spec_channels += 1
        _gate_note = " (freq axis gated at runtime)" if _ablation_gating else ""
        print(f"[ATT377-B] freq-axis concat mode: spec channels = {num_spec_channels}{_gate_note}")
    # ATT446: WST bypass — WST does NOT add channels to SpecEncoder when bypass is on
    _wst_enable = bool(getattr(cfg, "WST_ENABLE", False))
    _wst_bypass = bool(getattr(cfg, "WST_BYPASS_INJECT", False))
    if _wst_enable and _ms_enable and not _wst_bypass:
        _wst_ch = 1
        num_spec_channels += _wst_ch
        print(f"[ATT446-WST] WST concat mode: +{_wst_ch} ch, total spec channels = {num_spec_channels}")
    elif _wst_enable and _wst_bypass:
        print(f"[ATT446-WST] WST bypass mode: spec channels = {num_spec_channels} (unchanged, WST→TinyWSTEncoder)")
    # ATT464: SCD channel adds 1 to spec channels
    _scd_enable = bool(getattr(cfg, "SCD_CHANNEL_ENABLE", False))
    if _scd_enable:
        num_spec_channels += 1
        print(f"[ATT464-SCD] SCD channel enabled: +1 ch, total spec channels = {num_spec_channels}")
    
    # 域残差配置
    use_domain_residual = cfg.USE_DOMAIN_RESIDUAL
    num_domains = cfg.NUM_DOMAINS
    
    # SignalEncoder
    num_classes = len(cfg.FAULT_LABEL_TO_SEMANTIC)
    use_classification_head = getattr(cfg, 'USE_CLASSIFICATION_HEAD', False)
    signal_encoder = SignalEncoder(
        embed_dim=512, 
        use_domain_residual=use_domain_residual,
        num_domains=num_domains,
        num_classes=num_classes,
        use_classification_head=use_classification_head,
        use_domain_lora=cfg.USE_DOMAIN_LORA,
        domain_lora_rank=cfg.DOMAIN_LORA_RANK,
        domain_lora_alpha=cfg.DOMAIN_LORA_ALPHA,
        domain_lora_dropout=cfg.DOMAIN_LORA_DROPOUT
    ).to(device)
    
    if use_domain_residual:
        print(f"[Model] SignalEncoder with Domain Residual enabled ({num_domains} domains)")
    
    # SpecEncoder backend switch (Attempt 57 A/B): moe / light / standard
    spec_backend = str(getattr(cfg, "SPEC_ENCODER_BACKEND", "moe")).strip().lower()
    use_moe_backend = cfg.USE_MOE and spec_backend == "moe"

    spec_moe_mv = bool(getattr(cfg, "SPEC_MOE_MULTI_VIEW_ROUTER", False))
    spec_moe_band = bool(getattr(cfg, "SPEC_MOE_BAND_CROSS_ATTN", False))
    spec_moe_heads = int(getattr(cfg, "SPEC_MOE_BAND_ATTN_HEADS", 4))
    use_bayesian_effective = bool(cfg.USE_BAYESIAN_ROUTER) and (not spec_moe_mv)

    # SpecEncoder (普通或MoE版本)
    if use_moe_backend:
        num_bands = cfg.NUM_BANDS

        # MS-STFT + freq-axis concat → 7 channels: default NUM_BANDS=3 is invalid.
        if num_spec_channels % num_bands != 0 and num_spec_channels == 7:
            num_bands = 7
            print(f"[Model] MoE: adjusted NUM_BANDS→7 for num_spec_channels=7 (1 ch/band)")

        # 验证 num_spec_channels 必须能被 num_bands 整除
        if num_spec_channels % num_bands != 0:
            raise ValueError(
                f"num_spec_channels ({num_spec_channels}) must be divisible by num_bands ({num_bands}). "
                f"Valid num_bands values for {num_spec_channels} channels are: "
                f"{', '.join(str(i) for i in range(1, num_spec_channels + 1) if num_spec_channels % i == 0)}"
            )
        
        channels_per_band = num_spec_channels // num_bands
        expert_top_k = cfg.EXPERT_TOP_K
        num_experts_per_band = cfg.NUM_EXPERTS_PER_BAND

        spec_encoder = SpecEncoderMoE(
            in_channels=num_spec_channels,
            embed_dim=512,
            num_bands=num_bands,
            num_experts_per_band=num_experts_per_band,
            channels_per_band=channels_per_band,
            expert_top_k=expert_top_k,
            use_domain_router_bias=use_domain_residual,
            use_domain_film=False,
            num_domains=num_domains,
            use_domain_lora=cfg.USE_DOMAIN_LORA,
            domain_lora_rank=cfg.DOMAIN_LORA_RANK,
            domain_lora_alpha=cfg.DOMAIN_LORA_ALPHA,
            domain_lora_dropout=cfg.DOMAIN_LORA_DROPOUT,
            # Bayesian Router (disabled when multi-view routing — input dim mismatch)
            use_bayesian_router=use_bayesian_effective,
            router_prior_std=cfg.BAYESIAN_ROUTER_PRIOR_STD,
            router_num_samples_inference=cfg.BAYESIAN_ROUTER_NUM_SAMPLES_INFERENCE,
            # Expert Performance Tracking
            use_expert_tracking=cfg.USE_EXPERT_PERFORMANCE_TRACKING,
            expert_tracker_mode=cfg.EXPERT_TRACKER_MODE,
            expert_tracker_exploration_bonus=cfg.EXPERT_TRACKER_EXPLORATION_BONUS,
            # Expert Disagreement
            use_expert_disagreement=cfg.USE_EXPERT_DISAGREEMENT,
            disagreement_method=cfg.EXPERT_DISAGREEMENT_METHOD,
            disagreement_temp_scale=cfg.EXPERT_DISAGREEMENT_TEMP_SCALE,
            disagreement_temp_mode=cfg.EXPERT_DISAGREEMENT_TEMP_MODE,
            # Classification Head (for uncertainty)
            num_classes=num_classes,
            use_classification_head=use_classification_head,
            multi_view_router=spec_moe_mv,
            band_cross_attn_enable=spec_moe_band,
            band_attn_heads=spec_moe_heads,
        ).to(device)
        
        bayesian_info = f", Bayesian Router (prior_std={cfg.BAYESIAN_ROUTER_PRIOR_STD})" if use_bayesian_effective else ""
        tracking_info = f", Expert Tracking ({cfg.EXPERT_TRACKER_MODE})" if cfg.USE_EXPERT_PERFORMANCE_TRACKING else ""
        disagreement_info = f", Disagreement ({cfg.EXPERT_DISAGREEMENT_METHOD})" if cfg.USE_EXPERT_DISAGREEMENT else ""
        mv_info = ", ATT469-multi-view-router" if spec_moe_mv else ""
        ba_info = ", ATT469-band-cross-attn" if spec_moe_band else ""

        print(f"[Model] Using MoE SpecEncoder: {num_bands} bands, {num_experts_per_band} experts per band, "
              f"top-{expert_top_k} routing{bayesian_info}{tracking_info}{disagreement_info}{mv_info}{ba_info}")
    elif spec_backend == "light":
        spec_encoder = SpecEncoderLight(
            in_channels=num_spec_channels,
            embed_dim=512,
            use_domain_residual=use_domain_residual,
            num_domains=num_domains,
            base_channels=getattr(cfg, "LIGHT_SPEC_BASE_CHANNELS", 32),
            dropout=getattr(cfg, "LIGHT_SPEC_DROPOUT", 0.1),
            num_classes=num_classes,
            use_classification_head=use_classification_head,
        ).to(device)
        print(
            "[Model] Using SpecEncoderLight"
            f" (base_channels={getattr(cfg, 'LIGHT_SPEC_BASE_CHANNELS', 32)}, "
            f"dropout={getattr(cfg, 'LIGHT_SPEC_DROPOUT', 0.1)})"
            + (f" with Domain Residual ({num_domains} domains)" if use_domain_residual else "")
        )
    else:
        spec_encoder = SpecEncoder(
            embed_dim=512, 
            in_channels=num_spec_channels,
            use_domain_residual=use_domain_residual,
            num_domains=num_domains
        ).to(device)
        print(f"[Model] Using standard SpecEncoder" + 
              (f" with Domain Residual ({num_domains} domains)" if use_domain_residual else ""))
    
    # TextEncoder
    text_encoder = TextEncoder(embed_dim=512, device=device).to(device)
    
    # CLIP模型
    clip_model = FrozenCLIP(device)
    
    # 融合模块（根据配置选择类型）
    _use_shared_proj = bool(getattr(cfg, "USE_SHARED_PROJECTION_FUSION", False))
    _use_decision = bool(getattr(cfg, "USE_DECISION_LEVEL_FUSION", False))
    _use_gated = bool(getattr(cfg, "USE_GATED_ADDITIVE_FUSION", False))
    _use_att472 = bool(getattr(cfg, "ATT472_TWO_END_SHARED_NECK_ENABLE", False))
    _use_att473 = bool(getattr(cfg, "ATT473_SPECIALIST_ARBITER_ENABLE", False))
    _use_cbp = bool(getattr(cfg, "USE_COMPACT_BILINEAR_FUSION", False))
    _dgl_enable = bool(getattr(cfg, "DGL_ENABLE", False))
    _fixed_equal = bool(getattr(cfg, "FUSION_FIXED_EQUAL_WEIGHT", True))
    _cm_align_w = float(getattr(cfg, "FUSION_CROSS_MODAL_ALIGN_WEIGHT", 0.20))
    if _use_att473:
        _split_arch = bool(getattr(cfg, "SPLIT_ARCH_ENABLE", False))
        fusion_module = _fusion_cls("SpecialistArbiterFusion")(
            embed_dim=512,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
            split_arch_mode=_split_arch,
        ).to(device)
        _fus_fwd_detach = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
        print(f"[Model] Using SpecialistArbiterFusion (ATT473): embed_dim=512, "
              f"domain_embed={cfg.FUSION_DOMAIN_EMBED_DIM}, split_arch={_split_arch}, "
              f"fusion_forward_detach={_fus_fwd_detach}, cross_modal_align_w={_cm_align_w}")
    elif _use_att472:
        _split_arch = bool(getattr(cfg, "SPLIT_ARCH_ENABLE", False))
        fusion_module = _fusion_cls("TwoEndSeparatedSharedNeckFusion")(
            embed_dim=512,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
            split_arch_mode=_split_arch,
        ).to(device)
        _fus_fwd_detach = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
        print(f"[Model] Using TwoEndSeparatedSharedNeckFusion (ATT472): embed_dim=512, "
              f"domain_embed={cfg.FUSION_DOMAIN_EMBED_DIM}, split_arch={_split_arch}, "
              f"fusion_forward_detach={_fus_fwd_detach}, cross_modal_align_w={_cm_align_w}")
    elif _use_shared_proj:
        fusion_module = _fusion_cls("SharedProjectionFusion")(
            embed_dim=512,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
            dgl_enable=_dgl_enable,
        ).to(device)
        print(f"[Model] Using SharedProjectionFusion (ATT460): embed_dim=512, "
              f"domain_embed={cfg.FUSION_DOMAIN_EMBED_DIM}, "
              f"shared_proj=True, modal_adapters=True, "
              f"cross_modal_align_w={_cm_align_w}")
    elif _use_decision:
        fusion_module = _fusion_cls("DecisionLevelFusion")(
            embed_dim=512,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
            dgl_enable=_dgl_enable,
            fixed_equal_weight=_fixed_equal,
        ).to(device)
        _weight_mode = "fixed_equal(0.5/0.5)" if _fixed_equal else "adaptive_softmax"
        _att_tag = "ATT459" if _fixed_equal else "ATT458"
        print(f"[Model] Using DecisionLevelFusion ({_att_tag}): embed_dim=512, "
              f"domain_embed={cfg.FUSION_DOMAIN_EMBED_DIM}, "
              f"weight_mode={_weight_mode}, no_detached_norm=True")
    elif _use_gated:
        _res_skip = bool(getattr(cfg, "FUSION_RESIDUAL_SKIP_ENABLE", False))
        _res_alpha = float(getattr(cfg, "FUSION_RESIDUAL_SKIP_INIT_ALPHA", 0.1))
        _split_arch = bool(getattr(cfg, "SPLIT_ARCH_ENABLE", False))
        _mean_res = bool(getattr(cfg, "FUSION_MEAN_RESIDUAL_ENABLE", False))
        _std_uv = bool(getattr(cfg, "FUSION_USE_STANDARD_UNIT_VECTOR_NORM", False))
        _bounded_spec_alpha = float(os.environ.get(
            "NEW_IDEA_BOUNDED_SPEC_RESIDUAL_ALPHA", "0.0"
        ))
        _bounded_spec_enabled = bool(int(os.environ.get(
            "NEW_IDEA_BOUNDED_SPEC_RESIDUAL_ENABLE", "0"
        )))
        fusion_module = GatedAdditiveFusion(
            embed_dim=512,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
            split_arch_mode=_split_arch,
            use_mean_residual=_mean_res,
            use_standard_unit_vector_norm=_std_uv,
            use_cross_attention=bool(int(os.environ.get("NEW_IDEA_CROSS_ATTENTION", "0"))),
            use_token_pool_only=bool(int(os.environ.get("NEW_IDEA_TOKEN_POOL_ONLY", "0"))),
            num_attention_heads=int(os.environ.get("NEW_IDEA_ATTENTION_HEADS", "8")),
            bounded_spec_residual_alpha=_bounded_spec_alpha,
            bounded_spec_residual_enabled=_bounded_spec_enabled,
        ).to(device)
        _dgl_detach = bool(getattr(cfg, "DGL_DETACH_ENCODER_INPUTS", False))
        _fus_fwd_detach = bool(getattr(cfg, "FUSION_FORWARD_DETACH_ENCODER_INPUTS", False))
        _model_desc = f"[Model] Using GatedAdditiveFusion: embed_dim=512, domain_embed={cfg.FUSION_DOMAIN_EMBED_DIM}"
        if _split_arch:
            _model_desc += ", SPLIT_ARCH=True(fusion_bypass)"
        elif _res_skip:
            _model_desc += f", residual_skip=True(alpha_init={_res_alpha:.2f})"
        if _mean_res:
            _model_desc += ", mean_residual=True(no tanh gate, zero-init last layer)"
        elif _bounded_spec_enabled or _bounded_spec_alpha > 0.0:
            _model_desc += (
                f", bounded_spec_residual=True(alpha={_bounded_spec_alpha:.3f}, "
                "zero-init sigmoid gate)"
            )
        if _std_uv:
            _model_desc += ", standard_unit_L2=True(F.normalize)"
        print(_model_desc)
        if _dgl_enable:
            print(f"[ATT438-DGL] Auxiliary fusion head ENABLED (num_classes={len(cfg.FAULT_LABEL_TO_SEMANTIC)}), "
                  f"DGL_DETACH_ENCODER_INPUTS={_dgl_detach}, fusion_forward_detach={_fus_fwd_detach}")
        elif _fus_fwd_detach:
            print(f"[ATT471-AGIT] fusion_forward_detach_encoder_inputs=True "
                  f"(export ATT471_AGIT_ENABLE=1 or DGL_DETACH_ENCODER_INPUTS=1); DGL aux head OFF")
    elif _use_cbp:
        _cbp_dim = int(getattr(cfg, "CBP_DIM", 2048))
        fusion_module = _fusion_cls("CompactBilinearFusion")(
            embed_dim=512,
            cbp_dim=_cbp_dim,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
        ).to(device)
        print(f"[Model] Using CompactBilinearFusion: cbp_dim={_cbp_dim}, "
              f"embed_dim=512, domain_embed={cfg.FUSION_DOMAIN_EMBED_DIM}")
    elif cfg.USE_CROSS_ATTENTION_FUSION:
        fusion_module = _fusion_cls("DomainAwareCrossAttentionFusion")(
            embed_dim=512,
            num_heads=cfg.FUSION_NUM_HEADS,
            domain_embed_dim=cfg.FUSION_DOMAIN_EMBED_DIM,
            num_domains=num_domains,
            use_class_aware=cfg.USE_CLASS_AWARE_ATTENTION,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
            dropout=cfg.FUSION_DROPOUT,
            use_confidence=cfg.FUSION_USE_CONFIDENCE,
            use_multi_token=cfg.FUSION_USE_MULTI_TOKEN,
            use_reliability_gating=cfg.USE_RELIABILITY_GATING,
            uncertainty_method=cfg.UNCERTAINTY_METHOD,
            gate_type=cfg.GATE_TYPE,
            gate_temperature=cfg.GATE_TEMPERATURE,
            use_modality_dropout=cfg.USE_MODALITY_DROPOUT,
            modality_drop_prob=cfg.MODALITY_DROP_PROB,
            modality_drop_mode=cfg.MODALITY_DROP_MODE,
            use_attention_gate=cfg.USE_ATTENTION_GATE,
            attention_gate_alpha=cfg.ATTENTION_GATE_ALPHA,
            attention_dim=cfg.ATTENTION_GATE_DIM,
            attention_heads=cfg.ATTENTION_GATE_HEADS,
            cross_attn_depth=getattr(cfg, 'FUSION_CROSS_ATTN_DEPTH', 1),
            use_gate_masked_attention=getattr(cfg, 'USE_GATE_MASKED_ATTENTION', False)
        ).to(device)
        print(f"[Model] Using DomainAwareCrossAttentionFusion: {cfg.FUSION_NUM_HEADS} heads, "
              f"class_aware={cfg.USE_CLASS_AWARE_ATTENTION}, confidence={cfg.FUSION_USE_CONFIDENCE}, "
              f"multi_token={cfg.FUSION_USE_MULTI_TOKEN}, progressive={cfg.FUSION_PROGRESSIVE_TRAINING}, "
              f"reliability_gating={cfg.USE_RELIABILITY_GATING}, modality_dropout={cfg.USE_MODALITY_DROPOUT}, "
              f"attention_gate={cfg.USE_ATTENTION_GATE}")
    else:
        fusion_module = _fusion_cls("CrossGatedFusion")(
            dim=512,
            use_class_aware=cfg.USE_CLASS_AWARE_ATTENTION,
            num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC)
        ).to(device)
        print(f"[Model] Using CrossGatedFusion: class_aware={cfg.USE_CLASS_AWARE_ATTENTION}, "
              f"progressive={cfg.FUSION_PROGRESSIVE_TRAINING}")
    
    # FusionDecoder
    use_compile = cfg.USE_TORCH_COMPILE
    use_domain_aware_fault_head = getattr(cfg, 'USE_DOMAIN_AWARE_FAULT_HEAD', False)
    use_residual_proxies = getattr(cfg, 'USE_RESIDUAL_PROXIES', False)
    use_flat_diag_head = bool(getattr(cfg, "USE_FLAT_DIAG_HEAD", False))
    fusion_decoder = FusionDecoder(
        embed_dim=512, 
        num_classes=len(cfg.FAULT_LABEL_TO_SEMANTIC),
        use_cache=not use_compile,
        hierarchical=not use_flat_diag_head,
        use_domain_aware_fault_head=use_domain_aware_fault_head,
        num_domains=num_domains,
        use_residual_proxies=use_residual_proxies,
        residual_num_proxies=getattr(cfg, 'RESIDUAL_NUM_PROXIES_PER_FAULT', 8),
        residual_alpha=getattr(cfg, 'RESIDUAL_ALPHA', 0.5),
        use_prototype_head=getattr(cfg, 'USE_PROTOTYPE_HEAD', True),
        use_multi_prototype_head=getattr(cfg, 'USE_MULTI_PROTOTYPE_HEAD', True),
        multi_prototype_k=getattr(cfg, 'MULTI_PROTOTYPE_K', 1),
        multi_prototype_init_std=getattr(cfg, 'MULTI_PROTOTYPE_INIT_STD', 0.01),
        use_diag_linear_residual=getattr(cfg, 'USE_DIAG_LINEAR_RESIDUAL', True),
        diag_linear_scale_init=getattr(cfg, 'DIAG_LINEAR_SCALE_INIT', 0.0),
        use_diag_linear_gate=getattr(cfg, 'USE_DIAG_LINEAR_GATE', True),
        diag_linear_gate_init_bias=getattr(cfg, 'DIAG_LINEAR_GATE_INIT_BIAS', -8.0),
        use_prompt_tuning=getattr(cfg, 'USE_PROMPT_TUNING', True),
        use_domain_prompt_tuning=getattr(cfg, 'USE_DOMAIN_PROMPT_TUNING', True),
        prompt_residual_init_std=getattr(cfg, 'PROMPT_RESIDUAL_INIT_STD', 0.01)
    ).to(device)
    
    fusion_decoder.set_nf_calibrator_requires_grad(False)

    if use_flat_diag_head:
        print("[Model] FusionDecoder using flat 4-class diagnosis head")
        try:
            use_mc = bool(getattr(cfg, "USE_FLAT_MULTI_CENTER_ROLLING", False))
            k_roll = int(getattr(cfg, "FLAT_MULTI_CENTER_ROLLING_K", 1))
            pool = str(getattr(cfg, "FLAT_MULTI_CENTER_POOL", "logsumexp")).lower()
            if use_mc and k_roll > 1:
                print(f"[Model] Flat multi-center rolling enabled: k={k_roll}, pool={pool}")
        except Exception:
            pass
    if use_domain_aware_fault_head:
        print(f"[Model] FusionDecoder with domain-aware fault heads enabled ({num_domains} domains)")
    if use_residual_proxies:
        print(f"[Model] FusionDecoder with ResidualProxyBank enabled (num_proxies={getattr(cfg, 'RESIDUAL_NUM_PROXIES_PER_FAULT', 8)}, alpha={getattr(cfg, 'RESIDUAL_ALPHA', 0.5)})")
    try:
        _dt = bool(getattr(cfg, "FUSION_DECODER_DUAL_TOWER_ENABLE", False))
        _ds = bool(getattr(cfg, "USE_DUAL_SPACE_DECOUPLE", False))
        _md = bool(getattr(cfg, "DUAL_SPACE_METRIC_DETACH", False))
        print(f"[Model] FusionDecoder dual-tower: FUSION_DECODER_DUAL_TOWER_ENABLE={_dt}, "
              f"USE_DUAL_SPACE_DECOUPLE={_ds}, DUAL_SPACE_METRIC_DETACH={_md}")
    except Exception:
        pass

    # Diagnosis-branch adapter (z_cls) switch
    if hasattr(cfg, "FUSION_USE_CLS_ADAPTER"):
        try:
            fusion_decoder.use_cls_adapter = bool(cfg.FUSION_USE_CLS_ADAPTER)
        except Exception:
            pass
    
    # torch.compile优化
    if use_compile:
        try:
            if hasattr(torch, 'compile'):
                print("[Performance] Compiling models with torch.compile...")
                signal_encoder = torch.compile(signal_encoder, mode="reduce-overhead")
                spec_encoder = torch.compile(spec_encoder, mode="reduce-overhead")
                fusion_module = torch.compile(fusion_module, mode="reduce-overhead")
                fusion_decoder = torch.compile(fusion_decoder, mode="reduce-overhead")
                print("[Performance] Models compiled successfully")
            else:
                print("[Performance] torch.compile not available (requires PyTorch >= 2.0), skipping")
        except Exception as e:
            print(f"[Performance] torch.compile failed: {e}, continuing without compilation")
    
    # AnomalyDetector
    # ATT396: create Feature Decoupler modules for evaluation
    decouple_spec = None
    decouple_signal = None
    if bool(getattr(cfg, "USE_FEATURE_DECOUPLE", False)):
        from models.feature_decoupler import FeatureDecoupler
        _dc_embed = 512
        _dc_hidden = int(getattr(cfg, "DECOUPLE_HIDDEN_DIM", _dc_embed))
        decouple_spec = FeatureDecoupler(embed_dim=_dc_embed, hidden_dim=_dc_hidden).to(device)
        decouple_signal = FeatureDecoupler(embed_dim=_dc_embed, hidden_dim=_dc_hidden).to(device)
        decouple_spec.eval()
        decouple_signal.eval()

    # ATT446: create TinyWSTEncoder for WST bypass injection (before detector)
    wst_encoder = None
    if _wst_enable and _wst_bypass:
        from models.wst_encoder import TinyWSTEncoder
        _wst_dim = int(getattr(cfg, "WST_BYPASS_DIM", 128))
        wst_encoder = TinyWSTEncoder(in_channels=1, out_dim=_wst_dim).to(device)
        _n_params = sum(p.numel() for p in wst_encoder.parameters())
        print(f"[ATT446-WST] TinyWSTEncoder created: out_dim={_wst_dim}, params={_n_params:,}")

    # ATT449-S3: create TinyEnvEncoder for envelope spectrum bypass
    env_encoder = None
    _env_bypass = bool(getattr(cfg, "ENV_BYPASS_INJECT", False))
    if _env_bypass:
        from models.env_encoder import TinyEnvEncoder
        _env_dim = int(getattr(cfg, "ENV_BYPASS_DIM", 64))
        env_encoder = TinyEnvEncoder(in_channels=1, out_dim=_env_dim).to(device)
        _n_env = sum(p.numel() for p in env_encoder.parameters())
        print(f"[ATT449-Env] TinyEnvEncoder created: out_dim={_env_dim}, params={_n_env:,}")

    # ATT464b: create TinySCDEncoder for SCD bypass injection
    scd_encoder = None
    _scd_bypass = bool(getattr(cfg, "SCD_BYPASS_INJECT", False))
    if _scd_bypass:
        from models.scd_encoder import TinySCDEncoder
        _scd_dim = int(getattr(cfg, "SCD_BYPASS_DIM", 64))
        scd_encoder = TinySCDEncoder(in_channels=1, out_dim=_scd_dim).to(device)
        _n_scd = sum(p.numel() for p in scd_encoder.parameters())
        print(f"[ATT464b-SCD] TinySCDEncoder created: out_dim={_scd_dim}, params={_n_scd:,}")

    ablation_gate_bank = create_ablation_gate_bank(cfg, device)
    if ablation_gate_bank is not None:
        print(
            f"[AblationGating] enabled: {len(ablation_gate_bank.keys())} learnable gates, "
            f"init_default={float(getattr(cfg, 'ABLATION_GATE_INIT_DEFAULT', 0.5)):.2f}"
        )
        ablation_gate_bank.log_summary(prefix="[AblationGating] init")

    detector = AnomalyDetector(
        signal_encoder=signal_encoder,
        spec_encoder=spec_encoder,
        text_encoder=text_encoder,
        fusion_module=fusion_module,
        fusion_decoder=fusion_decoder,
        device=device,
        enable_finetuning=cfg.DETECTOR_FINETUNING_ENABLE,
        finetune_samples=cfg.DETECTOR_FINETUNING_SAMPLES,
        finetune_lr=cfg.DETECTOR_FINETUNING_LR,
        finetune_steps=cfg.DETECTOR_FINETUNING_STEPS,
        decouple_spec=decouple_spec,
        decouple_signal=decouple_signal,
        wst_encoder=wst_encoder,
        env_encoder=env_encoder,
        scd_encoder=scd_encoder,
    )
    attach_gate_bank(
        ablation_gate_bank,
        detector=detector,
        signal_encoder=signal_encoder,
        spec_encoder=spec_encoder,
        fusion_module=fusion_module,
    )

    return signal_encoder, spec_encoder, text_encoder, clip_model, fusion_module, fusion_decoder, detector, wst_encoder, env_encoder, scd_encoder
