import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ibn_mixstyle import MixStyle


class SharedBandFeatureExtractor(nn.Module):
    """
    共享的频带特征提取器（所有专家共享）
    输入: [B, C_band, F, T]
    输出: [B, 128] (pooled features)
    """
    def __init__(self, in_channels, use_mixstyle=False, mixstyle_p=0.5, mixstyle_alpha=0.3):
        super().__init__()
        
        # CNN backbone for band
        self.conv1 = nn.Conv2d(
            in_channels, 32,
            kernel_size=(3, 7),
            stride=(1, 2),
            padding=(1, 3)
        )
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=(3, 5),
            stride=(1, 2),
            padding=(1, 2)
        )
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 1)
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else None
    
    def forward(self, band_spec, band_mask, return_time_features: bool = False):
        """
        band_spec: [B, C_band, F, T]
        band_mask: [B, T]
        return_time_features: 如果True，同时返回 frequency pooling 后的时序特征与下采样mask

        返回:
            pooled: [B, 128] (pooled features)
            若 return_time_features=True:
                x_time: [B, 128, T']
                mask_1d: [B, T']
        """
        # CNN
        x = F.relu(self.bn1(self.conv1(band_spec)))
        if self.mixstyle is not None:
            x = self.mixstyle(x)
        x = F.relu(self.bn2(self.conv2(x)))
        if self.mixstyle is not None:
            x = self.mixstyle(x)
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, F', T']
        
        # Frequency pooling
        x = x.mean(dim=2)  # [B, 128, T']
        
        # Mask downsampling（优化：减少循环和squeeze/unsqueeze）
        mask = band_mask.unsqueeze(1)  # [B, 1, T]
        # 链式下采样，减少中间变量
        mask = F.avg_pool1d(
            F.avg_pool1d(
                F.avg_pool1d(mask, kernel_size=2, stride=2, ceil_mode=True),
                kernel_size=2, stride=2, ceil_mode=True
            ),
            kernel_size=2, stride=2, ceil_mode=True
        )  # [B, 1, T']
        mask_1d = mask.squeeze(1)  # [B, T']
        
        # Mask-aware temporal pooling
        x_masked = x * mask_1d.unsqueeze(1)  # [B, 128, T'] * [B, 1, T']
        sum_x = x_masked.sum(dim=-1)  # [B, 128]
        denom = mask_1d.sum(dim=-1).clamp(min=1e-6)
        pooled = sum_x / denom.unsqueeze(-1)  # [B, 128]
        
        if return_time_features:
            return pooled, x, mask_1d
        return pooled


class BandExpert(nn.Module):
    """
    单个频带的专家网络（轻量级，只负责投影）
    输入: [B, 128] (来自共享特征提取器)
    输出: [B, D]
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # 每个专家只负责最后的投影层
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, pooled_features):
        """
        pooled_features: [B, 128]
        返回: [B, D]
        """
        z = self.proj(pooled_features)
        return z


class SoftRouter(nn.Module):
    """
    轻量级软路由器：直接基于频带特征计算专家权重
    移除了复杂的descriptor计算，提升效率
    """
    def __init__(self, feature_dim=128, num_experts_per_band=4, temperature=1.0, router_dropout=0.1):
        super().__init__()
        self.num_experts_per_band = num_experts_per_band
        self.temperature = temperature
        self.router_dropout = router_dropout
        
        # 轻量级路由器：直接从频带特征映射到专家权重
        self.router = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(router_dropout),
            nn.Linear(feature_dim // 2, num_experts_per_band)
        )
    
    def forward(self, band_features, extra_bias=None, return_logits=False):
        """
        band_features: [B, feature_dim] 频带的pooled features
        extra_bias: [B, num_experts_per_band] or [num_experts_per_band] (optional)
        return_logits: if True, also return pre-softmax logits (for debugging)
        返回: [B, num_experts_per_band] 专家权重
        """
        # 计算logits
        logits = self.router(band_features)  # [B, num_experts_per_band]

        # 可选：添加外部bias（例如域条件化 router bias）
        if extra_bias is not None:
            if extra_bias.dim() == 1:
                logits = logits + extra_bias.unsqueeze(0)
            else:
                logits = logits + extra_bias
        
        # 应用温度参数（温度越高，分布越平滑）
        logits = logits / self.temperature
        
        # 标准softmax
        expert_weights = F.softmax(logits, dim=-1)

        if return_logits:
            return expert_weights, logits
        return expert_weights


class MultiViewContextEncoder(nn.Module):
    """Global spectrum pooling + channel-wise temporal statistics for router context."""

    def __init__(self, in_channels: int, global_dim: int = 64, sample_dim: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_g = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, global_dim),
            nn.ReLU(inplace=True),
        )
        hid = max(32, min(128, in_channels * 2))
        self.fc_s = nn.Sequential(
            nn.Linear(in_channels * 2, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, sample_dim),
        )

    def forward(self, spec: torch.Tensor):
        # spec [B,C,F,T]
        g = self.pool(spec).flatten(1)
        g = self.fc_g(g)
        m = spec.mean(dim=(2, 3))
        s = spec.std(dim=(2, 3))
        ms = torch.cat([m, s], dim=1)
        svec = self.fc_s(ms)
        return g, svec


class MultiViewSoftRouter(nn.Module):
    """Router logits from concat(band_feat, global_ctx, sample_ctx)."""

    def __init__(self, in_dim: int, num_experts_per_band: int, temperature: float = 1.0, router_dropout: float = 0.1):
        super().__init__()
        self.num_experts_per_band = num_experts_per_band
        self.temperature = temperature
        self.router_dropout = router_dropout
        h = max(in_dim // 2, 64)
        self.router = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(router_dropout),
            nn.Linear(h, num_experts_per_band),
        )

    def forward(self, features: torch.Tensor, extra_bias=None, return_logits=False):
        logits = self.router(features)
        if extra_bias is not None:
            if extra_bias.dim() == 1:
                logits = logits + extra_bias.unsqueeze(0)
            else:
                logits = logits + extra_bias
        logits = logits / self.temperature
        expert_weights = F.softmax(logits, dim=-1)
        if return_logits:
            return expert_weights, logits
        return expert_weights


from models.domain_lora import DomainLoRA


class SpecEncoderMoE(nn.Module):
    """
    MoE版本的Spectrogram Encoder
    将spec解耦为若干频带，每个频带使用多组专家学习表示
    """
    def __init__(
        self,
        in_channels=6,  # 通常是3个频带×2个n_fft
        embed_dim=256,
        num_bands=3,
        num_experts_per_band=4,
        channels_per_band=2,  # 每个频带的通道数（对应2个n_fft）
        expert_top_k=None,  # 稀疏路由：每个频带激活的专家数，None表示使用所有专家
        # ===== Domain conditioning (router bias / FiLM) =====
        use_domain_router_bias=False,
        use_domain_film=False,
        num_domains=3,
        use_domain_lora=False,
        domain_lora_rank=4,
        domain_lora_alpha=None,
        domain_lora_dropout=0.0,
        # ===== Bayesian Router =====
        use_bayesian_router=False,
        router_prior_std=1.0,
        router_num_samples_inference=5,
        # ===== Expert Performance Tracking =====
        use_expert_tracking=False,
        expert_tracker_mode='ucb',
        expert_tracker_exploration_bonus=0.1,
        # ===== Expert Disagreement =====
        use_expert_disagreement=False,
        disagreement_method='variance',
        disagreement_temp_scale=2.0,
        disagreement_temp_mode='increase',
        # ===== Classification Head (for uncertainty) =====
        num_classes=4,
        use_classification_head=False,
        # ===== ATT469: multi-view router + band-level attention =====
        multi_view_router=False,
        band_cross_attn_enable=False,
        band_attn_heads=4,
    ):
        super().__init__()
        
        self.num_bands = num_bands
        self.num_experts_per_band = num_experts_per_band
        self.channels_per_band = channels_per_band
        self.embed_dim = embed_dim
        self.expert_top_k = expert_top_k if expert_top_k is not None else num_experts_per_band

        # Domain conditioning flags
        self.use_domain_router_bias = use_domain_router_bias
        self.use_domain_film = use_domain_film
        self.num_domains = num_domains
        
        self.router_prior_std = router_prior_std
        
        # Expert tracking flags
        self.use_expert_tracking = use_expert_tracking
        self.expert_tracker_mode = expert_tracker_mode
        
        # Expert disagreement flags
        self.use_expert_disagreement = use_expert_disagreement
        self.disagreement_temp_scale = disagreement_temp_scale
        self.disagreement_temp_mode = disagreement_temp_mode
        self.multi_view_router = bool(multi_view_router)
        self.band_cross_attn_enable = bool(band_cross_attn_enable)
        self._mv_router_dim = 128 + 64 + 64 if self.multi_view_router else 128
        
        # 验证通道数
        assert in_channels == num_bands * channels_per_band, \
            f"in_channels ({in_channels}) must equal num_bands ({num_bands}) * channels_per_band ({channels_per_band})"
        
        # Attempt 66: MixStyle in spec early layers for domain generalization.
        try:
            import config as cfg
            use_mixstyle_spec = bool(getattr(cfg, "USE_MIXSTYLE_SPEC", False))
            mixstyle_p = float(getattr(cfg, "MIXSTYLE_P", 0.5))
            mixstyle_alpha = float(getattr(cfg, "MIXSTYLE_ALPHA", 0.3))
        except Exception:
            use_mixstyle_spec = False
            mixstyle_p = 0.5
            mixstyle_alpha = 0.3

        # 为每个频带创建共享的特征提取器（节省内存）
        self.shared_extractors = nn.ModuleList([
            SharedBandFeatureExtractor(
                in_channels=channels_per_band,
                use_mixstyle=use_mixstyle_spec,
                mixstyle_p=mixstyle_p,
                mixstyle_alpha=mixstyle_alpha
            )
            for _ in range(num_bands)
        ])
        
        # 为每个频带创建多个专家（轻量级，只负责投影）
        self.experts = nn.ModuleList([
            nn.ModuleList([
                BandExpert(embed_dim=embed_dim)
                for _ in range(num_experts_per_band)
            ])
            for _ in range(num_bands)
        ])

        self.mv_ctx = MultiViewContextEncoder(in_channels) if self.multi_view_router else None
        if self.band_cross_attn_enable:
            nh = int(band_attn_heads)
            if embed_dim % nh != 0:
                raise ValueError(f"ATT469 band_attn_heads={nh} must divide embed_dim={embed_dim}")
            self.band_attn_norm = nn.LayerNorm(embed_dim)
            self.band_cross_attn = nn.MultiheadAttention(
                embed_dim, nh, batch_first=True,
            )
        else:
            self.band_attn_norm = None
            self.band_cross_attn = None

        # Multi-view routing uses concatenated descriptors; BayesianRouter stays 128-d only.
        self.use_bayesian_router = bool(use_bayesian_router) and (not self.multi_view_router)
        if bool(use_bayesian_router) and self.multi_view_router:
            print("[ATT469-MoE] Bayesian router disabled when multi_view_router=True (input dim mismatch).")

        # 轻量级路由器（每个频带一个）
        if self.use_bayesian_router:
            from models.bayesian_router import BayesianRouter
            self.routers = nn.ModuleList([
                BayesianRouter(
                    feature_dim=128,
                    num_experts=num_experts_per_band,
                    temperature=1.0,
                    prior_std=router_prior_std,
                    num_samples_inference=router_num_samples_inference,
                    router_dropout=0.1
                )
                for _ in range(num_bands)
            ])
            print(f"[SpecEncoderMoE] Using Bayesian Router with prior_std={router_prior_std}")
        elif self.multi_view_router:
            self.routers = nn.ModuleList([
                MultiViewSoftRouter(
                    in_dim=self._mv_router_dim,
                    num_experts_per_band=num_experts_per_band,
                    temperature=1.0,
                    router_dropout=0.1,
                )
                for _ in range(num_bands)
            ])
            print(f"[ATT469-MoE] Multi-view SoftRouter per band, in_dim={self._mv_router_dim}")
        else:
            self.routers = nn.ModuleList([
                SoftRouter(
                    feature_dim=128,
                    num_experts_per_band=num_experts_per_band,
                    temperature=1.0,
                    router_dropout=0.1
                )
                for _ in range(num_bands)
            ])

        # ===== Domain router bias: per-band embedding -> bias over experts =====
        # This lets routing adapt per-domain without forcing experts to learn domain identity.
        if self.use_domain_router_bias:
            self.domain_router_bias = nn.ModuleList([
                nn.Embedding(self.num_domains, self.num_experts_per_band)
                for _ in range(self.num_bands)
            ])
            # init to zeros so it starts as no-op (stable)
            for emb in self.domain_router_bias:
                nn.init.zeros_(emb.weight)
        else:
            self.domain_router_bias = None

        # ===== Optional: Domain FiLM on pooled band features (128-d) =====
        if self.use_domain_film:
            # Simple FiLM: gamma/beta per domain per band
            self.domain_film_gamma = nn.ModuleList([
                nn.Embedding(self.num_domains, 128) for _ in range(self.num_bands)
            ])
            self.domain_film_beta = nn.ModuleList([
                nn.Embedding(self.num_domains, 128) for _ in range(self.num_bands)
            ])
            for g in self.domain_film_gamma:
                nn.init.zeros_(g.weight)  # start near identity
            for b in self.domain_film_beta:
                nn.init.zeros_(b.weight)
            # gate per band (sigmoid) to gradually enable
            self.domain_film_gate = nn.Parameter(torch.zeros(self.num_bands))
        else:
            self.domain_film_gamma = None
            self.domain_film_beta = None
            self.domain_film_gate = None
        
        # ===== Expert Performance Tracker =====
        if self.use_expert_tracking:
            from utils.expert_performance_tracker import ExpertPerformanceTracker
            self.expert_tracker = ExpertPerformanceTracker(
                num_domains=num_domains,
                num_bands=num_bands,
                num_experts_per_band=num_experts_per_band,
                exploration_bonus=expert_tracker_exploration_bonus,
                use_thompson_sampling=(expert_tracker_mode == 'thompson'),
                device='cpu'  # Will be moved to device later
            )
            print(f"[SpecEncoderMoE] Expert Performance Tracker enabled: mode={expert_tracker_mode}")
        else:
            self.expert_tracker = None
        
        # ===== Expert Disagreement Estimator =====
        if self.use_expert_disagreement:
            from models.expert_uncertainty import ExpertDisagreementEstimator
            self.disagreement_estimator = ExpertDisagreementEstimator(
                method=disagreement_method,
                normalize=True
            )
            print(f"[SpecEncoderMoE] Expert Disagreement Estimator enabled: method={disagreement_method}")
        else:
            self.disagreement_estimator = None
        
        # 移除了复杂的全局descriptor计算模块，提升效率
        # 路由器现在直接基于频带特征工作
        
        # 频带输出聚合
        self.band_aggregator = nn.Sequential(
            nn.Linear(embed_dim * num_bands, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.use_domain_lora = use_domain_lora
        if use_domain_lora:
            self.domain_lora_agg = DomainLoRA(
                in_features=embed_dim,
                out_features=embed_dim,
                rank=domain_lora_rank,
                num_domains=num_domains,
                alpha=domain_lora_alpha,
                dropout=domain_lora_dropout
            )
        
        # ========= Lightweight Classification Head (for uncertainty estimation) =========
        self.use_classification_head = use_classification_head
        self.num_classes = num_classes
        try:
            import config as _cfg_moe_cls
        except Exception:
            _cfg_moe_cls = None

        def _cg(attr, default):
            return default if _cfg_moe_cls is None else getattr(_cfg_moe_cls, attr, default)

        if use_classification_head:
            # ATT449/ATT450: align with SpecEncoderLight — widen classifier when bypass branches enabled.
            self._wst_bypass_dim = int(_cg("WST_BYPASS_DIM", 0)) if bool(_cg("WST_BYPASS_INJECT", False)) else 0
            self._env_bypass_dim = int(_cg("ENV_BYPASS_DIM", 0)) if bool(_cg("ENV_BYPASS_INJECT", False)) else 0
            self._env_ratio_dim = 1 if self._env_bypass_dim > 0 else 0
            self._scd_bypass_dim = int(_cg("SCD_BYPASS_DIM", 0)) if bool(_cg("SCD_BYPASS_INJECT", False)) else 0
            _spec_norm_init = float(embed_dim) ** 0.5
            if self._wst_bypass_dim > 0:
                self._wst_scale = nn.Parameter(torch.tensor(_spec_norm_init))
            else:
                self._wst_scale = None
            if self._env_bypass_dim > 0:
                self._env_scale = nn.Parameter(torch.tensor(_spec_norm_init))
            else:
                self._env_scale = None
            if self._env_ratio_dim > 0:
                self._ratio_scale = nn.Parameter(torch.tensor(5.0))
            else:
                self._ratio_scale = None
            if self._scd_bypass_dim > 0:
                self._scd_scale = nn.Parameter(torch.tensor(_spec_norm_init))
            else:
                self._scd_scale = None
            self._bypass_scale_log_count = 0

            _cls_in = embed_dim + self._wst_bypass_dim + self._env_bypass_dim + self._env_ratio_dim + self._scd_bypass_dim
            self.classifier = nn.Sequential(
                nn.Linear(_cls_in, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, num_classes),
            )
            _parts = [f"spec={embed_dim}"]
            if self._wst_bypass_dim > 0:
                _parts.append(f"wst={self._wst_bypass_dim}")
            if self._env_bypass_dim > 0:
                _parts.append(f"env={self._env_bypass_dim}")
            if self._env_ratio_dim > 0:
                _parts.append("ratio=1")
            if self._scd_bypass_dim > 0:
                _parts.append(f"scd={self._scd_bypass_dim}")
            print(f"[SpecEncoderMoE-SpecHead] MLP D_in={_cls_in} ({'+'.join(_parts)})")
        else:
            self.classifier = None
            self._wst_bypass_dim = 0
            self._env_bypass_dim = 0
            self._env_ratio_dim = 0
            self._scd_bypass_dim = 0
            self._wst_scale = None
            self._env_scale = None
            self._ratio_scale = None
            self._scd_scale = None
            self._bypass_scale_log_count = 0
        
        # MoE统计信息（用于损失计算）
        self.expert_loads = None  # 用于记录专家负载
        self.expert_weights_history = None  # 用于记录权重历史
        
        # 初始化专家和路由器，确保初始时就有差异
        self._init_experts()
        self._init_routers()
    
    def _init_experts(self):
        """初始化专家，确保初始时就有差异"""
        for band_idx, band_experts in enumerate(self.experts):
            for expert_idx, expert in enumerate(band_experts):
                # 为每个专家使用正交初始化
                nn.init.orthogonal_(expert.proj[0].weight, gain=1.0)
                # 添加更大的随机偏移，确保专家初始时就有明显差异
                # 偏移量随专家索引变化，使不同专家有不同的初始方向
                # 增加偏移量从0.01到0.05，使专家差异更明显
                offset = 0.05 * (expert_idx + 1) * torch.randn_like(expert.proj[0].weight)
                expert.proj[0].weight.data += offset
                # 初始化bias（如果有）
                if expert.proj[0].bias is not None:
                    # 为不同专家添加不同的bias偏移，增强差异
                    nn.init.zeros_(expert.proj[0].bias)
                    expert.proj[0].bias.data += 0.01 * (expert_idx + 1) * torch.randn_like(expert.proj[0].bias)
    
    def _init_routers(self):
        """初始化路由器，确保所有专家有公平的初始权重"""
        for router in self.routers:
            # Check router type - BayesianRouter vs SoftRouter
            if hasattr(router, 'fc2'):  # BayesianRouter
                # For BayesianRouter, initialize the final layer's mean weights
                # The BayesianLinear layer already has proper initialization in reset_parameters()
                # We just ensure uniform bias initialization for fair expert selection
                if hasattr(router.fc2, 'mu_b'):
                    nn.init.uniform_(router.fc2.mu_b, -0.01, 0.01)
            elif hasattr(router, 'router'):  # SoftRouter
                # Original initialization for SoftRouter
                router_output_layer = router.router[-1]  # 最后一层是输出层
                # 使用Xavier初始化，但添加小的随机偏移
                nn.init.xavier_uniform_(router_output_layer.weight, gain=0.1)  # 小的gain使初始权重接近0
                # 初始化bias为小的随机值，确保所有专家都有非零初始权重
                if router_output_layer.bias is not None:
                    # 使用小的均匀分布初始化bias，使所有专家初始权重接近1/num_experts
                    nn.init.uniform_(router_output_layer.bias, -0.01, 0.01)
    
    # 移除了compute_descriptor方法，路由器现在直接使用频带特征
    
    def split_spec_into_bands(self, spec):
        """
        将spec按通道分解为频带
        
        spec: [B, C, F, T] where C = num_bands * channels_per_band
        返回: List of [B, channels_per_band, F, T]
        """
        bands = []
        for i in range(self.num_bands):
            start_ch = i * self.channels_per_band
            end_ch = (i + 1) * self.channels_per_band
            band = spec[:, start_ch:end_ch, :, :]  # [B, channels_per_band, F, T]
            bands.append(band)
        return bands
    
    def forward(
        self,
        spec,
        spec_mask,
        text_emb=None,
        domain_ids=None,
        return_moe_stats=False,
        return_tokens: bool = False,
        return_logits=False,
        stat_vec=None,
        wst_aux=None,
        env_aux=None,
        ratio_aux=None,
        scd_aux=None,
    ):
        """
        spec: [B, C, F, T]
        spec_mask: [B, T]
        text_emb: [B, embed_dim] 文本embedding（可选）
        domain_ids: [B] 域id（可选）
        return_moe_stats: 是否返回MoE统计信息（用于损失计算）
        return_tokens: 是否返回时间步 token 序列（用于多token cross-attention）
        return_logits: bool, 如果True且use_classification_head=True，同时返回分类logits [B, num_classes]
        wst_aux/env_aux/ratio_aux/scd_aux: bypass features（与 SpecEncoderLight / train_alignment 约定一致）
        
        返回:
            z_spec: [B, embed_dim]
            moe_stats: dict (如果return_moe_stats=True)
            logits: [B, num_classes] (如果return_logits=True且use_classification_head=True)
            若 return_tokens=True:
                z_spec_tokens: [B, num_bands*T', embed_dim]
                mask_tokens: [B, num_bands*T']
        """
        B = spec.size(0)
        device = spec.device

        global_ctx = None
        sample_ctx = None
        if self.mv_ctx is not None:
            global_ctx, sample_ctx = self.mv_ctx(spec)
        
        # 将spec分解为频带
        band_specs = self.split_spec_into_bands(spec)  # List of [B, channels_per_band, F, T]
        
        # 为每个频带处理
        band_outputs = []
        band_tokens_list = []
        band_masks_list = []
        all_expert_weights = []
        all_expert_outputs = []
        all_router_uncertainties = []
        all_kl_losses = []
        all_disagreements = []
        all_entropy_losses = []
        
        for band_idx, (band_spec, router, shared_extractor) in enumerate(zip(band_specs, self.routers, self.shared_extractors)):
            # 使用共享特征提取器（只计算一次，所有专家共享）
            if return_tokens:
                pooled_features, x_time, mask_1d = shared_extractor(band_spec, spec_mask, return_time_features=True)  # pooled:[B,128], x_time:[B,128,T'], mask:[B,T']
            else:
                pooled_features = shared_extractor(band_spec, spec_mask)  # [B, 128]
                x_time, mask_1d = None, None

            # ===== Optional: domain FiLM on pooled features =====
            if self.use_domain_film and domain_ids is not None:
                gamma = self.domain_film_gamma[band_idx](domain_ids)  # [B, 128]
                beta = self.domain_film_beta[band_idx](domain_ids)    # [B, 128]
                gate = torch.sigmoid(self.domain_film_gate[band_idx])
                pooled_features = pooled_features + gate * (gamma * pooled_features + beta)
            
            # 计算该频带的专家权重（直接基于频带特征）
            extra_bias = None
            router_uncertainty = None
            kl_loss = None
            
            # Get domain-adaptive bias from expert tracker
            if self.use_expert_tracking and domain_ids is not None:
                extra_bias = self.expert_tracker.get_domain_bias(
                    domain_ids, band_idx, mode=self.expert_tracker_mode)
            elif self.use_domain_router_bias and domain_ids is not None and self.domain_router_bias is not None:
                extra_bias = self.domain_router_bias[band_idx](domain_ids)  # [B, num_experts_per_band]
            
            # Add input noise to router to encourage exploration (prevent collapse)
            if self.training:
                # Add 5% relative noise to features
                noise_scale = 0.05 * pooled_features.detach().std() if pooled_features.numel() > 1 else 0.05
                pooled_features_for_router = pooled_features + noise_scale * torch.randn_like(pooled_features)
            else:
                pooled_features_for_router = pooled_features

            if global_ctx is not None:
                router_in = torch.cat([pooled_features_for_router, global_ctx, sample_ctx], dim=-1)
            else:
                router_in = pooled_features_for_router

            # Call router (Bayesian or standard)
            if self.use_bayesian_router:
                expert_weights, router_uncertainty, kl_loss = router(
                    pooled_features_for_router, extra_bias=extra_bias, return_uncertainty=True)
                
                # SOTA Fix (NeurIPS 2024): Entropy Regularization (-H(p))
                # Penalize low entropy (certainty) to force router to be unsure initially, encouraging exploration.
                if self.training:
                    # Calculate entropy: -sum(p * log(p))
                    entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-10), dim=-1)  # [B]
                    # We want to maximize entropy -> minimize -H(p)
                    # wait, maximize entropy means minimize negative entropy?
                    # Entropy Reg term is usually: -H(p) (to minimize, we push p towards uniform)
                    # OR we penalize low entropy. 
                    # If we minimize -H(p), we MAXIMIZE H(p). 
                    # Yes, to prevent collapse (p=1.0 for one expert), we want higher entropy.
                    entropy_loss_val = -entropy.mean()
                    
                    # Store for return
                    entropy_loss_val = -entropy.mean()
                    
                    # Store for return
                    all_entropy_losses.append(entropy_loss_val)
            else:
                # SoftRouter / MultiViewSoftRouter: Expert Dropout (SOTA Fix)
                if self.training:
                     # Get raw logits
                    _, logits = router(router_in, extra_bias=extra_bias, return_logits=True)
                    
                    # Expert Dropout: mask the most likely expert to force second choice
                    try:
                        import config as _cfg_mod
                        expert_dropout_prob = float(getattr(_cfg_mod, 'MOE_EXPERT_DROPOUT', 0.0))
                    except Exception:
                        expert_dropout_prob = 0.0
                    if expert_dropout_prob > 0 and torch.rand(1).item() < expert_dropout_prob:
                        # Identify top-1 expert
                        top1_idx = logits.argmax(dim=-1) # [B]
                        # Create mask: set top-1 logit to -inf
                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        mask.scatter_(1, top1_idx.unsqueeze(1), True)
                        logits = logits.masked_fill(mask, float('-inf'))
                    
                    # Apply softmax
                    expert_weights = F.softmax(logits / router.temperature, dim=-1)
                    
                    # Entropy Loss calculation
                    entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-10), dim=-1)
                    entropy_loss_val = -entropy.mean() # Minimize -H(p) => Maximize H(p)
                    
                    entropy_loss_val = -entropy.mean() # Minimize -H(p) => Maximize H(p)
                    
                    all_entropy_losses.append(entropy_loss_val)
                    
                else:
                    expert_weights = router(router_in, extra_bias=extra_bias)  # [B, num_experts_per_band]
            
            all_expert_weights.append(expert_weights)
            
            # ===== 稀疏Top-k路由（优化：批量处理） =====
            if self.expert_top_k < self.num_experts_per_band:
                # 选择top-k个专家
                topk_weights, topk_indices = torch.topk(expert_weights, k=self.expert_top_k, dim=-1)  # [B, k], [B, k]
                # 重新归一化权重（只在选中的k个专家上归一化）
                topk_weights = F.softmax(topk_weights, dim=-1)  # [B, k]
                
                batch_size = pooled_features.size(0)
                
                # 优化：批量计算所有专家的输出，然后使用gather选择top-k
                # 虽然计算了所有专家，但避免了Python循环，在GPU上更快
                expert_outputs = []
                for expert in self.experts[band_idx]:
                    expert_out = expert(pooled_features)  # [B, embed_dim]
                    # 添加dropout和噪声（训练时）
                    if self.training:
                        expert_out = F.dropout(expert_out, p=0.15, training=True)
                        expert_out = expert_out + 0.01 * torch.randn_like(expert_out)
                    expert_outputs.append(expert_out)
                
                # Stack所有专家输出: [B, num_experts, embed_dim]
                all_expert_outs = torch.stack(expert_outputs, dim=1)
                
                # Compute expert disagreement for uncertainty estimation
                if self.use_expert_disagreement and return_moe_stats:
                    disagreement, _ = self.disagreement_estimator(all_expert_outs, expert_weights)
                    all_disagreements.append(disagreement)
                
                # 保存用于统计（如果需要）
                if return_moe_stats:
                    all_expert_outputs.append(all_expert_outs)
                    if router_uncertainty is not None:
                        all_router_uncertainties.append(router_uncertainty)
                    if kl_loss is not None:
                        all_kl_losses.append(kl_loss)
                
                # 使用gather批量选择top-k专家的输出: [B, k, embed_dim]
                # topk_indices: [B, k], 需要扩展维度用于gather
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.expert_top_k)  # [B, k]
                selected_expert_outs = all_expert_outs[batch_indices, topk_indices]  # [B, k, embed_dim]
                
                # 按权重加权求和: [B, k, embed_dim] * [B, k, 1] -> [B, embed_dim]
                topk_weights_expanded = topk_weights.unsqueeze(-1)  # [B, k, 1]
                band_output = (selected_expert_outs * topk_weights_expanded).sum(dim=1)  # [B, embed_dim]
                
                band_outputs.append(band_output)

                if return_tokens:
                    # x_time: [B, 128, T'] -> [B, T', 128]
                    x_time_bt = x_time.permute(0, 2, 1).contiguous()  # [B, T', 128]
                    # 先计算所有专家的 token 输出 [B, E, T', D]
                    expert_tok_outs = []
                    for expert in self.experts[band_idx]:
                        tok = expert.proj(x_time_bt)  # [B, T', D]
                        expert_tok_outs.append(tok)
                    expert_tok_outs = torch.stack(expert_tok_outs, dim=1)  # [B, E, T', D]
                    # 选择 top-k 专家
                    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.expert_top_k)  # [B, k]
                    selected_tok_outs = expert_tok_outs[batch_indices, topk_indices]  # [B, k, T', D]
                    # 加权求和
                    band_tokens = (selected_tok_outs * topk_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, T', D]
                    band_tokens_list.append(band_tokens)
                    band_masks_list.append(mask_1d)
            else:
                # 原始实现：使用所有专家
                expert_outputs = []
                for expert in self.experts[band_idx]:
                    expert_out = expert(pooled_features)  # [B, embed_dim]
                    # 添加dropout和小的噪声，增加鲁棒性和多样性
                    if self.training:
                        expert_out = F.dropout(expert_out, p=0.15, training=True)
                        # 添加小的噪声（std=0.01）
                        expert_out = expert_out + 0.01 * torch.randn_like(expert_out)
                    expert_outputs.append(expert_out)
                
                expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts_per_band, embed_dim]
                
                # Compute expert disagreement for uncertainty estimation
                if self.use_expert_disagreement and return_moe_stats:
                    disagreement, _ = self.disagreement_estimator(expert_outputs, expert_weights)
                    all_disagreements.append(disagreement)
                
                # Save for stats
                if return_moe_stats:
                    all_expert_outputs.append(expert_outputs)
                    if router_uncertainty is not None:
                        all_router_uncertainties.append(router_uncertainty)
                    if kl_loss is not None:
                        all_kl_losses.append(kl_loss)
                
                # 按权重聚合专家输出
                expert_weights_expanded = expert_weights.unsqueeze(-1)  # [B, num_experts_per_band, 1]
                band_output = (expert_outputs * expert_weights_expanded).sum(dim=1)  # [B, embed_dim]
                band_outputs.append(band_output)

                if return_tokens:
                    x_time_bt = x_time.permute(0, 2, 1).contiguous()  # [B, T', 128]
                    expert_tok_outs = []
                    for expert in self.experts[band_idx]:
                        tok = expert.proj(x_time_bt)  # [B, T', D]
                        expert_tok_outs.append(tok)
                    expert_tok_outs = torch.stack(expert_tok_outs, dim=1)  # [B, E, T', D]
                    band_tokens = (expert_tok_outs * expert_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, T', D]
                    band_tokens_list.append(band_tokens)
                    band_masks_list.append(mask_1d)

        band_sparsity_tensor = None
        if self.band_cross_attn is not None:
            tok = torch.stack(band_outputs, dim=1)
            tok = self.band_attn_norm(tok)
            proto = tok.mean(dim=1, keepdim=True)
            attn_out, attn_w = self.band_cross_attn(
                proto, tok, tok, need_weights=True, average_attn_weights=True,
            )
            importance = attn_w.squeeze(1).clamp_min(1e-10)
            importance = importance / importance.sum(dim=-1, keepdim=True)
            band_sparsity_tensor = (-(importance * (importance + 1e-10).log()).sum(dim=-1)).mean()
            refined = tok + attn_out.expand_as(tok)
            all_bands = refined.reshape(B, -1)
        else:
            all_bands = torch.cat(band_outputs, dim=-1)
        
        # 聚合所有频带输出
        z_spec_pre_norm = self.band_aggregator(all_bands)  # [B, embed_dim]
        if self.use_domain_lora:
            lora_delta = self.domain_lora_agg(z_spec_pre_norm, domain_ids)
            z_spec_pre_norm = z_spec_pre_norm + lora_delta
        z_spec = F.normalize(z_spec_pre_norm, dim=-1)
        try:
            z_spec._pre_l2_feature = z_spec_pre_norm
        except Exception:
            pass
        
        # Classification head (for uncertainty estimation)
        logits = None
        if return_logits and self.use_classification_head and self.classifier is not None:
            _cls_in = self._build_cls_input(z_spec_pre_norm, wst_aux, env_aux, ratio_aux, scd_aux)
            logits = self.classifier(_cls_in)  # [B, num_classes]
        
        # 保存MoE统计信息
        if return_moe_stats:
            # 计算专家负载（每个专家被选中的平均权重）
            expert_loads = []
            for band_weights in all_expert_weights:
                # band_weights: [B, num_experts_per_band]
                avg_load = band_weights.mean(dim=0)  # [num_experts_per_band]
                expert_loads.append(avg_load)
            
            # 计算权重熵（用于多样性）
            expert_entropies = []
            for band_weights in all_expert_weights:
                # 计算每个样本的熵
                entropy = -torch.sum(band_weights * torch.log(band_weights + 1e-10), dim=-1)  # [B]
                expert_entropies.append(entropy.mean())  # 标量
            
            moe_stats = {
                'expert_loads': expert_loads,  # List of [num_experts_per_band]
                'expert_weights': all_expert_weights,  # List of [B, num_experts_per_band]
                'expert_outputs': all_expert_outputs,  # List of [B, num_experts_per_band, embed_dim]
                'expert_entropies': expert_entropies,  # List of scalars
            }
            if band_sparsity_tensor is not None:
                moe_stats['band_sparsity_loss'] = band_sparsity_tensor
            
            # Add Bayesian router statistics
            if self.use_bayesian_router and len(all_router_uncertainties) > 0:
                moe_stats['router_uncertainties'] = all_router_uncertainties  # List of [B]
                moe_stats['kl_losses'] = all_kl_losses  # List of scalars
            
            # SOTA Fix: Add total entropy loss
            if len(all_entropy_losses) > 0:
                # Sum over bands
                moe_stats['entropy_loss'] = torch.stack(all_entropy_losses).sum()
            
            # Add expert disagreement statistics
            if self.use_expert_disagreement and len(all_disagreements) > 0:
                moe_stats['expert_disagreements'] = all_disagreements  # List of [B]
            
            if return_tokens:
                z_spec_tokens = torch.cat(band_tokens_list, dim=1)  # [B, num_bands*T', D]
                mask_tokens = torch.cat(band_masks_list, dim=1)      # [B, num_bands*T']
                if return_logits:
                    return z_spec, z_spec_tokens, mask_tokens, moe_stats, logits
                return z_spec, z_spec_tokens, mask_tokens, moe_stats
            if return_logits:
                return z_spec, moe_stats, logits
            return z_spec, moe_stats
        
        if return_tokens:
            z_spec_tokens = torch.cat(band_tokens_list, dim=1)
            mask_tokens = torch.cat(band_masks_list, dim=1)
            if return_logits:
                return z_spec, z_spec_tokens, mask_tokens, logits
            return z_spec, z_spec_tokens, mask_tokens
        if return_logits:
            return z_spec, logits
        return z_spec
    
    def compute_spec_head_logits(self, z_spec, wst_aux=None, env_aux=None, ratio_aux=None, scd_aux=None):
        """Match SpecEncoderLight / anomalyDetector: logits with bypass concatenation."""
        if self.classifier is None:
            return None
        z_pre = getattr(z_spec, "_pre_l2_feature", z_spec)
        _cls_input = self._build_cls_input(z_pre, wst_aux, env_aux, ratio_aux, scd_aux)
        return self.classifier(_cls_input)

    def _build_cls_input(self, z_pre, wst_aux=None, env_aux=None, ratio_aux=None, scd_aux=None):
        """Same convention as SpecEncoderLight._build_cls_input (ATT450-S1)."""
        parts = [z_pre]
        B = z_pre.size(0)
        dev = z_pre.device
        dt = z_pre.dtype
        if self._wst_bypass_dim > 0:
            if wst_aux is not None:
                _wst = F.normalize(wst_aux, dim=-1) * self._wst_scale
                parts.append(_wst)
            else:
                parts.append(torch.zeros(B, self._wst_bypass_dim, device=dev, dtype=dt))
        if self._env_bypass_dim > 0:
            if env_aux is not None:
                _env = F.normalize(env_aux, dim=-1) * self._env_scale
                parts.append(_env)
            else:
                parts.append(torch.zeros(B, self._env_bypass_dim, device=dev, dtype=dt))
        if self._env_ratio_dim > 0:
            if ratio_aux is not None:
                parts.append(ratio_aux * self._ratio_scale)
            else:
                parts.append(torch.zeros(B, self._env_ratio_dim, device=dev, dtype=dt))
        if self._scd_bypass_dim > 0:
            if scd_aux is not None:
                _scd = F.normalize(scd_aux, dim=-1) * self._scd_scale
                parts.append(_scd)
            else:
                parts.append(torch.zeros(B, self._scd_bypass_dim, device=dev, dtype=dt))

        if self.training and self._bypass_scale_log_count < 5:
            self._bypass_scale_log_count += 1
            _msg_parts = [f"[SpecEncoderMoE-ATT450-ScaleNorm] step={self._bypass_scale_log_count}"]
            _msg_parts.append(f"z_pre_norm={float(z_pre.detach().norm(dim=-1).mean()):.3f}")
            if self._wst_scale is not None:
                _msg_parts.append(f"wst_scale={float(self._wst_scale.detach()):.3f}")
            if self._env_scale is not None:
                _msg_parts.append(f"env_scale={float(self._env_scale.detach()):.3f}")
            if self._ratio_scale is not None:
                _msg_parts.append(f"ratio_scale={float(self._ratio_scale.detach()):.3f}")
            if self._scd_scale is not None:
                _msg_parts.append(f"scd_scale={float(self._scd_scale.detach()):.3f}")
            print(", ".join(_msg_parts))

        return torch.cat(parts, dim=-1)

    def to(self, device):
        """Override to method to handle expert tracker device transfer."""
        super().to(device)
        if self.expert_tracker is not None:
            self.expert_tracker.to(device)
        return self
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to exclude expert_tracker from automatic collection.
        
        expert_tracker is handled separately in checkpoint code to avoid issues
        with PyTorch's state_dict protocol.
        """
        # Temporarily remove expert_tracker from _modules to exclude it from state_dict
        expert_tracker = None
        if hasattr(self, 'expert_tracker') and 'expert_tracker' in self._modules:
            expert_tracker = self._modules.pop('expert_tracker')
        
        # Get standard state_dict (without expert_tracker)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Restore expert_tracker to _modules
        if expert_tracker is not None:
            self._modules['expert_tracker'] = expert_tracker
        
        return state

