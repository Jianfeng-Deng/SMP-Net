import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import roc_auc_score, roc_curve
from losses.supcon_loss import supervised_contrastive_loss

from adaptation.online_bayesian_finetuning import OnlineBayesianFinetuner


def _build_domain_map():
    try:
        import config as cfg
        return {
            "CWRU": int(getattr(cfg, "CWRU_GK2_DOMAIN_ID", 0)),
            "PU": int(getattr(cfg, "PU_DOMAIN_ID", 2)),
            "HUST": int(getattr(cfg, "HUST_DOMAIN_ID", 3)),
            "HIT": int(getattr(cfg, "HIT_DOMAIN_ID", 4)),
            "SEU": int(getattr(cfg, "SEU_DOMAIN_ID", 5)),
        }
    except Exception:
        return {"CWRU": 0, "PU": 2, "HUST": 3, "HIT": 4, "SEU": 5}


class AnomalyDetector:
    def __init__(self, signal_encoder, spec_encoder, text_encoder, fusion_module, fusion_decoder, device,
                 enable_finetuning=False, finetune_samples=50, finetune_lr=1e-4, finetune_steps=3,
                 text_prototypes=None, decouple_spec=None, decouple_signal=None, wst_encoder=None, env_encoder=None, scd_encoder=None):
        """
        Keep original naming. fusion_decoder may be None if you don't have decoder.
        
        Args:
            enable_finetuning: 是否启用在线微调
            finetune_samples: 用于微调的样本数
            finetune_lr: 微调学习率
            finetune_steps: 微调步数
            decouple_spec: ATT396 FeatureDecoupler for spec features
            decouple_signal: ATT396 FeatureDecoupler for signal features
        """
        self.signal_encoder = signal_encoder
        self.spec_encoder = spec_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.fusion_decoder = fusion_decoder
        self.device = device
        self.decouple_spec = decouple_spec
        self.decouple_signal = decouple_signal
        self.wst_encoder = wst_encoder
        self._wst_bypass_active = wst_encoder is not None
        self.env_encoder = env_encoder
        self._env_bypass_active = env_encoder is not None
        self.scd_encoder = scd_encoder
        self._scd_bypass_active = scd_encoder is not None
        self.enable_finetuning = enable_finetuning
        self.finetune_samples = finetune_samples
        self.finetune_lr = finetune_lr
        self.finetune_steps = finetune_steps

        # ATT446/ATT449: cached bypass features for current batch
        self._last_z_wst = None
        self._last_z_env = None
        self._last_ratio = None

        # 缓存已微调的数据集，避免重复微调
        self._finetuned_datasets = set()
        # 缓存原始模型状态（用于恢复）
        self._original_fusion_state = None
        self._original_decoder_state = None
        # 缓存统一的text prototypes（避免重复构建）
        self._cached_text_prototypes = None
        # 可选：注入训练时保存到checkpoint的 prototypes（优先于纯prompt重建）
        self._external_text_prototypes = None
        # 用于目标域隔离评估的快照状态
        self._snapshot_fusion_state = None
        self._snapshot_decoder_state = None
        self._snapshot_spec_state = None
        # ATT191(B): cache LP-FT best states for multiple validation metrics (e.g., ce/spec_ce).
        self._lpft_best_metric_states = {}
        self._lpft_best_metric_values = {}
        self._lpft_selected_metric = None
        self._lpft_ema_state = None
        self._lpft_ema_metric = None
        self._lpft_ema_step = None
        self._lpft_retrain_full_state = None
        self._lpft_retrain_full_step = None
        if text_prototypes is not None:
            self.set_text_prototypes(text_prototypes)
        # Att78: current eval dataset name for domain-specific gate (e.g. CWRU relax Spec gate).
        self._current_eval_dataset_name = None

    def set_eval_dataset_name(self, name):
        """Set current evaluation dataset name (e.g. 'CWRU', 'PU') for domain-specific behavior (CollapseGate relax)."""
        self._current_eval_dataset_name = name

    def prepare_spec_batch(self, batch):
        """Load spec with runtime ablation gates (freq concat / MS-STFT blend when enabled)."""
        from utils.ablation_gating import load_spec_from_batch
        return load_spec_from_batch(batch, self.device, self)

    def _active_text_prototypes(self):
        """Return the most recent prototypes (external > cached)."""
        protos = self._external_text_prototypes if self._external_text_prototypes is not None else self._cached_text_prototypes
        if protos is None:
            return None
        # ATT372 rollback safety: only apply prompt offsets when explicitly enabled.
        _enable_prompt_tuning = False
        try:
            import config as _cfg_local
            _enable_prompt_tuning = bool(getattr(_cfg_local, "USE_TEXT_PROMPT_TUNING", False))
        except Exception:
            _enable_prompt_tuning = False
        if _enable_prompt_tuning and self.fusion_decoder is not None and hasattr(self.fusion_decoder, "apply_text_prompt_tuning"):
            try:
                return self.fusion_decoder.apply_text_prompt_tuning(protos, detach_output=True)
            except Exception:
                return protos
        return protos

    def _set_nf_calibrator_grad(self, requires_grad: bool):
        if self.fusion_decoder is not None:
            self.fusion_decoder.set_nf_calibrator_requires_grad(requires_grad)
    
    def _freeze_all_except_calibrator_and_lora(
        self,
        freeze: bool,
        allow_fusion_module=False,
        allow_prototype_residual=False,
        allow_decoder_adapters=False,
    ):
        """
        Freeze/unfreeze all fusion_module and fusion_decoder parameters except:
        - NF calibrator (scale/bias in fusion_decoder)
        - Domain-LoRA adapters (if enabled in encoders)
        - fusion_module (if allow_fusion_module=True, for supervised finetuning)
        - prototype_residual_* (shared/domain) (if allow_prototype_residual=True, for supervised finetuning)
        
        Args:
            freeze: If True, freeze non-exceptional params; if False, unfreeze all
            allow_fusion_module: If True, keep fusion_module trainable even when freeze=True
            allow_prototype_residual: If True, keep prototype_residual_* trainable even when freeze=True
        """
        # Freeze/unfreeze fusion_module (unless explicitly allowed)
        if allow_fusion_module and freeze:
            # Keep fusion_module trainable
            for param in self.fusion_module.parameters():
                param.requires_grad = True
        else:
            for param in self.fusion_module.parameters():
                param.requires_grad = not freeze
        
        # Freeze/unfreeze fusion_decoder, except NF calibrator and prototype_residual_*
        if self.fusion_decoder is not None:
            for name, param in self.fusion_decoder.named_parameters():
                # Keep calibrator always trainable (controlled by _set_nf_calibrator_grad separately)
                if 'nf_calibrator' in name:
                    continue  # Don't change, managed by _set_nf_calibrator_grad
                # Keep prototype_residual_* trainable if allowed
                elif 'prototype_residual' in name and allow_prototype_residual and freeze:
                    param.requires_grad = True
                # Keep small diagnosis adapters trainable if allowed (for target supervised finetune)
                elif allow_decoder_adapters and freeze and (('cls_adapter' in name) or ('diag_linear' in name)):
                    param.requires_grad = True
                else:
                    param.requires_grad = not freeze
        
        # Keep Domain-LoRA params trainable if freeze=True (i.e., adaptation mode)
        if freeze:
            # Unfreeze LoRA in signal_encoder if present
            if hasattr(self.signal_encoder, 'use_domain_lora') and self.signal_encoder.use_domain_lora:
                if hasattr(self.signal_encoder, 'domain_lora_fc'):
                    for param in self.signal_encoder.domain_lora_fc.parameters():
                        param.requires_grad = True
            
            # Unfreeze LoRA in spec_encoder if present
            if hasattr(self.spec_encoder, 'use_domain_lora') and self.spec_encoder.use_domain_lora:
                if hasattr(self.spec_encoder, 'domain_lora_agg'):
                    for param in self.spec_encoder.domain_lora_agg.parameters():
                        param.requires_grad = True
    
    def _fuse_embeddings(self, z_signal, z_spec, signal=None, signal_mask=None, spec=None, spec_mask=None):
        """
        统一的融合方法，支持多token和类感知推理一致化
        
        Args:
            z_signal: [B, D] 或 [B, T, D] signal embedding/tokens
            z_spec: [B, D] 或 [B, T, D] spec embedding/tokens
            signal, signal_mask, spec, spec_mask: 原始输入（用于多token模式）
        
        Returns:
            z_fusion: [B, D] fused embedding
        """
        # 检查是否使用多token模式（MoE 现已支持 return_tokens）
        is_moe = hasattr(self.spec_encoder, 'num_bands')
        use_multi_token = bool(getattr(self.fusion_module, 'use_multi_token', False))
        signal_token_mask = None
        spec_token_mask = None
        
        if use_multi_token and signal is not None:
            # 多token模式：从encoder获取token序列
            if hasattr(self.signal_encoder, 'use_domain_residual') and self.signal_encoder.use_domain_residual:
                z_signal_tokens, signal_token_mask = self.signal_encoder(signal, signal_mask, domain_ids=None, return_tokens=True)
            else:
                z_signal_tokens, signal_token_mask = self.signal_encoder(signal, signal_mask, return_tokens=True)
            
            if is_moe:
                # SpecEncoderMoE(return_tokens=True) -> (z_spec, z_spec_tokens, mask_tokens)
                _, z_spec_tokens, spec_token_mask = self.spec_encoder(spec, spec_mask, return_tokens=True)
            else:
                if hasattr(self.spec_encoder, 'use_domain_residual') and self.spec_encoder.use_domain_residual:
                    z_spec_tokens, spec_token_mask = self.spec_encoder(spec, spec_mask, domain_ids=None, return_tokens=True)
                else:
                    z_spec_tokens, spec_token_mask = self.spec_encoder(spec, spec_mask, return_tokens=True)
            
            z_signal_input = z_signal_tokens
            z_spec_input = z_spec_tokens
        else:
            # 单token模式：使用传入的embeddings
            z_signal_input = z_signal
            z_spec_input = z_spec

        # ===== Collapse-aware spec gating (Attempt 42) =====
        # Motivation: In Attempt 41, HUST showed severe spec embedding collapse:
        #   stdmean(z_spec) ≪ stdmean(z_sig), causing fusion negative transfer.
        # This is an EVAL/TARGET-ADAPT policy (not used by the main training loop),
        # implemented here so both evaluation and target adaptation share the same behavior.
        try:
            import config as cfg
            enable_gate = bool(getattr(cfg, "COLLAPSE_AWARE_SPEC_GATING", False))
            abs_th = float(getattr(cfg, "SPEC_STDMEAN_ABS_TH", 0.0))
            ratio_th = float(getattr(cfg, "SPEC_STDMEAN_RATIO_TH", 0.0))
            log_gate = bool(getattr(cfg, "LOG_COLLAPSE_AWARE_GATING", True))
            # Att78: relax threshold for CWRU single-domain eval (low variance may be domain effect, not collapse).
            if getattr(self, "_current_eval_dataset_name", None) == "CWRU" and bool(getattr(cfg, "TASK1_CWRU_RELAX_SPEC_GATE", False)):
                abs_th = abs_th * 0.5
        except Exception:
            enable_gate = False
            abs_th = 0.0
            ratio_th = 0.0
            log_gate = True

        if enable_gate:
            std_sig = None
            std_spec = None
            try:
                if isinstance(z_signal, torch.Tensor) and z_signal.dim() == 2:
                    # Keep gate metric in the same normalized space across training/eval paths.
                    z_signal_n = F.normalize(z_signal.float(), dim=-1)
                    std_sig = float(z_signal_n.std(dim=0, unbiased=False).mean().item())
                if isinstance(z_spec, torch.Tensor) and z_spec.dim() == 2:
                    z_spec_n = F.normalize(z_spec.float(), dim=-1)
                    std_spec = float(z_spec_n.std(dim=0, unbiased=False).mean().item())
            except Exception:
                std_sig, std_spec = None, None

            gate = False
            try:
                if std_spec is not None and abs_th > 0 and std_spec < abs_th:
                    gate = True
                if (not gate) and (std_sig is not None) and (std_spec is not None) and ratio_th > 0 and std_sig > 1e-12:
                    if (std_spec / std_sig) < ratio_th:
                        gate = True
            except Exception:
                gate = False

            if gate:
                try:
                    if isinstance(z_spec_input, torch.Tensor):
                        z_spec_input = torch.zeros_like(z_spec_input)
                except Exception:
                    pass
            if log_gate:
                # Avoid per-call spam: print at most 1 time when not gating,
                # and at most 5 times when gating is active (enough to verify behavior).
                try:
                    if not hasattr(self, "_collapse_gate_log_n"):
                        self._collapse_gate_log_n = 0
                    limit = 5 if bool(gate) else 1
                    if int(self._collapse_gate_log_n) < int(limit):
                        rs = (std_spec / std_sig) if (std_sig is not None and std_spec is not None and std_sig > 1e-12) else None
                        print(
                            f"[CollapseGate] spec_gated={int(gate)} "
                            f"std_sig={std_sig if std_sig is not None else float('nan'):.6f} "
                            f"std_spec={std_spec if std_spec is not None else float('nan'):.6f} "
                            f"ratio={rs if rs is not None else float('nan'):.6f} "
                            f"(abs_th={abs_th:.6f}, ratio_th={ratio_th:.3f})"
                        )
                        self._collapse_gate_log_n += 1
                except Exception:
                    pass
        
        # 融合（支持类感知推理一致化）
        if hasattr(self.fusion_module, 'domain_embedding'):
            # DomainAwareCrossAttentionFusion
            use_class_aware = getattr(self.fusion_module, 'use_class_aware', False)
            text_prototypes = self._external_text_prototypes or self._cached_text_prototypes
            
            if use_class_aware and self.fusion_decoder is not None:
                # 两步融合：先普通融合得到预测，再用预测做类感知融合
                z_fusion_step1 = self.fusion_module(
                    z_signal_input, z_spec_input,
                    domain_ids=None, text_prototypes=text_prototypes,
                    class_ids=None, class_probs=None,
                    signal_token_mask=signal_token_mask,
                    spec_token_mask=spec_token_mask
                )
                
                # 获取预测的类别概率分布
                text_prototypes = self._active_text_prototypes()
                if hasattr(self.fusion_decoder, 'hierarchical') and self.fusion_decoder.hierarchical:
                    logits_step1, _ = self.fusion_decoder(z_fusion_step1, text_prototypes=text_prototypes, return_parts=True)
                else:
                    logits_step1 = self.fusion_decoder(z_fusion_step1, text_prototypes=text_prototypes)
                
                class_probs = F.softmax(logits_step1, dim=1)  # [B, C]
                
                # 第二步：使用预测分布做类感知融合
                z_fusion = self.fusion_module(
                    z_signal_input, z_spec_input,
                    domain_ids=None, text_prototypes=text_prototypes,
                    class_ids=None, class_probs=class_probs,
                    signal_token_mask=signal_token_mask,
                    spec_token_mask=spec_token_mask
                )
            else:
                # 不使用类感知：直接融合
                z_fusion = self.fusion_module(
                    z_signal_input, z_spec_input,
                    domain_ids=None, text_prototypes=text_prototypes,
                    class_ids=None, class_probs=None,
                    signal_token_mask=signal_token_mask,
                    spec_token_mask=spec_token_mask
                )
        else:
            # CrossGatedFusion
            z_fusion = self.fusion_module(z_signal_input, z_spec_input)
        
        return z_fusion
    
    def reset_finetuning_state(self):
        """
        重置微调状态，清除已微调的数据集记录，恢复原始模型状态
        用于在不同数据集之间切换时重置状态
        """
        self._finetuned_datasets.clear()
        if self._original_fusion_state is not None:
            self.fusion_module.load_state_dict(self._original_fusion_state)
        if self._original_decoder_state is not None and self.fusion_decoder is not None:
            self.fusion_decoder.load_state_dict(self._original_decoder_state)
        print("[AnomalyDetector] Finetuning state reset")
    
    def clear_prototype_cache(self):
        """
        清除prototype缓存，强制重新构建
        用于在需要更新prototypes时调用
        """
        self._cached_text_prototypes = None

    def set_text_prototypes(self, text_prototypes):
        """
        注入训练时保存的 text_prototypes（例如贝叶斯后验均值），用于 semantic / both / legacy scoring。
        text_prototypes: Dict[int, Tensor[D]]
        """
        if text_prototypes is None:
            self._external_text_prototypes = None
            self.clear_prototype_cache()
            return

        protos = {}
        for k, v in text_prototypes.items():
            if v is None:
                continue
            protos[int(k)] = v.detach().cpu()
        self._external_text_prototypes = protos
        self.clear_prototype_cache()

    # ======================
    # Legacy: build single normal prototype (your original method)
    # ======================
    def build_normal_text_prototype(self):
        """
        Build the single 'normal' text prototype (same behaviour as your original code).
        Returns a tensor of shape [D].
        """
        self.text_encoder.eval()

        prompts = [
            "A healthy bearing operating under normal conditions with low-amplitude, Gaussian-like background noise.",
            "A pristine bearing system exhibiting a smooth vibration signature devoid of periodic impulses or harmonic structures.",
            "Normal machinery operation characterized by stationary random vibration and the absence of fault characteristic frequencies.",
            "A time-frequency representation showing a uniform background texture with no distinct vertical lines or energy concentrations.",
            "A bearing system in a healthy state, displaying consistent entropy levels without rotational modulation signatures."
        ]

        with torch.no_grad():
            embs = []
            for p in prompts:
                emb = self.text_encoder([p]).to(self.device)  # expect [1, D]
                embs.append(F.normalize(emb, dim=-1))
            proto = torch.mean(torch.stack(embs), dim=0)  # [1, D]

        return F.normalize(proto.squeeze(0), dim=-1)  # [D]

    # ======================
    # New: build normal + fault text prototypes (for semantic scoring)
    # ======================
    def build_text_prototypes(self):
        """
        Build both a normal prototype and several fault prototypes.
        Returns (normal_proto [D], fault_protos [K, D]).
        """
        self.text_encoder.eval()

        normal_prompts = [
            "A healthy bearing operating under normal conditions with low-amplitude, Gaussian-like background noise.",
            "A pristine bearing system exhibiting a smooth vibration signature devoid of periodic impulses or harmonic structures.",
            "Normal machinery operation characterized by stationary random vibration and the absence of fault characteristic frequencies."
        ]

        fault_prompts = [
            "A bearing with an inner race fault characterized by periodic impulsive vibration patterns.",
            "A bearing exhibiting an outer race defect with repetitive shock responses and localized spectral energy.",
            "A rolling element bearing fault showing irregular impulsive components and elevated vibration energy.",
            "A defective bearing operating under abnormal conditions with non-stationary vibration and fault characteristic frequencies."
        ]

        with torch.no_grad():
            normal_embs = []
            for p in normal_prompts:
                emb = self.text_encoder([p]).to(self.device)
                normal_embs.append(F.normalize(emb, dim=-1))
            normal_proto = torch.mean(torch.stack(normal_embs), dim=0)
            normal_proto = F.normalize(normal_proto.squeeze(0), dim=-1)  # [D]

            fault_embs = []
            for p in fault_prompts:
                emb = self.text_encoder([p]).to(self.device)
                fault_embs.append(F.normalize(emb, dim=-1).squeeze(0))  # [D]
            fault_protos = torch.stack(fault_embs, dim=0)  # [K, D]
            fault_protos = F.normalize(fault_protos, dim=-1)

        return normal_proto, fault_protos

    # ======================
    # Build all text prototypes (for online finetuning)
    # ======================
    def _build_all_text_prototypes(self):
        """
        Build text prototypes for all classes (0: normal, 1-3: faults).
        Returns dict {label: tensor[D]}.
        """
        self.text_encoder.eval()
        
        # Import config to get FAULT_PROMPT_TEMPLATES
        try:
            import config
            fault_prompt_templates = config.FAULT_PROMPT_TEMPLATES
        except ImportError:
            # Fallback: use hardcoded prompts if config not available
            fault_prompt_templates = {
                0: ["A healthy bearing operating under normal conditions."],
                1: ["An inner race fault characterized by periodic impulses."],
                2: ["An outer race fault with stable periodic impulses."],
                3: ["A rolling element fault showing irregular impulses."]
            }
        
        text_prototypes = {}
        
        with torch.no_grad():
            for label, prompts in fault_prompt_templates.items():
                embs = []
                for p in prompts:
                    emb = self.text_encoder([p]).to(self.device)
                    embs.append(F.normalize(emb.squeeze(0), dim=0))
                
                # Average all prompts for this class
                proto = torch.stack(embs, dim=0).mean(dim=0)  # [D]
                proto = F.normalize(proto, dim=0)
                text_prototypes[label] = proto
        
        try:
            from utils.build_text_prototypes_main import apply_semantic_proto_ablation
            text_prototypes = apply_semantic_proto_ablation(text_prototypes)
        except Exception as exc:
            print(f"[SemanticProtoAblation] skipped in anomaly detector: {exc}")
        return text_prototypes

    # ======================
    # Apply online finetuning
    # ======================
    def _apply_online_finetuning(self, dataloader, dataset_name=None):
        """
        Apply online finetuning on fusion_module and fusion_decoder using samples from dataloader.
        无监督微调目标：NF置信校准、跨模态一致性、gate自洽
        **现在只更新 NF calibrator + Domain-LoRA (如果启用)**
        
        Args:
            dataloader: DataLoader containing samples for finetuning
            dataset_name: str, 数据集名称（用于日志）
        
        Returns:
            bool: True if finetuning succeeded, False otherwise
        """
        if not self.enable_finetuning or self.finetune_samples <= 0:
            return False
        
        if self.fusion_decoder is None:
            print("[Online Finetuning] Skipped: fusion_decoder is None")
            return False

        # Stability guard: HUST online unsupervised finetuning was observed to hurt NF/diagnosis in logs.
        # Apply the same policy switch as the two-stage target adaptation path.
        try:
            import config as cfg
            skip_hust = bool(getattr(cfg, "TARGET_SKIP_HUST_UNSUP_ADAPT", False))
        except Exception:
            skip_hust = False
        if skip_hust and (dataset_name is not None) and str(dataset_name).upper() == "HUST":
            print("[Online Finetuning] Skipped for HUST (TARGET_SKIP_HUST_UNSUP_ADAPT=True)")
            return False
        
        dataset_str = f" for {dataset_name}" if dataset_name else ""
        print(f"[Online Finetuning] Starting{dataset_str} with {self.finetune_samples} samples...")
        
        if self._original_fusion_state is None:
            self._original_fusion_state = self.fusion_module.state_dict().copy()
        if self._original_decoder_state is None:
            self._original_decoder_state = self.fusion_decoder.state_dict().copy()
        
        original_fusion_state = self._original_fusion_state
        original_decoder_state = self._original_decoder_state

        # Collect a small subset of batches for finetuning
        finetune_batches = []
        count = 0
        for batch in dataloader:
            if count < self.finetune_samples:
                finetune_batches.append(batch)
                count += batch["signal"].size(0)
            else:
                break

        if len(finetune_batches) == 0:
            print("[Online Finetuning] No batches collected for finetuning")
            return False

        if self._external_text_prototypes is not None:
            text_prototypes = self._external_text_prototypes
            print(f"[Unsupervised Adaptation] Using checkpoint text prototypes: {list(text_prototypes.keys())}")
        elif self._cached_text_prototypes is not None:
            text_prototypes = self._cached_text_prototypes
        else:
            text_prototypes = self._build_all_text_prototypes()
            self._cached_text_prototypes = text_prototypes
            
        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.train()
        self.fusion_decoder.train()
            
        finetune_inputs = []
        warned_nonfinite = False

        for batch in finetune_batches:
            signal = batch["signal"].to(self.device)
            signal_mask = batch["signal_mask"].to(self.device)
            spec = self.prepare_spec_batch(batch)
            spec_mask = batch["spec_mask"].to(self.device)

            # Skip non-finite batches to avoid poisoning adaptation (common on HIT finetune subsets)
            if not torch.isfinite(signal).all() or not torch.isfinite(spec).all():
                if not warned_nonfinite:
                    def _finite_stats(x):
                        x_det = x.detach()
                        return {
                            "nan": int(torch.isnan(x_det).sum().item()),
                            "inf": int(torch.isinf(x_det).sum().item()),
                            "min": float(torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0).min().item()),
                            "max": float(torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0).max().item()),
                        }

                    print(f"[Online Finetuning] Warning: Non-finite inputs detected{dataset_str}. Skipping batch.")
                    print(f"  signal stats: {_finite_stats(signal)}")
                    print(f"  spec   stats: {_finite_stats(spec)}")
                    warned_nonfinite = True
                continue

            with torch.no_grad():
                z_sig = self.signal_encoder(signal, signal_mask)
                z_spec = self.spec_encoder(spec, spec_mask)
                # ATT379e: removed F.normalize for train-eval consistency

            if not torch.isfinite(z_sig).all() or not torch.isfinite(z_spec).all():
                if not warned_nonfinite:
                    print(f"[Online Finetuning] Warning: Non-finite encoder outputs detected{dataset_str}. Skipping batch.")
                    warned_nonfinite = True
                continue

            finetune_inputs.append((z_sig.detach(), z_spec.detach()))

        finetuner = OnlineBayesianFinetuner(
            lr=self.finetune_lr,
            num_steps=self.finetune_steps,
            lambda_uncertainty=0.1,
            max_update_norm=0.01
        )

        try:
            # Only enable NF calibrator for unsupervised adaptation
            self._set_nf_calibrator_grad(True)
            # Freeze all other fusion_module and fusion_decoder params
            self._freeze_all_except_calibrator_and_lora(freeze=True)
            
            if len(finetune_inputs) == 0:
                print(f"[Online Finetuning] Skipped{dataset_str}: no finite batches collected")
                return False

            self.fusion_module, self.fusion_decoder = finetuner.finetune_with_recompute(
                self.fusion_module, self.fusion_decoder, text_prototypes, finetune_inputs, self.device
            )
            
            print(f"[Online Finetuning] Completed {self.finetune_steps} steps{dataset_str} (only NF calibrator + LoRA updated)")
            return True
        except Exception as e:
            print(f"[Online Finetuning] Failed: {e}, using original models")
            self.fusion_module.load_state_dict(original_fusion_state)
            self.fusion_decoder.load_state_dict(original_decoder_state)
            return False
        finally:
            self._set_nf_calibrator_grad(False)
            self._freeze_all_except_calibrator_and_lora(freeze=False)

    # ======================
    # Snapshot/Restore for isolated domain evaluation
    # ======================
    def snapshot_fusion_state(self):
        """
        创建 spec_encoder / fusion_module / fusion_decoder 的快照（深拷贝）
        用于目标域隔离评估：每个域使用独立的模型副本
        """
        if self.spec_encoder is not None:
            self._snapshot_spec_state = {
                k: v.clone().detach() for k, v in self.spec_encoder.state_dict().items()
            }
        else:
            self._snapshot_spec_state = None
        self._snapshot_fusion_state = {
            k: v.clone().detach() for k, v in self.fusion_module.state_dict().items()
        }
        if self.fusion_decoder is not None:
            self._snapshot_decoder_state = {
                k: v.clone().detach() for k, v in self.fusion_decoder.state_dict().items()
            }
        else:
            self._snapshot_decoder_state = None
    
    def restore_fusion_state(self):
        """
        恢复 spec_encoder / fusion_module / fusion_decoder 到快照状态
        """
        if self._snapshot_fusion_state is None:
            raise RuntimeError("Cannot restore: no snapshot state saved. Call snapshot_fusion_state() first.")
        if self._snapshot_spec_state is not None and self.spec_encoder is not None:
            self.spec_encoder.load_state_dict(self._snapshot_spec_state)
        self.fusion_module.load_state_dict(self._snapshot_fusion_state)
        if self._snapshot_decoder_state is not None and self.fusion_decoder is not None:
            self.fusion_decoder.load_state_dict(self._snapshot_decoder_state)

    def restore_lpft_best_state(self, metric="ce"):
        """
        Restore LP-FT best snapshot tracked by validation metric (e.g., ce/spec_ce).
        Returns True on success, False otherwise.
        """
        key = str(metric).strip().lower()
        pack = self._lpft_best_metric_states.get(key, None)
        if pack is None:
            return False
        dec_state = pack.get("decoder", None)
        spec_state = pack.get("spec", None)
        ok = False
        try:
            if dec_state is not None and self.fusion_decoder is not None:
                self.fusion_decoder.load_state_dict(dec_state)
                ok = True
            if spec_state is not None and self.spec_encoder is not None:
                self.spec_encoder.load_state_dict(spec_state)
                ok = True
        except Exception:
            return False
        return bool(ok)

    def get_lpft_best_metric_value(self, metric="ce"):
        key = str(metric).strip().lower()
        return self._lpft_best_metric_values.get(key, None)

    def restore_lpft_ema_state(self):
        """
        Restore LP-FT EMA snapshot for post-LPFT evaluation.
        Returns True on success, False otherwise.
        """
        pack = self._lpft_ema_state
        if not isinstance(pack, dict):
            return False
        dec_state = pack.get("decoder", None)
        spec_state = pack.get("spec", None)
        ok = False
        try:
            if dec_state is not None and self.fusion_decoder is not None:
                self.fusion_decoder.load_state_dict(dec_state)
                ok = True
            if spec_state is not None and self.spec_encoder is not None:
                self.spec_encoder.load_state_dict(spec_state)
                ok = True
        except Exception:
            return False
        return bool(ok)

    def restore_lpft_retrain_full_state(self):
        """
        Restore retrain-on-full LP-FT snapshot for post-LPFT non-parametric evaluation.
        Returns True on success, False otherwise.
        """
        pack = self._lpft_retrain_full_state
        if not isinstance(pack, dict):
            # ATT240(S3): graceful fallback when retrain-full snapshot is absent.
            # This prevents HIT/HUST non-parametric pathways from being skipped
            # due to missing cache in edge runs.
            try:
                for _metric in ("ce", "spec_ce"):
                    if self.restore_lpft_best_metric_state(_metric):
                        return True
            except Exception:
                pass
            try:
                if self.restore_lpft_ema_state():
                    return True
            except Exception:
                pass
            return False
        dec_state = pack.get("decoder", None)
        spec_state = pack.get("spec", None)
        ok = False
        try:
            if dec_state is not None and self.fusion_decoder is not None:
                self.fusion_decoder.load_state_dict(dec_state)
                ok = True
            if spec_state is not None and self.spec_encoder is not None:
                self.spec_encoder.load_state_dict(spec_state)
                ok = True
        except Exception:
            return False
        return bool(ok)

    # ======================
    # Target-domain adaptation (two-stage: unsupervised -> supervised)
    # ======================
    def unsupervised_adapt_on_loader(self, finetune_loader, dataset_name=None):
        """
        在目标域的无标签数据上进行无监督适配
        
        Args:
            finetune_loader: DataLoader for finetuning (5% of target domain)
            dataset_name: str, 数据集名称（用于日志）
        
        Returns:
            bool: True if adaptation succeeded
        """
        if self.fusion_decoder is None:
            print("[Unsupervised Adaptation] Skipped: fusion_decoder is None")
            return False

        # Stability guard: HUST unsupervised adaptation often hurts NF/diagnosis (observed in training logs).
        # Keep this as an evaluation-time policy switch via config.
        try:
            import config as cfg
            skip_hust = bool(getattr(cfg, "TARGET_SKIP_HUST_UNSUP_ADAPT", False))
        except Exception:
            skip_hust = False
        if skip_hust and (dataset_name is not None) and str(dataset_name).upper() == "HUST":
            print("[Unsupervised Adaptation] Skipped for HUST (TARGET_SKIP_HUST_UNSUP_ADAPT=True)")
            return False
        
        # 使用已有的无监督微调逻辑，但确保使用 checkpoint prototypes
        # 如果 _external_text_prototypes 存在（来自checkpoint），优先使用
        return self._apply_online_finetuning(finetune_loader, dataset_name=dataset_name)
    
    def supervised_finetune_on_loader(self, finetune_loader, dataset_name=None, num_classes_eval=4,
                                     num_steps=None, lr=None, nf_weight=0.5, diag_weight=1.0,
                                     unlabeled_loader=None, allow_full_retrain=True, force_no_val_es=False,
                                     holdout_test_loader=None, lpft_curve=None, record_test_every=None,
                                     lpft_curve_eval_fn=None):
        """
        在目标域的有标签数据上进行监督微调（支持 LP-FT + 渐进式 backbone 解冻）
        
        Args:
            finetune_loader: DataLoader containing labeled samples for finetuning (5% of target domain)
            dataset_name: str, 数据集名称（用于日志）
            num_classes_eval: int, 诊断类别数（HUST=4, HIT=3）
            num_steps: int, 微调步数（如果None，使用self.finetune_steps）
            lr: float, 学习率（如果None，使用self.finetune_lr）
            nf_weight: float, NF loss 权重
            diag_weight: float, 诊断 loss 权重
        
        Returns:
            bool: True if finetuning succeeded
        """
        if self.fusion_decoder is None:
            print("[Supervised Finetune] Skipped: fusion_decoder is None")
            return False
        
        num_steps = num_steps or self.finetune_steps
        lr = lr or self.finetune_lr
        dataset_str = f" for {dataset_name}" if dataset_name else ""
        _dataset_upper = str(dataset_name).upper() if dataset_name is not None else ""
        _dataset_name_norm = str(dataset_name).strip().lower() if dataset_name is not None else ""
        _is_selftrain_call = ("SELFTRAIN" in _dataset_upper)
        _is_retrain_full_call = ("fullretrain" in _dataset_name_norm)
        if (not _is_retrain_full_call) and (not _is_selftrain_call):
            # Prevent stale retrain-full snapshots from leaking into future evaluations.
            self._lpft_retrain_full_state = None
            self._lpft_retrain_full_step = None
        
        # 保存当前状态（用于失败恢复）
        current_fusion_state = self.fusion_module.state_dict().copy()
        current_decoder_state = self.fusion_decoder.state_dict().copy() if self.fusion_decoder is not None else None
        current_spec_state = self.spec_encoder.state_dict().copy() if self.spec_encoder is not None else None
        
        try:
            # Detect decoder type
            is_hier = bool(getattr(self.fusion_decoder, "hierarchical", False))
            is_flat = not is_hier
            # Flat-head LP defaults (Attempt 42): override steps/lr if provided in config.
            if is_flat:
                try:
                    import config as cfg
                    if not force_no_val_es:
                        num_steps = int(getattr(cfg, "TARGET_SUP_LP_STEPS", num_steps))
                    if not force_no_val_es:
                        lr = float(getattr(cfg, "TARGET_SUP_LP_LR", lr))
                except Exception:
                    pass
            # Log effective values after flat-head config override (prevents "steps=50 but run 200" mismatch).
            print(
                f"[Supervised Finetune] Starting{dataset_str} with labeled data "
                f"({num_classes_eval}-class), steps={num_steps}, lr={lr}"
            )

            # Optional (Attempt 45+): override flat cosine temperature during LP-FT.
            # Default None keeps existing behavior unchanged.
            _lp_temp_override = None
            _lp_temp_orig = None
            if is_flat:
                try:
                    import config as cfg
                    _lp_temp_override = getattr(cfg, "TARGET_SUP_LP_TEMPERATURE", None)
                    if _lp_temp_override is not None:
                        _lp_temp_override = float(_lp_temp_override)
                except Exception:
                    _lp_temp_override = None

            try:
                import config as cfg
                include_spec_head = bool(getattr(cfg, "TARGET_SUP_INCLUDE_SPEC_HEAD", True))
            except Exception:
                include_spec_head = True
            # ATT180: progressive unfreezing (flat LP-FT only).
            # Default behavior remains decoder/spec_head-only unless explicitly enabled in config.
            try:
                import config as cfg
                lp_prog_enable = bool(getattr(cfg, "TARGET_SUP_LP_PROGRESSIVE_UNFREEZE_ENABLE", False))
                # Keep raw config intent before any path-level safety guards mutate lp_prog_enable.
                lp_prog_enable_cfg = bool(lp_prog_enable)
                lp_unfreeze_stage2_start = int(getattr(cfg, "TARGET_SUP_LP_UNFREEZE_STAGE2_START", 50))
                lp_unfreeze_stage3_start = int(getattr(cfg, "TARGET_SUP_LP_UNFREEZE_STAGE3_START", 100))
                lp_backbone_ft_enable = bool(getattr(cfg, "TARGET_SUP_LP_BACKBONE_FT_ENABLE", False))
                lp_unfreeze_stage4_start = int(getattr(cfg, "TARGET_SUP_LP_UNFREEZE_STAGE4_START", 180))
                lp_l2sp_enable = bool(getattr(cfg, "TARGET_SUP_LP_L2SP_ENABLE", False))
                lp_l2sp_weight = float(getattr(cfg, "TARGET_SUP_LP_L2SP_WEIGHT", 0.01))
                lp_backbone_lr_mult = float(getattr(cfg, "TARGET_SUP_LP_BACKBONE_LR_MULT", 0.10))
                lp_backbone_stage_warmup_enable = bool(
                    getattr(cfg, "TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_ENABLE", False)
                )
                lp_backbone_stage_warmup_steps = int(
                    getattr(cfg, "TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_STEPS", 10)
                )
                lp_backbone_stage_warmup_min_scale = float(
                    getattr(cfg, "TARGET_SUP_LP_BACKBONE_STAGE_WARMUP_MIN_SCALE", 0.2)
                )
                lp_disable_prog_on_selftrain = bool(
                    getattr(cfg, "TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SELFTRAIN", True)
                )
                lp_disable_prog_on_task1 = bool(
                    getattr(cfg, "TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_TASK1", True)
                )
                lp_disable_prog_on_hit = bool(
                    getattr(cfg, "TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_HIT", False)
                )
                lp_disable_prog_on_retrain_full = bool(
                    getattr(cfg, "TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_RETRAIN_FULL", False)
                )
                lp_retrain_full_allow_prog_flat = bool(
                    getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_ALLOW_PROGRESSIVE_UNFREEZE_FLAT", True)
                )
                lp_disable_prog_tinyshot_threshold = int(
                    getattr(cfg, "TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_TINYSHOT_THRESHOLD", 40)
                )
                lp_sam_enable = bool(getattr(cfg, "TARGET_SUP_LP_SAM_ENABLE", False))
                lp_sam_rho = float(getattr(cfg, "TARGET_SUP_LP_SAM_RHO", 0.05))
                lp_sam_stage1_only = bool(getattr(cfg, "TARGET_SUP_LP_SAM_STAGE1_ONLY", True))
                lp_stage_scale_by_samples = bool(getattr(cfg, "TARGET_SUP_LP_STAGE_SCALE_BY_SAMPLES", False))
                lp_stage_scale_threshold = int(getattr(cfg, "TARGET_SUP_LP_STAGE_SCALE_SAMPLE_THRESHOLD", 50))
                lp_stage_scale_min_ratio = float(getattr(cfg, "TARGET_SUP_LP_STAGE_SCALE_MIN_RATIO", 0.25))
                lp_stage_scale_min_stage2_steps = int(getattr(cfg, "TARGET_SUP_LP_STAGE_SCALE_MIN_STAGE2_STEPS", 1))
                lp_l2sp_use_global_anchor = bool(getattr(cfg, "TARGET_SUP_LP_L2SP_USE_GLOBAL_ANCHOR", False))
                lp_aug_enable = bool(getattr(cfg, "TARGET_SUP_LP_AUG_ENABLE", False))
                lp_aug_prob = float(getattr(cfg, "TARGET_SUP_LP_AUG_PROB", 0.7))
                lp_aug_spec_noise_std = float(getattr(cfg, "TARGET_SUP_LP_AUG_SPEC_NOISE_STD", 0.02))
                lp_aug_signal_noise_std = float(getattr(cfg, "TARGET_SUP_LP_AUG_SIGNAL_NOISE_STD", 0.01))
                lp_aug_freq_mask_ratio = float(getattr(cfg, "TARGET_SUP_LP_AUG_FREQ_MASK_RATIO", 0.08))
                lp_aug_time_mask_ratio = float(getattr(cfg, "TARGET_SUP_LP_AUG_TIME_MASK_RATIO", 0.08))
                lp_aug_num_freq_masks = int(getattr(cfg, "TARGET_SUP_LP_AUG_NUM_FREQ_MASKS", 1))
                lp_aug_num_time_masks = int(getattr(cfg, "TARGET_SUP_LP_AUG_NUM_TIME_MASKS", 1))
            except Exception:
                lp_prog_enable = False
                lp_prog_enable_cfg = False
                lp_unfreeze_stage2_start = 50
                lp_unfreeze_stage3_start = 100
                lp_backbone_ft_enable = False
                lp_unfreeze_stage4_start = 180
                lp_l2sp_enable = False
                lp_l2sp_weight = 0.01
                lp_backbone_lr_mult = 0.10
                lp_backbone_stage_warmup_enable = False
                lp_backbone_stage_warmup_steps = 10
                lp_backbone_stage_warmup_min_scale = 0.2
                lp_disable_prog_on_selftrain = True
                lp_disable_prog_on_task1 = True
                lp_disable_prog_on_hit = False
                lp_disable_prog_on_retrain_full = False
                lp_retrain_full_allow_prog_flat = True
                lp_disable_prog_tinyshot_threshold = 40
                lp_sam_enable = False
                lp_sam_rho = 0.05
                lp_sam_stage1_only = True
                lp_stage_scale_by_samples = False
                lp_stage_scale_threshold = 50
                lp_stage_scale_min_ratio = 0.25
                lp_stage_scale_min_stage2_steps = 1
                lp_l2sp_use_global_anchor = False
                lp_aug_enable = False
                lp_aug_prob = 0.7
                lp_aug_spec_noise_std = 0.02
                lp_aug_signal_noise_std = 0.01
                lp_aug_freq_mask_ratio = 0.08
                lp_aug_time_mask_ratio = 0.08
                lp_aug_num_freq_masks = 1
                lp_aug_num_time_masks = 1

            # ATT202(A): self-training pseudo-label noise + progressive unfreeze can destabilize backbone.
            # Keep self-training on LP (head-only) path by default unless explicitly disabled in config.
            if _is_selftrain_call and lp_disable_prog_on_selftrain:
                lp_prog_enable = False
                print(
                    f"[Supervised Finetune] Progressive unfreeze disabled for self-train path{dataset_str} "
                    f"(TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_SELFTRAIN=1)"
                )
            else:
                _is_task1_call = ("task1" in _dataset_name_norm)
                _lp_dataset_samples = None
                try:
                    if finetune_loader is not None and hasattr(finetune_loader, "dataset") and finetune_loader.dataset is not None:
                        _lp_dataset_samples = int(len(finetune_loader.dataset))
                except Exception:
                    _lp_dataset_samples = None

                if lp_prog_enable and lp_disable_prog_on_task1 and _is_task1_call:
                    lp_prog_enable = False
                    print(
                        f"[Supervised Finetune] Progressive unfreeze disabled for Task1 path{dataset_str} "
                        f"(TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_TASK1=1)"
                    )
                elif lp_prog_enable and lp_disable_prog_on_hit and _dataset_name_norm == "hit":
                    lp_prog_enable = False
                    print(
                        f"[Supervised Finetune] Progressive unfreeze disabled for HIT path{dataset_str} "
                        f"(TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_HIT=1)"
                    )
                elif lp_prog_enable and lp_disable_prog_on_retrain_full and _is_retrain_full_call:
                    _keep_prog_for_flat = bool(is_flat and lp_retrain_full_allow_prog_flat)
                    if not _keep_prog_for_flat:
                        lp_prog_enable = False
                        print(
                            f"[Supervised Finetune] Progressive unfreeze disabled for retrain-full path{dataset_str} "
                            f"(TARGET_SUP_LP_DISABLE_PROGRESSIVE_UNFREEZE_ON_RETRAIN_FULL=1)"
                        )
                    else:
                        print(
                            f"[Supervised Finetune] Progressive unfreeze kept for flat retrain-full path{dataset_str} "
                            f"(TARGET_SUP_LP_RETRAIN_FULL_ALLOW_PROGRESSIVE_UNFREEZE_FLAT=1)"
                        )
                elif (
                    lp_prog_enable
                    and int(lp_disable_prog_tinyshot_threshold) > 0
                    and (_lp_dataset_samples is not None)
                    and (_lp_dataset_samples < int(lp_disable_prog_tinyshot_threshold))
                ):
                    lp_prog_enable = False
                    print(
                        f"[Supervised Finetune] Progressive unfreeze disabled by tiny-shot guard{dataset_str}: "
                        f"n_train={_lp_dataset_samples} < threshold={int(lp_disable_prog_tinyshot_threshold)}"
                    )

            # Set to train mode
            self.signal_encoder.eval()  # encoders stay frozen in target finetune (LP-FT trains decoder only by default)
            self.spec_encoder.eval()
            self.fusion_module.eval()
            self.fusion_decoder.train()
            
            # Enable small, stable adaptation set:
            # - NF calibrator (optional but helps NF)
            # - Domain-LoRA (if enabled)
            # - prototype_residual_domain (CoCoOp-like, for target-domain labeled tuning)
            #
            # IMPORTANT: do NOT train fusion_module by default here.
            # In prior logs this expanded trainable params to ~11.8M and often *hurt* HIT diagnosis.
            # NF calibrator is only meaningful for hierarchical decoders.
            self._set_nf_calibrator_grad(bool(is_hier))

            # Freeze everything else
            self._freeze_all_except_calibrator_and_lora(
                freeze=True,
                allow_fusion_module=False,
                allow_prototype_residual=bool(not is_flat),  # flat LP: keep prototype residual frozen
                allow_decoder_adapters=bool(not is_flat),     # flat LP: keep adapters frozen
            )

            # IMPORTANT: _freeze_all_except_calibrator_and_lora(...) may (re)freeze decoder params.
            # Configure per-decoder trainable knobs AFTER the freeze call so that LP-FT / finetune
            # policy is actually in effect.
            lp_full_backbone_direct = False
            if is_flat:
                # ===== Attempt 42 (LP-FT): linear probe for flat head =====
                # Train only the flat cosine classifier (and optional flat multi-center extras).
                if hasattr(self.fusion_decoder, "set_flat_classifier_requires_grad"):
                    self.fusion_decoder.set_flat_classifier_requires_grad(True)
                # Keep other adaptation knobs off to make this change falsifiable.
                if hasattr(self.fusion_decoder, "set_prototype_residual_shared_requires_grad"):
                    self.fusion_decoder.set_prototype_residual_shared_requires_grad(False)
                if hasattr(self.fusion_decoder, "set_prototype_residual_domain_requires_grad"):
                    self.fusion_decoder.set_prototype_residual_domain_requires_grad(False)
                if hasattr(self.fusion_decoder, "set_cls_adapter_requires_grad"):
                    self.fusion_decoder.set_cls_adapter_requires_grad(False)
                if hasattr(self.fusion_decoder, "set_diag_linear_requires_grad"):
                    self.fusion_decoder.set_diag_linear_requires_grad(False)

                # Flat LP: explicitly freeze LoRA (we only want the classifier to move).
                try:
                    if hasattr(self.signal_encoder, 'use_domain_lora') and self.signal_encoder.use_domain_lora and hasattr(self.signal_encoder, 'domain_lora_fc'):
                        for p in self.signal_encoder.domain_lora_fc.parameters():
                            p.requires_grad = False
                    if hasattr(self.spec_encoder, 'use_domain_lora') and self.spec_encoder.use_domain_lora and hasattr(self.spec_encoder, 'domain_lora_agg'):
                        for p in self.spec_encoder.domain_lora_agg.parameters():
                            p.requires_grad = False
                except Exception:
                    pass

                # Optional: include SpecHead classifier adaptation for target-domain LP-FT.
                # Only classifier head is enabled; spec backbone remains frozen.
                if include_spec_head and getattr(self.spec_encoder, "has_spec_head", False):
                    try:
                        if self.spec_encoder.classifier is not None:
                            for p in self.spec_encoder.classifier.parameters():
                                p.requires_grad = True
                            self.spec_encoder.classifier.train()
                        if getattr(self.spec_encoder, "spec_prototypes", None) is not None:
                            self.spec_encoder.spec_prototypes.requires_grad = True
                    except Exception as _e:
                        print(f"[Supervised Finetune] Warning: enable spec_head classifier failed{dataset_str}: {_e}")

                # ATT180/ATT281: collect spec-encoder backbone params for two LP-FT modes:
                # (A) progressive-unfreeze path, or (B) direct full-backbone path from step0.
                prog_params_stage2 = []
                prog_params_stage3 = []
                prog_params_stage4 = []
                lp_prog_active = bool(lp_prog_enable)
                # P1 fix:
                # direct full-backbone path should only be enabled when user CONFIG explicitly disables
                # progressive unfreeze. Do not activate direct mode when progressive is turned off by
                # runtime safety guards (self-train/task1/tiny-shot/retrain-full guards).
                lp_full_backbone_direct = bool(
                    (not lp_prog_enable_cfg) and (not lp_prog_active) and lp_backbone_ft_enable
                )

                def _is_lpft_frozen_struct_param(_name: str) -> bool:
                    # ATT248(S2): after removing shortcut path from SpecEncoderLight blocks,
                    # no structural branch parameters need special freezing.
                    return False

                if lp_prog_active or lp_full_backbone_direct:
                    # stage2: projection+normalization layers
                    for _m in ("proj", "norm", "proj_inv", "norm_inv"):
                        _mod = getattr(self.spec_encoder, _m, None)
                        if _mod is not None and hasattr(_mod, "parameters"):
                            prog_params_stage2.extend(list(_mod.parameters()))
                    # stage3: last conv block
                    _block3 = getattr(self.spec_encoder, "block3", None)
                    if _block3 is not None and hasattr(_block3, "named_parameters"):
                        for _n, _p in _block3.named_parameters():
                            if _is_lpft_frozen_struct_param(_n):
                                continue
                            prog_params_stage3.append(_p)
                    # stage4: broader backbone adaptation
                    if lp_backbone_ft_enable:
                        for _m in ("block2", "stem"):
                            _mod = getattr(self.spec_encoder, _m, None)
                            if _mod is not None and hasattr(_mod, "named_parameters"):
                                for _n, _p in _mod.named_parameters():
                                    if (_m != "stem") and _is_lpft_frozen_struct_param(_n):
                                        continue
                                    prog_params_stage4.append(_p)

                    if lp_prog_active:
                        # Progressive mode: keep backbone frozen at step0; open by stage transitions.
                        for _p in (prog_params_stage2 + prog_params_stage3 + prog_params_stage4):
                            _p.requires_grad = False
                    elif lp_full_backbone_direct:
                        # Direct mode: enable selected backbone params from step0.
                        for _p in (prog_params_stage2 + prog_params_stage3 + prog_params_stage4):
                            _p.requires_grad = True

                    n_s2 = sum(p.numel() for p in prog_params_stage2)
                    n_s3 = sum(p.numel() for p in prog_params_stage3)
                    n_s4 = sum(p.numel() for p in prog_params_stage4)
                    if lp_prog_active:
                        print(
                            f"[Supervised Finetune] Progressive unfreeze enabled{dataset_str}: "
                            f"stage2(start={lp_unfreeze_stage2_start}, params={n_s2:,}), "
                            f"stage3(start={lp_unfreeze_stage3_start}, params={n_s3:,}), "
                            f"stage4(start={lp_unfreeze_stage4_start}, params={n_s4:,}, backbone_ft={int(lp_backbone_ft_enable)})"
                        )
                    elif lp_full_backbone_direct:
                        print(
                            f"[Supervised Finetune] Full-backbone LP-FT enabled{dataset_str}: "
                            f"step0_params(stage2+3+4)={n_s2 + n_s3 + n_s4:,}, "
                            f"backbone_lr_mult={float(lp_backbone_lr_mult):.4f}"
                        )
            else:
                lp_prog_active = False
                lp_full_backbone_direct = False
                prog_params_stage2 = []
                prog_params_stage3 = []
                prog_params_stage4 = []
                # ===== Hierarchical head: prompt/domain residuals + adapters =====
                if hasattr(self.fusion_decoder, 'set_prototype_residual_shared_requires_grad'):
                    self.fusion_decoder.set_prototype_residual_shared_requires_grad(False)
                if hasattr(self.fusion_decoder, 'set_prototype_residual_domain_requires_grad'):
                    self.fusion_decoder.set_prototype_residual_domain_requires_grad(True)
                if hasattr(self.fusion_decoder, "set_cls_adapter_requires_grad"):
                    self.fusion_decoder.set_cls_adapter_requires_grad(True)
                if hasattr(self.fusion_decoder, "set_diag_linear_requires_grad"):
                    self.fusion_decoder.set_diag_linear_requires_grad(True)
            
            # Collect trainable parameters (STRICT: only those with requires_grad=True).
            # This avoids the "loss has no grad_fn" crash when frozen params are accidentally optimized.
            params_to_optimize = []
            params_to_optimize += [p for p in self.fusion_module.parameters() if p.requires_grad]
            if self.fusion_decoder is not None:
                params_to_optimize += [p for p in self.fusion_decoder.parameters() if p.requires_grad]
            if hasattr(self.signal_encoder, 'use_domain_lora') and self.signal_encoder.use_domain_lora and hasattr(self.signal_encoder, 'domain_lora_fc'):
                params_to_optimize += [p for p in self.signal_encoder.domain_lora_fc.parameters() if p.requires_grad]
            if hasattr(self.spec_encoder, 'use_domain_lora') and self.spec_encoder.use_domain_lora and hasattr(self.spec_encoder, 'domain_lora_agg'):
                params_to_optimize += [p for p in self.spec_encoder.domain_lora_agg.parameters() if p.requires_grad]
            if include_spec_head and getattr(self.spec_encoder, "has_spec_head", False):
                if self.spec_encoder.classifier is not None:
                    params_to_optimize += [p for p in self.spec_encoder.classifier.parameters() if p.requires_grad]
                if getattr(self.spec_encoder, "spec_prototypes", None) is not None:
                    params_to_optimize.append(self.spec_encoder.spec_prototypes)
            # ATT180: include progressive params in optimizer param groups from step 0.
            # They are frozen initially; once requires_grad is toggled later, optimizer can update them directly.
            if is_flat and (lp_prog_active or lp_full_backbone_direct):
                params_to_optimize += prog_params_stage2
                params_to_optimize += prog_params_stage3
                params_to_optimize += prog_params_stage4
            # deduplicate parameters by id (avoids accidental duplicates from shared references)
            _uniq_params = []
            _seen_ids = set()
            for _p in params_to_optimize:
                _pid = id(_p)
                if _pid in _seen_ids:
                    continue
                _seen_ids.add(_pid)
                _uniq_params.append(_p)
            params_to_optimize = _uniq_params
            
            if len(params_to_optimize) == 0:
                print("[Supervised Finetune] Warning: No trainable parameters found")
                return False
            
            try:
                n_fusion = sum(p.numel() for p in self.fusion_module.parameters() if p.requires_grad)
                n_dec = sum(p.numel() for p in self.fusion_decoder.parameters() if p.requires_grad) if self.fusion_decoder is not None else 0
                n_lora_sig = 0
                n_lora_spec = 0
                if hasattr(self.signal_encoder, "domain_lora_fc"):
                    n_lora_sig = sum(p.numel() for p in self.signal_encoder.domain_lora_fc.parameters() if p.requires_grad)
                if hasattr(self.spec_encoder, "domain_lora_agg"):
                    n_lora_spec = sum(p.numel() for p in self.spec_encoder.domain_lora_agg.parameters() if p.requires_grad)
                n_spec_head = 0
                if include_spec_head and getattr(self.spec_encoder, "has_spec_head", False):
                    if self.spec_encoder.classifier is not None:
                        n_spec_head += sum(p.numel() for p in self.spec_encoder.classifier.parameters() if p.requires_grad)
                    if getattr(self.spec_encoder, "spec_prototypes", None) is not None:
                        n_spec_head += self.spec_encoder.spec_prototypes.numel()
                print(
                    f"[Supervised Finetune] Trainable params: {sum(p.numel() for p in params_to_optimize):,} "
                    f"(fusion={n_fusion:,}, decoder={n_dec:,}, lora_sig={n_lora_sig:,}, "
                    f"lora_spec={n_lora_spec:,}, spec_head={n_spec_head:,})"
                )
            except Exception:
                print(f"[Supervised Finetune] Trainable params: {sum(p.numel() for p in params_to_optimize):,}")
            # Override weights for flat head: NF is not a meaningful objective here.
            if is_flat:
                nf_weight = 0.0
                diag_weight = 1.0

            # ATT201(A): use lower LR for progressively-unfrozen backbone params.
            _backbone_param_ids = set()
            if is_flat and (lp_prog_active or lp_full_backbone_direct):
                _backbone_param_ids = {id(p) for p in (prog_params_stage2 + prog_params_stage3 + prog_params_stage4)}
            _base_params = [p for p in params_to_optimize if id(p) not in _backbone_param_ids]
            _backbone_params = [p for p in params_to_optimize if id(p) in _backbone_param_ids]
            _backbone_group_idx = None
            _backbone_lr_target = float(lr) * float(lp_backbone_lr_mult)
            if _backbone_params and float(lp_backbone_lr_mult) > 0:
                optimizer = torch.optim.Adam(
                    [
                        {"params": _base_params, "lr": float(lr)},
                        {"params": _backbone_params, "lr": _backbone_lr_target},
                    ]
                )
                _backbone_group_idx = 1
            else:
                optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
            _initial_group_lrs = [float(pg.get("lr", float(lr))) for pg in optimizer.param_groups]
            _lp_sam_base_active = bool(lp_sam_enable and is_flat)
            if _lp_sam_base_active:
                print(
                    f"[Supervised Finetune] SAM enabled{dataset_str}: "
                    f"rho={float(lp_sam_rho):.4f} (flat LP-FT, stage1_only={bool(lp_sam_stage1_only)})"
                )

            # ATT226(S4): LP-FT global LR schedule (warmup + cosine decay).
            # Keep relative LR across param groups; progressive backbone warmup stays multiplicative.
            lp_lr_sched_enable = False
            lp_lr_warmup_steps = 0
            lp_lr_min_scale = 0.2
            try:
                import config as cfg
                lp_lr_sched_enable = bool(getattr(cfg, "TARGET_SUP_LP_LR_SCHEDULE_ENABLE", False))
                lp_lr_warmup_steps = int(getattr(cfg, "TARGET_SUP_LP_LR_WARMUP_STEPS", 0))
                lp_lr_min_scale = float(getattr(cfg, "TARGET_SUP_LP_LR_MIN_SCALE", 0.2))
            except Exception:
                pass
            lp_lr_warmup_steps = max(0, int(lp_lr_warmup_steps))
            lp_lr_min_scale = max(0.0, min(1.0, float(lp_lr_min_scale)))

            def _lp_global_lr_scale(_step: int) -> float:
                if (not lp_lr_sched_enable) or int(num_steps) <= 1:
                    return 1.0
                _warm = max(0, min(int(lp_lr_warmup_steps), int(num_steps) - 1))
                _t = int(_step) + 1
                if (_warm > 0) and (_t <= _warm):
                    return max(lp_lr_min_scale, float(_t) / float(_warm))
                _tail = max(1, int(num_steps) - _warm)
                _p = min(1.0, max(0.0, float(_t - _warm) / float(_tail)))
                _cos = 0.5 * (1.0 + math.cos(math.pi * _p))
                return float(lp_lr_min_scale + (1.0 - lp_lr_min_scale) * _cos)

            # ATT201(A): L2-SP reference snapshots for backbone params.
            l2sp_refs = {}
            if is_flat and (lp_prog_active or lp_full_backbone_direct) and lp_l2sp_enable and float(lp_l2sp_weight) > 0:
                _prog_all = (prog_params_stage2 + prog_params_stage3 + prog_params_stage4)
                if lp_l2sp_use_global_anchor:
                    _anchor = getattr(self, "_lp_l2sp_global_anchor", None)
                    if not isinstance(_anchor, dict) or len(_anchor) == 0:
                        _anchor = {}
                        for _p in _prog_all:
                            _anchor[id(_p)] = _p.detach().clone()
                        self._lp_l2sp_global_anchor = _anchor
                        print(f"[Supervised Finetune] Initialized global L2-SP anchor{dataset_str} (params={len(_anchor)})")
                    l2sp_refs = _anchor
                else:
                    for _p in _prog_all:
                        l2sp_refs[id(_p)] = _p.detach().clone()

            # Apply temperature override AFTER trainable params are finalized.
            if is_flat and (_lp_temp_override is not None) and hasattr(self.fusion_decoder, "temperature"):
                try:
                    _lp_temp_orig = float(getattr(self.fusion_decoder, "temperature"))
                    setattr(self.fusion_decoder, "temperature", float(_lp_temp_override))
                    print(
                        f"[Supervised Finetune] Override flat cosine temperature: "
                        f"{_lp_temp_orig:.6f} -> {_lp_temp_override:.6f}{dataset_str}"
                    )
                except Exception:
                    _lp_temp_orig = None
            
            # (Attempt 51) Optional ramp override for flat LP-FT.
            # NOTE: this controls `_diag_linear_ramp` in [0, 1], not the raw `diag_linear_scale` parameter.
            _lp_beta_override = None
            _lp_ramp_orig = None
            if is_flat:
                try:
                    import config as cfg
                    _lp_beta_override = getattr(cfg, "TARGET_SUP_LP_BETA_OVERRIDE", None)
                    if _lp_beta_override is not None:
                        _lp_beta_override = float(_lp_beta_override)
                except Exception:
                    _lp_beta_override = None
            
            if is_flat and (_lp_beta_override is not None) and hasattr(self.fusion_decoder, "set_diag_linear_ramp"):
                try:
                    # Save/restore the actual ramp attribute. Reading `diag_linear_scale` here is incorrect
                    # because it is the raw sigmoid input parameter, not the runtime ramp multiplier.
                    _lp_ramp_orig = float(getattr(self.fusion_decoder, "_diag_linear_ramp", 1.0))
                    self.fusion_decoder.set_diag_linear_ramp(float(_lp_beta_override))
                    print(
                        f"[Supervised Finetune] Override diag_linear ramp: "
                        f"{_lp_ramp_orig:.6f} -> {_lp_beta_override:.6f}{dataset_str}"
                    )
                except Exception as e:
                    print(f"[Supervised Finetune] Warning: ramp override failed: {e}")
                    _lp_ramp_orig = None

            # (Attempt 52) Validation-driven early stopping for flat LP-FT (decoder-only).
            # Split the 5% labeled finetune set into train/val (stratified) and early-stop on val metric.
            _use_val_es = False
            _val_ratio = 0.20
            _patience = 5
            _min_steps = 20
            _eval_every = 5
            _es_metric = "ce"
            _min_delta = 1e-4
            _min_val_samples = 16
            _val_multi_split_enable = False
            _val_multi_split_repeats = 1
            _val_multi_split_base_seed = 42
            _val_multi_split_metric_reduce = "mean"
            _lp_ema_enable = False
            _lp_ema_decay = 0.995
            _lp_ema_eval_enable = False
            _lp_pre_bn_adapt_enable = False
            _lp_pre_bn_adapt_domains = ["HIT"]
            _lp_pre_bn_adapt_num_batches = None
            _lp_pre_bn_adapt_on_selftrain = False
            _lp_pre_bn_adapt_on_retrain_full = True
            if is_flat:
                try:
                    import config as cfg
                    _use_val_es = bool(getattr(cfg, "TARGET_SUP_LP_USE_VAL_EARLY_STOP", False))
                    _val_ratio = float(getattr(cfg, "TARGET_SUP_LP_VAL_RATIO", _val_ratio))
                    _patience = int(getattr(cfg, "TARGET_SUP_LP_PATIENCE", _patience))
                    if "HIT" in _dataset_upper:
                        _patience = int(getattr(cfg, "TARGET_SUP_LP_PATIENCE_HIT", _patience))
                    elif "PU" in _dataset_upper:
                        _patience = int(getattr(cfg, "TARGET_SUP_LP_PATIENCE_PU", _patience))
                    _min_steps = int(getattr(cfg, "TARGET_SUP_LP_MIN_STEPS", _min_steps))
                    _eval_every = int(getattr(cfg, "TARGET_SUP_LP_EVAL_EVERY", _eval_every))
                    _es_metric = str(getattr(cfg, "TARGET_SUP_LP_EARLYSTOP_METRIC", _es_metric)).strip().lower()
                    _min_delta = float(getattr(cfg, "TARGET_SUP_LP_MIN_DELTA", _min_delta))
                    _min_val_samples = int(getattr(cfg, "TARGET_SUP_LP_MIN_VAL_SAMPLES", _min_val_samples))
                    _val_multi_split_enable = bool(
                        getattr(cfg, "TARGET_SUP_LP_VAL_MULTI_SPLIT_ENABLE", _val_multi_split_enable)
                    )
                    _val_multi_split_repeats = int(
                        getattr(cfg, "TARGET_SUP_LP_VAL_MULTI_SPLIT_REPEATS", _val_multi_split_repeats)
                    )
                    _val_multi_split_base_seed = int(
                        getattr(cfg, "TARGET_SUP_LP_VAL_MULTI_SPLIT_BASE_SEED", _val_multi_split_base_seed)
                    )
                    _val_multi_split_metric_reduce = str(
                        getattr(cfg, "TARGET_SUP_LP_VAL_MULTI_SPLIT_METRIC_REDUCE", _val_multi_split_metric_reduce)
                    ).strip().lower()
                    _lp_ema_enable = bool(getattr(cfg, "TARGET_SUP_LP_EMA_ENABLE", False))
                    _lp_ema_decay = float(getattr(cfg, "TARGET_SUP_LP_EMA_DECAY", 0.995))
                    _lp_ema_eval_enable = bool(getattr(cfg, "TARGET_SUP_LP_EMA_EVAL_ENABLE", _lp_ema_enable))
                    _lp_pre_bn_adapt_enable = bool(
                        getattr(cfg, "TARGET_SUP_LP_PRE_BN_ADAPT_ENABLE", _lp_pre_bn_adapt_enable)
                    )
                    _lp_pre_bn_adapt_domains = getattr(
                        cfg, "TARGET_SUP_LP_PRE_BN_ADAPT_DOMAINS", _lp_pre_bn_adapt_domains
                    )
                    _lp_pre_bn_adapt_num_batches = getattr(
                        cfg, "TARGET_SUP_LP_PRE_BN_ADAPT_NUM_BATCHES", _lp_pre_bn_adapt_num_batches
                    )
                    _lp_pre_bn_adapt_on_selftrain = bool(
                        getattr(
                            cfg,
                            "TARGET_SUP_LP_PRE_BN_ADAPT_ON_SELFTRAIN",
                            _lp_pre_bn_adapt_on_selftrain,
                        )
                    )
                    _lp_pre_bn_adapt_on_retrain_full = bool(
                        getattr(
                            cfg,
                            "TARGET_SUP_LP_PRE_BN_ADAPT_ON_RETRAIN_FULL",
                            _lp_pre_bn_adapt_on_retrain_full,
                        )
                    )
                except Exception:
                    pass
            if force_no_val_es:
                _use_val_es = False
                if is_flat:
                    print(f"[Supervised Finetune] (Val-ES) disabled by force_no_val_es{dataset_str}")
            _val_ratio = max(0.0, min(0.5, float(_val_ratio)))
            _patience = max(1, int(_patience))
            _min_steps = max(0, int(_min_steps))
            _eval_every = max(1, int(_eval_every))
            _min_val_samples = max(1, int(_min_val_samples))
            _lp_ema_decay = max(0.90, min(0.9999, float(_lp_ema_decay)))
            _val_multi_split_repeats = max(1, int(_val_multi_split_repeats))
            _val_multi_split_metric_reduce = str(_val_multi_split_metric_reduce or "mean").strip().lower()
            if _val_multi_split_metric_reduce not in ("mean", "median"):
                _val_multi_split_metric_reduce = "mean"
            if not bool(_val_multi_split_enable):
                _val_multi_split_repeats = 1
            if isinstance(_lp_pre_bn_adapt_domains, str):
                _lp_pre_bn_adapt_domains = [_lp_pre_bn_adapt_domains]
            if not isinstance(_lp_pre_bn_adapt_domains, (list, tuple, set)):
                _lp_pre_bn_adapt_domains = ["HIT"]
            _lp_pre_bn_adapt_domains = [
                str(_d).strip().upper() for _d in _lp_pre_bn_adapt_domains if str(_d).strip()
            ]
            if len(_lp_pre_bn_adapt_domains) == 0:
                _lp_pre_bn_adapt_domains = ["HIT"]
            if _es_metric not in ("ce", "macro_f1", "spec_ce", "auto"):
                _es_metric = "ce"
            if _es_metric == "auto":
                _has_spec_head_es = bool(
                    include_spec_head
                    and getattr(self.spec_encoder, "has_spec_head", False)
                )
                _es_metric = "spec_ce" if (is_flat and _has_spec_head_es) else "ce"

            train_loader = finetune_loader
            val_loader = None
            val_loaders = []
            if _use_val_es and (finetune_loader is not None) and hasattr(finetune_loader, "dataset"):
                try:
                    from torch.utils.data import Subset, DataLoader, ConcatDataset
                    import random as _random

                    ds = finetune_loader.dataset
                    n_total = len(ds)
                    _split_candidate_indices = list(range(n_total))
                    _val_split_source = "all-labeled"
                    # P2: for self-training calls, val-ES must be built only from
                    # original human-labeled samples (exclude pseudo-labeled tail)
                    # to avoid step0 metric leakage.
                    if _is_selftrain_call and isinstance(ds, ConcatDataset):
                        try:
                            if len(getattr(ds, "datasets", [])) >= 2:
                                _base_n = int(len(ds.datasets[0]))
                                if _base_n > 0:
                                    _split_candidate_indices = list(range(_base_n))
                                    _val_split_source = f"base-labeled-only(n={_base_n})"
                        except Exception:
                            pass
                    n_split_pool = int(len(_split_candidate_indices))
                    # Need at least 2 batches to make val meaningful.
                    if n_split_pool >= max(2 * int(num_classes_eval), 8):
                        # Robust label getter for Subset/dict/tuple datasets
                        def _get_label_from_item(item):
                            if isinstance(item, dict) and "label" in item:
                                return int(item["label"])
                            if isinstance(item, (tuple, list)) and len(item) > 0:
                                return int(item[-1])
                            raise RuntimeError("Unable to extract label from dataset sample.")

                        by_cls = {}
                        for i in _split_candidate_indices:
                            y = None
                            try:
                                # Fast path: underlying dataset attrs
                                for attr in ("labels", "y", "targets"):
                                    if hasattr(ds, attr):
                                        arr = getattr(ds, attr)
                                        y = int(arr[i])
                                        break
                            except Exception:
                                y = None
                            if y is None:
                                y = _get_label_from_item(ds[i])
                            by_cls.setdefault(int(y), []).append(i)

                        def _mk_loader(subset, shuffle: bool):
                            return DataLoader(
                                subset,
                                batch_size=finetune_loader.batch_size,
                                shuffle=shuffle,
                                collate_fn=getattr(finetune_loader, "collate_fn", None),
                                num_workers=getattr(finetune_loader, "num_workers", 0),
                                pin_memory=getattr(finetune_loader, "pin_memory", False),
                                prefetch_factor=getattr(finetune_loader, "prefetch_factor", None),
                                persistent_workers=getattr(finetune_loader, "persistent_workers", False),
                            )

                        def _build_stratified_split(seed_value: int):
                            rng = _random.Random(int(seed_value))
                            val_total = int(round(float(n_split_pool) * float(_val_ratio)))
                            val_total = max(int(num_classes_eval), val_total)
                            val_total = min(
                                n_split_pool - int(num_classes_eval),
                                max(int(num_classes_eval), val_total),
                            )
                            if val_total < int(_min_val_samples):
                                return None, None, int(val_total)
                            val_idx = []
                            for _c, _idxs in by_cls.items():
                                _idxs = list(_idxs)
                                rng.shuffle(_idxs)
                                _k = int(round(len(_idxs) * float(_val_ratio)))
                                _k = max(1, min(len(_idxs) - 1, _k)) if len(_idxs) >= 2 else 0
                                if _k > 0:
                                    val_idx.extend(_idxs[:_k])
                            val_idx = list(dict.fromkeys(val_idx))
                            if len(val_idx) > val_total:
                                rng.shuffle(val_idx)
                                val_idx = val_idx[:val_total]
                            elif len(val_idx) < val_total:
                                _val_set_now = set(val_idx)
                                leftovers = [i for i in _split_candidate_indices if i not in _val_set_now]
                                rng.shuffle(leftovers)
                                val_idx.extend(leftovers[: (val_total - len(val_idx))])
                            val_set = set(val_idx)
                            train_idx = [i for i in range(n_total) if i not in val_set]
                            if len(train_idx) <= 0 or len(val_idx) <= 0:
                                return None, None, int(val_total)
                            return train_idx, val_idx, int(val_total)

                        train_idx, val_idx, val_total = _build_stratified_split(_val_multi_split_base_seed)
                        if val_total < int(_min_val_samples):
                            print(
                                f"[Supervised Finetune] (Val-ES) Skip split{dataset_str}: "
                                f"val samples {val_total} < min_val_samples={int(_min_val_samples)}; "
                                f"fallback to fixed-step LP-FT."
                            )
                        elif (train_idx is not None) and (val_idx is not None):
                            train_subset = Subset(ds, train_idx)
                            val_subset = Subset(ds, val_idx)
                            train_loader = _mk_loader(train_subset, shuffle=True)
                            val_loader = _mk_loader(val_subset, shuffle=False)
                            val_loaders = [val_loader]

                            if bool(_val_multi_split_enable) and int(_val_multi_split_repeats) > 1:
                                train_set_primary = set(train_idx)
                                for _split_id in range(1, int(_val_multi_split_repeats)):
                                    _, val_idx_alt, _ = _build_stratified_split(
                                        int(_val_multi_split_base_seed) + int(_split_id)
                                    )
                                    if not val_idx_alt:
                                        continue
                                    # Eval-only secondary split: remove current train samples to avoid metric leakage.
                                    val_idx_alt = [i for i in val_idx_alt if i not in train_set_primary]
                                    if len(val_idx_alt) < max(int(num_classes_eval), int(_min_val_samples)):
                                        continue
                                    val_subset_alt = Subset(ds, val_idx_alt)
                                    val_loaders.append(_mk_loader(val_subset_alt, shuffle=False))

                            print(
                                f"[Supervised Finetune] (Val-ES) Split{dataset_str}: "
                                f"{len(train_idx)} train, {len(val_idx)} val (ratio={_val_ratio:.2f}), "
                                f"metric={_es_metric}, patience={_patience}, eval_every={_eval_every}, "
                                f"split_source={_val_split_source}"
                            )
                            if len(val_loaders) > 1:
                                print(
                                    f"[Supervised Finetune] (Val-ES) Multi-split{dataset_str}: "
                                    f"val_splits={len(val_loaders)}, base_seed={int(_val_multi_split_base_seed)}, "
                                    f"reduce={_val_multi_split_metric_reduce}"
                                )
                except Exception as _e:
                    # If anything fails, fall back to original fixed-step finetune_loader.
                    val_loader = None
                    val_loaders = []

            # ATT224(S3): semi-supervised LP-FT regularization (target unlabeled entropy + diversity).
            try:
                import config as cfg
                lp_semi_enable = bool(getattr(cfg, "TARGET_SUP_LP_SEMI_ENABLE", False))
                lp_semi_ent_w = float(getattr(cfg, "TARGET_SUP_LP_SEMI_ENT_WEIGHT", 0.10))
                lp_semi_div_w = float(getattr(cfg, "TARGET_SUP_LP_SEMI_DIV_WEIGHT", 0.10))
                lp_semi_warmup_steps = int(getattr(cfg, "TARGET_SUP_LP_SEMI_WARMUP_STEPS", 20))
                lp_semi_tinyshot_disable_threshold = int(
                    getattr(cfg, "TARGET_SUP_LP_SEMI_TINYSHOT_DISABLE_THRESHOLD", 0)
                )
            except Exception:
                lp_semi_enable = False
                lp_semi_ent_w = 0.10
                lp_semi_div_w = 0.10
                lp_semi_warmup_steps = 20
                lp_semi_tinyshot_disable_threshold = 0
            # ATT227(S4): class-balanced focal loss for LP-FT SpecHead on tiny/imbalanced target splits.
            try:
                import config as cfg
                lp_spec_cb_focal_enable = bool(
                    getattr(cfg, "TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_ENABLE", False)
                )
                lp_spec_cb_focal_gamma = float(
                    getattr(cfg, "TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_GAMMA", 1.0)
                )
                lp_spec_cb_focal_beta = float(
                    getattr(cfg, "TARGET_SUP_LP_SPEC_CLASS_BALANCED_FOCAL_BETA", 0.999)
                )
            except Exception:
                lp_spec_cb_focal_enable = False
                lp_spec_cb_focal_gamma = 1.0
                lp_spec_cb_focal_beta = 0.999
            lp_spec_cb_focal_gamma = max(0.0, float(lp_spec_cb_focal_gamma))
            lp_spec_cb_focal_beta = max(0.0, min(0.99999, float(lp_spec_cb_focal_beta)))
            lp_spec_cb_class_weights = None
            if is_flat and include_spec_head and lp_spec_cb_focal_enable:
                try:
                    from torch.utils.data import Subset

                    def _label_from_ds(_ds, _idx):
                        if isinstance(_ds, Subset):
                            return _label_from_ds(_ds.dataset, int(_ds.indices[int(_idx)]))
                        for _attr in ("labels", "y", "targets"):
                            if hasattr(_ds, _attr):
                                _arr = getattr(_ds, _attr)
                                return int(_arr[int(_idx)])
                        _item = _ds[int(_idx)]
                        if isinstance(_item, dict) and ("label" in _item):
                            return int(_item["label"])
                        if isinstance(_item, (tuple, list)) and len(_item) > 0:
                            return int(_item[-1])
                        raise RuntimeError("Unable to extract label from LP dataset.")

                    _ds_lp = getattr(train_loader, "dataset", None)
                    if _ds_lp is not None:
                        _n_cls = 3 if int(num_classes_eval) == 3 else 4
                        _counts = torch.zeros(_n_cls, dtype=torch.float32, device=self.device)
                        _n = int(len(_ds_lp))
                        for _i in range(_n):
                            _y = _label_from_ds(_ds_lp, _i)
                            _y = max(0, min(_n_cls - 1, int(_y)))
                            _counts[_y] += 1.0
                        _counts = torch.clamp(_counts, min=1.0)
                        _eff_num = 1.0 - torch.pow(
                            torch.full_like(_counts, float(lp_spec_cb_focal_beta)),
                            _counts
                        )
                        _alpha = (1.0 - float(lp_spec_cb_focal_beta)) / torch.clamp(_eff_num, min=1e-8)
                        lp_spec_cb_class_weights = _alpha / torch.clamp(_alpha.mean(), min=1e-8)
                        _counts_list = [int(x) for x in _counts.detach().cpu().tolist()]
                        _alpha_list = [round(float(x), 4) for x in lp_spec_cb_class_weights.detach().cpu().tolist()]
                        print(
                            f"[Supervised Finetune] LP Spec CB-Focal enabled{dataset_str}: "
                            f"gamma={lp_spec_cb_focal_gamma:.2f}, beta={lp_spec_cb_focal_beta:.5f}, "
                            f"class_counts={_counts_list}, class_alpha={_alpha_list}"
                        )
                except Exception as _e:
                    lp_spec_cb_class_weights = None
                    print(f"[Supervised Finetune] LP Spec CB-Focal fallback to CE{dataset_str}: {_e}")
            # ATT228(S4): semantic prototype anchor regularization in LP-FT (aux only, warmup protected).
            try:
                import config as cfg
                lp_proto_anchor_enable = bool(getattr(cfg, "TARGET_SUP_LP_PROTO_ANCHOR_ENABLE", False))
                lp_proto_anchor_w = float(getattr(cfg, "TARGET_SUP_LP_PROTO_ANCHOR_WEIGHT", 0.05))
                lp_proto_anchor_warmup_steps = int(getattr(cfg, "TARGET_SUP_LP_PROTO_ANCHOR_WARMUP_STEPS", 30))
            except Exception:
                lp_proto_anchor_enable = False
                lp_proto_anchor_w = 0.05
                lp_proto_anchor_warmup_steps = 30
            lp_proto_anchor_w = max(0.0, float(lp_proto_anchor_w))
            lp_proto_anchor_warmup_steps = max(0, int(lp_proto_anchor_warmup_steps))
            if not is_flat:
                lp_semi_enable = False
            if unlabeled_loader is None:
                lp_semi_enable = False
            if lp_semi_enable and int(lp_semi_tinyshot_disable_threshold) > 0:
                _semi_train_samples = None
                try:
                    if finetune_loader is not None and hasattr(finetune_loader, "dataset") and finetune_loader.dataset is not None:
                        _semi_train_samples = int(len(finetune_loader.dataset))
                except Exception:
                    _semi_train_samples = None
                if (
                    _semi_train_samples is not None
                    and _semi_train_samples <= int(lp_semi_tinyshot_disable_threshold)
                ):
                    lp_semi_enable = False
                    print(
                        f"[Supervised Finetune] LP-semi disabled by tiny-shot guard{dataset_str}: "
                        f"n_train={_semi_train_samples} <= threshold={int(lp_semi_tinyshot_disable_threshold)}"
                    )
            unlabeled_iter = iter(unlabeled_loader) if lp_semi_enable else None

            # ATT202(B) revised: auto-scale progressive stage thresholds by training sample count.
            # This avoids dataset-specific hardcoding and keeps schedule proportional under tiny-shot LP-FT.
            if is_flat and lp_prog_enable and lp_stage_scale_by_samples:
                n_train_samples = None
                try:
                    if train_loader is not None and hasattr(train_loader, "dataset") and train_loader.dataset is not None:
                        n_train_samples = int(len(train_loader.dataset))
                except Exception:
                    n_train_samples = None
                if n_train_samples is None:
                    try:
                        if finetune_loader is not None and hasattr(finetune_loader, "dataset") and finetune_loader.dataset is not None:
                            n_train_samples = int(len(finetune_loader.dataset))
                    except Exception:
                        n_train_samples = None

                th = max(1, int(lp_stage_scale_threshold))
                if (n_train_samples is not None) and (n_train_samples < th):
                    ratio = float(n_train_samples) / float(th)
                    ratio = max(float(lp_stage_scale_min_ratio), min(1.0, ratio))
                    s2_base = int(lp_unfreeze_stage2_start)
                    s3_base = int(lp_unfreeze_stage3_start)
                    s4_base = int(lp_unfreeze_stage4_start)
                    s2 = max(1, int(round(s2_base * ratio)))
                    s2 = max(int(lp_stage_scale_min_stage2_steps), s2)
                    s3 = max(s2 + 1, int(round(s3_base * ratio)))
                    s4 = max(s3 + 1, int(round(s4_base * ratio)))
                    lp_unfreeze_stage2_start = s2
                    lp_unfreeze_stage3_start = s3
                    lp_unfreeze_stage4_start = s4
                    print(
                        f"[Supervised Finetune] Stage thresholds scaled by samples{dataset_str}: "
                        f"n_train={n_train_samples}, threshold={th}, ratio={ratio:.3f}, "
                        f"stage2:{s2_base}->{s2}, stage3:{s3_base}->{s3}, stage4:{s4_base}->{s4}, "
                        f"min_stage2={int(lp_stage_scale_min_stage2_steps)}"
                    )
            
            # Micro helpers for early stopping
            best_val = None
            best_step = None
            best_decoder_state = None
            best_spec_state = None
            bad_epochs = 0
            ema_decoder_state = None
            ema_spec_state = None
            ema_metric_val = None
            ema_step = None
            # ATT191(B): keep per-metric best snapshots in the same LP-FT run.
            metric_best = {"ce": None, "spec_ce": None}
            metric_best_state = {
                "ce": {"decoder": None, "spec": None, "step": None},
                "spec_ce": {"decoder": None, "spec": None, "step": None},
            }

            def _ema_update_state(ema_state, cur_state, decay):
                if ema_state is None:
                    return {k: t.detach().clone() for k, t in cur_state.items()}
                out = {}
                for _k, _t in cur_state.items():
                    _cur = _t.detach()
                    _old = ema_state.get(_k, None)
                    if (
                        _old is not None
                        and _old.dtype.is_floating_point
                        and _cur.dtype.is_floating_point
                        and _old.shape == _cur.shape
                    ):
                        out[_k] = _old.mul(float(decay)).add(_cur, alpha=(1.0 - float(decay)))
                    else:
                        out[_k] = _cur.clone()
                return out

            def _macro_f1_from_logits(logits, y, num_classes: int):
                # logits: [N,C], y: [N]
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    f1s = []
                    for c in range(int(num_classes)):
                        tp = ((pred == c) & (y == c)).sum().item()
                        fp = ((pred == c) & (y != c)).sum().item()
                        fn = ((pred != c) & (y == c)).sum().item()
                        p = tp / max(1, (tp + fp))
                        r = tp / max(1, (tp + fn))
                        f1 = (2 * p * r) / max(1e-12, (p + r))
                        f1s.append(float(f1))
                    return float(sum(f1s) / max(1, len(f1s)))

            def _eval_on_val():
                _loaders = [vl for vl in (val_loaders if len(val_loaders) > 0 else [val_loader]) if vl is not None]
                if len(_loaders) == 0:
                    return None

                def _eval_one_loader(_loader):
                    self.fusion_decoder.eval()
                    losses = []
                    spec_losses = []
                    all_logits = []
                    all_spec_logits = []
                    all_y = []
                    with torch.no_grad():
                        for vb in _loader:
                            signal = vb["signal"].to(self.device)
                            signal_mask = vb["signal_mask"].to(self.device)
                            spec = self.prepare_spec_batch(vb)
                            spec_mask = vb["spec_mask"].to(self.device)
                            y = vb["label"].to(self.device)
                            domains = vb.get("domain", None)
                            domain_ids = None
                            if domains is not None:
                                domain_map = _build_domain_map()
                                domain_ids = torch.tensor([domain_map.get(d, 0) for d in domains], dtype=torch.long, device=self.device)
                            # ATT446/ATT449: compute bypass features
                            self._last_z_wst = None
                            self._last_z_env = None
                            self._last_ratio = None
                            self._last_z_scd = None
                            if self._wst_bypass_active:
                                _wst_raw = vb.get("wst_spec", None)
                                if _wst_raw is not None:
                                    self._last_z_wst = self.wst_encoder(_wst_raw.to(self.device))
                            if self._env_bypass_active:
                                _env_raw = vb.get("env_spec", None)
                                if _env_raw is not None:
                                    self._last_z_env = self.env_encoder(_env_raw.to(self.device))
                                _ratio_raw = vb.get("bpfi_bpfo_ratio", None)
                                if _ratio_raw is not None:
                                    self._last_ratio = _ratio_raw.to(self.device)
                            if self._scd_bypass_active:
                                _scd_raw = vb.get("scd_spec", None)
                                if _scd_raw is not None:
                                    self._last_z_scd = self.scd_encoder(_scd_raw.to(self.device))
                            z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                            _spec_kw = {"spec": spec, "spec_mask": spec_mask, "domain_ids": domain_ids}
                            if self._last_z_wst is not None:
                                _spec_kw["wst_aux"] = self._last_z_wst
                            if self._last_z_env is not None:
                                _spec_kw["env_aux"] = self._last_z_env
                            if self._last_ratio is not None:
                                _spec_kw["ratio_aux"] = self._last_ratio
                            if self._last_z_scd is not None:
                                _spec_kw["scd_aux"] = self._last_z_scd
                            z_spec = self.spec_encoder(**_spec_kw)
                            # ATT379e: removed F.normalize for train-eval consistency
                            z_fusion = self._fuse_embeddings(z_sig, z_spec, signal=None, signal_mask=None, spec=None, spec_mask=None)
                            logits4, _parts = self.fusion_decoder(
                                z_fusion,
                                domain_ids=domain_ids,
                                text_prototypes=self._active_text_prototypes(),
                                return_parts=True,
                            )
                            if int(num_classes_eval) == 3:
                                logits_use = logits4[:, :3]
                                y_use = y.clamp(0, 2)
                            else:
                                logits_use = logits4
                                y_use = y
                            ce = F.cross_entropy(logits_use, y_use)
                            losses.append(float(ce.item()))
                            if (
                                include_spec_head
                                and getattr(self.spec_encoder, "has_spec_head", False)
                            ):
                                if hasattr(self.spec_encoder, 'compute_spec_head_logits'):
                                    spec_logits = self.spec_encoder.compute_spec_head_logits(z_spec, wst_aux=self._last_z_wst, env_aux=self._last_z_env, ratio_aux=self._last_ratio)
                                elif self.spec_encoder.classifier is not None:
                                    spec_logits = self.spec_encoder.classifier(z_spec)
                                else:
                                    spec_logits = None
                                if spec_logits is not None:
                                    if int(num_classes_eval) == 3:
                                        spec_logits = spec_logits[:, :3]
                                    spec_ce = F.cross_entropy(spec_logits, y_use)
                                spec_losses.append(float(spec_ce.item()))
                                all_spec_logits.append(spec_logits.detach())
                            all_logits.append(logits_use.detach())
                            all_y.append(y_use.detach())
                    if len(losses) == 0:
                        return None
                    val_ce = float(sum(losses) / max(1, len(losses)))
                    val_f1 = None
                    val_spec_ce = None
                    val_spec_f1 = None
                    try:
                        lg = torch.cat(all_logits, dim=0)
                        yy = torch.cat(all_y, dim=0)
                        val_f1 = _macro_f1_from_logits(lg, yy, int(num_classes_eval))
                    except Exception:
                        val_f1 = None
                    if len(spec_losses) > 0:
                        val_spec_ce = float(sum(spec_losses) / max(1, len(spec_losses)))
                        try:
                            lg_sp = torch.cat(all_spec_logits, dim=0)
                            yy_sp = torch.cat(all_y, dim=0)
                            val_spec_f1 = _macro_f1_from_logits(lg_sp, yy_sp, int(num_classes_eval))
                        except Exception:
                            val_spec_f1 = None
                    return {"ce": val_ce, "macro_f1": val_f1, "spec_ce": val_spec_ce, "spec_macro_f1": val_spec_f1}

                split_metrics = []
                for _vl in _loaders:
                    _vm = _eval_one_loader(_vl)
                    if _vm is not None:
                        split_metrics.append(_vm)

                self.fusion_decoder.train()
                if len(split_metrics) == 0:
                    return None

                def _reduce_metric(_name: str):
                    _vals = [float(_m[_name]) for _m in split_metrics if _m.get(_name) is not None]
                    if len(_vals) == 0:
                        return None
                    if _val_multi_split_metric_reduce == "median":
                        return float(np.median(_vals))
                    return float(np.mean(_vals))

                return {
                    "ce": _reduce_metric("ce"),
                    "macro_f1": _reduce_metric("macro_f1"),
                    "spec_ce": _reduce_metric("spec_ce"),
                    "spec_macro_f1": _reduce_metric("spec_macro_f1"),
                    "_num_splits": int(len(split_metrics)),
                }

            def _augment_target_batch(signal_in, spec_in):
                """ATT201(D): lightweight target-domain augmentation for LP-FT."""
                if (not is_flat) or (not lp_aug_enable):
                    return signal_in, spec_in
                if torch.rand(1, device=spec_in.device).item() > float(lp_aug_prob):
                    return signal_in, spec_in
                signal_aug = signal_in
                spec_aug = spec_in
                if float(lp_aug_signal_noise_std) > 0:
                    signal_aug = signal_aug + float(lp_aug_signal_noise_std) * torch.randn_like(signal_aug)
                if float(lp_aug_spec_noise_std) > 0:
                    spec_aug = spec_aug + float(lp_aug_spec_noise_std) * torch.randn_like(spec_aug)
                if spec_aug.dim() == 4:
                    bsz, _c, fdim, tdim = spec_aug.shape
                    fmax = max(1, int(float(lp_aug_freq_mask_ratio) * float(fdim)))
                    tmax = max(1, int(float(lp_aug_time_mask_ratio) * float(tdim)))
                    for _b in range(int(bsz)):
                        for _ in range(max(0, int(lp_aug_num_freq_masks))):
                            fw = int(torch.randint(1, fmax + 1, (1,), device=spec_aug.device).item())
                            f0 = int(torch.randint(0, max(1, fdim - fw + 1), (1,), device=spec_aug.device).item())
                            spec_aug[_b, :, f0:f0 + fw, :] = 0.0
                        for _ in range(max(0, int(lp_aug_num_time_masks))):
                            tw = int(torch.randint(1, tmax + 1, (1,), device=spec_aug.device).item())
                            t0 = int(torch.randint(0, max(1, tdim - tw + 1), (1,), device=spec_aug.device).item())
                            spec_aug[_b, :, :, t0:t0 + tw] = 0.0
                return signal_aug, spec_aug

            def _run_lpft_pre_bn_adapt(_loader):
                if _loader is None:
                    return
                try:
                    import config as cfg
                    _bn_enable = bool(getattr(cfg, "TARGET_BN_ADAPT_ENABLE", True))
                    _bn_n_batch = int(getattr(cfg, "TARGET_BN_ADAPT_NUM_BATCHES", 10))
                    _bn_alpha = float(getattr(cfg, "TARGET_BN_ADAPT_ALPHA", 0.0))
                    _bn_auto_alpha = bool(getattr(cfg, "TARGET_BN_ADAPT_AUTO_ALPHA", False))
                    _bn_auto_alpha_min = float(getattr(cfg, "TARGET_BN_ADAPT_AUTO_ALPHA_MIN", 0.05))
                    _bn_auto_alpha_max = float(getattr(cfg, "TARGET_BN_ADAPT_AUTO_ALPHA_MAX", 0.95))
                except Exception:
                    _bn_enable = True
                    _bn_n_batch = 10
                    _bn_alpha = 0.0
                    _bn_auto_alpha = False
                    _bn_auto_alpha_min = 0.05
                    _bn_auto_alpha_max = 0.95
                if not _bn_enable:
                    return
                try:
                    if _lp_pre_bn_adapt_num_batches is not None:
                        _bn_n_batch = int(_lp_pre_bn_adapt_num_batches)
                except Exception:
                    pass
                if int(_bn_n_batch) == 0:
                    return
                if int(_bn_n_batch) < 0:
                    _max_batches = len(_loader) if hasattr(_loader, "__len__") else 10**9
                else:
                    _max_batches = int(_bn_n_batch)

                _saved_source = {
                    "signal_encoder": {k: v.detach().clone().cpu() for k, v in self.signal_encoder.state_dict().items()},
                    "spec_encoder": {k: v.detach().clone().cpu() for k, v in self.spec_encoder.state_dict().items()},
                }
                self.signal_encoder.train()
                self.spec_encoder.train()

                _domain_map = _build_domain_map()
                _domain_id_fallback = _domain_map.get(str(_dataset_upper).split("/")[0], 0)
                _n_run = 0
                with torch.no_grad():
                    for _bi, _batch in enumerate(_loader):
                        if _bi >= int(_max_batches):
                            break
                        _n_run += 1
                        _signal = _batch["signal"].to(self.device)
                        _spec = self.prepare_spec_batch(_batch)
                        _signal_mask = _batch.get("signal_mask")
                        _spec_mask = _batch.get("spec_mask")
                        if _signal_mask is None:
                            _signal_mask = torch.ones(
                                _signal.size(0), _signal.size(1), device=self.device, dtype=torch.bool
                            )
                        else:
                            _signal_mask = _signal_mask.to(self.device)
                        if _spec_mask is None:
                            _spec_mask = torch.ones(
                                _spec.size(0), _spec.size(1), device=self.device, dtype=torch.bool
                            )
                        else:
                            _spec_mask = _spec_mask.to(self.device)
                        _domains = _batch.get("domain", None)
                        if _domains is not None:
                            _domain_ids = torch.tensor(
                                [_domain_map.get(d, _domain_id_fallback) for d in _domains],
                                dtype=torch.long,
                                device=self.device,
                            )
                        else:
                            _domain_ids = torch.full(
                                (_signal.size(0),),
                                int(_domain_id_fallback),
                                dtype=torch.long,
                                device=self.device,
                            )
                        try:
                            self.signal_encoder(_signal, _signal_mask, domain_ids=_domain_ids)
                        except TypeError:
                            self.signal_encoder(_signal, _signal_mask)
                        try:
                            self.spec_encoder(_spec, _spec_mask, domain_ids=_domain_ids)
                        except TypeError:
                            self.spec_encoder(_spec, _spec_mask)

                _mixed_stats = 0
                _alpha_values = []
                if float(_bn_alpha) > 0.0:
                    _alpha = max(0.0, min(1.0, float(_bn_alpha)))
                    _a_min = max(0.0, min(1.0, float(_bn_auto_alpha_min)))
                    _a_max = max(_a_min, min(1.0, float(_bn_auto_alpha_max)))
                    with torch.no_grad():
                        for _module_name in ("signal_encoder", "spec_encoder"):
                            _module = getattr(self, _module_name, None)
                            if _module is None:
                                continue
                            _src_state = _saved_source.get(_module_name, {})
                            _alpha_by_prefix = {}
                            for _name, _tensor in _module.state_dict().items():
                                if ("running_mean" not in _name) and ("running_var" not in _name):
                                    continue
                                if _name not in _src_state:
                                    continue
                                _alpha_eff = _alpha
                                if _bn_auto_alpha:
                                    if "running_var" in _name:
                                        _src_var = _src_state[_name]
                                        _tgt_var = _tensor
                                        _src_m = float(torch.mean(torch.abs(_src_var)).item())
                                        _tgt_m = float(torch.mean(torch.abs(_tgt_var)).item())
                                        _alpha_eff = _src_m / (_src_m + _tgt_m + 1e-12)
                                        _alpha_eff = max(_a_min, min(_a_max, _alpha_eff))
                                        _alpha_by_prefix[_name.replace("running_var", "")] = _alpha_eff
                                    else:
                                        _k = _name.replace("running_mean", "")
                                        _alpha_eff = _alpha_by_prefix.get(_k, _alpha)
                                _src_t = _src_state[_name].to(device=_tensor.device, dtype=_tensor.dtype)
                                _tensor.copy_(_alpha_eff * _src_t + (1.0 - _alpha_eff) * _tensor)
                                _alpha_values.append(_alpha_eff)
                                _mixed_stats += 1

                self.signal_encoder.eval()
                self.spec_encoder.eval()
                if float(_bn_alpha) > 0.0:
                    _alpha_msg = f"alpha_bn={max(0.0, min(1.0, float(_bn_alpha))):.2f}"
                    if _bn_auto_alpha and _alpha_values:
                        _alpha_msg = (
                            f"alpha_bn=auto[{min(_alpha_values):.2f},{max(_alpha_values):.2f}]"
                            f"/mean={float(np.mean(_alpha_values)):.2f}"
                        )
                    print(
                        f"[Supervised Finetune] Pre-LPFT BN-Adapt{dataset_str}: "
                        f"updated BN stats with {_n_run} batches, {_alpha_msg}, mixed_stats={_mixed_stats}"
                    )
                else:
                    print(
                        f"[Supervised Finetune] Pre-LPFT BN-Adapt{dataset_str}: "
                        f"updated BN stats with {_n_run} batches"
                    )

            _lp_base_domain = str(_dataset_upper).split("/")[0]
            _lp_pre_bn_apply = bool(
                is_flat
                and _lp_pre_bn_adapt_enable
                and (_lp_base_domain in set(_lp_pre_bn_adapt_domains))
                and ((not _is_selftrain_call) or bool(_lp_pre_bn_adapt_on_selftrain))
                and ((not _is_retrain_full_call) or bool(_lp_pre_bn_adapt_on_retrain_full))
            )
            if _lp_pre_bn_apply:
                _bn_loader = finetune_loader if finetune_loader is not None else train_loader
                _run_lpft_pre_bn_adapt(_bn_loader)

            _record_test_every = int(record_test_every) if record_test_every is not None else None
            _lpft_recorded_steps = set()

            def _maybe_record_lpft_point(step_idx, avg_loss=None, val_metrics=None):
                if lpft_curve is None or holdout_test_loader is None:
                    return
                _every = _record_test_every if _record_test_every is not None else int(_eval_every)
                if int(step_idx) != 0 and (int(step_idx) % max(int(_every), 1) != 0):
                    return
                if int(step_idx) in _lpft_recorded_steps:
                    return
                _lpft_recorded_steps.add(int(step_idx))
                try:
                    _test_res = self.evaluate_target_domain_accuracy(
                        holdout_test_loader,
                        dataset_name=dataset_name,
                        diag_num_classes=int(num_classes_eval),
                    )
                    _test_payload = {
                        "test_top1": float(_test_res.get("diag_acc", float("nan"))),
                        "test_macro_f1": float("nan"),
                        "test_nf_acc": float(_test_res.get("nf_acc", float("nan"))),
                        "diag_acc_raw": float(_test_res.get("diag_acc_raw", float("nan"))),
                    }
                    if callable(lpft_curve_eval_fn):
                        try:
                            _extra = lpft_curve_eval_fn(
                                self,
                                holdout_test_loader,
                                dataset_name,
                                int(num_classes_eval),
                            )
                            if isinstance(_extra, dict):
                                for _ek, _ev in _extra.items():
                                    if _ev is None:
                                        continue
                                    if _ek == "final_decision_path":
                                        _test_payload[_ek] = str(_ev)
                                        continue
                                    try:
                                        _fv = float(_ev)
                                        if _fv == _fv:
                                            _test_payload[_ek] = _fv
                                    except (TypeError, ValueError):
                                        pass
                        except Exception as _e_fd:
                            _test_payload["final_decision_eval_error"] = str(_e_fd)
                except Exception as _e_rec:
                    _test_payload = {"error": str(_e_rec)}
                _point = {"step": int(step_idx)}
                if avg_loss is not None:
                    _point["train_loss"] = float(avg_loss)
                if isinstance(val_metrics, dict):
                    _point["val"] = {
                        k: (float(v) if v is not None and v == v else v)
                        for k, v in val_metrics.items()
                        if not str(k).startswith("_")
                    }
                _point["test"] = _test_payload
                lpft_curve.append(_point)

            _maybe_record_lpft_point(0)

            # 微调循环
            for step in range(num_steps):
                total_loss = 0.0
                n_batches = 0
                warned_nonfinite = False
                _global_lr_scale = float(_lp_global_lr_scale(step))
                for _i, _pg in enumerate(optimizer.param_groups):
                    _pg["lr"] = float(_initial_group_lrs[_i]) * float(_global_lr_scale)
                # ATT180: progressive unfreeze stage transition (0-based step).
                # stage1: [0, s2_start), stage2: [s2_start, s3_start), stage3: [s3_start, s4_start), stage4: [s4_start, ...)
                lp_stage = 4 if (is_flat and lp_full_backbone_direct) else 1
                if is_flat and lp_prog_active:
                    if int(step) >= int(lp_unfreeze_stage4_start):
                        lp_stage = 4
                    elif int(step) >= int(lp_unfreeze_stage3_start):
                        lp_stage = 3
                    elif int(step) >= int(lp_unfreeze_stage2_start):
                        lp_stage = 2
                    # Enable stage-2 params from stage>=2
                    for _p in prog_params_stage2:
                        _p.requires_grad = bool(lp_stage >= 2)
                    # Enable stage-3 params from stage>=3
                    for _p in prog_params_stage3:
                        _p.requires_grad = bool(lp_stage >= 3)
                    # Enable stage-4 params from stage>=4
                    for _p in prog_params_stage4:
                        _p.requires_grad = bool(lp_stage >= 4)
                    if int(step) in (int(lp_unfreeze_stage2_start), int(lp_unfreeze_stage3_start), int(lp_unfreeze_stage4_start)):
                        n_now = sum(p.numel() for p in params_to_optimize if p.requires_grad)
                        print(
                            f"[Supervised Finetune] Progressive stage -> {lp_stage}{dataset_str} "
                            f"(step={step}, active_trainable={n_now:,})"
                        )
                    if lp_stage >= 2:
                        self.spec_encoder.train()
                        # ATT254(S1): freeze BN running-stats from stage2 onward to avoid
                        # tiny-shot stage-transition shocks caused by BN train/eval statistic switches.
                        for _m in self.spec_encoder.modules():
                            if isinstance(_m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                                _m.eval()
                    else:
                        self.spec_encoder.eval()
                    # ATT204(C): warmup backbone LR after each progressive stage activation
                    # to avoid CE spikes when newly-unfrozen params receive full Adam step immediately.
                    if (
                        _backbone_group_idx is not None
                        and lp_backbone_stage_warmup_enable
                        and int(lp_backbone_stage_warmup_steps) > 0
                    ):
                        _warm_start = None
                        if int(step) >= int(lp_unfreeze_stage4_start):
                            _warm_start = int(lp_unfreeze_stage4_start)
                        elif int(step) >= int(lp_unfreeze_stage3_start):
                            _warm_start = int(lp_unfreeze_stage3_start)
                        elif int(step) >= int(lp_unfreeze_stage2_start):
                            _warm_start = int(lp_unfreeze_stage2_start)
                        if _warm_start is not None:
                            _delta = int(step) - int(_warm_start)
                            _ws = max(1, int(lp_backbone_stage_warmup_steps))
                            _p = min(1.0, max(0.0, float(_delta) / float(_ws)))
                            _min_scale = max(0.0, min(1.0, float(lp_backbone_stage_warmup_min_scale)))
                            _scale = _min_scale + (1.0 - _min_scale) * _p
                            optimizer.param_groups[_backbone_group_idx]["lr"] = (
                                float(_backbone_lr_target) * float(_scale) * float(_global_lr_scale)
                            )
                        else:
                            optimizer.param_groups[_backbone_group_idx]["lr"] = (
                                float(_backbone_lr_target) * float(_global_lr_scale)
                            )
                elif is_flat and lp_full_backbone_direct:
                    self.spec_encoder.train()
                    # Keep BN running stats frozen for tiny-shot stability.
                    for _m in self.spec_encoder.modules():
                        if isinstance(_m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                            _m.eval()
                    if (
                        _backbone_group_idx is not None
                        and lp_backbone_stage_warmup_enable
                        and int(lp_backbone_stage_warmup_steps) > 0
                    ):
                        _ws = max(1, int(lp_backbone_stage_warmup_steps))
                        _p = min(1.0, max(0.0, float(int(step)) / float(_ws)))
                        _min_scale = max(0.0, min(1.0, float(lp_backbone_stage_warmup_min_scale)))
                        _scale = _min_scale + (1.0 - _min_scale) * _p
                        optimizer.param_groups[_backbone_group_idx]["lr"] = (
                            float(_backbone_lr_target) * float(_scale) * float(_global_lr_scale)
                        )

                _lp_sam_active = bool(
                    _lp_sam_base_active
                    and ((not bool(lp_sam_stage1_only)) or int(lp_stage) <= 1)
                )
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    signal = batch["signal"].to(self.device)
                    signal_mask = batch["signal_mask"].to(self.device)
                    spec = self.prepare_spec_batch(batch)
                    spec_mask = batch["spec_mask"].to(self.device)
                    labels = batch["label"].to(self.device)  # [B]
                    domains = batch.get("domain", None)  # List[str] or None

                    # Skip non-finite batches early (prevents NaN logits and wasted steps)
                    if not torch.isfinite(signal).all() or not torch.isfinite(spec).all():
                        if not warned_nonfinite:
                            def _finite_stats(x):
                                x_det = x.detach()
                                return {
                                    "nan": int(torch.isnan(x_det).sum().item()),
                                    "inf": int(torch.isinf(x_det).sum().item()),
                                    "min": float(torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0).min().item()),
                                    "max": float(torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0).max().item()),
                                }
                            print(f"[Supervised Finetune] Warning: Non-finite inputs detected{dataset_str}. Skipping batch.")
                            print(f"  signal stats: {_finite_stats(signal)}")
                            print(f"  spec   stats: {_finite_stats(spec)}")
                            warned_nonfinite = True
                        continue
                    
                    # Encode domain_ids for domain-aware fault head and LoRA
                    domain_ids = None
                    if domains is not None:
                        domain_map = _build_domain_map()
                        domain_ids = torch.tensor([domain_map.get(d, 0) for d in domains], dtype=torch.long, device=self.device)
                    
                    # ATT201(D): target-domain augmentation during LP-FT.
                    signal, spec = _augment_target_batch(signal, spec)

                    # ATT446/ATT449/ATT464b: bypass features for LP-FT loop
                    self._last_z_wst = None
                    self._last_z_env = None
                    self._last_ratio = None
                    self._last_z_scd = None
                    if self._wst_bypass_active:
                        _wst_raw = batch.get("wst_spec", None)
                        if _wst_raw is not None:
                            with torch.no_grad():
                                self._last_z_wst = self.wst_encoder(_wst_raw.to(self.device))
                    if self._env_bypass_active:
                        _env_raw = batch.get("env_spec", None)
                        if _env_raw is not None:
                            with torch.no_grad():
                                self._last_z_env = self.env_encoder(_env_raw.to(self.device))
                        _ratio_raw = batch.get("bpfi_bpfo_ratio", None)
                        if _ratio_raw is not None:
                            self._last_ratio = _ratio_raw.to(self.device)
                    if self._scd_bypass_active:
                        _scd_raw = batch.get("scd_spec", None)
                        if _scd_raw is not None:
                            with torch.no_grad():
                                self._last_z_scd = self.scd_encoder(_scd_raw.to(self.device))

                    # Forward through encoders
                    # - Flat LP: no LoRA; we can safely detach embeddings for stability and speed.
                    # - Hierarchical finetune: keep grad enabled for LoRA/decoder adapters.
                    if is_flat:
                        # Signal encoder remains frozen in LP-FT.
                        # ATT379e: removed F.normalize for train-eval consistency
                        with torch.no_grad():
                            z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        z_sig = z_sig.detach()
                        _train_spec_encoder = bool((lp_prog_active and lp_stage >= 2) or lp_full_backbone_direct)
                        if _train_spec_encoder:
                            z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                        else:
                            with torch.no_grad():
                                z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                            z_spec = z_spec.detach()
                    else:
                        z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)

                    if not torch.isfinite(z_sig).all() or not torch.isfinite(z_spec).all():
                        if not warned_nonfinite:
                            print(f"[Supervised Finetune] Warning: Non-finite encoder outputs detected{dataset_str}. Skipping batch.")
                            warned_nonfinite = True
                        continue
                    
                    if is_flat:
                        z_fusion = self._fuse_embeddings(z_sig, z_spec, signal=None, signal_mask=None, spec=None, spec_mask=None)
                    else:
                        z_fusion = self._fuse_embeddings(
                            z_sig, z_spec,
                            signal=signal, signal_mask=signal_mask,
                            spec=spec, spec_mask=spec_mask
                        )

                    if not torch.isfinite(z_fusion).all():
                        if not warned_nonfinite:
                            print(f"[Supervised Finetune] Warning: Non-finite fusion embedding detected{dataset_str}. Skipping batch.")
                            warned_nonfinite = True
                        continue
                    
                    # Get decoder output with parts
                    logits4, parts = self.fusion_decoder(
                        z_fusion,
                        domain_ids=domain_ids,
                        text_prototypes=self._active_text_prototypes(),
                        return_parts=True
                    )
                    # Allow large finite negative masks (e.g., HIT rolling). Only skip true NaNs.
                    if torch.isnan(logits4).any():
                        if not warned_nonfinite:
                            print(f"[Supervised Finetune] Warning: Non-finite decoder logits detected{dataset_str}. Skipping batch.")
                            warned_nonfinite = True
                        continue

                    # (Attempt 44 diagnostic) Print logits distribution for the very first batch.
                    # Helps distinguish "scale/temperature mismatch" vs "embedding shift" when LP-FT starts with huge loss.
                    if int(step) == 0 and int(n_batches) == 0:
                        try:
                            with torch.no_grad():
                                lx = logits4.detach().float()
                                lmin = float(lx.min().item())
                                lmax = float(lx.max().item())
                                lmean = float(lx.mean().item())
                                lstd = float(lx.std(unbiased=False).item())
                                # label histogram
                                uniq, cnt = labels.detach().cpu().unique(return_counts=True)
                                hist = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(uniq.tolist(), cnt.tolist())])
                            print(
                                f"[Supervised Finetune] Step0 batch0 logits stats: "
                                f"min={lmin:.4f}, max={lmax:.4f}, mean={lmean:.4f}, std={lstd:.4f}; "
                                f"labels={{ {hist} }}{dataset_str}"
                            )
                        except Exception:
                            pass
                    
                    # NF loss: only meaningful for hierarchical head.
                    nf_loss = torch.tensor(0.0, device=self.device)
                    if not is_flat:
                        nf_labels = (labels != 0).long()  # [B] 0=normal, 1=fault
                        logits_nf_global = parts.get("logits_nf_global", None)
                        if logits_nf_global is None:
                            logit_normal = logits4[:, :1]
                            logit_fault = torch.logsumexp(logits4[:, 1:], dim=1, keepdim=True)
                            logits_nf_global = torch.cat([logit_normal, logit_fault], dim=1)
                        nf_loss = F.cross_entropy(logits_nf_global, nf_labels)

                    # Supervised finetune weights (configurable; keep nf_weight low for HUST)
                    try:
                        import config as cfg
                        nf_weight = float(getattr(cfg, "SUP_FT_NF_WEIGHT", 0.05))
                        diag_weight = float(getattr(cfg, "SUP_FT_DIAG_WEIGHT", 1.0))
                        lp_flat_nf_weight = float(getattr(cfg, "TARGET_SUP_LP_FLAT_NF_WEIGHT", 0.10))
                    except Exception:
                        nf_weight = 0.05
                        diag_weight = 1.0
                        lp_flat_nf_weight = 0.10
                    
                    # Diagnosis loss
                    proto_anchor_loss = torch.tensor(0.0, device=self.device)
                    if is_flat:
                        # ATT228(S2): add explicit NF supervision for flat LP-FT
                        # by deriving NF logits from diagnosis logits.
                        _nf_labels = (labels != 0).long()
                        _nf_logit_normal = logits4[:, :1]
                        _nf_logit_fault = torch.logsumexp(logits4[:, 1:], dim=1, keepdim=True)
                        _nf_logits_flat = torch.cat([_nf_logit_normal, _nf_logit_fault], dim=1)
                        nf_loss = F.cross_entropy(_nf_logits_flat, _nf_labels)
                        nf_weight = float(max(0.0, lp_flat_nf_weight))

                        # Flat LP (ATT186): use SpecHead CE as the primary supervised signal.
                        # Rationale: in cross-domain target adaptation, z_spec keeps more class-discriminative
                        # structure than z_fusion in hard domains (e.g., HIT). We keep fusion CE as an auxiliary
                        # term to preserve compatibility and avoid abruptly discarding decoder information.
                        if int(num_classes_eval) == 3:
                            labels_use = labels.clamp(0, 2)
                            fusion_diag_loss = F.cross_entropy(logits4[:, :3], labels_use)
                        else:
                            labels_use = labels
                            fusion_diag_loss = F.cross_entropy(logits4, labels_use)

                        diag_loss = fusion_diag_loss
                        try:
                            import config as cfg
                            lp_spec_w = float(getattr(cfg, "TARGET_SUP_LP_SPEC_LOSS_WEIGHT", 1.0))
                            lp_fusion_w = float(getattr(cfg, "TARGET_SUP_LP_FUSION_LOSS_WEIGHT", 0.30))
                            lp_spec_ls = float(getattr(cfg, "TARGET_SUP_LP_SPEC_LABEL_SMOOTHING", 0.0))
                            lp_supcon_enable = bool(getattr(cfg, "TARGET_SUP_LP_SUPCON_ENABLE", False))
                            lp_supcon_weight = float(getattr(cfg, "TARGET_SUP_LP_SUPCON_WEIGHT", 0.0))
                            lp_supcon_temp = float(getattr(cfg, "TARGET_SUP_LP_SUPCON_TEMPERATURE", 0.10))
                            lp_supcon_use_class_weights = bool(
                                getattr(cfg, "TARGET_SUP_LP_SUPCON_USE_CLASS_WEIGHTS", True)
                            )
                            lp_supcon_fault_only = bool(getattr(cfg, "TARGET_SUP_LP_SUPCON_FAULT_ONLY", False))
                        except Exception:
                            lp_spec_w = 1.0
                            lp_fusion_w = 0.30
                            lp_spec_ls = 0.0
                            lp_supcon_enable = False
                            lp_supcon_weight = 0.0
                            lp_supcon_temp = 0.10
                            lp_supcon_use_class_weights = True
                            lp_supcon_fault_only = False

                        if include_spec_head and getattr(self.spec_encoder, "has_spec_head", False):
                            if hasattr(self.spec_encoder, 'compute_spec_head_logits'):
                                spec_logits = self.spec_encoder.compute_spec_head_logits(z_spec, wst_aux=self._last_z_wst, env_aux=self._last_z_env, ratio_aux=self._last_ratio)
                            elif self.spec_encoder.classifier is not None:
                                spec_logits = self.spec_encoder.classifier(z_spec)
                            else:
                                spec_logits = None
                            if spec_logits is not None:
                                if int(num_classes_eval) == 3:
                                    spec_logits = spec_logits[:, :3]
                                if lp_spec_cb_focal_enable:
                                    _labels_use = labels_use.long().clamp(min=0, max=int(spec_logits.size(1)) - 1)
                                    _log_prob = F.log_softmax(spec_logits, dim=1)
                                    _prob = _log_prob.exp()
                                    _pt = _prob.gather(1, _labels_use.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
                                    _log_pt = _log_prob.gather(1, _labels_use.unsqueeze(1)).squeeze(1)
                                    if (
                                        lp_spec_cb_class_weights is not None
                                        and int(lp_spec_cb_class_weights.numel()) >= int(spec_logits.size(1))
                                    ):
                                        _alpha_t = lp_spec_cb_class_weights.to(spec_logits.device, spec_logits.dtype)[_labels_use]
                                    else:
                                        _alpha_t = torch.ones_like(_pt)
                                    _focal_factor = torch.pow(1.0 - _pt, float(lp_spec_cb_focal_gamma))
                                    spec_diag_loss = (-_alpha_t * _focal_factor * _log_pt).mean()
                                else:
                                    spec_diag_loss = F.cross_entropy(
                                        spec_logits,
                                        labels_use,
                                        label_smoothing=max(0.0, min(0.2, float(lp_spec_ls))),
                                    )
                                if (
                                    lp_proto_anchor_enable
                                    and float(lp_proto_anchor_w) > 0.0
                                    and int(step) >= int(lp_proto_anchor_warmup_steps)
                                ):
                                    try:
                                        _protos = self._active_text_prototypes()
                                        if _protos is not None and isinstance(_protos, torch.Tensor):
                                            _n_cls = int(spec_logits.size(1))
                                            _p = _protos.to(z_spec.device, z_spec.dtype)
                                            if int(_p.dim()) == 2 and int(_p.size(0)) >= _n_cls:
                                                _p = F.normalize(_p[:_n_cls], dim=-1)
                                                _l = labels_use.long().clamp(min=0, max=_n_cls - 1)
                                                _uniq = torch.unique(_l)
                                                _terms = []
                                                for _c in _uniq.tolist():
                                                    _m = (_l == int(_c))
                                                    if bool(_m.any()):
                                                        _mu_c = F.normalize(z_spec[_m].mean(dim=0, keepdim=False), dim=0)
                                                        _terms.append(1.0 - torch.sum(_mu_c * _p[int(_c)]))
                                                if len(_terms) > 0:
                                                    proto_anchor_loss = torch.stack(_terms).mean()
                                    except Exception:
                                        proto_anchor_loss = torch.tensor(0.0, device=self.device)
                                diag_loss = float(lp_spec_w) * spec_diag_loss + float(lp_fusion_w) * fusion_diag_loss
                                lp_supcon_loss = torch.tensor(0.0, device=self.device)
                                if bool(lp_supcon_enable) and float(lp_supcon_weight) > 0.0:
                                    _z_sc = z_spec
                                    _y_sc = labels_use
                                    if bool(lp_supcon_fault_only):
                                        _mask_sc = (_y_sc != 0)
                                        _z_sc = _z_sc[_mask_sc]
                                        _y_sc = _y_sc[_mask_sc]
                                    if int(_z_sc.size(0)) >= 2:
                                        lp_supcon_loss = supervised_contrastive_loss(
                                            _z_sc,
                                            _y_sc,
                                            temperature=float(lp_supcon_temp),
                                            use_class_weights=bool(lp_supcon_use_class_weights),
                                            return_stats=False,
                                        )
                                        diag_loss = diag_loss + float(lp_supcon_weight) * lp_supcon_loss
                            if int(step) == 0 and int(n_batches) == 0:
                                print(
                                    f"[Supervised Finetune] LP-loss blend{dataset_str}: "
                                    f"spec_w={lp_spec_w:.2f}, fusion_w={lp_fusion_w:.2f}, "
                                    f"spec_ls={max(0.0, min(0.2, float(lp_spec_ls))):.2f}, "
                                    f"spec_mode={'cb_focal' if lp_spec_cb_focal_enable else 'ce'}, "
                                    f"spec_ce={float(spec_diag_loss.item()):.4f}, "
                                    f"fusion_ce={float(fusion_diag_loss.item()):.4f}, "
                                    f"supcon_w={float(lp_supcon_weight):.2f}, "
                                    f"supcon={float(lp_supcon_loss.item()):.4f}, "
                                    f"nf_ce={float(nf_loss.item()):.4f}, "
                                    f"nf_w={float(nf_weight):.3f}, "
                                    f"proto_w={float(lp_proto_anchor_w):.3f}, "
                                    f"proto_loss={float(proto_anchor_loss.item()):.4f}"
                                )
                    else:
                        # Hierarchical head: fault-type CE on fault samples only (keeps NF and type decoupled).
                        logits_fault = parts.get("logits_fault", None)
                        if logits_fault is None:
                            logits_fault = logits4[:, 1:]

                        fault_mask = (labels != 0)
                        if fault_mask.any():
                            targets_fault = (labels[fault_mask] - 1).clamp(min=0, max=2)
                            if num_classes_eval == 3:
                                diag_logits = logits_fault[fault_mask, :2]
                                targets_fault = targets_fault.clamp(0, 1)
                            else:
                                diag_logits = logits_fault[fault_mask, :3]
                            diag_loss = F.cross_entropy(diag_logits, targets_fault.long())
                        else:
                            diag_loss = torch.tensor(0.0, device=self.device)

                    # Inner/Outer margin loss (STRUCTURAL constraints):
                    # Enforce inner/outer logits to beat others by a margin for true Inner/Outer samples.
                    inner_margin_loss = torch.tensor(0.0, device=self.device)
                    outer_margin_loss = torch.tensor(0.0, device=self.device)
                    # Defaults (safe even if config import fails)
                    outer_margin = 0.2
                    outer_margin_weight = 0.05
                    inner_margin = 0.2
                    inner_margin_weight = 0.05
                    try:
                        # Prefer config if available
                        try:
                            import config as cfg
                            outer_margin = float(getattr(cfg, "OUTER_MARGIN", outer_margin))
                            outer_margin_weight = float(getattr(cfg, "OUTER_MARGIN_WEIGHT", outer_margin_weight))
                            inner_margin = float(getattr(cfg, "INNER_MARGIN", inner_margin))
                            inner_margin_weight = float(getattr(cfg, "INNER_MARGIN_WEIGHT", inner_margin_weight))
                            use_normal_margin = bool(getattr(cfg, "USE_NORMAL_MARGIN_LOSS", False))
                            normal_margin = float(getattr(cfg, "NORMAL_MARGIN", 0.2))
                            normal_margin_weight = float(getattr(cfg, "NORMAL_MARGIN_WEIGHT", 0.05))
                            use_in_vn = bool(getattr(cfg, "USE_INNER_VS_NORMAL_MARGIN_LOSS", False))
                            in_vn_m = float(getattr(cfg, "INNER_VS_NORMAL_MARGIN", 0.2))
                            in_vn_w = float(getattr(cfg, "INNER_VS_NORMAL_MARGIN_WEIGHT", 0.0))
                            use_out_vn = bool(getattr(cfg, "USE_OUTER_VS_NORMAL_MARGIN_LOSS", False))
                            out_vn_m = float(getattr(cfg, "OUTER_VS_NORMAL_MARGIN", 0.2))
                            out_vn_w = float(getattr(cfg, "OUTER_VS_NORMAL_MARGIN_WEIGHT", 0.0))
                        except Exception:
                            use_normal_margin = False
                            normal_margin = 0.2
                            normal_margin_weight = 0.0
                            use_in_vn = False
                            in_vn_m = 0.2
                            in_vn_w = 0.0
                            use_out_vn = False
                            out_vn_m = 0.2
                            out_vn_w = 0.0
                            pass

                        # Only meaningful for 4-class label space (HUST/CWRU) and when diag logits are available
                        if num_classes_eval == 4 and (outer_margin_weight > 0 or inner_margin_weight > 0):
                            # IMPORTANT: apply margins on the same logits used for diagnosis decisions.
                            # Use `logits4` (decision logits). Do not rely on `logits_diag_outer`.
                            logits_outer_space = logits4
                            
                            # True inner class index = 1
                            inner_mask = (labels == 1)
                            if inner_margin_weight > 0 and inner_mask.any():
                                logits_i = logits_outer_space[inner_mask]  # [B_i, 4]
                                inner_logit = logits_i[:, 1]
                                other_max = torch.cat([logits_i[:, :1], logits_i[:, 2:]], dim=1).max(dim=1).values
                                inner_margin_loss = F.relu(other_max - inner_logit + inner_margin).mean()

                            # True outer class index = 2
                            outer_mask = (labels == 2)
                            if outer_margin_weight > 0 and outer_mask.any():
                                logits_o = logits_outer_space[outer_mask]  # [B_o, 4]
                                outer_logit = logits_o[:, 2]
                                other_max = torch.cat([logits_o[:, :2], logits_o[:, 3:]], dim=1).max(dim=1).values
                                outer_margin_loss = F.relu(other_max - outer_logit + outer_margin).mean()

                        # Normal margin loss (anti-collapse): encourage Normal to beat all faults by a margin.
                        normal_margin_loss = torch.tensor(0.0, device=self.device)
                        if use_normal_margin and normal_margin_weight > 0:
                            logits_outer_space = logits4
                            normal_mask = (labels == 0)
                            if normal_mask.any():
                                ln = logits_outer_space[normal_mask]
                                logit_n = ln[:, 0]
                                if num_classes_eval == 3:
                                    other_max = ln[:, 1:3].max(dim=1).values
                                else:
                                    other_max = ln[:, 1:].max(dim=1).values
                                normal_margin_loss = F.relu(other_max - logit_n + normal_margin).mean()

                        # Inner/Outer vs Normal margins (directly targets Inner/Outer→Normal absorption)
                        inner_vs_normal_loss = torch.tensor(0.0, device=self.device)
                        outer_vs_normal_loss = torch.tensor(0.0, device=self.device)
                        if num_classes_eval == 4:
                            if use_in_vn and in_vn_w > 0:
                                m = (labels == 1)
                                if m.any():
                                    li = logits4[m]
                                    inner_vs_normal_loss = F.relu(li[:, 0] - li[:, 1] + float(in_vn_m)).mean()
                            if use_out_vn and out_vn_w > 0:
                                m = (labels == 2)
                                if m.any():
                                    lo = logits4[m]
                                    outer_vs_normal_loss = F.relu(lo[:, 0] - lo[:, 2] + float(out_vn_m)).mean()
                    except Exception:
                        inner_margin_loss = torch.tensor(0.0, device=self.device)
                        outer_margin_loss = torch.tensor(0.0, device=self.device)
                        normal_margin_loss = torch.tensor(0.0, device=self.device)
                        inner_vs_normal_loss = torch.tensor(0.0, device=self.device)
                        outer_vs_normal_loss = torch.tensor(0.0, device=self.device)
                        in_vn_w = 0.0
                        out_vn_w = 0.0
                    
                    # Check for NaN (debugging)
                    if torch.isnan(diag_loss):
                        try:
                            if fault_mask.any():
                                p_min = float(probs_diag[fault_mask].min().item())
                                p_max = float(probs_diag[fault_mask].max().item())
                                uniq = diag_targets[fault_mask].unique().tolist()
                            else:
                                p_min = float(probs_diag.min().item()) if 'probs_diag' in locals() else 0.0
                                p_max = float(probs_diag.max().item()) if 'probs_diag' in locals() else 0.0
                                uniq = []
                        except Exception:
                            p_min, p_max, uniq = 0.0, 0.0, []
                        print(f"[Supervised Finetune] Warning: diag_loss is NaN. probs_diag min={p_min:.6f}, max={p_max:.6f}, labels={uniq}")
                        continue
                    
                    # Rolling vs Normal margin (HUST only): enforce rolling logit to beat normal by a margin.
                    rolling_vs_normal_loss = torch.tensor(0.0, device=self.device)
                    roll_vn_w = 0.0
                    if num_classes_eval == 4:
                        try:
                            import config as cfg
                            use_roll_vn = bool(getattr(cfg, "USE_ROLLING_VS_NORMAL_MARGIN_LOSS", False))
                            roll_vn_w = float(getattr(cfg, "ROLLING_VS_NORMAL_MARGIN_WEIGHT", 0.0))
                            roll_vn_m = float(getattr(cfg, "ROLLING_VS_NORMAL_MARGIN", 0.2))
                        except Exception:
                            use_roll_vn = False
                            roll_vn_w = 0.0
                            roll_vn_m = 0.2
                        if use_roll_vn and roll_vn_w > 0:
                            m = (labels == 3)
                            if m.any():
                                _roll_logits = logits4[m]
                                rolling_vs_normal_loss = F.relu(
                                    _roll_logits[:, 0] - _roll_logits[:, 3] + float(roll_vn_m)
                                ).mean()

                    # ATT224(S3): use unlabeled target-domain batches to regularize LP-FT decision boundary.
                    # - cond_entropy: pushes confident predictions on target unlabeled data.
                    # - diversity(KL to uniform): avoids degenerate one-class collapse.
                    semi_loss = torch.tensor(0.0, device=self.device)
                    semi_cond_entropy = None
                    semi_div_kl = None
                    if (
                        lp_semi_enable
                        and include_spec_head
                        and getattr(self.spec_encoder, "has_spec_head", False)
                        and int(step) >= int(lp_semi_warmup_steps)
                    ):
                        ubatch = None
                        if unlabeled_iter is not None:
                            try:
                                ubatch = next(unlabeled_iter)
                            except StopIteration:
                                try:
                                    unlabeled_iter = iter(unlabeled_loader)
                                    ubatch = next(unlabeled_iter)
                                except Exception:
                                    ubatch = None
                            except Exception:
                                ubatch = None
                        if ubatch is not None:
                            try:
                                u_spec = self.prepare_spec_batch(ubatch)
                                u_spec_mask = ubatch["spec_mask"].to(self.device)
                                u_domains = ubatch.get("domain", None)
                                u_domain_ids = None
                                if u_domains is not None:
                                    domain_map = _build_domain_map()
                                    u_domain_ids = torch.tensor(
                                        [domain_map.get(d, 0) for d in u_domains],
                                        dtype=torch.long,
                                        device=self.device,
                                    )
                                _train_spec_encoder = bool((lp_prog_active and lp_stage >= 2) or lp_full_backbone_direct)
                                if _train_spec_encoder:
                                    u_z_spec = self.spec_encoder(u_spec, u_spec_mask, domain_ids=u_domain_ids)
                                    u_z_spec = F.normalize(u_z_spec, dim=-1)
                                else:
                                    with torch.no_grad():
                                        u_z_spec = self.spec_encoder(u_spec, u_spec_mask, domain_ids=u_domain_ids)
                                        u_z_spec = F.normalize(u_z_spec, dim=-1)
                                    u_z_spec = u_z_spec.detach()
                                if hasattr(self.spec_encoder, 'compute_spec_head_logits'):
                                    u_spec_logits = self.spec_encoder.compute_spec_head_logits(u_z_spec, wst_aux=self._last_z_wst, env_aux=self._last_z_env, ratio_aux=self._last_ratio)
                                elif self.spec_encoder.classifier is not None:
                                    u_spec_logits = self.spec_encoder.classifier(u_z_spec)
                                else:
                                    u_spec_logits = None
                                if u_spec_logits is None:
                                    raise RuntimeError("spec_logits unavailable for semi-supervised")
                                if int(num_classes_eval) == 3:
                                    u_spec_logits = u_spec_logits[:, :3]
                                u_prob = torch.softmax(u_spec_logits, dim=1)
                                _eps = 1e-8
                                semi_cond_entropy = (-(u_prob * torch.log(u_prob + _eps)).sum(dim=1)).mean()
                                p_bar = u_prob.mean(dim=0)
                                _n_cls = max(2, int(p_bar.shape[0]))
                                u_uniform = torch.full_like(p_bar, 1.0 / float(_n_cls))
                                semi_div_kl = F.kl_div(torch.log(p_bar + _eps), u_uniform, reduction="sum")
                                semi_loss = (
                                    float(lp_semi_ent_w) * semi_cond_entropy
                                    + float(lp_semi_div_w) * semi_div_kl
                                )
                            except Exception:
                                semi_loss = torch.tensor(0.0, device=self.device)

                    # Combined loss
                    loss = (
                        float(nf_weight) * nf_loss
                        + float(diag_weight) * diag_loss
                        + outer_margin_weight * outer_margin_loss
                        + inner_margin_weight * inner_margin_loss
                        + normal_margin_weight * normal_margin_loss
                        + float(in_vn_w) * inner_vs_normal_loss
                        + float(out_vn_w) * outer_vs_normal_loss
                        + float(roll_vn_w) * rolling_vs_normal_loss
                        + float(lp_proto_anchor_w) * proto_anchor_loss
                        + semi_loss
                    )
                    if int(step) == 0 and int(n_batches) == 0 and lp_semi_enable:
                        try:
                            print(
                                f"[Supervised Finetune] LP-semi regularization{dataset_str}: "
                                f"enable=1, warmup={int(lp_semi_warmup_steps)}, "
                                f"ent_w={float(lp_semi_ent_w):.3f}, div_w={float(lp_semi_div_w):.3f}"
                            )
                        except Exception:
                            pass
                    # ATT201(A): L2-SP for progressively-unfrozen spec backbone params.
                    if l2sp_refs and float(lp_l2sp_weight) > 0:
                        _l2sp_loss = torch.tensor(0.0, device=self.device)
                        _l2sp_count = 0
                        for _p in params_to_optimize:
                            _pid = id(_p)
                            _ref = l2sp_refs.get(_pid, None)
                            if (_ref is None) or (not _p.requires_grad):
                                continue
                            _l2sp_loss = _l2sp_loss + (_p - _ref).pow(2).sum()
                            _l2sp_count += int(_p.numel())
                        if _l2sp_count > 0:
                            loss = loss + float(lp_l2sp_weight) * (_l2sp_loss / float(_l2sp_count))
                    if (not loss.requires_grad) or (loss.grad_fn is None):
                        raise RuntimeError(
                            "[Supervised Finetune] Loss has no grad_fn (no trainable params connected). "
                            "For flat LP-FT, ensure FusionDecoder.classifier is trainable and included in optimizer."
                        )
                    
                    # Backward (optionally with SAM two-pass update on flat LP-FT)
                    if _lp_sam_active:
                        loss.backward()
                        _sam_rho = max(0.0, float(lp_sam_rho))
                        _sam_scale = 0.0
                        _sam_perturb = {}
                        if _sam_rho > 0:
                            with torch.no_grad():
                                _grad_norm_sq = 0.0
                                for _p in params_to_optimize:
                                    if _p.grad is None:
                                        continue
                                    _g = _p.grad.detach()
                                    _grad_norm_sq += float(torch.sum(_g * _g).item())
                                _grad_norm = math.sqrt(max(_grad_norm_sq, 0.0))
                                if _grad_norm > 0:
                                    _sam_scale = float(_sam_rho) / float(_grad_norm + 1e-12)
                                    for _p in params_to_optimize:
                                        if _p.grad is None:
                                            continue
                                        _e = _p.grad.detach() * float(_sam_scale)
                                        _p.add_(_e)
                                        _sam_perturb[id(_p)] = _e
                        optimizer.zero_grad()
                        # Second pass on perturbed weights.
                        _sam_loss = loss
                        try:
                            if torch.isfinite(signal).all() and torch.isfinite(spec).all():
                                _sam_train_spec_encoder = bool((lp_prog_active and lp_stage >= 2) or lp_full_backbone_direct)
                                with torch.no_grad():
                                    _sam_z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                                    _sam_z_sig = F.normalize(_sam_z_sig, dim=-1)
                                if _sam_train_spec_encoder:
                                    _sam_z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                                    _sam_z_spec = F.normalize(_sam_z_spec, dim=-1)
                                else:
                                    with torch.no_grad():
                                        _sam_z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                                        _sam_z_spec = F.normalize(_sam_z_spec, dim=-1)
                                    _sam_z_spec = _sam_z_spec.detach()
                                _sam_z_fusion = self._fuse_embeddings(
                                    _sam_z_sig.detach(), _sam_z_spec, signal=None, signal_mask=None, spec=None, spec_mask=None
                                )
                                _sam_z_fusion = F.normalize(_sam_z_fusion, dim=-1)
                                _sam_logits4, _ = self.fusion_decoder(
                                    _sam_z_fusion,
                                    domain_ids=domain_ids,
                                    text_prototypes=self._active_text_prototypes(),
                                    return_parts=True
                                )
                                _sam_nf_labels = (labels != 0).long()
                                _sam_nf_logit_normal = _sam_logits4[:, :1]
                                _sam_nf_logit_fault = torch.logsumexp(_sam_logits4[:, 1:], dim=1, keepdim=True)
                                _sam_nf_logits = torch.cat([_sam_nf_logit_normal, _sam_nf_logit_fault], dim=1)
                                _sam_nf_loss = F.cross_entropy(_sam_nf_logits, _sam_nf_labels)
                                if int(num_classes_eval) == 3:
                                    _sam_labels_use = labels.clamp(0, 2)
                                    _sam_fusion_diag = F.cross_entropy(_sam_logits4[:, :3], _sam_labels_use)
                                else:
                                    _sam_labels_use = labels
                                    _sam_fusion_diag = F.cross_entropy(_sam_logits4, _sam_labels_use)
                                _sam_diag_loss = _sam_fusion_diag
                                if (
                                    include_spec_head
                                    and getattr(self.spec_encoder, "has_spec_head", False)
                                ):
                                    if hasattr(self.spec_encoder, 'compute_spec_head_logits'):
                                        _sam_spec_logits = self.spec_encoder.compute_spec_head_logits(_sam_z_spec, wst_aux=self._last_z_wst, env_aux=self._last_z_env, ratio_aux=self._last_ratio)
                                    elif self.spec_encoder.classifier is not None:
                                        _sam_spec_logits = self.spec_encoder.classifier(_sam_z_spec)
                                    else:
                                        _sam_spec_logits = None
                                    if _sam_spec_logits is not None:
                                        if int(num_classes_eval) == 3:
                                            _sam_spec_logits = _sam_spec_logits[:, :3]
                                        _sam_spec_diag = F.cross_entropy(
                                            _sam_spec_logits,
                                            _sam_labels_use,
                                            label_smoothing=max(0.0, min(0.2, float(lp_spec_ls))),
                                        )
                                        _sam_diag_loss = float(lp_spec_w) * _sam_spec_diag + float(lp_fusion_w) * _sam_fusion_diag
                                _sam_loss = float(nf_weight) * _sam_nf_loss + float(diag_weight) * _sam_diag_loss
                        except Exception:
                            _sam_loss = loss
                        _sam_loss.backward()
                        with torch.no_grad():
                            for _p in params_to_optimize:
                                _e = _sam_perturb.get(id(_p), None)
                                if _e is not None:
                                    _p.sub_(_e)
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=0.1)
                        optimizer.step()
                    else:
                        loss.backward()
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=0.1)
                        optimizer.step()

                    if _lp_ema_enable and val_loader is not None:
                        try:
                            if self.fusion_decoder is not None:
                                ema_decoder_state = _ema_update_state(
                                    ema_decoder_state,
                                    self.fusion_decoder.state_dict(),
                                    _lp_ema_decay,
                                )
                            if is_flat and self.spec_encoder is not None:
                                ema_spec_state = _ema_update_state(
                                    ema_spec_state,
                                    self.spec_encoder.state_dict(),
                                    _lp_ema_decay,
                                )
                            ema_step = int(step)
                        except Exception:
                            pass
                    
                    total_loss += loss.item()
                    n_batches += 1
                    
                
                if step % max(1, num_steps // 5) == 0:
                    avg_loss = total_loss / max(n_batches, 1)
                    try:
                        _lr_msg = ", ".join([f"{float(pg.get('lr', 0.0)):.2e}" for pg in optimizer.param_groups])
                    except Exception:
                        _lr_msg = "N/A"
                    print(
                        f"[Supervised Finetune] Step {step}/{num_steps}, avg_loss={avg_loss:.4f}, "
                        f"lr_scale={_global_lr_scale:.3f}, lr_groups=[{_lr_msg}]{dataset_str}"
                    )

                # (Attempt 52) Validate + early stop
                if val_loader is not None and (step % int(_eval_every) == 0):
                    v = _eval_on_val()
                    if v is not None:
                        cur = v.get(_es_metric, None)
                        if cur is None:
                            # Fallback to CE if F1 is unavailable
                            cur = v.get("ce", None)
                            _es_metric = "ce"
                        try:
                            # CE-type metrics are "lower is better", including spec_ce.
                            if _es_metric in ("ce", "spec_ce"):
                                improved = (best_val is None) or (float(cur) < float(best_val) - float(_min_delta))
                            else:
                                improved = (best_val is None) or (float(cur) > float(best_val) + float(_min_delta))
                        except Exception:
                            improved = False
                        v_ce = v.get("ce", None)
                        v_f1 = v.get("macro_f1", None)
                        v_spec_ce = v.get("spec_ce", None)
                        v_spec_f1 = v.get("spec_macro_f1", None)
                        try:
                            msg = f"[Supervised Finetune] (Val-ES){dataset_str} step={step}: val_ce={float(v_ce):.4f}"
                            if v_f1 is not None:
                                msg += f", val_macro_f1={float(v_f1):.4f}"
                            if v_spec_ce is not None:
                                msg += f", val_spec_ce={float(v_spec_ce):.4f}"
                            if v_spec_f1 is not None:
                                msg += f", val_spec_macro_f1={float(v_spec_f1):.4f}"
                            _n_splits = int(v.get("_num_splits", 1))
                            if _n_splits > 1:
                                msg += (
                                    f", val_splits={_n_splits}"
                                    f", reduce={_val_multi_split_metric_reduce}"
                                )
                            msg += f" | best_{_es_metric}={best_val if best_val is not None else 'None'}"
                            print(msg)
                        except Exception:
                            pass
                        _maybe_record_lpft_point(
                            int(step),
                            avg_loss=(total_loss / max(n_batches, 1)) if n_batches > 0 else None,
                            val_metrics=v if isinstance(v, dict) else None,
                        )

                        # ATT191(B): update ce/spec_ce best snapshots independently.
                        for _m in ("ce", "spec_ce"):
                            _mv = v.get(_m, None)
                            if _mv is None:
                                continue
                            try:
                                _improved_m = (
                                    metric_best[_m] is None
                                    or float(_mv) < float(metric_best[_m]) - float(_min_delta)
                                )
                            except Exception:
                                _improved_m = False
                            if not _improved_m:
                                continue
                            metric_best[_m] = float(_mv)
                            metric_best_state[_m]["step"] = int(step)
                            try:
                                metric_best_state[_m]["decoder"] = {
                                    k: t.clone().detach() for k, t in self.fusion_decoder.state_dict().items()
                                }
                            except Exception:
                                metric_best_state[_m]["decoder"] = None
                            if is_flat:
                                try:
                                    metric_best_state[_m]["spec"] = {
                                        k: t.clone().detach() for k, t in self.spec_encoder.state_dict().items()
                                    }
                                except Exception:
                                    metric_best_state[_m]["spec"] = None

                        if improved:
                            best_val = float(cur)
                            best_step = int(step)
                            bad_epochs = 0
                            try:
                                best_decoder_state = {k: t.clone().detach() for k, t in self.fusion_decoder.state_dict().items()}
                            except Exception:
                                best_decoder_state = None
                            if is_flat:
                                try:
                                    best_spec_state = {k: t.clone().detach() for k, t in self.spec_encoder.state_dict().items()}
                                except Exception:
                                    best_spec_state = None
                        else:
                            bad_epochs += 1
                            if (step >= int(_min_steps)) and (bad_epochs >= int(_patience)):
                                print(
                                    f"[Supervised Finetune] (Val-ES){dataset_str} early stop at step={step} "
                                    f"(patience={_patience}, best_step={best_step}, best_{_es_metric}={best_val})"
                                )
                                break

                # Record holdout-test curve points independently of Val-ES, so that
                # K-shot LP-FT trajectories are captured even when no val_loader is
                # provided (e.g. run_nshot_lpft_curves.py). Dedup inside the helper
                # avoids double-recording steps already logged by the Val-ES branch.
                _maybe_record_lpft_point(
                    int(step),
                    avg_loss=(total_loss / max(n_batches, 1)) if n_batches > 0 else None,
                )

            # Restore best decoder state (if early stopping used)
            if val_loader is not None and best_decoder_state is not None and self.fusion_decoder is not None:
                try:
                    self.fusion_decoder.load_state_dict(best_decoder_state)
                    print(f"[Supervised Finetune] (Val-ES){dataset_str} restored best decoder at step={best_step}")
                except Exception:
                    pass
            if val_loader is not None and best_spec_state is not None and self.spec_encoder is not None:
                try:
                    self.spec_encoder.load_state_dict(best_spec_state)
                    print(f"[Supervised Finetune] (Val-ES){dataset_str} restored best spec_encoder at step={best_step}")
                except Exception:
                    pass
            if (
                val_loader is not None
                and _lp_ema_enable
                and _lp_ema_eval_enable
                and ema_decoder_state is not None
                and self.fusion_decoder is not None
            ):
                try:
                    cur_decoder_state = {
                        k: t.clone().detach() for k, t in self.fusion_decoder.state_dict().items()
                    }
                except Exception:
                    cur_decoder_state = None
                try:
                    cur_spec_state = (
                        {k: t.clone().detach() for k, t in self.spec_encoder.state_dict().items()}
                        if (is_flat and self.spec_encoder is not None)
                        else None
                    )
                except Exception:
                    cur_spec_state = None
                try:
                    self.fusion_decoder.load_state_dict(ema_decoder_state)
                    if is_flat and (ema_spec_state is not None) and (self.spec_encoder is not None):
                        self.spec_encoder.load_state_dict(ema_spec_state)
                    _v_ema = _eval_on_val()
                    _cur_ema = _v_ema.get(_es_metric, None) if isinstance(_v_ema, dict) else None
                    if _cur_ema is not None:
                        ema_metric_val = float(_cur_ema)
                        _ema_better = False
                        if _es_metric in ("ce", "spec_ce"):
                            _ema_better = (best_val is None) or (
                                float(ema_metric_val) < float(best_val) - float(_min_delta)
                            )
                        else:
                            _ema_better = (best_val is None) or (
                                float(ema_metric_val) > float(best_val) + float(_min_delta)
                            )
                        print(
                            f"[Supervised Finetune] (LP-EMA){dataset_str} step={ema_step}, "
                            f"{_es_metric}={float(ema_metric_val):.6f}, best={best_val if best_val is not None else 'None'}"
                        )
                        if _ema_better:
                            best_val = float(ema_metric_val)
                            best_step = int(ema_step) if ema_step is not None else best_step
                            try:
                                if self.fusion_decoder is not None:
                                    best_decoder_state = {
                                        k: t.clone().detach() for k, t in self.fusion_decoder.state_dict().items()
                                    }
                                if is_flat and self.spec_encoder is not None:
                                    best_spec_state = {
                                        k: t.clone().detach() for k, t in self.spec_encoder.state_dict().items()
                                    }
                                print(
                                    f"[Supervised Finetune] (LP-EMA){dataset_str} selected EMA checkpoint "
                                    f"as final best_{_es_metric}"
                                )
                            except Exception:
                                pass
                except Exception:
                    pass
                finally:
                    try:
                        if cur_decoder_state is not None and self.fusion_decoder is not None:
                            self.fusion_decoder.load_state_dict(cur_decoder_state)
                        if is_flat and (cur_spec_state is not None) and (self.spec_encoder is not None):
                            self.spec_encoder.load_state_dict(cur_spec_state)
                    except Exception:
                        pass
                    try:
                        if best_decoder_state is not None and self.fusion_decoder is not None:
                            self.fusion_decoder.load_state_dict(best_decoder_state)
                        if is_flat and (best_spec_state is not None) and (self.spec_encoder is not None):
                            self.spec_encoder.load_state_dict(best_spec_state)
                    except Exception:
                        pass
            # Persist per-metric LP-FT best states for downstream eval path selection.
            # Keep previously cached snapshots when current run has no Val-ES (e.g., retrain-on-full pass).
            if val_loader is not None:
                self._lpft_best_metric_states = {}
                self._lpft_best_metric_values = {}
                self._lpft_selected_metric = str(_es_metric)
                for _m in ("ce", "spec_ce"):
                    _pack = metric_best_state.get(_m, None)
                    _mval = metric_best.get(_m, None)
                    if _pack is None or _mval is None:
                        continue
                    self._lpft_best_metric_states[_m] = {
                        "decoder": _pack.get("decoder", None),
                        "spec": _pack.get("spec", None),
                        "step": _pack.get("step", None),
                    }
                    self._lpft_best_metric_values[_m] = float(_mval)
                if _lp_ema_enable and (ema_decoder_state is not None):
                    self._lpft_ema_state = {
                        "decoder": ema_decoder_state,
                        "spec": ema_spec_state if is_flat else None,
                        "step": ema_step,
                    }
                    self._lpft_ema_metric = float(ema_metric_val) if (ema_metric_val is not None) else None
                    self._lpft_ema_step = ema_step
                else:
                    self._lpft_ema_state = None
                    self._lpft_ema_metric = None
                    self._lpft_ema_step = None

            # Cache retrain-on-full final state for optional non-parametric evaluation.
            if is_flat and _is_retrain_full_call and bool(force_no_val_es):
                try:
                    self._lpft_retrain_full_state = {
                        "decoder": (
                            {k: t.clone().detach() for k, t in self.fusion_decoder.state_dict().items()}
                            if self.fusion_decoder is not None
                            else None
                        ),
                        "spec": (
                            {k: t.clone().detach() for k, t in self.spec_encoder.state_dict().items()}
                            if self.spec_encoder is not None
                            else None
                        ),
                        "step": int(num_steps),
                    }
                    self._lpft_retrain_full_step = int(num_steps)
                except Exception:
                    self._lpft_retrain_full_state = None
                    self._lpft_retrain_full_step = None

            # ATT224(S4): for tiny-shot LP-FT, retrain on full labeled split using best_step budget.
            if (
                bool(allow_full_retrain)
                and is_flat
                and (_use_val_es and (val_loader is not None))
            ):
                try:
                    import config as cfg
                    lp_retrain_enable = bool(getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_ENABLE", False))
                    lp_retrain_max_samples = int(getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_MAX_SAMPLES", 160))
                    lp_retrain_min_steps = int(getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_MIN_STEPS", 80))
                    lp_retrain_enforce_stage4_coverage = bool(
                        getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_ENFORCE_STAGE4_COVERAGE", False)
                    )
                    lp_retrain_min_stage4_ft_steps = int(
                        getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_MIN_STAGE4_FT_STEPS", 20)
                    )
                    lp_retrain_adaptive_stage4_floor_enable = bool(
                        getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_ENABLE", False)
                    )
                    lp_retrain_adaptive_stage4_floor_ratio = float(
                        getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_RATIO", 0.30)
                    )
                    lp_retrain_adaptive_stage4_floor_min = int(
                        getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_ADAPTIVE_STAGE4_FLOOR_MIN", 20)
                    )
                    lp_retrain_use_best_step = bool(
                        getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_USE_BEST_STEP", True)
                    )
                except Exception:
                    lp_retrain_enable = False
                    lp_retrain_max_samples = 160
                    lp_retrain_min_steps = 80
                    lp_retrain_enforce_stage4_coverage = False
                    lp_retrain_min_stage4_ft_steps = 20
                    lp_retrain_adaptive_stage4_floor_enable = False
                    lp_retrain_adaptive_stage4_floor_ratio = 0.30
                    lp_retrain_adaptive_stage4_floor_min = 20
                    lp_retrain_use_best_step = True
                _n_ft = 0
                try:
                    _n_ft = int(len(finetune_loader.dataset))
                except Exception:
                    _n_ft = 0
                if lp_retrain_enable and (_n_ft > 0) and (_n_ft <= max(1, int(lp_retrain_max_samples))):
                    _best_pack = self._lpft_best_metric_states.get(str(_es_metric).strip().lower(), None)
                    _best_step = None
                    if _best_pack is not None:
                        _best_step = _best_pack.get("step", None)
                    if _best_step is None:
                        _best_step = best_step
                    _retrain_steps = int(num_steps)
                    if lp_retrain_use_best_step and (_best_step is not None):
                        _retrain_steps = int(_best_step) + 1
                    _retrain_steps = max(int(lp_retrain_min_steps), min(int(num_steps), int(_retrain_steps)))
                    if (
                        bool(lp_retrain_enforce_stage4_coverage)
                        and bool(lp_prog_enable)
                        and bool(lp_backbone_ft_enable)
                    ):
                        # P1: floor budget must use retrain-full's own progressive schedule,
                        # not the caller path's already-scaled stage4 start.
                        _retrain_s2 = int(lp_unfreeze_stage2_start)
                        _retrain_s3 = int(lp_unfreeze_stage3_start)
                        _retrain_s4 = int(lp_unfreeze_stage4_start)
                        try:
                            _s2_base = int(getattr(cfg, "TARGET_SUP_LP_UNFREEZE_STAGE2_START", _retrain_s2))
                            _s3_base = int(getattr(cfg, "TARGET_SUP_LP_UNFREEZE_STAGE3_START", _retrain_s3))
                            _s4_base = int(getattr(cfg, "TARGET_SUP_LP_UNFREEZE_STAGE4_START", _retrain_s4))
                        except Exception:
                            _s2_base, _s3_base, _s4_base = _retrain_s2, _retrain_s3, _retrain_s4
                        if bool(lp_stage_scale_by_samples):
                            _th = max(1, int(lp_stage_scale_threshold))
                            if int(_n_ft) < _th:
                                _ratio = float(_n_ft) / float(_th)
                                _ratio = max(float(lp_stage_scale_min_ratio), min(1.0, _ratio))
                                _retrain_s2 = max(
                                    int(lp_stage_scale_min_stage2_steps),
                                    max(1, int(round(float(_s2_base) * _ratio))),
                                )
                                _retrain_s3 = max(_retrain_s2 + 1, int(round(float(_s3_base) * _ratio)))
                                _retrain_s4 = max(_retrain_s3 + 1, int(round(float(_s4_base) * _ratio)))
                            else:
                                _retrain_s2, _retrain_s3, _retrain_s4 = _s2_base, _s3_base, _s4_base
                        else:
                            _retrain_s2, _retrain_s3, _retrain_s4 = _s2_base, _s3_base, _s4_base

                        _stage4_ft_min_steps = max(0, int(lp_retrain_min_stage4_ft_steps))
                        if (
                            bool(lp_retrain_adaptive_stage4_floor_enable)
                            and (_best_step is not None)
                        ):
                            _ratio = max(0.0, float(lp_retrain_adaptive_stage4_floor_ratio))
                            _adaptive_floor = int(round(float(int(_best_step) + 1) * _ratio))
                            _adaptive_floor = max(int(lp_retrain_adaptive_stage4_floor_min), _adaptive_floor)
                            _stage4_ft_min_steps = min(_stage4_ft_min_steps, _adaptive_floor)

                        _stage4_floor = int(_retrain_s4) + int(_stage4_ft_min_steps)
                        if _retrain_steps < _stage4_floor:
                            _retrain_steps = min(int(num_steps), int(_stage4_floor))
                        print(
                            f"[Supervised Finetune] Retrain-on-full stage4 floor applied{dataset_str}: "
                            f"stage4_start={int(_retrain_s4)}, "
                            f"min_stage4_ft_steps={int(_stage4_ft_min_steps)}, "
                            f"adaptive_floor={int(bool(lp_retrain_adaptive_stage4_floor_enable))}, "
                            f"floor_steps={int(_stage4_floor)}"
                        )
                    if _retrain_steps > 0:
                        print(
                            f"[Supervised Finetune] Retrain-on-full enabled{dataset_str}: "
                            f"n_train={_n_ft}, steps={_retrain_steps}, best_step={_best_step}"
                        )
                        try:
                            if current_spec_state is not None and self.spec_encoder is not None:
                                self.spec_encoder.load_state_dict(current_spec_state)
                            if current_decoder_state is not None and self.fusion_decoder is not None:
                                self.fusion_decoder.load_state_dict(current_decoder_state)
                        except Exception:
                            pass
                        try:
                            self.supervised_finetune_on_loader(
                                finetune_loader,
                                dataset_name=(f"{dataset_name}/fullretrain" if dataset_name is not None else "fullretrain"),
                                num_classes_eval=num_classes_eval,
                                num_steps=int(_retrain_steps),
                                lr=lr,
                                nf_weight=nf_weight,
                                diag_weight=diag_weight,
                                unlabeled_loader=unlabeled_loader,
                                allow_full_retrain=False,
                                force_no_val_es=True,
                            )
                        except Exception as _retrain_e:
                            try:
                                import traceback
                                traceback.print_exc()
                            except Exception:
                                pass
                            print(f"[Supervised Finetune] Retrain-on-full failed{dataset_str}: {_retrain_e}")

            # Restore temperature if overridden
            if (_lp_temp_orig is not None) and hasattr(self.fusion_decoder, "temperature"):
                try:
                    setattr(self.fusion_decoder, "temperature", float(_lp_temp_orig))
                    print(f"[Supervised Finetune] Restored flat cosine temperature to {_lp_temp_orig:.6f}{dataset_str}")
                except Exception:
                    pass
            
            # Set back to eval mode
            self.fusion_module.eval()
            self.fusion_decoder.eval()
            
            print(f"[Supervised Finetune] Completed {num_steps} steps{dataset_str}")
            # Debug: report whether the D2 correction head has moved off zero
            try:
                if hasattr(self.fusion_decoder, "diag_linear_scale"):
                    # Print effective beta in [0,beta_max] to match forward behavior.
                    try:
                        import config as cfg
                        beta_max = float(getattr(cfg, "DIAG_LINEAR_BETA_MAX", 0.5))
                    except Exception:
                        beta_max = 0.5
                    beta_max = max(0.0, beta_max)
                    beta_raw = self.fusion_decoder.diag_linear_scale.detach().cpu()
                    beta_eff = float(beta_max * torch.sigmoid(beta_raw).item())
                    print(f"[Supervised Finetune] diag_linear_scale(beta)={beta_eff:.6f}{dataset_str}")
            except Exception:
                pass
            return True
            
        except Exception as e:
            print(f"[Supervised Finetune] Failed: {e}, restoring previous state{dataset_str}")
            # Restore previous state
            self.fusion_module.load_state_dict(current_fusion_state)
            if current_spec_state is not None and self.spec_encoder is not None:
                self.spec_encoder.load_state_dict(current_spec_state)
            if current_decoder_state is not None:
                self.fusion_decoder.load_state_dict(current_decoder_state)
            return False
        finally:
            # (Attempt 51) Restore overridden ramp and temperature
            if _lp_ramp_orig is not None and hasattr(self.fusion_decoder, "set_diag_linear_ramp"):
                try:
                    self.fusion_decoder.set_diag_linear_ramp(float(_lp_ramp_orig))
                except Exception:
                    pass
            if _lp_temp_orig is not None and hasattr(self.fusion_decoder, "temperature"):
                try:
                    setattr(self.fusion_decoder, "temperature", float(_lp_temp_orig))
                except Exception:
                    pass
            
            self._set_nf_calibrator_grad(False)
            # Restore prototype residual trainability
            if self.fusion_decoder is not None:
                try:
                    if hasattr(self.fusion_decoder, "set_flat_classifier_requires_grad"):
                        self.fusion_decoder.set_flat_classifier_requires_grad(False)
                except Exception:
                    pass
                if hasattr(self.fusion_decoder, 'set_prototype_residual_domain_requires_grad'):
                    self.fusion_decoder.set_prototype_residual_domain_requires_grad(False)
                if hasattr(self.fusion_decoder, 'set_prototype_residual_shared_requires_grad'):
                    try:
                        import config as cfg
                        self.fusion_decoder.set_prototype_residual_shared_requires_grad(bool(getattr(cfg, 'USE_PROMPT_TUNING', True)))
                    except Exception:
                        self.fusion_decoder.set_prototype_residual_shared_requires_grad(True)
            self._freeze_all_except_calibrator_and_lora(freeze=False)

    # ======================
    # Extract embeddings/logits (returns CPU tensors)
    # modes:
    #   - "spec": spec encoder outputs [B,D]
    #   - "signal": signal encoder outputs [B,D]
    #   - "fusion": fusion_module(z_signal, z_spec) -> z_fusion [B,D]
    #   - "fusion_decoder": fusion_decoder(z_fusion) -> logits [B,C]
    # ======================
    def extract_embeddings(self, dataloader, mode="spec"):
        assert mode in ["spec", "signal", "fusion", "fusion_decoder"]

        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.eval()
        if self.fusion_decoder is not None:
            self.fusion_decoder.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                if mode == "spec":
                    x = self.prepare_spec_batch(batch)
                    mask = batch["spec_mask"].to(self.device)
                    emb = self.spec_encoder(x, mask)    # [B, D]
                elif mode == "signal":
                    x = batch["signal"].to(self.device)
                    mask = batch["signal_mask"].to(self.device)
                    emb = self.signal_encoder(x, mask)  # [B, D]
                else:
                    # need both modalities
                    signal = batch["signal"].to(self.device)
                    spec = self.prepare_spec_batch(batch)
                    signal_mask = batch["signal_mask"].to(self.device)
                    spec_mask = batch["spec_mask"].to(self.device)

                    z_signal = self.signal_encoder(signal, signal_mask)  # [B, D]
                    z_spec = self.spec_encoder(spec, spec_mask)          # [B, D]

                    z_fusion = self._fuse_embeddings(z_signal, z_spec, signal, signal_mask, spec, spec_mask)  # [B, D]

                    if mode == "fusion":
                        emb = z_fusion
                    else:  # fusion_decoder
                        if self.fusion_decoder is None:
                            raise RuntimeError("fusion_decoder is None but mode 'fusion_decoder' requested.")
                        # Get logits (hierarchical decoder still returns 4-class logits)
                        text_prototypes = self._active_text_prototypes()
                        if hasattr(self.fusion_decoder, 'hierarchical') and self.fusion_decoder.hierarchical:
                            logits, _ = self.fusion_decoder(
                                z_fusion,
                                text_prototypes=text_prototypes,
                                return_parts=True
                            )
                        else:
                            logits = self.fusion_decoder(z_fusion, text_prototypes=text_prototypes)  # [B, C]
                        emb = logits  # logits used for decoder-based scoring

                embeddings.append(emb.cpu())
                labels.append(batch["label"].cpu())

        if len(embeddings) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels

    # ======================
    # Extract multiple embeddings in one pass (optimized)
    # Returns dict with keys: "signal", "spec", "fusion", "fusion_decoder", "labels"
    # Only computes what's needed based on requested modes
    # ======================
    def extract_multimodal_embeddings(self, dataloader, modes=None):
        """
        Extract multiple types of embeddings in a single pass to avoid redundant computation.
        
        Args:
            dataloader: DataLoader
            modes: List of modes to extract, e.g., ["fusion", "fusion_decoder"]
                   If None, extracts all available modes
        
        Returns:
            dict: {
                "signal": tensor [N, D] or None,
                "spec": tensor [N, D] or None,
                "fusion": tensor [N, D] or None,
                "fusion_decoder": tensor [N, C] or None,
                "labels": tensor [N]
            }
        """
        if modes is None:
            modes = ["signal", "spec", "fusion", "fusion_decoder"]
        
        # Determine what needs to be computed
        need_signal = "signal" in modes
        need_spec = "spec" in modes
        need_fusion = "fusion" in modes or "fusion_decoder" in modes  # fusion_decoder needs fusion
        need_decoder = "fusion_decoder" in modes
        save_fusion = "fusion" in modes  # Whether to save fusion embeddings (vs just compute for decoder)
        
        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.eval()
        if self.fusion_decoder is not None and need_decoder:
            self.fusion_decoder.eval()
        
        # Initialize storage (only allocate lists for modes that are explicitly requested)
        results = {
            "signal": [] if need_signal else None,
            "spec": [] if need_spec else None,
            "fusion": [] if save_fusion else None,  # Only save if explicitly requested
            "fusion_decoder": [] if need_decoder else None,
            "labels": []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(self.device)
                spec = self.prepare_spec_batch(batch)
                signal_mask = batch["signal_mask"].to(self.device)
                spec_mask = batch["spec_mask"].to(self.device)
                labels = batch["label"].cpu()
                
                # Compute signal embedding if needed
                if need_signal:
                    z_signal = self.signal_encoder(signal, signal_mask)  # [B, D]
                    # ATT396: apply decoupler for condition-invariant features
                    if self.decouple_signal is not None:
                        z_signal, _ = self.decouple_signal(z_signal)
                    results["signal"].append(z_signal.cpu())
                
                # Compute spec embedding if needed
                if need_spec:
                    z_spec = self.spec_encoder(spec, spec_mask)  # [B, D]
                    if self.decouple_spec is not None:
                        z_spec, _ = self.decouple_spec(z_spec)
                    results["spec"].append(z_spec.cpu())
                
                # Compute fusion if needed
                if need_fusion:
                    if not need_signal:
                        z_signal = self.signal_encoder(signal, signal_mask)  # [B, D]
                        if self.decouple_signal is not None:
                            z_signal, _ = self.decouple_signal(z_signal)
                    if not need_spec:
                        z_spec = self.spec_encoder(spec, spec_mask)  # [B, D]
                        if self.decouple_spec is not None:
                            z_spec, _ = self.decouple_spec(z_spec)
                    
                    z_fusion = self._fuse_embeddings(z_signal, z_spec, signal, signal_mask, spec, spec_mask)  # [B, D]
                    
                    # Save fusion embeddings if requested
                    if save_fusion:
                        results["fusion"].append(z_fusion.cpu())
                    
                    # Compute decoder logits if needed
                    if need_decoder:
                        if self.fusion_decoder is None:
                            raise RuntimeError("fusion_decoder is None but 'fusion_decoder' mode requested.")
                        # Get logits (hierarchical decoder still returns 4-class logits)
                        prototypes = self._active_text_prototypes()
                        if hasattr(self.fusion_decoder, 'hierarchical') and self.fusion_decoder.hierarchical:
                            logits, _ = self.fusion_decoder(
                                z_fusion,
                                text_prototypes=prototypes,
                                return_parts=True
                            )
                        else:
                            logits = self.fusion_decoder(z_fusion, text_prototypes=prototypes)  # [B, C]
                        results["fusion_decoder"].append(logits.cpu())
                
                results["labels"].append(labels)
        
        # Concatenate all results
        if len(results["labels"]) == 0:
            return {k: (torch.empty(0) if k != "labels" else torch.empty(0, dtype=torch.long)) 
                    for k in results}
        
        results["labels"] = torch.cat(results["labels"], dim=0)
        for key in ["signal", "spec", "fusion", "fusion_decoder"]:
            if results[key] is not None:
                # Only concatenate if the list is not empty
                if len(results[key]) > 0:
                    results[key] = torch.cat(results[key], dim=0)
                else:
                    # If empty, set to None (shouldn't happen, but handle gracefully)
                    results[key] = None
        
        return results

    # ======================
    # Legacy scoring: 1 - cos(sim to normal)
    # ======================
    @staticmethod
    def compute_text_anomaly_score_legacy(embeddings, normal_proto):
        """
        embeddings: [N, D], normal_proto: [D]
        returns: scores [N] (higher => more anomalous)
        """
        if embeddings.numel() == 0:
            return torch.tensor([], device=embeddings.device)

        embeddings = F.normalize(embeddings, dim=-1)
        normal_proto = normal_proto.to(embeddings.device)
        normal_proto = F.normalize(normal_proto, dim=-1)

        sims = (embeddings @ normal_proto.unsqueeze(1)).squeeze(1)  # [N]
        scores = 1.0 - sims
        return scores

    # ======================
    # Semantic scoring: max(sim_to_faults) - sim_to_normal
    # ======================
    @staticmethod
    def compute_text_anomaly_score(embeddings, normal_proto, fault_protos):
        if embeddings.numel() == 0:
            return torch.tensor([], device=embeddings.device)

        embeddings = F.normalize(embeddings, dim=-1)
        normal_proto = normal_proto.to(embeddings.device)
        fault_protos = fault_protos.to(embeddings.device)

        normal_proto = F.normalize(normal_proto, dim=-1)
        fault_protos = F.normalize(fault_protos, dim=-1)

        sim_normal = embeddings @ normal_proto.unsqueeze(1)   # [N,1]
        sim_fault = embeddings @ fault_protos.T               # [N,K]
        sim_fault_max, _ = torch.max(sim_fault, dim=1, keepdim=True)  # [N,1]

        scores = (sim_fault_max - sim_normal).squeeze(1)  # [N]
        return scores

    # ======================
    # Decoder-based scoring: 1 - p(normal_class_idx)
    # assume normal class index is 0
    # ======================
    @staticmethod
    def compute_decoder_anomaly_score(logits, normal_idx=0):
        if logits.numel() == 0:
            return torch.tensor([], device=logits.device)

        probs = F.softmax(logits, dim=1)
        p_normal = probs[:, normal_idx]
        scores = 1.0 - p_normal
        return scores
    
    # ======================
    # NF-based anomaly scoring (hierarchical decoder only)
    # Uses the Normal vs Fault (nf) head for AUC calculation
    # ======================
    def compute_nf_anomaly_score(self, z_fusion, score_type="logit_diff", evidence_temperature=1.0):
        """
        计算基于层次化nf头的异常分数
        
        Args:
            z_fusion: [N, D] fusion embeddings
            score_type: "logit_diff" or "prob" or "evidence_max"
            evidence_temperature: temperature for evidence aggregation (used in evidence_max mode)
        
        Returns:
            scores: [N] anomaly scores (higher = more anomalous)
        """
        if self.fusion_decoder is None:
            raise RuntimeError("fusion_decoder is None")
        
        if not hasattr(self.fusion_decoder, 'hierarchical'):
            raise RuntimeError("fusion_decoder missing 'hierarchical' flag")
        
        if z_fusion.numel() == 0:
            return torch.tensor([], device=z_fusion.device)
        
        # Get anomaly score from nf head
        scores = self.fusion_decoder.get_anomaly_score(z_fusion, score_type=score_type, evidence_temperature=evidence_temperature)
        return scores

    @staticmethod
    def build_binary_labels(labels):
        return (labels != 0).long()
    
    # ======================
    # Target-domain accuracy evaluation
    # ======================
    def evaluate_target_domain_accuracy(self, dataloader, dataset_name=None, diag_num_classes=4):
        """
        评估目标域的准确率（NF accuracy + Diagnosis accuracy）
        
        Args:
            dataloader: DataLoader for evaluation
            dataset_name: str, 数据集名称（用于日志）
            diag_num_classes: int, 诊断类别数（HUST=4, HIT=3）
        
        Returns:
            dict: {
                'nf_acc': float, NF accuracy (Normal vs Fault)
                'diag_acc': float, Diagnosis accuracy (multi-class)
                'nf_correct': int, NF correct count
                'nf_total': int, NF total count
                'diag_correct': int, Diagnosis correct count
                'diag_total': int, Diagnosis total count
            }
        """
        if self.fusion_decoder is None:
            raise RuntimeError("fusion_decoder is None, cannot evaluate accuracy")
        
        if not hasattr(self.fusion_decoder, 'hierarchical'):
            raise RuntimeError("fusion_decoder missing 'hierarchical' flag")
        
        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.eval()
        self.fusion_decoder.eval()
        
        nf_correct = 0
        nf_total = 0
        diag_correct = 0
        diag_total = 0
        diag_correct_raw = 0
        
        all_nf_preds = []
        all_nf_labels = []
        all_diag_preds = []
        all_diag_labels = []
        all_diag_preds_raw = []
        
        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(self.device)
                signal_mask = batch["signal_mask"].to(self.device)
                spec = self.prepare_spec_batch(batch)
                spec_mask = batch["spec_mask"].to(self.device)
                labels = batch["label"].to(self.device)  # [B]
                domains = batch.get("domain", None)  # List[str] or None
                
                # Encode domain_ids (used by: domain-aware fault head, LoRA, fusion_module domain embedding,
                # nf_calibrator, and prototype/prompt masking)
                domain_ids = None
                if domains is not None:
                    domain_map = _build_domain_map()
                    domain_ids = torch.tensor([domain_map.get(d, 0) for d in domains], dtype=torch.long, device=self.device)
                
                # Forward (IMPORTANT: pass domain_ids so Domain-LoRA can be applied during evaluation)
                if domain_ids is not None:
                    try:
                        z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_sig = self.signal_encoder(signal, signal_mask)
                    try:
                        z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_spec = self.spec_encoder(spec, spec_mask)
                else:
                    z_sig = self.signal_encoder(signal, signal_mask)
                    z_spec = self.spec_encoder(spec, spec_mask)
                # ATT379e(S3): removed F.normalize — training does NOT normalize before decoder,
                # normalization here caused train-eval mismatch (Diag-Acc 0.4570 vs FusionRaw 0.9516)

                # Fusion (pass domain_ids + text_prototypes when supported)
                text_prototypes = self._active_text_prototypes()
                if hasattr(self.fusion_module, "domain_embedding"):
                    z_fusion = self.fusion_module(
                        z_sig, z_spec,
                        domain_ids=domain_ids,
                        text_prototypes=text_prototypes
                    )
                else:
                    z_fusion = self.fusion_module(z_sig, z_spec)
                
                # Get decoder output with parts (pass domain_ids for: nf_calibrator + prototype/prompt masking)
                logits4, parts = self.fusion_decoder(
                    z_fusion,
                    domain_ids=domain_ids,
                    text_prototypes=text_prototypes,
                    return_parts=True
                )
                
                # Unified decision: nf_gate / argmax / nf_protect
                logits_nf_global = parts.get("logits_nf_global", None)
                if logits_nf_global is None:
                    logit_normal = logits_decision[:, :1]
                    logit_fault = torch.logsumexp(logits_decision[:, 1:], dim=1, keepdim=True)
                    logits_nf_global = torch.cat([logit_normal, logit_fault], dim=1)
                nf_labels = (labels != 0).long()  # [B] 0=normal, 1=fault

                if diag_num_classes == 3:
                    diag_labels = torch.clamp(labels.clone(), 0, 2)
                else:
                    diag_labels = labels

                # Load decision config
                try:
                    import config as cfg
                    decision_mode = str(getattr(cfg, "DIAG_DECISION_MODE", "nf_protect")).lower()
                    nf_mode = str(getattr(cfg, "NF_GATE_NF_MODE", "threshold")).lower()
                    gray_low = float(getattr(cfg, "NF_GATE_GRAY_LOW", 0.35))
                    gray_high = float(getattr(cfg, "NF_GATE_GRAY_HIGH", 0.65))
                    gray_fallback = str(getattr(cfg, "NF_GATE_GRAY_FALLBACK", "raw")).lower()
                    use_nf_protected = bool(getattr(cfg, "USE_NF_PROTECTED_ARGMAX", True))
                    pfault_max = float(getattr(cfg, "NF_PROTECT_PFAULT_MAX", 0.01))
                    protect_mode = str(getattr(cfg, "NF_PROTECT_MODE", "soft_bias"))
                    protect_bias = float(getattr(cfg, "NF_PROTECT_BIAS", 4.0))
                    protect_diag_margin = float(getattr(cfg, "NF_PROTECT_DIAG_MARGIN", 0.5))
                except Exception:
                    decision_mode = "nf_protect"
                    nf_mode = "threshold"
                    gray_low, gray_high, gray_fallback = 0.35, 0.65, "raw"
                    use_nf_protected = True
                    pfault_max = 0.01
                    protect_mode = "soft_bias"
                    protect_bias = 4.0
                    protect_diag_margin = 0.5

                # Eval-time threshold + Platt (if set by evaluation_utils)
                nf_threshold = float(self._eval_nf_threshold) if hasattr(self, "_eval_nf_threshold") and self._eval_nf_threshold is not None else None
                platt = None
                if hasattr(self, "_eval_nf_platt") and self._eval_nf_platt is not None:
                    try:
                        platt = (float(self._eval_nf_platt[0]), float(self._eval_nf_platt[1]))
                    except Exception:
                        platt = None

                from utils.diag_decision import decide_diagnosis_from_logits

                # Prefer supervised diagnosis logits for decisions (structural split-head / two-stage decoupling).
                # Keep semantic prototypes for semantic scoring, but do not let biased prototype logits dominate diagnosis.
                logits_decision = None
                try:
                    logits_decision = parts.get("logits_diag_supervised", None)
                except Exception:
                    logits_decision = None
                if logits_decision is None:
                    logits_decision = logits4

                out = decide_diagnosis_from_logits(
                    logits_diag=logits_decision,
                    logits_nf_global=logits_nf_global,
                    diag_num_classes=int(diag_num_classes),
                    decision_mode=decision_mode,
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
                )

                diag_preds = out["diag_preds"]
                diag_preds_raw = out["diag_preds_raw"]
                nf_preds = out["nf_preds"]
                nf_gate_gray_frac = float(out.get("nf_gate_gray_frac", 0.0))
                open_set_unknown_frac = float(out.get("open_set_unknown_frac", 0.0))
                
                # Accumulate
                nf_correct += (nf_preds == nf_labels).sum().item()
                nf_total += nf_labels.size(0)
                # Open-set: unknown label (-1) is treated as incorrect for standard accuracy.
                diag_correct += (diag_preds == diag_labels).sum().item()
                diag_correct_raw += (diag_preds_raw == diag_labels).sum().item()
                diag_total += diag_labels.size(0)
                
                # Store for confusion matrix (optional)
                all_nf_preds.append(nf_preds.cpu())
                all_nf_labels.append(nf_labels.cpu())
                # Avoid breaking downstream confusion logic: clamp unknown (-1) to 0 in stored preds,
                # while still reporting unknown_frac separately.
                try:
                    dp_store = diag_preds.clone()
                    dp_store[dp_store < 0] = 0
                except Exception:
                    dp_store = diag_preds
                all_diag_preds.append(dp_store.cpu())
                all_diag_labels.append(diag_labels.cpu())
                all_diag_preds_raw.append(diag_preds_raw.cpu())
        
        nf_acc = nf_correct / max(nf_total, 1)
        diag_acc = diag_correct / max(diag_total, 1)
        diag_acc_raw = diag_correct_raw / max(diag_total, 1)
        
        result = {
            'nf_acc': nf_acc,
            'diag_acc': diag_acc,
            'diag_acc_raw': diag_acc_raw,
            'nf_gate_gray_frac': nf_gate_gray_frac if 'nf_gate_gray_frac' in locals() else 0.0,
            'open_set_unknown_frac': open_set_unknown_frac if 'open_set_unknown_frac' in locals() else 0.0,
            'nf_correct': nf_correct,
            'nf_total': nf_total,
            'diag_correct': diag_correct,
            'diag_total': diag_total,
            'nf_preds': torch.cat(all_nf_preds, dim=0).numpy() if all_nf_preds else None,
            'nf_labels': torch.cat(all_nf_labels, dim=0).numpy() if all_nf_labels else None,
            'diag_preds': torch.cat(all_diag_preds, dim=0).numpy() if all_diag_preds else None,
            'diag_preds_raw': torch.cat(all_diag_preds_raw, dim=0).numpy() if all_diag_preds_raw else None,
            'diag_labels': torch.cat(all_diag_labels, dim=0).numpy() if all_diag_labels else None,
        }
        
        return result

    # ======================
    # NF threshold selection (for labeled target-domain split)
    # ======================
    def set_eval_nf_threshold(self, threshold=None):
        """Set a temporary NF threshold used by evaluate_target_domain_accuracy()."""
        self._eval_nf_threshold = None if threshold is None else float(threshold)

    # ======================
    # (A1-1) NF Platt calibration (scale + bias on anomaly_logit)
    # ======================
    def set_eval_nf_platt(self, scale=None, bias=None):
        """
        Set eval-time Platt calibration parameters:
            anomaly_logit_cal = scale * anomaly_logit + bias
        If scale/bias is None, disable Platt calibration.
        """
        if scale is None or bias is None:
            self._eval_nf_platt = None
        else:
            self._eval_nf_platt = (float(scale), float(bias))

    def fit_eval_nf_platt(self, dataloader, dataset_name=None, max_samples=512, steps=200, lr=0.1, l2=1e-3):
        """
        Fit Platt scaling on anomaly_logit using labeled data (tiny split).
        Returns: (scale, bias, final_loss)
        """
        if self.fusion_decoder is None or not hasattr(self.fusion_decoder, "hierarchical") or not self.fusion_decoder.hierarchical:
            # If decoder is not hierarchical, NF logits are not defined; Platt calibration is inapplicable.
            ds = f" for {dataset_name}" if dataset_name else ""
            print(f"[NF] Platt calibration skipped{ds}: fusion_decoder is not hierarchical (NF head unavailable).")
            self.set_eval_nf_platt(None, None)
            return 1.0, 0.0, 0.0

        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.eval()
        self.fusion_decoder.eval()

        xs = []
        ys = []
        seen = 0
        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(self.device)
                signal_mask = batch["signal_mask"].to(self.device)
                spec = self.prepare_spec_batch(batch)
                spec_mask = batch["spec_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                domains = batch.get("domain", None)

                domain_ids = None
                if domains is not None:
                    domain_map = _build_domain_map()
                    domain_ids = torch.tensor([domain_map.get(d, 0) for d in domains], dtype=torch.long, device=self.device)

                # encoder forward (LoRA-aware)
                if domain_ids is not None:
                    try:
                        z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_sig = self.signal_encoder(signal, signal_mask)
                    try:
                        z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_spec = self.spec_encoder(spec, spec_mask)
                else:
                    z_sig = self.signal_encoder(signal, signal_mask)
                    z_spec = self.spec_encoder(spec, spec_mask)
                # ATT379e: removed F.normalize for train-eval consistency

                text_prototypes = self._active_text_prototypes()
                if hasattr(self.fusion_module, "domain_embedding"):
                    z_fusion = self.fusion_module(z_sig, z_spec, domain_ids=domain_ids, text_prototypes=text_prototypes)
                else:
                    z_fusion = self.fusion_module(z_sig, z_spec)

                _, parts = self.fusion_decoder(z_fusion, domain_ids=domain_ids, text_prototypes=text_prototypes, return_parts=True)
                logits_nf_global = parts.get("logits_nf_global", None)
                if logits_nf_global is None:
                    continue
                x = (logits_nf_global[:, 1] - logits_nf_global[:, 0]).detach().float().cpu()
                y = (labels != 0).float().detach().cpu()
                xs.append(x)
                ys.append(y)
                seen += int(labels.size(0))
                if seen >= int(max_samples):
                    break

        if not xs:
            ds = f" for {dataset_name}" if dataset_name else ""
            print(f"[NF] Platt calibration skipped{ds}: no usable anomaly logits collected (missing logits_nf_global).")
            self.set_eval_nf_platt(None, None)
            return 1.0, 0.0, 0.0

        x = torch.cat(xs, dim=0).float()
        y = torch.cat(ys, dim=0).float()
        # Filter non-finite values (can happen under extreme domain shift / numerical issues).
        finite = torch.isfinite(x) & torch.isfinite(y)
        if int(finite.sum().item()) != int(x.numel()):
            x = x[finite]
            y = y[finite]
        if x.numel() < 4:
            ds = f" for {dataset_name}" if dataset_name else ""
            print(f"[NF] Platt calibration skipped{ds}: too few finite samples (n={int(x.numel())}).")
            self.set_eval_nf_platt(None, None)
            return 1.0, 0.0, 0.0

        # If the labeled split contains only one class (all-normal or all-fault), Platt is ill-posed.
        uniq = torch.unique(y)
        if int(uniq.numel()) < 2:
            ds = f" for {dataset_name}" if dataset_name else ""
            c0 = int((y == 0).sum().item())
            c1 = int((y == 1).sum().item())
            print(f"[NF] Platt calibration skipped{ds}: single-class labels (normal={c0}, fault={c1}).")
            self.set_eval_nf_platt(None, None)
            return 1.0, 0.0, 0.0

        # Clamp logits for numerical stability (match training-time clamp policy).
        try:
            import config as cfg
            clamp_val = float(getattr(cfg, "NF_LOGIT_CLAMP", 0.0))
        except Exception:
            clamp_val = 0.0
        if clamp_val and clamp_val > 0:
            x = torch.clamp(x, min=-clamp_val, max=clamp_val)

        # Optimize scale,bias in float32 on CPU for numerical stability.
        s = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([s, b], lr=float(lr), max_iter=int(steps), line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = s * x + b
            loss = F.binary_cross_entropy_with_logits(logits, y)
            # tiny L2 keeps scale bounded (prevents extreme thresholds under tiny splits)
            loss = loss + float(l2) * (s * s + b * b)
            loss.backward()
            return loss

        loss_tensor = opt.step(closure)
        try:
            loss_final = float(loss_tensor.detach().cpu().item())
        except Exception:
            loss_final = float("nan")
        scale = float(s.detach().cpu().item())
        bias = float(b.detach().cpu().item())
        if (not (loss_final == loss_final)) or (not torch.isfinite(torch.tensor([scale, bias, loss_final])).all().item()):
            ds = f" for {dataset_name}" if dataset_name else ""
            print(f"[NF] Platt calibration warning{ds}: non-finite result (scale={scale}, bias={bias}, loss={loss_final}). Disabled.")
            self.set_eval_nf_platt(None, None)
            return 1.0, 0.0, 0.0

        self.set_eval_nf_platt(scale, bias)
        return scale, bias, loss_final

    def find_best_nf_threshold(self, dataloader, dataset_name=None, num_candidates=51, objective="balanced_acc"):
        """
        Find the best NF threshold t on anomaly_logit = logit_fault - logit_normal using labeled data.
        This is intended for the 5% labeled target-domain split (allowed by your setting).
        """
        if self.fusion_decoder is None or not hasattr(self.fusion_decoder, "hierarchical") or not self.fusion_decoder.hierarchical:
            return 0.0, 0.0

        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.eval()
        self.fusion_decoder.eval()

        scores = []
        labels_all = []
        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(self.device)
                signal_mask = batch["signal_mask"].to(self.device)
                spec = self.prepare_spec_batch(batch)
                spec_mask = batch["spec_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                domains = batch.get("domain", None)

                domain_ids = None
                if domains is not None:
                    domain_map = _build_domain_map()
                    domain_ids = torch.tensor([domain_map.get(d, 0) for d in domains], dtype=torch.long, device=self.device)

                # encoder forward (LoRA-aware)
                if domain_ids is not None:
                    try:
                        z_sig = self.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_sig = self.signal_encoder(signal, signal_mask)
                    try:
                        z_spec = self.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_spec = self.spec_encoder(spec, spec_mask)
                else:
                    z_sig = self.signal_encoder(signal, signal_mask)
                    z_spec = self.spec_encoder(spec, spec_mask)
                # ATT379e: removed F.normalize for train-eval consistency

                text_prototypes = self._active_text_prototypes()
                if hasattr(self.fusion_module, "domain_embedding"):
                    z_fusion = self.fusion_module(z_sig, z_spec, domain_ids=domain_ids, text_prototypes=text_prototypes)
                else:
                    z_fusion = self.fusion_module(z_sig, z_spec)

                _, parts = self.fusion_decoder(
                    z_fusion, domain_ids=domain_ids, text_prototypes=text_prototypes, return_parts=True
                )
                logits_nf_global = parts.get("logits_nf_global", None)
                if logits_nf_global is None:
                    logits_nf_global = parts.get("logits_nf", None)
                if logits_nf_global is None:
                    continue
                anomaly_logit = logits_nf_global[:, 1] - logits_nf_global[:, 0]
                # (A1-1) Apply eval-time Platt calibration if available
                if hasattr(self, "_eval_nf_platt") and self._eval_nf_platt is not None:
                    try:
                        s, b = self._eval_nf_platt
                        anomaly_logit = (float(s) * anomaly_logit) + float(b)
                    except Exception:
                        pass
                # Clamp for stability
                try:
                    import config as cfg
                    clamp_val = float(getattr(cfg, "NF_LOGIT_CLAMP", 0.0))
                except Exception:
                    clamp_val = 0.0
                if clamp_val and clamp_val > 0:
                    anomaly_logit = torch.clamp(anomaly_logit, min=-clamp_val, max=clamp_val)
                scores.append(anomaly_logit.detach().cpu())
                labels_all.append((labels != 0).long().detach().cpu())

        if not scores:
            return 0.0, 0.0

        scores = torch.cat(scores, dim=0).float()
        labels_all = torch.cat(labels_all, dim=0).long()
        def _acc_metrics(preds, y):
            # y: 0/1
            y = y.long()
            preds = preds.long()
            tp = int(((preds == 1) & (y == 1)).sum().item())
            tn = int(((preds == 0) & (y == 0)).sum().item())
            fp = int(((preds == 1) & (y == 0)).sum().item())
            fn = int(((preds == 0) & (y == 1)).sum().item())
            p = tp + fn
            n = tn + fp
            tpr = tp / max(p, 1)
            tnr = tn / max(n, 1)
            acc = (tp + tn) / max(p + n, 1)
            bal = 0.5 * (tpr + tnr)
            return acc, bal

        if scores.numel() < 2:
            t0 = float(scores.mean().item()) if scores.numel() > 0 else 0.0
            acc0, bal0 = _acc_metrics((scores > t0).long(), labels_all)
            return t0, (bal0 if objective == "balanced_acc" else acc0)

        # candidate thresholds from quantiles + include 0
        qs = torch.linspace(0.0, 1.0, steps=max(2, int(num_candidates)), dtype=torch.float32)
        cand = torch.quantile(scores, qs).unique()
        cand = torch.cat([cand, torch.tensor([0.0], dtype=torch.float32)]).unique()

        best_score = -1.0
        best_t = 0.0
        for t in cand:
            preds = (scores > t).long()
            acc, bal = _acc_metrics(preds, labels_all)
            score = bal if objective == "balanced_acc" else acc
            if (score > best_score) or (score == best_score and abs(float(t.item())) < abs(best_t)):
                best_score = score
                best_t = float(t.item())

        return best_t, float(best_score)
    
    def _diagnose_semantic_scoring(self, z, normal_proto, fault_protos, dataset_name=""):
        """
        诊断semantic scoring问题
        
        Args:
            z: [N, D] fusion embeddings
            normal_proto: [D] normal prototype
            fault_protos: [K, D] fault prototypes
            dataset_name: str, dataset name for logging
        
        Returns:
            scores: [N] computed scores
        """
        # 确保 prototypes 在正确的设备上
        normal_proto = normal_proto.to(z.device)
        fault_protos = fault_protos.to(z.device)
        
        z_norm = F.normalize(z, dim=-1)
        normal_norm = F.normalize(normal_proto, dim=-1)
        fault_norm = F.normalize(fault_protos, dim=-1)
        
        sim_normal = z_norm @ normal_norm.unsqueeze(1)  # [N, 1]
        sim_fault = z_norm @ fault_norm.T  # [N, K]
        sim_fault_max, _ = torch.max(sim_fault, dim=1, keepdim=True)  # [N, 1]
        
        scores = (sim_fault_max - sim_normal).squeeze(1)  # [N]
        
        print(f"\n[Diagnosis {dataset_name}] Semantic Scoring:")
        print(f"  z shape: {z.shape}, z norm: {z_norm.norm(dim=1).mean():.4f}")
        print(f"  normal_proto shape: {normal_proto.shape}, norm: {normal_norm.norm():.4f}")
        print(f"  fault_protos shape: {fault_protos.shape}, norm: {fault_norm.norm(dim=1)}")
        print(f"  sim_normal: min={sim_normal.min():.4f}, max={sim_normal.max():.4f}, mean={sim_normal.mean():.4f}")
        print(f"  sim_fault_max: min={sim_fault_max.min():.4f}, max={sim_fault_max.max():.4f}, mean={sim_fault_max.mean():.4f}")
        print(f"  scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        return scores

    # ======================
    # Evaluate
    # mode param controls which encoder path to use for embeddings/logits:
    #   - "spec" or "signal" -> use respective encoder (legacy scoring by default)
    #   - "fusion" -> use fusion embedding (text-based scoring)
    #   - "fusion_decoder" -> use fusion_decoder logits (decoder scoring)
    #
    # scoring parameter:
    #   - "legacy" (default): use single normal prototype and 1 - cos(sim)
    #   - "semantic": use fault prototypes (max_fault - normal)
    #   - "decoder": use decoder logits scoring
    #   - "both": fuse semantic + decoder (alpha weighting)
    # ======================
    def _filter_prototypes_for_domain(self, prototypes, domain_name):
        if domain_name is None:
            return prototypes
        domain_name = domain_name.upper()
        domain_keys_map = {
            "HIT": [0, 1, 2],
            "HUST": [0, 1, 2, 3]
        }
        desired_keys = domain_keys_map.get(domain_name)
        if desired_keys is None:
            return prototypes
        filtered = {k: prototypes[k] for k in desired_keys if k in prototypes}
        return filtered if len(filtered) > 1 else prototypes

    def evaluate(self, dataloader, mode="spec", scoring="legacy", alpha=0.5, beta=0.0, dataset_name=None, enable_diagnosis=False, evidence_temperature=1.0, prototype_domain=None):
        """
        scoring: "legacy" | "semantic" | "decoder" | "both" | "both_legacy" | "evidence_or" | "evidence_max"
        alpha: weight for semantic (when scoring == "both"), or weight for legacy (when scoring == "both_legacy")
        beta: weight for decoder (when scoring == "both"), gamma = 1 - alpha - beta for fusion (legacy)
        dataset_name: str, name of the dataset (e.g., "HUST"). If provided and enable_finetuning is True,
                      online finetuning will be applied before evaluation.
        enable_diagnosis: bool, whether to enable detailed diagnosis output
        evidence_temperature: float, temperature for evidence aggregation (used in evidence_or/evidence_max modes)
        """
        assert scoring in ["legacy", "semantic", "decoder", "both", "both_legacy", "evidence_or", "evidence_max", "soft_hierarchical"]
        
        # 确保所有模型都在eval模式
        self.signal_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_module.eval()
        if self.fusion_decoder is not None:
            self.fusion_decoder.eval()
        self.text_encoder.eval()
        
        # 保存原始模型状态（如果还没有保存）
        if self._original_fusion_state is None:
            self._original_fusion_state = self.fusion_module.state_dict().copy()
        if self._original_decoder_state is None and self.fusion_decoder is not None:
            self._original_decoder_state = self.fusion_decoder.state_dict().copy()
        
        # Apply online finetuning if enabled and dataset is in target domains list (only once per dataset)
        finetuning_applied = False
        if self.enable_finetuning and dataset_name is not None and dataset_name not in self._finetuned_datasets:
            # 检查是否在目标域列表中（从config读取）
            import config as cfg
            target_domains = getattr(cfg, 'DETECTOR_FINETUNING_TARGET_DOMAINS', ["HUST", "HIT"])
            if dataset_name in target_domains:
                print(f"[Online Finetuning] Target domain '{dataset_name}' detected, applying unsupervised finetuning...")
                finetuning_applied = self._apply_online_finetuning(dataloader, dataset_name)
            if finetuning_applied:
                self._finetuned_datasets.add(dataset_name)
                # Ensure models are in eval mode after finetuning
                self.fusion_module.eval()
                if self.fusion_decoder is not None:
                    self.fusion_decoder.eval()
            else:
                # 如果微调失败，恢复原始状态
                if self._original_fusion_state is not None:
                    self.fusion_module.load_state_dict(self._original_fusion_state)
                if self._original_decoder_state is not None and self.fusion_decoder is not None:
                    self.fusion_decoder.load_state_dict(self._original_decoder_state)

        # 获取统一的text prototypes：优先使用训练checkpoint注入的 prototypes，
        # 否则回退为基于 FAULT_PROMPT_TEMPLATES 的纯文本 prototypes（zero-shot）。
        if self._cached_text_prototypes is None:
            if self._external_text_prototypes is not None:
                self._cached_text_prototypes = self._external_text_prototypes
                print(f"[Debug] Using checkpoint text prototypes: {list(self._cached_text_prototypes.keys())}")
            else:
                print("[Debug] Building text prototypes from prompts...")
                self._cached_text_prototypes = self._build_all_text_prototypes()
                print(f"[Debug] Prototypes built: {list(self._cached_text_prototypes.keys())}")
        
        # ===== Build prototypes for semantic scoring =====
        # Base: text prototypes (from checkpoint or prompts)
        text_protos = self._cached_text_prototypes

        # Optional: mix in EMA visual prototypes learned on SOURCE domains
        mixed = dict(text_protos) if text_protos is not None else {}
        try:
            import config as cfg
            use_vp = bool(getattr(cfg, "USE_VISUAL_PROTOTYPES", True))
            a_vis = float(getattr(cfg, "SEMANTIC_MIX_VISUAL_ALPHA", 0.70))
        except Exception:
            use_vp, a_vis = True, 0.70
        a_vis = max(0.0, min(1.0, float(a_vis)))
        if use_vp and self.fusion_decoder is not None and hasattr(self.fusion_decoder, "get_visual_prototypes_dict"):
            try:
                vis = self.fusion_decoder.get_visual_prototypes_dict()
            except Exception:
                vis = {}
            if isinstance(vis, dict) and len(vis) > 0:
                keys = sorted(set(list(mixed.keys()) + list(vis.keys())))
                out = {}
                for k in keys:
                    t = mixed.get(k, None)
                    v = vis.get(k, None)
                    if t is None and v is None:
                        continue
                    if t is None:
                        p = v.to(self.device)
                    elif v is None:
                        p = t.to(self.device)
                    else:
                        # Ensure same device BEFORE arithmetic (fixes cpu/cuda mismatch)
                        v_dev = v.to(self.device)
                        t_dev = t.to(self.device)
                        p = a_vis * v_dev + (1.0 - a_vis) * t_dev
                    out[int(k)] = F.normalize(p, dim=-1)
                mixed = out if len(out) > 0 else mixed

        filtered_prototypes = self._filter_prototypes_for_domain(mixed, prototype_domain)
        prototypes_to_use = filtered_prototypes if filtered_prototypes else mixed
        if 0 not in prototypes_to_use:
            raise KeyError("text/visual prototypes must contain label 0 (normal).")
        normal_proto = prototypes_to_use[0]
        fault_keys = [i for i in sorted(prototypes_to_use.keys()) if i > 0]
        if len(fault_keys) == 0:
            raise KeyError("prototypes must contain at least one fault class (label > 0).")
        fault_protos = torch.stack([prototypes_to_use[i] for i in fault_keys], dim=0)  # [K, D]
        
        # 添加诊断输出
        if enable_diagnosis:
            print(f"[Debug] normal_proto shape: {normal_proto.shape}, norm: {normal_proto.norm():.4f}")
            print(f"[Debug] fault_protos shape: {fault_protos.shape}, num_faults: {fault_protos.shape[0]}")
        
        # Determine what embeddings we need based on scoring and mode
        # This avoids redundant computation
        if scoring == "both":
            # Triplet fusion: alpha*semantic + beta*decoder + gamma*legacy_fusion
            # where gamma = 1 - alpha - beta
            modes = ["fusion", "fusion_decoder"]
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=modes)
            labels_cpu = embeddings_dict["labels"]
            
            z = embeddings_dict["fusion"].to(self.device)
            logits = embeddings_dict["fusion_decoder"].to(self.device)

            # Use semantic space embedding for semantic scoring (reduces "all looks normal" bias)
            z_sem = z
            try:
                import config as cfg
                use_z_sem = bool(getattr(cfg, "SEMANTIC_SCORE_USE_Z_SEM", True))
            except Exception:
                use_z_sem = True
            if use_z_sem and self.fusion_decoder is not None and hasattr(self.fusion_decoder, "get_sem_embedding"):
                try:
                    z_sem = self.fusion_decoder.get_sem_embedding(z)
                except Exception:
                    z_sem = z
            
            # Compute three scores:
            # 1. semantic: max(sim_fault) - sim_normal
            if enable_diagnosis:
                semantic_scores_tensor = self._diagnose_semantic_scoring(z_sem, normal_proto, fault_protos, dataset_name or "unknown")
                semantic_scores = semantic_scores_tensor.detach().cpu().numpy()
            else:
                semantic_scores = self.compute_text_anomaly_score(z_sem, normal_proto, fault_protos).detach().cpu().numpy()
            
            # 2. decoder: 1 - p(normal)
            decoder_scores = self.compute_decoder_anomaly_score(logits, normal_idx=0).cpu().numpy()
            
            # 3. fusion (legacy): 1 - sim(z, normal_proto)
            legacy_scores = self.compute_text_anomaly_score_legacy(z, normal_proto).detach().cpu().numpy()
            
            if len(semantic_scores) != len(decoder_scores) or len(semantic_scores) != len(legacy_scores):
                raise RuntimeError("Length mismatch between semantic, decoder, and legacy scores; check dataloader ordering.")
            
            # Rank-based normalization: convert scores to [0,1] ranking probabilities
            # This preserves AUC while aligning scales for linear combination
            def to_rank_prob(scores):
                # argsort().argsort() gives rank of each element (0 for min, N-1 for max)
                # Divide by (N-1) to get [0, 1] range
                n = len(scores)
                if n <= 1:
                    return np.ones_like(scores) * 0.5
                ranks = scores.argsort().argsort().astype(np.float32)
                return ranks / (n - 1.0)
            
            semantic_ranks = to_rank_prob(semantic_scores)
            decoder_ranks = to_rank_prob(decoder_scores)
            legacy_ranks = to_rank_prob(legacy_scores)
            
            # Compute gamma (ensure non-negative)
            gamma = max(0.0, 1.0 - alpha - beta)
            
            # Triplet linear combination
            scores = alpha * semantic_ranks + beta * decoder_ranks + gamma * legacy_ranks
            
            if enable_diagnosis:
                print(f"\n[Diagnosis {dataset_name or 'unknown'}] Both Scoring (Triplet Fusion, alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}):")
                print(f"  Semantic scores: min={semantic_scores.min():.4f}, max={semantic_scores.max():.4f}, mean={semantic_scores.mean():.4f}")
                print(f"  Decoder scores: min={decoder_scores.min():.4f}, max={decoder_scores.max():.4f}, mean={decoder_scores.mean():.4f}")
                print(f"  Legacy scores: min={legacy_scores.min():.4f}, max={legacy_scores.max():.4f}, mean={legacy_scores.mean():.4f}")
                print(f"  Semantic ranks: min={semantic_ranks.min():.4f}, max={semantic_ranks.max():.4f}, mean={semantic_ranks.mean():.4f}")
                print(f"  Decoder ranks: min={decoder_ranks.min():.4f}, max={decoder_ranks.max():.4f}, mean={decoder_ranks.mean():.4f}")
                print(f"  Legacy ranks: min={legacy_ranks.min():.4f}, max={legacy_ranks.max():.4f}, mean={legacy_ranks.mean():.4f}")
                print(f"  Combined scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        elif scoring == "both_legacy":
            # Fuse legacy(fusion) + decoder using Noisy-OR with alpha weighting
            # This ensures alpha=1.0 recovers legacy performance (0.855 for HUST)
            modes = ["fusion", "fusion_decoder"]
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=modes)
            labels_cpu = embeddings_dict["labels"]
            
            z = embeddings_dict["fusion"].to(self.device)
            logits = embeddings_dict["fusion_decoder"].to(self.device)
            
            # legacy(fusion): distance to normal prototype (higher = more anomalous)
            legacy_scores = self.compute_text_anomaly_score_legacy(z, normal_proto).detach().cpu().numpy()
            
            # decoder score: 1 - p(normal) (already in [0,1] probability space)
            decoder_probs = self.compute_decoder_anomaly_score(logits, normal_idx=0).detach().cpu().numpy()
            
            # Rank-normalize legacy scores to [0,1] probability space (monotonic transformation preserves AUC)
            # This ensures alpha=1.0 => pure legacy => AUC unchanged
            order = legacy_scores.argsort()
            legacy_probs = np.empty_like(legacy_scores, dtype=np.float32)
            legacy_probs[order] = np.linspace(0.0, 1.0, num=len(legacy_scores), endpoint=True)
            
            # Weighted Noisy-OR aggregation: P_final = 1 - (1-P_legacy)^alpha * (1-P_decoder)^(1-alpha)
            # alpha=1.0 => P_final = 1 - (1-P_legacy) = P_legacy (pure legacy, recovers 0.855)
            # alpha=0.0 => P_final = 1 - (1-P_decoder) = P_decoder (pure decoder)
            legacy_weight = alpha
            decoder_weight = 1.0 - alpha
            combined_probs = 1.0 - np.power(1.0 - legacy_probs, legacy_weight) * np.power(1.0 - decoder_probs, decoder_weight)
            
            # Convert back to logit space for AUC computation
            combined_probs = np.clip(combined_probs, 1e-7, 1.0 - 1e-7)
            scores = np.log(combined_probs / (1.0 - combined_probs))
            
            if enable_diagnosis:
                print(f"\n[Diagnosis {dataset_name or 'unknown'}] Both-Legacy Scoring (Noisy-OR, alpha={alpha:.2f}):")
                print(f"  Legacy scores: min={legacy_scores.min():.4f}, max={legacy_scores.max():.4f}, mean={legacy_scores.mean():.4f}")
                print(f"  Legacy probs (rank-norm): min={legacy_probs.min():.4f}, max={legacy_probs.max():.4f}, mean={legacy_probs.mean():.4f}")
                print(f"  Decoder probs: min={decoder_probs.min():.4f}, max={decoder_probs.max():.4f}, mean={decoder_probs.mean():.4f}")
                print(f"  Combined probs: min={combined_probs.min():.4f}, max={combined_probs.max():.4f}, mean={combined_probs.mean():.4f}")
        
        elif scoring == "semantic":
            # Need fusion embeddings
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=["fusion"])
            labels_cpu = embeddings_dict["labels"]
            z = embeddings_dict["fusion"].to(self.device)
            z_sem = z
            try:
                import config as cfg
                use_z_sem = bool(getattr(cfg, "SEMANTIC_SCORE_USE_Z_SEM", True))
            except Exception:
                use_z_sem = True
            if use_z_sem and self.fusion_decoder is not None and hasattr(self.fusion_decoder, "get_sem_embedding"):
                try:
                    z_sem = self.fusion_decoder.get_sem_embedding(z)
                except Exception:
                    z_sem = z
            # 使用统一的prototypes
            if enable_diagnosis:
                scores_tensor = self._diagnose_semantic_scoring(z_sem, normal_proto, fault_protos, dataset_name or "unknown")
                scores = scores_tensor.detach().cpu().numpy()
            else:
                scores = self.compute_text_anomaly_score(z_sem, normal_proto, fault_protos).detach().cpu().numpy()
        
        elif scoring == "decoder":
            # Need fusion_decoder logits
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=["fusion_decoder"])
            labels_cpu = embeddings_dict["labels"]
            logits = embeddings_dict["fusion_decoder"].to(self.device)
            scores = self.compute_decoder_anomaly_score(logits, normal_idx=0).cpu().numpy()
        
        elif scoring == "legacy":
            # For legacy, use the mode requested
            if mode in ["spec", "signal"]:
                embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=[mode])
                labels_cpu = embeddings_dict["labels"]
                emb = embeddings_dict[mode].to(self.device)
            elif mode == "fusion":
                embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=["fusion"])
                labels_cpu = embeddings_dict["labels"]
                emb = embeddings_dict["fusion"].to(self.device)
            else:  # mode == "fusion_decoder" but scoring == "legacy"
                # get fusion embeddings (not logits) to compute legacy scoring
                embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=["fusion"])
                labels_cpu = embeddings_dict["labels"]
                emb = embeddings_dict["fusion"].to(self.device)
            
            # 使用统一的normal prototype（而不是build_normal_text_prototype()）
            if enable_diagnosis:
                print(f"\n[Diagnosis {dataset_name or 'unknown'}] Legacy Scoring ({mode}):")
                print(f"  emb shape: {emb.shape}, emb norm: {F.normalize(emb, dim=-1).norm(dim=1).mean():.4f}")
                print(f"  normal_proto shape: {normal_proto.shape}, norm: {F.normalize(normal_proto, dim=-1).norm():.4f}")
            
            scores = self.compute_text_anomaly_score_legacy(emb, normal_proto).cpu().numpy()
            
            if enable_diagnosis:
                print(f"  scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        elif scoring == "evidence_or":
            # Multi-evidence aggregation: Noisy-OR combination of semantic and NF scores
            # This provides robustness when one modality is unreliable
            # Additionally uses entropy-based dynamic weighting for HUST robustness
            modes = ["fusion", "fusion_decoder"]
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=modes)
            labels_cpu = embeddings_dict["labels"]
            z = embeddings_dict["fusion"].to(self.device)
            z_sem = z
            try:
                import config as cfg
                use_z_sem = bool(getattr(cfg, "SEMANTIC_SCORE_USE_Z_SEM", True))
            except Exception:
                use_z_sem = True
            if use_z_sem and self.fusion_decoder is not None and hasattr(self.fusion_decoder, "get_sem_embedding"):
                try:
                    z_sem = self.fusion_decoder.get_sem_embedding(z)
                except Exception:
                    z_sem = z
            
            # Compute semantic scores (normalized to probability space)
            semantic_scores_raw = self.compute_text_anomaly_score(z_sem, normal_proto, fault_protos).detach().cpu().numpy()
            # Normalize to [0, 1] using sigmoid
            semantic_probs = 1.0 / (1.0 + np.exp(-semantic_scores_raw))
            
            # Compute NF scores (normalized to probability space)
            nf_scores_raw = self.compute_nf_anomaly_score(z, score_type="logit_diff").detach().cpu().numpy()
            nf_probs = 1.0 / (1.0 + np.exp(-nf_scores_raw))
            
            # Compute decoder entropy for dynamic weighting
            # High entropy = uncertain decoder -> rely more on semantic
            logits = embeddings_dict["fusion_decoder"].to(self.device)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # [B]
            # Normalize entropy to [0, 1] range (assuming max entropy ~ log(num_classes))
            num_classes = logits.shape[1]
            max_entropy = np.log(num_classes)
            normalized_entropy = (entropy / max_entropy).cpu().numpy()  # [B]
            
            # Dynamic weighting: high entropy -> more weight on semantic
            # semantic_weight = 0.3 + 0.4 * normalized_entropy (range: [0.3, 0.7])
            # nf_weight = 0.7 - 0.4 * normalized_entropy (range: [0.3, 0.7])
            semantic_weight = 0.3 + 0.4 * normalized_entropy
            nf_weight = 0.7 - 0.4 * normalized_entropy
            
            # Weighted Noisy-OR aggregation
            # P_final = 1 - (1 - P_semantic)^w_sem * (1 - P_nf)^w_nf
            combined_probs = 1.0 - np.power(1.0 - semantic_probs, semantic_weight) * np.power(1.0 - nf_probs, nf_weight)
            
            # Convert back to logit space for AUC computation
            combined_probs = np.clip(combined_probs, 1e-7, 1.0 - 1e-7)
            scores = np.log(combined_probs / (1.0 - combined_probs))
            
            if enable_diagnosis:
                print(f"\n[Diagnosis {dataset_name or 'unknown'}] Evidence-OR Scoring (Entropy-Weighted):")
                print(f"  Semantic probs: min={semantic_probs.min():.4f}, max={semantic_probs.max():.4f}, mean={semantic_probs.mean():.4f}")
                print(f"  NF probs: min={nf_probs.min():.4f}, max={nf_probs.max():.4f}, mean={nf_probs.mean():.4f}")
                print(f"  Entropy: min={normalized_entropy.min():.4f}, max={normalized_entropy.max():.4f}, mean={normalized_entropy.mean():.4f}")
                print(f"  Semantic weight: min={semantic_weight.min():.4f}, max={semantic_weight.max():.4f}, mean={semantic_weight.mean():.4f}")
                print(f"  NF weight: min={nf_weight.min():.4f}, max={nf_weight.max():.4f}, mean={nf_weight.mean():.4f}")
                print(f"  Combined probs: min={combined_probs.min():.4f}, max={combined_probs.max():.4f}, mean={combined_probs.mean():.4f}")
        
        elif scoring == "evidence_max":
            # Use the decoder's internal multi-granularity evidence aggregation
            modes = ["fusion"]
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=modes)
            labels_cpu = embeddings_dict["labels"]
            z = embeddings_dict["fusion"].to(self.device)
            
            # Use decoder's evidence_max scoring with temperature control
            scores_logit = self.compute_nf_anomaly_score(z, score_type="evidence_max", evidence_temperature=evidence_temperature).detach().cpu().numpy()
            scores = scores_logit
            
            if enable_diagnosis:
                print(f"\n[Diagnosis {dataset_name or 'unknown'}] Evidence-Max Scoring:")
                print(f"  scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, std={scores.std():.4f}")

        elif scoring == "soft_hierarchical":
            # Soft hierarchical probability combination: P(k) = P(fault) * P(k|fault)
            # Enhanced with nf_multi aggregation for robustness
            modes = ["fusion"]
            embeddings_dict = self.extract_multimodal_embeddings(dataloader, modes=modes)
            labels_cpu = embeddings_dict["labels"]
            z = embeddings_dict["fusion"].to(self.device)

            # Use FusionDecoder's soft hierarchical probability method
            if self.fusion_decoder is not None and hasattr(self.fusion_decoder, 'get_class_probs_soft_hierarchical'):
                probs4 = self.fusion_decoder.get_class_probs_soft_hierarchical(z, temperature=1.0)  # [B, 4]
                # Use normal probability as anomaly score (lower = more normal = less anomalous)
                # Convert to anomaly score: 1 - P(normal)
                scores = (1.0 - probs4[:, 0]).detach().cpu().numpy()

                if enable_diagnosis:
                    print(f"\n[Diagnosis {dataset_name or 'unknown'}] Soft Hierarchical Scoring:")
                    print(f"  P(normal): min={probs4[:, 0].min():.4f}, max={probs4[:, 0].max():.4f}, mean={probs4[:, 0].mean():.4f}")
                    print(f"  P(inner): min={probs4[:, 1].min():.4f}, max={probs4[:, 1].max():.4f}, mean={probs4[:, 1].mean():.4f}")
                    print(f"  P(outer): min={probs4[:, 2].min():.4f}, max={probs4[:, 2].max():.4f}, mean={probs4[:, 2].mean():.4f}")
                    print(f"  P(rolling): min={probs4[:, 3].min():.4f}, max={probs4[:, 3].max():.4f}, mean={probs4[:, 3].mean():.4f}")
                    print(f"  Anomaly scores (1-P(normal)): min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            else:
                # Fallback to regular decoder scoring
                print(f"[Warning] Soft hierarchical scoring requires hierarchical fusion_decoder, falling back to decoder scoring")
                embeddings_dict_logits = self.extract_multimodal_embeddings(dataloader, modes=["fusion_decoder"])
                logits = embeddings_dict_logits["fusion_decoder"].to(self.device)
                scores = self.compute_decoder_anomaly_score(logits, normal_idx=0).cpu().numpy()

        # binary labels (CPU numpy)
        binary_labels = self.build_binary_labels(labels_cpu).numpy()

        # compute ROC AUC (guard against single-class)
        try:
            auc = roc_auc_score(binary_labels, scores)
            fpr, tpr, _ = roc_curve(binary_labels, scores)
        except ValueError:
            auc = float("nan")
            fpr, tpr = [], []
        
        # Compute AUC_nf from decoder NF anomaly score.
        # NOTE:
        # - hierarchical head: uses global-NF / hierarchical score.
        # - flat head: compute_nf_anomaly_score() falls back to "1 - P(normal)".
        auc_nf = float("nan")
        if self.fusion_decoder is not None:
            try:
                # Get fusion embeddings if not already computed
                if scoring in ["semantic", "both"] or (scoring == "legacy" and mode == "fusion"):
                    z_fusion = embeddings_dict["fusion"].to(self.device)
                else:
                    # Need to compute fusion embeddings
                    embeddings_dict_fusion = self.extract_multimodal_embeddings(dataloader, modes=["fusion"])
                    z_fusion = embeddings_dict_fusion["fusion"].to(self.device)
                
                # Compute nf-based anomaly scores
                # 使用概率融合的 nf score (prob_hier_or) 以提升跨域性能，特别是 HIT
                # 这会让 AUC_nf 受益于 global nf 和 nf_multi 的融合
                try:
                    nf_scores = self.compute_nf_anomaly_score(z_fusion, score_type="prob_hier_or").detach().cpu().numpy()
                except (RuntimeError, AttributeError):
                    # Fallback to logit_diff if prob_hier_or is not available
                    nf_scores = self.compute_nf_anomaly_score(z_fusion, score_type="logit_diff").detach().cpu().numpy()
                auc_nf = roc_auc_score(binary_labels, nf_scores)
            except (ValueError, RuntimeError) as e:
                # Handle cases where nf scoring fails
                auc_nf = float("nan")
                if enable_diagnosis:
                    print(f"[Warning] Failed to compute AUC_nf: {e}")

        # Return results with clear labeling:
        # - auc_nf: PRIMARY metric, from hierarchical nf head (normal vs fault)
        # - auc: LEGACY/SEMANTIC/BOTH metric (depends on scoring parameter)
        # - scoring_method: indicates which method was used for 'auc'
        return {
            "auc_nf": auc_nf,  # PRIMARY: AUC from hierarchical nf head
            "auc": auc,        # LEGACY: AUC from scoring method (legacy/semantic/decoder/both)
            "auc_legacy": auc if scoring in ["legacy", "both"] else float("nan"),  # For backward compatibility
            "auc_semantic": auc if scoring in ["semantic", "both"] else float("nan"),  # For backward compatibility
            "auc_decoder": auc if scoring in ["decoder", "both"] else float("nan"),  # For backward compatibility
            "fpr": fpr,
            "tpr": tpr,
            "scoring_method": scoring  # Document which scoring method was used
        }
