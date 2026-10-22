"""
训练相关工具函数
包括checkpoint加载、训练参数构建等
"""
import os
import re

from utils.build_text_prototypes_main import build_text_prototypes
from utils.checkpoint import load_checkpoint, load_text_prototypes


def infer_checkpoint_tag(checkpoint_path, default_tag="withNewSpecEncoder_x2"):
    """
    从checkpoint路径推断tag：epoch_{k}_{tag}.pth -> tag
    """
    base = os.path.basename(checkpoint_path or "")
    m = re.search(r"epoch_\d+_(.+)\.pth$", base)
    return m.group(1) if m else default_tag


def load_model_checkpoint(checkpoint_path, signal_encoder, spec_encoder, text_encoder, 
                          fusion_module, fusion_decoder, optimizer=None, device="cpu"):
    """加载模型checkpoint的辅助函数"""
    return load_checkpoint(
        checkpoint_path=checkpoint_path,
        signal_encoder=signal_encoder,
        spec_encoder=spec_encoder,
        text_encoder=text_encoder,
        fusion_module=fusion_module,
        fusion_decoder=fusion_decoder,
        optimizer=optimizer,
        device=device,
    )


def build_train_alignment_kwargs(cfg, train_loader, signal_encoder, spec_encoder, text_encoder,
                                  fusion_module, fusion_decoder, optimizer, device, clip_model,
                                  hust_loader, start_epoch, precomputed_class_weights, checkpoint_tag,
                                  hit_loader=None, val_loader=None, wst_encoder=None, env_encoder=None, scd_encoder=None):
    """构建train_alignment函数的参数字典"""
    return {
        "train_loader": train_loader,
        "signal_encoder": signal_encoder,
        "spec_encoder": spec_encoder,
        "text_encoder": text_encoder,
        "fusion_module": fusion_module,
        "fusion_decoder": fusion_decoder,
        "optimizer": optimizer,
        "device": device,
        "clip_model": clip_model,
        "lambda_clip": cfg.LAMBDA_CLIP,
        "start_epoch": start_epoch,
        "num_epochs": cfg.TOTAL_EPOCHS,
        "hust_loader": hust_loader,
        "hit_loader": hit_loader,
        "val_loader": val_loader,
        "build_text_prototypes_fn": build_text_prototypes,
        "fault_prompt_templates": cfg.FAULT_PROMPT_TEMPLATES,
        "test_interval": cfg.TEST_INTERVAL,
        "use_moe": cfg.USE_MOE,
        "moe_loss_weight": cfg.MOE_LOSS_WEIGHT,
        "use_bayesian_prototypes": cfg.USE_BAYESIAN_PROTOTYPES,
        "bayesian_prior_strength": cfg.BAYESIAN_PRIOR_STRENGTH,
        "bayesian_update_until_epoch": cfg.BAYESIAN_UPDATE_UNTIL_EPOCH,
        # 训练时zero_shot评估的在线微调配置
        "enable_finetuning": cfg.ZERO_SHOT_FINETUNING_ENABLE,
        "finetune_samples": cfg.ZERO_SHOT_FINETUNING_SAMPLES,
        "finetune_lr": cfg.ZERO_SHOT_FINETUNING_LR,
        "finetune_steps": cfg.ZERO_SHOT_FINETUNING_STEPS,
        # 一致性正则配置
        "use_consistency_regularization": cfg.USE_CONSISTENCY_REGULARIZATION,
        "consistency_weight": cfg.CONSISTENCY_WEIGHT,
        "consistency_temperature": cfg.CONSISTENCY_TEMPERATURE,
        # Schedule 和 Early stopping 配置
        "use_schedule": cfg.USE_SCHEDULE,
        "schedule_epoch_threshold": cfg.SCHEDULE_EPOCH_THRESHOLD,
        "w_fusion_cls_early": cfg.W_FUSION_CLS_EARLY,
        "w_fusion_cls_late": cfg.W_FUSION_CLS_LATE,
        "lr_decay_factor": cfg.LR_DECAY_FACTOR,
        "use_early_stopping": cfg.USE_EARLY_STOPPING,
        "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
        "early_stop_metric": cfg.EARLY_STOP_METRIC,
        "early_stop_min_epochs": getattr(cfg, 'EARLY_STOP_MIN_EPOCHS', 10),
        "early_stop_skip_prewindow_baseline": getattr(cfg, 'EARLY_STOP_SKIP_PREWINDOW_BASELINE', False),
        # checkpoint 命名
        "checkpoint_dir": getattr(cfg, "CHECKPOINT_DIR", "checkpoints"),
        "checkpoint_tag": checkpoint_tag,
        # 性能优化配置
        "use_amp": cfg.USE_AMP,
        "precomputed_class_weights": precomputed_class_weights,
        # 样本不均衡处理配置
        "use_progressive_weights": cfg.USE_PROGRESSIVE_WEIGHTS,
        "initial_weight_scale": cfg.INITIAL_WEIGHT_SCALE,
        "weight_schedule": cfg.WEIGHT_SCHEDULE,
        "use_focal_loss": cfg.USE_FOCAL_LOSS,
        "focal_gamma": cfg.FOCAL_GAMMA,
        "focal_alpha": cfg.FOCAL_ALPHA,
        "label_smoothing": cfg.LABEL_SMOOTHING,
        "enable_class_monitoring": cfg.ENABLE_CLASS_MONITORING,
        # 类别权重策略（IMPORTANT: must be passed explicitly; otherwise train_alignment uses its default "fixed")
        "class_weight_strategy": getattr(cfg, "CLASS_WEIGHT_STRATEGY", "fixed"),
        "class_weight_fixed_epochs": getattr(cfg, "CLASS_WEIGHT_FIXED_EPOCHS", 5),
        # sampler-first: avoid double-compensation when WeightedSampler is enabled
        "use_class_weights_in_loss": getattr(cfg, "USE_CLASS_WEIGHTS_IN_LOSS", True),
        # 渐进式学习率和类感知正则化
        "use_progressive_lr": cfg.USE_PROGRESSIVE_LR,
        "progressive_lr_schedule": cfg.PROGRESSIVE_LR_SCHEDULE,
        "fusion_class_aware_regularization": cfg.FUSION_CLASS_AWARE_REGULARIZATION,
        # EMA 权重平均
        "use_ema": cfg.USE_EMA,
        "ema_decay": cfg.EMA_DECAY,
        "ema_warmup_steps": cfg.EMA_WARMUP_STEPS,
        "ema_update_after_step": cfg.EMA_UPDATE_AFTER_STEP,
        # 校准损失（新增）
        "use_calibration_loss": cfg.USE_CALIBRATION_LOSS,
        "calibration_loss_weight": cfg.CALIBRATION_LOSS_WEIGHT,
        "calibration_n_bins": cfg.CALIBRATION_N_BINS,
        # 正交性正则化（新增）
        "use_orthogonality_reg": getattr(cfg, 'USE_ORTHOGONALITY_REG', False),
        "orthogonality_reg_weight": getattr(cfg, 'ORTHOGONALITY_REG_WEIGHT', 0.001),
        # 元学习配置（可选）
        "use_meta_learning": getattr(cfg, 'USE_META_LEARNING', False),
        "meta_learning_start_epoch": getattr(cfg, 'META_LEARNING_START_EPOCH', 5),
        "meta_learning_interval": getattr(cfg, 'META_LEARNING_INTERVAL', 2),
        "meta_lr": getattr(cfg, 'META_LR', 1e-4),
        "meta_inner_lr": getattr(cfg, 'META_INNER_LR', 1e-3),
        "meta_inner_steps": getattr(cfg, 'META_INNER_STEPS', 3),
        "meta_first_order": getattr(cfg, 'META_FIRST_ORDER', True),
        # ATT446: WST bypass encoder
        "wst_encoder": wst_encoder,
        # ATT449: Envelope bypass encoder
        "env_encoder": env_encoder,
        # ATT464b: SCD bypass encoder
        "scd_encoder": scd_encoder,
    }


def load_checkpoint_prototypes(checkpoint_path, detector, cfg, device):
    """
    尝试从checkpoint加载text_prototypes并注入到detector.
    ATT396: also loads Feature Decoupler states if present.
    
    Returns:
        tuple: (checkpoint_path, checkpoint_exists) - 更新后的checkpoint路径和存在状态
    """
    checkpoint_exists = os.path.exists(checkpoint_path)
    checkpoint_prototypes = None
    
    if checkpoint_exists:
        try:
            checkpoint_prototypes = load_text_prototypes(checkpoint_path, device=device)
            if checkpoint_prototypes is not None:
                detector.set_text_prototypes(checkpoint_prototypes)
                print(f"[Checkpoint] Loaded text_prototypes for evaluation from {checkpoint_path}")
                print(f"[Checkpoint] Prototypes keys: {sorted(checkpoint_prototypes.keys())}")
            else:
                print(f"[Checkpoint] No text_prototypes found in {checkpoint_path}, will use prompt-based prototypes")
        except Exception as e:
            print(f"[Checkpoint] Failed to load text_prototypes: {e}, will use prompt-based prototypes")

        # ATT396: load Feature Decoupler states
        try:
            import torch
            _ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if hasattr(detector, "decouple_spec") and detector.decouple_spec is not None and "decouple_spec" in _ckpt:
                detector.decouple_spec.load_state_dict(_ckpt["decouple_spec"], strict=False)
                detector.decouple_spec.eval()
                print("[Checkpoint] Loaded decouple_spec into detector")
            if hasattr(detector, "decouple_signal") and detector.decouple_signal is not None and "decouple_signal" in _ckpt:
                detector.decouple_signal.load_state_dict(_ckpt["decouple_signal"], strict=False)
                detector.decouple_signal.eval()
                print("[Checkpoint] Loaded decouple_signal into detector")
            del _ckpt
        except Exception as _dc_e:
            print(f"[Checkpoint] Feature Decoupler load skipped: {_dc_e}")
    else:
        # 尝试加载最新的checkpoint
        try:
            checkpoint_tag = infer_checkpoint_tag(checkpoint_path)
            latest_epoch = cfg.TOTAL_EPOCHS
            checkpoint_dir = getattr(cfg, "CHECKPOINT_DIR", "checkpoints")
            latest_checkpoint = f"{checkpoint_dir}/epoch_{latest_epoch}_{checkpoint_tag}.pth"
            if os.path.exists(latest_checkpoint):
                checkpoint_prototypes = load_text_prototypes(latest_checkpoint, device=device)
                if checkpoint_prototypes is not None:
                    detector.set_text_prototypes(checkpoint_prototypes)
                    print(f"[Checkpoint] Loaded text_prototypes for evaluation from {latest_checkpoint}")
                    print(f"[Checkpoint] Prototypes keys: {sorted(checkpoint_prototypes.keys())}")
        except Exception as e:
            print(f"[Checkpoint] Failed to load from latest checkpoint: {e}")
    
    return checkpoint_path, checkpoint_exists

