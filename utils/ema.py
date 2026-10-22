"""
EMA (Exponential Moving Average) 权重管理模块
用于训练过程中维护模型权重的指数移动平均，提高评估稳定性
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ModelEMA:
    """
    模型权重的指数移动平均 (EMA)
    
    在训练过程中维护模型参数的 EMA 版本，用于评估和推理。
    这可以显著提高模型在验证集上的稳定性，减少评估指标的波动。
    
    Args:
        model: 要跟踪的模型
        decay: EMA 衰减率，默认 0.9999。值越接近 1，平均窗口越大
        warmup_steps: 预热步数，在此期间 decay 会从 0 线性增长到目标值
        update_after_step: 在此步数之后才开始更新 EMA（默认 0）
        device: 设备
    
    Example:
        >>> model = MyModel()
        >>> ema = ModelEMA(model, decay=0.9999, warmup_steps=1000)
        >>> 
        >>> # 训练循环
        >>> for step, batch in enumerate(dataloader):
        >>>     loss = train_step(model, batch)
        >>>     optimizer.step()
        >>>     ema.update(model, step)  # 每步更新 EMA
        >>> 
        >>> # 评估时使用 EMA 权重
        >>> ema.apply_shadow()  # 临时替换为 EMA 权重
        >>> eval_metrics = evaluate(model, val_loader)
        >>> ema.restore()  # 恢复原始权重
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.device = device or torch.device('cpu')
        
        # 保存 EMA 权重的影子副本
        self.shadow = {}
        self.backup = {}  # 用于临时保存原始权重
        
        # 初始化：复制当前模型权重作为 EMA 的初始值
        self._register(model)
        
        print(f"[EMA] Initialized with decay={decay}, warmup_steps={warmup_steps}, update_after_step={update_after_step}")
    
    def _register(self, model: nn.Module):
        """注册模型参数，创建影子副本"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(self.device)
    
    def update(self, model: nn.Module, step: int):
        """
        更新 EMA 权重
        
        Args:
            model: 当前训练的模型
            step: 当前训练步数（用于 warmup）
        """
        if step < self.update_after_step:
            return
        
        # 计算当前步的有效 decay（考虑 warmup）
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # 线性 warmup：decay 从 0 增长到目标值
            effective_decay = self.decay * (step / self.warmup_steps)
        else:
            effective_decay = self.decay
        
        # 更新所有参数的 EMA
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA 更新公式：shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(effective_decay).add_(
                        param.data.to(self.device), alpha=1 - effective_decay
                    )
    
    def apply_shadow(self, model: nn.Module):
        """
        将模型参数临时替换为 EMA 权重（用于评估）
        
        注意：调用此方法后必须调用 restore() 来恢复原始权重
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
    
    def restore(self, model: nn.Module):
        """恢复模型的原始权重（在 apply_shadow 之后调用）"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict[str, Any]:
        """返回 EMA 状态字典（用于保存 checkpoint）"""
        return {
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'update_after_step': self.update_after_step,
            'shadow': {k: v.cpu() for k, v in self.shadow.items()}
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """从状态字典加载 EMA 权重"""
        self.decay = state_dict.get('decay', self.decay)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
        
        shadow_dict = state_dict.get('shadow', {})
        for name, value in shadow_dict.items():
            self.shadow[name] = value.to(self.device)
        
        print(f"[EMA] Loaded state dict with {len(self.shadow)} parameters")


class EMAManager:
    """
    管理多个模型的 EMA（用于多模型训练场景）
    
    Example:
        >>> ema_manager = EMAManager(decay=0.9999)
        >>> ema_manager.register('encoder', encoder_model)
        >>> ema_manager.register('decoder', decoder_model)
        >>> 
        >>> # 训练循环
        >>> for step, batch in enumerate(dataloader):
        >>>     loss = train_step(...)
        >>>     optimizer.step()
        >>>     ema_manager.update_all(step)
        >>> 
        >>> # 评估
        >>> ema_manager.apply_all()
        >>> eval_metrics = evaluate(...)
        >>> ema_manager.restore_all()
    """
    
    def __init__(
        self,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.device = device or torch.device('cpu')
        
        self.emas = {}  # {model_name: ModelEMA}
        
        print(f"[EMAManager] Initialized with decay={decay}, warmup_steps={warmup_steps}")
    
    def register(self, name: str, model: nn.Module):
        """注册一个模型到 EMA 管理器"""
        self.emas[name] = ModelEMA(
            model=model,
            decay=self.decay,
            warmup_steps=self.warmup_steps,
            update_after_step=self.update_after_step,
            device=self.device
        )
        print(f"[EMAManager] Registered model '{name}' with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    def update_all(self, step: int, models: Dict[str, nn.Module]):
        """更新所有已注册模型的 EMA"""
        for name, model in models.items():
            if name in self.emas:
                self.emas[name].update(model, step)
    
    def apply_all(self, models: Dict[str, nn.Module]):
        """将所有模型临时替换为 EMA 权重"""
        for name, model in models.items():
            if name in self.emas:
                self.emas[name].apply_shadow(model)
    
    def restore_all(self, models: Dict[str, nn.Module]):
        """恢复所有模型的原始权重"""
        for name, model in models.items():
            if name in self.emas:
                self.emas[name].restore(model)
    
    def state_dict(self) -> Dict[str, Any]:
        """返回所有 EMA 的状态字典"""
        return {
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'update_after_step': self.update_after_step,
            'emas': {name: ema.state_dict() for name, ema in self.emas.items()}
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """从状态字典加载所有 EMA"""
        self.decay = state_dict.get('decay', self.decay)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
        
        emas_dict = state_dict.get('emas', {})
        for name, ema_state in emas_dict.items():
            if name in self.emas:
                self.emas[name].load_state_dict(ema_state)
        
        print(f"[EMAManager] Loaded {len(emas_dict)} EMA states")

