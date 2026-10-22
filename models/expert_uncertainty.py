"""
Expert Disagreement Estimator for Uncertainty-based Routing

Estimates uncertainty based on disagreement among expert outputs,
and modulates routing temperature dynamically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertDisagreementEstimator(nn.Module):
    """
    Estimates uncertainty based on disagreement among expert outputs.
    
    Provides multiple methods:
    - variance: Compute variance of expert outputs
    - entropy: Compute entropy of expert output distribution
    - pairwise_distance: Compute pairwise distances between expert outputs
    
    Args:
        method: Disagreement estimation method ('variance', 'entropy', 'pairwise_distance')
        normalize: Whether to normalize disagreement to [0, 1] range
    """
    
    def __init__(self, method='variance', normalize=True):
        super().__init__()
        self.method = method
        self.normalize = normalize
        
        if method not in ['variance', 'entropy', 'pairwise_distance']:
            raise ValueError(f"Unknown disagreement method: {method}. "
                           f"Choose from: 'variance', 'entropy', 'pairwise_distance'")
        
        print(f"[ExpertDisagreementEstimator] Initialized with method={method}, normalize={normalize}")
    
    def forward(self, expert_outputs, expert_weights=None):
        """
        Compute expert disagreement.
        
        Args:
            expert_outputs: Expert outputs [B, num_experts, D]
            expert_weights: Optional expert weights [B, num_experts] for weighted disagreement
        
        Returns:
            disagreement: Overall disagreement per sample [B]
            per_expert_disagreement: Disagreement contribution per expert [B, num_experts] (optional)
        """
        return self.compute_disagreement(expert_outputs, expert_weights)
    
    def compute_disagreement(self, expert_outputs, expert_weights=None):
        """
        Compute disagreement based on selected method.
        
        Args:
            expert_outputs: Expert outputs [B, num_experts, D]
            expert_weights: Optional weights [B, num_experts]
        
        Returns:
            disagreement: [B] overall disagreement score
            per_expert_disagreement: [B, num_experts] per-expert contribution (for some methods)
        """
        if self.method == 'variance':
            return self._compute_variance_disagreement(expert_outputs, expert_weights)
        elif self.method == 'entropy':
            return self._compute_entropy_disagreement(expert_outputs, expert_weights)
        elif self.method == 'pairwise_distance':
            return self._compute_pairwise_disagreement(expert_outputs, expert_weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_variance_disagreement(self, expert_outputs, expert_weights=None):
        """
        Compute variance-based disagreement.
        
        Measures variance of expert outputs across the expert dimension.
        High variance indicates disagreement.
        """
        # expert_outputs: [B, num_experts, D]
        B, num_experts, D = expert_outputs.shape
        
        if expert_weights is not None:
            # Weighted variance
            # Expand weights: [B, num_experts, 1]
            weights_expanded = expert_weights.unsqueeze(-1)
            
            # Weighted mean: [B, 1, D]
            weighted_mean = (expert_outputs * weights_expanded).sum(dim=1, keepdim=True)
            
            # Weighted variance: sum(w * (x - mean)^2)
            squared_diff = (expert_outputs - weighted_mean) ** 2
            variance = (weights_expanded * squared_diff).sum(dim=1)  # [B, D]
        else:
            # Unweighted variance
            variance = expert_outputs.var(dim=1)  # [B, D]
        
        # Aggregate variance across feature dimension
        disagreement = variance.mean(dim=-1)  # [B]
        
        # Per-expert contribution (variance from mean)
        mean_output = expert_outputs.mean(dim=1, keepdim=True)  # [B, 1, D]
        per_expert_var = ((expert_outputs - mean_output) ** 2).mean(dim=-1)  # [B, num_experts]
        
        if self.normalize:
            # Normalize to [0, 1] using sigmoid
            disagreement = torch.sigmoid(disagreement)
            per_expert_var = torch.sigmoid(per_expert_var)
        
        return disagreement, per_expert_var
    
    def _compute_entropy_disagreement(self, expert_outputs, expert_weights=None):
        """
        Compute entropy-based disagreement.
        
        Treats expert outputs as probability distributions and computes entropy.
        High entropy indicates high disagreement/uncertainty.
        """
        # expert_outputs: [B, num_experts, D]
        B, num_experts, D = expert_outputs.shape
        
        # Normalize expert outputs to probabilities (softmax over experts)
        expert_probs = F.softmax(expert_outputs, dim=1)  # [B, num_experts, D]
        
        # Compute entropy for each feature dimension
        epsilon = 1e-10
        entropy = -torch.sum(expert_probs * torch.log(expert_probs + epsilon), dim=1)  # [B, D]
        
        # Aggregate entropy across feature dimension
        disagreement = entropy.mean(dim=-1)  # [B]
        
        # Per-expert contribution (KL divergence from mean distribution)
        mean_probs = expert_probs.mean(dim=1, keepdim=True)  # [B, 1, D]
        kl_div = torch.sum(
            expert_probs * torch.log((expert_probs + epsilon) / (mean_probs + epsilon)),
            dim=-1
        )  # [B, num_experts]
        
        if self.normalize:
            # Normalize entropy by max entropy (log(num_experts))
            max_entropy = math.log(num_experts)
            disagreement = disagreement / max_entropy
            kl_div = kl_div / max_entropy
        
        return disagreement, kl_div
    
    def _compute_pairwise_disagreement(self, expert_outputs, expert_weights=None):
        """
        Compute pairwise distance-based disagreement.
        
        Measures average pairwise distance between expert outputs.
        High pairwise distance indicates disagreement.
        """
        # expert_outputs: [B, num_experts, D]
        B, num_experts, D = expert_outputs.shape
        
        # Compute pairwise distances
        # Expand: [B, num_experts, 1, D] and [B, 1, num_experts, D]
        expert_i = expert_outputs.unsqueeze(2)  # [B, num_experts, 1, D]
        expert_j = expert_outputs.unsqueeze(1)  # [B, 1, num_experts, D]
        
        # Pairwise L2 distances: [B, num_experts, num_experts]
        pairwise_dist = torch.norm(expert_i - expert_j, p=2, dim=-1)
        
        if expert_weights is not None:
            # Weight pairwise distances
            # weights: [B, num_experts]
            # Create weight matrix: [B, num_experts, num_experts]
            weight_i = expert_weights.unsqueeze(2)  # [B, num_experts, 1]
            weight_j = expert_weights.unsqueeze(1)  # [B, 1, num_experts]
            weight_matrix = weight_i * weight_j  # [B, num_experts, num_experts]
            
            # Weighted average of pairwise distances
            disagreement = (pairwise_dist * weight_matrix).sum(dim=(1, 2)) / (weight_matrix.sum(dim=(1, 2)) + 1e-10)
        else:
            # Average pairwise distance (exclude diagonal)
            # Create mask to exclude diagonal
            mask = ~torch.eye(num_experts, dtype=torch.bool, device=expert_outputs.device)
            mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, num_experts, num_experts]
            
            disagreement = (pairwise_dist * mask.float()).sum(dim=(1, 2)) / (num_experts * (num_experts - 1))
        
        # Per-expert contribution (average distance to other experts)
        per_expert_dist = pairwise_dist.mean(dim=2)  # [B, num_experts]
        
        if self.normalize:
            # Normalize using sigmoid
            disagreement = torch.sigmoid(disagreement)
            per_expert_dist = torch.sigmoid(per_expert_dist)
        
        return disagreement, per_expert_dist
    
    def modulate_temperature(self, base_temperature, disagreement, scale=2.0, mode='increase'):
        """
        Modulate routing temperature based on disagreement.
        
        High disagreement -> higher temperature (more exploration)
        Low disagreement -> lower temperature (more exploitation)
        
        Args:
            base_temperature: Base temperature value (scalar or tensor)
            disagreement: Disagreement scores [B] or scalar
            scale: Scaling factor for temperature modulation
            mode: 'increase' (high disagreement -> higher temp) or 
                  'decrease' (high disagreement -> lower temp)
        
        Returns:
            modulated_temperature: Adjusted temperature [B] or scalar
        """
        if mode == 'increase':
            # Higher disagreement -> higher temperature
            # temp = base_temp * (1 + scale * disagreement)
            modulated_temperature = base_temperature * (1.0 + scale * disagreement)
        elif mode == 'decrease':
            # Higher disagreement -> lower temperature
            # temp = base_temp / (1 + scale * disagreement)
            modulated_temperature = base_temperature / (1.0 + scale * disagreement)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'increase' or 'decrease'")
        
        return modulated_temperature
    
    def compute_disagreement_statistics(self, expert_outputs, expert_weights=None):
        """
        Compute comprehensive disagreement statistics for monitoring.
        
        Args:
            expert_outputs: [B, num_experts, D]
            expert_weights: Optional [B, num_experts]
        
        Returns:
            dict with various disagreement metrics
        """
        stats = {}
        
        # Current method
        disagreement, per_expert = self.compute_disagreement(expert_outputs, expert_weights)
        stats[f'{self.method}_disagreement'] = disagreement.mean().item()
        stats[f'{self.method}_disagreement_std'] = disagreement.std().item()
        stats[f'{self.method}_per_expert_mean'] = per_expert.mean().item()
        
        # Additional metrics
        # Mean output magnitude
        output_norm = torch.norm(expert_outputs, p=2, dim=-1)  # [B, num_experts]
        stats['expert_output_norm_mean'] = output_norm.mean().item()
        stats['expert_output_norm_std'] = output_norm.std().item()
        
        # Coefficient of variation of expert outputs
        mean_output = expert_outputs.mean(dim=1)  # [B, D]
        std_output = expert_outputs.std(dim=1)  # [B, D]
        cv = (std_output / (mean_output.abs() + 1e-10)).mean(dim=-1)  # [B]
        stats['expert_coefficient_of_variation'] = cv.mean().item()
        
        return stats


class DynamicTemperatureScheduler:
    """
    Scheduler for dynamically adjusting router temperature based on disagreement.
    
    Implements adaptive temperature control with smoothing and bounds.
    
    Args:
        base_temperature: Initial/base temperature
        min_temperature: Minimum allowed temperature
        max_temperature: Maximum allowed temperature
        smoothing_factor: EMA smoothing factor (0 = no smoothing, 1 = no update)
        adaptation_rate: Rate of temperature adaptation
    """
    
    def __init__(
        self,
        base_temperature=1.0,
        min_temperature=0.5,
        max_temperature=3.0,
        smoothing_factor=0.9,
        adaptation_rate=0.1
    ):
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.smoothing_factor = smoothing_factor
        self.adaptation_rate = adaptation_rate
        
        self.current_temperature = base_temperature
        self.disagreement_history = []
    
    def update(self, disagreement_score):
        """
        Update temperature based on disagreement score.
        
        Args:
            disagreement_score: Scalar disagreement value
        
        Returns:
            current_temperature: Updated temperature
        """
        # Record disagreement
        self.disagreement_history.append(disagreement_score)
        if len(self.disagreement_history) > 100:
            self.disagreement_history.pop(0)
        
        # Compute target temperature
        # Higher disagreement -> higher temperature
        target_temperature = self.base_temperature * (1.0 + self.adaptation_rate * disagreement_score)
        
        # Clip to bounds
        target_temperature = max(self.min_temperature, min(self.max_temperature, target_temperature))
        
        # EMA smoothing
        self.current_temperature = (
            self.smoothing_factor * self.current_temperature +
            (1 - self.smoothing_factor) * target_temperature
        )
        
        return self.current_temperature
    
    def get_statistics(self):
        """Get scheduler statistics."""
        if not self.disagreement_history:
            return {}
        
        return {
            'current_temperature': self.current_temperature,
            'mean_disagreement': sum(self.disagreement_history) / len(self.disagreement_history),
            'recent_disagreement': self.disagreement_history[-1] if self.disagreement_history else 0.0
        }

