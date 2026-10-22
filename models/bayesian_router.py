"""
Bayesian Router for MoE with Uncertainty Estimation

Implements Bayesian Neural Network layers for router with weight uncertainty
and KL divergence regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty.
    
    Weights are parameterized as Gaussian distributions:
    - w ~ N(mu_w, sigma_w^2)
    - Uses reparameterization trick: w = mu_w + sigma_w * epsilon
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        prior_std: Standard deviation of prior distribution N(0, prior_std^2)
    """
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters: mean and log variance
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_var_w = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters: mean and log variance
        self.mu_b = nn.Parameter(torch.Tensor(out_features))
        self.log_var_b = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization for mean."""
        # Initialize weight mean
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        
        # Initialize weight log variance (small variance initially)
        nn.init.constant_(self.log_var_w, -6.0)  # exp(-6) ≈ 0.0025
        
        # Initialize bias mean
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.mu_b, -bound, bound)
        
        # Initialize bias log variance (small variance initially)
        nn.init.constant_(self.log_var_b, -6.0)
    
    def forward(self, x, sample=True):
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor [B, in_features]
            sample: If True, sample weights; if False, use mean weights
        
        Returns:
            output: Linear transformation [B, out_features]
            kl_div: KL divergence between posterior and prior
        """
        if sample and self.training:
            # Reparameterization trick: w = mu + sigma * epsilon
            sigma_w = torch.exp(0.5 * self.log_var_w)
            epsilon_w = torch.randn_like(sigma_w)
            w = self.mu_w + sigma_w * epsilon_w
            
            sigma_b = torch.exp(0.5 * self.log_var_b)
            epsilon_b = torch.randn_like(sigma_b)
            b = self.mu_b + sigma_b * epsilon_b
        else:
            # Use mean weights (deterministic)
            w = self.mu_w
            b = self.mu_b
        
        # Linear transformation
        output = F.linear(x, w, b)
        
        # Compute KL divergence: KL(q(w) || p(w))
        # For Gaussian: KL = 0.5 * (sigma^2/prior_sigma^2 + mu^2/prior_sigma^2 - 1 - log(sigma^2/prior_sigma^2))
        kl_div = self._kl_divergence()
        
        return output, kl_div
    
    def _kl_divergence(self):
        """
        Compute KL divergence between posterior q(w) and prior p(w).
        
        KL(q || p) = 0.5 * sum(sigma_q^2/sigma_p^2 + mu_q^2/sigma_p^2 - 1 - log(sigma_q^2/sigma_p^2))
        """
        prior_var = self.prior_std ** 2
        
        # KL for weights
        var_w = torch.exp(self.log_var_w)
        kl_w = 0.5 * torch.sum(
            var_w / prior_var + self.mu_w ** 2 / prior_var - 1 - torch.log(var_w / prior_var)
        )
        
        # KL for bias
        var_b = torch.exp(self.log_var_b)
        kl_b = 0.5 * torch.sum(
            var_b / prior_var + self.mu_b ** 2 / prior_var - 1 - torch.log(var_b / prior_var)
        )
        
        return kl_w + kl_b


class BayesianRouter(nn.Module):
    """
    Bayesian Router for MoE with uncertainty estimation.
    
    Two-layer Bayesian neural network that computes expert weights
    with uncertainty quantification.
    
    Args:
        feature_dim: Input feature dimension (band features)
        num_experts: Number of experts per band
        temperature: Temperature for softmax (higher = more uniform)
        prior_std: Prior standard deviation for Bayesian layers
        num_samples_inference: Number of samples for uncertainty estimation during inference
        router_dropout: Dropout rate between layers
    """
    
    def __init__(
        self,
        feature_dim=128,
        num_experts=4,
        temperature=1.0,
        prior_std=1.0,
        num_samples_inference=5,
        router_dropout=0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.num_samples_inference = num_samples_inference
        self.router_dropout = router_dropout
        
        # Two-layer Bayesian network
        self.fc1 = BayesianLinear(feature_dim, feature_dim // 2, prior_std)
        self.fc2 = BayesianLinear(feature_dim // 2, num_experts, prior_std)
        self.dropout = nn.Dropout(router_dropout)
        
        print(f"[BayesianRouter] Initialized: feature_dim={feature_dim}, num_experts={num_experts}, "
              f"temperature={temperature}, prior_std={prior_std}, num_samples_inference={num_samples_inference}")
    
    def forward(self, x, extra_bias=None, return_uncertainty=True):
        """
        Forward pass through Bayesian router.
        
        Args:
            x: Input features [B, feature_dim]
            extra_bias: Optional bias to add to logits [B, num_experts] or [num_experts]
            return_uncertainty: If True, return uncertainty estimates
        
        Returns:
            expert_weights: Softmax weights [B, num_experts]
            router_uncertainty: Uncertainty estimate [B] (if return_uncertainty=True)
            kl_loss: Total KL divergence loss (scalar)
        """
        if self.training:
            # Training: single sample
            logits, kl_loss = self._forward_single(x)
            
            # Apply extra bias (e.g., domain-adaptive bias)
            if extra_bias is not None:
                if extra_bias.dim() == 1:
                    logits = logits + extra_bias.unsqueeze(0)
                else:
                    logits = logits + extra_bias
            
            # Apply temperature and softmax
            logits = logits / self.temperature
            expert_weights = F.softmax(logits, dim=-1)
            
            # Estimate uncertainty from logits entropy
            if return_uncertainty:
                router_uncertainty = self._compute_entropy_uncertainty(expert_weights)
            else:
                router_uncertainty = None
        
        else:
            # Inference: multiple samples for uncertainty estimation
            all_logits = []
            total_kl_loss = 0.0
            
            for _ in range(self.num_samples_inference):
                logits_sample, kl_sample = self._forward_single(x)
                all_logits.append(logits_sample)
                total_kl_loss += kl_sample
            
            # Average KL loss
            kl_loss = total_kl_loss / self.num_samples_inference
            
            # Stack logits: [num_samples, B, num_experts]
            all_logits = torch.stack(all_logits, dim=0)
            
            # Apply extra bias
            if extra_bias is not None:
                if extra_bias.dim() == 1:
                    all_logits = all_logits + extra_bias.unsqueeze(0).unsqueeze(0)
                else:
                    all_logits = all_logits + extra_bias.unsqueeze(0)
            
            # Apply temperature and softmax
            all_logits = all_logits / self.temperature
            all_weights = F.softmax(all_logits, dim=-1)  # [num_samples, B, num_experts]
            
            # Mean weights
            expert_weights = all_weights.mean(dim=0)  # [B, num_experts]
            
            # Estimate uncertainty from sample variance
            if return_uncertainty:
                # Variance across samples
                weight_variance = all_weights.var(dim=0)  # [B, num_experts]
                # Sum variance across experts as overall uncertainty
                router_uncertainty = weight_variance.sum(dim=-1)  # [B]
            else:
                router_uncertainty = None
        
        return expert_weights, router_uncertainty, kl_loss
    
    def _forward_single(self, x):
        """Single forward pass with weight sampling."""
        # First layer
        h, kl1 = self.fc1(x, sample=True)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second layer
        logits, kl2 = self.fc2(h, sample=True)
        
        # Total KL divergence
        kl_loss = kl1 + kl2
        
        return logits, kl_loss
    
    def _compute_entropy_uncertainty(self, probs):
        """
        Compute entropy-based uncertainty from probability distribution.
        
        Args:
            probs: Probability distribution [B, num_experts]
        
        Returns:
            uncertainty: Entropy values [B]
        """
        # Entropy: H = -sum(p * log(p))
        epsilon = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        # Normalize by max entropy (log(num_experts))
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def get_weight_statistics(self):
        """
        Get statistics of Bayesian weights for monitoring.
        
        Returns:
            dict with weight means and stds
        """
        stats = {}
        
        # Layer 1
        stats['fc1_weight_mean'] = self.fc1.mu_w.mean().item()
        stats['fc1_weight_std'] = torch.exp(0.5 * self.fc1.log_var_w).mean().item()
        stats['fc1_bias_mean'] = self.fc1.mu_b.mean().item()
        stats['fc1_bias_std'] = torch.exp(0.5 * self.fc1.log_var_b).mean().item()
        
        # Layer 2
        stats['fc2_weight_mean'] = self.fc2.mu_w.mean().item()
        stats['fc2_weight_std'] = torch.exp(0.5 * self.fc2.log_var_w).mean().item()
        stats['fc2_bias_mean'] = self.fc2.mu_b.mean().item()
        stats['fc2_bias_std'] = torch.exp(0.5 * self.fc2.log_var_b).mean().item()
        
        return stats

