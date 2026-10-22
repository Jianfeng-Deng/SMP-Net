"""
Expert Performance Tracker for Domain-Adaptive Routing

Tracks expert performance per domain using Bayesian statistics and provides
domain-adaptive routing bias for exploration-exploitation trade-off.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertPerformanceStats:
    """
    Bayesian statistics for expert performance in a specific domain.
    
    Maintains Gaussian posterior over expert performance:
    - mu: Mean performance
    - var: Variance (uncertainty)
    - n_samples: Number of observations
    
    Uses Bayesian updating similar to BayesianPrototype.
    
    Args:
        num_experts: Number of experts to track
        prior_mean: Prior mean performance (0.5 = neutral)
        prior_var: Prior variance (higher = more uncertain)
        device: Device for tensors
    """
    
    def __init__(self, num_experts, prior_mean=0.5, prior_var=0.1, device='cpu'):
        self.num_experts = num_experts
        self.device = device
        
        # Initialize posterior as prior
        self.mu = torch.ones(num_experts, device=device) * prior_mean
        self.var = torch.ones(num_experts, device=device) * prior_var
        self.n_samples = torch.zeros(num_experts, device=device)
        
        # Store prior for reset
        self.prior_mean = prior_mean
        self.prior_var = prior_var
    
    def update(self, expert_scores):
        """
        Bayesian update of expert performance statistics.
        
        Uses conjugate Gaussian update:
        posterior_precision = prior_precision + data_precision
        posterior_mean = (prior_precision * prior_mean + data_precision * data_mean) / posterior_precision
        
        Args:
            expert_scores: Performance scores [num_experts] (higher = better)
        """
        expert_scores = expert_scores.to(self.device)
        
        # Current precision (inverse variance)
        precision = 1.0 / (self.var + 1e-10)
        
        # Update for each expert
        # Assume single observation per expert (n=1, so data_var = prior_var for simplicity)
        data_precision = 1.0 / (self.prior_var + 1e-10)
        
        # Updated precision
        new_precision = precision + data_precision
        
        # Updated mean
        self.mu = (precision * self.mu + data_precision * expert_scores) / new_precision
        
        # Updated variance
        self.var = 1.0 / new_precision
        
        # Increment sample count
        self.n_samples += 1
    
    def get_ucb_scores(self, exploration_bonus=0.1):
        """
        Get Upper Confidence Bound (UCB) scores for exploration-exploitation.
        
        UCB = mean + exploration_bonus * sqrt(var / n)
        
        Args:
            exploration_bonus: Controls exploration (higher = more exploration)
        
        Returns:
            ucb_scores: [num_experts]
        """
        # UCB with sample-dependent exploration
        exploration_term = exploration_bonus * torch.sqrt(self.var / (self.n_samples + 1))
        ucb_scores = self.mu + exploration_term
        return ucb_scores
    
    def get_thompson_sample(self):
        """
        Sample from posterior distribution (Thompson sampling).
        
        Returns:
            samples: [num_experts] sampled performance scores
        """
        # Sample from N(mu, var)
        std = torch.sqrt(self.var)
        samples = torch.randn(self.num_experts, device=self.device) * std + self.mu
        return samples
    
    def get_mean_scores(self):
        """Get mean performance scores."""
        return self.mu.clone()
    
    def reset(self):
        """Reset statistics to prior."""
        self.mu = torch.ones(self.num_experts, device=self.device) * self.prior_mean
        self.var = torch.ones(self.num_experts, device=self.device) * self.prior_var
        self.n_samples = torch.zeros(self.num_experts, device=self.device)
    
    def get_statistics(self):
        """Get statistics for monitoring."""
        return {
            'mean': self.mu.cpu().numpy().tolist(),
            'var': self.var.cpu().numpy().tolist(),
            'n_samples': self.n_samples.cpu().numpy().tolist(),
            'ucb': self.get_ucb_scores().cpu().numpy().tolist()
        }


class ExpertPerformanceTracker(nn.Module):
    """
    Tracks expert performance per domain and band for adaptive routing.
    
    Maintains Bayesian statistics for each (domain, band, expert) combination
    and provides domain-adaptive routing bias.
    
    Args:
        num_domains: Number of domains
        num_bands: Number of frequency bands
        num_experts_per_band: Number of experts per band
        prior_mean: Prior mean performance
        prior_var: Prior variance
        exploration_bonus: Exploration bonus for UCB
        use_thompson_sampling: Whether to use Thompson sampling
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_domains,
        num_bands,
        num_experts_per_band,
        prior_mean=0.5,
        prior_var=0.1,
        exploration_bonus=0.1,
        use_thompson_sampling=False,
        device='cpu'
    ):
        super().__init__()
        self.num_domains = num_domains
        self.num_bands = num_bands
        self.num_experts_per_band = num_experts_per_band
        self.exploration_bonus = exploration_bonus
        self.use_thompson_sampling = use_thompson_sampling
        self.device = device
        
        # Create stats for each domain and band
        # stats[domain_id][band_id] = ExpertPerformanceStats
        self.stats = []
        for _ in range(num_domains):
            domain_stats = []
            for _ in range(num_bands):
                band_stats = ExpertPerformanceStats(
                    num_experts_per_band,
                    prior_mean=prior_mean,
                    prior_var=prior_var,
                    device=device
                )
                domain_stats.append(band_stats)
            self.stats.append(domain_stats)
        
        print(f"[ExpertPerformanceTracker] Initialized: "
              f"domains={num_domains}, bands={num_bands}, experts={num_experts_per_band}, "
              f"exploration_bonus={exploration_bonus}, thompson_sampling={use_thompson_sampling}")
    
    def get_domain_bias(self, domain_ids, band_idx, mode='ucb'):
        """
        Get domain-adaptive routing bias.
        
        Args:
            domain_ids: Domain indices [B]
            band_idx: Band index (scalar)
            mode: Bias mode ('mean', 'ucb', 'thompson')
        
        Returns:
            bias: Routing bias [B, num_experts]
        """
        B = domain_ids.size(0)
        bias = torch.zeros(B, self.num_experts_per_band, device=self.device)
        
        # Get bias for each domain in batch
        unique_domains = domain_ids.unique()
        for domain_id in unique_domains:
            mask = (domain_ids == domain_id)
            stats = self.stats[domain_id.item()][band_idx]
            
            if mode == 'mean':
                domain_bias = stats.get_mean_scores()
            elif mode == 'ucb':
                domain_bias = stats.get_ucb_scores(self.exploration_bonus)
            elif mode == 'thompson':
                domain_bias = stats.get_thompson_sample()
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            bias[mask] = domain_bias.unsqueeze(0)
        
        return bias
    
    def update_from_batch(
        self,
        domain_ids,
        band_idx,
        expert_weights,
        expert_outputs,
        batch_loss
    ):
        """
        Update expert performance statistics from batch results.
        
        Args:
            domain_ids: Domain indices [B]
            band_idx: Band index (scalar)
            expert_weights: Expert weights used [B, num_experts]
            expert_outputs: Expert outputs [B, num_experts, D]
            batch_loss: Batch loss value (scalar, lower = better)
        """
        # Convert loss to performance score (lower loss = higher performance)
        # Use negative exponential: score = exp(-loss)
        performance_score = math.exp(-batch_loss) if batch_loss < 10 else 0.0
        
        # Update stats for each domain
        unique_domains = domain_ids.unique()
        for domain_id in unique_domains:
            mask = (domain_ids == domain_id)
            
            # Get expert weights for this domain
            domain_expert_weights = expert_weights[mask]  # [N, num_experts]
            
            # Aggregate expert usage: mean weight per expert
            expert_usage = domain_expert_weights.mean(dim=0)  # [num_experts]
            
            # Score each expert based on usage and performance
            # Experts that were used more get credited/blamed more
            expert_scores = expert_usage * performance_score
            
            # Normalize scores to [0, 1]
            expert_scores = torch.sigmoid(expert_scores)
            
            # Update stats
            stats = self.stats[domain_id.item()][band_idx]
            stats.update(expert_scores)
    
    def update_from_accuracy(
        self,
        domain_ids,
        band_idx,
        expert_weights,
        predictions,
        targets
    ):
        """
        Update expert performance based on prediction accuracy.
        
        Args:
            domain_ids: Domain indices [B]
            band_idx: Band index (scalar)
            expert_weights: Expert weights [B, num_experts]
            predictions: Predicted labels [B]
            targets: True labels [B]
        """
        # Compute per-sample accuracy
        correct = (predictions == targets).float()  # [B]
        
        # Update stats for each domain
        unique_domains = domain_ids.unique()
        for domain_id in unique_domains:
            mask = (domain_ids == domain_id)
            
            # Get expert weights and accuracy for this domain
            domain_expert_weights = expert_weights[mask]  # [N, num_experts]
            domain_correct = correct[mask]  # [N]
            
            # Weight accuracy by expert usage
            # If an expert was used and prediction was correct, credit it
            # expert_scores[e] = mean(weight[e] * correct)
            expert_scores = (domain_expert_weights.T * domain_correct).mean(dim=1)  # [num_experts]
            
            # Normalize to [0, 1]
            expert_scores = torch.sigmoid(expert_scores * 2 - 1)  # Center around 0.5
            
            # Update stats
            stats = self.stats[domain_id.item()][band_idx]
            stats.update(expert_scores)
    
    def get_all_statistics(self):
        """
        Get statistics for all domains and bands.
        
        Returns:
            dict: Nested dict with statistics
        """
        all_stats = {}
        for domain_id in range(self.num_domains):
            domain_stats = {}
            for band_idx in range(self.num_bands):
                stats = self.stats[domain_id][band_idx]
                domain_stats[f'band_{band_idx}'] = stats.get_statistics()
            all_stats[f'domain_{domain_id}'] = domain_stats
        return all_stats
    
    def get_domain_expert_preferences(self, domain_id, mode='mean'):
        """
        Get expert preferences for a specific domain across all bands.
        
        Args:
            domain_id: Domain index
            mode: 'mean', 'ucb', or 'thompson'
        
        Returns:
            preferences: [num_bands, num_experts] expert scores
        """
        preferences = []
        for band_idx in range(self.num_bands):
            stats = self.stats[domain_id][band_idx]
            
            if mode == 'mean':
                scores = stats.get_mean_scores()
            elif mode == 'ucb':
                scores = stats.get_ucb_scores(self.exploration_bonus)
            elif mode == 'thompson':
                scores = stats.get_thompson_sample()
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            preferences.append(scores)
        
        preferences = torch.stack(preferences, dim=0)  # [num_bands, num_experts]
        return preferences
    
    def reset_domain(self, domain_id):
        """Reset statistics for a specific domain."""
        for band_idx in range(self.num_bands):
            self.stats[domain_id][band_idx].reset()
    
    def reset_all(self):
        """Reset all statistics."""
        for domain_id in range(self.num_domains):
            self.reset_domain(domain_id)
    
    def to(self, device):
        """Move tracker to device."""
        self.device = device
        for domain_id in range(self.num_domains):
            for band_idx in range(self.num_bands):
                stats = self.stats[domain_id][band_idx]
                stats.mu = stats.mu.to(device)
                stats.var = stats.var.to(device)
                stats.n_samples = stats.n_samples.to(device)
                stats.device = device
        return self
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Save tracker state.
        
        Compatible with PyTorch's state_dict() signature for nn.Module.
        """
        state = {
            'num_domains': self.num_domains,
            'num_bands': self.num_bands,
            'num_experts_per_band': self.num_experts_per_band,
            'exploration_bonus': self.exploration_bonus,
            'use_thompson_sampling': self.use_thompson_sampling
        }
        
        # Save stats for each domain and band
        stats_data = []
        for domain_id in range(self.num_domains):
            domain_data = []
            for band_idx in range(self.num_bands):
                stats = self.stats[domain_id][band_idx]
                band_data = {
                    'mu': stats.mu.cpu(),
                    'var': stats.var.cpu(),
                    'n_samples': stats.n_samples.cpu(),
                    'prior_mean': stats.prior_mean,
                    'prior_var': stats.prior_var
                }
                domain_data.append(band_data)
            stats_data.append(domain_data)
        
        state['stats'] = stats_data
        
        # Handle PyTorch's state_dict() protocol
        if destination is not None:
            # If destination is provided, update it (PyTorch protocol)
            for key, value in state.items():
                destination[prefix + key] = value
            return destination
        else:
            # Otherwise, return the state dict directly
            return state
    
    def load_state_dict(self, state):
        """Load tracker state."""
        # Verify dimensions match
        assert state['num_domains'] == self.num_domains
        assert state['num_bands'] == self.num_bands
        assert state['num_experts_per_band'] == self.num_experts_per_band
        
        # Load stats
        stats_data = state['stats']
        for domain_id in range(self.num_domains):
            for band_idx in range(self.num_bands):
                band_data = stats_data[domain_id][band_idx]
                stats = self.stats[domain_id][band_idx]
                
                stats.mu = band_data['mu'].to(self.device)
                stats.var = band_data['var'].to(self.device)
                stats.n_samples = band_data['n_samples'].to(self.device)
                stats.prior_mean = band_data['prior_mean']
                stats.prior_var = band_data['prior_var']
        
        print(f"[ExpertPerformanceTracker] Loaded state from checkpoint")

