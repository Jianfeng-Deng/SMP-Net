"""
Domain Residual Module for Cross-Domain Adaptation

Implements FiLM (Feature-wise Linear Modulation) and Residual Adapter
to compensate for domain-specific feature distribution shifts.
"""

import torch
import torch.nn as nn


class DomainFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for domain adaptation.
    
    For each domain, learns scale (gamma) and shift (beta) parameters
    to modulate features: output = gamma * x + beta
    
    Args:
        num_domains: Number of domains (e.g., 3 for CWRU/PU/HUST)
        num_features: Number of feature channels to modulate
        init_scale: Initial scale for gamma (default 0.0 for gradual activation)
    """
    
    def __init__(self, num_domains, num_features, init_scale=0.0):
        super().__init__()
        self.num_domains = num_domains
        self.num_features = num_features
        
        # Learnable scale and shift for each domain
        # Initialize gamma close to 0 (controlled by init_scale)
        # Initialize beta to 0
        self.gamma = nn.Parameter(torch.ones(num_domains, num_features) * init_scale)
        self.beta = nn.Parameter(torch.zeros(num_domains, num_features))
        
        # Gate parameter to control overall residual strength (starts from 0)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, domain_ids):
        """
        Apply domain-specific FiLM modulation.
        
        Args:
            x: Input features [B, num_features] or [B, num_features, *]
            domain_ids: Domain indices [B] (0, 1, 2, ...)
            
        Returns:
            Modulated features with same shape as x
        """
        # Get domain-specific parameters
        gamma = self.gamma[domain_ids]  # [B, num_features]
        beta = self.beta[domain_ids]    # [B, num_features]
        
        # Apply gating (sigmoid to keep in [0, 1])
        gate_weight = torch.sigmoid(self.gate)
        
        # Reshape for broadcasting if x has spatial dimensions
        if x.dim() > 2:
            # x: [B, C, ...], gamma/beta: [B, C]
            # Reshape to [B, C, 1, 1, ...] for broadcasting
            shape = [x.size(0), x.size(1)] + [1] * (x.dim() - 2)
            gamma = gamma.view(shape)
            beta = beta.view(shape)
        
        # Apply FiLM: x + gate * (gamma * x + beta)
        # This allows gradual activation from identity (gate=0)
        return x + gate_weight * (gamma * x + beta)
    
    def get_gate_value(self):
        """Get current gate value for monitoring."""
        return torch.sigmoid(self.gate).item()


class DomainResidualAdapter(nn.Module):
    """
    Residual Adapter for domain adaptation.
    
    For each domain, learns a small MLP adapter:
    output = x + gate(domain) * Adapter(x)
    
    Args:
        num_domains: Number of domains
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension of adapter MLP
        init_gate_value: Initial gate value (default 0.0)
    """
    
    def __init__(self, num_domains, input_dim, hidden_dim=128, init_gate_value=0.0):
        super().__init__()
        self.num_domains = num_domains
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Adapter MLP for each domain
        # Use ModuleList to properly register parameters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_domains)
        ])
        
        # Initialize adapter weights to small values
        for adapter in self.adapters:
            for m in adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        
        # Learnable gate for each domain (starts from init_gate_value)
        self.gates = nn.Parameter(torch.ones(num_domains) * init_gate_value)
        
    def forward(self, x, domain_ids):
        """
        Apply domain-specific residual adapter.
        
        Args:
            x: Input features [B, input_dim]
            domain_ids: Domain indices [B]
            
        Returns:
            Adapted features [B, input_dim]
        """
        batch_size = x.size(0)
        output = torch.zeros_like(x)
        
        # Process each unique domain in the batch
        unique_domains = domain_ids.unique()
        for domain_id in unique_domains:
            mask = (domain_ids == domain_id)
            x_domain = x[mask]
            
            # Apply adapter
            adapter_out = self.adapters[domain_id](x_domain)
            
            # Apply gate (sigmoid to keep in [0, 1])
            gate_weight = torch.sigmoid(self.gates[domain_id])
            
            # Residual connection
            output[mask] = x_domain + gate_weight * adapter_out
        
        return output
    
    def get_gate_values(self):
        """Get current gate values for all domains."""
        return torch.sigmoid(self.gates).detach().cpu().numpy()


class DomainResidualWrapper(nn.Module):
    """
    Wrapper to easily add domain residual to existing encoders.
    
    Args:
        encoder: Base encoder module
        residual_type: 'film' or 'adapter'
        num_domains: Number of domains
        feature_dim: Feature dimension (for FiLM: channel dim, for Adapter: output dim)
        hidden_dim: Hidden dimension (only for adapter)
        init_scale: Initial scale/gate value
        apply_after_layer: Which layer to apply residual after (e.g., 'conv3', 'fc')
    """
    
    def __init__(self, encoder, residual_type='film', num_domains=3, 
                 feature_dim=128, hidden_dim=128, init_scale=0.0,
                 apply_after_layer='conv3'):
        super().__init__()
        self.encoder = encoder
        self.residual_type = residual_type
        self.apply_after_layer = apply_after_layer
        
        # Create domain residual module
        if residual_type == 'film':
            self.domain_residual = DomainFiLM(
                num_domains=num_domains,
                num_features=feature_dim,
                init_scale=init_scale
            )
        elif residual_type == 'adapter':
            self.domain_residual = DomainResidualAdapter(
                num_domains=num_domains,
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                init_gate_value=init_scale
            )
        else:
            raise ValueError(f"Unknown residual_type: {residual_type}")
    
    def forward(self, *args, domain_ids=None, **kwargs):
        """
        Forward pass with domain residual.
        
        Note: This is a simplified wrapper. For production use,
        you should modify the encoder's forward method directly
        to insert domain residual at the desired location.
        """
        # This is a placeholder - actual implementation should be
        # integrated into the encoder's forward method
        raise NotImplementedError(
            "DomainResidualWrapper is a template. "
            "Please integrate domain residual directly into encoder's forward method."
        )


def create_domain_residual(residual_type='film', num_domains=3, 
                          feature_dim=128, hidden_dim=128, init_scale=0.0):
    """
    Factory function to create domain residual module.
    
    Args:
        residual_type: 'film' or 'adapter'
        num_domains: Number of domains
        feature_dim: Feature dimension
        hidden_dim: Hidden dimension (for adapter)
        init_scale: Initial scale/gate value
        
    Returns:
        Domain residual module
    """
    if residual_type == 'film':
        return DomainFiLM(
            num_domains=num_domains,
            num_features=feature_dim,
            init_scale=init_scale
        )
    elif residual_type == 'adapter':
        return DomainResidualAdapter(
            num_domains=num_domains,
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            init_gate_value=init_scale
        )
    else:
        raise ValueError(f"Unknown residual_type: {residual_type}")

