"""
Domain-conditional Low-Rank Adaptation (LoRA) for cross-domain fine-tuning.

Implements LoRA adapters that are domain-specific, allowing efficient
fine-tuning on target domains with minimal parameter overhead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainLoRA(nn.Module):
    """
    Domain-conditional LoRA adapter.
    
    For each domain, learns low-rank matrices A and B such that:
    W' = W + A @ B (where rank(A) = rank(B) = r)
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        rank: LoRA rank (r in the formula above)
        num_domains: Number of domains (e.g., 4 for CWRU/PU/HUST/HIT)
        alpha: Scaling factor for LoRA weights (default: rank)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(self, in_features, out_features, rank, num_domains, alpha=None, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_domains = num_domains
        self.alpha = alpha if alpha is not None else rank
        self.dropout = dropout
        
        # LoRA matrices A and B for each domain
        # A: [num_domains, out_features, rank] (or [num_domains, rank, out_features] depending on convention)
        # B: [num_domains, rank, in_features]
        # Common convention: W' = W + (B @ A) * (alpha / rank), where B @ A gives [out_features, in_features]
        self.lora_A = nn.Parameter(torch.zeros(num_domains, rank, in_features))  # [num_domains, r, in]
        self.lora_B = nn.Parameter(torch.zeros(num_domains, out_features, rank))  # [num_domains, out, r]
        
        # Initialize A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x, domain_ids):
        """
        Apply domain-specific LoRA adaptation.
        
        Args:
            x: [B, in_features] input features
            domain_ids: [B] integer domain IDs (0, 1, 2, ...)
        
        Returns:
            adapted_x: [B, out_features] adapted features
        """
        batch_size = x.size(0)
        device = x.device
        
        if domain_ids is None:
            return torch.zeros(batch_size, self.out_features, device=device, dtype=x.dtype)

        # Apply dropout if enabled
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        
        # Collect LoRA outputs per domain
        output = torch.zeros(batch_size, self.out_features, device=device, dtype=x.dtype)
        
        # Process each unique domain in the batch
        unique_domains = domain_ids.unique()
        for domain_id in unique_domains:
            mask = (domain_ids == domain_id)
            x_domain = x[mask]  # [B_domain, in_features]
            
            # Get domain-specific LoRA matrices
            A = self.lora_A[domain_id]  # [r, in]
            B = self.lora_B[domain_id]  # [out, r]
            
            # Compute LoRA adaptation: (B @ A) @ x^T, then scale by alpha/rank
            # B @ A: [out, r] @ [r, in] = [out, in]
            # (B @ A) @ x_domain^T: [out, in] @ [in, B_domain] = [out, B_domain]
            lora_delta = (B @ A) @ x_domain.t()  # [out, B_domain]
            lora_delta = lora_delta.t()  # [B_domain, out]
            lora_delta = lora_delta * (self.alpha / self.rank)
            
            output[mask] = lora_delta.to(output.dtype)
        
        return output


class DomainLoRAWrapper(nn.Module):
    """
    Wrapper to easily add domain LoRA to existing linear layers.
    
    Wraps an existing nn.Linear layer and adds domain LoRA adapters.
    During forward, applies base layer then LoRA adaptation.
    """
    
    def __init__(self, base_layer, rank, num_domains, alpha=None, dropout=0.0, 
                 freeze_base=True):
        """
        Args:
            base_layer: Existing nn.Linear layer to wrap
            rank: LoRA rank
            num_domains: Number of domains
            alpha: LoRA scaling factor
            dropout: Dropout probability
            freeze_base: If True, freeze base layer weights (only LoRA trainable)
        """
        super().__init__()
        self.base_layer = base_layer
        self.freeze_base = freeze_base
        
        # Freeze base layer if requested
        if freeze_base:
            for param in self.base_layer.parameters():
                param.requires_grad = False
        
        # Create LoRA adapter
        self.lora = DomainLoRA(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            num_domains=num_domains,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x, domain_ids=None):
        """
        Apply base layer then LoRA adaptation.
        
        Args:
            x: [B, in_features] input features
            domain_ids: [B] integer domain IDs (optional, defaults to None = no adaptation)
        
        Returns:
            output: [B, out_features] output features
        """
        # Apply base layer
        base_output = self.base_layer(x)
        
        # Apply LoRA adaptation if domain_ids provided
        if domain_ids is not None:
            lora_output = self.lora(x, domain_ids)
            return base_output + lora_output
        else:
            return base_output


# Helper function to create domain LoRA adapters
def apply_domain_lora_to_layer(layer, rank, num_domains, alpha=None, dropout=0.0, 
                                freeze_base=True):
    """
    Apply domain LoRA to an existing linear layer.
    
    Returns a DomainLoRAWrapper that wraps the original layer.
    """
    return DomainLoRAWrapper(layer, rank, num_domains, alpha, dropout, freeze_base)

