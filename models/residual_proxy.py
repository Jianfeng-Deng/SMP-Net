import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualProxyBank(nn.Module):
    """
    Residual proxy bank for each fault type.
    Computes residual logits and a residual vector r that is injected into the diagnostic branch.
    """
    def __init__(self, num_faults=3, num_proxies_per_fault=8, embed_dim=512, temperature=0.1):
        super().__init__()
        self.num_faults = num_faults
        self.num_proxies = num_proxies_per_fault
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Proxies per fault: [num_faults, num_proxies, embed_dim]
        self.proxies = nn.Parameter(torch.randn(num_faults, num_proxies_per_fault, embed_dim) * 0.02)

    def forward(self, z_cls, fault_mask=None):
        """
        Args:
            z_cls: [B, D]
            fault_mask: [B, num_faults] boolean mask of valid faults per sample (optional)
        Returns:
            logits_residual: [B, num_faults]
            residual_vector: [B, D]
        """
        z = F.normalize(z_cls, dim=-1)  # [B, D]
        proxies = F.normalize(self.proxies, dim=-1)  # [F, P, D]

        # Similarity between z and each proxy: [B, F, P]
        sim = torch.einsum("bd,fpd->bfp", z, proxies) / self.temperature

        # Use a large finite negative value instead of -inf to avoid NaN in softmax
        mask_value = -1e4  # Large finite negative, avoids non-finite guards

        if fault_mask is not None:
            sim = sim.masked_fill(~fault_mask.unsqueeze(-1), mask_value)

        # Aggregate per fault over proxies
        logits_residual = torch.logsumexp(sim, dim=-1)  # [B, F]

        if fault_mask is not None:
            # Masked faults get mask_value (ensures softmax probability ≈ 0, but remains finite)
            logits_residual = logits_residual.masked_fill(~fault_mask, mask_value)

        proxy_attn = F.softmax(sim, dim=-1)  # [B, F, P] (now safe: no -inf → no NaN)
        # Weighted sum of proxy vectors for each fault: [B, F, D]
        r_fault = torch.einsum("bfp,fpd->bfd", proxy_attn, proxies)

        if fault_mask is not None:
            # Explicitly zero out masked faults' vector contributions to prevent NaN propagation
            r_fault = r_fault.masked_fill(~fault_mask.unsqueeze(-1), 0.0)

        # Combine faults with softmax over logits_residual
        fault_probs = F.softmax(logits_residual, dim=-1)  # [B, F] (masked faults ≈ 0)
        residual_vector = torch.einsum("bf,bfd->bd", fault_probs, r_fault)  # [B, D]

        return logits_residual, residual_vector
