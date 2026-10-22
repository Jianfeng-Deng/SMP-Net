"""
ATT465: DomainBN — Domain-Specific Batch Normalization.

Maintains per-domain running statistics (mean, var) while sharing learnable
affine parameters (gamma, beta). At inference, unknown domains use the
uniform average of all domain statistics.

SOTA basis:
- Seo et al. (2020) "Learning to Optimize Domain-Specific Normalization"
- Segu et al. (2023) "Batch Normalization Embeddings for DG"
"""

import torch
import torch.nn as nn


class DomainBN(nn.Module):

    def __init__(self, num_features: int, num_domains: int = 5,
                 momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.momentum = momentum
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        for d in range(num_domains):
            self.register_buffer(f"running_mean_{d}", torch.zeros(num_features))
            self.register_buffer(f"running_var_{d}", torch.ones(num_features))
            self.register_buffer(f"num_batches_{d}", torch.tensor(0, dtype=torch.long))

    def _get_running_stats(self, domain_id: int):
        mean = getattr(self, f"running_mean_{domain_id}")
        var = getattr(self, f"running_var_{domain_id}")
        return mean, var

    def _update_running_stats(self, domain_id: int, batch_mean: torch.Tensor,
                              batch_var: torch.Tensor):
        mean = getattr(self, f"running_mean_{domain_id}")
        var = getattr(self, f"running_var_{domain_id}")
        nb = getattr(self, f"num_batches_{domain_id}")
        mean.mul_(1 - self.momentum).add_(batch_mean.detach(), alpha=self.momentum)
        var.mul_(1 - self.momentum).add_(batch_var.detach(), alpha=self.momentum)
        nb.add_(1)

    def _average_running_stats(self):
        """Uniform average of all domain statistics for unknown domains."""
        means, vars_ = [], []
        total_nb = 0
        for d in range(self.num_domains):
            nb = getattr(self, f"num_batches_{d}").item()
            if nb > 0:
                means.append(getattr(self, f"running_mean_{d}"))
                vars_.append(getattr(self, f"running_var_{d}"))
                total_nb += 1
        if total_nb == 0:
            return torch.zeros(self.num_features, device=self.weight.device), \
                   torch.ones(self.num_features, device=self.weight.device)
        return torch.stack(means).mean(0), torch.stack(vars_).mean(0)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps
            domain_ids: [B] integer domain IDs (0..num_domains-1), or None
        """
        if x.dim() != 4:
            return x

        if self.training:
            if domain_ids is not None and torch.is_tensor(domain_ids):
                return self._forward_domain_aware_train(x, domain_ids)
            else:
                return self._forward_shared_train(x)
        else:
            if domain_ids is not None and torch.is_tensor(domain_ids):
                return self._forward_domain_aware_eval(x, domain_ids)
            else:
                return self._forward_shared_eval(x)

    def _forward_shared_train(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: standard BN when no domain_ids available."""
        return nn.functional.batch_norm(
            x, None, None, self.weight, self.bias,
            training=True, momentum=self.momentum, eps=self.eps,
        )

    def _forward_domain_aware_train(self, x: torch.Tensor,
                                     domain_ids: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = torch.empty_like(x)
        unique_domains = domain_ids.unique().tolist()

        for d in unique_domains:
            mask = (domain_ids == d)
            x_d = x[mask]
            if x_d.size(0) == 0:
                continue
            batch_mean = x_d.mean(dim=(0, 2, 3))
            batch_var = x_d.var(dim=(0, 2, 3), unbiased=False)
            d_int = int(d)
            if 0 <= d_int < self.num_domains:
                self._update_running_stats(d_int, batch_mean, batch_var)
            x_normed = (x_d - batch_mean[None, :, None, None]) / \
                       (batch_var[None, :, None, None] + self.eps).sqrt()
            out[mask] = x_normed * self.weight[None, :, None, None] + \
                        self.bias[None, :, None, None]

        return out

    def _forward_domain_aware_eval(self, x: torch.Tensor,
                                    domain_ids: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = torch.empty_like(x)
        unique_domains = domain_ids.unique().tolist()
        avg_mean, avg_var = None, None

        for d in unique_domains:
            mask = (domain_ids == d)
            x_d = x[mask]
            if x_d.size(0) == 0:
                continue
            d_int = int(d)
            if 0 <= d_int < self.num_domains:
                nb = getattr(self, f"num_batches_{d_int}").item()
                if nb > 0:
                    mean, var = self._get_running_stats(d_int)
                else:
                    if avg_mean is None:
                        avg_mean, avg_var = self._average_running_stats()
                    mean, var = avg_mean, avg_var
            else:
                if avg_mean is None:
                    avg_mean, avg_var = self._average_running_stats()
                mean, var = avg_mean, avg_var

            x_normed = (x_d - mean[None, :, None, None]) / \
                       (var[None, :, None, None] + self.eps).sqrt()
            out[mask] = x_normed * self.weight[None, :, None, None] + \
                        self.bias[None, :, None, None]

        return out

    def _forward_shared_eval(self, x: torch.Tensor) -> torch.Tensor:
        avg_mean, avg_var = self._average_running_stats()
        x_normed = (x - avg_mean[None, :, None, None]) / \
                   (avg_var[None, :, None, None] + self.eps).sqrt()
        return x_normed * self.weight[None, :, None, None] + \
               self.bias[None, :, None, None]
