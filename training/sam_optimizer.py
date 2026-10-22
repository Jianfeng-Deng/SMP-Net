"""
SAM (Sharpness-Aware Minimization) optimizer wrapper.
Ref: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    Wraps any base optimizer with SAM's two-step update:
      1) first_step: ascend along the sharpest direction (maximize loss in ε-ball)
      2) second_step: descend from the perturbed point (minimize worst-case loss)
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore_weights(self):
        """Restore original weights saved during first_step (without stepping)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore weights + base optimizer step. For non-AMP only."""
        self.restore_weights()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
