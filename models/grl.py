import torch


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Gradient Reversal Layer (GRL) for DANN.
    Forward: identity
    Backward: multiply gradients by -alpha
    """
    if alpha is None:
        alpha = 1.0
    try:
        a = float(alpha)
    except Exception:
        a = 1.0
    if a != a:  # NaN guard
        a = 1.0
    return _GradReverseFn.apply(x, a)

