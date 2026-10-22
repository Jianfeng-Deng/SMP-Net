"""
Simplex Equiangular Tight Frame (ETF) prototype generation.

Generates C prototypes in d-dimensional space such that:
  - All prototypes have unit L2 norm
  - All pairwise cosine similarities equal -1/(C-1)
  - This is the optimal geometry for classification (Neural Collapse, Papyan et al. PNAS 2020)

Reference:
  - Papyan, Han, Donoho. "Prevalence of neural collapse..." PNAS 2020.
  - Yang et al. "We Don't Need a Classifier: ..." CVPR 2022.
"""

import torch
import torch.nn.functional as F


def generate_simplex_etf(
    num_classes: int,
    embed_dim: int,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate Simplex ETF prototypes.

    The resulting prototypes satisfy:
      proto @ proto.T ≈ I*(C/(C-1)) - ones*(1/(C-1))
    i.e., diagonal = 1.0, off-diagonal = -1/(C-1).

    Args:
        num_classes: C, number of classes
        embed_dim: d, embedding dimension (must be >= num_classes - 1)
        seed: random seed for reproducibility
        device: target device

    Returns:
        prototypes: [C, d] tensor, each row is a unit-norm prototype
    """
    assert embed_dim >= num_classes - 1, (
        f"embed_dim ({embed_dim}) must be >= num_classes-1 ({num_classes - 1})"
    )
    C = num_classes

    if C == 1:
        g = torch.Generator(device="cpu").manual_seed(seed)
        v = torch.randn(1, embed_dim, generator=g)
        return F.normalize(v, dim=-1).to(device)

    # Step 1: Construct the centered identity in C-dim
    #   M = I_C - (1/C) * ones(C,C)
    #   M has rank C-1 with eigenvalues: (C-1)/C repeated (C-1) times, 0 once
    I_C = torch.eye(C, dtype=torch.float64)
    M = I_C - torch.ones(C, C, dtype=torch.float64) / C

    # Step 2: SVD to get the C-1 non-zero directions
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    # Take first C-1 columns (non-zero singular values)
    base = U[:, : C - 1] * S[: C - 1].sqrt().unsqueeze(0)  # [C, C-1]

    # Step 3: Random orthogonal projection from (C-1) to embed_dim
    g = torch.Generator(device="cpu").manual_seed(seed)
    random_matrix = torch.randn(embed_dim, C - 1, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(random_matrix)  # [d, C-1] orthonormal columns
    proj = Q[:, : C - 1]  # [d, C-1]

    # Step 4: Project base vectors into d-dim and normalize
    prototypes = (base @ proj.T).float()  # [C, d]
    prototypes = F.normalize(prototypes, dim=-1)

    # Verification
    cos_matrix = prototypes @ prototypes.T
    expected_off_diag = -1.0 / (C - 1)
    off_diag_mask = ~torch.eye(C, dtype=torch.bool)
    actual_off_diag = cos_matrix[off_diag_mask].mean().item()
    if abs(actual_off_diag - expected_off_diag) > 0.01:
        print(
            f"[SimplexETF] WARNING: off-diagonal cos mean={actual_off_diag:.4f}, "
            f"expected={expected_off_diag:.4f}"
        )

    return prototypes.to(device)


def print_etf_stats(prototypes: torch.Tensor, class_names=None):
    """Print cosine similarity matrix for ETF prototypes."""
    C = prototypes.size(0)
    cos_mat = prototypes @ prototypes.T
    names = class_names or [str(i) for i in range(C)]

    header = "         " + "  ".join(f"{n:>8s}" for n in names)
    print(f"[SimplexETF] Cosine similarity matrix ({C} classes, dim={prototypes.size(1)}):")
    print(header)
    for i in range(C):
        row = f"  {names[i]:>6s} " + "  ".join(f"{cos_mat[i, j]:>8.4f}" for j in range(C))
        print(row)

    off_diag_mask = ~torch.eye(C, dtype=torch.bool, device=prototypes.device)
    off_diag_vals = cos_mat[off_diag_mask]
    print(
        f"[SimplexETF] off-diag: mean={off_diag_vals.mean():.4f}, "
        f"min={off_diag_vals.min():.4f}, max={off_diag_vals.max():.4f}, "
        f"expected={-1.0/(C-1):.4f}"
    )
