import torch
import torch.nn.functional as F


def _build_physics_attribute_prototypes(prototypes, seed=2026, shuffled=False):
    """Build fixed, interpretable fault-attribute prototypes in model space."""
    labels = sorted(prototypes.keys())
    if labels != [0, 1, 2, 3]:
        raise ValueError("Physics attribute prototypes require labels [0, 1, 2, 3]")

    # Axes: healthy, impulsive, periodic, amplitude modulation, shaft sidebands,
    # fixed defect, rotating defect, irregular impacts, broadband resonance,
    # narrowband harmonics, cage modulation, elevated energy.
    attrs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.3, 0.5, 0.3, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ], dtype=torch.float32)

    ref = prototypes[labels[0]]
    dim = int(ref.numel())
    if dim < attrs.size(1):
        raise ValueError(f"Prototype dimension {dim} is smaller than attribute dimension")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    basis_raw = torch.randn((dim, attrs.size(1)), generator=generator)
    basis, _ = torch.linalg.qr(basis_raw, mode="reduced")
    projected = F.normalize(attrs @ basis.T, dim=1).to(device=ref.device, dtype=ref.dtype)

    if shuffled:
        # Deterministic derangement: every class receives an incorrect prototype.
        shift = int(seed) % (len(labels) - 1) + 1
        perm = torch.roll(torch.arange(len(labels)), shifts=-shift).tolist()
        projected = projected[perm]

    return {label: projected[i] for i, label in enumerate(labels)}

def apply_semantic_proto_ablation(prototypes, mode=None, seed=None):
    """Apply deterministic prototype controls for supplementary runs."""
    import os

    mode = str(mode or os.environ.get("SEMANTIC_PROTO_MODE", "correct")).strip().lower()
    if mode in {"", "correct", "text", "semantic"} or len(prototypes) < 2:
        return prototypes
    seed = int(seed if seed is not None else os.environ.get("SEMANTIC_PROTO_SEED", "2026"))
    labels = sorted(prototypes.keys())
    if mode in {"attribute", "physics_attribute", "pgsp"}:
        print(f"[SemanticProtoAblation] mode=attribute seed={seed} labels={labels}")
        return _build_physics_attribute_prototypes(prototypes, seed=seed, shuffled=False)
    if mode in {"attribute_shuffled", "physics_attribute_shuffled", "pgsp_shuffled"}:
        print(f"[SemanticProtoAblation] mode=attribute_shuffled seed={seed} labels={labels}")
        return _build_physics_attribute_prototypes(prototypes, seed=seed, shuffled=True)
    if mode == "shuffled":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        perm = torch.randperm(len(labels), generator=generator).tolist()
        out = {label: prototypes[labels[src]].clone() for label, src in zip(labels, perm)}
        print(f"[SemanticProtoAblation] mode=shuffled seed={seed} labels={labels}")
        return out
    if mode == "random":
        ref = prototypes[labels[0]]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn((len(labels), ref.numel()), generator=generator, dtype=ref.dtype)
        noise = F.normalize(noise, dim=1).to(ref.device)
        out = {label: noise[i] for i, label in enumerate(labels)}
        print(f"[SemanticProtoAblation] mode=random seed={seed} labels={labels}")
        return out
    raise ValueError(
        f"Unknown SEMANTIC_PROTO_MODE={mode!r}; use correct, shuffled, random, "
        "attribute, or attribute_shuffled"
    )


def build_text_prototypes(text_encoder, label_to_prompts, device):
    """
    label_to_prompts: Dict[int, List[str]]
    """
    text_encoder.eval()
    prototypes = {}

    with torch.no_grad():
        for label, prompt_list in label_to_prompts.items():
            tokens = prompt_list
            emb = text_encoder(tokens)      # [K, D]
            emb = F.normalize(emb, dim=1)
            prototypes[label] = emb.mean(dim=0)
            prototypes[label] = F.normalize(prototypes[label], dim=0)

    return apply_semantic_proto_ablation(prototypes)
