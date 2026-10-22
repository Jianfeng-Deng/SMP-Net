import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ablation_gates import apply_residual_gate
from utils.ablation_gating import get_gate_bank


class GatedAdditiveFusion(nn.Module):
    """
    Production fusion: gated additive mix or mean-residual path, with optional encoder detach
    for dual-contract training (FUSION_FORWARD_DETACH_ENCODER_INPUTS).

    Default mix: z_fusion = LayerNorm(0.5*(sig + spec) + tanh(gate)*(sig - spec))
    Optional (use_mean_residual): z_fusion = LayerNorm(mean + residual_net(sig,spec,dom)) — no tanh gate.
    Optional bounded spectral residual:
        z_fusion = LayerNorm(sig + alpha*sigmoid(gate)*(spec - sig))
    where alpha is constrained to [0, 1]. This keeps the signal view as the
    backbone and prevents negative modality weights.

    Norm: default detached L2 (ATT437); optional F.normalize full backward (use_standard_unit_vector_norm).

    split_arch_mode: bypass — z_fusion from z_signal only (experimental / SplitArch experiments).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        domain_embed_dim: int = 64,
        num_domains: int = 5,
        num_classes: int = 4,
        dropout: float = 0.1,
        gate_hidden_dim: int = 128,
        split_arch_mode: bool = False,
        use_mean_residual: bool = False,
        use_standard_unit_vector_norm: bool = False,
        use_cross_attention: bool = False,
        use_token_pool_only: bool = False,
        num_attention_heads: int = 8,
        bounded_spec_residual_alpha: float = 0.0,
        bounded_spec_residual_enabled: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.domain_embed_dim = domain_embed_dim
        self.num_classes = num_classes
        self.split_arch_mode = split_arch_mode
        self.use_mean_residual = use_mean_residual
        self.use_standard_unit_vector_norm = use_standard_unit_vector_norm
        self.use_cross_attention = bool(use_cross_attention)
        self.use_multi_token = self.use_cross_attention or bool(use_token_pool_only)
        self.bounded_spec_residual_alpha = float(bounded_spec_residual_alpha)
        self.bounded_spec_residual_enabled = bool(
            bounded_spec_residual_enabled or self.bounded_spec_residual_alpha > 0.0
        )
        if not 0.0 <= self.bounded_spec_residual_alpha <= 1.0:
            raise ValueError("bounded_spec_residual_alpha must be in [0, 1]")
        if self.use_cross_attention:
            if embed_dim % num_attention_heads != 0:
                raise ValueError("embed_dim must be divisible by num_attention_heads")
            self.signal_queries_spec = nn.MultiheadAttention(embed_dim, num_attention_heads, dropout=dropout, batch_first=True)
            self.spec_queries_signal = nn.MultiheadAttention(embed_dim, num_attention_heads, dropout=dropout, batch_first=True)
            self.cross_attention_scale = nn.Parameter(torch.full((2,), 0.1))
        else:
            self.signal_queries_spec = None
            self.spec_queries_signal = None
            self.register_parameter("cross_attention_scale", None)

        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)
        nn.init.zeros_(self.domain_embedding.weight)
        self._trained_domain_ids = set()

        proj_in = embed_dim + domain_embed_dim
        self.signal_proj = nn.Sequential(
            nn.Linear(proj_in, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.spec_proj = nn.Sequential(
            nn.Linear(proj_in, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        gate_in = embed_dim * 2 + domain_embed_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, embed_dim),
        )
        if use_mean_residual:
            self.residual_net = nn.Sequential(
                nn.Linear(gate_in, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, embed_dim),
            )
        else:
            self.residual_net = None

        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_drop = nn.Dropout(dropout)

        self._last_gate_info = None
        self._last_uncertainties = None
        self._last_reg_loss = None
        self._last_signal_pooled = None
        self._last_spec_pooled = None
        self._last_signal_entropy_gate = None
        self._last_spec_residual_alpha = None
        self._last_gate_values = None
        self._gate_for_reg = None
        self._last_z_normed = None

        self._init_weights()
        if use_mean_residual and self.residual_net is not None:
            nn.init.zeros_(self.residual_net[-1].weight)
            nn.init.zeros_(self.residual_net[-1].bias)
        if self.bounded_spec_residual_enabled:
            # Start from a neutral, non-saturated gate (sigmoid(0)=0.5).
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.gate_net[-1].bias, 0.0)

    def forward(
        self,
        z_signal,
        z_spec,
        domain_ids=None,
        text_prototypes=None,
        class_ids=None,
        class_probs=None,
        signal_token_mask=None,
        spec_token_mask=None,
        signal_logits=None,
        spec_logits=None,
        spec_moe_disagreement=None,
        dgl_detach_inputs=False,
    ):
        B = z_signal.size(0)
        device = z_signal.device
        # Bidirectional cross-attention; detach before attention for the isolated variant.
        cross_attention_active = self.use_cross_attention and z_signal.dim() == 3 and z_spec.dim() == 3
        if cross_attention_active:
            if dgl_detach_inputs and self.training:
                z_signal = z_signal.detach()
                z_spec = z_spec.detach()
            sig_key_padding = ~signal_token_mask.bool() if signal_token_mask is not None else None
            spec_key_padding = ~spec_token_mask.bool() if spec_token_mask is not None else None
            sig_context, _ = self.signal_queries_spec(z_signal, z_spec, z_spec, key_padding_mask=spec_key_padding, need_weights=False)
            spec_context, _ = self.spec_queries_signal(z_spec, z_signal, z_signal, key_padding_mask=sig_key_padding, need_weights=False)
            z_signal = z_signal + torch.tanh(self.cross_attention_scale[0]) * sig_context
            z_spec = z_spec + torch.tanh(self.cross_attention_scale[1]) * spec_context

        if z_signal.dim() == 3:
            if signal_token_mask is not None:
                m = signal_token_mask.to(dtype=z_signal.dtype).unsqueeze(-1)
                z_signal = (z_signal * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
            else:
                z_signal = z_signal.mean(dim=1)
        if z_spec.dim() == 3:
            if spec_token_mask is not None:
                m = spec_token_mask.to(dtype=z_spec.dtype).unsqueeze(-1)
                z_spec = (z_spec * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
            else:
                z_spec = z_spec.mean(dim=1)

        if self.split_arch_mode:
            z_normed = self.out_norm(z_signal)
            self._last_z_normed = z_normed
            self._last_gate_values = None
            self._gate_for_reg = None
            self._last_signal_pooled = z_signal.detach()
            self._last_spec_pooled = z_spec.detach()
            z_pre_l2 = self.out_drop(z_normed)
            _norm = z_pre_l2.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z_fusion_norm = z_pre_l2 / _norm
            try:
                z_fusion_norm._pre_l2_feature = z_pre_l2
            except Exception:
                pass
            return z_fusion_norm

        _gate_bank = get_gate_bank(self)
        if dgl_detach_inputs and self.training and not cross_attention_active:
            z_signal = apply_residual_gate(
                _gate_bank, "gidc_dual_contract", z_signal.detach(), z_signal,
            )
            z_spec = apply_residual_gate(
                _gate_bank, "gidc_dual_contract", z_spec.detach(), z_spec,
            )

        if domain_ids is not None:
            if self.training:
                for _did in domain_ids.unique().tolist():
                    self._trained_domain_ids.add(int(_did))
            d_emb = self.domain_embedding(domain_ids)
        else:
            d_emb = torch.zeros(B, self.domain_embed_dim, device=device)

        sig_in = torch.cat([z_signal, d_emb], dim=-1)
        spe_in = torch.cat([z_spec, d_emb], dim=-1)

        sig_proj = self.signal_proj(sig_in)
        spe_proj = self.spec_proj(spe_in)

        self._last_signal_pooled = sig_proj
        self._last_spec_pooled = spe_proj

        gate_input = torch.cat([sig_proj, spe_proj, d_emb], dim=-1)
        z_mean = 0.5 * (sig_proj + spe_proj)
        if self.use_mean_residual and self.residual_net is not None:
            _z_res = z_mean + self.residual_net(gate_input)
            z_fused = apply_residual_gate(_gate_bank, "gidc_mean_residual", _z_res, z_mean)
            self._gate_for_reg = None
            self._last_gate_values = None
        else:
            gate_raw = self.gate_net(gate_input)
            if self.bounded_spec_residual_enabled:
                gate = torch.sigmoid(gate_raw)
                # Regularize the pre-sigmoid value. Unlike a penalty on a
                # saturated tanh output, this retains a useful gradient.
                self._gate_for_reg = gate_raw
                self._last_gate_values = gate.detach()
                z_fused = sig_proj + self.bounded_spec_residual_alpha * gate * (
                    spe_proj - sig_proj
                )
            else:
                gate = torch.tanh(gate_raw)
                self._gate_for_reg = gate
                self._last_gate_values = gate.detach()
                z_diff = sig_proj - spe_proj
                z_fused = z_mean + gate * z_diff
        z_normed = self.out_norm(z_fused)

        self._last_z_normed = z_normed
        z_pre_l2 = self.out_drop(z_normed)
        if self.use_standard_unit_vector_norm:
            z_fusion_norm = F.normalize(z_pre_l2, dim=-1, eps=1e-6)
        else:
            _norm = z_pre_l2.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z_fusion_norm = z_pre_l2 / _norm

        try:
            z_fusion_norm._pre_l2_feature = z_pre_l2
        except Exception:
            pass

        return z_fusion_norm

    def get_gate_info(self):
        return {
            'gate_info': self._last_gate_info,
            'uncertainties': self._last_uncertainties,
            'signal_entropy_gate': self._last_signal_entropy_gate,
            'spec_residual_alpha': self._last_spec_residual_alpha,
            'gate_values': self._last_gate_values,
        }

    def get_cross_modal_align_loss(self):
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        z = torch.zeros((), device=dev, dtype=dt)
        if self.split_arch_mode:
            return z
        if self._last_signal_pooled is None or self._last_spec_pooled is None:
            return z
        h_sig_n = F.normalize(self._last_signal_pooled, dim=-1)
        h_spec_n = F.normalize(self._last_spec_pooled, dim=-1)
        return 1.0 - (h_sig_n * h_spec_n).sum(dim=-1).mean()

    def get_reg_loss(self):
        return self._last_reg_loss
