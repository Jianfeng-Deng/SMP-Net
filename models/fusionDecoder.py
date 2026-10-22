import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.residual_proxy import ResidualProxyBank
from models.domain_discriminator import DomainDiscriminator
from models.grl import grad_reverse
try:
    from config import Config
except Exception:
    Config = None


class FusionDecoder(nn.Module):
    """
    Cosine-based classifier for fault diagnosis.

    Supports a hierarchical head:
      - Stage A: normal-vs-fault (2-way) [shared across domains]
      - Stage B: fault-type [domain-aware: HUST 3-way, HIT 2-way]

    For backward compatibility, `forward()` returns 4-class logits by default.
    """
    def __init__(
        self,
        embed_dim,
        num_classes,
        temperature=0.07,
        use_cache=True,
        hierarchical=True,
        use_domain_aware_fault_head=False,
        num_domains=5,
        use_residual_proxies=False,
        residual_alpha=0.5,
        residual_temperature=0.1,
        residual_num_proxies=8,
        use_prototype_head=True,
        # (S1) multi-prototype diagnosis head
        use_multi_prototype_head=True,
        multi_prototype_k=1,
        multi_prototype_init_std=0.01,
        # (D2) Prototype + linear residual diagnosis head
        use_diag_linear_residual=True,
        diag_linear_scale_init=0.0,
        # (D2+) sample-adaptive beta(x) gate
        use_diag_linear_gate=True,
        diag_linear_gate_init_bias=-8.0,
        # Prompt/Prototype tuning (CoOp/CoCoOp-like) in embedding space
        use_prompt_tuning=True,
        use_domain_prompt_tuning=True,
        prompt_residual_init_std=0.01
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_cache = use_cache  # 使用torch.compile()时应设为False
        self.hierarchical = hierarchical
        self.use_domain_aware_fault_head = use_domain_aware_fault_head
        self.num_domains = num_domains
        self.use_residual_proxies = use_residual_proxies
        self.residual_alpha = residual_alpha
        self.residual_temperature = residual_temperature
        self.residual_num_proxies = residual_num_proxies
        self.use_prototype_head = use_prototype_head
        self.use_multi_prototype_head = bool(use_multi_prototype_head)
        self.multi_prototype_k = int(multi_prototype_k) if multi_prototype_k is not None else 1
        self.use_diag_linear_residual = bool(use_diag_linear_residual)
        self.use_diag_linear_gate = bool(use_diag_linear_gate)
        self.use_prompt_tuning = use_prompt_tuning
        self.use_domain_prompt_tuning = use_domain_prompt_tuning
        try:
            import config as cfg
            self.use_dual_space_decouple = bool(getattr(cfg, "USE_DUAL_SPACE_DECOUPLE", False))
            self.dual_space_metric_detach = bool(getattr(cfg, "DUAL_SPACE_METRIC_DETACH", False))
            self.use_text_prompt_tuning = bool(getattr(cfg, "USE_TEXT_PROMPT_TUNING", False))
            self.text_prompt_tuning_init_std = float(getattr(cfg, "TEXT_PROMPT_TUNING_INIT_STD", 0.01))
            self.text_prompt_tuning_scale_init = float(getattr(cfg, "TEXT_PROMPT_TUNING_SCALE_INIT", 1.0))
        except Exception:
            self.use_dual_space_decouple = False
            self.dual_space_metric_detach = False
            self.use_text_prompt_tuning = False
            self.text_prompt_tuning_init_std = 0.01
            self.text_prompt_tuning_scale_init = 1.0
        # Use a large finite negative number instead of -inf for missing classes (avoids "non-finite" guards).
        self._logit_mask_value = -1e4

        # ===== (Fault-type, structural) Split heads for Rolling =====
        # Two binary heads:
        #   - Inner vs Outer (2-way)
        #   - Rolling vs Non-rolling (1 logit)
        # They are used to build a unified [inner, outer, rolling] fault-logit vector.
        try:
            import config as cfg
            self.use_split_fault_heads = bool(getattr(cfg, "USE_SPLIT_FAULT_HEADS", True))
        except Exception:
            self.use_split_fault_heads =False

        # Rolling head gradient isolation (reduce IO/rolling interference)
        try:
            import config as cfg
            self.use_roll_isolated_adapter = bool(getattr(cfg, "USE_ROLL_ISOLATED_ADAPTER", False))
            roll_dim = int(getattr(cfg, "ROLL_ADAPTER_DIM", max(32, embed_dim // 4)))
            self.roll_adapter_detach = bool(getattr(cfg, "ROLLING_ADAPTER_DETACH", False))
        except Exception:
            self.use_roll_isolated_adapter = False
            roll_dim = max(32, embed_dim // 4)
            self.roll_adapter_detach = False
        if self.use_roll_isolated_adapter:
            self.roll_adapter = nn.Sequential(
                nn.Linear(embed_dim, roll_dim),
                nn.LayerNorm(roll_dim),
                nn.ReLU(),
                nn.Linear(roll_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.roll_adapter = None

        # ===== (MTFD) Multi-Task Fault Decomposition: Independent Binary Heads =====
        # Replace unstable 4-class Softmax with 3 independent binary classifiers.
        try:
            import config as cfg
            self.use_mtfd = bool(getattr(cfg, "USE_MTFD", False))
            self.mtfd_nf_weight = float(getattr(cfg, "MTFD_NF_LOSS_WEIGHT", 1.0))
            self.mtfd_io_weight = float(getattr(cfg, "MTFD_IO_LOSS_WEIGHT", 1.0))
            self.mtfd_rolling_weight = float(getattr(cfg, "MTFD_ROLLING_LOSS_WEIGHT", 2.0))
        except Exception:
            self.use_mtfd = False
            self.mtfd_nf_weight = 1.0
            self.mtfd_io_weight = 1.0
            self.mtfd_rolling_weight = 2.0

        # ===== (A2) Missing-class masking policy =====
        # For heterogeneous label spaces (e.g., PU/HIT have no rolling), DO NOT hard-mask logits by default.
        # Instead, prefer partial-label loss (masked CE) at the loss level to avoid "unlearnable samples".
        try:
            import config as cfg
            self.mask_missing_classes_in_logits = bool(getattr(cfg, "MASK_MISSING_CLASSES_IN_LOGITS", False))
        except Exception:
            self.mask_missing_classes_in_logits = False

        # Attempt 8: Diagnostic Decision Mode
        try:
            import config as cfg
            self.diag_decision_mode = getattr(cfg, "DIAG_DECISION_MODE", "normal")
        except:
            self.diag_decision_mode = "normal"

        # ===== (S-SEM) Semantic adapter (z_sem) for decoupling semantic alignment from diagnosis space =====
        # z_sem = stopgrad(z) + sem_adapter(z); used only for prototype/CLIP alignment losses.
        try:
            import config as cfg
            self.use_sem_adapter = bool(getattr(cfg, "USE_SEM_ADAPTER", True))
            sem_h = int(getattr(cfg, "SEM_ADAPTER_HIDDEN", 256))
            sem_init_zero = bool(getattr(cfg, "SEM_ADAPTER_INIT_ZERO", True))
        except Exception:
            self.use_sem_adapter, sem_h, sem_init_zero = True, 256, True
        self.sem_adapter = nn.Sequential(
            nn.Linear(embed_dim, sem_h),
            nn.ReLU(),
            nn.Linear(sem_h, embed_dim),
        )
        # ATT239(S1): linear SupCon projection to preserve stronger gradient flow.
        self.supcon_proj = nn.Linear(embed_dim, embed_dim)
        if sem_init_zero:
            try:
                nn.init.zeros_(self.sem_adapter[-1].weight)
                nn.init.zeros_(self.sem_adapter[-1].bias)
            except Exception:
                pass
        for p in self.sem_adapter.parameters():
            p.requires_grad = False

        # ===== Dual-space decoupling heads =====
        # z_cls: diagnosis logits / supervised path
        # z_metric: semantic alignment / prototype path
        self.proj_cls = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.use_metric_proj_residual_mlp = False
        self.metric_proj_residual_scale = 1.0
        try:
            import config as cfg
            self.use_metric_proj_residual_mlp = bool(
                getattr(cfg, "FUSION_METRIC_PROJ_RESIDUAL_MLP_ENABLE", False)
            )
            _hid_mul = float(getattr(cfg, "FUSION_METRIC_PROJ_HIDDEN_MULT", 2.0))
            _hid_mul = max(1.0, min(8.0, _hid_mul))
            metric_hid = int(round(float(embed_dim) * _hid_mul))
        except Exception:
            metric_hid = int(embed_dim * 2)
            self.use_metric_proj_residual_mlp = False

        if self.use_metric_proj_residual_mlp:
            self.proj_metric = nn.Sequential(
                nn.Linear(embed_dim, metric_hid),
                nn.GELU(),
                nn.Linear(metric_hid, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.proj_metric = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        try:
            nn.init.xavier_uniform_(self.proj_cls[0].weight)
            nn.init.zeros_(self.proj_cls[0].bias)
            if self.use_metric_proj_residual_mlp:
                nn.init.xavier_uniform_(self.proj_metric[0].weight)
                nn.init.zeros_(self.proj_metric[0].bias)
                # Residual-MLP starts as near-identity to avoid abrupt geometry drift.
                nn.init.zeros_(self.proj_metric[2].weight)
                nn.init.zeros_(self.proj_metric[2].bias)
            else:
                nn.init.xavier_uniform_(self.proj_metric[0].weight)
                nn.init.zeros_(self.proj_metric[0].bias)
        except Exception:
            pass

        # ===== Learnable text-prompt offsets (class-wise) =====
        # This branch tunes text prototypes without unfreezing the text encoder backbone.
        self.text_prompt_offsets = nn.Parameter(torch.zeros(int(num_classes), int(embed_dim), dtype=torch.float32))
        self.text_prompt_scale = nn.Parameter(torch.tensor(float(self.text_prompt_tuning_scale_init), dtype=torch.float32))
        try:
            nn.init.normal_(self.text_prompt_offsets, mean=0.0, std=float(self.text_prompt_tuning_init_std))
        except Exception:
            pass
        if not bool(self.use_text_prompt_tuning):
            self.text_prompt_offsets.requires_grad = False
            self.text_prompt_scale.requires_grad = False

        # ===== (S-SEM2) Source-domain EMA visual prototypes =====
        # These prototypes are updated from source training embeddings (z_sem) and can be mixed with text prototypes
        # for more robust semantic scoring under domain shift.
        try:
            import config as cfg
            self.use_visual_prototypes = bool(getattr(cfg, "USE_VISUAL_PROTOTYPES", True))
            self.visual_proto_momentum = float(getattr(cfg, "VISUAL_PROTO_MOMENTUM", 0.99))
        except Exception:
            self.use_visual_prototypes = True
            self.visual_proto_momentum = 0.99
        # Buffers: [C,D] and counts [C]
        self.register_buffer("visual_prototypes", torch.zeros(int(num_classes), int(embed_dim), dtype=torch.float32))
        self.register_buffer("visual_proto_counts", torch.zeros(int(num_classes), dtype=torch.float32))
        self.register_buffer("visual_proto_initialized", torch.zeros(1, dtype=torch.uint8))

        # ===== (DANN) Domain adversarial branch (GRL + discriminator) =====
        # Enabled/weighted in training loop; should not affect inference.
        try:
            import config as cfg
            self.use_dann = bool(getattr(cfg, "USE_DANN", False))
            self.dann_hidden_dim = int(getattr(cfg, "DANN_HIDDEN_DIM", 256))
            self.dann_dropout = float(getattr(cfg, "DANN_DROPOUT", 0.1))
            self.dann_mode = str(getattr(cfg, "DANN_MODE", "vanilla")).lower()  # "vanilla" | "cdan"
            self.dann_feature = str(getattr(cfg, "DANN_FEATURE", "z_gate")).lower()  # z_gate|z_cls|z_res|z
            self.dann_detach_probs = bool(getattr(cfg, "DANN_DETACH_PROBS", True))
        except Exception:
            self.use_dann = False
            self.dann_hidden_dim = 256
            self.dann_dropout = 0.1
            self.dann_mode = "vanilla"
            self.dann_feature = "z_gate"
            self.dann_detach_probs = True
        dann_in_dim = embed_dim
        if str(self.dann_mode).lower() == "cdan":
            dann_in_dim = int(embed_dim) * int(num_classes)
        self.domain_discriminator = DomainDiscriminator(
            in_dim=dann_in_dim,
            num_domains=num_domains,
            hidden_dim=self.dann_hidden_dim,
            dropout=self.dann_dropout
        )

        # ===== (Rolling-DANN) Domain adversarial on rolling branch =====
        try:
            import config as cfg
            self.use_rolling_dann = bool(getattr(cfg, "USE_ROLLING_DANN", False))
            self.rolling_dann_hidden_dim = int(getattr(cfg, "DANN_HIDDEN_DIM", 256))
            self.rolling_dann_dropout = float(getattr(cfg, "DANN_DROPOUT", 0.1))
        except Exception:
            self.use_rolling_dann = False
            self.rolling_dann_hidden_dim = 256
            self.rolling_dann_dropout = 0.1
        self.roll_domain_discriminator = DomainDiscriminator(
            in_dim=embed_dim,
            num_domains=num_domains,
            hidden_dim=self.rolling_dann_hidden_dim,
            dropout=self.rolling_dann_dropout
        )
        
        # ===== (B) Outer margin auxiliary head =====
        # Only used for an auxiliary margin loss; does not affect inference API.
        self.outer_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if self.use_residual_proxies:
            self.residual_proxy_bank = ResidualProxyBank(
                num_faults=3,
                num_proxies_per_fault=residual_num_proxies,
                embed_dim=embed_dim,
                temperature=residual_temperature
            )
        else:
            self.residual_proxy_bank = None

        # Domain mapping (must stay consistent with training.encode_domain_ids)
        cwru_gk2_id = int(getattr(Config, "CWRU_GK2_DOMAIN_ID", 0)) if Config is not None else 0
        cwru_gk4_id = int(getattr(Config, "CWRU_GK4_DOMAIN_ID", 1)) if Config is not None else 1
        pu_id = int(getattr(Config, "PU_DOMAIN_ID", 2)) if Config is not None else 2
        hust_id = int(getattr(Config, "HUST_DOMAIN_ID", 3)) if Config is not None else 3
        hit_id = int(getattr(Config, "HIT_DOMAIN_ID", 4)) if Config is not None else 4
        seu_id = int(getattr(Config, "SEU_DOMAIN_ID", 5)) if Config is not None else 5
        self.domain_map = {
            "CWRU": cwru_gk2_id,
            "CWRU_GK2": cwru_gk2_id,
            "CWRU_GK4": cwru_gk4_id,
            "PU": pu_id,
            "HUST": hust_id,
            "HIT": hit_id,
            "SEU": seu_id,
        }
        self.domain_map_upper = {str(k).upper(): int(v) for k, v in self.domain_map.items()}
        self.default_domain_id = self.domain_map.get("CWRU_GK2", 0)

        if self.hierarchical:
            # ===== Hybrid nf (global 2-way + per-type nf_k) =====
            # 1) global nf head: normal vs any fault (专门负责“放行故障”，所有域共享)
            # 2) per-type nf_k: normal vs fault_k (域感知：HUST 3-way, HIT 2-way)
            #
            # For per-type, we implement it with a shared normal weight + domain-specific fault-type weights:
            #   logits_nf_multi[k] = logit(fault_k) - logit(normal)
            # bias=True is important for cross-domain calibration:
            # it allows shifting the decision boundary when the target domain has a global score offset.
            self.classifier_nf_global = nn.Linear(embed_dim, 2, bias=True)  # [normal, fault_any] (shared)
            self.classifier_normal = nn.Linear(embed_dim, 1, bias=False)  # normal logit (shared)
            
            # Domain-aware fault heads
            if use_domain_aware_fault_head:
                # HUST: 3 faults {inner, outer, rolling}
                self.classifier_fault_hust = nn.Linear(embed_dim, 3, bias=False)  # [inner, outer, rolling]
                # HIT: 2 faults {inner, outer}
                self.classifier_fault_hit = nn.Linear(embed_dim, 2, bias=False)   # [inner, outer]
                # Backward compatibility: shared fault head for source domains (CWRU, PU)
                # These can still use the shared 3-way head, or we can create domain-specific ones later
                self.classifier_fault_shared = nn.Linear(embed_dim, 3, bias=False)  # [inner, outer, rolling] (fallback)
            else:
                # Legacy: shared fault head for all domains
                self.classifier_fault = nn.Linear(embed_dim, 3, bias=False)   # fault-type logits (1/2/3)

            # (Split-head) Rolling-specific + Inner/Outer head (structural)
            # NOTE: keep bias=False because we use normalized-weight cosine logits in forward.
            if bool(getattr(self, "use_split_fault_heads", False)):
                self.classifier_fault_io = nn.Linear(embed_dim, 2, bias=False)     # [inner, outer]
                try:
                    nn.init.orthogonal_(self.classifier_fault_io.weight, gain=1.0)
                except Exception:
                    pass
                # IMPORTANT: use 2-logit softmax for stability:
                #   logits_roll2 = [non_roll, roll]  => p_roll = softmax(logits_roll2)[:,1]
                # This avoids sigmoid saturation and makes the head explicitly conditional.
                self.classifier_fault_roll = nn.Linear(embed_dim, 2, bias=False)  # [non_roll, roll]
                try:
                    nn.init.orthogonal_(self.classifier_fault_roll.weight, gain=1.0)
                except Exception:
                    pass
                
                # MTFD: Independent NF Binary Head (Normal vs Fault)
                if bool(getattr(self, "use_mtfd", False)):
                    self.mtfd_nf_head = nn.Linear(embed_dim, 2, bias=False)  # [normal, fault]
                    try:
                        nn.init.orthogonal_(self.mtfd_nf_head.weight, gain=1.0)
                    except Exception:
                        pass
                else:
                    self.mtfd_nf_head = None
            else:
                self.mtfd_nf_head = None

            # ===== Decoupled gate branch (z_gate) for stable global nf =====
            # z_gate = stopgrad(z_fusion) + small_mlp(z_fusion)
            # This isolates global nf training from fault-type/融合扰动, preventing
            # the "gate collapse" issue where fault samples are incorrectly gated as normal.
            self.gate_adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.LayerNorm(embed_dim),
            )

            # Optional diagnostic adapter for z_cls (keeps semantic z_fusion less distorted)
            # Exposed via get_cls_embedding(); forward() uses z_cls for logits when enabled.
            self.use_cls_adapter = True
            self.cls_adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
            self._init_flat_classifier()
            self._init_flat_multi_center()

        # ===== (D2) Linear residual diagnosis head (cosine classifier) =====
        # logits_diag = logits_proto + beta * logits_linear
        # beta starts at 0 so the default behavior is unchanged until trained.
        self.diag_linear_head = nn.Linear(embed_dim, num_classes, bias=False)
        self.register_parameter(
            "diag_linear_scale",
            nn.Parameter(torch.tensor(float(diag_linear_scale_init), dtype=torch.float32))
        )
        # Default: keep this head frozen; enable in target-domain supervised finetune for stability.
        for p in self.diag_linear_head.parameters():
            p.requires_grad = False
        self.diag_linear_scale.requires_grad = False

        # ===== (D2+) Sample-adaptive beta(x) gate for mixing proto/linear =====
        # beta_eff(x) = (diag_linear_scale * ramp) * sigmoid(gate(z_res))
        # Gate is initialized to output near 0 so default behavior stays prototype-only.
        self.diag_linear_gate = nn.Sequential(
            nn.Linear(embed_dim, max(32, embed_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(32, embed_dim // 4), 1),
        )
        try:
            nn.init.zeros_(self.diag_linear_gate[-1].weight)
            nn.init.constant_(self.diag_linear_gate[-1].bias, float(diag_linear_gate_init_bias))
        except Exception:
            pass
        for p in self.diag_linear_gate.parameters():
            p.requires_grad = False

        # ===== (S-B) Class-bias prior on diagnosis logits =====
        # A small learnable bias per class to correct systematic logit shifts under domain shift.
        self.register_parameter(
            "diag_class_bias",
            nn.Parameter(torch.zeros(self.num_classes, dtype=torch.float32))
        )
        self.diag_class_bias.requires_grad = False

        # ===== (S1) Multi-prototype extra bank (shared across domains) =====
        # Base prototype comes from text_prototypes (+ prompt tuning residuals).
        # We add (K-1) learnable extra prototypes per class to model multi-mode fault patterns.
        k = int(self.multi_prototype_k)
        if self.use_multi_prototype_head and k > 1:
            self.register_parameter(
                "prototype_extra",
                nn.Parameter(torch.zeros(self.num_classes, k - 1, embed_dim))
            )
            nn.init.normal_(self.prototype_extra, mean=0.0, std=float(multi_prototype_init_std))
            # Default: keep frozen; enable late in training for stability.
            self.prototype_extra.requires_grad = False
            # Default: do NOT use multi-proto during early training (prevents random extra prototypes acting as absorbers).
            self._use_multi_prototype_now = False
        else:
            self.prototype_extra = None
            self._use_multi_prototype_now = False
        # NF calibrator parameters (per-domain scale+shift)
        self.register_parameter("nf_calibrator_scale", nn.Parameter(torch.ones(num_domains)))
        self.register_parameter("nf_calibrator_bias", nn.Parameter(torch.zeros(num_domains)))

        # (1b) When True, z_cls = z.detach() + cls_adapter(z) so fault-type loss does not update backbone.
        try:
            import config as cfg
            self.use_cls_detach_fault_type = bool(getattr(cfg, "USE_CLS_DETACH_FAULT_TYPE", True))
        except Exception:
            self.use_cls_detach_fault_type = True
        # (New scheme 2) Fault-type dedicated representation: z_fault = fault_projection(z.detach()).
        # Only fault-type loss updates fault_projection; NF uses z_gate so representations are decoupled.
        try:
            import config as cfg
            self.use_fault_dedicated_rep = bool(getattr(cfg, "USE_FAULT_DEDICATED_REP", True))
        except Exception:
            self.use_fault_dedicated_rep = True
        # (Attempt 13) Fault-type logit adjustment for long-tail bias.
        try:
            import config as cfg
            self.use_fault_logit_adjust = bool(getattr(cfg, "USE_FAULT_LOGIT_ADJUST", False))
            self.fault_logit_adjust_tau = float(getattr(cfg, "FAULT_LOGIT_ADJUST_TAU", 1.0))
            prior = getattr(cfg, "FAULT_LOGIT_ADJUST_PRIOR", None)
            if prior is None:
                self.fault_logit_adjust_prior = None
            else:
                prior_tensor = torch.tensor(prior, dtype=torch.float32)
                self.fault_logit_adjust_prior = prior_tensor
        except Exception:
            self.use_fault_logit_adjust = False
            self.fault_logit_adjust_tau = 1.0
            self.fault_logit_adjust_prior = None
        # (Attempt 14) Hierarchical fault probabilities (roll gate × IO) instead of logit concat.
        try:
            import config as cfg
            self.use_hierarchical_fault_prob = bool(getattr(cfg, "USE_HIERARCHICAL_FAULT_PROB", False))
        except Exception:
            self.use_hierarchical_fault_prob = False
        if self.use_fault_dedicated_rep:
            self.fault_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.fault_projection = None
        # (3b) Learnable normal reference [D] for pull-normal / push-fault loss (EMA updated).
        self.register_buffer("normal_reference", torch.zeros(embed_dim))
        self.normal_reference = F.normalize(self.normal_reference + 1e-6, dim=-1)

        # ========= Prompt/Prototype tuning residuals =========
        # Shared residual (CoOp-like): trainable on source training (improves prototype separability)
        self.register_parameter(
            "prototype_residual_shared",
            nn.Parameter(torch.zeros(num_classes, embed_dim))
        )
        nn.init.normal_(self.prototype_residual_shared, mean=0.0, std=float(prompt_residual_init_std))
        self.prototype_residual_shared.requires_grad = bool(self.use_prompt_tuning)

        # Domain residual (CoCoOp-like): used mainly in target-domain supervised finetune
        self.register_parameter(
            "prototype_residual_domain",
            nn.Parameter(torch.zeros(num_domains, num_classes, embed_dim))
        )
        nn.init.normal_(self.prototype_residual_domain, mean=0.0, std=float(prompt_residual_init_std))
        self.prototype_residual_domain.requires_grad = False
        
        # 缓存normalized权重（仅在use_cache=True且eval模式下使用）
        # 注意：使用torch.compile()时必须设置use_cache=False以避免CUDA Graphs冲突
        if self.use_cache:
            self._cached = {}
        else:
            self._cached = None  # 占位符，避免 AttributeError

    def set_nf_calibrator_requires_grad(self, requires_grad: bool):
        """Toggle whether the NF calibrator (scale/bias) participates in gradient updates."""
        if hasattr(self, "nf_calibrator_scale"):
            self.nf_calibrator_scale.requires_grad = requires_grad
        if hasattr(self, "nf_calibrator_bias"):
            self.nf_calibrator_bias.requires_grad = requires_grad
    
    def set_prototype_residual_shared_requires_grad(self, requires_grad: bool):
        """Toggle whether the shared prototype residual (CoOp-like) participates in gradient updates."""
        if hasattr(self, "prototype_residual_shared"):
            self.prototype_residual_shared.requires_grad = requires_grad

    def set_prototype_residual_domain_requires_grad(self, requires_grad: bool):
        """Toggle whether the domain prototype residual (CoCoOp-like) participates in gradient updates."""
        if hasattr(self, "prototype_residual_domain"):
            self.prototype_residual_domain.requires_grad = requires_grad

    def set_diag_linear_requires_grad(self, requires_grad: bool):
        """Toggle whether the linear residual diagnosis head participates in gradient updates."""
        if hasattr(self, "diag_linear_head"):
            for p in self.diag_linear_head.parameters():
                p.requires_grad = requires_grad
        if hasattr(self, "diag_linear_scale"):
            self.diag_linear_scale.requires_grad = requires_grad
        if hasattr(self, "diag_linear_gate"):
            for p in self.diag_linear_gate.parameters():
                p.requires_grad = requires_grad

    def set_flat_classifier_requires_grad(self, requires_grad: bool):
        """
        Toggle whether the flat-head cosine classifier (and optional multi-center extras) are trainable.

        Intended for target-domain supervised LP-FT (linear probe) when `hierarchical=False`.
        """
        # Flat head base classifier
        if hasattr(self, "classifier") and self.classifier is not None:
            try:
                for p in self.classifier.parameters():
                    p.requires_grad = requires_grad
            except Exception:
                pass

        # Optional multi-center extras (flat_*_extra are Parameters when enabled)
        for extra_name in ("flat_roll_extra", "flat_inner_extra", "flat_outer_extra"):
            try:
                extra = getattr(self, extra_name, None)
                if extra is not None and hasattr(extra, "requires_grad"):
                    extra.requires_grad = requires_grad
            except Exception:
                pass

        # Optional small class-bias prior
        if hasattr(self, "diag_class_bias") and self.diag_class_bias is not None:
            try:
                self.diag_class_bias.requires_grad = requires_grad
            except Exception:
                pass

    def set_sem_adapter_requires_grad(self, requires_grad: bool):
        """Toggle whether the semantic adapter participates in gradient updates."""
        if hasattr(self, "sem_adapter") and self.sem_adapter is not None:
            for p in self.sem_adapter.parameters():
                p.requires_grad = requires_grad

    def _init_flat_classifier(self):
        """
        Optional ETF initialization for flat 4-class classifier.
        Provides a simplex-like geometry for early separation stability.
        """
        try:
            import config as cfg
            use_etf = bool(getattr(cfg, "USE_ETF_CLASSIFIER_INIT", False))
        except Exception:
            use_etf = False
        if not use_etf:
            return
        if not hasattr(self, "classifier") or self.classifier is None:
            return
        try:
            with torch.no_grad():
                num_classes = int(self.num_classes)
                embed_dim = int(self.classifier.weight.size(1))
                if embed_dim < num_classes:
                    nn.init.orthogonal_(self.classifier.weight, gain=1.0)
                    return
                device = self.classifier.weight.device
                eye = torch.eye(num_classes, device=device, dtype=torch.float32)
                ones = torch.ones((num_classes, num_classes), device=device, dtype=torch.float32)
                simplex = eye - (1.0 / float(num_classes)) * ones  # [C,C]
                w = torch.zeros((num_classes, embed_dim), device=device, dtype=torch.float32)
                w[:, :num_classes] = simplex
                w = F.normalize(w, dim=1)
                self.classifier.weight.copy_(w.to(dtype=self.classifier.weight.dtype))
        except Exception:
            pass

    def fix_classifier_to_etf(self, etf_prototypes):
        """
        ATT377-C: Replace flat classifier weights with ETF prototypes and freeze.
        etf_prototypes: [C, D] tensor — the same ETF prototypes used for PACL/text alignment.
        """
        if not hasattr(self, "classifier") or self.classifier is None:
            return False
        C_etf, D_etf = int(etf_prototypes.size(0)), int(etf_prototypes.size(1))
        C_cls, D_cls = int(self.classifier.weight.size(0)), int(self.classifier.weight.size(1))
        if D_etf != D_cls:
            print(f"[ETF-FIX] dim mismatch: etf={D_etf}, classifier={D_cls}, skip")
            return False
        C_use = min(C_etf, C_cls)
        with torch.no_grad():
            w = F.normalize(etf_prototypes[:C_use].float(), dim=-1)
            self.classifier.weight[:C_use].copy_(w.to(dtype=self.classifier.weight.dtype, device=self.classifier.weight.device))
            self.classifier.weight.requires_grad_(False)
        print(
            f"[ETF-FIX] classifier frozen to ETF: C={C_use}, D={D_cls}, "
            f"weight_norm={float(self.classifier.weight.norm(dim=-1).mean().item()):.4f}"
        )
        return True

    def _init_flat_multi_center(self):
        """
        Optional multi-center expansion for flat diagnosis classes (NCMC-style).
        Adds K-1 extra centers for selected classes and pools by max/logsumexp.
        Notes:
          - We compensate logsumexp with -log(K) to remove the inherent +log(K) bias.
          - This is critical; otherwise multi-center classes can become "absorbers" even when untrained.
        """
        try:
            import config as cfg
            # Per-class switches
            use_mc_roll = bool(getattr(cfg, "USE_FLAT_MULTI_CENTER_ROLLING", False))
            k_roll = int(getattr(cfg, "FLAT_MULTI_CENTER_ROLLING_K", 1))
            use_mc_inner = bool(getattr(cfg, "USE_FLAT_MULTI_CENTER_INNER", False))
            k_inner = int(getattr(cfg, "FLAT_MULTI_CENTER_INNER_K", 1))
            use_mc_outer = bool(getattr(cfg, "USE_FLAT_MULTI_CENTER_OUTER", False))
            k_outer = int(getattr(cfg, "FLAT_MULTI_CENTER_OUTER_K", 1))
            pool = str(getattr(cfg, "FLAT_MULTI_CENTER_POOL", "logsumexp")).lower()
            tau = float(getattr(cfg, "FLAT_MULTI_CENTER_TAU", 0.07))
            init_std = float(getattr(cfg, "FLAT_MULTI_CENTER_INIT_STD", 0.01))
        except Exception:
            use_mc_roll = False
            k_roll = 1
            use_mc_inner = False
            k_inner = 1
            use_mc_outer = False
            k_outer = 1
            pool = "logsumexp"
            tau = 0.07
            init_std = 0.01
        self.use_flat_multi_center_rolling = bool(use_mc_roll and k_roll > 1)
        self.use_flat_multi_center_inner = bool(use_mc_inner and k_inner > 1)
        self.use_flat_multi_center_outer = bool(use_mc_outer and k_outer > 1)
        self.flat_multi_center_pool = pool
        self.flat_multi_center_tau = float(tau)
        # Default to None (so forward can check existence safely)
        self.flat_roll_extra = None
        self.flat_inner_extra = None
        self.flat_outer_extra = None

        try:
            embed_dim = int(self.classifier.weight.size(1))
        except Exception:
            embed_dim = None

        if embed_dim is None:
            return

        # Rolling (class-3)
        if self.use_flat_multi_center_rolling:
            try:
                self.register_parameter(
                    "flat_roll_extra",
                    nn.Parameter(torch.zeros(k_roll - 1, embed_dim, dtype=self.classifier.weight.dtype))
                )
                nn.init.normal_(self.flat_roll_extra, mean=0.0, std=float(init_std))
            except Exception:
                self.flat_roll_extra = None

        # Inner (class-1)
        if self.use_flat_multi_center_inner:
            try:
                self.register_parameter(
                    "flat_inner_extra",
                    nn.Parameter(torch.zeros(k_inner - 1, embed_dim, dtype=self.classifier.weight.dtype))
                )
                nn.init.normal_(self.flat_inner_extra, mean=0.0, std=float(init_std))
            except Exception:
                self.flat_inner_extra = None

        # Outer (class-2)
        if self.use_flat_multi_center_outer:
            try:
                self.register_parameter(
                    "flat_outer_extra",
                    nn.Parameter(torch.zeros(k_outer - 1, embed_dim, dtype=self.classifier.weight.dtype))
                )
                nn.init.normal_(self.flat_outer_extra, mean=0.0, std=float(init_std))
            except Exception:
                self.flat_outer_extra = None

    def get_sem_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """
        z_sem for semantic alignment:
          z_sem = stopgrad(z) + sem_adapter(z)
        """
        z = self.get_metric_embedding(z)
        if not bool(getattr(self, "use_sem_adapter", True)) or not hasattr(self, "sem_adapter") or self.sem_adapter is None:
            return z
        # stopgrad trunk to avoid semantic loss dragging diagnosis space
        z_sem = z.detach() + self.sem_adapter(z)
        z_sem = torch.nan_to_num(z_sem, nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(z_sem, dim=-1)

    @torch.no_grad()
    def update_visual_prototypes(self, z_sem: torch.Tensor, labels: torch.Tensor):
        """
        Update EMA visual prototypes from (source) semantic embeddings.
        This should be called during training on SOURCE domains only.
        """
        if not bool(getattr(self, "use_visual_prototypes", True)):
            return
        if z_sem is None or labels is None:
            return
        if z_sem.ndim != 2:
            return
        if labels.ndim != 1 or labels.size(0) != z_sem.size(0):
            return

        z_sem = F.normalize(z_sem.detach(), dim=-1).to(dtype=self.visual_prototypes.dtype, device=self.visual_prototypes.device)
        labels = labels.detach().to(device=self.visual_prototypes.device)
        C = int(self.visual_prototypes.size(0))

        m = float(getattr(self, "visual_proto_momentum", 0.99))
        m = max(0.0, min(0.9999, m))

        for c in range(C):
            idx = (labels == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            proto = z_sem[idx].mean(dim=0)
            proto = F.normalize(proto, dim=-1)

            if int(self.visual_proto_initialized.item()) == 0 or float(self.visual_proto_counts[c].item()) < 1.0:
                self.visual_prototypes[c] = proto
                self.visual_proto_counts[c] = self.visual_proto_counts[c] + float(idx.numel())
            else:
                self.visual_prototypes[c] = F.normalize(m * self.visual_prototypes[c] + (1.0 - m) * proto, dim=-1)
                self.visual_proto_counts[c] = self.visual_proto_counts[c] + float(idx.numel())

        self.visual_proto_initialized.fill_(1)

    @torch.no_grad()
    def get_visual_prototypes_dict(self) -> dict:
        """
        Return {class_id: prototype[D]} from EMA visual prototypes (on current device).
        """
        out = {}
        if not bool(getattr(self, "use_visual_prototypes", True)):
            return out
        C = int(self.visual_prototypes.size(0))
        for c in range(C):
            if float(self.visual_proto_counts[c].item()) <= 0:
                continue
            out[int(c)] = F.normalize(self.visual_prototypes[c], dim=-1)
        return out

    def set_multi_prototype_requires_grad(self, requires_grad: bool):
        """Toggle whether the extra multi-prototype bank participates in gradient updates."""
        if hasattr(self, "prototype_extra") and self.prototype_extra is not None:
            self.prototype_extra.requires_grad = requires_grad
            # Also control whether multi-proto is active in forward.
            self._use_multi_prototype_now = bool(requires_grad)

    def set_diag_linear_ramp(self, ramp: float):
        """
        Set a multiplicative ramp factor in [0,1] applied to beta during forward:
            beta_effective = beta * ramp
        This provides a stable warmup when enabling the linear residual head late in training.
        """
        try:
            r = float(ramp)
        except Exception:
            r = 1.0
        if r != r:  # NaN guard
            r = 1.0
        self._diag_linear_ramp = max(0.0, min(1.0, r))

    def set_cls_adapter_requires_grad(self, requires_grad: bool):
        """Toggle whether the cls_adapter participates in gradient updates."""
        if hasattr(self, "cls_adapter"):
            for p in self.cls_adapter.parameters():
                p.requires_grad = requires_grad

    def set_diag_class_bias_requires_grad(self, requires_grad: bool):
        """Toggle whether the diagnosis class-bias prior participates in gradient updates."""
        if hasattr(self, "diag_class_bias"):
            self.diag_class_bias.requires_grad = requires_grad
    
    def _get_normalized_weight(self, key, weight_tensor):
        """
        获取normalized权重，根据use_cache决定是否使用缓存机制
        
        - use_cache=False: 直接计算（适用于torch.compile()模式）
        - use_cache=True: 在eval模式下缓存（适用于非编译模式，提升性能）
        """
        # 如果禁用缓存或在训练模式，直接计算
        if not self.use_cache or self.training:
            return F.normalize(weight_tensor, dim=-1)
        
        # Eval模式 + 启用缓存：检查并使用缓存
        # 确保 _cached 已初始化
        if not hasattr(self, "_cached") or self._cached is None:
            self._cached = {}
        
        current_weight = weight_tensor
        
        # 检查是否需要更新缓存
        entry = self._cached.get(key, None)
        if entry is None or not torch.equal(entry["version"], current_weight):
            # 更新缓存
            normalized = F.normalize(current_weight, dim=-1)
            self._cached[key] = {
                "normalized": normalized.clone(),
                "version": current_weight.detach().clone()
            }
        
        return self._cached[key]["normalized"].clone()

    def apply_text_prompt_tuning(self, text_prototypes, detach_output: bool = False):
        """
        Apply learnable class-wise prompt offsets on text prototypes.
        This keeps the text encoder frozen while allowing prototype geometry to adapt.
        """
        if not isinstance(text_prototypes, dict):
            return text_prototypes

        tuned = {}
        for _k, _v in text_prototypes.items():
            try:
                _idx = int(_k)
            except Exception:
                _idx = -1
            if not torch.is_tensor(_v):
                continue
            _proto = _v
            if _proto.dim() != 1:
                _proto = _proto.view(-1)
            _proto = _proto.to(dtype=torch.float32)
            _proto = F.normalize(_proto, dim=-1, eps=1e-6)

            if bool(getattr(self, "use_text_prompt_tuning", False)) and 0 <= _idx < int(self.num_classes):
                _off = self.text_prompt_offsets[_idx].to(device=_proto.device, dtype=_proto.dtype)
                _scale = F.softplus(self.text_prompt_scale).to(device=_proto.device, dtype=_proto.dtype)
                _proto = F.normalize(_proto + _scale * _off, dim=-1, eps=1e-6)

            if detach_output:
                _proto = _proto.detach()
            tuned[_idx] = _proto
        return tuned

    def get_text_prompt_offset_norms(self):
        try:
            _off = self.text_prompt_offsets.detach().float()
            _scale = float(F.softplus(self.text_prompt_scale.detach().float()).item())
            _norm_mean = float(_off.norm(dim=-1).mean().item())
            _norm_max = float(_off.norm(dim=-1).max().item())
            return {
                "scale": _scale,
                "offset_norm_mean": _norm_mean,
                "offset_norm_max": _norm_max,
            }
        except Exception:
            return {
                "scale": 0.0,
                "offset_norm_mean": 0.0,
                "offset_norm_max": 0.0,
            }

    def _resolve_domain_ids(self, domain_ids, domain_names, batch_size, device):
        if domain_ids is not None:
            return domain_ids

        if domain_names is not None:
            resolved = torch.tensor(
                [self.domain_map_upper.get(str(name).upper(), self.default_domain_id) for name in domain_names],
                                    dtype=torch.long, device=device)
            return resolved

        if batch_size > 0:
            return torch.full((batch_size,), int(self.default_domain_id), dtype=torch.long, device=device)
        return None

    def _apply_prompt_tuning(self, proto_mat, domain_ids=None):
        """
        proto_mat: [C, D] base prototypes (on correct device)

        Returns:
          - domain_ids is None: [C, D] adjusted prototypes
          - domain_ids provided: [B, C, D] adjusted prototypes per-sample domain
        """
        if not self.use_prompt_tuning:
            return proto_mat

        # IMPORTANT: do prompt/proto math in fp32 to avoid NaNs during adaptation (esp. HIT supervised finetune)
        orig_dtype = proto_mat.dtype
        proto_fp32 = proto_mat.float()
        shared = self.prototype_residual_shared.float()  # [C, D]

        # (D) Freeze rolling-element residual updates to avoid "all→rolling" absorber.
        # This masks gradients for class index 3 (rolling) in both shared and domain residuals.
        freeze_rolling = True
        try:
            import config as cfg
            freeze_rolling = bool(getattr(cfg, "FREEZE_ROLLING_PROMPT_RESIDUAL", True))
        except Exception:
            freeze_rolling = True
        if freeze_rolling and shared.size(0) >= 4:
            shared = shared.clone()
            shared[3, :] = 0.0

        if domain_ids is None or (not self.use_domain_prompt_tuning):
            out = proto_fp32 + shared
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out.to(dtype=orig_dtype)

        delta_domain = self.prototype_residual_domain[domain_ids].float()  # [B, C, D]

        # Mask rolling (class 3) for domains that don't have it:
        # - HIT: only has {normal, inner, outer}
        # - PU: only has {normal, inner, outer} (no rolling element fault)
        hit_id = self.domain_map.get("HIT", 4)
        pu_id = self.domain_map.get("PU", 2)
        hit_mask = domain_ids == hit_id
        pu_mask = domain_ids == pu_id
        if hit_mask.any() or pu_mask.any():
            delta_domain = delta_domain.clone()
            if hit_mask.any():
                delta_domain[hit_mask, 3, :] = 0.0  # HIT: mask rolling
            if pu_mask.any():
                delta_domain[pu_mask, 3, :] = 0.0   # PU: mask rolling (no rolling element fault)

        # Freeze rolling residual updates globally (all domains)
        if freeze_rolling and delta_domain.size(1) >= 4:
            delta_domain = delta_domain.clone()
            delta_domain[:, 3, :] = 0.0

        B = delta_domain.size(0)
        shared_b = shared.unsqueeze(0).expand(B, -1, -1)
        base_b = proto_fp32.unsqueeze(0).expand(B, -1, -1)
        out = base_b + shared_b + delta_domain
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(dtype=orig_dtype)

    def _build_fault_mask(self, domain_ids, batch_size, device):
        mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        if domain_ids is None:
            return mask
        hit_id = self.domain_map.get("HIT", 4)
        pu_id = self.domain_map.get("PU", 2)
        # Domains without rolling element fault should mask rolling (index 2 in {inner, outer, rolling})
        mask[domain_ids == hit_id, 2] = False  # HIT only has inner/outer
        mask[domain_ids == pu_id, 2] = False   # PU only has inner/outer
        return mask

    def _compute_split_fault_logits(self, z_res: torch.Tensor, domain_ids: torch.Tensor | None):
        """
        Compute split-head fault logits:
          - logits_io:   [B,2]  (inner, outer)
          - logits_roll: [B,2]  (non_roll, roll)
          - logits_fault_raw: [B,3] (inner, outer, rolling) without domain masking
          - logits_fault_masked: [B,3] with rolling masked for domains without rolling (PU/HIT)
        """
        if not bool(getattr(self, "use_split_fault_heads", False)):
            return None, None, None, None, None
        if (not hasattr(self, "classifier_fault_io")) or (not hasattr(self, "classifier_fault_roll")):
            return None, None, None, None, None
        try:
            z_io = z_res
            z_roll = z_res
            if bool(getattr(self, "use_roll_isolated_adapter", False)) and self.roll_adapter is not None:
                # Isolate rolling gradients if configured; otherwise allow gradients to flow into z_res.
                roll_in = z_res.detach() if bool(getattr(self, "roll_adapter_detach", False)) else z_res
                z_roll = F.normalize(z_res + self.roll_adapter(roll_in), dim=-1)
            w_io = self._get_normalized_weight("fault_io", self.classifier_fault_io.weight)        # [2,D]
            w_roll = self._get_normalized_weight("fault_roll", self.classifier_fault_roll.weight)  # [2,D]
            logits_io = torch.matmul(z_io, w_io.t()) / float(self.temperature)        # [B,2]
            logits_roll2 = torch.matmul(z_roll, w_roll.t()) / float(self.temperature)  # [B,2] (non_roll, roll)
            # For legacy/debug: keep a "rolling logit" scalar (roll - non_roll) to visualize separation.
            logits_roll = (logits_roll2[:, 1:2] - logits_roll2[:, 0:1])  # [B,1]
            logits_fault_raw = torch.cat([logits_io, logits_roll], dim=1)            # [B,3]
            logits_fault_raw = torch.nan_to_num(
                logits_fault_raw,
                nan=self._logit_mask_value,
                posinf=1e4,
                neginf=self._logit_mask_value
            ).to(dtype=z_res.dtype)

            logits_fault_masked = logits_fault_raw
            if domain_ids is not None:
                hit_id = self.domain_map.get("HIT", 4)
                pu_id = self.domain_map.get("PU", 2)
                no_roll = (domain_ids == hit_id) | (domain_ids == pu_id)
                if no_roll.any():
                    logits_fault_masked = logits_fault_raw.clone()
                    logits_fault_masked[no_roll, 2] = self._logit_mask_value
            # Return both: logits_roll2 is stored in "logits_fault_roll_raw" for loss/probe;
            # logits_roll (scalar) is stored in "logits_fault_roll" for backward compatibility.
            return logits_fault_masked, logits_io, logits_roll2, logits_fault_raw, z_roll
        except Exception:
            return None, None, None, None, None

    def reset_roll_adapter(self) -> bool:
        if self.roll_adapter is None:
            return False
        try:
            for m in self.roll_adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            return True
        except Exception:
            return False

    def _compute_prototype_diag_logits(self, z_res, domain_ids, text_prototypes):
        """
        Prototype-tied unified 4-class diagnosis logits:
          [normal, inner, outer, rolling]
        For HIT, rolling is masked with a large finite negative logit.
        
        Applies prototype residual (Δproto) if enabled and domain_ids are provided.
        """
        device = z_res.device
        for k in [0, 1, 2, 3]:
            if k not in text_prototypes:
                raise KeyError(f"text_prototypes must contain label {k}. Got keys={sorted(list(text_prototypes.keys()))}.")

        proto_mat = torch.stack([text_prototypes[i] for i in [0, 1, 2, 3]], dim=0).to(device)  # [4, D]

        # Multi-prototype bank: [C, K, D] where K = 1 + (K-1 extra learnable protos)
        # Default K=1 -> identical to old behavior.
        k_total = 1
        extra = None
        if self.use_multi_prototype_head and bool(getattr(self, "_use_multi_prototype_now", False)) and getattr(self, "prototype_extra", None) is not None:
            extra = self.prototype_extra.to(device=device, dtype=proto_mat.dtype)  # [C,K-1,D]
            k_total = 1 + extra.size(1)

        # Apply prompt tuning to base text prototypes
        if domain_ids is not None:
            base = self._apply_prompt_tuning(proto_mat, domain_ids=domain_ids)  # [B,4,D]
        else:
            base = self._apply_prompt_tuning(proto_mat, domain_ids=None)        # [4,D]

        # Build the full prototype set including extra prototypes (extras are shared across domains).
        # NOTE: keep all prototype math in fp32 for stability.
        z_fp32 = F.normalize(z_res.float(), dim=-1, eps=1e-6)
        if extra is None:
            # Old single-prototype path
            if domain_ids is not None:
                base_fp32 = F.normalize(base.float(), dim=-1, eps=1e-6)  # [B,4,D]
                logits4 = torch.bmm(z_fp32.unsqueeze(1), base_fp32.transpose(-1, -2)).squeeze(1) / float(self.temperature)  # [B,4]
            else:
                base_fp32 = F.normalize(base.float(), dim=-1, eps=1e-6)  # [4,D]
                logits4 = torch.matmul(z_fp32, base_fp32.t()) / float(self.temperature)  # [B,4]
            logits4 = torch.nan_to_num(logits4, nan=self._logit_mask_value, posinf=1e4, neginf=self._logit_mask_value).to(dtype=z_res.dtype)
        else:
            # Multi-prototype aggregation (stable): logits_c = max_k sim(z, p_{c,k})/T
            # This prevents "class dropout" under domain shift by allowing multi-mode class structure.
            extra_fp32 = extra.float()  # [C,K-1,D]
            # IMPORTANT: never allow rolling extra prototypes to participate (they easily become an "absorber").
            # We mask them at similarity level (not by zeroing vectors, which breaks normalization).
            disable_roll_extra = True
            try:
                import config as cfg
                disable_roll_extra = bool(getattr(cfg, "MULTI_PROTOTYPE_DISABLE_ROLLING_EXTRA", True))
            except Exception:
                disable_roll_extra = True

            if domain_ids is not None:
                base_fp32 = base.float()  # [B,C,D]
                B = base_fp32.size(0)
                # Expand extras to [B,C,K-1,D]
                extra_b = extra_fp32.unsqueeze(0).expand(B, -1, -1, -1)
                base_b = base_fp32.unsqueeze(2)  # [B,C,1,D]
                protos = torch.cat([base_b, extra_b], dim=2)  # [B,C,K,D]
                protos = F.normalize(protos, dim=-1, eps=1e-6)
                # sim: [B,C,K]
                sim = torch.einsum("bd,bckd->bck", z_fp32, protos)
                if disable_roll_extra and sim.size(1) >= 4 and sim.size(2) >= 2:
                    sim = sim.clone()
                    sim[:, 3, 1:] = -1e4
                logits4 = (sim / float(self.temperature)).max(dim=2).values  # [B,C]
            else:
                base_fp32 = base.float()  # [C,D]
                protos = torch.cat([base_fp32.unsqueeze(1), extra_fp32], dim=1)  # [C,K,D]
                protos = F.normalize(protos, dim=-1, eps=1e-6)
                # sim: [B,C,K]
                sim = torch.einsum("bd,ckd->bck", z_fp32, protos)
                if disable_roll_extra and sim.size(1) >= 4 and sim.size(2) >= 2:
                    sim = sim.clone()
                    sim[:, 3, 1:] = -1e4
                logits4 = (sim / float(self.temperature)).max(dim=2).values  # [B,C]

            logits4 = torch.nan_to_num(logits4, nan=self._logit_mask_value, posinf=1e4, neginf=self._logit_mask_value).to(dtype=z_res.dtype)

        if domain_ids is None:
            return logits4
        hit_id = self.domain_map.get("HIT", 4)
        pu_id = self.domain_map.get("PU", 2)
        # Domains without rolling element fault should mask rolling (class index 3 in [normal,inner,outer,rolling])
        no_rolling_mask = (domain_ids == hit_id) | (domain_ids == pu_id)
        if no_rolling_mask.any():
            logits4 = logits4.clone()
            logits4[no_rolling_mask, 3] = self._logit_mask_value
        return logits4

    def compute_prototype_separation_loss(self, text_prototypes, margin=0.2):
        """
        Encourage prototypes (after prompt tuning) to be separated:
          L = mean(relu(sim_ij - margin)^2) over off-diagonal pairs.
        Uses:
          - HUST domain: classes [0,1,2,3]
          - HIT domain: classes [0,1,2] (rolling excluded)
        """
        if text_prototypes is None:
            return None
        device = next(self.parameters()).device
        base = torch.stack([text_prototypes[i] for i in [0, 1, 2, 3]], dim=0).to(device)  # [4,D]

        losses = []

        hust_id = self.domain_map.get("HUST", 3)
        dom_hust = torch.tensor([hust_id], dtype=torch.long, device=device)
        p_hust = self._apply_prompt_tuning(base, domain_ids=dom_hust).squeeze(0)  # [4,D]
        p_hust = F.normalize(p_hust, dim=-1)
        sim = p_hust @ p_hust.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        off = sim.masked_fill(eye, -1.0)
        losses.append(F.relu(off - float(margin)).pow(2).mean())

        hit_id = self.domain_map.get("HIT", 4)
        dom_hit = torch.tensor([hit_id], dtype=torch.long, device=device)
        p_hit = self._apply_prompt_tuning(base, domain_ids=dom_hit).squeeze(0)  # [4,D]
        p_hit = F.normalize(p_hit[:3], dim=-1)  # [3,D]
        sim2 = p_hit @ p_hit.t()
        eye2 = torch.eye(sim2.size(0), device=sim2.device, dtype=torch.bool)
        off2 = sim2.masked_fill(eye2, -1.0)
        losses.append(F.relu(off2 - float(margin)).pow(2).mean())

        return torch.stack(losses).mean()

    def _compute_prototype_fault_logits(self, z_res, domain_ids, text_prototypes):
        device = z_res.device
        sorted_keys = sorted(text_prototypes.keys())
        proto_mat = torch.stack([text_prototypes[k] for k in sorted_keys], dim=0).to(device)  # [C, D]
        proto_mat = F.normalize(proto_mat, dim=-1)
        logits_all = torch.matmul(z_res, proto_mat.t()) / self.temperature  # [B, C]
        batch_size = z_res.size(0)
        hit_id = self.domain_map.get("HIT", 4)
        pu_id = self.domain_map.get("PU", 2)
        logits_fault = torch.full((batch_size, 3), self._logit_mask_value, device=device)
        for i in range(batch_size):
            d_id = domain_ids[i].item()
            if d_id == hit_id or d_id == pu_id:
                logits_fault[i, :2] = logits_all[i, 1:3]
            else:
                logits_fault[i] = logits_all[i, 1:4]
        return logits_fault

    def _calibrate_nf_logits(self, logits_nf_global, domain_ids):
        """
        Calibrate NF logits by applying affine transformation to anomaly_logit (logit_fault - logit_normal).
        This allows the bias to actually shift the decision boundary, affecting NF Accuracy.
        
        Args:
            logits_nf_global: [B, 2] logits [normal, fault]
            domain_ids: [B] integer domain IDs
        
        Returns:
            logits_nf_global_calibrated: [B, 2] calibrated logits
        """
        if domain_ids is None:
            return logits_nf_global

        domain_ids = domain_ids.to(logits_nf_global.device)
        scales = self.nf_calibrator_scale[domain_ids]  # [B]
        biases = self.nf_calibrator_bias[domain_ids]   # [B]
        
        # Compute anomaly_logit = logit_fault - logit_normal
        anomaly_logit = logits_nf_global[:, 1] - logits_nf_global[:, 0]  # [B]
        
        # Apply affine transformation: a' = s * a + b
        anomaly_logit_calibrated = scales * anomaly_logit + biases  # [B]
        
        # Reconstruct 2-class logits from calibrated anomaly_logit
        # We keep the original logit_normal unchanged, and set logit_fault = logit_normal + a'
        # This ensures the calibration affects the decision boundary
        logit_normal = logits_nf_global[:, 0]  # [B]
        logit_fault_calibrated = logit_normal + anomaly_logit_calibrated  # [B]
        
        # Stack back to [B, 2]
        logits_nf_global_calibrated = torch.stack([logit_normal, logit_fault_calibrated], dim=1)  # [B, 2]
        
        return logits_nf_global_calibrated

    def forward(self, z, domain_ids=None, domain_names=None, text_prototypes=None, return_parts=False, dann_alpha=None, rolling_dann_alpha=None, z_signal=None):
        """
        z: (B, D) fusion embedding
        domain_ids: (B,) integer domain IDs (CWRU_gk2=0, CWRU_gk4=1, PU=2, HUST=3, HIT=4), optional
        domain_names: (B,) list of domain name strings, optional (alternative to domain_ids)
        return_parts: bool, if True, return intermediate logits in a dict
        """
        z_pre_norm = z
        z = F.normalize(z, dim=-1)
        z_metric = self.get_metric_embedding(z)
        if not self.hierarchical:
            w = self._get_normalized_weight("flat", self.classifier.weight)
            logits = torch.matmul(z, w.t()) / self.temperature
            # ===== Flat multi-center pooling (NCMC-like) =====
            # Apply per-class multi-center pooling and replace corresponding logits columns.
            try:
                if logits.size(1) >= 4:
                    pool = str(getattr(self, "flat_multi_center_pool", "logsumexp")).lower()
                    tau = max(1e-6, float(getattr(self, "flat_multi_center_tau", 0.07)))

                    def _pool_centers(base_w: torch.Tensor, extra_param: torch.Tensor) -> torch.Tensor:
                        # base_w: [1,D], extra_param: [K-1,D] (unnormalized params)
                        extra = F.normalize(extra_param, dim=-1, eps=1e-6)
                        centers = torch.cat([base_w, extra], dim=0)  # [K,D]
                        logits_k = torch.matmul(z, centers.t()) / self.temperature  # [B,K]
                        K = int(centers.size(0))
                        if pool == "max":
                            return logits_k.max(dim=1, keepdim=True).values
                        # logsumexp bias compensation: -log(K)
                        return tau * torch.logsumexp(logits_k / tau, dim=1, keepdim=True) - math.log(K)

                    # Start from current single-center logits
                    logit0 = logits[:, 0:1]
                    logit1 = logits[:, 1:2]
                    logit2 = logits[:, 2:3]
                    logit3 = logits[:, 3:4]

                    # Inner (class-1)
                    if bool(getattr(self, "use_flat_multi_center_inner", False)) and getattr(self, "flat_inner_extra", None) is not None:
                        inner_w = w[1:2]
                        logit1 = _pool_centers(inner_w, self.flat_inner_extra)

                    # Outer (class-2)
                    if bool(getattr(self, "use_flat_multi_center_outer", False)) and getattr(self, "flat_outer_extra", None) is not None:
                        outer_w = w[2:3]
                        logit2 = _pool_centers(outer_w, self.flat_outer_extra)

                    # Rolling (class-3)
                    if bool(getattr(self, "use_flat_multi_center_rolling", False)) and getattr(self, "flat_roll_extra", None) is not None:
                        roll_w = w[3:4]
                        logit3 = _pool_centers(roll_w, self.flat_roll_extra)

                    logits = torch.cat([logit0, logit1, logit2, logit3], dim=1)
            except Exception:
                pass
            if return_parts:
                logit_normal = logits[:, :1]
                logit_fault = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True)
                logits_nf_global = torch.cat([logit_normal, logit_fault], dim=1)
                parts = {
                    "logits_diag_supervised": logits,
                    "logits_diag": logits,
                    "logits_diag_teacher": logits,
                    "logits_nf_global": logits_nf_global,
                    "logits_nf": logits_nf_global,  # Fallback alias for loss_fusion_cls compatibility
                    "z_cls": self.get_cls_embedding(z),
                    "z_metric": z_metric,
                    "z_cls_pre_norm": z_pre_norm,
                    "z_res": z,
                    "z_gate": z,
                    "flat_mode": True,
                }
                return logits, parts
            return logits

        batch_size = z.size(0)
        domain_ids = self._resolve_domain_ids(domain_ids, domain_names, batch_size, z.device)

        # ===== hierarchical heads (Hybrid nf + fault type) =====
        # Decoupled branches:
        # - z_gate: isolated branch for global nf (stable gating, not affected by fault-type training)
        # - z_cls: diagnostic branch for fault-type classification
        z_gate = self.get_gate_embedding(z)  # Isolated gate branch
        z_cls = self.get_cls_embedding(z)    # Diagnostic branch
        residual_mask = None
        logits_residual = None
        z_res = z_cls
        # (New scheme 2) Fault-type dedicated representation: only fault-type loss updates it.
        if bool(getattr(self, "use_fault_dedicated_rep", False)) and self.fault_projection is not None:
            z_fault = F.normalize(self.fault_projection(z.detach()), dim=-1)
            z_for_fault = z_fault
        else:
            z_fault = None
            z_for_fault = z_res

        if self.use_residual_proxies and self.residual_proxy_bank is not None:
            residual_mask = self._build_fault_mask(domain_ids, batch_size, z.device)
            logits_residual, residual_vec = self.residual_proxy_bank(z_cls, fault_mask=residual_mask)
            z_res = z_cls + self.residual_alpha * residual_vec
            z_res = F.normalize(z_res, dim=-1)

        w_nf_global = self._get_normalized_weight("nf_global", self.classifier_nf_global.weight)  # [2,D]
        w_normal = self._get_normalized_weight("normal", self.classifier_normal.weight)          # [1,D]

        # Global nf uses z_gate (isolated from fault-type training)
        logits_nf_global = torch.matmul(z_gate, w_nf_global.t()) / self.temperature              # [B,2]
        logits_nf_global = self._calibrate_nf_logits(logits_nf_global, domain_ids)
        logits_normal_sup = torch.matmul(z_cls, w_normal.t()) / self.temperature             # [B,1]
        logits_normal = logits_normal_sup

        # ===== Supervised (non-prototype) diagnosis logits (for two-stage decoupling) =====
        # MAINLINE (A): When split heads are enabled, fault-type is P(type|fault) via 3-way softmax
        # over [l_inner, l_outer, l_roll] so inner/outer/rolling compete on the same scale (no roll-gate factor).
        #   p_fault = P(fault_any) from global NF head
        #   logits_3way = [l_io_inner, l_io_outer, l_roll]  (same temperature / scale)
        #   p_type = softmax(logits_3way); PU/HIT: mask roll column then renormalize
        #   p_inner = p_fault * p_type[:,0], p_outer = p_fault * p_type[:,1], p_rolling = p_fault * p_type[:,2]
        #   p_normal = 1 - p_fault
        # Then log(p+eps) as diagnosis logits. This avoids "roll gate × IO" where high p_roll crushes inner/outer.
        logits_fault_sup = None
        logits_fault_sup_raw = None
        logits_fault_io = None
        logits_fault_roll = None
        logits_fault_roll_raw = None
        z_roll = None
        if bool(getattr(self, "use_split_fault_heads", False)):
            lf_masked, lio, lroll, lf_raw, z_roll = self._compute_split_fault_logits(z_for_fault, domain_ids)
            if lf_masked is not None:
                logits_fault_sup = lf_masked
                logits_fault_sup_raw = lf_raw
                logits_fault_io = lio
                logits_fault_roll_raw = lroll  # [B,2] (non_roll, roll)
                # Keep scalar roll logit diff for compatibility/debugging
                try:
                    logits_fault_roll = (lroll[:, 1:2] - lroll[:, 0:1]) if (lroll is not None and lroll.dim() == 2 and lroll.size(1) == 2) else lroll
                except Exception:
                    logits_fault_roll = lroll

        if logits_fault_sup is None:
            # Fallback to original fault head logits (domain-aware if enabled)
            if self.use_domain_aware_fault_head and domain_ids is not None:
                logits_fault_sup = self._get_domain_aware_fault_logits(z_for_fault, domain_ids)
                logits_fault_sup_raw = logits_fault_sup
            else:
                if hasattr(self, 'classifier_fault'):
                    w_fault_sup = self._get_normalized_weight("fault", self.classifier_fault.weight)
                else:
                    w_fault_sup = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)
                logits_fault_sup = torch.matmul(z_for_fault, w_fault_sup.t()) / self.temperature
                logits_fault_sup = torch.nan_to_num(
                    logits_fault_sup,
                    nan=self._logit_mask_value,
                    posinf=1e4,
                    neginf=self._logit_mask_value
                ).to(dtype=z_res.dtype)
                logits_fault_sup_raw = logits_fault_sup

        # Build hierarchical diagnosis logits in log-probability space (stable & comparable across classes).
        logits_3way_out = None  # (New scheme 1) exposed for two-stage loss (3-class CE on fault samples only)
        try:
            eps = 1e-6
            # p_fault from calibrated NF head (use softmax rather than sigmoid so nf_calibrator is respected).
            p_nf = torch.softmax(logits_nf_global.float(), dim=1)  # [B,2]
            p_normal = p_nf[:, 0].clamp(min=eps, max=1.0 - eps)    # [B]
            p_fault = p_nf[:, 1].clamp(min=eps, max=1.0 - eps)     # [B]

            if bool(getattr(self, "use_split_fault_heads", False)) and (logits_fault_io is not None) and (logits_fault_roll_raw is not None):
                l_io = logits_fault_io.float()   # [B,2]
                l_roll = logits_fault_roll_raw.float()

                # Mask rolling for domains without it (PU/HIT)
                no_roll = None
                if domain_ids is not None:
                    hit_id = self.domain_map.get("HIT", 4)
                    pu_id = self.domain_map.get("PU", 2)
                    no_roll = (domain_ids == hit_id) | (domain_ids == pu_id)

                if self.use_hierarchical_fault_prob:
                    # (Attempt 14) Hierarchical probabilities (roll gate × IO), then log-probs for CE.
                    # p_roll = P(rolling | fault), p_io = P(inner/outer | fault, non-roll)
                    if l_roll.ndim == 2 and l_roll.size(1) == 2:
                        p_roll = torch.softmax(l_roll, dim=1)[:, 1]
                    else:
                        l_roll_col = l_roll.view(-1) if l_roll.dim() == 1 else l_roll.squeeze(-1)
                        p_roll = torch.sigmoid(l_roll_col)
                    p_io = torch.softmax(l_io, dim=1)  # [B,2]
                    p_inner = (1.0 - p_roll) * p_io[:, 0]
                    p_outer = (1.0 - p_roll) * p_io[:, 1]
                    p_rolling = p_roll
                    p_type = torch.stack([p_inner, p_outer, p_rolling], dim=1).clamp(min=eps, max=1.0)
                    if no_roll is not None and no_roll.any():
                        p_type = p_type.clone()
                        p_type[no_roll, 2] = 0.0
                        row_sum = p_type[no_roll].sum(dim=1, keepdim=True).clamp(min=eps)
                        p_type[no_roll] = p_type[no_roll] / row_sum
                    logits_3way_out = torch.log(p_type.clamp(min=eps))
                else:
                    # MAINLINE (legacy): 3-way softmax over [inner, outer, roll] logits.
                    if l_roll.ndim == 2 and l_roll.size(1) == 2:
                        l_roll_col = (l_roll[:, 1:2] - l_roll[:, 0:1])   # [B,1]
                    else:
                        l_roll_col = l_roll.view(-1, 1) if l_roll.dim() == 1 else l_roll
                    logits_3way = torch.cat([l_io, l_roll_col], dim=1)  # [B,3] (inner, outer, roll)

                    # (Attempt 13) Logit adjustment using long-tail prior (inner/outer/rolling).
                    if self.use_fault_logit_adjust and self.fault_logit_adjust_prior is not None:
                        prior = self.fault_logit_adjust_prior.to(device=logits_3way.device, dtype=logits_3way.dtype)
                        prior = prior / prior.sum().clamp(min=eps)
                        log_prior = torch.log(prior + eps).view(1, 3)
                        logits_3way = logits_3way - self.fault_logit_adjust_tau * log_prior

                    if no_roll is not None and no_roll.any():
                        logits_3way = logits_3way.clone()
                        logits_3way[no_roll, 2] = -1e9
                    logits_3way_out = logits_3way

                    p_type = torch.softmax(logits_3way, dim=1).clamp(min=eps, max=1.0)
                    if no_roll is not None and no_roll.any():
                        p_type = p_type.clone()
                        p_type[no_roll, 2] = 0.0
                        row_sum = p_type[no_roll].sum(dim=1, keepdim=True).clamp(min=eps)
                        p_type[no_roll] = p_type[no_roll] / row_sum

                    p_inner = p_fault * p_type[:, 0]
                    p_outer = p_fault * p_type[:, 1]
                    p_rolling = p_fault * p_type[:, 2]
            else:
                # Fallback: use softmax over 3-way fault logits as P(type | fault)
                p_type = torch.softmax(logits_fault_sup.float(), dim=1)  # [B,3]
                p_type = torch.clamp(p_type, min=eps, max=1.0)
                p_inner = p_fault * p_type[:, 0]
                p_outer = p_fault * p_type[:, 1]
                p_rolling = p_fault * p_type[:, 2]

            probs4 = torch.stack([p_normal, p_inner, p_outer, p_rolling], dim=1)  # [B,4]
            probs4 = probs4 / probs4.sum(dim=1, keepdim=True).clamp(min=eps)
            logits_diag_supervised = torch.log(probs4.clamp(min=eps)).to(dtype=z_res.dtype)
        except Exception:
            # Backward compatibility: fall back to old 4-way concatenation
            logits_diag_supervised = torch.cat([logits_normal_sup, logits_fault_sup], dim=1)  # [B,4] unified

        # ===== (MTFD) Multi-Task Fault Decomposition Outputs =====
        # Independent binary classifiers computed alongside main diagnosis
        # This allows "active" training of MTFD heads without breaking the main decision path.
        logits_mtfd_nf = None
        logits_mtfd_io = None 
        logits_mtfd_roll = None
        
        if self.use_mtfd and self.mtfd_nf_head is not None:
            # 1. NF Binary Task: Uses global z_gate (aligned with Normal vs Other)
            w_mtfd_nf = self._get_normalized_weight("mtfd_nf", self.mtfd_nf_head.weight)
            logits_mtfd_nf = torch.matmul(z_gate, w_mtfd_nf.t()) / self.temperature # [B, 2]
            
            # 2. IO & Rolling Tasks: Use fault-dedicated representation z_for_fault
            # Reuse the split-head logic but ensure we get the raw outputs for independent losses
            if bool(getattr(self, "use_split_fault_heads", False)):
                # We leverage the existing split-head architecture for IO/Rolling tasks
                # but we treat them as independent binary problems in the loss function
                # _compute_split_fault_logits returns: lb_masked, lio, lroll, lb_raw
                _, l_io_raw, l_roll_raw, _, _ = self._compute_split_fault_logits(z_for_fault, domain_ids)
                logits_mtfd_io = l_io_raw     # [B, 2] Inner/Outer
                logits_mtfd_roll = l_roll_raw # [B, 2] Roll/Non-Roll
            else:
                 pass # MTFD requires split heads for IO/Roll decomposition

        if text_prototypes is not None:
            try:
                text_prototypes = self.apply_text_prompt_tuning(
                    text_prototypes,
                    detach_output=(not bool(self.training))
                )
            except Exception:
                pass

        use_proto = self.use_prototype_head and text_prototypes is not None and domain_ids is not None

        if self.use_prototype_head and text_prototypes is not None and domain_ids is not None:
            # Prototype-tied diagnosis with stability fix:
            # - Fault logits use z_res (residual-enhanced, good for inter-fault separation)
            # - Normal logit uses z_cls (avoids "normal recall = 0" collapse when z_res becomes too fault-biased)
            _proto_metric = z_metric if bool(getattr(self, "use_dual_space_decouple", False)) else z_res
            logits_diag_fault = self._compute_prototype_diag_logits(_proto_metric, domain_ids, text_prototypes)  # [B,4]
            logits_diag_norm = self._compute_prototype_diag_logits(z_cls, domain_ids, text_prototypes)    # [B,4]
            logits_diag = torch.cat([logits_diag_norm[:, :1], logits_diag_fault[:, 1:4]], dim=1)         # [B,4]
            logits_normal = logits_diag[:, :1]  # [B,1]
            logits_fault = logits_diag[:, 1:4]  # [B,3] (HIT rolling masked)
        elif self.use_domain_aware_fault_head and domain_ids is not None:
            logits_fault = self._get_domain_aware_fault_logits(z_res, domain_ids)
            logits_diag = self._assemble_domain_diagnosis_logits(logits_normal, logits_fault, domain_ids)
        else:
            if hasattr(self, 'classifier_fault'):
                w_fault = self._get_normalized_weight("fault", self.classifier_fault.weight)
            else:
                w_fault = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)
            logits_fault = torch.matmul(z_res, w_fault.t()) / self.temperature
            logits_diag = torch.cat([logits_normal, logits_fault], dim=1)

        # ===== Decision Mode Logic =====
        decision_mode = getattr(self, "diag_decision_mode", "argmax")

        # Attempt 8: Hierarchical Compositional Inference
        # Force the main diagnosis logits to be the composition of binary heads.
        if decision_mode == "hierarchical_composition":
            logits_diag = logits_diag_supervised
            # Update parts
            logits_normal = logits_diag[:, :1]
            logits_fault = logits_diag[:, 1:4]

        # Attempt 9: Sequential Thresholding Logic
        # Sequential Decision: Detection (NF) -> Diagnosis (Type)
        # Prevents "Vote Dilution" where Normal wins against split fault probabilities.
        elif decision_mode == "sequential":
            # IMPORTANT (Attempt 10 fix): decouple training vs inference.
            # - Training/loss MUST see soft (finite) logits to keep CE well-defined.
            # - Sequential hard decision should be applied ONLY in decision code (e.g., utils/diag_decision.py)
            #   for metrics/inference, NOT by masking logits inside forward().
            logits_diag = logits_diag_supervised
            logits_normal = logits_diag[:, :1]
            logits_fault = logits_diag[:, 1:4]

        # Save the "teacher" diagnosis logits BEFORE any linear residual / bias composition.
        # This is used for semantic distillation into the learnable linear diagnosis head.
        logits_diag_teacher = logits_diag

        # ===== (D2) Add linear residual diagnosis logits =====
        # This helps when prototype similarity is biased on target domains (e.g., Inner/Outer absorbed into Normal).
        if self.use_diag_linear_residual and hasattr(self, "diag_linear_head") and logits_diag is not None:
            try:
                import config as cfg
                enable_lin = bool(getattr(cfg, "USE_DIAG_LINEAR_RESIDUAL", True))
            except Exception:
                enable_lin = True
            if enable_lin:
                w_lin = self._get_normalized_weight("diag_linear", self.diag_linear_head.weight)  # [C,D]
                logits_lin = torch.matmul(z_res, w_lin.t()) / float(self.temperature)  # [B,C]
                logits_lin = torch.nan_to_num(
                    logits_lin,
                    nan=self._logit_mask_value,
                    posinf=1e4,
                    neginf=self._logit_mask_value
                ).to(dtype=logits_diag.dtype)

                # Apply domain mask for rolling (class index 3) where not present.
                if self.mask_missing_classes_in_logits and domain_ids is not None and logits_lin.size(1) >= 4:
                    hit_id = self.domain_map.get("HIT", 4)
                    pu_id = self.domain_map.get("PU", 2)
                    no_rolling_mask = (domain_ids == hit_id) | (domain_ids == pu_id)
                    if no_rolling_mask.any():
                        logits_lin = logits_lin.clone()
                        logits_lin[no_rolling_mask, 3] = self._logit_mask_value

                # Beta is strictly in [0, beta_max] to prevent the linear term from dominating.
                try:
                    import config as cfg
                    beta_max = float(getattr(cfg, "DIAG_LINEAR_BETA_MAX", 0.5))
                except Exception:
                    beta_max = 0.5
                beta_max = max(0.0, beta_max)
                beta = (float(beta_max) * torch.sigmoid(self.diag_linear_scale.to(dtype=logits_diag.dtype, device=logits_diag.device)))
                ramp = float(getattr(self, "_diag_linear_ramp", 1.0))
                # Keep ramp stable in fp32
                if ramp != ramp:
                    ramp = 1.0
                ramp = max(0.0, min(1.0, ramp))
                # Sample-adaptive gate: beta(x) in [0,1]
                gate = 1.0
                try:
                    import config as cfg
                    use_gate = bool(getattr(cfg, "USE_DIAG_LINEAR_GATE", True))
                except Exception:
                    use_gate = True
                if use_gate and hasattr(self, "diag_linear_gate") and self.diag_linear_gate is not None:
                    g = self.diag_linear_gate(z_res.float())  # [B,1]
                    gate = torch.sigmoid(g).to(dtype=logits_diag.dtype)  # [B,1]
                logits_diag = logits_diag + ((beta * float(ramp)) * gate) * logits_lin

                # Re-sync downstream logits after composition
                logits_normal = logits_diag[:, :1]
                logits_fault = logits_diag[:, 1:4]

        # ===== (S-B) Apply class-bias prior (small) =====
        # Applied after all compositions to correct systematic class logit shift.
        if hasattr(self, "diag_class_bias") and logits_diag is not None:
            try:
                import config as cfg
                use_bias = bool(getattr(cfg, "USE_DIAG_CLASS_BIAS", True))
            except Exception:
                use_bias = True
            if use_bias and self.diag_class_bias is not None and logits_diag.dim() == 2 and logits_diag.size(1) == self.diag_class_bias.numel():
                logits_diag = logits_diag + self.diag_class_bias.to(device=logits_diag.device, dtype=logits_diag.dtype).view(1, -1)
                logits_normal = logits_diag[:, :1]
                logits_fault = logits_diag[:, 1:4]

        # nf multibinary logit diffs: fault_k - normal (for backward compatibility)
        num_faults = logits_fault.size(1)
        logits_nf_multi = logits_fault - logits_normal  # [B, num_faults]

        if return_parts:
            # Report effective beta (bounded) for debugging/logging
            beta_eff_log = 0.0
            try:
                import config as cfg
                beta_max_log = float(getattr(cfg, "DIAG_LINEAR_BETA_MAX", 0.5))
            except Exception:
                beta_max_log = 0.5
            try:
                beta_max_log = max(0.0, float(beta_max_log))
                beta_eff_log = float((beta_max_log * torch.sigmoid(self.diag_linear_scale.detach().float().cpu())).item())
            except Exception:
                beta_eff_log = 0.0
            parts = {
                "logits_normal": logits_normal.squeeze(1),   # [B]
                "logits_nf_global": logits_nf_global,        # [B,2]
                "logits_nf_multi": logits_nf_multi,          # [B, num_faults]
                "logits_fault": logits_fault,                # [B, num_faults]
                "logits_diag": logits_diag,
                # Supervised diagnosis logits (no prototypes): used by training/evaluation as the main decision logits.
                "logits_diag_supervised": logits_diag_supervised,
                "logits_normal_supervised": logits_normal_sup.squeeze(1),
                "logits_fault_supervised": logits_fault_sup,
                "logits_fault_supervised_raw": logits_fault_sup_raw,
                # Split-head outputs (optional)
                "logits_fault_io": logits_fault_io,
                "logits_fault_roll": logits_fault_roll,
                "logits_fault_roll_raw": logits_fault_roll_raw,
                "logits_3way": logits_3way_out,
                
                # MTFD Independent Logits (for multi-target training)
                "logits_mtfd_nf": logits_mtfd_nf,
                "logits_mtfd_io": logits_mtfd_io,
                "logits_mtfd_roll": logits_mtfd_roll,

                # Teacher logits (pre-linear, pre-bias) for distillation.
                "logits_diag_teacher": logits_diag_teacher,
                "logits_diag_linear": logits_lin if ("logits_lin" in locals()) else None,
                "diag_linear_scale": beta_eff_log,
                "diag_linear_ramp": float(getattr(self, "_diag_linear_ramp", 1.0)),
                "diag_linear_gate_mean": float(gate.mean().detach().cpu().item()) if ("gate" in locals() and isinstance(gate, torch.Tensor)) else 1.0,
                "diag_class_bias_norm": float(self.diag_class_bias.detach().float().norm().cpu().item()) if hasattr(self, "diag_class_bias") else 0.0,
                "z_cls": z_cls,                              # [B,D]
                "z_metric": z_metric,                        # [B,D] dual-space semantic/metric embedding
                "z_cls_pre_norm": getattr(self, "_last_z_cls_pre_norm", None),  # [B,D], before L2 normalize
                "z_gate": z_gate,                            # [B,D] (isolated gate branch)
                "z_roll": z_roll,                            # [B,D] (rolling branch rep, optional)
                "logits_residual": logits_residual,
                "residual_mask": residual_mask,
                "z_res": z_res,
                "domain_ids": domain_ids
            }

            # Optional: DANN domain logits (via GRL); only used when enabled and alpha>0.
            try:
                if bool(getattr(self, "use_dann", False)) and dann_alpha is not None:
                    a = float(dann_alpha)
                    if a > 0.0 and hasattr(self, "domain_discriminator") and self.domain_discriminator is not None:
                        # Select feature for domain adversarial training.
                        feat_name = str(getattr(self, "dann_feature", "z_gate")).lower()
                        if feat_name == "z_cls":
                            z_for_domain = z_cls
                        elif feat_name == "z_res":
                            z_for_domain = z_res
                        elif feat_name == "z":
                            z_for_domain = z
                        elif feat_name == "z_signal" and z_signal is not None:
                            # ATT117: DANN on raw signal embedding (load-shift source).
                            # Preserves z_spec geometry (no DANN collateral on spectral features).
                            z_for_domain = F.normalize(z_signal.float(), dim=-1)
                        else:
                            z_for_domain = z_gate

                        # CDAN: use feature ⊗ prediction distribution as input.
                        if str(getattr(self, "dann_mode", "vanilla")).lower() == "cdan":
                            logits_for_p = logits_diag
                            try:
                                if "logits_diag_teacher" in parts and parts["logits_diag_teacher"] is not None:
                                    logits_for_p = parts["logits_diag_teacher"]
                            except Exception:
                                pass
                            p = torch.softmax(logits_for_p.float(), dim=1)
                            if bool(getattr(self, "dann_detach_probs", True)):
                                p = p.detach()
                            # outer product -> [B, C*D]
                            cdan = torch.bmm(p.unsqueeze(2), z_for_domain.float().unsqueeze(1)).flatten(1)
                            parts["domain_logits"] = self.domain_discriminator(grad_reverse(cdan, alpha=a))
                        else:
                            parts["domain_logits"] = self.domain_discriminator(grad_reverse(z_for_domain, alpha=a))
                    else:
                        parts["domain_logits"] = None
                else:
                    parts["domain_logits"] = None
            except Exception:
                parts["domain_logits"] = None

            # Optional: Rolling-domain adversarial logits (GRL on rolling branch)
            try:
                parts["roll_domain_logits"] = None
                if bool(getattr(self, "use_rolling_dann", False)) and rolling_dann_alpha is not None:
                    a = float(rolling_dann_alpha)
                    if a > 0.0 and z_roll is not None and domain_ids is not None:
                        if hasattr(self, "roll_domain_discriminator") and self.roll_domain_discriminator is not None:
                            parts["roll_domain_logits"] = self.roll_domain_discriminator(grad_reverse(z_roll, alpha=a))
            except Exception:
                parts["roll_domain_logits"] = None

            # Prototype separation loss (only meaningful when prompt tuning is enabled)
            if self.use_prototype_head and text_prototypes is not None and self.use_prompt_tuning:
                parts["proto_sep_loss"] = self.compute_prototype_separation_loss(text_prototypes, margin=0.2)
            else:
                parts["proto_sep_loss"] = None
            
            # Outer margin auxiliary logits (prototype space over a dedicated projection of z_cls)
            # Used by an auxiliary loss to prevent Outer collapsing into other classes.
            use_proto = self.use_prototype_head and text_prototypes is not None and domain_ids is not None
            if use_proto:
                z_outer = F.normalize(self.outer_adapter(z_cls), dim=-1, eps=1e-6)
                z_outer = torch.nan_to_num(z_outer, nan=0.0, posinf=0.0, neginf=0.0)
                parts["logits_diag_outer"] = self._compute_prototype_diag_logits(z_outer, domain_ids, text_prototypes)
            else:
                parts["logits_diag_outer"] = None

            return logits_diag, parts
        return logits_diag
    
    def get_anomaly_score(self, z, score_type="logit_diff", evidence_temperature=1.0, domain_ids=None):
        """
        计算异常检测分数（用于AUC评估）
        
        Args:
            z: (B, D) fusion embedding
            score_type: "logit_diff" (fault_logit - normal_logit) or "prob" (P(fault))
                       or "evidence_max" (Noisy-OR aggregation of multiple evidence sources)
                       or "prob_hier_or" (Probability fusion: global nf + nf_multi via noisy-or)
            evidence_temperature: temperature for smoothing evidence aggregation (default: 1.0)
        
        Returns:
            anomaly_score: (B,) 异常分数，值越大表示越可能是故障
        """
        if not self.hierarchical:
            z = F.normalize(z, dim=-1)
            w = self._get_normalized_weight("flat", self.classifier.weight)
            logits = torch.matmul(z, w.t()) / self.temperature
            score_type = str(score_type).lower()
            if score_type in {"prob", "prob_hier_or", "evidence_or", "evidence_max"}:
                probs = torch.softmax(logits, dim=1)
                return 1.0 - probs[:, 0]
            logit_normal = logits[:, 0]
            logit_fault = torch.logsumexp(logits[:, 1:], dim=1)
            return logit_fault - logit_normal
        
        z = F.normalize(z, dim=-1)
        
        # Default: use global nf head for stable AUC_nf & gating.
        # Use z_gate (isolated branch) instead of z_cls to avoid fault-type training interference
        z_gate = self.get_gate_embedding(z)
        w_nf_global = self._get_normalized_weight("nf_global", self.classifier_nf_global.weight)  # [2,D]
        logits_nf_global = torch.matmul(z_gate, w_nf_global.t()) / self.temperature  # [B,2]
        logits_nf_global = self._calibrate_nf_logits(logits_nf_global, domain_ids)

        if score_type == "logit_diff":
            # fault_any - normal
            return logits_nf_global[:, 1] - logits_nf_global[:, 0]
        if score_type == "prob":
            return F.softmax(logits_nf_global, dim=1)[:, 1]
        if score_type == "logit_fused":
            # Optional fused score: global + per-type smooth-OR (can be used for ablations)
            # Note: For per-type, we still use z_cls (diagnostic branch)
            z_cls = self.get_cls_embedding(z)
            w_normal = self._get_normalized_weight("normal", self.classifier_normal.weight)
            logits_normal = torch.matmul(z_cls, w_normal.t()) / self.temperature
            # Prefer split-head fault logits if enabled (keeps rolling separable).
            logits_fault = None
            if bool(getattr(self, "use_split_fault_heads", False)):
                lf_masked, _, _, _, _ = self._compute_split_fault_logits(z_cls, domain_ids=None)
                logits_fault = lf_masked
            if logits_fault is None:
                # Use shared fault head for backward compatibility (get_anomaly_score doesn't have domain info)
                if hasattr(self, 'classifier_fault'):
                    w_fault = self._get_normalized_weight("fault", self.classifier_fault.weight)
                else:
                    w_fault = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)
                logits_fault = torch.matmul(z_cls, w_fault.t()) / self.temperature
            logits_nf_multi = logits_fault - logits_normal
            return (logits_nf_global[:, 1] - logits_nf_global[:, 0]) + 0.5 * torch.logsumexp(logits_nf_multi, dim=1)
        if score_type == "prob_hier_or":
            # 概率融合的 nf score：使用 soft-hierarchical 的概率分解
            # P(fault) = 融合(global nf, nf_multi) 使用 noisy-or
            z_cls = self.get_cls_embedding(z)
            w_normal = self._get_normalized_weight("normal", self.classifier_normal.weight)
            logits_normal = torch.matmul(z_cls, w_normal.t()) / self.temperature  # [B, 1]
            logits_fault = None
            if bool(getattr(self, "use_split_fault_heads", False)):
                lf_masked, _, _, _, _ = self._compute_split_fault_logits(z_cls, domain_ids=None)
                logits_fault = lf_masked
            if logits_fault is None:
                # Use shared fault head for backward compatibility (get_anomaly_score doesn't have domain info)
                if hasattr(self, 'classifier_fault'):
                    w_fault = self._get_normalized_weight("fault", self.classifier_fault.weight)
                else:
                    w_fault = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)
                logits_fault = torch.matmul(z_cls, w_fault.t()) / self.temperature    # [B, 3]
            logits_nf_multi = logits_fault - logits_normal  # [B, 3] fault_k vs normal
            
            # P(fault|global) from global nf head
            p_fault_global = F.softmax(logits_nf_global, dim=1)[:, 1]  # [B]
            
            # P(fault|nf_multi) using noisy-or aggregation
            p_fault_multi_per_type = torch.sigmoid(logits_nf_multi)  # [B, 3]
            p_fault_multi = 1.0 - torch.prod(1.0 - p_fault_multi_per_type, dim=1)  # [B] noisy-or
            
            # Noisy-OR fusion: P_final = 1 - (1 - P_global) * (1 - P_multi)
            p_fault_fused = 1.0 - (1.0 - p_fault_global) * (1.0 - p_fault_multi)  # [B]
            
            # 转换为 logit score（保持与 logit_diff 类似的尺度）
            # 使用 logit = log(p / (1-p))，但为了避免数值问题，使用 logit 近似
            # 这里直接返回概率作为分数（因为 AUC 只需要单调性）
            return p_fault_fused
        
        if score_type == "evidence_max":
            # Multi-granularity evidence aggregation using Noisy-OR logic
            # Combines global nf evidence with per-type nf evidence
            # Formula: P_final = 1 - (1 - P_global) * (1 - P_multi_max)
            # This allows any strong fault signal to trigger detection
            
            # Global evidence: P(fault | any type)
            logits_global_scaled = logits_nf_global / evidence_temperature
            p_global = F.softmax(logits_global_scaled, dim=1)[:, 1]  # [B]
            
            # Per-type evidence: max over P(fault_k | type k)
            z_cls = self.get_cls_embedding(z)
            w_normal = self._get_normalized_weight("normal", self.classifier_normal.weight)
            logits_normal = torch.matmul(z_cls, w_normal.t()) / self.temperature  # [B, 1]
            logits_fault = None
            if bool(getattr(self, "use_split_fault_heads", False)):
                lf_masked, _, _, _, _ = self._compute_split_fault_logits(z_cls, domain_ids=None)
                logits_fault = lf_masked
            if logits_fault is None:
                # Use shared fault head for backward compatibility (get_anomaly_score doesn't have domain info)
                if hasattr(self, 'classifier_fault'):
                    w_fault = self._get_normalized_weight("fault", self.classifier_fault.weight)
                else:
                    w_fault = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)
                logits_fault = torch.matmul(z_cls, w_fault.t()) / self.temperature  # [B, 3]
            
            # Per-type nf probabilities (each is P(fault_k) vs P(normal))
            logits_nf_multi = torch.cat([logits_normal, logits_fault], dim=1)  # [B, 4]
            # For each fault type, compute P(fault_k) / (P(normal) + P(fault_k))
            # We use a temperature-scaled softmax for stability
            logits_nf_multi_scaled = logits_nf_multi / evidence_temperature
            probs_multi = F.softmax(logits_nf_multi_scaled, dim=1)  # [B, 4]
            p_multi_max = probs_multi[:, 1:].max(dim=1)[0]  # [B] max over fault types
            
            # Noisy-OR aggregation: at least one evidence source detects fault
            # P(fault) = 1 - P(not_fault_global) * P(not_fault_multi)
            p_combined = 1.0 - (1.0 - p_global) * (1.0 - p_multi_max)
            
            # Return as logit for consistency with other score types
            # Use safe clipping to avoid log(0) or log(1)
            p_combined = torch.clamp(p_combined, min=1e-7, max=1.0 - 1e-7)
            return torch.log(p_combined / (1.0 - p_combined))

        raise ValueError(f"Unknown score_type: {score_type}")

    def get_gate_embedding(self, z):
        """
        Gate branch embedding (z_gate) for global NF. (1b) No detach: backbone receives NF gradient.
        """
        if not self.hierarchical:
            return F.normalize(z, dim=-1)
        z = F.normalize(z, dim=-1)
        z_gate = z + self.gate_adapter(z)  # Backbone gets gradient from NF path
        return F.normalize(z_gate, dim=-1)

    def get_metric_embedding(self, z):
        """
        Metric branch embedding (z_metric) used by semantic/prototype paths.
        When dual-space is enabled, this branch is explicitly decoupled from z_cls.
        """
        if z is None:
            return z
        z = F.normalize(z, dim=-1)
        if bool(getattr(self, "use_dual_space_decouple", False)) and hasattr(self, "proj_metric"):
            z_input = z.detach() if bool(getattr(self, "dual_space_metric_detach", False)) else z
            z_metric_delta = self.proj_metric(z_input)
            if bool(getattr(self, "use_metric_proj_residual_mlp", False)):
                z_metric = z_input + float(getattr(self, "metric_proj_residual_scale", 1.0)) * z_metric_delta
            else:
                z_metric = z_metric_delta
            z_metric = torch.nan_to_num(z_metric, nan=0.0, posinf=0.0, neginf=0.0)
            return F.normalize(z_metric, dim=-1)
        return z

    def get_cls_embedding(self, z):
        """
        Diagnosis branch embedding (z_cls). (1b) When use_cls_detach_fault_type, use z.detach()
        so fault-type loss does not update backbone; only cls_adapter and fault heads get gradient.
        """
        z = F.normalize(z, dim=-1)
        if not self.hierarchical:
            z_cls = self.proj_cls(z) if bool(getattr(self, "use_dual_space_decouple", False)) else z
            self._last_z_cls_pre_norm = z_cls
            return F.normalize(z_cls, dim=-1)
        if getattr(self, "use_cls_adapter", False):
            if getattr(self, "use_cls_detach_fault_type", False):
                z_input = z.detach()
            else:
                z_input = z
            z_cls = z_input + 0.5 * self.cls_adapter(z)
        else:
            z_cls = z
        if bool(getattr(self, "use_dual_space_decouple", False)):
            z_cls = self.proj_cls(z_cls)
        self._last_z_cls_pre_norm = z_cls
        return F.normalize(z_cls, dim=-1)

    def project_for_supcon(self, z: torch.Tensor):
        """
        ATT240(S1): use direct normalized diagnosis embedding for SupCon.
        Keep projection parameters defined for checkpoint compatibility.
        """
        if z is None:
            return z
        return F.normalize(z, dim=-1)

    def update_normal_reference(self, z_gate: torch.Tensor, labels: torch.Tensor, momentum: float = 0.99):
        """(3b) EMA update of normal reference from normal samples' z_gate."""
        if z_gate is None or labels is None or z_gate.size(0) == 0:
            return
        normal_mask = (labels == 0)
        if not normal_mask.any():
            return
        z_n = z_gate[normal_mask].detach().float()
        z_n = F.normalize(z_n, dim=-1)
        mean_n = z_n.mean(dim=0)
        mean_n = F.normalize(mean_n.unsqueeze(0), dim=-1).squeeze(0)
        ref = self.normal_reference.to(z_n.device).float()
        new_ref = momentum * ref + (1.0 - momentum) * mean_n
        self.normal_reference.copy_(F.normalize(new_ref.unsqueeze(0), dim=-1).squeeze(0))
    
    def _get_domain_aware_fault_logits(self, z_cls, domain_ids):
        """
        Get domain-specific fault logits.
        
        Args:
            z_cls: [B, D] diagnostic branch embedding
            domain_ids: [B] integer domain IDs (CWRU_gk2=0, CWRU_gk4=1, PU=2, HUST=3, HIT=4)
        
        Returns:
            logits_fault: [B, num_faults] fault logits (varies by domain: HUST=3, HIT=2)
        """
        batch_size = z_cls.size(0)
        device = z_cls.device
        
        if not self.use_domain_aware_fault_head:
            # Fallback to shared head
            if hasattr(self, 'classifier_fault'):
                w_fault = self._get_normalized_weight("fault", self.classifier_fault.weight)
            else:
                w_fault = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)
            return torch.matmul(z_cls, w_fault.t()) / self.temperature
        
        # Collect fault logits per domain
        all_fault_logits = []
        hust_id = self.domain_map.get("HUST", 3)
        hit_id = self.domain_map.get("HIT", 4)
        for i in range(batch_size):
            domain_id = domain_ids[i].item()
            if domain_id == hust_id:  # HUST: 3 faults
                w_fault = self._get_normalized_weight("fault_hust", self.classifier_fault_hust.weight)  # [3,D]
                fault_logits_i = torch.matmul(z_cls[i:i+1], w_fault.t()) / self.temperature  # [1,3]
            elif domain_id == hit_id:  # HIT: 2 faults
                w_fault = self._get_normalized_weight("fault_hit", self.classifier_fault_hit.weight)  # [2,D]
                fault_logits_i = torch.matmul(z_cls[i:i+1], w_fault.t()) / self.temperature  # [1,2]
            else:  # CWRU/PU: use shared head (3 faults for backward compatibility)
                w_fault = self._get_normalized_weight("fault_shared", self.classifier_fault_shared.weight)  # [3,D]
                fault_logits_i = torch.matmul(z_cls[i:i+1], w_fault.t()) / self.temperature  # [1,3]
            all_fault_logits.append(fault_logits_i)
        
        # Stack into a tensor (different domains may have different num_faults, so we pad to max)
        max_faults = max(3, 3, 2)  # HUST=3, HIT=2, shared=3
        padded_logits = []
        for i, logits_i in enumerate(all_fault_logits):
            num_faults_i = logits_i.size(1)
            if num_faults_i < max_faults:
                # Pad with a large finite negative number for missing faults (avoid non-finite guards)
                pad_size = max_faults - num_faults_i
                padding = torch.full((1, pad_size), self._logit_mask_value, device=device)
                logits_i = torch.cat([logits_i, padding], dim=1)
            padded_logits.append(logits_i)
        logits_fault_padded = torch.cat(padded_logits, dim=0)  # [B, max_faults]
        
        return logits_fault_padded
    
    def _assemble_domain_diagnosis_logits(self, logits_normal, logits_fault, domain_ids):
        """
        Assemble domain-specific diagnosis logits.
        **Unified 4-class output**: Always returns [B, 4] logits [normal, inner, outer, rolling].
        For HIT (which has no rolling), rolling logit is set to -inf (will be masked in softmax).
        This allows mixed-batch training with unified 4-class CE loss.
        
        Args:
            logits_normal: [B, 1] normal logits
            logits_fault: [B, num_faults] fault logits (may be padded to max_faults=3)
            domain_ids: [B] integer domain IDs
        
        Returns:
            logits_diag: [B, 4] unified 4-class diagnosis logits
              - All domains: [normal, inner, outer, rolling]
              - HIT: rolling logit = -inf (masked)
              - HUST/CWRU/PU: all 4 classes valid
        """
        batch_size = logits_normal.size(0)
        device = logits_normal.device
        
        if not self.use_domain_aware_fault_head:
            # Legacy: always return 4-class (already unified)
            return torch.cat([logits_normal, logits_fault], dim=1)  # [B,4]
        
        # Unified 4-class output: [normal, inner, outer, rolling]
        all_diag_logits = []
        hit_id = self.domain_map.get("HIT", 4)
        for i in range(batch_size):
            domain_id = domain_ids[i].item() if domain_ids is not None else 0
            normal_i = logits_normal[i:i+1]  # [1,1]
            fault_i = logits_fault[i:i+1]    # [1, max_faults] (already padded to 3 if needed)
            
            if domain_id == hit_id:  # HIT: 3 classes [normal, inner, outer], rolling masked
                # Use first 2 fault logits (inner, outer), set rolling to a large finite negative
                rolling_logit = torch.full((1, 1), self._logit_mask_value, device=device, dtype=logits_normal.dtype)
                diag_i = torch.cat([normal_i, fault_i[:, :2], rolling_logit], dim=1)  # [1,4]
            else:  # HUST/CWRU/PU: 4 classes [normal, inner, outer, rolling]
                # Use all 3 fault logits (inner, outer, rolling)
                diag_i = torch.cat([normal_i, fault_i[:, :3]], dim=1)  # [1,4]
            all_diag_logits.append(diag_i)
        
        # Stack to [B, 4] - unified 4-class output
        diag_logits_stacked = torch.cat(all_diag_logits, dim=0)  # [B, 4]
        
        return diag_logits_stacked

    def get_class_probs_soft_hierarchical(
        self,
        z,
        temperature=1.0,
        domain_ids=None,
        domain_names=None,
        text_prototypes=None,
        use_nf_multi=True,
        nf_global_weight=0.7,
        nf_logit_alpha=1.0,
    ):
        """
        Soft hierarchical inference (LOGIT-SPACE, more stable than probability multiply):
          Let s be a scalar "faultness logit" (from NF head, optionally fused with nf_multi).
          Let l_k be fault-type logits over {inner, outer, rolling} (masked per-domain).

          We form 4-class logits:
            logit_normal = model's diagnosis normal logit (prototype-tied / domain-aware)
            logit_k      = l_k + alpha * s
          Then:
            P = softmax([logit_normal, logit_inner, logit_outer, logit_rolling] / T)

        Key design point (fixes "Normal -> Rolling" collapse in evaluation):
        - P(fault) uses the isolated global NF head (z_gate), including nf_calibrator if domain_ids provided.
        - P(k|fault) uses the *same* fault logits produced by forward() (prototype-tied / domain-aware / masked),
          so the label-space masking (HIT/PU no-rolling) is respected.

        This method is for inference/evaluation only and does not change training losses.
        """
        if not self.hierarchical:
            logits = self.forward(z, domain_ids=domain_ids, domain_names=domain_names, text_prototypes=text_prototypes)
            return F.softmax(logits / float(temperature), dim=1)

        z = F.normalize(z, dim=-1)

        # Use forward() to ensure we reuse the exact current structure:
        # - residual proxies (z_res)
        # - prototype-tied logits (if enabled)
        # - domain-aware masking (HIT/PU no-rolling)
        # - nf_calibrator (if domain_ids provided)
        logits_diag, parts = self.forward(
            z,
            domain_ids=domain_ids,
            domain_names=domain_names,
            text_prototypes=text_prototypes,
            return_parts=True,
        )

        logits_nf_global = parts["logits_nf_global"]  # [B,2] already calibrated if domain_ids is not None
        # Prefer supervised fault logits (split-head / domain-aware) for stability under domain shift.
        logits_fault = parts.get("logits_fault_supervised", None)
        if logits_fault is None:
            logits_fault = parts["logits_fault"]          # [B,3] (rolling masked for HIT/PU via large negative)
        logits_nf_multi = parts.get("logits_nf_multi", None)  # [B,3] or None

        # Faultness from global NF head (logit space)
        # anomaly_logit_global > 0 => fault
        s_global = (logits_nf_global[:, 1] - logits_nf_global[:, 0])  # [B]

        # Optional nf_multi enhancement:
        # convert noisy-or probability to logit, then fuse in logit space
        if use_nf_multi and logits_nf_multi is not None:
            p_multi_per_type = torch.sigmoid(logits_nf_multi)  # [B,3]
            p_multi = 1.0 - torch.prod(1.0 - p_multi_per_type, dim=1)  # [B]
            p_multi = p_multi.clamp(1e-6, 1.0 - 1e-6)
            s_multi = torch.log(p_multi) - torch.log1p(-p_multi)  # logit(p_multi)
            w = float(nf_global_weight)
            s = w * s_global + (1.0 - w) * s_multi
        else:
            s = s_global

        # Build 4-class logits and softmax
        # IMPORTANT: do NOT use a constant normal logit; reuse the model's normal logit so diagnosis
        # isn't entirely hostage to NF scaling.
        alpha = float(nf_logit_alpha)
        base_diag = parts.get("logits_diag_supervised", None)
        if base_diag is None:
            base_diag = logits_diag
        logit_normal = base_diag[:, :1]  # [B,1]
        logits4 = torch.cat([logit_normal, (logits_fault + alpha * s.unsqueeze(1))], dim=1)  # [B,4]
        probs4 = F.softmax(logits4 / float(temperature), dim=1)
        probs4 = torch.clamp(probs4, min=0.0, max=1.0)
        return probs4