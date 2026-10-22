import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class FrozenCLIP:
    def __init__(self, device):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        # ===== 延迟初始化 =====
        self.spec_adapter = None

    def _build_spec_adapter(self, in_channels):
        """
        Lazy init: 根据 spec channel 数自动创建 adapter
        """
        adapter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=1,
            bias=False
        ).to(self.device)

        # adapter 可训练
        for p in adapter.parameters():
            p.requires_grad = True

        self.spec_adapter = adapter

        print(f"[FrozenCLIP] Spec adapter initialized: {in_channels} → 3")

    @torch.no_grad()
    def encode_spec(self, spec):
        """
        spec:
            [B, H, W]
            [B, 1, H, W]
            [B, C, H, W]
        """

        spec = spec.float().to(self.device)

        # [B, H, W] → [B, 1, H, W]
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)

        # ===== Lazy adapter init =====
        if self.spec_adapter is None:
            self._build_spec_adapter(spec.size(1))

        # Spec → RGB
        spec = self.spec_adapter(spec)   # [B, 3, H, W]

        # resize for CLIP
        spec = F.interpolate(
            spec,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        emb = self.model.encode_image(spec)
        return F.normalize(emb, dim=-1)