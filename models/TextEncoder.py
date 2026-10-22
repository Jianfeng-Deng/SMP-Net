import os
# 设置 Hugging Face 镜像站点（必须在导入 sentence_transformers 之前）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        embed_dim=256,
        freeze_backbone=True,
        device='cpu'
    ):
        """
        model_name: SentenceTransformer model name
        embed_dim: final embedding dim for multimodal alignment
        freeze_backbone: if True -> freeze SentenceTransformer weights (recommended)
        device: 'cuda' or 'cpu' where the projection will live and encode outputs will be sent
        """
        super().__init__()

        self.device = device
        self.text_model = SentenceTransformer(model_name)
        text_dim = self.text_model.get_sentence_embedding_dimension()

        # freeze only backbone if requested (DO NOT freeze proj)
        if freeze_backbone:
            for p in self.text_model.parameters():
                p.requires_grad = False

        # A small adapter/proj that is trainable
        # you may replace with a small MLP if you want
        self.proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # initialize proj (optional)
        nn.init.xavier_uniform_(self.proj[0].weight)
        if self.proj[0].bias is not None:
            nn.init.zeros_(self.proj[0].bias)

        # move proj to device
        self.to(self.device)

    def forward(self, texts):
        """
        texts: List[str], length = B
        returns: [B, embed_dim] tensor on self.device
        """
        # SentenceTransformer.encode supports device argument (so we avoid extra copies)
        # We run encode under torch.no_grad() because backbone is frozen and to save memory.
        with torch.no_grad():
            # convert_to_tensor True gives torch.Tensor directly on desired device
            text_feat = self.text_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )  # [B, text_dim], dtype=torch.float32 on device

        # Clone to escape inference-mode tensor constraint: tensors produced inside
        # torch.no_grad() cannot be saved for backward by autograd when proj is trainable.
        z_text = self.proj(text_feat.clone())  # [B, embed_dim]

        # optional: normalize output into unit sphere (if your other embeddings are normalized)
        z_text = F.normalize(z_text, dim=-1)

        return z_text