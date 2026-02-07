# Program: Cervical fracture detection of C1-C7 (stage 2)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   EfficientnetV2 backbone (timm) on intensity channels + small mask encoder.
#   Sequence model: BiLSTM + temporal attention pooling.
#
# Input:
#   x: (bs, T, 6, H, W)
#   Channels:
#     0..4 = intensity slices (float, normalized)
#     5    = vertebra mask (0/1)
#
# Output:
#   logits: (bs,)  binary logits
#
# Notes:
#   - This is 2D feature extraction per timestep, not 3D CNN.
#   - timm: use num_classes=0 + global_pool="avg". 

from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import timm


class MaskEncoder(nn.Module):
    """
    Tiny encoder for the mask channel.
    We keep it small because it's region of interest signal, not a second backbone.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_dim, 3, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(1)


class FractureSeqModel(nn.Module):
    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s_in21ft1k",
        pretrained: bool = True,
        intensity_chans: int = 5,
        mask_feat_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        drop_rate: float = 0.0,
        drop_rate_last: float = 0.3,
    ):
        super().__init__()

        self.intensity_chans = int(intensity_chans)

        self.encoder = timm.create_model(
            backbone,
            in_chans=self.intensity_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        hdim = self.encoder.num_features

        self.mask_encoder = MaskEncoder(out_dim=int(mask_feat_dim))
        feat_dim = hdim + int(mask_feat_dim)

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(drop_rate) if int(num_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
            batch_first=True,
        )
        lstm_out_dim = int(hidden_size) * (2 if bidirectional else 1)

        # Temporal attention pooling. This replaces the "take last timestep" hack.
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.LayerNorm(256),           # batch-size safe
            nn.Dropout(float(drop_rate_last)),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # only accept 5d tensor
        if x.ndim != 5:
            raise ValueError(f"Expected (bs,T,C,H,W) got {tuple(x.shape)}")

        bs, T, C, H, W = x.shape
        if C < self.intensity_chans + 1:
            raise ValueError(f"Expected >= {self.intensity_chans+1} channels, got {C}")

        x_int = x[:, :, :self.intensity_chans].reshape(bs * T, self.intensity_chans, H, W)
        x_msk = x[:, :, self.intensity_chans:self.intensity_chans+1].reshape(bs * T, 1, H, W)

        f_int = self.encoder(x_int)
        f_msk = self.mask_encoder(x_msk)

        feat = torch.cat([f_int, f_msk], dim=1).view(bs, T, -1)

        seq, _ = self.lstm(feat)

        att_logits = self.attn(seq)                 # (bs, T, 1)
        att_w = torch.softmax(att_logits, dim=1)    # (bs, T, 1)
        pooled = (seq * att_w).sum(dim=1)           # (bs, lstm_out_dim)

        logits = self.head(pooled).squeeze(-1)      # (bs,)
        return logits

    @torch.no_grad()
    def forward_with_attention(self, x: torch.Tensor):
        # Debugging function. Returns attention weights over timesteps.
        bs, T, C, H, W = x.shape
        x_int = x[:, :, :self.intensity_chans].reshape(bs * T, self.intensity_chans, H, W)
        x_msk = x[:, :, self.intensity_chans:self.intensity_chans+1].reshape(bs * T, 1, H, W)
        f_int = self.encoder(x_int)
        f_msk = self.mask_encoder(x_msk)
        feat = torch.cat([f_int, f_msk], dim=1).view(bs, T, -1)
        seq, _ = self.lstm(feat)
        att_w = torch.softmax(self.attn(seq), dim=1).squeeze(-1)  # (bs, T)
        pooled = (seq * att_w.unsqueeze(-1)).sum(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits, att_w


def build_model(cfg: Dict[str, Any]) -> FractureSeqModel:
    # cfg keys are explicit. If backbone is missing, that's a config error.
    return FractureSeqModel(
        backbone=str(cfg.get("backbone", "tf_efficientnetv2_s_in21ft1k")),
        pretrained=bool(cfg.get("pretrained", True)),
        intensity_chans=int(cfg.get("intensity_chans", 5)),
        mask_feat_dim=int(cfg.get("mask_feat_dim", 64)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        bidirectional=bool(cfg.get("bidirectional", True)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
        drop_rate_last=float(cfg.get("drop_rate_last", 0.3)),
    )
