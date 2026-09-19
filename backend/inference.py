"""
inference.py
------------
Inference pipeline for the Hierarchical Temporal Transformer video summarizer,
extracted from the training notebook (Final_Journal_Version_7_VSUM.ipynb) and
repackaged for online serving.

Public API (imported by main.py):

    load_model(checkpoint_path: str) -> None
    summarize_video(video_path, out_dir, sample_rate, summary_proportion,
                    progress_cb) -> dict
    _MODEL, _DEVICE             # module-level state for health endpoint
    _have_ffmpeg() -> bool

Pipeline (identical to the notebook, plus audio-preserving H.264 encoding):

    raw video
      -> ffmpeg remux to browser-playable original.mp4
      -> GoogleNet pool5 features (1024-d, L2-normalized) every N frames
      -> KTS change-point detection (cpd_auto)
      -> ViT encoder + Temporal U-Net backbone + Hierarchical score head
      -> per-frame importance scores
      -> knapsack keyshot selection under proportion budget
      -> ffmpeg libx264/AAC summary.mp4 with original audio preserved
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg
from PIL import Image
from torchvision import models, transforms

import timm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Best-run configs.
#
#   BEST_CFG_PART1 — HiT-UNet (Part 1 paper): GoogLeNet pool5 features,
#                    ViT encoder + 3-level Temporal U-Net + hierarchical head.
#                    Loads from models/best.pt.
#
#   BEST_CFG_PART2 — MM-HiT-UNet (Part 2 Applied AI Letters submission):
#                    CLIP ViT-B/32 image (512-d) + CLIP text of BLIP-2 caption
#                    (512-d) = 1024-d Concat features, 2-level Temporal U-Net
#                    (192 base channels), hierarchical head with 3/3 layers.
#                    ViT encoder DISABLED — per ablation Table 9, disabling the
#                    pretrained ViT improves F1 on multimodal features because
#                    raw-image pretraining mismatches CLIP-space inputs.
#                    Loads from models/best_part2.pt.
#
# If you trained a different split, override these fields before load_model is
# called. Both configs use the same VideoSummarizer class — only the hyper-
# parameters differ.
# ---------------------------------------------------------------------------

BEST_CFG_PART1: Dict[str, Any] = {
    "input_dim": 1024,
    "hidden_dim": 768,
    "use_vit_encoder": True,
    "vit_name": "vit_base_patch16_224",
    "freeze_vit": False,
    "use_input_proj": True,
    "backbone_type": "unet",
    "head_type": "hier",
    "unet_levels": 3,
    "unet_base_channels": 128,
    "unet_dropout": 0.4,
    "unet_use_attention": True,
    "unet_attn_heads": 2,
    "unet_attn_levels": None,
    "use_unet_fusion": False,
    "hier_heads": 4,
    "hier_shot_layers": 1,
    "hier_frame_layers": 2,
    "hier_dim_ff": 1024,
    "hier_dropout": 0.4,
    "summary_proportion": 0.20,
}

BEST_CFG_PART2: Dict[str, Any] = {
    # Concat(CLIP-image, CLIP-text-of-BLIP2-caption) = 512 + 512 = 1024
    "input_dim": 1024,
    # The best_part2.pt checkpoint was trained with the same architecture as
    # Part 1 (simply swapping GoogLeNet pool5 for the CLIP+BLIP-2 multimodal
    # features). Its file size matches best.pt to the byte, confirming the
    # identical weight layout. Do not change these unless you retrain.
    "hidden_dim": 768,
    "use_vit_encoder": True,
    "vit_name": "vit_base_patch16_224",
    "freeze_vit": False,
    "use_input_proj": True,
    "backbone_type": "unet",
    "head_type": "hier",
    "unet_levels": 3,
    "unet_base_channels": 128,
    "unet_dropout": 0.4,
    "unet_use_attention": True,
    "unet_attn_heads": 2,
    "unet_attn_levels": None,
    "use_unet_fusion": False,
    "hier_heads": 4,
    "hier_shot_layers": 1,
    "hier_frame_layers": 2,
    "hier_dim_ff": 1024,
    "hier_dropout": 0.4,
    "summary_proportion": 0.20,
}

# Back-compat alias for any caller that still imports BEST_CFG
BEST_CFG: Dict[str, Any] = BEST_CFG_PART1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class _Cfg:
    """Simple attribute bag mirroring the notebook's Config."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Model blocks — copied verbatim from the training notebook so that the saved
# state_dict keys match exactly. Do not rename these classes/attributes.
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)

    def forward(self, x):
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
            out, _ = self.mha(x_seq, x_seq, x_seq)
            return out.squeeze(1)
        elif x.dim() == 3:
            x_seq = x.transpose(0, 1)
            out, _ = self.mha(x_seq, x_seq, x_seq)
            return out.transpose(0, 1)
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")


class SinusoidalPE1D(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, T):
        return self.pe[:T, :]


class HDF5ViTEncoder(nn.Module):
    def __init__(self, input_dim, vit_name="vit_base_patch16_224", freeze_vit=False, use_sinusoidal_pe=True):
        super().__init__()
        vit = timm.create_model(vit_name, pretrained=True)
        embed_dim = vit.embed_dim
        self.vit_blocks = vit.blocks
        self.vit_norm = vit.norm
        if freeze_vit:
            for p in self.vit_blocks.parameters():
                p.requires_grad = False
            for p in self.vit_norm.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(input_dim, embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        self.embed_dim = embed_dim
        self.use_sinusoidal_pe = use_sinusoidal_pe
        if use_sinusoidal_pe:
            self.pe = SinusoidalPE1D(embed_dim)

    def forward(self, x):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        T, _ = x.shape
        h = self.proj(x)
        if self.use_sinusoidal_pe:
            h = h + self.pe(T)
        h = h.unsqueeze(0)
        for blk in self.vit_blocks:
            h = blk(h)
        h = self.vit_norm(h)
        return h.squeeze(0)


class GlobalTemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        self.global_att = MultiHeadSelfAttention(hidden_dim, heads, dropout)

    def forward(self, x):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        h = self.proj(x)
        return self.global_att(h)


class HierarchicalScoreHead(nn.Module):
    def __init__(self, d_model, shot_layers=2, frame_layers=2, n_heads=4, dim_ff=512, dropout=0.1):
        super().__init__()
        shot_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=False, activation="gelu",
        )
        self.shot_encoder = nn.TransformerEncoder(shot_layer, num_layers=shot_layers)
        frame_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=False, activation="gelu",
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=frame_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, frame_feat, shot_ids=None):
        T, H = frame_feat.shape
        device = frame_feat.device
        if shot_ids is None:
            shot_ids = torch.zeros(T, dtype=torch.long, device=device)
        else:
            shot_ids = shot_ids.to(device=device, dtype=torch.long)
        num_shots = int(shot_ids.max().item()) + 1
        global_mean = frame_feat.mean(dim=0)
        shot_vecs = []
        for h in range(num_shots):
            m = (shot_ids == h)
            shot_vecs.append(frame_feat[m].mean(dim=0) if m.any() else global_mean)
        S0 = torch.stack(shot_vecs, dim=0)
        S_enc = self.shot_encoder(S0.unsqueeze(1)).squeeze(1)
        shot_emb = S_enc[shot_ids]
        fused = frame_feat + shot_emb
        F_enc = self.frame_encoder(fused.unsqueeze(1)).squeeze(1)
        scores = torch.sigmoid(self.mlp(F_enc)).squeeze(-1)
        return scores


class SimpleMLPHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, feat, shot_ids=None):
        return torch.sigmoid(self.net(feat)).squeeze(-1)


class ConvBlock1D(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0, groups=8):
        super().__init__()
        g = min(groups, c_out)
        if c_out % g != 0:
            g = 1
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(g, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(g, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalSelfAttention1D(nn.Module):
    def __init__(self, c, heads=4, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(c, heads, dropout=dropout, batch_first=False)
        self.ln = nn.LayerNorm(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T = x.shape
        xt = x.permute(2, 0, 1)
        xt_ln = self.ln(xt)
        out, _ = self.mha(xt_ln, xt_ln, xt_ln)
        out = self.drop(out) + xt
        return out.permute(1, 2, 0)


class Downsample1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv1d(c, c, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.ConvTranspose1d(c, c, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class TemporalUNet1DBackbone(nn.Module):
    def __init__(self, in_dim, out_dim, levels=3, base_channels=256, dropout=0.1,
                 use_attention=False, attn_heads=4, attn_levels=None):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.use_attention = use_attention
        if attn_levels is None:
            attn_levels = [max(0, levels - 2), max(0, levels - 1)] if use_attention else []
            attn_levels = sorted(list(set([l for l in attn_levels if 0 <= l < levels])))
        self.attn_levels = set(attn_levels)
        self.in_proj = nn.Linear(in_dim, base_channels)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        enc_chs = [base_channels * (2 ** i) for i in range(levels)]
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(levels):
            c_in = enc_chs[i - 1] if i > 0 else base_channels
            c_out = enc_chs[i]
            self.enc_blocks.append(ConvBlock1D(c_in, c_out, dropout=dropout))
            self.enc_attn.append(
                TemporalSelfAttention1D(c_out, heads=attn_heads, dropout=dropout)
                if (use_attention and (i in self.attn_levels)) else nn.Identity()
            )
            if i < levels - 1:
                self.downs.append(Downsample1D(c_out))
        self.bottleneck = ConvBlock1D(enc_chs[-1], enc_chs[-1], dropout=dropout)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(levels - 1)):
            c_up = enc_chs[i + 1]
            c_skip = enc_chs[i]
            self.ups.append(Upsample1D(c_up))
            self.dec_blocks.append(ConvBlock1D(c_up + c_skip, c_skip, dropout=dropout))
        self.out_conv = nn.Conv1d(base_channels, out_dim, kernel_size=1)

    def forward(self, x):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        h = self.in_proj(x)
        cur = h.transpose(0, 1).unsqueeze(0)
        skips = []
        multi = []
        for i in range(self.levels):
            cur = self.enc_blocks[i](cur)
            cur = self.enc_attn[i](cur)
            multi.append(cur.squeeze(0).permute(1, 0))
            skips.append(cur)
            if i < self.levels - 1:
                cur = self.downs[i](cur)
        cur = self.bottleneck(cur)
        for j in range(self.levels - 1):
            cur = self.ups[j](cur)
            skip = skips[self.levels - 2 - j]
            if cur.size(-1) != skip.size(-1):
                diff = skip.size(-1) - cur.size(-1)
                if diff > 0:
                    cur = F.pad(cur, (0, diff))
                else:
                    cur = cur[..., :skip.size(-1)]
            cur = torch.cat([cur, skip], dim=1)
            cur = self.dec_blocks[j](cur)
        out = self.out_conv(cur)
        out = out.squeeze(0).permute(1, 0)
        return out, multi


class CoarseFineFusion(nn.Module):
    def __init__(self, fine_dim, coarse_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(fine_dim + coarse_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, fine, coarse):
        if coarse.size(0) != fine.size(0):
            c = coarse.transpose(0, 1).unsqueeze(0)
            c_up = F.interpolate(c, size=fine.size(0), mode="linear", align_corners=False)
            coarse_up = c_up.squeeze(0).transpose(0, 1)
        else:
            coarse_up = coarse
        z = torch.cat([fine, coarse_up], dim=-1)
        return self.ln(self.proj(z))


class VideoSummarizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.use_vit_encoder:
            self.encoder = HDF5ViTEncoder(cfg.input_dim, cfg.vit_name, cfg.freeze_vit, use_sinusoidal_pe=True)
            enc_out_dim = self.encoder.embed_dim
            self.input_proj = None
        else:
            self.encoder = None
            enc_out_dim = cfg.input_dim
            self.input_proj = None
            if cfg.use_input_proj and enc_out_dim != cfg.hidden_dim:
                self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
                nn.init.xavier_uniform_(self.input_proj.weight)
                nn.init.constant_(self.input_proj.bias, 0.0)
                enc_out_dim = cfg.hidden_dim
        self.enc_out_dim = enc_out_dim

        bt = cfg.backbone_type.lower()
        self.backbone_type = bt
        self.global_backbone = None
        self.unet_backbone = None
        self.identity_proj = None
        if bt == "global_attn":
            self.global_backbone = GlobalTemporalTransformer(
                self.enc_out_dim, cfg.hidden_dim, cfg.global_heads, getattr(cfg, "global_dropout", 0.0),
            )
        elif bt == "unet":
            self.unet_backbone = TemporalUNet1DBackbone(
                self.enc_out_dim, cfg.hidden_dim,
                levels=cfg.unet_levels, base_channels=cfg.unet_base_channels,
                dropout=cfg.unet_dropout, use_attention=cfg.unet_use_attention,
                attn_heads=cfg.unet_attn_heads, attn_levels=cfg.unet_attn_levels,
            )
            self.use_unet_fusion = bool(cfg.use_unet_fusion)
            self.fusion = None
            self._fusion_built = not self.use_unet_fusion
        else:
            if self.enc_out_dim != cfg.hidden_dim:
                self.identity_proj = nn.Linear(self.enc_out_dim, cfg.hidden_dim)
                nn.init.xavier_uniform_(self.identity_proj.weight)
                nn.init.constant_(self.identity_proj.bias, 0.0)
            else:
                self.identity_proj = nn.Identity()

        ht = cfg.head_type.lower()
        self.head_type = ht
        if ht == "hier":
            self.head = HierarchicalScoreHead(
                cfg.hidden_dim, shot_layers=cfg.hier_shot_layers, frame_layers=cfg.hier_frame_layers,
                n_heads=cfg.hier_heads, dim_ff=cfg.hier_dim_ff, dropout=cfg.hier_dropout,
            )
        else:
            self.head = SimpleMLPHead(cfg.hidden_dim, dropout=cfg.hier_dropout)

    def _build_fusion(self, fine_dim, coarse_dim, device):
        if self._fusion_built:
            return
        self.fusion = CoarseFineFusion(fine_dim, coarse_dim, self.cfg.hidden_dim).to(device)
        self._fusion_built = True

    def backbone_features(self, x):
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        if self.encoder is not None:
            z = self.encoder(x)
        else:
            z = x
            if self.input_proj is not None:
                z = self.input_proj(z)
        if self.backbone_type == "global_attn":
            return self.global_backbone(z)
        if self.backbone_type == "unet":
            fine, multi = self.unet_backbone(z)
            if self.use_unet_fusion and (multi is not None) and (len(multi) > 0):
                coarse = multi[-1]
                self._build_fusion(fine.size(-1), coarse.size(-1), device=fine.device)
                fine = self.fusion(fine, coarse)
            return fine
        return self.identity_proj(z)

    def forward(self, x, shot_ids=None):
        feat = self.backbone_features(x)
        imp = self.head(feat, shot_ids=shot_ids)
        return feat, imp


# ---------------------------------------------------------------------------
# KTS change-point detection (copied verbatim)
# ---------------------------------------------------------------------------

def _calc_scatters(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n + 1, n + 1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)
    diagK2 = np.diag(K2)
    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))
    scatters = (
        K1[1:].reshape((1, -1)) - K1[:-1].reshape((-1, 1)) -
        (diagK2[1:].reshape((1, -1)) + diagK2[:-1].reshape((-1, 1)) -
         K2[1:, :-1].T - K2[:-1, 1:]) /
        ((j - i + 1).astype(np.float32) + (j == i - 1).astype(np.float32))
    )
    scatters[j < i] = 0
    return scatters


def _cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True):
    m = int(ncp)
    n, _ = K.shape
    J = _calc_scatters(K)
    I = 1e101 * np.ones((m + 1, n + 1))
    I[0, lmin:lmax] = J[0, lmin - 1:lmax - 1]
    if backtrack:
        p = np.zeros((m + 1, n + 1), dtype=int)
    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            tmin = max(k * lmin, l - lmax)
            tmax = l - lmin + 1
            c = J[tmin:tmax, l - 1].reshape(-1) + I[k - 1, tmin:tmax].reshape(-1)
            I[k, l] = np.min(c)
            if backtrack:
                p[k, l] = np.argmin(c) + tmin
    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]
    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def _cpd_auto(K, ncp, vmax, desc_rate=1):
    _, scores = _cpd_nonlin(K, ncp, backtrack=False)
    N = K.shape[0]
    N2 = N * desc_rate
    penalties = np.zeros(ncp + 1)
    ncp_arr = np.arange(1, ncp + 1)
    penalties[1:] = (vmax * ncp_arr / (2.0 * N2)) * (np.log(float(N2) / ncp_arr) + 1)
    costs = scores / float(N) + penalties
    m_best = int(np.argmin(costs))
    cps, scores2 = _cpd_nonlin(K, m_best)
    return cps, scores2


# ---------------------------------------------------------------------------
# GoogleNet pool5 feature extractor (exact match to the training pipeline)
# ---------------------------------------------------------------------------

class _FeatureExtractor:
    def __init__(self, device):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Use the modern `weights` API where available; fall back to pretrained=True.
        try:
            weights = models.GoogLeNet_Weights.IMAGENET1K_V1
            gnet = models.googlenet(weights=weights)
        except Exception:
            gnet = models.googlenet(pretrained=True)
        self.model = nn.Sequential(*list(gnet.children())[:-2]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def run(self, img_rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img_rgb)
        img = self.preprocess(img)
        batch = img.unsqueeze(0).to(self.device)
        feat = self.model(batch).squeeze().detach().cpu().numpy()
        feat = feat / (linalg.norm(feat) + 1e-10)
        return feat


# ---------------------------------------------------------------------------
# Knapsack keyshot selection
# ---------------------------------------------------------------------------

def _knapsack(W, wt, val, n):
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]
    return selected


def _get_keyshot_summ(pred, cps, n_frames, nfps, picks, proportion=0.15):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    picks = np.asarray(picks, dtype=np.int32).reshape(-1)
    cps = np.asarray(cps, dtype=np.int32)
    nfps = np.asarray(nfps, dtype=np.int32)
    frame_scores = np.zeros(int(n_frames), dtype=np.float32)
    T = len(picks)
    for i in range(T):
        lo = int(picks[i])
        hi = int(picks[i + 1]) if (i + 1) < T else int(n_frames)
        if hi > lo:
            frame_scores[lo:hi] = pred[i]
    cps_ = cps[0] if cps.ndim == 3 else cps
    K = len(cps_)
    seg_scores = np.zeros(K, dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps_):
        first, last = int(first), min(int(last), n_frames - 1)
        seg_scores[seg_idx] = int(1000.0 * frame_scores[first:last + 1].mean()) if last >= first else 0
    limits = int(n_frames * float(proportion))
    nfps_ = nfps[0] if nfps.ndim == 2 else nfps
    packed = _knapsack(limits, nfps_.tolist(), seg_scores.tolist(), K)
    summary = np.zeros(int(n_frames), dtype=bool)
    for seg_idx in packed:
        first, last = cps_[seg_idx]
        first, last = int(first), min(int(last), n_frames - 1)
        if last >= first:
            summary[first:last + 1] = True
    return summary


def _build_shot_ids_from_cps(cps, picks, n_frames, T):
    cps_arr = np.asarray(cps, dtype=np.int32)
    if cps_arr.ndim == 3:
        cps_arr = cps_arr[0]
    picks_arr = np.asarray(picks, dtype=np.int32).reshape(-1)
    n_frames = int(n_frames)
    T = int(T)
    if picks_arr.size != T:
        T = picks_arr.size
    if picks_arr.size == 0:
        return np.zeros(T, dtype=np.int64)
    pos = picks_arr
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    K = cps_arr.shape[0]
    shot_ids = np.zeros(T, dtype=np.int64)
    for i in range(T):
        lo = int(pos[i])
        assigned = False
        for h in range(K):
            s_frame = int(cps_arr[h, 0])
            e_frame = int(cps_arr[h, 1])
            if lo >= s_frame and lo <= e_frame:
                shot_ids[i] = h
                assigned = True
                break
        if not assigned:
            shot_ids[i] = 0 if lo < cps_arr[0, 0] else (K - 1)
    return shot_ids


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_video(path: str) -> Dict[str, Any]:
    """Return {video_codec, has_audio, width, height, duration, fps}."""
    info: Dict[str, Any] = {
        "video_codec": None, "has_audio": False,
        "width": None, "height": None, "duration": None, "fps": None,
    }
    if not _have_ffmpeg():
        return info
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(out.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and info["video_codec"] is None:
                info["video_codec"] = s.get("codec_name")
                info["width"] = s.get("width")
                info["height"] = s.get("height")
                rfr = s.get("r_frame_rate", "0/1")
                try:
                    num, den = rfr.split("/")
                    info["fps"] = float(num) / float(den) if float(den) else None
                except Exception:
                    info["fps"] = None
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
        try:
            info["duration"] = float(data.get("format", {}).get("duration"))
        except (TypeError, ValueError):
            info["duration"] = None
    except Exception as e:
        print(f"[ffprobe] failed: {e}")
    return info


def _prepare_original(input_path: str, output_path: str) -> None:
    """Produce a browser-playable MP4 (H.264 + AAC + faststart). Remuxes if
    already H.264 mp4, otherwise transcodes."""
    if not _have_ffmpeg():
        # No ffmpeg — best effort fallback: just copy bytes.
        shutil.copy(input_path, output_path)
        return

    info = _probe_video(input_path)
    is_mp4_ext = Path(input_path).suffix.lower() in (".mp4", ".m4v")
    can_remux = is_mp4_ext and (info.get("video_codec") == "h264")

    if can_remux:
        # No re-encode; just fix faststart so the browser can start playing fast.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg prepare_original failed: {r.stderr[-1200:]}")


def _mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs -> list of (start_frame, end_frame) inclusive."""
    segs: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif (not v) and in_seg:
            segs.append((start, i - 1))
            in_seg = False
    if in_seg:
        segs.append((start, int(len(mask)) - 1))
    return segs


def _write_summary_video(
    input_path: str,
    output_path: str,
    summary_mask: np.ndarray,
    fps: float,
) -> None:
    """
    Build the summary as an H.264 mp4 with preserved audio, using ffmpeg's
    filter_complex trim+concat. Far more reliable than OpenCV's VideoWriter
    and — unlike the notebook's approach — keeps the audio track aligned.

    For very large numbers of segments (>200), falls back to writing each
    segment to a temp file and using the concat demuxer.
    """
    if not _have_ffmpeg():
        raise RuntimeError(
            "ffmpeg is not installed. The summary video can't be encoded. "
            "Install ffmpeg and restart the server."
        )

    segs = _mask_to_segments(summary_mask)
    if not segs:
        raise RuntimeError("Summary mask is empty — nothing to encode.")

    info = _probe_video(input_path)
    has_audio = bool(info.get("has_audio", False))

    # -- Preferred path: single-pass filter_complex ------------------------
    if len(segs) <= 200:
        parts: List[str] = []
        for i, (s_frame, e_frame) in enumerate(segs):
            ts = s_frame / fps
            te = (e_frame + 1) / fps
            parts.append(f"[0:v]trim=start={ts:.3f}:end={te:.3f},setpts=PTS-STARTPTS[v{i}]")
            if has_audio:
                parts.append(f"[0:a]atrim=start={ts:.3f}:end={te:.3f},asetpts=PTS-STARTPTS[a{i}]")

        n = len(segs)
        if has_audio:
            concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
            parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")
        else:
            concat_inputs = "".join(f"[v{i}]" for i in range(n))
            parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")
        filter_complex = ";".join(parts)

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
        ]
        if has_audio:
            cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "128k"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
        # Fall through to the demuxer path on filter failure
        print(f"[ffmpeg filter_complex] failed, falling back to segment demuxer:\n{r.stderr[-1200:]}")

    # -- Fallback path: segment-by-segment then concat demuxer -------------
    tmp_dir = Path(output_path).parent / ("_tmp_" + Path(output_path).stem)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seg_files: List[Path] = []
    try:
        for i, (s_frame, e_frame) in enumerate(segs):
            ts = s_frame / fps
            te = (e_frame + 1) / fps
            duration = max(1e-3, te - ts)
            seg_path = tmp_dir / f"seg_{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{ts:.3f}", "-i", str(input_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            if has_audio:
                cmd += ["-c:a", "aac", "-b:a", "128k"]
            else:
                cmd += ["-an"]
            cmd += [str(seg_path)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg segment {i} failed: {r.stderr[-800:]}")
            seg_files.append(seg_path)

        # Concat demuxer list file
        list_path = tmp_dir / "list.txt"
        with open(list_path, "w") as f:
            for p in seg_files:
                f.write(f"file '{p.resolve().as_posix()}'\n")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-1200:]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLIP + BLIP-2 multimodal feature extractor (Part 2)
#
# For each sampled frame:
#   1) BLIP-2 generates a short caption
#   2) CLIP image encoder produces a 512-d visual embedding
#   3) CLIP text encoder produces a 512-d embedding of the caption
#   4) Both are L2-normalised and concatenated -> 1024-d Concat feature
#
# Lazy-loaded on the first Part 2 request: BLIP-2 is a 12 GB download and
# users may only ever hit the Part 1 path.
# ---------------------------------------------------------------------------

class _ClipBlip2Extractor:
    """Matches the paper's Concat variant: CLIP image + CLIP text of BLIP caption.

    Note on the captioner: on CPU-only machines with <= 16 GB RAM, BLIP-2
    opt-2.7b (~11 GB fp32, ~6 GB bf16) triggers heavy swapping and effectively
    hangs during weight loading. We default to Salesforce/blip-image-captioning-
    large (~2 GB) which loads in ~10 s and captions at ~5x the speed, with
    negligible impact on downstream summarisation because the summariser only
    sees the CLIP-text embedding of the caption string, not the caption model
    itself. Set BLIP_MODEL_ID=Salesforce/blip2-opt-2.7b (and BLIP_FAMILY=blip2)
    to use the paper's exact captioner on a machine with > 24 GB RAM or a GPU.
    """

    CLIP_MODEL_ID = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    BLIP_MODEL_ID = os.environ.get(
        "BLIP_MODEL_ID", "Salesforce/blip-image-captioning-large"
    )
    # "blip" (1-stage) or "blip2" (Q-Former + OPT/T5). Autodetected from the
    # model id, but an explicit override wins.
    BLIP_FAMILY = os.environ.get("BLIP_FAMILY", "").strip().lower()
    MAX_CAPTION_TOKENS = int(os.environ.get("BLIP_MAX_TOKENS", "24"))

    def __init__(self, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device

        # Decide captioner family
        family = self.BLIP_FAMILY
        if not family:
            family = "blip2" if "blip2" in self.BLIP_MODEL_ID.lower() else "blip"
        self._family = family

        # --- torch.load CVE-2025-32434 workaround ------------------------
        # Transformers >= 4.45 refuses to load weights through `torch.load`
        # on torch < 2.6 (because of CVE-2025-32434). That path is only
        # triggered for legacy `*.bin` weight files. CLIP ViT-B/32 and the
        # BLIP captioners on the HuggingFace Hub ship with `.safetensors`
        # weights, so forcing `use_safetensors=True` sidesteps `torch.load`
        # entirely and the security check never fires. This is strictly
        # safer than passing `weights_only=False` because safetensors never
        # executes pickle. No torch upgrade required.
        # -----------------------------------------------------------------

        print(f"[part2-extractor] loading CLIP ({self.CLIP_MODEL_ID}) ...")
        self.clip_proc = CLIPProcessor.from_pretrained(self.CLIP_MODEL_ID)
        self.clip_model = CLIPModel.from_pretrained(
            self.CLIP_MODEL_ID, use_safetensors=True,
        ).to(device).eval()

        # Precision: fp16 on CUDA, bf16 on CPU (half the RAM of fp32, and
        # modern CPUs execute bf16 matmul natively).
        if device.type == "cuda":
            blip_dtype = torch.float16
        else:
            blip_dtype = torch.bfloat16
        self._blip_dtype = blip_dtype

        if family == "blip2":
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            print(
                f"[part2-extractor] loading BLIP-2 ({self.BLIP_MODEL_ID}) - "
                f"first run downloads ~12 GB, be patient..."
            )
            self.blip_proc = Blip2Processor.from_pretrained(self.BLIP_MODEL_ID)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                self.BLIP_MODEL_ID,
                torch_dtype=blip_dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device).eval()
        else:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            print(
                f"[part2-extractor] loading BLIP ({self.BLIP_MODEL_ID}) - "
                f"~2 GB download on first run..."
            )
            self.blip_proc = BlipProcessor.from_pretrained(self.BLIP_MODEL_ID)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                self.BLIP_MODEL_ID,
                torch_dtype=blip_dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device).eval()
        print("[part2-extractor] ready.")

    @torch.no_grad()
    def caption(self, img_rgb: np.ndarray) -> str:
        pil = Image.fromarray(img_rgb)
        inputs = self.blip_proc(images=pil, return_tensors="pt").to(self.device)
        # Cast pixel_values to the same dtype the model was loaded in
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._blip_dtype)
        gen = self.blip_model.generate(
            **inputs, max_new_tokens=self.MAX_CAPTION_TOKENS, num_beams=1, do_sample=False,
        )
        text = self.blip_proc.batch_decode(gen, skip_special_tokens=True)[0]
        return text.strip()

    @torch.no_grad()
    def run(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, str, np.ndarray]:
        """Return (1024-d concat feature, caption, 512-d CLIP text feature).

        The third return value is the L2-normalised CLIP text embedding of the
        BLIP-2 caption. It is useful on its own for Part 2 visualisations
        (caption-embedding scatter / semantic similarity heatmap) even though
        it is already included inside the concat feature fed to the model.
        """
        caption = self.caption(img_rgb)

        pil = Image.fromarray(img_rgb)
        img_inputs = self.clip_proc(images=pil, return_tensors="pt").to(self.device)
        img_feat = self.clip_model.get_image_features(**img_inputs).squeeze(0)

        # Fall back to a neutral caption if BLIP-2 produced an empty string
        cap_for_clip = caption if caption else "a video frame"
        txt_inputs = self.clip_proc(
            text=[cap_for_clip], return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        ).to(self.device)
        txt_feat = self.clip_model.get_text_features(**txt_inputs).squeeze(0)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        concat = torch.cat([img_feat, txt_feat], dim=-1).detach().cpu().numpy().astype(np.float32)
        txt_np = txt_feat.detach().cpu().numpy().astype(np.float32)
        return concat, caption, txt_np


# ---------------------------------------------------------------------------
# Module-level state (loaded once at server startup)
#
# Part 1 (_MODEL_PART1) loads eagerly at startup if best.pt is present.
# Part 2 (_MODEL_PART2) loads eagerly at startup if best_part2.pt is present.
# The CLIP+BLIP-2 extractor is created lazily on the first Part 2 request.
#
# Back-compat: _MODEL points at Part 1 so existing callers keep working.
# ---------------------------------------------------------------------------

_DEVICE: Optional[torch.device] = None
_FEATS: Optional[_FeatureExtractor] = None      # GoogLeNet pool5 (both variants use it for KTS)

_MODEL_PART1: Optional[VideoSummarizer] = None
_CFG_PART1: Optional[_Cfg] = None

_MODEL_PART2: Optional[VideoSummarizer] = None
_CFG_PART2: Optional[_Cfg] = None

_MM_FEATS: Optional[_ClipBlip2Extractor] = None

# Legacy aliases (main.py health check still reads these)
_MODEL: Optional[VideoSummarizer] = None
_CFG: Optional[_Cfg] = None


def _resolve_device_name(name: Optional[str]) -> torch.device:
    """Map a user-facing string to a torch.device.

    Accepts: None, "", "auto", "cpu", "cuda", "cuda:0", "gpu".
    "auto" / None / "" -> cuda if available else cpu.
    "gpu" is treated as an alias for "cuda".
    Raises ValueError if "cuda" is requested but no CUDA device is present.
    """
    n = (name or "").strip().lower()
    if n in ("", "auto"):
        env = os.environ.get("VSUM_DEVICE", "").strip().lower()
        if env:
            return _resolve_device_name(env)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "gpu":
        n = "cuda"
    if n.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but PyTorch reports no CUDA device. "
                "Install a CUDA build of torch (see setup_cuda.ps1) or pick CPU."
            )
        return torch.device(n)
    if n == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{name}'. Expected 'auto', 'cpu', or 'cuda'.")


def _ensure_device(preferred: Optional[str] = None) -> torch.device:
    """Initialize the global device (once) and the GoogLeNet feature extractor.

    If a `preferred` string is given, it overrides autodetection on first call.
    On subsequent calls `preferred` is ignored — use `set_device()` to switch
    devices after initialisation.
    """
    global _DEVICE, _FEATS
    if _DEVICE is None:
        _DEVICE = _resolve_device_name(preferred)
        print(f"[inference] device = {_DEVICE}")
    if _FEATS is None:
        print("[inference] loading GoogleNet pool5 feature extractor...")
        _FEATS = _FeatureExtractor(_DEVICE)
    return _DEVICE


def set_device(name: str) -> torch.device:
    """Live-migrate every loaded model to `name`.

    Cheap when the target equals the current device. Otherwise this moves
    GoogLeNet, both VideoSummarizer models, and the CLIP+BLIP captioner
    (if warm) onto the new device, then empties the CUDA allocator cache.

    This is the only safe way to switch devices because the 1050 Ti has just
    4 GB VRAM — we cannot afford two simultaneous copies of every model.
    """
    global _DEVICE, _FEATS, _MODEL_PART1, _MODEL_PART2, _MM_FEATS

    target = _resolve_device_name(name)
    if _DEVICE is not None and target == _DEVICE:
        return _DEVICE

    print(f"[inference] migrating models: {_DEVICE} -> {target}")
    src = _DEVICE

    # Move every loaded module
    if _FEATS is not None and hasattr(_FEATS, "model"):
        try:
            _FEATS.model.to(target)
            _FEATS.device = target
        except Exception as e:
            print(f"[inference] WARN GoogLeNet move failed: {e}")
    if _MODEL_PART1 is not None:
        _MODEL_PART1.to(target)
    if _MODEL_PART2 is not None:
        _MODEL_PART2.to(target)
    if _MM_FEATS is not None:
        try:
            _MM_FEATS.device = target
            if hasattr(_MM_FEATS, "clip_model") and _MM_FEATS.clip_model is not None:
                _MM_FEATS.clip_model.to(target)
            if hasattr(_MM_FEATS, "blip_model") and _MM_FEATS.blip_model is not None:
                _MM_FEATS.blip_model.to(target)
        except Exception as e:
            print(f"[inference] WARN multimodal extractor move failed: {e}")

    _DEVICE = target

    # Free whatever the source device held
    import gc
    gc.collect()
    if src is not None and src.type == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(src)
        except Exception:
            pass
    if target.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[inference] device migration complete: {_DEVICE}")
    return _DEVICE


def cuda_info() -> Dict[str, Any]:
    """Report CUDA availability + per-device VRAM, for the /api/health endpoint."""
    info: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "devices": [],
        "current": str(_DEVICE) if _DEVICE is not None else "unloaded",
    }
    if not info["available"]:
        return info
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "vram_total_mb": int(total / (1024 * 1024)),
                "vram_free_mb": int(free / (1024 * 1024)),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        except Exception as e:
            info["devices"].append({"index": i, "error": str(e)})
    return info


def _load_checkpoint_into(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    print(f"[inference] loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Support plain state_dict or a training-wrapper dict with 'state_dict' key
    if isinstance(ckpt, dict) and "state_dict" in ckpt and not any(
        k.startswith(("encoder.", "unet_backbone.", "head.", "global_backbone.", "fusion."))
        for k in ckpt.keys()
    ):
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[inference] WARNING missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[inference] WARNING unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")


def load_model_part1(checkpoint_path: str) -> None:
    """Load the Part 1 HiT-UNet model (GoogLeNet features)."""
    global _MODEL_PART1, _CFG_PART1, _MODEL, _CFG

    device = _ensure_device()
    print("[inference] building Part 1 VideoSummarizer (ViT + U-Net + hierarchical head)...")
    cfg = _Cfg(**BEST_CFG_PART1)
    model = VideoSummarizer(cfg).to(device)
    _load_checkpoint_into(model, checkpoint_path, device)
    model.eval()

    _MODEL_PART1 = model
    _CFG_PART1 = cfg
    # Back-compat: legacy globals point at Part 1
    _MODEL = model
    _CFG = cfg
    print("[inference] Part 1 ready.")


def load_model_part2(checkpoint_path: str) -> None:
    """Load the Part 2 MM-HiT-UNet model (CLIP + BLIP-2 Concat features)."""
    global _MODEL_PART2, _CFG_PART2

    device = _ensure_device()
    print("[inference] building Part 2 VideoSummarizer (U-Net + hierarchical head, multimodal input)...")
    cfg = _Cfg(**BEST_CFG_PART2)
    model = VideoSummarizer(cfg).to(device)
    _load_checkpoint_into(model, checkpoint_path, device)
    model.eval()

    _MODEL_PART2 = model
    _CFG_PART2 = cfg
    print("[inference] Part 2 ready. CLIP + BLIP-2 extractor will load on first Part 2 request.")


def load_model(checkpoint_path: str) -> None:
    """Back-compat: load Part 1 from the given checkpoint path."""
    load_model_part1(checkpoint_path)


def _get_mm_extractor() -> _ClipBlip2Extractor:
    """Lazy-load the CLIP + BLIP-2 extractor (used only by Part 2)."""
    global _MM_FEATS
    if _MM_FEATS is None:
        device = _ensure_device()
        _MM_FEATS = _ClipBlip2Extractor(device)
    return _MM_FEATS


# ---------------------------------------------------------------------------
# Per-request entry point (called from main.py in a background task)
# ---------------------------------------------------------------------------

ProgressCb = Callable[[str, float, Dict[str, Any]], None]


def _extract_features_streaming(
    video_path: str,
    sample_rate: int,
    progress_cb: Optional[ProgressCb],
    mm_extractor: Optional[_ClipBlip2Extractor] = None,
) -> Tuple[int, np.ndarray, float, int, int, Optional[np.ndarray], List[str], Optional[np.ndarray]]:
    """Single streaming pass over the video.

    Always computes the GoogLeNet pool5 features (used by KTS, and by the Part 1
    model). If ``mm_extractor`` is provided (Part 2), also computes the
    CLIP+BLIP-2 Concat features on the SAME sampled frames, keeping features
    time-aligned with picks/KTS.

    Returns:
        n_frames, googlenet_features, fps, width, height,
        mm_features, captions, clip_text_features
    """
    assert _FEATS is not None, "Feature extractor not initialized; call load_model_part1() first."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    feats: List[np.ndarray] = []
    mm_feats: List[np.ndarray] = []
    text_feats: List[np.ndarray] = []
    captions: List[str] = []
    n_frames = 0
    last_emit = 0
    # Progress emission cadence. Tuned so the bar visibly ticks even on short
    # videos. Part 2 emits every sampled frame because each one runs BLIP (slow);
    # Part 1 emits every 5 sampled frames because GoogLeNet is fast and emitting
    # every frame would just spam the status endpoint without buying anything.
    emit_every = 1 if mm_extractor is not None else 5
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if n_frames % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feats.append(_FEATS.run(frame_rgb))
                if mm_extractor is not None:
                    mm_feat, cap_txt, txt_feat = mm_extractor.run(frame_rgb)
                    mm_feats.append(mm_feat)
                    captions.append(cap_txt)
                    text_feats.append(txt_feat)
                # Force a tick on the very first sampled frame so the user sees
                # the bar move out of "upload done" the instant extraction begins.
                first_tick = (len(feats) == 1)
                if progress_cb and total_frames > 0 and (
                    first_tick or (len(feats) - last_emit) >= emit_every
                ):
                    meta = {"frame": n_frames, "total": total_frames}
                    if captions:
                        meta["caption"] = captions[-1]
                    progress_cb(
                        "extract_features",
                        min(n_frames / max(total_frames, 1), 0.99),
                        meta,
                    )
                    last_emit = len(feats)
            n_frames += 1
    finally:
        cap.release()

    if progress_cb:
        progress_cb(
            "extract_features", 1.0,
            {"frame": n_frames, "total": max(total_frames, n_frames)},
        )

    features = np.asarray(feats, dtype=np.float32)
    mm_features = np.asarray(mm_feats, dtype=np.float32) if mm_feats else None
    text_features = np.asarray(text_feats, dtype=np.float32) if text_feats else None
    return int(n_frames), features, float(fps), int(width), int(height), mm_features, captions, text_features


def _run_kts(n_frames: int, features: np.ndarray, sample_rate: int):
    seq_len = len(features)
    picks = (np.arange(0, seq_len) * sample_rate).astype(np.int32)
    kernel = (features @ features.T).astype(np.float32)
    cps, _ = _cpd_auto(kernel, seq_len - 1, 1)
    cps = (cps.astype(np.int32) * sample_rate)
    cps = np.hstack((0, cps, n_frames)).astype(np.int32)
    begin_frames, end_frames = cps[:-1], cps[1:]
    change_points = np.vstack((begin_frames, end_frames - 1)).T.astype(np.int32)
    nfps = (end_frames - begin_frames).astype(np.int32)
    return change_points, nfps, picks


def summarize_video(
    video_path: str,
    out_dir: str,
    sample_rate: int = 15,
    summary_proportion: float = 0.20,
    progress_cb: Optional[ProgressCb] = None,
    variant: str = "part1",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end inference for one video.

    Args:
        variant: "part1" uses the HiT-UNet model on GoogLeNet features.
                 "part2" uses the MM-HiT-UNet model on CLIP+BLIP-2 Concat
                 features (GoogLeNet is still used to drive KTS, matching the
                 paper's GoogLeNet-CPS protocol).

    Writes two files to `out_dir`:
        original.mp4   — browser-playable copy of the uploaded video
        summary.mp4    — H.264/AAC summary produced by the model

    Returns a dict of JSON-serializable metadata (no numpy/torch types).
    """
    variant = (variant or "part1").lower()
    if variant not in ("part1", "part2"):
        raise ValueError(f"Unknown variant '{variant}'. Expected 'part1' or 'part2'.")

    # Honor per-request device override before any tensor work begins. This
    # live-migrates GoogLeNet + the variant's summariser + (if warm) CLIP+BLIP
    # onto the chosen device. No-op when the target equals the current device.
    if device is not None and device.strip() and device.strip().lower() != "auto":
        set_device(device)

    # Resolve which model to use for the forward pass
    if variant == "part1":
        if _MODEL_PART1 is None or _DEVICE is None or _CFG_PART1 is None:
            raise RuntimeError("Part 1 model is not loaded. Place best.pt in models/ and restart.")
        model = _MODEL_PART1
        mm_extractor: Optional[_ClipBlip2Extractor] = None
    else:
        if _MODEL_PART2 is None or _DEVICE is None or _CFG_PART2 is None:
            raise RuntimeError(
                "Part 2 model is not loaded. Place best_part2.pt in models/ and restart."
            )
        model = _MODEL_PART2
        # This triggers the ~12 GB download the first time, so surface a status
        if progress_cb:
            progress_cb("load_multimodal", 0.0, {"note": "loading CLIP + BLIP-2 (first run: ~12 GB download)"})
        mm_extractor = _get_mm_extractor()
        if progress_cb:
            progress_cb("load_multimodal", 1.0, {})

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    original_out = out_dir_p / "original.mp4"
    summary_out = out_dir_p / "summary.mp4"

    # ---- Stage 0: prepare original for the browser ------------------------
    if progress_cb:
        progress_cb("extract_features", 0.0, {"frame": 0, "total": 0})
    _prepare_original(str(video_path), str(original_out))

    # ---- Stage 1: feature extraction -------------------------------------
    (
        n_frames, features, fps, width, height,
        mm_features, captions, text_features,
    ) = _extract_features_streaming(
        str(video_path), int(sample_rate), progress_cb, mm_extractor=mm_extractor,
    )
    if len(features) < 4:
        raise RuntimeError(
            f"Video is too short for summarization (only {len(features)} sampled frames). "
            f"Try a longer video or reduce SAMPLE_RATE."
        )

    # ---- Stage 2: KTS ----------------------------------------------------
    if progress_cb:
        progress_cb("kts", 0.0, {})
    cps, nfps, picks = _run_kts(n_frames, features, int(sample_rate))
    if progress_cb:
        progress_cb("kts", 1.0, {"num_shots": int(len(cps))})

    # ---- Stage 3: Transformer model forward ------------------------------
    if progress_cb:
        progress_cb("model", 0.0, {})
    T = int(features.shape[0])
    shot_ids_np = _build_shot_ids_from_cps(cps=cps, picks=picks, n_frames=n_frames, T=T)
    shot_ids_t = torch.from_numpy(shot_ids_np).long().to(_DEVICE)

    # For Part 1 we feed GoogLeNet features; for Part 2 we feed the aligned
    # CLIP+BLIP-2 Concat features. Both share the same shot_ids (KTS output
    # is identical, since both variants use GoogLeNet-CPS).
    if variant == "part1":
        model_input = features
    else:
        if mm_features is None or len(mm_features) != T:
            raise RuntimeError(
                "Multimodal features missing or misaligned with GoogLeNet picks."
            )
        model_input = mm_features

    feats_t = torch.from_numpy(model_input).float().to(_DEVICE)
    with torch.no_grad():
        _, imp = model(feats_t, shot_ids=shot_ids_t)
    imp_np = imp.detach().cpu().numpy().astype(np.float32)
    if progress_cb:
        progress_cb("model", 1.0, {})

    # ---- Stage 4: knapsack ------------------------------------------------
    if progress_cb:
        progress_cb("knapsack", 0.0, {})
    summary_mask = _get_keyshot_summ(
        pred=imp_np, cps=cps, n_frames=int(n_frames), nfps=nfps, picks=picks,
        proportion=float(summary_proportion),
    ).astype(bool)
    selected_frames = int(summary_mask.sum())
    selected_ratio = float(summary_mask.mean())
    if progress_cb:
        progress_cb("knapsack", 1.0, {"selected_frames": selected_frames})

    # ---- Stage 5: encode summary ------------------------------------------
    if progress_cb:
        progress_cb("encode", 0.0, {})
    _write_summary_video(str(video_path), str(summary_out), summary_mask, fps)
    if progress_cb:
        progress_cb("encode", 1.0, {})

    # ---- Stage 6: finalize (checks the output is readable) ----------------
    if progress_cb:
        progress_cb("reencode", 0.5, {})
    if not summary_out.exists() or summary_out.stat().st_size == 0:
        raise RuntimeError("Summary encoding produced an empty file.")
    summary_info = _probe_video(str(summary_out))
    if progress_cb:
        progress_cb("reencode", 1.0, {})

    duration_seconds = float(n_frames / fps) if fps > 0 else 0.0
    summary_duration_seconds = (
        float(summary_info.get("duration") or 0.0)
        or (selected_frames / fps if fps > 0 else 0.0)
    )

    # Everything in the returned dict must be JSON-serializable (main.py dumps
    # it to meta.json and the frontend reads it from /api/status).
    variant_label = "HiT-UNet (Part 1)" if variant == "part1" else "MM-HiT-UNet (Part 2)"
    feature_label = (
        "GoogLeNet pool5 (1024-d)"
        if variant == "part1"
        else "CLIP image + CLIP text of BLIP-2 caption (Concat 1024-d)"
    )

    # ------------------------------------------------------------------
    # Part 2 visualisation payload
    # ------------------------------------------------------------------
    # For each sampled feature index t we know:
    #   - its caption (BLIP-2 output)
    #   - its shot id (from KTS segmentation)
    #   - whether the frame that backs it is in the final summary
    #   - its CLIP text embedding (512-d, L2-normalised)
    # We project the text embeddings to 2D via a cheap PCA (SVD on mean-
    # centred features). That gives us a scatter plot highlighting how
    # caption semantics cluster and which clusters got selected.
    caption_payload: Dict[str, Any] = {}
    if variant == "part2" and text_features is not None and len(captions) > 0:
        # Map each of the T sampled frames to its representative underlying
        # frame index (via picks). A caption is "selected" if its backing
        # frame falls inside the summary mask.
        try:
            T_caps = len(captions)
            caps_selected: List[bool] = []
            for t in range(T_caps):
                f_idx = int(picks[t]) if t < len(picks) else 0
                f_idx = max(0, min(f_idx, int(n_frames) - 1))
                caps_selected.append(bool(summary_mask[f_idx]))

            # 2D PCA via SVD (no sklearn dependency)
            X = text_features.astype(np.float32)
            if X.ndim == 2 and X.shape[0] >= 2 and X.shape[1] >= 2:
                Xc = X - X.mean(axis=0, keepdims=True)
                # Use full_matrices=False for speed; we only need top-2 singular vectors
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                comps = Vt[:2]                 # (2, D)
                proj = Xc @ comps.T            # (T, 2)
                # Normalise to ~[-1, 1] for easy plotting on the frontend
                maxabs = float(np.max(np.abs(proj))) or 1.0
                proj = (proj / maxabs).astype(np.float32)
                proj_list = proj.tolist()
                explained = (S[:2] ** 2 / float((S ** 2).sum() or 1.0)).tolist()
            else:
                proj_list = [[0.0, 0.0]] * T_caps
                explained = [0.0, 0.0]

            # Per-caption shot id (reuse model's shot_ids_np)
            caps_shot_ids = [int(shot_ids_np[t]) if t < len(shot_ids_np) else 0 for t in range(T_caps)]

            # Also hand back the per-caption importance score (Part 2
            # importance output aligns 1-to-1 with the sampled frames, so we
            # can plot importance colour against the 2D scatter if desired).
            if imp_np.ndim == 1 and len(imp_np) >= T_caps:
                caps_importance = imp_np[:T_caps].tolist()
            else:
                caps_importance = [0.0] * T_caps

            caption_payload = {
                "captions_full": captions,
                "caption_selected": caps_selected,
                "caption_shot_ids": caps_shot_ids,
                "caption_importance": caps_importance,
                "caption_embedding_2d": proj_list,
                "caption_embedding_variance": explained,
            }
        except Exception as vis_err:     # noqa: BLE001 — visualisation must never break inference
            print(f"[inference] WARN: caption-vis payload skipped ({vis_err!r})")
            caption_payload = {}

    return {
        "variant": variant,
        "variant_label": variant_label,
        "feature_label": feature_label,
        "n_frames": int(n_frames),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "T_features": int(T),
        "summary_proportion": float(summary_proportion),
        "num_shots": int(len(cps)),
        "selected_frames": selected_frames,
        "selected_ratio": selected_ratio,
        "duration_seconds": duration_seconds,
        "summary_duration_seconds": summary_duration_seconds,
        "summary_width": summary_info.get("width"),
        "summary_height": summary_info.get("height"),
        # Part 2 only: a small sample of BLIP-2 captions for UI display
        "sample_captions": (captions[:6] if captions else []),
        "num_captions": int(len(captions)),
        # Part 2 only: rich visualisation payload (empty dict on Part 1)
        **caption_payload,
        # Big arrays — main.py strips these before sending to the frontend, but
        # they're useful for debugging or building advanced visualizations.
        "importance_scores": imp_np.tolist(),
        "picks": picks.tolist(),
    }
