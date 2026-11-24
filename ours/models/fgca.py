"""Frequency-guided cross-attention (FGCA) blocks of FSFNet (Section 3.3 of the paper).

The implementation of this module will be released upon acceptance of the paper.
This file defines the interface used by ``models/networks.py``.
"""
import torch.nn as nn


class FrequencyGuidedTransformer(nn.Module):
    """Stack of L Transformer blocks with FGCA and GAGM.

    In each block, the spatial feature is normalized by LayerNorm and split into two
    channel halves that provide the keys and values. Layer-specific projections of
    F_low and F_high provide the queries of a regular-window (W-MCA) and a
    shifted-window (SW-MCA) multi-head cross-attention stream, and the low- and
    high-frequency queries swap between the two streams across layers. The two
    streams are weighted by GAGM (channel-aware gate, position-aware gate driven by
    the granularity map M, and progressive fusion with coefficient alpha(e)),
    concatenated, projected, and followed by a pre-norm residual MLP.

    Args:
        depths (list[int]): number of blocks, L = 12.
        num_heads (list[int]): attention heads per stream, N_h = 12 (head dim 16).
        embed_dim (int): channel dimension C = 384.
        window_size (int): window size w = 4 (shift of 2 for SW-MCA).

    Inputs:
        x (Tensor): spatial bottleneck feature F_4, shape (B, 384, H/16, W/16).
        freq_low (Tensor): low-frequency representation F_low, shape (B, 192, H/16, W/16).
        freq_high (Tensor): high-frequency representation F_high, shape (B, 192, H/16, W/16).
        granularity_map (Tensor or None): granularity map M, shape (B, 1, H/16, W/16).
        alpha (float): progressive-fusion coefficient alpha(e) in [0, 1].

    Returns:
        Tensor: fused bottleneck feature F_fused, shape (B, 384, H/16, W/16).
    """

    def __init__(self, depths=(12,), num_heads=(12,), embed_dim=384, window_size=4):
        super().__init__()
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.embed_dim = embed_dim
        self.window_size = window_size

    def forward(self, x, freq_low, freq_high, granularity_map=None, alpha=0.0):
        raise NotImplementedError(
            "The FGCA and GAGM blocks will be released upon acceptance of the paper.")
