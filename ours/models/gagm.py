"""Granularity map of the granularity-aware adaptive gating mechanism (GAGM)
of FSFNet (Section 3.4 of the paper).

The implementation of this module will be released upon acceptance of the paper.
This file defines the interface used by ``models/networks.py``. The channel-aware
gate, position-aware gate, and progressive fusion of GAGM are applied inside the
blocks defined in ``models/fgca.py``.
"""
import torch.nn as nn


class GranularityMap(nn.Module):
    """Parameter-free granularity map M.

    The bottleneck feature is detached (stop-gradient), its channel-averaged local
    variance is computed with window sizes {2, 4}, the variance maps are resized by
    bilinear interpolation and combined by an element-wise maximum, and the result is
    normalized by its 95th percentile and clipped at 1.

    Args:
        window_sizes (tuple[int]): granule window sizes w_s.
        percentile (float): normalization percentile.
        eps (float): constant that prevents division by zero.

    Inputs:
        x (Tensor): spatial bottleneck feature F_4, shape (B, 384, H/16, W/16).

    Returns:
        Tensor: granularity map M in [0, 1], shape (B, 1, H/16, W/16).
    """

    def __init__(self, window_sizes=(2, 4), percentile=0.95, eps=1e-6):
        super().__init__()
        self.window_sizes = tuple(window_sizes)
        self.percentile = percentile
        self.eps = eps

    def forward(self, x):
        raise NotImplementedError(
            "The GAGM granularity map will be released upon acceptance of the paper.")
