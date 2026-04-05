"""
Tool adapter H: maps heterogeneous tool / physics outputs into the shared
latent space R^D.

External tools (math, physics engines, APIs) return raw vectors; H standardizes
them so the GeometricReasoner can be trained against geometry-grounded targets
rather than symbolic strings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ergm.constants import LATENT_DIM


class ToolAdapter(nn.Module):
    """
    Linear projection from raw tool observation space to latent ground truth x_hat.

    Input:  raw tensor of shape (batch, raw_dim) — floats from a tool or simulator.
    Output: x_hat of shape (batch, latent_dim), the target next-state embedding.
    """

    def __init__(self, raw_dim: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        self.latent_dim = latent_dim
        self.projection = nn.Linear(raw_dim, latent_dim, bias=True)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw: (B, raw_dim) physical / numeric tool output.
        Returns:
            x_hat: (B, latent_dim) ground-truth state in latent space.
        """
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        return self.projection(raw.float())
