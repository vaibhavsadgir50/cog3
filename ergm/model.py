"""
GeometricReasoner: predicts next latent state via continuous dynamics.

This replaces autoregressive token prediction with a map
(s_t, a_t) -> s_hat_{t+1} in R^D, suitable for geometric / physics-aligned loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ergm.constants import LATENT_DIM


class GeometricReasoner(nn.Module):
    """
    Maps current latent state and action embedding to predicted next latent state.

    Uses an MLP (stack of linear + normalization + GELU). This is the minimal
    "transformer-like" stack without self-attention; you can swap the core for
    a small TransformerEncoder over sequence [s_t, a_t] if you later need
    relational mixing over multiple tokens.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = 64,
        hidden_dim: int | None = None,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        h = hidden_dim or max(latent_dim, action_dim) * 2

        in_dim = latent_dim + action_dim
        layers: list[nn.Module] = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(d, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                ]
            )
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_t: (B, latent_dim) current world state embedding.
            a_t: (B, action_dim) action embedding (precomputed or from an action encoder).

        Returns:
            s_hat_{t+1}: (B, latent_dim) predicted next state.
        """
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)
        x = torch.cat([s_t, a_t], dim=-1)
        return self.net(x)
