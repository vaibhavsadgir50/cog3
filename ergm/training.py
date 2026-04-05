"""
Training utilities: geometric MSE loss and a single-step train demo.

Loss aligns predicted latent next state with tool-projected ground truth x_hat,
not cross-entropy over vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ergm.model import GeometricReasoner


def geometric_prediction_loss(
    pred_next: torch.Tensor,
    target_next: torch.Tensor,
) -> torch.Tensor:
    """
    L = MSE(s_hat_{t+1}, x_hat).

    pred_next:   (B, D) output of GeometricReasoner
    target_next: (B, D) output of ToolAdapter on raw tool / physics result
    """
    return nn.functional.mse_loss(pred_next, target_next)


def train_step(
    reasoner: GeometricReasoner,
    optimizer: torch.optim.Optimizer,
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    target_x_hat: torch.Tensor,
) -> float:
    """
    One optimization step on a batch (s_t, a_t, target_x_hat).

    s_t / target_x_hat must come from a fresh forward each step if they share
    modules trained jointly (e.g. ToolAdapter); reusing precomputed tensors across
    backward() calls causes "backward through the graph a second time".

    Demonstrates the full cycle: zero_grad -> forward -> loss -> backward -> step.
    Returns detached scalar loss for logging.
    """
    optimizer.zero_grad(set_to_none=True)
    s_hat_next = reasoner(s_t, a_t)
    loss = geometric_prediction_loss(s_hat_next, target_x_hat)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
