"""
Direct geometric regression (no diffusion): (s_t, a_t) -> s_hat_{t+1} with MSE.

Uses LATENT_DIM=512 ToolAdapter targets and GeometricReasoner. CPU-friendly for
small batches; scale hidden layers via GeometricReasoner(..., hidden_dim=...).

Run: python -m ergm.demo_mse
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ergm.constants import LATENT_DIM
from ergm.environment import get_initial_state, simulate_action_and_observe
from ergm.model import GeometricReasoner
from ergm.tool_adapter import ToolAdapter
from ergm.training import geometric_prediction_loss, train_step


def main() -> None:
    raw_dim = 8
    action_dim = 64
    batch_size = 16
    device = torch.device("cpu")

    tool_adapter = ToolAdapter(raw_dim=raw_dim, latent_dim=LATENT_DIM).to(device)
    reasoner = GeometricReasoner(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
    ).to(device)
    action_encoder = nn.Linear(raw_dim, action_dim, bias=False).to(device)

    n_params = sum(p.numel() for p in reasoner.parameters() if p.requires_grad)
    print(f"GeometricReasoner (D={LATENT_DIM}, action_dim={action_dim}): {n_params:,} params")

    initial_rows: list[np.ndarray] = []
    next_raw_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []

    for i in range(batch_size):
        s0 = get_initial_state(seed=1000 + i)
        a_np = np.random.default_rng(2000 + i).standard_normal(raw_dim).astype(np.float64)
        obs_next = simulate_action_and_observe(s0, a_np)
        initial_rows.append(s0.astype(np.float32))
        next_raw_rows.append(obs_next.astype(np.float32))
        action_rows.append(a_np.astype(np.float32))

    initial_raw = torch.tensor(np.stack(initial_rows), device=device)
    raw_next_t = torch.tensor(np.stack(next_raw_rows), device=device)
    raw_actions = torch.tensor(np.stack(action_rows), device=device)

    s_t = tool_adapter(initial_raw)
    target_x_hat = tool_adapter(raw_next_t)
    a_t = action_encoder(raw_actions)

    pred = reasoner(s_t, a_t)
    loss_before = geometric_prediction_loss(pred, target_x_hat).item()
    print(f"MSE before train_step: {loss_before:.6f}")

    params = list(reasoner.parameters()) + list(tool_adapter.parameters()) + list(action_encoder.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    loss_val = train_step(reasoner, optimizer, s_t, a_t, target_x_hat)
    with torch.no_grad():
        loss_after = geometric_prediction_loss(reasoner(s_t, a_t), target_x_hat).item()
    print(f"MSE (train_step forward scalar): {loss_val:.6f}")
    print(f"MSE after one optimizer.step (re-forward): {loss_after:.6f}")
    print("Loss: geometric_prediction_loss = MSE(s_hat_{t+1}, x_hat)")


if __name__ == "__main__":
    main()
