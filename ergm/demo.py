"""
End-to-end mock on CPU: light latent DLLM (diffusion) + ToolAdapter.

Run: python -m ergm.demo

Direct MSE (s_t, a_t) -> s_hat_{t+1} at LATENT_DIM=512: python -m ergm.demo_mse
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ergm.constants import DEFAULT_ACTION_DIM_LIGHT, LATENT_DIM_LIGHT
from ergm.diffusion import LightLatentDLLM, count_parameters, diffusion_train_step
from ergm.environment import get_initial_state, simulate_action_and_observe
from ergm.tool_adapter import ToolAdapter


def main() -> None:
    raw_dim = 8
    action_dim = DEFAULT_ACTION_DIM_LIGHT
    batch_size = 16
    device = torch.device("cpu")

    tool_adapter = ToolAdapter(raw_dim=raw_dim, latent_dim=LATENT_DIM_LIGHT).to(device)
    dllm = LightLatentDLLM(
        latent_dim=LATENT_DIM_LIGHT,
        action_dim=action_dim,
    ).to(device)
    action_encoder = nn.Linear(raw_dim, action_dim, bias=False).to(device)

    print(
        f"CPU light latent DLLM: D={LATENT_DIM_LIGHT}, T={dllm.num_timesteps}, "
        f"denoiser params={count_parameters(dllm):,}"
    )

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

    params = list(dllm.parameters()) + list(tool_adapter.parameters()) + list(action_encoder.parameters())
    optimizer = optim.Adam(params, lr=2e-3)

    loss_before = dllm.diffusion_loss(target_x_hat, s_t, a_t).item()
    print(f"Diffusion loss before train_step: {loss_before:.6f}")

    loss_val = diffusion_train_step(dllm, optimizer, s_t, a_t, target_x_hat)
    with torch.no_grad():
        loss_after = dllm.diffusion_loss(target_x_hat, s_t, a_t).item()
    print(f"Diffusion loss (train_step scalar): {loss_val:.6f}")
    print(f"Diffusion loss after one optimizer.step: {loss_after:.6f}")

    with torch.no_grad():
        x_sample = dllm.sample(s_t, a_t)
    print(f"Sampled next latent shape: {tuple(x_sample.shape)} (same as target_x_hat)")

    print("Optimizer: zero_grad -> diffusion noise MSE -> backward -> step")


if __name__ == "__main__":
    main()
