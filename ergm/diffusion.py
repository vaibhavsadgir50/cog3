"""
Light latent DLLM: diffusion (denoising) in R^D conditioned on (s_t, a_t).

Same spirit as discrete diffusion LMs (iterative denoise) but continuous latent
vectors and a tiny denoiser so training/sampling stays feasible on CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ergm.constants import (
    DEFAULT_ACTION_DIM_LIGHT,
    DENOISER_HIDDEN_LIGHT,
    DIFFUSION_STEPS_LIGHT,
    LATENT_DIM_LIGHT,
    TIME_EMB_DIM,
)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Integer timestep t -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)


class LightLatentDLLM(nn.Module):
    """
    Predicts noise epsilon in a DDPM-style latent diffusion, conditioned on state + action.

    Training target: MSE(eps_pred, eps_true) on noisy x_t where x_0 is tool-grounded x_hat.
    Inference: sample p(x_0 | s_t, a_t) by iterative denoising (CPU-light T).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM_LIGHT,
        action_dim: int = DEFAULT_ACTION_DIM_LIGHT,
        hidden_dim: int = DENOISER_HIDDEN_LIGHT,
        time_emb_dim: int = TIME_EMB_DIM,
        num_timesteps: int = DIFFUSION_STEPS_LIGHT,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.time_emb_dim = time_emb_dim

        betas = _linear_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        self.time_in = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.time_act = nn.GELU()
        in_dim = latent_dim + latent_dim + action_dim + self.time_emb_dim
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, latent_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, D) noisy latent
            t:   (B,) int64 timesteps in [0, num_timesteps)
            s_t: (B, D) current state embedding
            a_t: (B, action_dim) action embedding
        Returns:
            epsilon prediction (B, D)
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)
        te = _sinusoidal_time_embedding(t, self.time_emb_dim)
        te = self.time_act(self.time_in(te))
        h = torch.cat([x_t, s_t, a_t, te], dim=-1)
        return self.net(h)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        ac = self.alphas_cumprod.gather(0, t).view(-1, 1)
        return torch.sqrt(ac) * x_0 + torch.sqrt(1.0 - ac) * noise

    def diffusion_loss(self, x_0: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Single-batch DDPM noise-prediction loss."""
        b = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.forward(x_t, t, s_t, a_t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample_step(self, x: torch.Tensor, t: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """One reverse step x_t -> x_{t-1} (batched, same scalar t per batch as in loop)."""
        eps = self.forward(x, t, s_t, a_t)
        b = self.betas.gather(0, t).view(-1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1)
        sqrt_ra = self.sqrt_recip_alphas.gather(0, t).view(-1, 1)
        model_mean = sqrt_ra * (x - b / sqrt_om * eps)
        post_var = self.posterior_variance.gather(0, t).view(-1, 1)
        noise = torch.randn_like(x)
        nonzero = (t > 0).float().view(-1, 1)
        return model_mean + nonzero * torch.sqrt(post_var) * noise

    @torch.no_grad()
    def sample(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Sample x_0 ~ p(x|s,a) by full reverse chain (T small for CPU)."""
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)
        bsz = s_t.shape[0]
        device = s_t.device
        x = torch.randn(bsz, self.latent_dim, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((bsz,), i, device=device, dtype=torch.long)
            x = self.p_sample_step(x, t, s_t, a_t)
        return x


def diffusion_train_step(
    dllm: LightLatentDLLM,
    optimizer: torch.optim.Optimizer,
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    target_x_hat: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = dllm.diffusion_loss(target_x_hat, s_t, a_t)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
