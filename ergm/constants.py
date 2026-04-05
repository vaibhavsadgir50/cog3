"""
Global geometric world model constants.

Reasoning is represented as continuous dynamics in a single latent manifold
R^LATENT_DIM instead of discrete next-token prediction.

LATENT_DIM_LIGHT + small diffusion: intended for CPU-friendly latent DLLM
(denoising in R^D, not a large token vocabulary).
"""

# Full-width latent (original ERGM / heavier runs)
LATENT_DIM: int = 512

# CPU-light profile: small manifold + few diffusion steps
LATENT_DIM_LIGHT: int = 128
DIFFUSION_STEPS_LIGHT: int = 12
DENOISER_HIDDEN_LIGHT: int = 192
TIME_EMB_DIM: int = 32
DEFAULT_ACTION_DIM_LIGHT: int = 64
