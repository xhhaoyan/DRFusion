"""
Noise schedules for diffusion models
Adapted from diffusion-forcing-transformer
"""

from typing import Literal
import math
import torch


def make_beta_schedule(
    schedule: Literal["cosine", "sigmoid", "sd", "linear", "alphas_cumprod_linear"],
    timesteps: int = 1000,
    shift: float = 1.0,
    clip_min: float = 1e-9,
    zero_terminal_snr: bool = True,
    **kwargs,
):
    """
    Create beta schedule for diffusion.

    Args:
        schedule: Schedule type
        timesteps: Number of timesteps
        shift: Scale factor for SNR
        clip_min: Minimum beta value
        zero_terminal_snr: Enforce zero SNR at T
        **kwargs: Additional schedule parameters
    """
    schedule_fn = {
        "alphas_cumprod_linear": alphas_cumprod_linear_schedule,
        "cosine": cosine_schedule,
        "cosine_simple_diffusion": cosine_simple_diffusion_schedule,
        "sigmoid": sigmoid_schedule,
        "sd": sd_schedule,
        "linear": beta_linear_schedule,
    }[schedule]

    alphas_cumprod = schedule_fn(timesteps, **kwargs)

    if schedule not in ["cosine", "cosine_simple_diffusion"] and zero_terminal_snr:
        alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)

    if shift != 1.0 and schedule != "cosine_simple_diffusion":
        alphas_cumprod = shift_beta_schedule(alphas_cumprod, shift)

    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas

    return torch.clip(betas, clip_min, 1.0)


def cosine_schedule(timesteps, s=0.008):
    """Cosine schedule from DDPM paper"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def cosine_simple_diffusion_schedule(
    timesteps,
    logsnr_min=-15.0,
    logsnr_max=15.0,
    shifted: float = 1.0,
    interpolated: bool = False,
):
    """
    Cosine schedule from Simple Diffusion
    Supports shifted and interpolated variants
    """
    t_min = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float64)))
    t_max = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float64)))
    t = torch.linspace(0, 1, timesteps, dtype=torch.float64)
    logsnr = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    if shifted != 1.0:
        shifted_logsnr = logsnr + 2 * torch.log(
            torch.tensor(shifted, dtype=torch.float64)
        )
        if interpolated:
            logsnr = t * logsnr + (1 - t) * shifted_logsnr
        else:
            logsnr = shifted_logsnr

    alphas_cumprod = 1 / (1 + torch.exp(-logsnr))
    return alphas_cumprod


def alphas_cumprod_linear_schedule(timesteps: int) -> torch.Tensor:
    """Linear schedule for alphas_cumprod"""
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    return (1 - t)[1:]


def beta_linear_schedule(
    timesteps: int, start: float = 0.0001, end: float = 0.02
) -> torch.Tensor:
    """Linear beta schedule from original DDPM"""
    betas = torch.linspace(start, end, timesteps, dtype=torch.float64)
    return (1 - betas).cumprod(dim=0)


def sigmoid_schedule(timesteps, start=-3, end=3, tau=1):
    """Sigmoid schedule, better for larger images"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def sd_schedule(timesteps, start=0.00085, end=0.0120):
    """Stable Diffusion's noise schedule"""
    betas = torch.linspace(start**0.5, end**0.5, timesteps, dtype=torch.float64) ** 2
    alphas_cumprod = (1 - betas).cumprod(dim=0)
    return alphas_cumprod


def shift_beta_schedule(alphas_cumprod: torch.Tensor, shift: float):
    """Scale alphas_cumprod so SNR is multiplied by shift^2"""
    snr_scale = shift**2
    return (snr_scale * alphas_cumprod) / (
        snr_scale * alphas_cumprod + 1 - alphas_cumprod
    )


def enforce_zero_terminal_snr(alphas_cumprod):
    """Enforce zero terminal SNR"""
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)

    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / alphas_cumprod_sqrt[0]

    alphas_cumprod = alphas_cumprod_sqrt**2
    assert alphas_cumprod[-1] == 0, "terminal SNR not zero"
    return alphas_cumprod
