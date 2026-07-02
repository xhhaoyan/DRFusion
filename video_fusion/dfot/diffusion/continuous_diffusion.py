"""
Continuous Diffusion for Video Fusion
Adapted from diffusion-forcing-transformer with v-prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .noise_schedule import make_beta_schedule


class CosineNoiseSchedule(nn.Module):
    """Continuous-time noise schedule using cosine function"""

    def __init__(self, logsnr_min=-15.0, logsnr_max=15.0, shift=1.0):
        super().__init__()
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "shift",
            2 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Given timestep t ∈ [0, 1], return logSNR value

        Args:
            t: (batch_size,) timesteps in [0, 1]

        Returns:
            logSNR: (batch_size,) log signal-to-noise ratio
        """
        return (
            -2 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min)))
            + self.shift
        )

    @property
    def max_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(0.0, dtype=torch.float32, device=self.shift.device)
        )

    @property
    def min_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(1.0, dtype=torch.float32, device=self.shift.device)
        )


class ContinuousDiffusion(nn.Module):
    """
    Continuous-time diffusion model with v-prediction.
    This is the core diffusion mechanism from Diffusion Forcing.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        shift: float = 1.0,
        precond_scale: float = 1.0,
        sigmoid_bias: float = 0.0,
        clip_noise: float = 1000.0,
    ):
        """
        Args:
            model: 3D backbone model (DiT/UNet)
            timesteps: Total training timesteps
            sampling_timesteps: Sampling timesteps (can be less)
            logsnr_min: Minimum log SNR
            logsnr_max: Maximum log SNR
            shift: Schedule shift factor
            precond_scale: Preconditioning scale
            sigmoid_bias: Bias for sigmoid loss weighting
            clip_noise: Clip noise to this value
        """
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.precond_scale = precond_scale
        self.sigmoid_bias = sigmoid_bias
        self.clip_noise = clip_noise

        # Continuous noise schedule
        self.training_schedule = CosineNoiseSchedule(
            logsnr_min=logsnr_min,
            logsnr_max=logsnr_max,
            shift=shift
        )

        # For discrete sampling, we still use timestep indices
        # But convert them to continuous time via logSNR
        betas = make_beta_schedule(
            schedule="cosine_simple_diffusion",
            timesteps=timesteps,
            logsnr_min=logsnr_min,
            logsnr_max=logsnr_max,
            shifted=shift,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32), persistent=False
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # For sampling
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("logsnr", torch.log(snr))

    def add_shape_channels(self, x):
        """Add dimensions to match input shape"""
        while x.ndimension() < 5:  # (B, T, C, H, W)
            x = x.unsqueeze(-1)
        return x

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to x_start

        Args:
            x_start: (B, T, C, H, W) Clean video
            t: (B,) Timestep indices
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy video at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        """Extract values from a at indices t and reshape"""
        batch_size = t.shape[0]
        out = a[t.long()]
        # Reshape to (B, 1, 1, 1, ...) for broadcasting with x_shape
        # x_shape is a tensor, so we need x_shape.shape to get dimensions
        num_dims = len(x_shape.shape)
        return out.reshape(batch_size, *((1,) * (num_dims - 1)))

    def predict_start_from_v(self, x_t, t, v):
        """
        Predict x_0 from velocity prediction v

        v = alpha_t * eps - sigma_t * x_0
        x_0 = alpha_t * x_t - sigma_t * v
        """
        logsnr = self.logsnr[t.long()]
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        x_start = alpha_t * x_t - sigma_t * v
        return x_start

    def predict_noise_from_v(self, x_t, t, v):
        """
        Predict noise from velocity prediction v

        v = alpha_t * eps - sigma_t * x_0
        eps = alpha_t * v + sigma_t * x_t
        """
        logsnr = self.logsnr[t.long()]
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        pred_noise = alpha_t * v + sigma_t * x_t
        return pred_noise

    def model_predictions(self, x_t, t, external_cond=None):
        """
        Get model predictions (v-prediction)

        Args:
            x_t: (B, T, C, H, W) Noisy input
            t: (B,) Timestep indices
            external_cond: Optional external conditioning

        Returns:
            dict with 'pred_noise', 'pred_x_start', 'v'
        """
        logsnr = self.logsnr[t.long()]

        # Model forward pass
        v = self.model(x_t, self.precond_scale * logsnr, external_cond)

        # Convert v-prediction to x_0 and noise
        pred_x_start = self.predict_start_from_v(x_t, t, v)
        pred_noise = self.predict_noise_from_v(x_t, t, v)

        return {
            'pred_noise': pred_noise,
            'pred_x_start': pred_x_start,
            'v': v
        }

    @torch.no_grad()
    def ddim_sample_step(self, x_t, t, t_next, pred_x_start, eta=0.0):
        """
        Single DDIM sampling step

        Args:
            x_t: Current noisy sample
            t: Current timestep
            t_next: Next timestep
            pred_x_start: Predicted clean sample
            eta: DDIM eta parameter (0=deterministic)

        Returns:
            x_{t-1}: Less noisy sample
        """
        alpha_bar = self.alphas_cumprod[t.long()]
        alpha_bar_prev = self.alphas_cumprod[t_next.long()] if t_next >= 0 else torch.ones_like(alpha_bar)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Mean calculation
        alpha_bar = self.add_shape_channels(alpha_bar)
        alpha_bar_prev = self.add_shape_channels(alpha_bar_prev)
        sigma = self.add_shape_channels(sigma)

        mean_pred = (
            torch.sqrt(alpha_bar_prev) * pred_x_start
            + torch.sqrt(1 - alpha_bar_prev - sigma**2)
            * (x_t - torch.sqrt(alpha_bar) * pred_x_start)
            / torch.sqrt(1 - alpha_bar)
        )

        # Add noise
        noise = torch.randn_like(x_t) if t_next > 0 else torch.zeros_like(x_t)
        x_prev = mean_pred + sigma * noise

        return x_prev

    def forward(self, x, t):
        """
        Training forward pass with v-prediction

        Args:
            x: (B, T, C, H, W) Clean video
            t: (B,) Continuous timesteps in [0, 1]

        Returns:
            x_pred: Predicted clean video
            loss: Weighted MSE loss
        """
        # Sample continuous time and get logSNR
        logsnr = self.training_schedule(t)

        # Add noise
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())
        x_t = alpha_t * x + sigma_t * noise

        # v-prediction
        v_pred = self.model(x_t, self.precond_scale * logsnr, None)

        # Convert to x_0 and noise predictions
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred

        # MSE loss on noise
        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")

        # Sigmoid loss weighting (Simple Diffusion 2)
        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss
