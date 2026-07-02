"""
Video Fusion Sampler integrating Diffusion Forcing with EM-based fusion
This is the core component that combines both methods
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.EM_onestep import EM_onestep, EM_Initial
from RF.sampleWithEM import compute_alphas_cumprod
from util.common import decode_first_stage, encode_first_stage


class VideoFusionSampler:
    """
    Combines Diffusion Forcing with EM-based image fusion for video.

    Key innovations:
    1. Uses 3D DiT for temporal consistency
    2. History Guidance for stable autoregressive generation
    3. EM algorithm for IR/VI fusion at each denoising step
    """

    def __init__(
        self,
        diffusion_model,  # ContinuousDiffusion with 3D DiT
        autoencoder,      # VAE for latent space
        history_guidance,  # HistoryGuidance configuration
        timesteps=1000,
        sampling_timesteps=50,
        lamb=0.5,         # EM fusion parameter
        rho=0.01,         # EM fusion parameter
        temporal_smooth_alpha=0.2,  # Temporal smoothing weight
        device='cuda',
    ):
        """
        Args:
            diffusion_model: ContinuousDiffusion instance
            autoencoder: VAE model for encoding/decoding
            history_guidance: HistoryGuidance instance
            timesteps: Total diffusion timesteps
            sampling_timesteps: Number of sampling steps
            lamb: EM algorithm lambda parameter
            rho: EM algorithm rho parameter
            temporal_smooth_alpha: Weight for temporal smoothing
            device: torch device
        """
        self.diffusion_model = diffusion_model
        self.autoencoder = autoencoder
        self.history_guidance = history_guidance
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.lamb = lamb
        self.rho = rho
        self.temporal_smooth_alpha = temporal_smooth_alpha
        self.device = device

        # Precompute alpha schedules for EM algorithm
        betas = [0.1] * 1000
        self.alphas_cumprod, self.alphas_cumprod_prev = compute_alphas_cumprod(betas)

        # Autoencoder to eval mode
        self.autoencoder.eval()

    @torch.no_grad()
    def sample(
        self,
        infrared_seq,
        visible_seq,
        context_frames=2,
        guidance_scale=2.0,
        stabilization_level=0.02,
        eta=0.0,
    ):
        """
        Sample fused video sequence.

        Args:
            infrared_seq: (B, T, 1, H, W) IR video tensor
            visible_seq: (B, T, 3, H, W) VI video tensor
            context_frames: Number of initial context frames
            guidance_scale: History guidance scale
            stabilization_level: Stabilization for generated frames
            eta: DDIM eta (0=deterministic, 1=stochastic)

        Returns:
            fused_video: (B, T, 1, H, W) fused video tensor
        """
        B, T, C, H, W = visible_seq.shape
        device = self.device

        print(f"Sampling fused video: {B} batch, {T} frames, {H}x{W}")

        # ===== Step 1: Encode visible frames to latent space =====
        print("Step 1: Encoding frames to latent space...")
        vi_latents = []
        for t in range(T):
            latent = encode_first_stage(visible_seq[:, t], self.autoencoder)
            vi_latents.append(latent)
        vi_latents = torch.stack(vi_latents, dim=1)  # (B, T, C_lat, H_lat, W_lat)

        _, _, C_lat, H_lat, W_lat = vi_latents.shape

        # ===== Step 1.5: Encode IR frames and create conditioning =====
        print("Step 1.5: Encoding IR frames for conditioning...")
        ir_latents = []
        for t in range(T):
            # Convert IR to 3-channel for VAE
            ir_3ch = infrared_seq[:, t].repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
            latent = encode_first_stage(ir_3ch, self.autoencoder)
            ir_latents.append(latent)
        ir_latents = torch.stack(ir_latents, dim=1)  # (B, T, C_lat, H_lat, W_lat)

        # DDFM Approach: Use only IR as conditioning (VI is in initialization)
        # Shape: (B, T, C_lat, H_lat, W_lat)
        cond_latents = ir_latents  # Only IR for guidance
        print(f"   Conditioning shape (IR only): {cond_latents.shape}")

        # ===== Step 2: Initialize with reduced-noise VI =====
        print("Step 2: Initializing from VI with REDUCED noise...")
        print("   OPTIMIZATION: Lower noise level to preserve VI structure")

        # Use VI latent as starting point
        x_0 = vi_latents.clone()  # (B, T, C_lat, H_lat, W_lat)

        # REDUCED noise: t=200 instead of t=999
        # At t=200: ~90% VI signal + ~10% noise (preserves structure)
        # At t=999: ~1% VI signal + ~99% noise (destroys structure)
        t_init = torch.full((B,), 200, device=device, dtype=torch.long)  # OPTIMIZATION: 200 instead of 999
        noise = torch.randn_like(x_0)

        # q_sample: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x = self.diffusion_model.q_sample(x_0, t_init, noise)

        # Calculate preservation ratio
        alpha_bar_t = self.diffusion_model.alphas_cumprod[t_init[0]]
        vi_preservation = (alpha_bar_t ** 0.5).item()
        noise_ratio = ((1 - alpha_bar_t) ** 0.5).item()

        print(f"   VI latent:      mean={x_0.mean():.3f}, std={x_0.std():.3f}")
        print(f"   Noised VI (t={t_init[0].item()}): mean={x.mean():.3f}, std={x.std():.3f}")
        print(f"   Signal preservation: {vi_preservation*100:.1f}% VI + {noise_ratio*100:.1f}% noise")

        # ===== Step 3: Create mask =====
        # 0 = to be generated, 1 = ground truth context, 2 = generated history
        mask = torch.zeros(B, T, device=device)
        mask[:, :context_frames] = 1  # First N frames are context

        # ===== Step 4: Setup History Guidance =====
        print(f"Step 4: Setting up History Guidance (scale={guidance_scale})...")
        if self.history_guidance is None:
            from video_fusion.dfot.history_guidance import HistoryGuidance
            self.history_guidance = HistoryGuidance.stabilized_vanilla(
                guidance_scale=guidance_scale,
                stabilization_level=stabilization_level,
                timesteps=self.timesteps,
            )

        # ===== Step 5: Diffusion Forcing sampling loop =====
        print(f"Step 5: Diffusion Forcing sampling ({self.sampling_timesteps} steps)...")
        print("   OPTIMIZED: EM fusion DISABLED in loop to reduce VAE operations")
        print(f"   VAE operations: {self.sampling_timesteps * T * 2} (old) → {T} (new) = {100 * T / (self.sampling_timesteps * T * 2):.1f}% reduction")

        # Create sampling schedule
        timesteps_schedule = torch.linspace(
            self.timesteps - 1, 0, self.sampling_timesteps, dtype=torch.long, device=device
        )

        for step_idx, t_curr in enumerate(tqdm(timesteps_schedule, desc="Sampling")):
            t_next = timesteps_schedule[step_idx + 1] if step_idx < len(timesteps_schedule) - 1 else torch.tensor(-1, device=device)

            # Current batch timesteps
            t = torch.full((B,), t_curr.item(), device=device, dtype=torch.long)

            # ===== History Guidance Context Manager =====
            with self.history_guidance(mask) as hg:
                # Prepare inputs with history conditioning
                noise_levels_from = t.unsqueeze(1).expand(B, T)
                noise_levels_to = noise_levels_from.clone()

                x_input, from_noise, to_noise, cond_mask = hg.prepare(
                    x,
                    from_noise_levels=noise_levels_from,
                    to_noise_levels=noise_levels_to,
                    replacement_fn=self._add_noise_to_latent,
                )

                # Model prediction with all conditions
                # x_input shape: (B * num_hist * num_gen, T, C_lat, H_lat, W_lat)
                B_expanded = x_input.shape[0]
                t_expanded = t[0].repeat(B_expanded)

                # Expand conditioning to match batch size
                cond_expanded = cond_latents.repeat(B_expanded // B, 1, 1, 1, 1)

                pred_dict = self.diffusion_model.model_predictions(
                    x_input,
                    t_expanded,
                    external_cond=cond_expanded  # ← IR conditioning
                )

                # Compose predictions from different history conditions
                pred_x_start = hg.compose(pred_dict['pred_x_start'])

            # ===== DDIM update step (directly in latent space, NO EM fusion) =====
            # Optimization: Skip intermediate EM fusion to reduce VAE operations
            # EM fusion will only be applied at the final step (Step 6)
            x = self.diffusion_model.ddim_sample_step(
                x, t, t_next, pred_x_start, eta=eta
            )

            # Update mask: mark newly generated frames
            # (Simplified: all non-context frames become "generated")
            if step_idx == len(timesteps_schedule) - 1:
                mask[mask == 0] = 2

        # ===== Step 6: Final EM fusion on the last denoised result =====
        print("Step 6: Final EM fusion on denoised video...")
        print("   Applying EM fusion ONCE at the end to preserve clarity")
        fused_video = []
        bfHP = None

        for t in range(T):
            # Decode latent to RGB
            frame_latent = x[:, t]
            frame_rgb = decode_first_stage(frame_latent, self.autoencoder)

            # Convert to Y channel
            pred_y = self._rgb_to_y_channel(frame_rgb)

            # Initialize EM on first frame
            if t == 0:
                bfHP = EM_Initial(infrared_seq[:, t])

            # EM fusion
            ir_frame = infrared_seq[:, t]
            vi_y_frame = self._rgb_to_y_channel(visible_seq[:, t])

            # DDFM Approach: Use final prediction for EM fusion
            fused_dict, bfHP = self._em_fusion_step(
                pred_y,      # Use final denoised prediction
                ir_frame,
                vi_y_frame,
                bfHP
            )

            fused_y = fused_dict["sample"]
            fused_video.append(fused_y)

        fused_video = torch.stack(fused_video, dim=1)

        print("Sampling complete!")
        return fused_video

    def _rgb_to_y_channel(self, rgb_img):
        """
        Convert RGB image to Y channel (grayscale).

        Args:
            rgb_img: (B, 3, H, W) or (B, 1, H, W)

        Returns:
            (B, 1, H, W) Y channel
        """
        if rgb_img.shape[1] == 1:
            return rgb_img  # Already grayscale

        # Standard RGB to Y conversion weights
        r, g, b = rgb_img[:, 0:1], rgb_img[:, 1:2], rgb_img[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y

    def _y_channel_to_rgb(self, y_img):
        """
        Convert Y channel (grayscale) back to RGB by replicating across channels.

        Args:
            y_img: (B, 1, H, W) Y channel

        Returns:
            (B, 3, H, W) RGB image
        """
        if y_img.shape[1] == 3:
            return y_img  # Already RGB

        # Replicate Y channel to RGB
        return y_img.repeat(1, 3, 1, 1)

    def _em_fusion_step(self, pred_y, infrared, visible, bfHP):
        """
        Single EM fusion step.

        Args:
            pred_y: (B, 1, H, W) Predicted Y channel in [-1, 1]
            infrared: (B, 1, H, W) IR frame in [-1, 1]
            visible: (B, 1, H, W) VI Y channel in [-1, 1]
            bfHP: EM hyperparameters

        Returns:
            fused_dict: Dictionary with 'sample' key in [-1, 1]
            bfHP: Updated hyperparameters
        """
        # EM expects [0, 1] range, convert from [-1, 1]
        pred_y_01 = (pred_y + 1.0) / 2.0  # [-1,1] -> [0,1]
        infrared_01 = (infrared + 1.0) / 2.0
        visible_01 = (visible + 1.0) / 2.0

        # Call EM fusion (expects [0,1] range)
        fused_y_01, bfHP = EM_onestep(
            f_pre=pred_y_01,
            I=infrared_01,
            V=visible_01,
            HyperP=bfHP,
            lamb=self.lamb,
            rho=self.rho,
        )

        # Convert back to [-1, 1] range
        fused_y = fused_y_01 * 2.0 - 1.0  # [0,1] -> [-1,1]

        return {"sample": fused_y}, bfHP

    def _add_noise_to_latent(self, x_lat, noise_level):
        """
        Add noise to latent based on noise level.

        Args:
            x_lat: (B, T, C, H, W) Latent
            noise_level: (B, T) Noise level indices

        Returns:
            Noisy latent
        """
        # Handle different input shapes
        if noise_level.dim() == 1:
            noise_level = noise_level.unsqueeze(1)

        # Use q_sample from diffusion model
        B, T = noise_level.shape[:2]
        noisy_latents = []

        for t_idx in range(T):
            t = noise_level[:, t_idx].long()
            x_t = self.diffusion_model.q_sample(
                x_lat[:, t_idx:t_idx+1],
                t,
            )
            noisy_latents.append(x_t)

        return torch.cat(noisy_latents, dim=1)
