"""
Autoregressive Video Fusion Sampler - True Diffusion Forcing Implementation
This implements frame-by-frame generation with proper causal structure.
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


class AutoregressiveVideoFusionSampler:
    """
    True Diffusion Forcing: Autoregressive frame-by-frame video fusion.

    Key differences from batch version:
    1. Frame-by-frame generation loop (outer loop over frames)
    2. Each frame has independent denoising trajectory
    3. History accumulates with fused frames
    4. Causal: frame t only sees frames 1 to t-1
    5. Suitable for real-time applications

    Architecture:
        for t in range(context_frames, T):
            x_t = noise  # Initialize current frame
            for step in range(50):  # Denoise current frame
                x_t = denoise(x_t, history=[fused_1, ..., fused_{t-1}])
            fused_t = EM_fusion(x_t, IR_t, VI_t)
            history.append(fused_t)
    """

    def __init__(
        self,
        diffusion_model,  # ContinuousDiffusion with 3D DiT
        autoencoder,      # VAE for latent space
        history_guidance_config=None,  # Will create instance per frame
        timesteps=1000,
        sampling_timesteps=50,
        lamb=0.5,         # EM fusion parameter
        rho=0.01,         # EM fusion parameter
        max_context_frames=8,  # Maximum window size for efficiency
        device='cuda',
    ):
        """
        Args:
            diffusion_model: ContinuousDiffusion instance
            autoencoder: VAE model for encoding/decoding
            history_guidance_config: Dict with guidance_scale and stabilization_level
            timesteps: Total diffusion timesteps
            sampling_timesteps: Number of sampling steps per frame
            lamb: EM algorithm lambda parameter
            rho: EM algorithm rho parameter
            max_context_frames: Max history frames to use (for memory efficiency)
            device: torch device
        """
        self.diffusion_model = diffusion_model
        self.autoencoder = autoencoder
        self.history_guidance_config = history_guidance_config or {
            'guidance_scale': 2.0,
            'stabilization_level': 0.02
        }
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.lamb = lamb
        self.rho = rho
        self.max_context_frames = max_context_frames
        self.device = device

        # Precompute alpha schedules for EM algorithm
        betas = [0.1] * 1000
        self.alphas_cumprod, self.alphas_cumprod_prev = compute_alphas_cumprod(betas)

        # Autoencoder to eval mode
        self.autoencoder.eval()

    @torch.no_grad()
    def sample_autoregressive(
        self,
        infrared_seq,
        visible_seq,
        context_frames=2,
        guidance_scale=2.0,
        stabilization_level=0.02,
        eta=0.0,
        verbose=True,
    ):
        """
        Autoregressive frame-by-frame video fusion.

        Args:
            infrared_seq: (B, T, 1, H, W) IR video tensor
            visible_seq: (B, T, 3, H, W) VI video tensor
            context_frames: Number of initial context frames (GT)
            guidance_scale: History guidance scale
            stabilization_level: Stabilization for generated history frames
            eta: DDIM eta (0=deterministic, 1=stochastic)
            verbose: Print progress

        Returns:
            fused_video: (B, T, 1, H, W) fused video tensor
        """
        B, T, C, H, W = visible_seq.shape
        device = self.device

        if verbose:
            print("=" * 60)
            print("AUTOREGRESSIVE VIDEO FUSION (True Diffusion Forcing)")
            print("=" * 60)
            print(f"Input: {B} batch, {T} frames, {H}x{W}")
            print(f"Context frames: {context_frames} (will also be denoised)")
            print(f"All frames generation: {T} frames total")
            print(f"Max context window: {self.max_context_frames}")
            print(f"Sampling steps per frame: {self.sampling_timesteps}")
            print(f"Total denoising iterations: {T * self.sampling_timesteps}")
            print("=" * 60)

        # ===== Step 1: Unified initialization - All frames go through diffusion =====
        if verbose:
            print("\n[Step 1] Unified Initialization (Diffusion Forcing)...")
            print("  All frames will go through diffusion process for consistent quality")

        fused_frames = []
        fused_latents = []
        bfHP = None

        # ===== Step 2: Autoregressive generation loop (including context frames) =====
        if verbose:
            print(f"\n[Step 2] Autoregressive generation (ALL {T} frames)...")
            print(f"  Context frames (0-{context_frames-1}): Minimal history")
            print(f"  Generated frames ({context_frames}-{T-1}): Full history")

        for t in tqdm(range(T), desc="Generating frames", disable=not verbose):
            if verbose and (t < context_frames or t == context_frames):
                frame_type = "Context" if t < context_frames else "Generated"
                print(f"\n--- Frame {t+1}/{T} ({frame_type}) ---")

            # Step 2.1: Encode current VI frame
            vi_frame = visible_seq[:, t]
            vi_latent = encode_first_stage(vi_frame, self.autoencoder)

            # Step 2.2: Initialize with low noise
            t_init = torch.full((B,), 200, device=device, dtype=torch.long)
            noise = torch.randn_like(vi_latent)
            x_t = self.diffusion_model.q_sample(vi_latent, t_init, noise)

            if verbose and (t == 0 or t == context_frames):
                alpha_bar = self.diffusion_model.alphas_cumprod[t_init[0]]
                vi_pres = (alpha_bar ** 0.5).item()
                print(f"  Initialized: {vi_pres*100:.1f}% VI + {(1-vi_pres)*100:.1f}% noise")

            # Step 2.3: Prepare history window
            if t == 0:
                # First frame: no history
                history_fused_latents = torch.zeros(B, 0, vi_latent.shape[1],
                                                   vi_latent.shape[2], vi_latent.shape[3],
                                                   device=device)
                history_IR_latents = torch.zeros(B, 0, vi_latent.shape[1],
                                                vi_latent.shape[2], vi_latent.shape[3],
                                                device=device)
            else:
                # Use history (limited by max_context_frames)
                history_window_size = min(t, self.max_context_frames)
                history_start = max(0, t - history_window_size)

                history_fused_latents = torch.stack(
                    fused_latents[history_start:t], dim=1
                )  # (B, hist_len, C_lat, H_lat, W_lat)

                history_IR_latents = []
                for h_idx in range(history_start, t):
                    ir_3ch = infrared_seq[:, h_idx].repeat(1, 3, 1, 1)
                    ir_lat = encode_first_stage(ir_3ch, self.autoencoder)
                    history_IR_latents.append(ir_lat)
                history_IR_latents = torch.stack(history_IR_latents, dim=1)

            # Current IR
            current_ir_3ch = infrared_seq[:, t].repeat(1, 3, 1, 1)
            current_ir_latent = encode_first_stage(current_ir_3ch, self.autoencoder)

            if verbose and (t == 0 or t == context_frames):
                hist_size = history_fused_latents.shape[1] if t > 0 else 0
                if t == 0:
                    print(f"  History window: No history (first frame)")
                else:
                    hist_start = max(0, t - self.max_context_frames)
                    print(f"  History window: frames {hist_start+1}-{t} ({hist_size} frames)")

            # Step 2.4: Denoise current frame
            x_t = self._denoise_single_frame(
                x_t=x_t,
                history_fused_latents=history_fused_latents,
                history_IR_latents=history_IR_latents,
                current_ir_latent=current_ir_latent,
                guidance_scale=guidance_scale,
                stabilization_level=stabilization_level,
                eta=eta,
                verbose=verbose and (t == 0 or t == context_frames),
            )

            # Step 2.5: Decode and EM fusion
            frame_rgb = decode_first_stage(x_t, self.autoencoder)
            pred_y = self._rgb_to_y_channel(frame_rgb)

            ir_frame = infrared_seq[:, t]
            vi_y_frame = self._rgb_to_y_channel(visible_seq[:, t])

            # Initialize EM on first frame
            if t == 0:
                bfHP = EM_Initial(ir_frame)

            fused_y, bfHP = self._em_fusion_step(pred_y, ir_frame, vi_y_frame, bfHP)

            # Step 2.6: Add to history
            fused_frames.append(fused_y)
            fused_latents.append(x_t)

            if verbose and (t == context_frames - 1 or t == context_frames):
                if t == context_frames - 1:
                    print(f"  Context frames complete: {len(fused_frames)} frames")
                elif t == context_frames:
                    print(f"  First generated frame complete. History size: {len(fused_frames)}")

        # Stack all frames
        fused_video = torch.stack(fused_frames, dim=1)  # (B, T, 1, H, W)

        if verbose:
            print("\n" + "=" * 60)
            print("AUTOREGRESSIVE GENERATION COMPLETE")
            print(f"Output: {fused_video.shape}")
            print("=" * 60)

        return fused_video

    def _denoise_single_frame(
        self,
        x_t,
        history_fused_latents,
        history_IR_latents,
        current_ir_latent,
        guidance_scale,
        stabilization_level,
        eta,
        verbose=False,
    ):
        """
        Denoise a single frame given history.

        Args:
            x_t: (B, C_lat, H_lat, W_lat) Current noisy frame
            history_fused_latents: (B, hist_len, C_lat, H_lat, W_lat) History
            history_IR_latents: (B, hist_len, C_lat, H_lat, W_lat) History IR
            current_ir_latent: (B, C_lat, H_lat, W_lat) Current IR conditioning
            guidance_scale: Guidance strength
            stabilization_level: Stabilization level
            eta: DDIM eta
            verbose: Print denoising progress

        Returns:
            x_0: (B, C_lat, H_lat, W_lat) Denoised frame
        """
        B = x_t.shape[0]
        device = x_t.device
        hist_len = history_fused_latents.shape[1]

        # Create History Guidance instance for this frame
        from video_fusion.dfot.history_guidance import HistoryGuidance
        history_guidance = HistoryGuidance.stabilized_vanilla(
            guidance_scale=guidance_scale,
            stabilization_level=stabilization_level,
            timesteps=self.timesteps,
        )

        # Create window: [history, current]
        # x_window: (B, hist_len+1, C_lat, H_lat, W_lat)
        x_window = torch.cat([
            history_fused_latents,
            x_t.unsqueeze(1)  # Add time dimension
        ], dim=1)

        # Mask: history=2 (generated), current=0 (to generate)
        mask = torch.ones(B, hist_len + 1, device=device) * 2
        mask[:, -1] = 0  # Last frame is current (to generate)

        # Conditioning: [history_IR, current_IR]
        cond_window = torch.cat([
            history_IR_latents,
            current_ir_latent.unsqueeze(1)
        ], dim=1)

        # Sampling schedule
        timesteps_schedule = torch.linspace(
            self.timesteps - 1, 0, self.sampling_timesteps,
            dtype=torch.long, device=device
        )

        if verbose:
            print(f"    Denoising: {self.sampling_timesteps} steps")

        # Denoising loop
        for step_idx, t_curr in enumerate(timesteps_schedule):
            t_next = timesteps_schedule[step_idx + 1] if step_idx < len(timesteps_schedule) - 1 else torch.tensor(-1, device=device)
            t = torch.full((B,), t_curr.item(), device=device, dtype=torch.long)

            # History Guidance
            with history_guidance(mask) as hg:
                noise_levels_from = t.unsqueeze(1).expand(B, hist_len + 1)
                noise_levels_to = noise_levels_from.clone()

                x_input, from_noise, to_noise, cond_mask = hg.prepare(
                    x_window,
                    from_noise_levels=noise_levels_from,
                    to_noise_levels=noise_levels_to,
                    replacement_fn=self._add_noise_to_latent,
                )

                # Expand for batch
                B_expanded = x_input.shape[0]
                t_expanded = t[0].repeat(B_expanded)
                cond_expanded = cond_window.repeat(B_expanded // B, 1, 1, 1, 1)

                # Model prediction
                pred_dict = self.diffusion_model.model_predictions(
                    x_input,
                    t_expanded,
                    external_cond=cond_expanded
                )

                # Compose predictions
                pred_x_start_window = hg.compose(pred_dict['pred_x_start'])

            # Extract current frame prediction
            pred_x_start_current = pred_x_start_window[:, -1:]  # (B, 1, C, H, W)

            # DDIM update (only update current frame)
            x_t_window = x_window[:, -1:].clone()  # (B, 1, C, H, W)
            x_t_updated = self.diffusion_model.ddim_sample_step(
                x_t_window, t, t_next, pred_x_start_current, eta=eta
            )

            # Update window
            x_window = torch.cat([
                history_fused_latents,
                x_t_updated
            ], dim=1)

            # Update x_t for next iteration
            x_t = x_t_updated.squeeze(1)  # Remove time dimension

        return x_t

    def _add_noise_to_latent(self, x, t):
        """
        Add noise to latent according to diffusion schedule

        Args:
            x: Latent tensor, shape (B, T, C, H, W)
            t: Noise level tensor, shape (B, T)

        Returns:
            Noised tensor with same shape as x
        """
        B, T, C, H, W = x.shape

        # Process frame by frame
        noised_frames = []
        for frame_idx in range(T):
            x_frame = x[:, frame_idx]  # (B, C, H, W)
            t_frame = t[:, frame_idx]  # (B,)

            noise = torch.randn_like(x_frame)
            noised_frame = self.diffusion_model.q_sample(x_frame, t_frame, noise)
            noised_frames.append(noised_frame)

        # Stack back to (B, T, C, H, W)
        return torch.stack(noised_frames, dim=1)

    def _rgb_to_y_channel(self, rgb):
        """Convert RGB to Y channel (luminance)"""
        if rgb.shape[1] == 1:
            return rgb
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y

    def _em_fusion_step(self, f_pre, I, V, HyperP):
        """
        EM fusion single step with proper range conversion.

        Args:
            f_pre: (B, 1, H, W) Predicted Y channel in [-1, 1]
            I: (B, 1, H, W) IR frame in [-1, 1]
            V: (B, 1, H, W) VI Y channel in [-1, 1]
            HyperP: EM hyperparameters dict

        Returns:
            fused_y: (B, 1, H, W) Fused result in [-1, 1]
            HyperP: Updated hyperparameters dict
        """
        # EM expects [0, 1] range, convert from [-1, 1]
        f_pre_01 = (f_pre + 1.0) / 2.0  # [-1,1] -> [0,1]
        I_01 = (I + 1.0) / 2.0
        V_01 = (V + 1.0) / 2.0

        # Call EM fusion (returns tensor, dict)
        fused_y_01, HyperP = EM_onestep(
            f_pre=f_pre_01,
            I=I_01,
            V=V_01,
            HyperP=HyperP,
            lamb=self.lamb,
            rho=self.rho,
        )

        # Convert back to [-1, 1] range
        fused_y = fused_y_01 * 2.0 - 1.0  # [0,1] -> [-1,1]

        return fused_y, HyperP

    # Wrapper for backward compatibility
    @torch.no_grad()
    def sample(self, *args, **kwargs):
        """Alias for sample_autoregressive"""
        return self.sample_autoregressive(*args, **kwargs)
