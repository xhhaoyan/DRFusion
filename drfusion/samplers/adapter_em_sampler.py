"""Stage-2 adapter sampler with EM replacement for DRFusion."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from guided_diffusion.EM_onestep import EM_Initial, EM_onestep
from util.common import decode_first_stage, encode_first_stage
from video_fusion.dfot.history_guidance import HistoryGuidance


class AdapterEMReplacementDiffusionForcingSampler:
    """Autoregressive sampler that uses the stage-2 ConditionAdapter.

    Stage-2 training calls the adapter as
    ``adapter(noisy_latent, raw_timestep, ir_cond=IR)``. Inference follows the
    same interface and applies EM replacement to the predicted clean frame every
    ``em_every_n_steps`` denoising steps.
    """

    def __init__(
        self,
        diffusion_model,
        autoencoder,
        timesteps=1000,
        sampling_timesteps=50,
        lamb=0.5,
        rho=0.01,
        max_context_frames=8,
        vae_size=256,
        em_every_n_steps=5,
        use_vi_cond=False,
        device="cuda",
    ):
        self.diffusion_model = diffusion_model
        self.autoencoder = autoencoder
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.lamb = lamb
        self.rho = rho
        self.max_context_frames = max_context_frames
        self.vae_size = vae_size
        self.em_every_n_steps = max(1, int(em_every_n_steps))
        self.use_vi_cond = use_vi_cond
        self.device = device
        self.original_size = None

        self.autoencoder.eval()

    def _resize_for_vae(self, images):
        return F.interpolate(
            images,
            size=(self.vae_size, self.vae_size),
            mode="bicubic",
            align_corners=False,
        )

    def _resize_to_original(self, images):
        return F.interpolate(
            images,
            size=self.original_size,
            mode="bicubic",
            align_corners=False,
        )

    def _encode_with_resize(self, images):
        return encode_first_stage(self._resize_for_vae(images), self.autoencoder)

    def _decode_with_resize(self, latents):
        return self._resize_to_original(decode_first_stage(latents, self.autoencoder))

    @staticmethod
    def _rgb_to_y_channel(rgb_tensor):
        if rgb_tensor.shape[1] == 1:
            return rgb_tensor
        r, g, b = rgb_tensor[:, 0:1], rgb_tensor[:, 1:2], rgb_tensor[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _adapter_predictions(self, x_t, t, ir_cond, vi_cond=None):
        raw_t = t.float()
        v = self.diffusion_model.model(
            x_t,
            raw_t,
            ir_cond=ir_cond,
            vi_cond=vi_cond,
        )
        return {
            "pred_noise": self.diffusion_model.predict_noise_from_v(x_t, t, v),
            "pred_x_start": self.diffusion_model.predict_start_from_v(x_t, t, v),
            "v": v,
        }

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
        del context_frames  # This path uses the generated history window directly.

        batch_size, num_frames, _, height, width = visible_seq.shape
        self.original_size = (height, width)

        if verbose:
            print("=" * 70)
            print("DRFusion inference: stage-2 adapter + EM replacement")
            print("=" * 70)
            print(f"Input: {batch_size} batch, {num_frames} frames, {height}x{width}")
            print(f"Adapter condition size: {self.vae_size}x{self.vae_size}")
            print(f"VI condition into adapter: {self.use_vi_cond}")
            print(f"EM replacement: every {self.em_every_n_steps} denoising step(s)")
            print("=" * 70)

        fused_frames = []
        fused_latents = []
        bfHP = None
        bfHP_final = None

        for frame_idx in tqdm(range(num_frames), desc="Generating frames", disable=not verbose):
            if verbose and frame_idx == 0:
                print(f"\n--- Frame {frame_idx + 1}/{num_frames} ---")

            ir_frame_original = infrared_seq[:, frame_idx]
            vi_frame_original = visible_seq[:, frame_idx]

            vi_latent = self._encode_with_resize(vi_frame_original)
            t_init = torch.full((batch_size,), 200, device=self.device, dtype=torch.long)
            x_t = self.diffusion_model.q_sample(
                vi_latent,
                t_init,
                torch.randn_like(vi_latent),
            )

            if frame_idx == 0:
                history_fused_latents = torch.zeros(
                    batch_size,
                    0,
                    *vi_latent.shape[1:],
                    device=self.device,
                )
                history_ir_cond = torch.zeros(
                    batch_size,
                    0,
                    1,
                    self.vae_size,
                    self.vae_size,
                    device=self.device,
                )
                history_vi_cond = None
                if self.use_vi_cond:
                    history_vi_cond = torch.zeros(
                        batch_size,
                        0,
                        3,
                        self.vae_size,
                        self.vae_size,
                        device=self.device,
                    )
            else:
                history_window_size = min(frame_idx, self.max_context_frames)
                history_start = max(0, frame_idx - history_window_size)
                history_fused_latents = torch.stack(
                    fused_latents[history_start:frame_idx],
                    dim=1,
                )
                history_ir_cond = torch.stack(
                    [
                        self._resize_for_vae(infrared_seq[:, hist_idx])
                        for hist_idx in range(history_start, frame_idx)
                    ],
                    dim=1,
                )
                history_vi_cond = None
                if self.use_vi_cond:
                    history_vi_cond = torch.stack(
                        [
                            self._resize_for_vae(visible_seq[:, hist_idx])
                            for hist_idx in range(history_start, frame_idx)
                        ],
                        dim=1,
                    )

            current_ir_cond = self._resize_for_vae(ir_frame_original)
            current_vi_cond = self._resize_for_vae(vi_frame_original) if self.use_vi_cond else None

            x_t, bfHP = self._denoise_with_adapter_em_replacement(
                x_t=x_t,
                history_fused_latents=history_fused_latents,
                history_ir_cond=history_ir_cond,
                current_ir_cond=current_ir_cond,
                history_vi_cond=history_vi_cond,
                current_vi_cond=current_vi_cond,
                ir_frame_original=ir_frame_original,
                vi_frame_original=vi_frame_original,
                bfHP=bfHP,
                guidance_scale=guidance_scale,
                stabilization_level=stabilization_level,
                eta=eta,
                verbose=verbose and frame_idx == 0,
            )

            frame_rgb_original = self._decode_with_resize(x_t)
            pred_y_original = self._rgb_to_y_channel(frame_rgb_original)
            vi_y_original = self._rgb_to_y_channel(vi_frame_original)

            if frame_idx == 0:
                bfHP_final = EM_Initial(ir_frame_original)

            fused_y_original, bfHP_final = self._em_fusion_step(
                pred_y_original,
                ir_frame_original,
                vi_y_original,
                bfHP_final,
            )

            fused_frames.append(fused_y_original)
            fused_latents.append(x_t)

        fused_video = torch.stack(fused_frames, dim=1)

        if verbose:
            print("\n" + "=" * 70)
            print("Generation complete")
            print(f"Output: {fused_video.shape}")
            print("=" * 70)

        return fused_video

    def _denoise_with_adapter_em_replacement(
        self,
        x_t,
        history_fused_latents,
        history_ir_cond,
        current_ir_cond,
        history_vi_cond,
        current_vi_cond,
        ir_frame_original,
        vi_frame_original,
        bfHP,
        guidance_scale,
        stabilization_level,
        eta,
        verbose=False,
    ):
        batch_size = x_t.shape[0]
        hist_len = history_fused_latents.shape[1]
        device = x_t.device

        history_guidance = HistoryGuidance.stabilized_vanilla(
            guidance_scale=guidance_scale,
            stabilization_level=stabilization_level,
            timesteps=self.timesteps,
        )

        if bfHP is None:
            bfHP = EM_Initial(ir_frame_original)

        timesteps_schedule = torch.linspace(
            self.timesteps - 1,
            0,
            self.sampling_timesteps,
            dtype=torch.long,
            device=device,
        )

        if verbose:
            print(f"    Denoising with adapter: {self.sampling_timesteps} steps")
            print(f"    History frames: {hist_len}")

        for step_idx, t_curr in enumerate(timesteps_schedule):
            t_next = (
                timesteps_schedule[step_idx + 1]
                if step_idx < len(timesteps_schedule) - 1
                else torch.tensor(-1, device=device)
            )
            t = torch.full((batch_size,), t_curr.item(), device=device, dtype=torch.long)

            x_window = torch.cat([history_fused_latents, x_t.unsqueeze(1)], dim=1)
            ir_cond_window = torch.cat([history_ir_cond, current_ir_cond.unsqueeze(1)], dim=1)

            vi_cond_window = None
            if self.use_vi_cond:
                vi_cond_window = torch.cat(
                    [history_vi_cond, current_vi_cond.unsqueeze(1)],
                    dim=1,
                )

            mask = torch.ones(batch_size, hist_len + 1, device=device) * 2
            mask[:, -1] = 0

            with history_guidance(mask) as hg:
                noise_levels_from = t.unsqueeze(1).expand(batch_size, hist_len + 1)
                noise_levels_to = noise_levels_from.clone()

                x_input, _, _, _ = hg.prepare(
                    x_window,
                    from_noise_levels=noise_levels_from,
                    to_noise_levels=noise_levels_to,
                    replacement_fn=self._add_noise_to_latent,
                )

                expanded_batch = x_input.shape[0]
                repeat_factor = expanded_batch // batch_size
                t_expanded = t[0].repeat(expanded_batch)
                ir_cond_expanded = ir_cond_window.repeat_interleave(repeat_factor, dim=0)
                vi_cond_expanded = None
                if vi_cond_window is not None:
                    vi_cond_expanded = vi_cond_window.repeat_interleave(repeat_factor, dim=0)

                pred_dict = self._adapter_predictions(
                    x_input,
                    t_expanded,
                    ir_cond=ir_cond_expanded,
                    vi_cond=vi_cond_expanded,
                )
                pred_x_start_window = hg.compose(pred_dict["pred_x_start"])

            pred_x_start_current = pred_x_start_window[:, -1:]

            if step_idx % self.em_every_n_steps == 0:
                pred_rgb_original = self._decode_with_resize(pred_x_start_current.squeeze(1))
                pred_y_original = self._rgb_to_y_channel(pred_rgb_original)
                vi_y_original = self._rgb_to_y_channel(vi_frame_original)

                fused_y_em, bfHP = self._em_fusion_step(
                    pred_y_original,
                    ir_frame_original,
                    vi_y_original,
                    bfHP,
                )
                pred_x_start_current = self._encode_with_resize(
                    fused_y_em.repeat(1, 3, 1, 1)
                ).unsqueeze(1)

                if verbose and step_idx == 0:
                    diff = (fused_y_em - pred_y_original).abs().mean().item()
                    print(f"      Applied EM replacement (diff={diff:.4f})")

            x_t_window = x_window[:, -1:].clone()
            x_t = self.diffusion_model.ddim_sample_step(
                x_t_window,
                t,
                t_next,
                pred_x_start_current,
                eta=eta,
            ).squeeze(1)

        return x_t, bfHP

    def _em_fusion_step(self, pred_y, ir_frame, vi_y_frame, bfHP):
        return EM_onestep(
            f_pre=pred_y,
            I=ir_frame,
            V=vi_y_frame,
            HyperP=bfHP,
            lamb=self.lamb,
            rho=self.rho,
        )

    def _add_noise_to_latent(self, x, t):
        noised_frames = []
        for frame_idx in range(x.shape[1]):
            noised_frames.append(
                self.diffusion_model.q_sample(
                    x[:, frame_idx],
                    t[:, frame_idx],
                    torch.randn_like(x[:, frame_idx]),
                )
            )
        return torch.stack(noised_frames, dim=1)
