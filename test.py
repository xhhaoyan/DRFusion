"""Run DRFusion inference with the stage-2 adapter checkpoint."""

import argparse
import os

import torch

from drfusion.inference import (
    build_models,
    get_video_list,
    load_config,
    load_video_frames,
    save_colored_fused_video,
)
from drfusion.samplers import AdapterEMReplacementDiffusionForcingSampler


def parse_args():
    parser = argparse.ArgumentParser(description="DRFusion stage-2 adapter inference")
    parser.add_argument("--config", type=str, default="configs/video_fusion_config.yaml")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--max_context_frames", type=int, default=8)
    parser.add_argument("--vae_size", type=int, default=256)
    parser.add_argument("--em_every_n_steps", type=int, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Override autoencoder.ckpt_path.")
    parser.add_argument("--dit_ckpt", type=str, default=None, help="Override model.pretrained_path.")
    parser.add_argument("--adapter_ckpt", type=str, default=None, help="Override adapter.ckpt_path.")
    parser.add_argument(
        "--use_vi_cond",
        action="store_true",
        help="Feed visible frames to the adapter. The released default is IR-only.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    autoencoder, diffusion_model = build_models(
        config,
        device,
        vae_ckpt=args.vae_ckpt,
        dit_ckpt=args.dit_ckpt,
        adapter_ckpt=args.adapter_ckpt,
    )

    adapter_config = config.get("adapter", {})
    use_vi_cond = args.use_vi_cond or bool(adapter_config.get("use_vi_cond", False))
    em_every_n_steps = args.em_every_n_steps or config.get("fusion", {}).get("em_every_n_steps", 5)

    sampler = AdapterEMReplacementDiffusionForcingSampler(
        diffusion_model=diffusion_model,
        autoencoder=autoencoder,
        timesteps=config["diffusion"]["timesteps"],
        sampling_timesteps=config["diffusion"]["sampling_timesteps"],
        lamb=config["fusion"]["lamb"],
        rho=config["fusion"]["rho"],
        max_context_frames=args.max_context_frames,
        vae_size=args.vae_size,
        em_every_n_steps=em_every_n_steps,
        use_vi_cond=use_vi_cond,
        device=device,
    )

    video_list = get_video_list(args.video_dir)
    print(f"Found {len(video_list)} video(s) to process.")

    for video_idx, video_path in enumerate(video_list, 1):
        print("\n" + "=" * 70)
        print(f"Processing video {video_idx}/{len(video_list)}: {video_path.name}")
        print("=" * 70)

        ir_tensor, vi_tensor, frame_files, video_name, original_size = load_video_frames(
            video_path,
            frame_limit=args.frame_limit,
        )
        print(f"  Original size: {original_size[0]}x{original_size[1]}")

        ir_tensor = ir_tensor.to(device)
        vi_tensor = vi_tensor.to(device)

        with torch.no_grad():
            fused_tensor = sampler.sample_autoregressive(
                infrared_seq=ir_tensor,
                visible_seq=vi_tensor,
                context_frames=config["sampling"]["context_frames"],
                guidance_scale=config["sampling"]["history_guidance"]["guidance_scale"],
                stabilization_level=config["sampling"]["history_guidance"].get("stabilization_level", 0.02),
                eta=config["sampling"].get("eta", 0.0),
                verbose=not args.quiet and video_idx == 1,
            )

        save_colored_fused_video(
            fused_tensor,
            os.path.join(args.output_dir, video_name),
            frame_files,
            vi_tensor,
        )

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
