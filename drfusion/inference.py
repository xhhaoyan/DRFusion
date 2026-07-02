"""Model loading and IO helpers for DRFusion inference."""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from util.common import instantiate_from_config, load_model
from video_fusion.dfot.adapters import create_adapter_from_pretrained
from video_fusion.dfot.backbones import SimpleDiT3D
from video_fusion.dfot.diffusion import ContinuousDiffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def load_config(config_path):
    with open(resolve_path(config_path), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    """Resolve a config path relative to the public repository root."""
    if path_value is None:
        return None

    path = Path(str(path_value)).expanduser()
    if path.exists():
        return str(path)
    if not path.is_absolute():
        return str((base_dir / path).resolve())
    return str(path)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("adapter_state_dict", "state_dict", "model", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _strip_prefixes(state_dict, prefixes):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _load_compatible_state_dict(module, state_dict, label):
    model_state = module.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value

    if skipped:
        print(f"   Skipping {len(skipped)} {label} key(s) with mismatched shapes:")
        for key, ckpt_shape, model_shape in skipped[:8]:
            print(f"     {key}: checkpoint {ckpt_shape} -> model {model_shape}")
        if len(skipped) > 8:
            print(f"     ... {len(skipped) - 8} more")

    missing, unexpected = module.load_state_dict(compatible, strict=False)
    print(f"   {label} loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    return missing, unexpected


def load_backbone_checkpoint(backbone, ckpt_path, device):
    ckpt_path = resolve_path(ckpt_path)
    if not ckpt_path or not Path(ckpt_path).exists():
        print("   No valid 3D-DiT checkpoint found; using initialized backbone weights.")
        return

    print(f"   Loading 3D-DiT checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    state_dict = _strip_prefixes(state_dict, prefixes=("diffusion_model.model.", "model."))
    _load_compatible_state_dict(backbone, state_dict, "3D-DiT")


def load_adapter_checkpoint(adapter, ckpt_path, device):
    ckpt_path = resolve_path(ckpt_path)
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {ckpt_path}")

    print(f"   Loading stage-2 adapter checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    state_dict = _strip_prefixes(state_dict, prefixes=())
    _load_compatible_state_dict(adapter, state_dict, "ConditionAdapter")

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(f"   Adapter checkpoint epoch: {checkpoint['epoch']}")
        if "loss" in checkpoint:
            print(f"   Adapter checkpoint loss: {checkpoint['loss']}")


def build_models(config, device, vae_ckpt=None, dit_ckpt=None, adapter_ckpt=None):
    print("=" * 70)
    print("Building VAE + 3D-DiT + stage-2 ConditionAdapter")
    print("=" * 70)

    print("\n1. Loading VAE...")
    autoencoder = instantiate_from_config(config["autoencoder"]).to(device)
    vae_path = resolve_path(vae_ckpt or config["autoencoder"]["ckpt_path"])
    if not Path(vae_path).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    load_model(autoencoder, vae_path, device)
    autoencoder.eval()
    print(f"   VAE loaded from {vae_path}")

    print("\n2. Creating 3D-DiT backbone...")
    dit_config = config["model"]["dit"]
    backbone = SimpleDiT3D(
        input_channels=dit_config["input_channels"],
        hidden_size=dit_config["hidden_size"],
        depth=dit_config["depth"],
        num_heads=dit_config["num_heads"],
        mlp_ratio=dit_config["mlp_ratio"],
        patch_size=dit_config["patch_size"],
        img_size=dit_config["img_size"],
        max_frames=dit_config["max_frames"],
    ).to(device)

    load_backbone_checkpoint(backbone, dit_ckpt or config["model"].get("pretrained_path"), device)

    print("\n3. Wrapping backbone with ConditionAdapter...")
    adapter_config = config.get("adapter", {})
    if not adapter_config.get("enabled", True):
        raise ValueError("adapter.enabled is false. DRFusion inference requires the stage-2 adapter.")

    adapter = create_adapter_from_pretrained(
        pretrained_model=backbone,
        freeze_backbone=True,
    ).to(device)

    adapter_path = adapter_ckpt or adapter_config.get("ckpt_path")
    if not adapter_path:
        raise ValueError("Adapter checkpoint is not configured. Set adapter.ckpt_path in the config.")
    load_adapter_checkpoint(adapter, adapter_path, device)
    adapter.eval()

    print("\n4. Creating ContinuousDiffusion wrapper...")
    diffusion_config = config["diffusion"]
    diffusion_model = ContinuousDiffusion(
        model=adapter,
        timesteps=diffusion_config["timesteps"],
        sampling_timesteps=diffusion_config["sampling_timesteps"],
        logsnr_min=diffusion_config.get("logsnr_min", -15.0),
        logsnr_max=diffusion_config.get("logsnr_max", 15.0),
        shift=diffusion_config.get("shift", 1.0),
        precond_scale=diffusion_config.get("precond_scale", 1.0),
        sigmoid_bias=diffusion_config.get("sigmoid_bias", 0.0),
    ).to(device)

    print("\nModels built successfully.\n")
    return autoencoder, diffusion_model


def _collect_images(folder):
    files = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(Path(folder).glob(pattern))
    return sorted(files)


def get_video_list(video_dir):
    video_dir = Path(resolve_path(video_dir))
    ir_dir = video_dir / "infrared"
    vi_dir = video_dir / "visible"

    if ir_dir.exists() and vi_dir.exists():
        return [video_dir]

    valid_videos = []
    for subdir in video_dir.iterdir():
        if not subdir.is_dir():
            continue
        if (subdir / "infrared").exists() and (subdir / "visible").exists():
            valid_videos.append(subdir)

    if not valid_videos:
        raise ValueError(f"No valid video directories found in {video_dir}")

    return sorted(valid_videos)


def load_video_frames(video_dir, frame_limit=None):
    video_dir = Path(resolve_path(video_dir))
    video_name = video_dir.name

    ir_files = _collect_images(video_dir / "infrared")
    vi_files = _collect_images(video_dir / "visible")

    if frame_limit:
        ir_files = ir_files[:frame_limit]
        vi_files = vi_files[:frame_limit]

    if not ir_files or not vi_files:
        raise ValueError(f"No IR/VI frames found in {video_dir}")

    num_frames = min(len(ir_files), len(vi_files))
    ir_files = ir_files[:num_frames]
    vi_files = vi_files[:num_frames]
    print(f"  Loading {num_frames} frames from {video_name} at original size...")

    ir_frames = []
    vi_frames = []
    for ir_file, vi_file in zip(ir_files, vi_files):
        ir = cv2.imread(str(ir_file), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise ValueError(f"Failed to read IR frame: {ir_file}")
        ir = ir.astype(np.float32) / 255.0
        ir_frames.append(ir * 2.0 - 1.0)

        vi = cv2.imread(str(vi_file), cv2.IMREAD_COLOR)
        if vi is None:
            raise ValueError(f"Failed to read VI frame: {vi_file}")
        vi = cv2.cvtColor(vi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        vi_frames.append(vi * 2.0 - 1.0)

    ir_tensor = torch.from_numpy(np.stack(ir_frames)).unsqueeze(0).unsqueeze(2)
    vi_tensor = torch.from_numpy(np.stack(vi_frames)).permute(0, 3, 1, 2).unsqueeze(0)
    original_h, original_w = ir_frames[0].shape
    return ir_tensor, vi_tensor, ir_files, video_name, (original_w, original_h)


def save_colored_fused_video(fused_tensor, output_dir, frame_files, visible_tensor):
    """Save fused luminance with visible-image chroma."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fused_tensor.dim() == 5:
        fused_tensor = fused_tensor[0]
    if visible_tensor.dim() == 5:
        visible_tensor = visible_tensor[0]

    fused_frames = fused_tensor[:, 0].detach().cpu().numpy()
    visible_frames = visible_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

    for frame_idx, (frame, original_file) in enumerate(zip(fused_frames, frame_files)):
        frame = (frame + 1.0) / 2.0
        frame_min, frame_max = frame.min(), frame.max()
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        gray = (frame * 255.0).astype(np.uint8)

        visible_rgb = (visible_frames[frame_idx] + 1.0) / 2.0
        visible_rgb = np.clip(visible_rgb, 0.0, 1.0)
        visible_rgb = (visible_rgb * 255.0).astype(np.uint8)

        if gray.shape[:2] != visible_rgb.shape[:2]:
            gray = cv2.resize(
                gray,
                (visible_rgb.shape[1], visible_rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        visible_ycrcb = cv2.cvtColor(visible_rgb, cv2.COLOR_RGB2YCrCb)
        visible_ycrcb[:, :, 0] = gray
        colored_rgb = cv2.cvtColor(visible_ycrcb, cv2.COLOR_YCrCb2RGB)
        colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / original_file.name), colored_bgr)

    print(f"Saved {len(fused_frames)} frames to {output_dir}")
