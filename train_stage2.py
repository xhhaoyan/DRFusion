import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from video_fusion.dfot.backbones import SimpleDiT3D
from video_fusion.dfot.diffusion import ContinuousDiffusion
from video_fusion.dfot.adapters import create_adapter_from_pretrained
from video_fusion.data.video_fusion_dataset import VideoFusionDataset
from util.common import instantiate_from_config, load_model

def load_video_list_from_paths_file(paths_file):
    video_names = set()
    with open(paths_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if 'channel' in parts or 'channel2' in parts:
                channel_idx = parts.index('channel') if 'channel' in parts else parts.index('channel2')
                if channel_idx > 0:
                    video_name = parts[channel_idx - 1]
                    video_names.add(video_name)
    return video_names

def encode_with_vae(autoencoder, images):
    with torch.no_grad():
        h = autoencoder.encoder(images)
        h = autoencoder.quant_conv(h)
        quant, _, _ = autoencoder.quantize(h)
    return quant

def decode_with_vae(autoencoder, latents):
    with torch.no_grad():
        quant = autoencoder.post_quant_conv(latents)
        decoded = autoencoder.decoder(quant)
    return decoded

class UnsupervisedFusionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features[:16].to(device).eval()
        self.vgg = vgg
    def forward(self, fused, ir, vi):
        fused_3ch = fused.repeat(1, 3, 1, 1)
        vi_gray = vi.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        ir_3ch = ir.repeat(1, 3, 1, 1)
        fused_feat = self.vgg(fused_3ch)
        with torch.no_grad():
            ir_feat = self.vgg(ir_3ch)
            vi_feat = self.vgg(vi_gray)
        perceptual_loss = (
            F.mse_loss(fused_feat, ir_feat.detach()) * 0.5 +
            F.mse_loss(fused_feat, vi_feat.detach()) * 0.5
        )
        ssim_ir = self.ssim_loss(fused, ir)
        ssim_vi_gray = self.ssim_loss(fused, vi.mean(dim=1, keepdim=True))
        ssim_loss = (ssim_ir + ssim_vi_gray) / 2
        grad_loss = self.gradient_loss(fused, ir, vi)
        intensity_loss = self.intensity_preservation_loss(fused, ir, vi)
        total_loss = (
            perceptual_loss * 2.0 +
            ssim_loss * 1.0 +
            grad_loss * 1.0 +
            intensity_loss * 0.5
        )
        loss_dict = {
            'perceptual': perceptual_loss.item(),
            'ssim': ssim_loss.item(),
            'gradient': grad_loss.item(),
            'intensity': intensity_loss.item(),
        }
        return total_loss, loss_dict
    def ssim_loss(self, img1, img2, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
        ssim_map = (
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        )
        return 1 - ssim_map.mean()
    def gradient_loss(self, fused, ir, vi):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        fused_grad_x = F.conv2d(fused, sobel_x, padding=1)
        fused_grad_y = F.conv2d(fused, sobel_y, padding=1)
        fused_grad = torch.sqrt(fused_grad_x ** 2 + fused_grad_y ** 2 + 1e-6)
        ir_grad_x = F.conv2d(ir, sobel_x, padding=1)
        ir_grad_y = F.conv2d(ir, sobel_y, padding=1)
        ir_grad = torch.sqrt(ir_grad_x ** 2 + ir_grad_y ** 2 + 1e-6)
        vi_gray = vi.mean(dim=1, keepdim=True)
        vi_grad_x = F.conv2d(vi_gray, sobel_x, padding=1)
        vi_grad_y = F.conv2d(vi_gray, sobel_y, padding=1)
        vi_grad = torch.sqrt(vi_grad_x ** 2 + vi_grad_y ** 2 + 1e-6)
        max_grad = torch.max(ir_grad, vi_grad)
        grad_loss = F.l1_loss(fused_grad, max_grad)
        return grad_loss
    def intensity_preservation_loss(self, fused, ir, vi):
        vi_gray = vi.mean(dim=1, keepdim=True)
        fused_mean = fused.mean()
        ir_mean = ir.mean()
        vi_mean = vi_gray.mean()
        min_mean = torch.min(ir_mean, vi_mean)
        max_mean = torch.max(ir_mean, vi_mean)
        if fused_mean < min_mean:
            loss = (min_mean - fused_mean) ** 2
        elif fused_mean > max_mean:
            loss = (fused_mean - max_mean) ** 2
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss

def create_dataloaders(config):
    train_list_file = config['data'].get('train_list_file')
    val_list_file = config['data'].get('val_list_file')
    print(f"Loading train video list from: {train_list_file}")
    train_videos = load_video_list_from_paths_file(train_list_file)
    print(f"  Found {len(train_videos)} training videos")
    print(f"Loading val video list from: {val_list_file}")
    val_videos = load_video_list_from_paths_file(val_list_file)
    print(f"  Found {len(val_videos)} validation videos")
    train_dataset = VideoFusionDataset(
        root_dir=config['data']['train_root'],
        n_frames=config['data']['n_frames'],
        frame_skip=config['data'].get('frame_skip', 1),
        img_size=config['data'].get('img_size', 256),
        return_video_name=True,
        filter_video_names=train_videos,
    )
    val_dataset = VideoFusionDataset(
        root_dir=config['data']['train_root'],
        n_frames=config['data']['n_frames'],
        frame_skip=config['data'].get('frame_skip', 1),
        img_size=config['data'].get('img_size', 256),
        return_video_name=True,
        filter_video_names=val_videos,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
    )
    return train_loader, val_loader

def train_step_unsupervised(batch, autoencoder, adapter, diffusion_model,
                           optimizer, fusion_loss_fn, device):
    ir = batch['infrared'].to(device)
    vi = batch['visible'].to(device)
    B, T = vi.shape[:2]
    vi_flat = vi.reshape(B * T, *vi.shape[2:])
    ir_flat = ir.reshape(B * T, *ir.shape[2:])
    with torch.no_grad():
        vi_latent = encode_with_vae(autoencoder, vi_flat)
    vi_latent = vi_latent.reshape(B, T, *vi_latent.shape[1:])
    total_loss = 0.0
    loss_details = {
        'perceptual': 0.0,
        'ssim': 0.0,
        'gradient': 0.0,
        'intensity': 0.0,
    }
    for frame_idx in range(T):
        t = torch.randint(0, diffusion_model.timesteps, (B,), device=device).float()
        current_vi_latent = vi_latent[:, frame_idx]
        noise = torch.randn_like(current_vi_latent)
        noisy_vi = diffusion_model.q_sample(current_vi_latent, t, noise)
        history_frames = noisy_vi.unsqueeze(1)
        ir_cond = ir[:, frame_idx:frame_idx+1]
        v_pred = adapter(
            history_frames, t,
            ir_cond=ir_cond,
            vi_cond=None,
        )
        logsnr = diffusion_model.logsnr[t.long()]
        alpha_t = torch.sigmoid(logsnr).sqrt()
        sigma_t = torch.sigmoid(-logsnr).sqrt()
        alpha_t = alpha_t[:, None, None, None]
        sigma_t = sigma_t[:, None, None, None]
        v_pred_current = v_pred[:, -1]
        pred_latent = alpha_t * noisy_vi - sigma_t * v_pred_current
        pred_latent_flat = pred_latent.reshape(B, *pred_latent.shape[1:])
        fused_img = decode_with_vae(autoencoder, pred_latent_flat)
        fused_gray = fused_img.mean(dim=1, keepdim=True)
        current_ir = ir_flat[frame_idx * B:(frame_idx + 1) * B]
        current_vi = vi_flat[frame_idx * B:(frame_idx + 1) * B]
        frame_loss, loss_dict = fusion_loss_fn(fused_gray, current_ir, current_vi)
        total_loss += frame_loss
        for key in loss_details:
            loss_details[key] += loss_dict[key]
    loss = total_loss / T
    for key in loss_details:
        loss_details[key] /= T
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), loss_details
@torch.no_grad()

def validate(val_loader, autoencoder, adapter, diffusion_model, fusion_loss_fn, device):
    adapter.eval()
    total_loss = 0.0
    loss_details = {
        'perceptual': 0.0,
        'ssim': 0.0,
        'gradient': 0.0,
        'intensity': 0.0,
    }
    pbar = tqdm(val_loader, desc="Validating")
    for batch in pbar:
        ir = batch['infrared'].to(device)
        vi = batch['visible'].to(device)
        B, T = vi.shape[:2]
        vi_flat = vi.reshape(B * T, *vi.shape[2:])
        ir_flat = ir.reshape(B * T, *ir.shape[2:])
        vi_latent = encode_with_vae(autoencoder, vi_flat)
        vi_latent = vi_latent.reshape(B, T, *vi_latent.shape[1:])
        for frame_idx in range(T):
            t = torch.randint(0, diffusion_model.timesteps, (B,), device=device).float()
            current_vi_latent = vi_latent[:, frame_idx]
            noise = torch.randn_like(current_vi_latent)
            noisy_vi = diffusion_model.q_sample(current_vi_latent, t, noise)
            history_frames = noisy_vi.unsqueeze(1)
            ir_cond = ir[:, frame_idx:frame_idx+1]
            v_pred = adapter(history_frames, t, ir_cond=ir_cond, vi_cond=None)
            logsnr = diffusion_model.logsnr[t.long()]
            alpha_t = torch.sigmoid(logsnr).sqrt()[:, None, None, None]
            sigma_t = torch.sigmoid(-logsnr).sqrt()[:, None, None, None]
            v_pred_current = v_pred[:, -1]
            pred_latent = alpha_t * noisy_vi - sigma_t * v_pred_current
            fused_img = decode_with_vae(autoencoder, pred_latent)
            fused_gray = fused_img.mean(dim=1, keepdim=True)
            current_ir = ir_flat[frame_idx * B:(frame_idx + 1) * B]
            current_vi = vi_flat[frame_idx * B:(frame_idx + 1) * B]
            frame_loss, loss_dict = fusion_loss_fn(fused_gray, current_ir, current_vi)
            total_loss += frame_loss.item()
            for key in loss_details:
                loss_details[key] += loss_dict[key]
    adapter.train()
    num_samples = len(val_loader) * T
    avg_loss = total_loss / num_samples
    for key in loss_details:
        loss_details[key] /= num_samples
    return avg_loss, loss_details

def main():
    parser = argparse.ArgumentParser(description='Train Adapter (Unsupervised)')
    parser.add_argument('--config', type=str, default='configs/train_stage2_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    print("\n" + "=" * 70)
    print("UNSUPERVISED ADAPTER TRAINING")
    print("=" * 70)
    print("Training mode: Unsupervised (no ground truth needed)")
    print("Losses: Perceptual + SSIM + Gradient + Intensity")
    print("Conditioning: IR only (VI is the input to be denoised)")
    print("=" * 70 + "\n")
    print("1. Loading VAE...")
    autoencoder = instantiate_from_config(config['autoencoder']).to(device)
    checkpoint = torch.load(config['autoencoder']['ckpt_path'], map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
    autoencoder.load_state_dict(filtered_state_dict, strict=False)
    autoencoder.eval()
    print(f"   [OK] VAE loaded from {config['autoencoder']['ckpt_path']}")
    print(f"   [OK] Filtered out {len(state_dict) - len(filtered_state_dict)} loss-related keys")
    print(f"   [OK] VAE in eval mode (gradients enabled for backprop)")
    print("\n2. Creating 3D DiT backbone...")
    dit_config = config['model']['dit']
    backbone = SimpleDiT3D(
        input_channels=dit_config['input_channels'],
        hidden_size=dit_config['hidden_size'],
        depth=dit_config['depth'],
        num_heads=dit_config['num_heads'],
        mlp_ratio=dit_config['mlp_ratio'],
        patch_size=dit_config['patch_size'],
        img_size=dit_config['img_size'],
        max_frames=dit_config['max_frames'],
    ).to(device)
    if 'ckpt_path' in dit_config and dit_config['ckpt_path']:
        print(f"   Loading pretrained weights from {dit_config['ckpt_path']}...")
        checkpoint = torch.load(dit_config['ckpt_path'], map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('diffusion_model.model.', '')
            new_state_dict[new_key] = v
        backbone.load_state_dict(new_state_dict, strict=False)
        print("   [OK] Pretrained weights loaded")
    else:
        print("   [WARN] No pretrained weights specified; training from scratch")
    print("\n3. Creating Condition Adapter...")
    adapter = create_adapter_from_pretrained(
        pretrained_model=backbone,
        freeze_backbone=True,
    )
    adapter.train()
    print(f"   [OK] Adapter created with {adapter.count_trainable_params() / 1e6:.2f}M trainable params")
    print("\n4. Creating Continuous Diffusion wrapper...")
    diffusion_config = config['diffusion']
    diffusion_model = ContinuousDiffusion(
        model=adapter,
        timesteps=diffusion_config['timesteps'],
        sampling_timesteps=diffusion_config['sampling_timesteps'],
        logsnr_min=diffusion_config.get('logsnr_min', -15.0),
        logsnr_max=diffusion_config.get('logsnr_max', 15.0),
        shift=diffusion_config.get('shift', 1.0),
        precond_scale=diffusion_config.get('precond_scale', 1.0),
        sigmoid_bias=diffusion_config.get('sigmoid_bias', 0.0),
    ).to(device)
    print("   [OK] Diffusion wrapper created")
    print("\n5. Creating unsupervised fusion loss...")
    fusion_loss_fn = UnsupervisedFusionLoss(device)
    print("   [OK] Loss function created (VGG16 loaded for perceptual loss)")
    print("\n6. Creating optimizer...")
    optimizer = torch.optim.AdamW(
        adapter.condition_encoder.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
    )
    print(f"   [OK] AdamW optimizer created (lr={config['training']['learning_rate']})")
    print("\n7. Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"   [OK] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    start_epoch = 0
    if args.resume:
        print(f"\n8. Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"   [OK] Resumed from epoch {checkpoint['epoch']}")
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    num_epochs = config['training']['num_epochs']
    save_interval = config['training'].get('save_interval', 10)
    for epoch in range(start_epoch, num_epochs):
        adapter.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_loss_details = {
            'perceptual': 0.0,
            'ssim': 0.0,
            'gradient': 0.0,
            'intensity': 0.0,
        }
        for batch in pbar:
            loss, loss_details = train_step_unsupervised(
                batch, autoencoder, adapter, diffusion_model,
                optimizer, fusion_loss_fn, device
            )
            epoch_loss += loss
            for key in epoch_loss_details:
                epoch_loss_details[key] += loss_details[key]
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        avg_loss = epoch_loss / len(train_loader)
        for key in epoch_loss_details:
            epoch_loss_details[key] /= len(train_loader)
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        for key, val in epoch_loss_details.items():
            writer.add_scalar(f'Train/{key}', val, epoch)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        print(f"  Perceptual: {epoch_loss_details['perceptual']:.4f}")
        print(f"  SSIM: {epoch_loss_details['ssim']:.4f}")
        print(f"  Gradient: {epoch_loss_details['gradient']:.4f}")
        print(f"  Intensity: {epoch_loss_details['intensity']:.4f}")
        if (epoch + 1) % 5 == 0:
            val_loss, val_loss_details = validate(
                val_loader, autoencoder, adapter, diffusion_model,
                fusion_loss_fn, device
            )
            writer.add_scalar('Val/Loss', val_loss, epoch)
            for key, val in val_loss_details.items():
                writer.add_scalar(f'Val/{key}', val, epoch)
            print(f"Validation Loss: {val_loss:.4f}")
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = output_dir / f'adapter_unsup_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'adapter_state_dict': adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"[OK] Saved checkpoint: {checkpoint_path}")
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    writer.close()

if __name__ == '__main__':
    main()
