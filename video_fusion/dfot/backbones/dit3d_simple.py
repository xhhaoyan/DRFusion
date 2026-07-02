"""
Simplified 3D DiT for Video Fusion
Adapted from diffusion-forcing-transformer with essential features only
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps/noise levels into vectors.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: (N,) tensor of timesteps
            dim: embedding dimension
            max_period: controls the minimum frequency

        Returns:
            (N, dim) tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Initialize modulation layers to zero
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) input tokens
            c: (B, D) conditioning vector

        Returns:
            (B, N, D) output tokens
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention with modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        # MLP with modulation
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding layer for video.
    Converts (B, T, C, H, W) to (B, T*num_patches, D)
    """

    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        # Initialize weights
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)

        Returns:
            (B, T*num_patches, D)
        """
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)  # (B*T, D, H', W')
        x = rearrange(x, '(b t) d h w -> b (t h w) d', b=B, t=T)
        return x


class SimpleDiT3D(nn.Module):
    """
    Simplified 3D Diffusion Transformer for video fusion.

    This is a streamlined version focusing on core functionality.
    """

    def __init__(
        self,
        input_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        patch_size=16,
        img_size=256,
        max_frames=32,
    ):
        """
        Args:
            input_channels: Number of input channels (3 for RGB)
            hidden_size: Hidden dimension of transformer
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            patch_size: Spatial patch size
            img_size: Input image size
            max_frames: Maximum number of frames
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_channels,
            embed_dim=hidden_size
        )

        # Positional embeddings
        # We use learned positional embeddings for simplicity
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_frames * self.num_patches, hidden_size)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # Noise level embedding
        self.noise_level_embedder = TimestepEmbedder(hidden_size)

        # Conditional embedding (for IR conditioning only - DDFM approach)
        # Input: C channels (IR latents only, VI is in initialization)
        self.cond_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_channels,  # Only IR (DDFM style)
            embed_dim=hidden_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Output projection
        out_channels = patch_size ** 2 * input_channels
        self.final_linear = nn.Linear(hidden_size, out_channels, bias=True)

        # Initialize output layer to zero (helps training stability)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def unpatchify(self, x, T):
        """
        Convert patches back to image format.

        Args:
            x: (B, T*num_patches, patch_size^2*C)
            T: Number of frames

        Returns:
            (B, T, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        c = self.input_channels

        x = rearrange(x, 'b (t h w) (p q c) -> b t c (h p) (w q)',
                      t=T, h=h, w=w, p=p, q=p, c=c)
        return x

    def forward(self, x, noise_levels, external_cond=None):
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) noisy video
            noise_levels: (B,) or (B, T) noise levels (logSNR)
            external_cond: Optional external conditioning

        Returns:
            (B, T, C, H, W) predicted velocity
        """
        B, T, C, H, W = x.shape

        # Handle noise levels
        if noise_levels.dim() == 1:
            # (B,) -> (B, T) by repeating
            noise_levels = repeat(noise_levels, 'b -> b t', t=T)

        # Embed patches
        x = self.patch_embed(x)  # (B, T*num_patches, D)

        # Add positional embeddings
        x = x + self.pos_embed[:, :T * self.num_patches, :]

        # Embed noise levels (average over time for conditioning)
        c = self.noise_level_embedder(noise_levels.mean(dim=1))  # (B, D)

        # Add external conditioning if provided (IR only - DDFM approach)
        if external_cond is not None:
            # external_cond shape: (B, T, C, H, W) - only IR latents
            cond_tokens = self.cond_embed(external_cond)  # (B, T*num_patches, D)
            # Add as additional tokens (simple approach)
            x = x + cond_tokens  # Element-wise addition

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final norm and projection
        x = self.final_norm(x)
        x = self.final_linear(x)

        # Unpatchify
        x = self.unpatchify(x, T)

        return x
