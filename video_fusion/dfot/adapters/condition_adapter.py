"""
Lightweight Condition Adapter for IR/VI Fusion
轻量级条件适配器：冻结预训练3D-DiT，只训练适配器

核心思路：
1. 冻结预训练的3D-DiT主干
2. 添加小型CNN编码器处理IR+VI条件
3. 通过残差连接注入到3D-DiT的条件输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """简单的ResNet Block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.shortcut(x)


class ConditionEncoder(nn.Module):
    """
    轻量级条件编码器

    输入：IR (1通道) + VI (3通道) = 4通道
    输出：与潜空间相同维度的条件特征

    设计原则：
    - 轻量级（<5M参数）
    - 输出与VAE潜空间相同尺寸
    - 使用残差块提升表达能力
    """

    def __init__(
        self,
        input_channels=4,  # IR(1) + VI(3)
        output_channels=3,  # 与VAE潜空间相同
        base_channels=64,
        num_res_blocks=3,
        downsample_factor=4,  # 256->64的下采样倍数
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsample_factor = downsample_factor

        # 初始卷积：4通道 -> 64通道
        self.init_conv = nn.Conv2d(input_channels, base_channels, 7, 1, 3)

        # 下采样层（256 -> 64, 需要4倍下采样）
        # 使用两个stride=2的卷积实现4倍下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),  # 2x下采样
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),  # 再2x下采样
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(base_channels, base_channels)
            for _ in range(num_res_blocks)
        ])

        # 输出投影：64通道 -> 3通道（潜空间维度）
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, 3, 1, 1),
        )

        # 初始化输出层为小值（避免破坏预训练特征）
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(self, ir, vi):
        """
        Args:
            ir: (B, T, 1, H, W) - 红外图像 (256x256)
            vi: (B, T, 3, H, W) - 可见光图像 (256x256)

        Returns:
            (B, T, 3, H', W') - 条件特征 (64x64, 潜空间尺寸)
        """
        B, T = ir.shape[:2]

        # 合并批次和时间维度
        ir = ir.reshape(B * T, 1, *ir.shape[-2:])
        vi = vi.reshape(B * T, 3, *vi.shape[-2:])

        # 通道拼接
        x = torch.cat([ir, vi], dim=1)  # (B*T, 4, H, W)

        # 编码
        x = self.init_conv(x)

        # 下采样到潜空间尺寸 (256 -> 64)
        x = self.downsample(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.out_conv(x)

        # 恢复时间维度
        x = x.reshape(B, T, self.output_channels, *x.shape[-2:])

        return x


class ConditionAdapter(nn.Module):
    """
    条件适配器：包装预训练模型 + 条件编码器

    使用方法：
    1. 加载预训练的3D-DiT
    2. 冻结3D-DiT参数
    3. 只训练ConditionEncoder
    4. 推理时将条件特征注入到external_cond
    """

    def __init__(
        self,
        pretrained_model,
        latent_channels=3,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = pretrained_model

        # 冻结主干
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen (only adapter will be trained)")

        # 条件编码器（可训练）
        self.condition_encoder = ConditionEncoder(
            input_channels=4,  # IR(1) + VI(3)
            output_channels=latent_channels,
            base_channels=64,
            num_res_blocks=3,
        )

        print(f"✓ Condition Adapter initialized")
        print(f"  - Trainable params: {self.count_trainable_params() / 1e6:.2f}M")
        print(f"  - Frozen params: {self.count_frozen_params() / 1e6:.2f}M")

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_frozen_params(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def forward(self, x, t, ir_cond, vi_cond=None, external_cond=None):
        """
        Args:
            x: (B, T, C, H, W) - 噪声潜空间
            t: (B,) - 时间步
            ir_cond: (B, T, 1, H, W) - 红外条件
            vi_cond: (B, T, 3, H, W) - 可见光条件（可选，如果为None则用零填充）
            external_cond: 可选的额外条件（如历史帧）

        Returns:
            预测的去噪结果
        """
        # 如果vi_cond为None，用零填充（仅使用IR条件）
        if vi_cond is None:
            B, T = ir_cond.shape[:2]
            H, W = ir_cond.shape[-2:]
            vi_cond = torch.zeros(B, T, 3, H, W, device=ir_cond.device, dtype=ir_cond.dtype)

        # 编码条件
        cond_feat = self.condition_encoder(ir_cond, vi_cond)  # (B, T, 3, H, W)

        # 如果有external_cond（如历史帧），与之相加
        if external_cond is not None:
            cond_feat = cond_feat + external_cond

        # 通过主干
        return self.backbone(x, t, external_cond=cond_feat)

    def get_trainable_parameters(self):
        """返回只需要训练的参数（即条件编码器）"""
        return self.condition_encoder.parameters()


def create_adapter_from_pretrained(pretrained_model, freeze_backbone=True):
    """
    从预训练模型创建适配器

    Args:
        pretrained_model: 预训练的3D-DiT模型
        freeze_backbone: 是否冻结主干

    Returns:
        ConditionAdapter实例
    """
    return ConditionAdapter(
        pretrained_model=pretrained_model,
        latent_channels=3,
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    # 测试代码
    from video_fusion.dfot.backbones import SimpleDiT3D

    # 创建预训练模型
    backbone = SimpleDiT3D(
        input_channels=3,
        hidden_size=512,
        depth=8,
        num_heads=8,
        patch_size=16,
        img_size=64,
        max_frames=16,
    )

    # 创建适配器
    adapter = create_adapter_from_pretrained(backbone, freeze_backbone=True)

    # 测试前向传播
    B, T, C, H, W = 2, 4, 3, 64, 64
    x = torch.randn(B, T, C, H, W)
    t = torch.randint(0, 1000, (B,))
    ir_cond = torch.randn(B, T, 1, H, W)
    vi_cond = torch.randn(B, T, 3, H, W)

    output = adapter(x, t, ir_cond, vi_cond)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"✓ Adapter test passed")
