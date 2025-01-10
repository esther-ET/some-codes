import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 1. Semantic Segmentation Branch
# ==============================
class SemanticSegmentationBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemanticSegmentationBranch, self).__init__()

        # 定义残差块
        self.residual_blocks = nn.Sequential(
            self._make_residual_block(in_channels, 64),
            self._make_residual_block(64, 128),
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 512),
            self._make_residual_block(512, 512)
        )

        # 最大池化层
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 上采样层
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 卷积层
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1)

        # BatchNorm 和 ReLU
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def _make_residual_block(self, in_channels, out_channels):
        # 残差块
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 残差块
        x = self.residual_blocks(x)

        # 最大池化
        x = self.maxpool1(x)
        x = self.maxpool2(x)

        # 上采样
        x = self.upsample1(x)
        x = self.upsample2(x)

        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        # 输出语义掩码
        return torch.sigmoid(x)  # 使用 sigmoid 激活函数生成概率图


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # 定义U-Net的下采样和上采样层
        self.down1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 下采样
        x1 = F.relu(self.down1(x))
        x2 = F.relu(self.down2(x1))

        # 上采样
        x3 = F.relu(self.up1(x2))
        x4 = self.up2(x3)

        return x4

# ==============================
# 2. Fusion Module
# ==============================
class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()

    def forward(self, features, seg_mask):
        # 特征融合：根据语义分割掩码重新加权特征图
        return (1 + seg_mask) * features


# ==============================
# 3. 完整的 Semantic Context Encoder (SCE)
# ==============================
class SemanticContextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemanticContextEncoder, self).__init__()

        # 主检测分支（U-Net结构）
        self.unet = UNet(in_channels, out_channels)

        # 语义分割分支
        self.segmentation_branch = SemanticSegmentationBranch(in_channels, out_channels)

        # 融合模块
        self.fusion = FusionModule()

    def forward(self, x):
        # 主检测分支
        unet_output = self.unet(x)

        # 语义分割分支生成语义掩码
        seg_mask = self.segmentation_branch(x)

        # 特征融合
        fused_features = self.fusion(unet_output, seg_mask)

        return fused_features, seg_mask


# ==============================
# 4. 损失函数
# ==============================
def semantic_segmentation_loss(seg_mask, gt_mask):
    # 计算交叉熵损失
    return F.binary_cross_entropy(seg_mask, gt_mask)


# ==============================
# 测试代码
# ==============================
if __name__ == "__main__":
    # 输入 BEV 特征图（假设来自 VFE）
    input_bev = torch.randn(1, 64, 8, 8)  # (batch_size, channels, height, width)

    # 初始化 SCE
    sce = SemanticContextEncoder(in_channels=64, out_channels=64)

    # 前向传播
    fused_features, seg_mask = sce(input_bev)

    # 假设有 ground truth 语义掩码
    gt_mask = torch.rand(1, 64, 8, 8).float()  # 二值掩码  [4,3,496,432]

    # 计算语义分割损失
    loss = semantic_segmentation_loss(seg_mask, gt_mask)

    print("Fused Features Shape:", fused_features.shape)
    print("Segmentation Mask Shape:", seg_mask.shape)
    print("Semantic Segmentation Loss:", loss.item())

