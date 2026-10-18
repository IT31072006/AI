"""
Custom Convolutional Neural Network (CNN) for Football Player Face Recognition.
Designed specifically for multi-class facial image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard Conv-BN-ReLU Block with optional residual connection and dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True, dropout_rate: float = 0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout(out)
        return out


class PlayerFaceCNN(nn.Module):
    """
    Hierarchical Deep CNN for Football Player Face Identification.
    Architecture:
      Input (3, H, W)
      -> Block 1: 3 -> 32 channels (MaxPool: H/2, W/2)
      -> Block 2: 32 -> 64 channels (MaxPool: H/4, W/4)
      -> Block 3: 64 -> 128 channels (MaxPool: H/8, W/8)
      -> Block 4: 128 -> 256 channels (MaxPool: H/16, W/16)
      -> AdaptiveAvgPool2d((1, 1))
      -> Dense Classifier: 256 -> 128 -> num_classes
    """
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.4):
        super(PlayerFaceCNN, self).__init__()
        self.num_classes = num_classes

        # Feature Extractor
        self.block1 = ConvBlock(in_channels=3, out_channels=32, pool=True, dropout_rate=0.15)
        self.block2 = ConvBlock(in_channels=32, out_channels=64, pool=True, dropout_rate=0.2)
        self.block3 = ConvBlock(in_channels=64, out_channels=128, pool=True, dropout_rate=0.25)
        self.block4 = ConvBlock(in_channels=128, out_channels=256, pool=True, dropout_rate=0.3)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck feature vector (256-dim)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        out = self.classifier(x)
        return out

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_custom_cnn(num_classes: int = 8) -> PlayerFaceCNN:
    return PlayerFaceCNN(num_classes=num_classes)
