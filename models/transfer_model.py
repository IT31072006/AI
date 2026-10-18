"""
Transfer Learning Models for Football Player Face Recognition.
Supports MobileNetV2 and ResNet18 fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_transfer_model(
    model_name: str = "mobilenet_v2",
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Build transfer learning model with custom classification head.
    
    Args:
        model_name: 'mobilenet_v2' or 'resnet18'
        num_classes: number of output football player classes
        pretrained: whether to load ImageNet pretrained weights
        freeze_backbone: whether to freeze convolutional feature extractor
    """
    model_name = model_name.lower()
    
    if "resnet" in model_name:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace FC layer
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    else:  # Default: MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            # Freeze generic low-level edge features (0-13)
            for param in model.features[:14].parameters():
                param.requires_grad = False
            # Fine-tune high-level facial semantic filters (14-18)
            for param in model.features[14:].parameters():
                param.requires_grad = True
                
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    return model
