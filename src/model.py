"""
Model Architecture Module: ResNet-18 with Transfer Learning.
=============================================================
Defines the deep convolutional neural network based on the ResNet-18 architecture,
implements feature extraction freezing for baseline training, and enables full
backbone unfreezing for fine-tuning.

Why ResNet-18?
--------------
ResNet (Residual Network), introduced by He et al., introduces 'residual skip connections'
that allow gradients to flow directly backwards across layers, mitigating the vanishing
gradient problem. ResNet-18 has ~11.2 million parameters:
1. Fast training and lightweight footprint (ideal for fast inference and deployment).
2. Pretrained on ImageNet (1.28 million natural images), its early convolutional layers
   already know how to detect universal visual primitives: edges, gradients, textures,
   and curves—which are directly relevant to finding lung opacities in X-rays.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.config import config


def build_resnet18(
    pretrained: bool = True,
    num_classes: int = config.NUM_CLASSES,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Constructs a ResNet-18 model tailored for binary Chest X-Ray classification.
    
    Args:
        pretrained: If True, downloads/loads ImageNet pretrained weights (DEFAULT).
        num_classes: Number of output logits (2: NORMAL, PNEUMONIA).
        freeze_backbone: If True, freezes all convolutional backbone layers
                         so only the newly attached classification head is trained.
                         
    Returns:
        Configured PyTorch nn.Module.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # 1. Optionally freeze all feature extraction layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("[MODEL] ResNet-18 convolutional backbone layers FROZEN (Feature Extraction mode).")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("[MODEL] ResNet-18 backbone layers UNFROZEN (Full Fine-Tuning mode).")

    # 2. Replace the final fully connected layer (head)
    # The default ImageNet head outputs 1000 classes. We replace it with our 2-class linear layer.
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    
    # Ensure the new classification head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    print(f"[MODEL] Replaced final FC head: Linear(in_features={in_features}, out_features={num_classes})")
    return model


def unfreeze_all_layers(model: nn.Module) -> None:
    """
    Unfreezes every parameter across all layers in the ResNet-18 network.
    Used before starting fine-tuning so the convolutional filters can adapt
    specifically to chest radiographs.
    """
    for name, param in model.named_parameters():
        param.requires_grad = True
    print("[MODEL] All ResNet-18 layers successfully UNFROZEN for fine-tuning.")


def get_parameter_summary(model: nn.Module) -> Dict[str, int]:
    """
    Computes total, trainable, and frozen parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n--- Model Parameter Summary ---")
    print(f"  - Total Parameters    : {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"  - Frozen Parameters   : {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }

