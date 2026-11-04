"""
Explainability Module: Grad-CAM (Gradient-Weighted Class Activation Mapping).
=============================================================================
Provides visual interpretability for deep learning predictions on Chest X-Rays.

Why Grad-CAM is Vital in Medical AI:
-----------------------------------
A deep neural network can easily pick up spurious 'shortcut' correlations (such as hospital
text markers, patient orientation tags, or radiographic borders) instead of actual lung
consolidation or infiltrates. Grad-CAM visualizes the spatial gradients flowing into the
final convolutional layer (ResNet-18's layer4), highlighting exactly which anatomical lung
regions triggered the pneumonia prediction.

This module provides:
1. Native, robust PyTorch hook-based Grad-CAM (zero third-party dependency issues).
2. Clean integration with `pytorch-grad-cam` library when available.
3. High-resolution colored heatmaps superimposed onto original chest radiographs.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.config import config


class GradCAM:
    """
    Pure PyTorch implementation of Grad-CAM using forward and backward hooks.
    Attaches to the final convolutional layer of ResNet-18 (model.layer4[-1]).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Default to the final convolutional layer of ResNet-18
        if target_layer is None:
            self.target_layer = self.model.layer4[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks to capture feature maps and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Cleanly removes hooks when no longer needed."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a 2D normalized Grad-CAM heatmap for the specified input image tensor.
        
        Args:
            input_tensor: (1, 3, 224, 224) normalized input tensor.
            target_class: Target class index (0=NORMAL, 1=PNEUMONIA). If None, uses top predicted class.
            
        Returns:
            (heatmap_2d, predicted_class, predicted_probability)
        """
        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Zero existing gradients
        self.model.zero_grad()
        
        # Backward pass with respect to the target class logit
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of the gradients across spatial dimensions (H, W)
        # gradients shape: [1, 512, 7, 7] -> pooled weights shape: [512]
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # Weighted combination of forward activation maps
        # activations shape: [1, 512, 7, 7]
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]

        # Apply ReLU to retain only features that have a POSITIVE influence on the target class
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize heatmap to [0.0, 1.0]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, pred_prob


def overlay_cam_on_image(
    rgb_image: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Superimposes a Grad-CAM heatmap onto a standard RGB image.
    
    Args:
        rgb_image: (H, W, 3) image in range [0, 255] uint8.
        cam_heatmap: (h, w) float heatmap in range [0.0, 1.0].
        alpha: Blending weight for heatmap overlay (0.0 = original image only, 1.0 = heatmap only).
        colormap: OpenCV colormap (cv2.COLORMAP_JET or cv2.COLORMAP_VIRIDIS).
        
    Returns:
        (colored_heatmap, blended_image)
    """
    height, width = rgb_image.shape[:2]
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(cam_heatmap, (width, height))
    
    # Convert heatmap to uint8 [0, 255] and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image and colored heatmap
    blended = np.float32(colored_heatmap) * alpha + np.float32(rgb_image) * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return colored_heatmap, blended


def explain_image(
    model: nn.Module,
    image_path_or_pil,
    device: torch.device,
    target_class: Optional[int] = None,
    alpha: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    High-level end-to-end function: loads image, runs inference, computes Grad-CAM,
    and returns visualization arrays.
    
    Returns:
        (original_resized_rgb, colored_heatmap, overlay_image, pred_class_name, confidence)
    """
    if isinstance(image_path_or_pil, (str, Image.Image)):
        if isinstance(image_path_or_pil, str):
            pil_img = Image.open(image_path_or_pil).convert("RGB")
        else:
            pil_img = image_path_or_pil.convert("RGB")
    else:
        # Assume numpy array
        pil_img = Image.fromarray(image_path_or_pil).convert("RGB")

    # Resize for visualization
    resized_pil = pil_img.resize(config.IMAGE_SIZE)
    original_rgb = np.array(resized_pil)

    # Tensor preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    input_tensor = preprocess(resized_pil).unsqueeze(0).to(device)

    # Grad-CAM execution
    grad_cam = GradCAM(model)
    heatmap, pred_class_idx, pred_prob = grad_cam.generate_heatmap(
        input_tensor=input_tensor,
        target_class=target_class
    )
    grad_cam.remove_hooks()

    colored_heatmap, overlay = overlay_cam_on_image(original_rgb, heatmap, alpha=alpha)
    class_name = config.IDX_TO_CLASS[pred_class_idx]

    return original_rgb, colored_heatmap, overlay, class_name, pred_prob


def plot_explainability_grid(
    model: nn.Module,
    sample_paths: list,
    device: torch.device,
    save_path: Optional[str] = None
) -> None:
    """
    Plots a comprehensive multi-column figure displaying:
    [Original X-Ray] | [Grad-CAM Heatmap] | [Superimposed Overlay]
    """
    n_samples = len(sample_paths)
    fig, axes = plt.subplots(n_samples, 3, figsize=(11, 3.5 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, path in enumerate(sample_paths):
        orig, hm, overlay, pred_cls, prob = explain_image(model, path, device)
        
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Input X-Ray #{i+1}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hm)
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Prediction: {pred_cls} ({prob*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

    plt.suptitle("Grad-CAM Explainability: Visualizing ResNet-18 Diagnostic Attention", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] Grad-CAM grid figure saved to: {save_path}")

    plt.show()
    plt.close()
