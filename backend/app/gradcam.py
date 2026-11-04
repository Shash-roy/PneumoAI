"""
Grad-CAM Explainability Service for PneumoAI Backend.
=====================================================
Calculates gradient-weighted class activation maps targeting ResNet-18 layer4[-1].
Encodes output images to base64 Data URLs for seamless transmission to the React frontend.
"""

import base64
import io
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from backend.app.model_loader import model_manager

# Normalization pipeline matching training
norm_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def pil_to_base64_data_uri(pil_img: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image to a base64 Data URI."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, quality=92)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def numpy_to_base64_data_uri(np_arr: np.ndarray, format: str = "JPEG") -> str:
    """Converts an RGB numpy uint8 array to a base64 Data URI."""
    pil_img = Image.fromarray(np_arr)
    return pil_to_base64_data_uri(pil_img, format=format)


class GradCAMService:
    """
    Hook-based Grad-CAM computation on ResNet-18.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = self.model.layer4[-1]
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def f_hook(module, input, output):
            self.activations = output.detach()

        def b_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(f_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(b_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Executes forward + backward pass to generate a 2D normalized activation map.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3))[0]

        # Weight the activation maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]

        cam = torch.relu(cam).cpu().numpy()

        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def generate_gradcam_explanation(
    pil_image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.45
) -> Dict:
    """
    High-level generator returning original, heatmap, and overlay images as base64 strings.
    """
    model, device = model_manager.get_model()

    # Standardize image to 224x224 RGB
    orig_resized = pil_image.resize((224, 224))
    orig_np = np.array(orig_resized)

    input_tensor = norm_pipeline(orig_resized).unsqueeze(0).to(device)

    # Determine predicted class if target class not explicitly selected
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_class_idx].item())

    effective_target_idx = target_class_idx if target_class_idx is not None else top_class_idx
    target_name = "PNEUMONIA" if effective_target_idx == 1 else "NORMAL"
    predicted_name = "PNEUMONIA" if top_class_idx == 1 else "NORMAL"

    # Compute CAM
    cam_engine = GradCAMService(model)
    cam_heatmap_2d = cam_engine.compute_cam(input_tensor, effective_target_idx)
    cam_engine.remove_hooks()

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(cam_heatmap_2d, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colored heatmap using JET colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.float32(colored_heatmap) * alpha + np.float32(orig_np) * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    explanation_text = (
        f"Grad-CAM visualizes the neural network's gradient activations across the final convolutional layer "
        f"(ResNet-18 layer4[-1]). The red and warm-colored highlights identify the localized pulmonary regions "
        f"that contributed most heavily to the classification of '{target_name}'."
    )

    return {
        "predicted_class": predicted_name,
        "confidence": round(confidence, 4),
        "target_class_explained": target_name,
        "target_layer": "model.layer4[-1] (512 feature channels)",
        "original_image_base64": numpy_to_base64_data_uri(orig_np),
        "heatmap_base64": numpy_to_base64_data_uri(colored_heatmap),
        "overlay_base64": numpy_to_base64_data_uri(overlay),
        "explanation_text": explanation_text,
        "disclaimer": (
            "Grad-CAM indicates regions of neural network attention, but does NOT prove clinical correctness. "
            "It must not be used as an independent diagnostic test."
        )
    }
