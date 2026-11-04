"""
Inference & Image Preprocessing Pipeline for PneumoAI Backend.
==============================================================
Ensures exact fidelity to the training preprocessing pipeline:
- RGB conversion (3-channel)
- Resizing to 224x224
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Validation-selected thresholding for sensitive pneumonia screening.
"""

import io
import time
from typing import Dict, Tuple

try:
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from torchvision import transforms  # type: ignore
except ImportError:
    torch = None
    Image = None
    transforms = None

from backend.app.model_loader import model_manager

# Exact training preprocessing pipeline
preprocess_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def validate_image_bytes(image_bytes: bytes, max_size_mb: int = 15) -> Image.Image:
    """
    Validates file size, integrity, and image formats (JPEG, PNG).
    Raises ValueError on invalid formats or corruption.
    """
    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
        
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb} MB.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verify integrity without decoding fully
        # Re-open after verify()
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Uploaded file is not a valid or readable image: {str(e)}")

    if pil_img.format not in ["JPEG", "PNG", "MPO", "WEBP"]:
        raise ValueError(f"Unsupported image format: {pil_img.format}. Please upload a JPEG or PNG chest radiograph.")

    # Convert to 3-channel RGB (critical for grayscale X-rays)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    return pil_img


def run_prediction(
    pil_image: Image.Image,
    custom_threshold: float = None
) -> Dict:
    """
    Runs forward pass through ResNet-18 and calculates class probabilities.
    
    Args:
        pil_image: PIL RGB image.
        custom_threshold: Optional decision threshold override.
        
    Returns:
        Dictionary with predicted class, probabilities, threshold, inference timing.
    """
    model, device = model_manager.get_model()
    threshold = custom_threshold if custom_threshold is not None else model_manager.optimal_threshold

    # Preprocess
    input_tensor = preprocess_pipeline(pil_image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    inference_time = (time.perf_counter() - t0) * 1000  # ms

    prob_normal = float(probs[0].item())
    prob_pneumonia = float(probs[1].item())

    # Apply decision threshold to classify as PNEUMONIA
    is_pneumonia = prob_pneumonia >= threshold
    predicted_class = "PNEUMONIA" if is_pneumonia else "NORMAL"

    checkpoint_status = (
        "Trained Fine-Tuned Checkpoint"
        if model_manager.is_custom_checkpoint
        else "Pretrained ResNet-18 Base (Checkpoint Pending)"
    )

    return {
        "predicted_class": predicted_class,
        "normal_probability": round(prob_normal, 4),
        "pneumonia_probability": round(prob_pneumonia, 4),
        "threshold_used": round(threshold, 2),
        "is_pneumonia": is_pneumonia,
        "inference_time_ms": round(inference_time, 2),
        "checkpoint_status": checkpoint_status,
        "model_name": "ResNet-18 (Transfer Learning)",
        "disclaimer": (
            "PneumoAI is an educational machine learning project. It is not a medical device "
            "and must not be used for diagnosis, treatment, or medical decision-making. "
            "Always consult a qualified healthcare professional for medical concerns."
        )
    }

