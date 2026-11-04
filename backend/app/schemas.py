"""
Pydantic Schemas for Request & Response Data Validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_name: str = Field(..., example="ResNet-18 (Transfer Learning)")
    checkpoint_available: bool = Field(..., example=True)
    checkpoint_path: Optional[str] = Field(None, example="checkpoints/best_finetuned_resnet18.pth")
    is_custom_checkpoint: bool = Field(..., example=True)
    device: str = Field(..., example="cuda")
    default_threshold: float = Field(0.40, example=0.40)
    classes: List[str] = Field(default=["NORMAL", "PNEUMONIA"])
    message: str = Field(..., example="Model is active and ready for inference.")


class PredictResponse(BaseModel):
    success: bool = True
    predicted_class: str = Field(..., example="PNEUMONIA")
    normal_probability: float = Field(..., example=0.042)
    pneumonia_probability: float = Field(..., example=0.958)
    threshold_used: float = Field(..., example=0.40)
    is_pneumonia: bool = Field(..., example=True)
    model_name: str = Field("ResNet-18")
    inference_time_ms: float = Field(..., example=42.5)
    checkpoint_status: str = Field(..., example="Fine-Tuned Checkpoint")
    disclaimer: str = Field(
        "PneumoAI is an educational machine learning project. It is not a medical device and must not be used for diagnosis, treatment, or medical decision-making."
    )


class ExplainResponse(BaseModel):
    success: bool = True
    predicted_class: str = Field(..., example="PNEUMONIA")
    confidence: float = Field(..., example=0.958)
    target_class_explained: str = Field(..., example="PNEUMONIA")
    target_layer: str = Field("layer4[-1]")
    original_image_base64: str
    heatmap_base64: str
    overlay_base64: str
    explanation_text: str
    disclaimer: str = Field(
        "Grad-CAM visualizes spatial gradients in convolutional layer4. It indicates regions of neural network activation, but does NOT prove clinical correctness."
    )


class ModelPerformanceStats(BaseModel):
    is_evaluated: bool
    evaluation_status_message: str
    baseline_metrics: Dict[str, float]
    finetuned_default_metrics: Dict[str, float]
    finetuned_optimal_metrics: Dict[str, float]
    optimal_threshold: float
    target_recall: float
    test_set_size: int
    confusion_matrix: Optional[Dict[str, int]] = None
