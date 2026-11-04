"""
Interactive Gradio Web Application for Pneumonia Detection & Explainability.
=============================================================================
Allows clinicians, researchers, and students to:
1. Upload any Chest X-Ray image (PNG, JPEG, DICOM-converted).
2. Adjust the classification decision threshold dynamically.
3. View instant prediction probabilities (NORMAL vs. PNEUMONIA).
4. Inspect the Grad-CAM saliency heatmap and overlay to see which lung
   regions influenced the network's prediction.
5. Review medical disclaimer and educational guidance.
"""

from pathlib import Path
from typing import Dict, Tuple

try:
    import gradio as gr  # type: ignore
except ImportError:
    gr = None

try:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from torchvision import transforms  # type: ignore
except ImportError:
    transforms = None

from src.config import config
from src.explainability import explain_image
from src.model import build_resnet18


# ------------------------------------------------------------------------------
# Model Initialization & Loading Helper
# ------------------------------------------------------------------------------
device = torch.device(config.DEVICE)
_model_instance = None


def load_inference_model() -> torch.nn.Module:
    """
    Loads the best fine-tuned ResNet-18 model checkpoint.
    If the checkpoint hasn't been trained yet, initializes with ImageNet pretrained weights.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    print("[APP] Loading ResNet-18 model for inference...")
    model = build_resnet18(pretrained=True, num_classes=config.NUM_CLASSES, freeze_backbone=False)
    
    checkpoint_candidates = [
        config.BEST_MODEL_PATH,
        config.BASELINE_MODEL_PATH,
        Path("best_finetuned_resnet18.pth"),
        Path("checkpoints/best_finetuned_resnet18.pth")
    ]
    
    loaded = False
    for ckpt in checkpoint_candidates:
        if ckpt.exists():
            print(f"[APP] Loading weights from checkpoint: {ckpt}")
            state = torch.load(str(ckpt), map_location=device)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            loaded = True
            break
            
    if not loaded:
        print("[APP] Notice: No custom trained checkpoint found. Using ImageNet pretrained ResNet-18.")

    model.to(device)
    model.eval()
    _model_instance = model
    return _model_instance


# ------------------------------------------------------------------------------
# Prediction & Explanation Pipeline
# ------------------------------------------------------------------------------
def predict_and_explain(
    input_image: Image.Image,
    threshold: float = 0.50,
    overlay_opacity: float = 0.45
) -> Tuple[Dict[str, float], Image.Image, Image.Image, str]:
    """
    Processes an uploaded X-ray image and returns diagnosis, probabilities, and Grad-CAM.
    
    Args:
        input_image: PIL Image uploaded via Gradio.
        threshold: Decision threshold for flagging Pneumonia (default: 0.50).
        overlay_opacity: Opacity alpha for Grad-CAM blending.
        
    Returns:
        (confidence_dict, heatmap_img, overlay_img, summary_text)
    """
    if input_image is None:
        return {}, None, None, "Please upload a chest X-ray image to begin diagnosis."

    model = load_inference_model()

    # Generate Grad-CAM arrays
    orig_rgb, colored_heatmap, overlay, top_class, top_prob = explain_image(
        model=model,
        image_path_or_pil=input_image,
        device=device,
        alpha=overlay_opacity
    )

    # Re-compute exact softmax probabilities for both classes
    if transforms is None:
        return {}, None, None, "PyTorch / torchvision is not installed in the active environment."
    preprocess = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    pil_rgb = Image.fromarray(orig_rgb)
    tensor = preprocess(pil_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prob_normal = float(probs[0])
    prob_pneumonia = float(probs[1])

    # Apply decision threshold to classify as PNEUMONIA
    is_pneumonia = prob_pneumonia >= threshold
    diagnosis = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL (NO PNEUMONIA DETECTED)"
    status_color = "#d9534f" if is_pneumonia else "#5cb85c"

    confidences = {
        "NORMAL": prob_normal,
        "PNEUMONIA": prob_pneumonia
    }

    summary_html = f"""
    <div style='padding: 15px; border-radius: 8px; background-color: #f8f9fa; border: 1px solid #dee2e6;'>
        <h3 style='color: {status_color}; margin-top: 0;'>Diagnostic Result: <b>{diagnosis}</b></h3>
        <p><b>Pneumonia Probability:</b> {prob_pneumonia * 100:.2f}%</p>
        <p><b>Normal Probability:</b> {prob_normal * 100:.2f}%</p>
        <p><b>Current Operating Threshold:</b> {threshold:.2f} <i>(Pneumonia flagged when probability ≥ threshold)</i></p>
        <p style='font-size: 0.9em; color: #6c757d;'>
            Grad-CAM heatmap highlights the key anatomical regions of the lung field that
            heavily contributed to the neural network's activation score.
        </p>
    </div>
    """

    heatmap_pil = Image.fromarray(colored_heatmap)
    overlay_pil = Image.fromarray(overlay)

    return confidences, heatmap_pil, overlay_pil, summary_html


# ------------------------------------------------------------------------------
# Gradio UI Layout
# ------------------------------------------------------------------------------
def create_app() -> "gr.Blocks":
    """Builds the complete Gradio interface with responsive components."""
    if gr is None:
        raise ImportError(
            "\n" + "=" * 70 + "\n"
            "[ERROR] Gradio is not installed in the active Python environment!\n"
            "To install all required dependencies, run:\n"
            "    pip install -r requirements.txt\n"
            "=" * 70
        )
    
    custom_css = """
    .gradio-container { font-family: 'Helvetica Neue', Arial, sans-serif; }
    .header-banner { text-align: center; margin-bottom: 20px; }
    .disclaimer-box { background: #fff3cd; border-left: 5px solid #ffeeba; padding: 12px; margin-top: 15px; border-radius: 4px; }
    """

    with gr.Blocks(title="Pneumonia Detection & Explainability", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🩺 Pneumonia Detection from Chest X-Rays
            ### Transfer Learning (ResNet-18) + Grad-CAM Visual Explainability
            Upload a pediatric chest radiograph to evaluate the likelihood of pneumonia and visualize which anatomical lung regions informed the model's prediction.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Chest X-Ray (AP/PA View)")
                
                with gr.Accordion("⚙️ Diagnostic Sensitivity Controls", open=True):
                    threshold_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.45,
                        step=0.01,
                        label="Decision Threshold for Pneumonia",
                        info="Lower threshold increases Recall (Sensitivity) to minimize false negatives."
                    )
                    opacity_slider = gr.Slider(
                        minimum=0.20,
                        maximum=0.80,
                        value=0.45,
                        step=0.05,
                        label="Heatmap Overlay Opacity",
                        info="Adjust blending between original X-ray and colored Grad-CAM."
                    )

                submit_btn = gr.Button("🔍 Analyze Chest X-Ray", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_html = gr.HTML(label="Diagnostic Assessment")
                label_output = gr.Label(num_top_classes=2, label="Classification Probabilities")
                
                with gr.Row():
                    cam_output = gr.Image(type="pil", label="Grad-CAM Saliency Map")
                    overlay_output = gr.Image(type="pil", label="Superimposed Diagnostic Overlay")

        submit_btn.click(
            fn=predict_and_explain,
            inputs=[image_input, threshold_slider, opacity_slider],
            outputs=[label_output, cam_output, overlay_output, result_html]
        )

        gr.Markdown(
            """
            <div class="disclaimer-box">
                <b>⚠️ Educational Disclaimer:</b> This application is built strictly for academic, research, and educational demonstration purposes. It has not been approved by the FDA or any medical authority and <b>must not</b> be utilized as a substitute for professional clinical judgment, medical diagnosis, or radiological interpretation.
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)

