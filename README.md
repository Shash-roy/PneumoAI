# 🩺 Pneumonia Detection from Chest X-Rays Using Transfer Learning & Grad-CAM Explainability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15%2B-blue.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-App%20UI-orange.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![Dataset](https://img.shields.io/badge/Kaggle-Chest%20X--Ray%20Dataset-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Deep Learning project that classifies pediatric chest radiographs into **NORMAL** and **PNEUMONIA** using **ResNet-18** transfer learning, handles severe class imbalance via dynamic class-weighted loss, optimizes decision thresholds to achieve **~95% diagnostic recall**, and provides visual interpretability via **Grad-CAM heatmaps** overlaid directly on X-rays.

> ⚠️ **IMPORTANT EDUCATIONAL DISCLAIMER**  
> This project is developed strictly for **academic, research, and educational demonstration purposes**. It has **not** been certified or cleared by the FDA, CE, or any medical regulatory authority and **must never** be used as a substitute for professional clinical judgment, medical diagnosis, or radiological review.

---

## 📌 Table of Contents
1. [Project Overview & Key Objectives](#-project-overview--key-objectives)
2. [Dataset & Fixing the Kaggle Validation Split Flaw](#-dataset--fixing-the-kaggle-validation-split-flaw)
3. [Deep Learning Architecture & Methodology](#-deep-learning-architecture--methodology)
4. [Class Imbalance & Threshold Optimization](#-class-imbalance--threshold-optimization)
5. [Grad-CAM Explainability](#-grad-cam-explainability)
6. [Project Structure](#-project-structure)
7. [Quickstart: Colab, Kaggle & Local Setup](#-quickstart-colab-kaggle--local-setup)
8. [Running the CLI Pipeline](#-running-the-cli-pipeline)
9. [Interactive Gradio Web Application](#-interactive-gradio-web-application)
10. [Diagnostic Results & Comparison](#-diagnostic-results--comparison)

---

## 🔬 Project Overview & Key Objectives

Pneumonia is an inflammatory lung condition primarily caused by bacterial or viral infections, accounting for substantial childhood mortality worldwide. While chest radiography is the most standard imaging modality, human radiological review can be prone to fatigue and subtle feature ambiguity.

This repository demonstrates how to build a reliable, reproducible, and explainable Deep Learning computer vision system:
- **Transfer Learning:** Fine-tunes a pretrained **ResNet-18** CNN on ImageNet features.
- **Data Hygiene:** Rectifies the dataset's critical validation split flaw without data leakage.
- **Imbalance Mitigation:** Dynamically weights the minority class (NORMAL) in the loss function.
- **Clinical Threshold Calibration:** Optimizes the decision threshold on the validation set targeting **~95% Recall (Sensitivity)** to minimize catastrophic False Negatives.
- **Explainability:** Implements **Grad-CAM (Gradient-Weighted Class Activation Mapping)** on `model.layer4[-1]` to visualize spatial attention maps.
- **Interactive UI:** Delivers a modern **Gradio** web app where users can adjust thresholds in real time and view instant Grad-CAM overlays.

---

## 📊 Dataset & Fixing the Kaggle Validation Split Flaw

We use the standard Kaggle pediatric dataset:  
🔗 **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** (`paultimothymooney/chest-xray-pneumonia`)

### The Original Kaggle Flaw
The original Kaggle distribution contains:
* `train/`: 5,216 images (1,341 Normal, 3,875 Pneumonia)
* `val/`: **Only 16 images** (8 Normal, 8 Pneumonia) ⚠️
* `test/`: 624 images (234 Normal, 390 Pneumonia)

Evaluating hyperparameter choices or checkpoint quality on only **16 validation images** produces massive statistical variance and leads to overfitting.

### The Professional Fix
1. **Stratified 85:15 Split:** We pool the original `train/` and `val/` images (~5,232 images) and execute a stratified shuffle split:
   * **New Train Set (85%):** ~4,447 images (preserves ~3:1 ratio)
   * **New Val Set (15%):** ~785 images (statistically robust validation)
2. **Untouched Test Set:** The official `test/` folder (624 images) is kept completely untouched until final evaluation to avoid data leakage.

```
       Original Folders                  Engineered Pipeline
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ train/ (~5,216 images)      │───────>│ 85% Train Set (~4,447 imgs)  │
│ val/   (16 images - flawed!)│        │ 15% Val Set   (~785 imgs)    │
└─────────────────────────────┘        └──────────────────────────────┘
┌─────────────────────────────┐        ┌──────────────────────────────┐
│ test/  (624 images)         │───────>│ Untouched Final Test Set     │
└─────────────────────────────┘        └──────────────────────────────┘
```

---

## 🏗️ Deep Learning Architecture & Methodology

```
Input (224x224 RGB) ──> Conv1 ──> Layer1 ──> Layer2 ──> Layer3 ──> Layer4 (512-d) ──> AvgPool ──> Linear(512, 2)
                                                                         │
                                                                   Grad-CAM Hook
                                                                         ▼
                                                             Saliency Heatmap Overlay
```

### Phase 1: Baseline Feature Extractor
1. Load `models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
2. Freeze all convolutional layers (`requires_grad = False`).
3. Replace the final classification head: `model.fc = nn.Linear(512, 2)`.
4. Train only the head with `CrossEntropyLoss` and `AdamW` (learning rate `1e-3`) for 5 epochs.

### Phase 2: Full Fine-Tuning
1. Unfreeze all ResNet-18 layers (`requires_grad = True`).
2. Train with a lower learning rate (`1e-4`) to gently adapt convolutional filters without catastrophic forgetting.
3. Apply dynamic class-weighted `CrossEntropyLoss`:
   $$\text{weight}_{\text{NORMAL}} = \frac{N_{\text{Pneumonia}}}{N_{\text{Normal}}} \approx 2.89, \quad \text{weight}_{\text{PNEUMONIA}} = 1.0$$
4. Checkpoint the model whenever validation **ROC-AUC** improves, with **Early Stopping** (patience = 4).

### Preprocessing & Safe Medical Augmentation
- Resize all images to standard `224 × 224`.
- Convert 1-channel grayscale X-rays to 3-channel RGB.
- Normalize using ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Safe Augmentations (Train only):** Subtle rotation ($\pm 10^\circ$), gentle zoom/crop (0.90 to 1.00), and slight brightness/contrast variance ($\pm 10\%$).  
  *Never apply vertical flips, because chest anatomy is strictly upright.*

---

## ⚖️ Class Imbalance & Threshold Optimization

### Why 0.50 is the Wrong Threshold in Healthcare
Standard classifiers output probabilities and default to a `0.50` decision threshold. However, clinical diagnostics has asymmetric cost penalties:
* **False Negative (FN):** Sending a sick patient home undiagnosed can be fatal.
* **False Positive (FP):** Scheduling a healthy patient for sputum culture or follow-up is an inconvenience.

### Validation-Tuned Threshold
We sweep 500 thresholds between `0.01` and `0.99` strictly on the **Validation Set** to find the decision cutoff that achieves **$\ge 95\%$ Recall (Sensitivity)** while maximizing F1-score and Precision.

```
Operating Point Comparison on Untouched Test Set (624 Images):
-----------------------------------------------------------------------------
Metric                   | Default Thresh (0.50)  | Optimal Thresh (~0.40)
-----------------------------------------------------------------------------
Recall (Sensitivity)     | ~92.5%                 | ~96.8% (Target >= 95% met!)
Missed Pneumonia (FN)    | 29 cases               | 12 cases (Reduced by >58%!)
Specificity              | ~86.2%                 | ~81.4%
ROC-AUC                  | ~0.965                 | ~0.965
-----------------------------------------------------------------------------
```

---

## 🔍 Grad-CAM Explainability

Neural networks can inadvertently learn background shortcuts (e.g., patient position labels `L` / `R`, hospital hardware borders). Grad-CAM computes gradients with respect to the feature maps of the final convolutional layer:
$$A^k = \text{Feature map } k \text{ of } \texttt{layer4[-1]}$$
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

The resulting 2D activation map is resized and superimposed onto the original radiograph using an OpenCV `JET` colormap, highlighting exactly whether opacity in the lower or middle pulmonary lobes drove the decision.

---

## 📁 Project Structure

```
PneumoniaML/
│
├── .gitignore                      # Safe ignore rules (credentials, weights, datasets)
├── requirements.txt                # Production dependencies
├── README.md                       # Comprehensive guide & documentation
├── app.py                          # Interactive Gradio web application
│
├── notebook/
│   └── pneumonia_detection_colab_kaggle.ipynb # 1-click end-to-end GPU notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration & hyperparameters
│   ├── dataset.py                  # Auto-detection, stratified 85:15 split, transforms
│   ├── eda.py                      # Exploratory data analysis & imbalance charts
│   ├── model.py                    # ResNet-18 builder & freezing/unfreezing logic
│   ├── train.py                    # Baseline & fine-tuning training loops
│   ├── evaluate.py                 # Clinical metrics, threshold tuning, ROC/PR curves
│   ├── explainability.py           # Native Grad-CAM hook engine & visual overlays
│   └── utils.py                    # Seeds, device logger, checkpointing, loss plots
│
├── scripts/
│   ├── download_dataset.py         # Secure Kaggle download assistant
│   ├── run_eda.py                  # CLI script for EDA
│   ├── run_train.py                # CLI script for training baseline + fine-tuning
│   └── run_evaluate.py             # CLI script for test evaluation & Grad-CAM grids
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py            # Unit test suite verifying logic and configs
│
├── checkpoints/                    # Saved model weights (.pth)
└── outputs/                        # Output plots, confusion matrices, ROC curves
```

---

## 🚀 Quickstart: Colab, Kaggle & Local Setup

### Option 1: Google Colab (Recommended for Free GPU)
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Upload `notebook/pneumonia_detection_colab_kaggle.ipynb` or clone the repository:
   ```bash
   !git clone https://github.com/<your-username>/PneumoniaML.git
   %cd PneumoniaML
   !pip install -r requirements.txt
   ```
4. Authenticate Kaggle securely via Colab Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or run the notebook cells sequentially!

### Option 2: Kaggle Notebooks
1. Go to Kaggle: [kaggle.com/code](https://www.kaggle.com/code) -> Create New Notebook.
2. In the right panel, set **Accelerator** to **GPU P100** or **GPU T4 x2**.
3. Click **+ Add Input** -> Search for `paultimothymooney/chest-xray-pneumonia` -> Click Add.
4. Upload `notebook/pneumonia_detection_colab_kaggle.ipynb` and run all cells! The code auto-detects `/kaggle/input` seamlessly.

### Option 3: Local Machine
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/PneumoniaML.git
cd PneumoniaML

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# or: venv\Scripts\activate    # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Acquire dataset
python scripts/download_dataset.py
# (Or manually extract chest_xray into PneumoniaML/data/chest_xray/)
```

---

## 💻 Running the CLI Pipeline

Once the dataset is in place:

### 1. Run Exploratory Data Analysis
Generates class distribution graphs and image resolution metrics:
```bash
python scripts/run_eda.py
```

### 2. Train Baseline and Fine-Tuned ResNet-18
Performs the stratified 85:15 split, trains the baseline head, fine-tunes all layers with class weights, and saves the best model:
```bash
python scripts/run_train.py
```

### 3. Evaluate on Untouched Test Set & Generate Grad-CAM
Calibrates the optimal threshold on validation and evaluates on the untouched test set:
```bash
python scripts/run_evaluate.py
```

### 4. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 🖥️ Interactive Gradio Web Application

Launch the browser-based graphical diagnosis assistant:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser:
* **Upload:** Drag-and-drop any frontal chest radiograph (JPEG/PNG).
* **Threshold Slider:** Dynamically shift the diagnostic sensitivity threshold.
* **Opacity Slider:** Adjust the Grad-CAM blending ratio over the original X-ray.
* **Instant Feedback:** Displays probabilities, diagnostic verdict, and high-resolution Grad-CAM saliency maps.

---

## 📈 Diagnostic Metrics Summary

| Evaluation Metric | Baseline Model (Frozen) | Fine-Tuned Model (Default 0.50) | Fine-Tuned Model (Optimal ~0.40) |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | ~0.942 | **~0.985** | **~0.985** |
| **Test ROC-AUC** | ~0.920 | **~0.965** | **~0.965** |
| **Test Recall (Sensitivity)** | ~88.5% | ~92.5% | **~96.8%** |
| **Test Specificity** | ~82.0% | **~86.2%** | ~81.4% |
| **Test F1-Score** | ~0.875 | **~0.912** | ~0.908 |
| **Missed Pneumonia Cases (FN)** | 45 | 29 | **12 (Critical Drop)** |

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
