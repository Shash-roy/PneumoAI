# PneumoAI — AI-Powered Chest X-Ray Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15%2B-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**PneumoAI** is a full-stack deep learning healthcare platform that detects pediatric pneumonia from frontal chest radiographs with visual clinical interpretability. It integrates a **ResNet-18** transfer learning engine, dynamic class-weighted loss, sensitivity threshold calibration (~95% Recall), a high-performance **FastAPI** backend, and a modern **React + Vite + TypeScript** dashboard with interactive **Grad-CAM heatmaps**.

> ⚠️ **IMPORTANT EDUCATIONAL & CLINICAL DISCLAIMER**  
> **PneumoAI is an educational machine learning project. It is not a medical device and must not be used for diagnosis, treatment, or medical decision-making. Always consult a qualified healthcare professional for medical concerns.**

---

## 📌 Table of Contents
1. [Platform Features & Architecture](#-platform-features--architecture)
2. [Full-Stack Architecture Overview](#-full-stack-architecture-overview)
3. [Website Pages & Capabilities](#-website-pages--capabilities)
4. [Deep Learning Model & Preprocessing](#-deep-learning-model--preprocessing)
5. [Class Imbalance & Threshold Sensitivity (~95% Recall)](#-class-imbalance--threshold-sensitivity-95-recall)
6. [Grad-CAM Visual Explainability](#-grad-cam-visual-explainability)
7. [Local Development Commands (Windows 11)](#-local-development-commands-windows-11)
8. [Backend API Reference](#-backend-api-reference)
9. [Deployment Guide (Vercel & Hugging Face Spaces)](#-deployment-guide-vercel--hugging-face-spaces)
10. [Diagnostic Benchmarks & Metrics](#-diagnostic-benchmarks--metrics)

---

## 🔬 Platform Features & Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PneumoAI Full-Stack Architecture                      │
└─────────────────────────────────────────────────────────────────────────────┘

       CLIENT BROWSER                                FASTAPI BACKEND
┌───────────────────────────┐                 ┌─────────────────────────────┐
│  React 18 + TypeScript    │                 │   FastAPI (Python 3.10+)    │
│  Tailwind CSS UI          │──[POST /predict]──>│                             │
│  Lucide Icons + Recharts  │<─[JSON Probs]───│   ModelManager (Singleton) │
│                           │                 │   ResNet-18 (512-dim head)  │
│  Drag-and-Drop Radiograph │                 │   ImageNet Preprocessing    │
│  Sensitivity Slider       │──[POST /explain]──>│   Grad-CAM Engine (layer4)  │
│  Grad-CAM Saliency Maps   │<─[Base64 Heatmap]─│   Dynamic Loss Weighting    │
└───────────────────────────┘                 └─────────────────────────────┘
```

- **Frontend:** Responsive React SPA built with Vite, TypeScript, and Tailwind CSS.
- **Backend:** Async FastAPI service providing singleton model inference, image validation, and Grad-CAM generation.
- **Machine Learning Core:** PyTorch ResNet-18 fine-tuned on Guangzhou Women and Children's Medical Center chest radiographs.
- **Explainability Engine:** Native gradient hook implementation extracting spatial activation maps from `model.layer4[-1]`.

---

## 🖥️ Website Pages & Capabilities

| Page / Route | Purpose & Key Features |
| :--- | :--- |
| **1. Home / Dashboard** | Hero introduction, quick statistics, pipeline walkthrough, technology stack, and real-time backend connectivity badge. |
| **2. X-Ray Analysis** | Drag-and-drop file uploader, file size/type validation, image preview, sensitivity threshold slider, and loading spinner. |
| **3. Prediction Results** | Classification verdict badge (`PNEUMONIA` or `NORMAL`), probability progress bars, inference latency counter, and operating threshold. |
| **4. Grad-CAM Explainability** | Side-by-side or superimposed views of original X-ray, colormap heatmap, and blended overlay showing localized lung regions of interest. |
| **5. Model Information** | Deep dive into pediatric pneumonia pathology, Kaggle dataset design flaw & stratified 85:15 fix, ResNet-18 residual skip connections, and loss weighting. |
| **6. Model Performance** | Benchmark progression table (Baseline vs. Fine-Tuned vs. Optimal Cutoff), interactive 4-quadrant Confusion Matrix, and ROC curve chart. |

---

## 🚀 Local Development Commands (Windows 11)

### 1. Prerequisites
- **Node.js:** v18+ or v20+ (with npm)
- **Python:** 3.10+ (with pip)

### 2. Backend Setup & Startup
Open a terminal in the project root:
```powershell
# Navigate to project root
cd PneumoniaML

# Optional: activate virtual environment
# python -m venv venv
# .\venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Start FastAPI backend server (Port 8000)
uvicorn backend.app.main:app --reload --port 8000
```
- **Backend URL:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/health`

### 3. Frontend Setup & Startup
Open a second terminal:
```powershell
# Navigate to frontend directory
cd PneumoniaML\frontend

# Install Node dependencies
npm.cmd install

# Start Vite development server
npm.cmd run dev
```
- **Frontend URL:** `http://localhost:5173`

---

## 📡 Backend API Reference

### 1. `GET /health`
Returns system status, device information, and model checkpoint state:
```json
{
  "status": "healthy",
  "model_name": "ResNet-18 (Transfer Learning)",
  "checkpoint_available": true,
  "checkpoint_path": "checkpoints/best_finetuned_resnet18.pth",
  "is_custom_checkpoint": true,
  "device": "cuda",
  "default_threshold": 0.40,
  "classes": ["NORMAL", "PNEUMONIA"],
  "message": "Active: Custom fine-tuned checkpoint loaded"
}
```

### 2. `POST /predict`
Form-data upload with optional threshold:
- `file`: UploadFile (JPEG, PNG, max 15MB)
- `threshold`: Optional float (`0.10` to `0.90`, default `0.40`)

**Response:**
```json
{
  "success": true,
  "predicted_class": "PNEUMONIA",
  "normal_probability": 0.038,
  "pneumonia_probability": 0.962,
  "threshold_used": 0.40,
  "is_pneumonia": true,
  "model_name": "ResNet-18 (Transfer Learning)",
  "inference_time_ms": 38.4,
  "checkpoint_status": "Fine-Tuned Checkpoint"
}
```

### 3. `POST /explain`
Form-data upload:
- `file`: UploadFile (JPEG, PNG)
- `target_class`: Optional integer (`0` for NORMAL, `1` for PNEUMONIA)
- `alpha`: Optional float overlay opacity (`0.45`)

**Response:**
```json
{
  "success": true,
  "predicted_class": "PNEUMONIA",
  "confidence": 0.962,
  "target_class_explained": "PNEUMONIA",
  "target_layer": "model.layer4[-1] (512 feature channels)",
  "original_image_base64": "data:image/jpeg;base64,...",
  "heatmap_base64": "data:image/jpeg;base64,...",
  "overlay_base64": "data:image/jpeg;base64,...",
  "explanation_text": "Grad-CAM visualizes the neural network's gradient activations..."
}
```

---

## ☁️ Deployment Guide (Vercel & Hugging Face Spaces)

### Frontend Deployment (Vercel)
1. Push your repository to GitHub.
2. Link the repository on [Vercel](https://vercel.com/):
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
3. Under **Environment Variables**, set:
   ```env
   VITE_API_BASE_URL=https://<your-backend-domain>.hf.space
   ```
*(The included `frontend/vercel.json` automatically manages single-page client routing).*

### Backend Deployment (Hugging Face Spaces)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces):
   - **SDK:** Docker
   - **Hardware:** CPU Basic (Free) or T4 GPU
2. The project contains [`backend/Dockerfile`](file:///c:/Users/Shashwat/Downloads/PneumoniaML/backend/Dockerfile). Point the space to use this Dockerfile.
3. Hugging Face Spaces serves on port `7860` automatically.

---

## 📈 Diagnostic Benchmarks & Metrics

Evaluated on the untouched 624-image Kaggle test set:

| Evaluation Metric | Baseline Model (Frozen) | Fine-Tuned (Default 0.50 Cutoff) | Fine-Tuned (Optimal ~0.40 Cutoff) ★ |
| :--- | :---: | :---: | :---: |
| **Diagnostic Sensitivity (Recall)** | 88.5% | 92.5% | **96.8% (Target &ge; 95% met)** |
| **Diagnostic Specificity** | 82.0% | **86.2%** | 81.4% |
| **Test ROC-AUC** | 0.920 | **0.965** | **0.965** |
| **Test F1-Score** | 0.875 | **0.912** | 0.908 |
| **Missed Pneumonia Cases (FN)** | 45 cases | 29 cases | **12 cases (Reduced by 58.6%)** |

---

## 📜 License & Disclaimers
This project is open-source software under the [MIT License](LICENSE).  
**Remember:** PneumoAI is for educational demonstration only and is not certified for clinical patient diagnosis.
