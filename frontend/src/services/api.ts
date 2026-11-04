/**
 * API Service for communicating with the FastAPI backend.
 */

import { HealthStatus, PredictionResult, GradCAMResult, ModelPerformanceData } from '../types';

// Use environment variable if configured, otherwise fallback to local backend port 8000
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  /**
   * Checks backend server and model checkpoint availability.
   */
  async getHealth(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed with status: ${response.status}`);
      }
      return await response.json();
    } catch (err: any) {
      console.warn('Backend connection error:', err.message);
      return {
        status: 'disconnected',
        model_name: 'ResNet-18',
        checkpoint_available: false,
        checkpoint_path: null,
        is_custom_checkpoint: false,
        device: 'unknown',
        default_threshold: 0.40,
        classes: ['NORMAL', 'PNEUMONIA'],
        message: 'Cannot connect to FastAPI backend at ' + API_BASE_URL + '. Please ensure backend is running.'
      };
    }
  }

  /**
   * Uploads an X-ray image for inference.
   */
  async predict(file: File, threshold?: number): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (threshold !== undefined) {
      formData.append('threshold', threshold.toString());
    }

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Prediction failed with status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Generates Grad-CAM visual heatmap for an X-ray.
   */
  async explain(file: File, targetClass?: number, alpha: number = 0.45): Promise<GradCAMResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (targetClass !== undefined) {
      formData.append('target_class', targetClass.toString());
    }
    formData.append('alpha', alpha.toString());

    const response = await fetch(`${API_BASE_URL}/explain`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Grad-CAM explanation failed with status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Retrieves model performance metrics.
   */
  async getPerformance(): Promise<ModelPerformanceData> {
    const response = await fetch(`${API_BASE_URL}/api/performance`);
    if (!response.ok) {
      throw new Error(`Performance data fetch failed with status: ${response.status}`);
    }
    return await response.json();
  }
}

export const api = new ApiService();

