/**
 * TypeScript Data Models and Interfaces for PneumoAI.
 */

export interface HealthStatus {
  status: string;
  model_name: string;
  checkpoint_available: boolean;
  checkpoint_path: string | null;
  is_custom_checkpoint: boolean;
  device: string;
  default_threshold: number;
  classes: string[];
  message: string;
}

export interface PredictionResult {
  success: boolean;
  predicted_class: 'NORMAL' | 'PNEUMONIA';
  normal_probability: number;
  pneumonia_probability: number;
  threshold_used: number;
  is_pneumonia: boolean;
  model_name: string;
  inference_time_ms: number;
  checkpoint_status: string;
  disclaimer: string;
}

export interface GradCAMResult {
  success: boolean;
  predicted_class: string;
  confidence: number;
  target_class_explained: string;
  target_layer: string;
  original_image_base64: string;
  heatmap_base64: string;
  overlay_base64: string;
  explanation_text: string;
  disclaimer: string;
}

export interface ModelPerformanceData {
  is_evaluated: boolean;
  evaluation_status_message: string;
  baseline_metrics: {
    accuracy: number;
    recall: number;
    specificity: number;
    precision: number;
    f1_score: number;
    roc_auc: number;
    false_negatives: number;
    false_positives: number;
  };
  finetuned_default_metrics: {
    accuracy: number;
    recall: number;
    specificity: number;
    precision: number;
    f1_score: number;
    roc_auc: number;
    false_negatives: number;
    false_positives: number;
  };
  finetuned_optimal_metrics: {
    accuracy: number;
    recall: number;
    specificity: number;
    precision: number;
    f1_score: number;
    roc_auc: number;
    false_negatives: number;
    false_positives: number;
  };
  optimal_threshold: number;
  target_recall: number;
  test_set_size: number;
  confusion_matrix?: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
}

export type PageId = 'home' | 'analyze' | 'model-info' | 'performance';
