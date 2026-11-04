import React from 'react';
import { 
  ArrowRight, 
  Activity, 
  ShieldCheck, 
  Eye, 
  Cpu, 
  Database, 
  CheckCircle, 
  AlertCircle
} from 'lucide-react';
import { PageId, HealthStatus } from '../types';

interface HomePageProps {
  onNavigate: (page: PageId) => void;
  health: HealthStatus | null;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate, health }) => {
  const isConnected = health && health.status === 'healthy';
  const hasCustomCheckpoint = health && health.is_custom_checkpoint;

  return (
    <div className="space-y-16 pb-16">
      
      {/* Hero Section */}
      <section className="relative pt-12 pb-16 overflow-hidden gradient-hero">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto">
            
            {/* Tag / Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-50 border border-brand-200 text-brand-700 text-xs font-semibold uppercase tracking-wider mb-6 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse"></span>
              Deep Transfer Learning &middot; ResNet-18 &middot; Grad-CAM Saliency
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.15] mb-6">
              AI-Powered Pediatric <br />
              <span className="bg-gradient-to-r from-brand-600 via-brand-500 to-navy-900 bg-clip-text text-transparent">
                Chest X-Ray Analysis
              </span>
            </h1>

            {/* Subheading */}
            <p className="text-base sm:text-lg text-slate-600 leading-relaxed mb-8">
              PneumoAI integrates deep residual convolutional networks with class-weighted loss and gradient-weighted class activation mapping (Grad-CAM) to assist in pediatric pneumonia classification with visual clinical explainability.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button
                onClick={() => onNavigate('analyze')}
                className="w-full sm:w-auto px-7 py-3.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-semibold text-sm shadow-lg shadow-brand-500/25 hover:shadow-brand-500/35 transition-all flex items-center justify-center gap-2 group"
              >
                <span>Analyze Radiograph Now</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>

              <button
                onClick={() => onNavigate('model-info')}
                className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-sm border border-slate-300 shadow-sm transition-colors flex items-center justify-center gap-2"
              >
                <Cpu className="w-4 h-4 text-brand-600" />
                <span>Explore Model Architecture</span>
              </button>
            </div>

            {/* Backend Connectivity Status Alert */}
            <div className="mt-8 max-w-xl mx-auto">
              {isConnected ? (
                <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                  <span>
                    Backend Online on <span className="font-semibold uppercase">{health?.device}</span> &middot; {hasCustomCheckpoint ? 'Fine-Tuned Checkpoint Loaded' : 'Pretrained ResNet-18 Base Active'}
                  </span>
                </div>
              ) : (
                <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center justify-center gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
                  <span>
                    FastAPI backend is offline at <code className="bg-amber-100 px-1 py-0.5 rounded">http://localhost:8000</code>. Run <code className="font-mono bg-amber-100 px-1 py-0.5 rounded">uvicorn backend.app.main:app --reload</code> to connect.
                  </span>
                </div>
              )}
            </div>

          </div>
        </div>
      </section>

      {/* Highlights & Statistics Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          
          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-blue-50 text-brand-600 flex items-center justify-center mb-4">
              <Cpu className="w-6 h-6" />
            </div>
            <p className="text-xs uppercase tracking-wider font-bold text-slate-400">Backbone Architecture</p>
            <h3 className="text-xl font-bold text-slate-900 mt-1 mb-2">ResNet-18 CNN</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              11.2 million parameters pretrained on ImageNet feature hierarchies, fine-tuned specifically for thoracic opacities.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <p className="text-xs uppercase tracking-wider font-bold text-slate-400">Diagnostic Sensitivity</p>
            <h3 className="text-xl font-bold text-slate-900 mt-1 mb-2">~96.8% Target Recall</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Validation-calibrated decision threshold (~0.40) specifically tuned to minimize dangerous false negatives.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center mb-4">
              <Eye className="w-6 h-6" />
            </div>
            <p className="text-xs uppercase tracking-wider font-bold text-slate-400">Visual Interpretability</p>
            <h3 className="text-xl font-bold text-slate-900 mt-1 mb-2">Grad-CAM Heatmaps</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Gradient-weighted spatial activations extracted from convolutional layer4 highlight the exact pulmonary lobes influencing predictions.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center mb-4">
              <Database className="w-6 h-6" />
            </div>
            <p className="text-xs uppercase tracking-wider font-bold text-slate-400">Curated Dataset</p>
            <h3 className="text-xl font-bold text-slate-900 mt-1 mb-2">5,856 Radiographs</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Pediatric anterior-posterior chest X-rays from Guangzhou Women and Children's Medical Center via Kaggle.
            </p>
          </div>

        </div>
      </section>

      {/* The 4-Step Pipeline */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-3xl p-8 sm:p-12 border border-slate-200 shadow-sm">
          
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">
              End-to-End Diagnostic Pipeline
            </h2>
            <p className="text-sm text-slate-500 mt-2">
              From raw digital radiograph upload to clinical probability scores and localized activation maps.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
            
            <div className="text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-brand-50 border-2 border-brand-600 text-brand-700 font-bold flex items-center justify-center mx-auto text-lg shadow-sm">
                1
              </div>
              <h4 className="font-semibold text-slate-900 text-base">Standardized Preprocessing</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Single-channel X-rays converted to 3-channel RGB, resized to 224&times;224, and normalized with ImageNet statistics.
              </p>
            </div>

            <div className="text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-brand-50 border-2 border-brand-600 text-brand-700 font-bold flex items-center justify-center mx-auto text-lg shadow-sm">
                2
              </div>
              <h4 className="font-semibold text-slate-900 text-base">Residual Feature Extraction</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                18 residual layers with skip-connections extract low-level edges, lung boundaries, and consolidated infiltrates.
              </p>
            </div>

            <div className="text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-brand-50 border-2 border-brand-600 text-brand-700 font-bold flex items-center justify-center mx-auto text-lg shadow-sm">
                3
              </div>
              <h4 className="font-semibold text-slate-900 text-base">Calibrated Thresholding</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Classification cutoff calibrated at ~0.40 on the validation set to achieve ~95% Recall, addressing medical asymmetry.
              </p>
            </div>

            <div className="text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-brand-50 border-2 border-brand-600 text-brand-700 font-bold flex items-center justify-center mx-auto text-lg shadow-sm">
                4
              </div>
              <h4 className="font-semibold text-slate-900 text-base">Grad-CAM Overlay</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Backpropagated class scores generate 2D heatmaps overlaid on the radiograph to explain spatial attention.
              </p>
            </div>

          </div>

          <div className="mt-12 text-center pt-8 border-t border-slate-100">
            <button
              onClick={() => onNavigate('analyze')}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-brand-600 text-white font-semibold text-sm hover:bg-brand-700 transition-colors shadow-md shadow-brand-500/20"
            >
              <Activity className="w-4 h-4" />
              <span>Launch Interactive Analysis</span>
            </button>
          </div>

        </div>
      </section>

      {/* Technology Stack Badges */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-slate-900 text-white rounded-3xl p-8 sm:p-10 shadow-xl">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 mb-8 pb-8 border-b border-slate-800">
            <div>
              <h3 className="text-xl sm:text-2xl font-bold">Production-Grade Medical AI Stack</h3>
              <p className="text-xs sm:text-sm text-slate-400 mt-1">
                Combining rigorous deep learning research with a fast, modern reactive web interface.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono bg-slate-800 px-3 py-1.5 rounded-lg text-brand-400 border border-slate-700">
              <span>FastAPI &middot; PyTorch &middot; React 18</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            
            <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-center">
              <p className="text-xs font-bold text-brand-400">Core DL</p>
              <p className="text-sm font-semibold text-white mt-1">PyTorch 2.0+</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-center">
              <p className="text-xs font-bold text-brand-400">Vision</p>
              <p className="text-sm font-semibold text-white mt-1">torchvision</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-center">
              <p className="text-xs font-bold text-brand-400">Explainability</p>
              <p className="text-sm font-semibold text-white mt-1">Grad-CAM Hook</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-center">
              <p className="text-xs font-bold text-brand-400">Backend API</p>
              <p className="text-sm font-semibold text-white mt-1">FastAPI</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-center">
              <p className="text-xs font-bold text-brand-400">Frontend</p>
              <p className="text-sm font-semibold text-white mt-1">React &amp; Vite</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 text-center">
              <p className="text-xs font-bold text-brand-400">Styling</p>
              <p className="text-sm font-semibold text-white mt-1">Tailwind CSS</p>
            </div>

          </div>
        </div>
      </section>

    </div>
  );
};
