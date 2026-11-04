import React from 'react';
import { Activity, Shield, ExternalLink, GitBranch } from 'lucide-react';
import { PageId } from '../types';

interface FooterProps {
  onNavigate: (page: PageId) => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  return (
    <footer className="bg-slate-900 text-slate-400 text-sm border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          
          {/* Brand & Mission */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 text-white font-bold text-lg mb-3">
              <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-white">
                <Activity className="w-5 h-5" />
              </div>
              <span>PneumoAI</span>
            </div>
            <p className="text-slate-400 text-xs sm:text-sm leading-relaxed max-w-md mb-4">
              An educational deep learning healthcare platform demonstrating transfer learning with ResNet-18, dynamic class-weighted loss, threshold sensitivity calibration, and Grad-CAM visual interpretability on pediatric chest radiographs.
            </p>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Shield className="w-4 h-4 text-brand-400" />
              <span>Strictly Academic Demonstration &middot; Non-Commercial Research</span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-white text-xs font-semibold uppercase tracking-wider mb-3">Platform Views</h4>
            <ul className="space-y-2 text-xs sm:text-sm">
              <li>
                <button onClick={() => onNavigate('home')} className="hover:text-white transition-colors">
                  System Dashboard
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('analyze')} className="hover:text-white transition-colors">
                  Chest X-Ray Analysis
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('model-info')} className="hover:text-white transition-colors">
                  Architecture &amp; Preprocessing
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('performance')} className="hover:text-white transition-colors">
                  Clinical Metrics &amp; Curves
                </button>
              </li>
            </ul>
          </div>

          {/* Resources & Dataset */}
          <div>
            <h4 className="text-white text-xs font-semibold uppercase tracking-wider mb-3">Resources &amp; Data</h4>
            <ul className="space-y-2 text-xs sm:text-sm">
              <li>
                <a 
                  href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" 
                  target="_blank" 
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 hover:text-white transition-colors"
                >
                  <span>Kaggle Dataset</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </li>
              <li>
                <a 
                  href="https://pytorch.org/hub/pytorch_vision_resnet/" 
                  target="_blank" 
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 hover:text-white transition-colors"
                >
                  <span>PyTorch ResNet-18</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </li>
              <li>
                <a 
                  href="http://localhost:8000/docs" 
                  target="_blank" 
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 hover:text-white transition-colors"
                >
                  <span>FastAPI Swagger Docs</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </li>
              <li>
                <span className="inline-flex items-center gap-1.5 text-slate-500">
                  <GitBranch className="w-3.5 h-3.5" />
                  <span>Version 1.0.0 (main)</span>
                </span>
              </li>
            </ul>
          </div>

        </div>

        {/* Legal Disclaimer Box */}
        <div className="border-t border-slate-800 pt-6 mt-6">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-400 leading-relaxed mb-4">
            <span className="font-semibold text-amber-400">Notice:</span> PneumoAI is an educational machine learning project. It is not a medical device and must not be used for diagnosis, treatment, or medical decision-making. Always consult a qualified healthcare professional for medical concerns.
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
            <p>&copy; {new Date().getFullYear()} PneumoAI. Open-source educational project under MIT License.</p>
            <p className="flex items-center gap-1">
              Built with PyTorch, FastAPI, React &amp; Tailwind CSS
            </p>
          </div>
        </div>

      </div>
    </footer>
  );
};
