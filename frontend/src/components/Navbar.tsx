import React from 'react';
import { Activity, ShieldCheck, Cpu, BarChart2, AlertCircle, CheckCircle2, ChevronRight } from 'lucide-react';
import { PageId, HealthStatus } from '../types';

interface NavbarProps {
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
  health: HealthStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate, health }) => {
  const isConnected = health && health.status === 'healthy';
  const hasCustomCheckpoint = health && health.is_custom_checkpoint;

  return (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Brand */}
          <div 
            onClick={() => onNavigate('home')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-600 to-navy-900 flex items-center justify-center text-white shadow-md shadow-brand-500/20 group-hover:scale-105 transition-transform">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xl font-bold tracking-tight text-slate-900">Pneumo<span className="text-brand-600">AI</span></span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-brand-100 text-brand-700">v1.0</span>
              </div>
              <p className="text-[11px] text-slate-500 hidden sm:block">AI-Powered Chest Radiograph Analysis</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            <button
              onClick={() => onNavigate('home')}
              className={`px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                currentPage === 'home'
                  ? 'bg-brand-50 text-brand-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => onNavigate('analyze')}
              className={`px-3.5 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                currentPage === 'analyze'
                  ? 'bg-brand-600 text-white shadow-sm shadow-brand-600/30'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <span>Analyze X-Ray</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate('model-info')}
              className={`px-3.5 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                currentPage === 'model-info'
                  ? 'bg-brand-50 text-brand-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span>Model Architecture</span>
            </button>
            <button
              onClick={() => onNavigate('performance')}
              className={`px-3.5 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                currentPage === 'performance'
                  ? 'bg-brand-50 text-brand-700'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <BarChart2 className="w-4 h-4" />
              <span>Evaluation & Metrics</span>
            </button>
          </nav>

          {/* Right Status Indicator */}
          <div className="flex items-center gap-3">
            <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs border border-slate-200 bg-slate-50">
              <span className="text-slate-500 font-medium">Model:</span>
              <span className="font-semibold text-slate-700">ResNet-18</span>
              <span className="w-1.5 h-1.5 rounded-full bg-slate-300"></span>
              {isConnected ? (
                hasCustomCheckpoint ? (
                  <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                    Fine-Tuned Checkpoint
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-blue-700 font-medium">
                    <ShieldCheck className="w-3.5 h-3.5 text-blue-500" />
                    Pretrained ResNet-18
                  </span>
                )
              ) : (
                <span className="inline-flex items-center gap-1 text-amber-700 font-medium">
                  <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                  Backend Offline
                </span>
              )}
            </div>

            {/* Mobile CTA */}
            <button
              onClick={() => onNavigate('analyze')}
              className="md:hidden px-3 py-1.5 bg-brand-600 text-white rounded-lg text-xs font-semibold"
            >
              Analyze
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Nav Bar */}
      <div className="md:hidden flex items-center justify-around border-t border-slate-100 bg-white py-2 px-2 text-xs font-medium text-slate-600">
        <button 
          onClick={() => onNavigate('home')}
          className={`py-1 px-2.5 rounded ${currentPage === 'home' ? 'text-brand-600 font-bold' : ''}`}
        >
          Home
        </button>
        <button 
          onClick={() => onNavigate('analyze')}
          className={`py-1 px-2.5 rounded ${currentPage === 'analyze' ? 'text-brand-600 font-bold' : ''}`}
        >
          Analyze
        </button>
        <button 
          onClick={() => onNavigate('model-info')}
          className={`py-1 px-2.5 rounded ${currentPage === 'model-info' ? 'text-brand-600 font-bold' : ''}`}
        >
          Model Info
        </button>
        <button 
          onClick={() => onNavigate('performance')}
          className={`py-1 px-2.5 rounded ${currentPage === 'performance' ? 'text-brand-600 font-bold' : ''}`}
        >
          Performance
        </button>
      </div>
    </header>
  );
};

