import React, { useEffect, useState } from 'react';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { HomePage } from './pages/HomePage';
import { AnalysisPage } from './pages/AnalysisPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { PerformancePage } from './pages/PerformancePage';
import { api } from './services/api';
import { PageId, HealthStatus } from './types';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('home');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  // Check backend health on initial load and periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await api.getHealth();
        setHealth(res);
      } catch (err) {
        console.warn('Initial backend health check failed:', err);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Scroll to top on page transition
  const handleNavigate = (page: PageId) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 font-sans text-slate-900 antialiased selection:bg-brand-500 selection:text-white">
      {/* 1. Medical Disclaimer Banner */}
      <DisclaimerBanner />

      {/* 2. Top Navigation Bar */}
      <Navbar 
        currentPage={currentPage} 
        onNavigate={handleNavigate} 
        health={health} 
      />

      {/* 3. Main Dynamic Content */}
      <main className="flex-grow">
        {currentPage === 'home' && (
          <HomePage onNavigate={handleNavigate} health={health} />
        )}
        {currentPage === 'analyze' && (
          <AnalysisPage health={health} />
        )}
        {currentPage === 'model-info' && (
          <ModelInfoPage />
        )}
        {currentPage === 'performance' && (
          <PerformancePage />
        )}
      </main>

      {/* 4. Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
};

export default App;

