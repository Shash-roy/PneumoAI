import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div className="bg-amber-50 border-b border-amber-200 py-2.5 px-4 text-xs sm:text-sm text-amber-900 shadow-inner">
      <div className="max-w-7xl mx-auto flex items-center justify-center gap-2.5 text-center">
        <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
        <p className="font-medium">
          <span className="font-bold">Medical Disclaimer:</span> PneumoAI is an educational machine learning project. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making. Always consult a qualified healthcare professional.
        </p>
      </div>
    </div>
  );
};
