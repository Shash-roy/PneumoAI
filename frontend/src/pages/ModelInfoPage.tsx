import React from 'react';
import { 
  Cpu, 
  Database, 
  Scale, 
  Layers, 
  Eye, 
  BookOpen,
  ExternalLink
} from 'lucide-react';

export const ModelInfoPage: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      
      {/* Page Title */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-50 border border-brand-200 text-brand-700 text-xs font-semibold uppercase tracking-wider mb-3">
          Architecture &amp; Methodology Specification
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          PneumoAI Deep Learning System Architecture
        </h1>
        <p className="text-slate-600 text-sm sm:text-base mt-2 max-w-3xl leading-relaxed">
          Comprehensive breakdown of the computer vision transfer learning pipeline, dataset split engineering, class imbalance mitigation, and visual saliency hooks.
        </p>
      </div>

      {/* 1. Problem Statement & Clinical Context */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2.5">
          <BookOpen className="w-5 h-5 text-brand-600" />
          <span>1. Problem Statement &amp; Pediatric Pneumonia Context</span>
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or purulent material (infiltrates and consolidations). Globally, pneumonia remains one of the single largest infectious causes of death in children under the age of 5. While chest radiography is the foremost diagnostic tool, interpreting pediatric radiographs is complicated by respiratory motion artifacts, variable lung inflation, and overlapping cardiovascular structures.
        </p>
        <div className="p-4 rounded-xl bg-blue-50/60 border border-blue-200 text-xs text-blue-900 leading-relaxed">
          <span className="font-bold">Asymmetric Clinical Cost:</span> A False Negative (failing to identify an infected child) risks catastrophic clinical progression into respiratory failure or sepsis. Conversely, a False Positive leads to a secondary clinician review or follow-up laboratory culture. Therefore, our optimization criterion prioritizes high <b>Diagnostic Sensitivity (Recall &ge; 95%)</b>.
        </div>
      </section>

      {/* 2. Dataset & The 16-Image Validation Flaw Fix */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2.5">
            <Database className="w-5 h-5 text-brand-600" />
            <span>2. Kaggle Dataset &amp; Validation Split Correction</span>
          </h2>
          <a
            href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
            target="_blank"
            rel="noreferrer"
            className="text-xs font-semibold text-brand-600 hover:text-brand-800 flex items-center gap-1"
          >
            <span>Kaggle: paultimothymooney/chest-xray-pneumonia</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        <p className="text-sm text-slate-600 leading-relaxed">
          The dataset consists of 5,856 verified anterior-posterior pediatric chest X-rays collected from patients between one and five years old at Guangzhou Women and Children’s Medical Center.
        </p>

        {/* Flaw explanation box */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-xl bg-red-50/60 border border-red-200 space-y-2">
            <span className="text-xs font-bold text-red-700 uppercase tracking-wider">The Kaggle Design Flaw</span>
            <h4 className="font-bold text-slate-900 text-sm">Flawed Original Validation Folder (Only 16 Images)</h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              The original dataset placed 5,216 images in <code className="bg-white px-1 py-0.5 rounded">train/</code>, but only <b>16 images</b> (8 Normal, 8 Pneumonia) in <code className="bg-white px-1 py-0.5 rounded">val/</code>. Evaluating validation loss or ROC-AUC on only 16 images causes extreme statistical noise, making early stopping unreliable.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-emerald-50/60 border border-emerald-200 space-y-2">
            <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider">Our Rigorous Engineering Fix</span>
            <h4 className="font-bold text-slate-900 text-sm">Stratified 85:15 Split (Untouched Test Set)</h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              We pool the original <code className="bg-white px-1 py-0.5 rounded">train/</code> and <code className="bg-white px-1 py-0.5 rounded">val/</code> pools (~5,232 images) and execute a reproducible <b>stratified 85:15 split</b> (85% train, 15% validation). The official <b>624-image test set is kept 100% untouched</b> until final evaluation to avoid data leakage.
            </p>
          </div>
        </div>
      </section>

      {/* 3. Deep Learning Architecture: ResNet-18 */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2.5">
          <Cpu className="w-5 h-5 text-brand-600" />
          <span>3. Deep Transfer Learning with ResNet-18</span>
        </h2>

        <p className="text-sm text-slate-600 leading-relaxed">
          Rather than training a convolutional network from scratch with random weights, we leverage <b>Transfer Learning</b> using the ResNet-18 architecture pretrained on ImageNet (1.28 million natural images).
        </p>

        {/* Residual connection explanation */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
          <h4 className="text-sm font-bold text-slate-800">Why ResNet (Residual Networks)?</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Traditional deep networks suffer from the vanishing gradient problem as depth increases. ResNet introduces identity shortcut connections:
          </p>
          <div className="font-mono text-xs bg-slate-900 text-brand-300 p-3 rounded-lg text-center">
            y = F(x, &#123;W_i&#125;) + x
          </div>
          <p className="text-xs text-slate-500">
            This bypasses intermediate layers and allows gradient signals to backpropagate directly across the entire network depth without degradation.
          </p>
        </div>

        {/* Head Replacement */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[11px] font-bold uppercase text-slate-400">Total Parameters</span>
            <p className="text-lg font-bold text-slate-800 mt-1">11,177,538</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[11px] font-bold uppercase text-slate-400">Feature Extractor</span>
            <p className="text-lg font-bold text-slate-800 mt-1">512 Channels</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[11px] font-bold uppercase text-slate-400">Classification Head</span>
            <p className="text-lg font-bold text-brand-700 mt-1">Linear(512 &rarr; 2)</p>
          </div>
        </div>
      </section>

      {/* 4. Preprocessing & Safe Medical Augmentation */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-brand-600" />
          <span>4. Preprocessing &amp; Clinically Safe Augmentation</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="text-sm font-bold text-slate-800">Standardized Pipeline</h4>
            <ul className="text-xs text-slate-600 space-y-2 list-disc list-inside">
              <li><b>3-Channel RGB Conversion:</b> Grayscale radiographs are mapped to 3 identical RGB channels to match ImageNet filter weights.</li>
              <li><b>Resolution Standardization:</b> Resized to uniform 224&times;224 pixels.</li>
              <li><b>ImageNet Normalization:</b> Zero-centered using ImageNet mean <code className="bg-slate-100 px-1 py-0.5 rounded">[0.485, 0.456, 0.406]</code> and std <code className="bg-slate-100 px-1 py-0.5 rounded">[0.229, 0.224, 0.225]</code>.</li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-bold text-slate-800">Medical Augmentation Safeguards</h4>
            <ul className="text-xs text-slate-600 space-y-2 list-disc list-inside">
              <li><b>Small Rotation (&plusmn;10&deg;):</b> Mimics slight pediatric patient movement or tilt.</li>
              <li><b>Gentle Zoom / Crop (0.90 to 1.00):</b> Simulates minor distance variance from detector.</li>
              <li><b>Brightness Variance (&plusmn;10%):</b> Emulates radiographic exposure variance.</li>
              <li><b className="text-red-700">Strict Prohibition:</b> NEVER flip vertically. Chest anatomy is strictly upright; flipping distorts cardiac-diaphragm orientation.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. Class Imbalance & Loss Weighting */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2.5">
          <Scale className="w-5 h-5 text-brand-600" />
          <span>5. Class Imbalance &amp; Dynamic Loss Penalties</span>
        </h2>

        <p className="text-sm text-slate-600 leading-relaxed">
          The training split contains roughly 3,875 Pneumonia images vs. 1,341 Normal images (~3:1 ratio). In standard Cross-Entropy, gradients are dominated by the majority class. We apply dynamic weighting:
        </p>

        <div className="font-mono text-xs bg-slate-900 text-brand-300 p-3 rounded-lg text-center">
          w_NORMAL = n_Pneumonia / n_Normal &approx; 2.89, &nbsp; w_PNEUMONIA = 1.0
        </div>

        <p className="text-xs text-slate-500 leading-relaxed">
          By weighting the loss, any prediction error on the minority Normal class incurs ~2.89&times; greater penalty, preventing the network from trivial majority-class shortcutting.
        </p>
      </section>

      {/* 6. Threshold Sensitivity & Grad-CAM */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2.5">
          <Eye className="w-5 h-5 text-brand-600" />
          <span>6. Grad-CAM Mathematical Formulation</span>
        </h2>

        <p className="text-sm text-slate-600 leading-relaxed">
          Grad-CAM (Gradient-Weighted Class Activation Mapping) computes the gradients of the target class score with respect to feature activation maps of ResNet-18's final convolutional layer (<code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-xs">layer4[-1]</code>):
        </p>

        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs text-slate-700">
          <p>1. Compute gradient of score <i>y<sup>c</sup></i> with respect to feature map <i>A<sup>k</sup></i>:</p>
          <div className="font-mono text-center py-1 font-bold text-slate-900">
            &alpha;<sub>k</sub><sup>c</sup> = (1/Z) &sum;<sub>i</sub> &sum;<sub>j</sub> (&part; y<sup>c</sup> / &part; A<sub>i,j</sub><sup>k</sup>)
          </div>
          <p>2. Take weighted linear combination and apply ReLU to keep features with positive influence:</p>
          <div className="font-mono text-center py-1 font-bold text-brand-700">
            L<sub>Grad-CAM</sub><sup>c</sup> = ReLU( &sum;<sub>k</sub> &alpha;<sub>k</sub><sup>c</sup> A<sup>k</sup> )
          </div>
        </div>

        <p className="text-xs text-slate-500 leading-relaxed">
          The resulting coarse 2D activation map (7&times;7) is upsampled to 224&times;224 via bicubic interpolation and colorized with OpenCV's JET colormap for superimposition onto the radiograph.
        </p>
      </section>

    </div>
  );
};
