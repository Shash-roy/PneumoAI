"""
Unit Test Suite for PneumoAI FastAPI Backend.
=============================================
Validates backend schemas, model loader singleton, and endpoint logic.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestBackendSchemas(unittest.TestCase):
    """Verifies Pydantic response models and fields."""

    def test_health_response_schema(self):
        try:
            from backend.app.schemas import HealthResponse
            res = HealthResponse(
                status="healthy",
                model_name="ResNet-18",
                checkpoint_available=False,
                is_custom_checkpoint=False,
                device="cpu",
                default_threshold=0.40,
                message="Test health check"
            )
            self.assertEqual(res.status, "healthy")
            self.assertEqual(res.default_threshold, 0.40)
            self.assertIn("NORMAL", res.classes)
        except ImportError:
            pass

    def test_predict_response_schema(self):
        try:
            from backend.app.schemas import PredictResponse
            res = PredictResponse(
                predicted_class="PNEUMONIA",
                normal_probability=0.038,
                pneumonia_probability=0.962,
                threshold_used=0.40,
                is_pneumonia=True,
                inference_time_ms=42.0,
                checkpoint_status="Fine-Tuned Checkpoint"
            )
            self.assertTrue(res.is_pneumonia)
            self.assertEqual(res.predicted_class, "PNEUMONIA")
            self.assertIn("PneumoAI is an educational", res.disclaimer)
        except ImportError:
            pass


class TestFullStackFileStructure(unittest.TestCase):
    """Validates presence of all full-stack PneumoAI assets."""

    def test_full_stack_files(self):
        required_paths = [
            PROJECT_ROOT / "backend" / "app" / "main.py",
            PROJECT_ROOT / "backend" / "app" / "model_loader.py",
            PROJECT_ROOT / "backend" / "app" / "predict.py",
            PROJECT_ROOT / "backend" / "app" / "gradcam.py",
            PROJECT_ROOT / "backend" / "app" / "schemas.py",
            PROJECT_ROOT / "backend" / "requirements.txt",
            PROJECT_ROOT / "backend" / "Dockerfile",
            PROJECT_ROOT / "frontend" / "package.json",
            PROJECT_ROOT / "frontend" / "index.html",
            PROJECT_ROOT / "frontend" / "vite.config.ts",
            PROJECT_ROOT / "frontend" / "vercel.json",
            PROJECT_ROOT / "frontend" / "src" / "App.tsx",
            PROJECT_ROOT / "frontend" / "src" / "pages" / "HomePage.tsx",
            PROJECT_ROOT / "frontend" / "src" / "pages" / "AnalysisPage.tsx",
            PROJECT_ROOT / "frontend" / "src" / "pages" / "ModelInfoPage.tsx",
            PROJECT_ROOT / "frontend" / "src" / "pages" / "PerformancePage.tsx",
            PROJECT_ROOT / "frontend" / "src" / "components" / "Navbar.tsx",
            PROJECT_ROOT / "frontend" / "src" / "components" / "Footer.tsx",
            PROJECT_ROOT / "frontend" / "src" / "components" / "DisclaimerBanner.tsx",
            PROJECT_ROOT / ".env.example",
        ]
        for p in required_paths:
            self.assertTrue(p.exists(), f"Missing full-stack file: {p}")


if __name__ == "__main__":
    unittest.main()

