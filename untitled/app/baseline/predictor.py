import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib


class BaselinePredictor:
    """
    Loads and runs the TF-IDF + LogisticRegression baseline.
    Expects artifacts:
      - tfidf_vectorizer.joblib
      - logreg_model.joblib

    Default location (relative to this file):
      <project_root>/app/models/baseline/
    where <project_root> is the folder that contains the 'app/' directory.
    """

    def __init__(self):
        # Resolve project root robustly, independent of where uvicorn is started.
        # This file is: <project_root>/app/baseline/predictor.py
        self.project_root = Path(__file__).resolve().parents[2]  # -> <project_root>/app/.. = <project_root>
        default_dir = self.project_root / "app" / "models" / "baseline"

        # Allow overriding via env var (useful for Docker / CI).
        self.baseline_dir = Path(os.getenv("BASELINE_DIR", str(default_dir))).resolve()

        self.vec_path = self.baseline_dir / "tfidf_vectorizer.joblib"
        self.model_path = self.baseline_dir / "logreg_model.joblib"

        self.vectorizer = None
        self.model = None

        self.model_version = os.getenv("BASELINE_MODEL_VERSION", "baseline-v1")

        # Optional debug logging
        self.debug = os.getenv("BASELINE_DEBUG", "false").lower() in ("1", "true", "yes")

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[BaselinePredictor] {msg}")

    def load(self) -> None:
        """
        Loads artifacts if present. If not present, keeps predictor in "not ready" state.
        The FastAPI service can still start; /health will show baselineReady=false.
        """
        self._log(f"Project root: {self.project_root}")
        self._log(f"Baseline dir: {self.baseline_dir}")
        self._log(f"Vectorizer path: {self.vec_path} (exists={self.vec_path.exists()})")
        self._log(f"Model path: {self.model_path} (exists={self.model_path.exists()})")

        if not self.vec_path.exists() or not self.model_path.exists():
            self.vectorizer = None
            self.model = None
            return

        self.vectorizer = joblib.load(self.vec_path)
        self.model = joblib.load(self.model_path)
        self._log("✅ Baseline artifacts loaded successfully.")

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def predict(self, text: str) -> Tuple[str, float, str]:
        """
        Returns:
          (priority_label, score, model_version)

        priority_label in {"HIGH","MEDIUM","LOW"}
        score = max predicted probability (0..1)
        """
        if not self.is_ready():
            raise RuntimeError(
                "Baseline model not loaded. "
                f"Expected artifacts at: {self.vec_path} and {self.model_path} "
                "(or set BASELINE_DIR env var)."
            )

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        idx = int(proba.argmax())

        label = str(self.model.classes_[idx])
        score = float(proba[idx])
        return label, score, self.model_version

    def predict_proba_map(self, text: str) -> Dict[str, float]:
        """
        Convenience method: returns a dict of label -> probability.
        Useful for debugging and evaluation plots.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Baseline model not loaded. "
                f"Expected artifacts at: {self.vec_path} and {self.model_path} "
                "(or set BASELINE_DIR env var)."
            )

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        return {str(label): float(p) for label, p in zip(self.model.classes_, proba)}
