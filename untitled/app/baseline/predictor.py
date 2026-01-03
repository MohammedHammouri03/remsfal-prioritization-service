import os
import joblib
from typing import Tuple

BASELINE_DIR = os.getenv("BASELINE_DIR", "models/baseline")

class BaselinePredictor:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_version = os.getenv("BASELINE_MODEL_VERSION", "baseline-v1")

    def load(self):
        vec_path = os.path.join(BASELINE_DIR, "tfidf_vectorizer.joblib")
        model_path = os.path.join(BASELINE_DIR, "logreg_model.joblib")

        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            return

        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def predict(self, text: str) -> Tuple[str, float, str]:
        if not self.is_ready():
            raise RuntimeError("Baseline model not loaded. Train and place artifacts in models/baseline/")

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        idx = int(proba.argmax())
        label = self.model.classes_[idx]
        score = float(proba[idx])
        return str(label), score, self.model_version
