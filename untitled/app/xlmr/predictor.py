import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class XlmrPredictor:
    """
    Loads and runs a fine-tuned XLM-R sequence classification model.
    Expects artifacts in:
      <project_root>/app/models/xlmr/

    Must be compatible with Baseline endpoint:
      predict(text) -> (label, score, model_version)
    """

    def __init__(self):
        # This file: <project_root>/app/xlmr/predictor.py  -> parents[2] = <project_root>
        self.project_root = Path(__file__).resolve().parents[2]
        default_dir = self.project_root / "app" / "models" / "xlmr"

        self.model_dir = Path(os.getenv("XLMR_DIR", str(default_dir))).resolve()
        self.model_version = os.getenv("XLMR_MODEL_VERSION", "xlmr-v1")

        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.debug = os.getenv("XLMR_DEBUG", "false").lower() in ("1", "true", "yes")

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[XlmrPredictor] {msg}")

    def load(self) -> None:
        """
        Loads model artifacts if present. If not present -> not ready.
        Service should still start; /health indicates readiness.
        """
        self._log(f"Project root: {self.project_root}")
        self._log(f"Model dir: {self.model_dir} (exists={self.model_dir.exists()})")

        # Basic sanity check: needs config + tokenizer files + weights
        if not self.model_dir.exists():
            self.tokenizer = None
            self.model = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self.model.to(self.device)
            self.model.eval()
            self._log(f"✅ XLM-R model loaded on {self.device}")
        except Exception as e:
            # Keep service up, but mark predictor not ready
            self._log(f"❌ Failed to load model: {e}")
            self.tokenizer = None
            self.model = None

    def is_ready(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float, str]:
        """
        Returns (priority_label, score, model_version)
        priority_label in {"HIGH","MEDIUM","LOW"}
        score = max softmax probability (0..1)
        """
        if not self.is_ready():
            raise RuntimeError(
                "XLM-R model not loaded. "
                f"Expected artifacts at: {self.model_dir} (or set XLMR_DIR env var)."
            )

        inputs = self.tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]
        probs = softmax(logits)

        idx = int(np.argmax(probs))
        label = self.model.config.id2label.get(idx, str(idx))
        score = float(probs[idx])

        # Enforce exact labels like baseline: HIGH/MEDIUM/LOW
        label = str(label).upper().strip()
        return label, score, self.model_version

    @torch.no_grad()
    def predict_proba_map(self, text: str) -> Dict[str, float]:
        """
        Returns a dict label -> probability (useful for debugging/eval).
        """
        if not self.is_ready():
            raise RuntimeError(
                "XLM-R model not loaded. "
                f"Expected artifacts at: {self.model_dir} (or set XLMR_DIR env var)."
            )

        inputs = self.tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]
        probs = softmax(logits)

        out = {}
        for i, p in enumerate(probs):
            lbl = self.model.config.id2label.get(i, str(i))
            out[str(lbl).upper().strip()] = float(p)
        return out


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
