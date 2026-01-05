import os
import re
from typing import Dict, Tuple

from openai import OpenAI


ALLOWED_LABELS = ["HIGH", "MEDIUM", "LOW"]


class OpenAIPredictor:
    """
    Calls OpenAI to classify into HIGH/MEDIUM/LOW.
    Returns (label, score, model_version) just like baseline/xlmr.
    score is derived from model's self-reported confidence (0..1).
    """

    def __init__(self):
        self.client = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.model_version = os.getenv("OPENAI_MODEL_VERSION", "openai-gpt4o-mini-v1")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

        self.debug = os.getenv("OPENAI_DEBUG", "false").lower() in ("1", "true", "yes")

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[OpenAIPredictor] {msg}")

    def load(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self.client = None
            return
        self.client = OpenAI(api_key=api_key)
        self._log("✅ OpenAI client ready.")

    def is_ready(self) -> bool:
        return self.client is not None

    def predict(self, text: str) -> Tuple[str, float, str]:
        if not self.is_ready():
            raise RuntimeError("OpenAI not configured. Set OPENAI_API_KEY.")

        prompt = build_prompt(text)

        # Use Responses API (recommended) via openai-python
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )

        out = extract_text(resp)
        label, score = parse_label_and_confidence(out)

        # hard guard
        if label not in ALLOWED_LABELS:
            # fallback: try to find label anywhere
            label2 = find_label_anywhere(out)
            if label2:
                label = label2
            else:
                raise RuntimeError(f"OpenAI returned invalid label: {out}")

        if score is None:
            # fallback if model doesn't provide confidence
            score = 0.5

        return label, float(score), self.model_version

    def predict_proba_map(self, text: str) -> Dict[str, float]:
        """
        Optional: Ask OpenAI for calibrated probabilities.
        To keep fairness with other models, we keep it simple and return one score.
        But for debugging, we can request pseudo-probs.
        """
        label, score, _ = self.predict(text)
        # Put all mass on predicted class (simple)
        return {l: (score if l == label else max(0.0, (1.0 - score) / 2.0)) for l in ALLOWED_LABELS}


def build_prompt(text: str) -> str:
    # Keep consistent instructions; allow only 3 labels.
    return f"""
You are a classification function for facility-management issue priority.

Classify the following issue into exactly one of:
HIGH, MEDIUM, LOW

Rules (high-level):
- HIGH: safety risk, outage, urgent operational impact, legal/compliance risk.
- MEDIUM: significant but not critical, workaround exists, limited scope.
- LOW: minor inconvenience, cosmetic, can be scheduled.

Return ONLY valid JSON, no extra text.
Schema:
{{
  "priority": "HIGH|MEDIUM|LOW",
  "confidence": 0.0-1.0
}}

Issue text:
\"\"\"{text}\"\"\"
""".strip()


def extract_text(resp) -> str:
    # openai-python Responses API: easiest is resp.output_text
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def parse_label_and_confidence(s: str):
    # Expect JSON, but be robust
    s = s.strip()
    # Try JSON object extraction
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None, None
    blob = m.group(0)

    # parse manually to avoid extra deps
    pr = re.search(r'"priority"\s*:\s*"([^"]+)"', blob, re.IGNORECASE)
    cf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', blob, re.IGNORECASE)

    label = pr.group(1).upper().strip() if pr else None
    conf = float(cf.group(1)) if cf else None

    # clamp confidence
    if conf is not None:
        conf = max(0.0, min(1.0, conf))

    return label, conf


def find_label_anywhere(s: str):
    s2 = s.upper()
    for l in ALLOWED_LABELS:
        if l in s2:
            return l
    return None
