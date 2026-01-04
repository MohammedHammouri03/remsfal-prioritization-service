import os, json, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from openai import OpenAI


MODEL_VERSION = os.getenv("OPENAI_MODEL_VERSION", "openai-gpt4o-mini-v1")
REPORT_DIR = os.getenv("REPORT_DIR", f"reports/{MODEL_VERSION}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("TRAIN_CSV", os.path.join(BASE_DIR, "data", "train.csv"))

RANDOM_STATE = 42
LABEL_COL = "priority_label"
TITLE_COL = "title"
DESC_COL = "description"
ALLOWED_LABELS = ["HIGH", "MEDIUM", "LOW"]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
MAX_CALLS = int(os.getenv("OPENAI_MAX_CALLS", "300"))


def build_text(row: pd.Series) -> str:
    title = str(row.get(TITLE_COL, "") or "").strip()
    desc = str(row.get(DESC_COL, "") or "").strip()
    return f"{title}\n{desc}".strip()


def percentile_ms(arr, p):
    return float(np.percentile(arr, p))


def build_prompt(text: str) -> str:
    return f"""
You are a classification function for facility-management issue priority.

Classify the following issue into exactly one of:
HIGH, MEDIUM, LOW

Return ONLY valid JSON, no extra text.
Schema:
{{
  "priority": "HIGH|MEDIUM|LOW",
  "confidence": 0.0-1.0
}}

Issue text:
\"\"\"{text}\"\"\"
""".strip()


def parse_json_like(s: str):
    import re
    s = s.strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None, None
    blob = m.group(0)
    pr = re.search(r'"priority"\s*:\s*"([^"]+)"', blob, re.IGNORECASE)
    cf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', blob, re.IGNORECASE)
    label = pr.group(1).upper().strip() if pr else None
    conf = float(cf.group(1)) if cf else None
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    return label, conf


def main():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = {LABEL_COL, TITLE_COL, DESC_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.upper().str.strip()
    df = df[df[LABEL_COL].isin(ALLOWED_LABELS)].copy()

    df["text"] = df.apply(build_text, axis=1).astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].copy()

    X = df["text"].tolist()
    y = df[LABEL_COL].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    client = OpenAI(api_key=api_key)

    n = min(len(X_test), MAX_CALLS)
    X_eval = X_test[:n]
    y_eval = y_test[:n]

    preds = []
    confs = []
    times_ms = []

    for i, text in enumerate(X_eval):
        prompt = build_prompt(text)
        start = time.perf_counter()
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=TEMPERATURE
        )
        end = time.perf_counter()

        out = getattr(resp, "output_text", str(resp))
        label, conf = parse_json_like(out)

        if label not in ALLOWED_LABELS:
            # fallback
            label = "MEDIUM"

        if conf is None:
            conf = 0.5

        preds.append(label)
        confs.append(float(conf))
        times_ms.append((end - start) * 1000.0)

        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{n}] done...")

    acc = float(accuracy_score(y_eval, preds))
    f1_macro = float(f1_score(y_eval, preds, average="macro"))
    report = classification_report(y_eval, preds, digits=4, output_dict=True)

    # confusion matrix
    cm = confusion_matrix(y_eval, preds, labels=ALLOWED_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ALLOWED_LABELS)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix – {MODEL_VERSION}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # per-class f1
    per_class_f1 = [float(report[label]["f1-score"]) for label in ALLOWED_LABELS]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ALLOWED_LABELS, per_class_f1)
    ax.set_ylim(0, 1)
    ax.set_title(f"F1 per class – {MODEL_VERSION}")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "per_class_f1.png"), dpi=200)
    plt.close(fig)

    # latency histogram
    times_ms = np.array(times_ms, dtype=float)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(times_ms, bins=30)
    ax.set_title(f"Model-only latency (ms) – {MODEL_VERSION}")
    ax.set_xlabel("ms")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "latency_model.png"), dpi=200)
    plt.close(fig)

    latency_stats = {
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "mean_ms": float(times_ms.mean()),
        "n": int(n),
    }

    metrics = {
        "modelVersion": MODEL_VERSION,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "dataPath": DATA_PATH,
        "rowsTotal": int(len(df)),
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        "metrics": {"accuracy": acc, "f1_macro": f1_macro},
        "per_class": {
            lbl: {
                "f1": float(report[lbl]["f1-score"]),
                "precision": float(report[lbl]["precision"]),
                "recall": float(report[lbl]["recall"]),
            } for lbl in ALLOWED_LABELS
        },
        "latency_model_only": latency_stats,
        "openai": {
            "model": OPENAI_MODEL,
            "temperature": TEMPERATURE,
            "calls": int(n),
            "mean_confidence": float(np.mean(confs)),
        }
    }

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ OpenAI evaluation complete")
    print(f"Accuracy: {acc:.4f} | F1_macro: {f1_macro:.4f}")
    print(f"Saved plots + metrics to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
