import os, json, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

MODEL_VERSION = os.getenv("XLMR_MODEL_VERSION", "xlmr-v1")
REPORT_DIR = os.getenv("REPORT_DIR", f"reports/{MODEL_VERSION}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.getenv("TRAIN_CSV", os.path.join(BASE_DIR, "data", "train.csv"))
OUT_DIR = os.getenv("XLMR_OUT_DIR", os.path.join(BASE_DIR, "app", "models", "xlmr"))

RANDOM_STATE = 42
LABEL_COL = "priority_label"
TITLE_COL = "title"
DESC_COL = "description"
ALLOWED_LABELS = ["HIGH", "MEDIUM", "LOW"]  # feste Reihenfolge für Plots

BASE_MODEL = os.getenv("XLMR_BASE", "xlm-roberta-base")
MAX_LEN = 256

def build_text(row: pd.Series) -> str:
    title = str(row.get(TITLE_COL, "") or "").strip()
    desc = str(row.get(DESC_COL, "") or "").strip()
    return f"{title}\n{desc}".strip()

def percentile_ms(arr, p):
    return float(np.percentile(arr, p) * 1000.0)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1m)}

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"train.csv not found at: {DATA_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
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
    y_labels = df[LABEL_COL].tolist()

    label2id = {l: i for i, l in enumerate(ALLOWED_LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    y = [label2id[l] for l in y_labels]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LEN)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=MAX_LEN)

    train_ds = SimpleDataset(train_enc, y_train)
    test_ds  = SimpleDataset(test_enc,  y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(ALLOWED_LABELS),
        label2id=label2id,
        id2label=id2label
    )

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "_runs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # --- Evaluation (same as baseline) ---
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    y_pred = np.argmax(logits, axis=-1)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(
        y_test, y_pred, target_names=ALLOWED_LABELS, digits=4, output_dict=True
    )

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(ALLOWED_LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ALLOWED_LABELS)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix – {MODEL_VERSION}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # --- Per-class F1 plot ---
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

    # --- Model-only latency benchmark (warm) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Warm-up
    with torch.no_grad():
        sample = tokenizer([X_test[0]], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        _ = model(**sample).logits

    N = min(200, len(X_test))
    times = []

    with torch.no_grad():
        for i in range(N):
            sample = tokenizer([X_test[i]], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            start = time.perf_counter()
            _ = model(**sample).logits
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)

    latency_stats = {
        "p50_ms": percentile_ms(times, 50),
        "p95_ms": percentile_ms(times, 95),
        "p99_ms": percentile_ms(times, 99),
        "mean_ms": float(times.mean() * 1000.0),
        "n": int(N),
        "device": device
    }

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(times * 1000.0, bins=30)
    ax.set_title(f"Model-only latency (ms) – {MODEL_VERSION}")
    ax.set_xlabel("ms")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "latency_model.png"), dpi=200)
    plt.close(fig)

    # --- Persist artifacts into app/models/xlmr (for FastAPI predictor.load()) ---
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

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
        "baseModel": BASE_MODEL,
        "max_length": MAX_LEN
    }

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ XLM-R Evaluation complete")
    print(f"Accuracy: {acc:.4f} | F1_macro: {f1_macro:.4f}")
    print(f"Saved plots + metrics to: {REPORT_DIR}")
    print(f"Saved model to: {OUT_DIR}")


if __name__ == "__main__":
    main()
