import os, json, time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_VERSION = os.getenv("BASELINE_MODEL_VERSION", "baseline-v1")
REPORT_DIR = os.getenv("REPORT_DIR", f"reports/{MODEL_VERSION}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("TRAIN_CSV", os.path.join(BASE_DIR, "data", "train.csv"))
OUT_DIR = os.getenv("BASELINE_OUT_DIR", os.path.join(BASE_DIR, "models", "baseline"))

RANDOM_STATE = 42
LABEL_COL = "priority_label"
TITLE_COL = "title"
DESC_COL = "description"
ALLOWED_LABELS = ["HIGH", "MEDIUM", "LOW"]  # feste Reihenfolge für Plots


def build_text(row: pd.Series) -> str:
    title = str(row.get(TITLE_COL, "") or "").strip()
    desc = str(row.get(DESC_COL, "") or "").strip()
    return f"{title}\n{desc}".strip()


def percentile_ms(arr, p):
    return float(np.percentile(arr, p) * 1000.0)


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
    y = df[LABEL_COL].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -------------------------------------------------------------------------
    # CHANGED TRAINING LOGIC:
    # - same TF-IDF + Logistic Regression
    # - but with light hyperparameter tuning via CV (macro-F1)
    # - plus sublinear_tf to reduce overweighting repeated terms
    # -------------------------------------------------------------------------
    base_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=2,
        sublinear_tf=True,      # <- new (often helps)
        norm="l2"               # keep default explicit
    )

    base_clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    # Pipeline without importing sklearn.pipeline to keep your structure simple:
    # We tune vectorizer+clf by fitting vectorizer inside CV through a manual wrapper:
    # -> easiest: use a real Pipeline + GridSearchCV (clean + reproducible)
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", base_vectorizer),
        ("clf", base_clf),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "tfidf__min_df": [1, 2],
        "tfidf__max_features": [30_000, 50_000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.3, 1.0, 3.0, 10.0],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    vectorizer = best_pipe.named_steps["tfidf"]
    clf = best_pipe.named_steps["clf"]

    # Transform once for evaluation/latency (keeps your later logic unchanged)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Evaluation metrics ---
    y_pred = clf.predict(X_test_vec)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred, labels=ALLOWED_LABELS)
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
    _ = clf.predict_proba(X_test_vec[:1])  # warm-up

    N = min(300, X_test_vec.shape[0])
    times = []
    for i in range(N):
        start = time.perf_counter()
        _ = clf.predict_proba(X_test_vec[i:i+1])  # <- small fix: always 2D slice
        end = time.perf_counter()
        times.append(end - start)
    times = np.array(times)

    latency_stats = {
        "p50_ms": percentile_ms(times, 50),
        "p95_ms": percentile_ms(times, 95),
        "p99_ms": percentile_ms(times, 99),
        "mean_ms": float(times.mean() * 1000.0),
        "n": int(N),
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

    # --- Persist artifacts (as before) ---
    joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(OUT_DIR, "logreg_model.joblib"))

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
        "tuning": {
            "best_params": grid.best_params_,
            "cv_best_f1_macro": float(grid.best_score_),
            "cv_folds": 5,
        }
    }

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ Evaluation complete")
    print(f"Accuracy: {acc:.4f} | F1_macro: {f1_macro:.4f}")
    print(f"Saved plots + metrics to: {REPORT_DIR}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")


if __name__ == "__main__":
    main()
