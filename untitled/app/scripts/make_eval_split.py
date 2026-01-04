import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("TRAIN_CSV", os.path.join(BASE_DIR, "data", "train.csv"))

OUT_TRAIN = os.getenv("EVAL_TRAIN_OUT", os.path.join(BASE_DIR, "data", "train_split.csv"))
OUT_TEST = os.getenv("EVAL_TEST_OUT", os.path.join(BASE_DIR, "data", "eval_test.csv"))

LABEL_COL = "priority_label"
TITLE_COL = "title"
DESC_COL = "description"
ID_COL = "id"
ALLOWED_LABELS = ["HIGH", "MEDIUM", "LOW"]

def build_text(row: pd.Series) -> str:
    title = str(row.get(TITLE_COL, "") or "").strip()
    desc = str(row.get(DESC_COL, "") or "").strip()
    return f"{title}\n{desc}".strip()

def main():
    df = pd.read_csv(DATA_PATH)

    for col in [LABEL_COL, TITLE_COL, DESC_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.upper().str.strip()
    df = df[df[LABEL_COL].isin(ALLOWED_LABELS)].copy()

    df["text"] = df.apply(build_text, axis=1).astype(str).str.strip()
    df = df[df["text"].str.len() >= 5].copy()

    # Falls id fehlt, erzeugen wir eine stabile ID (für Predictions)
    if ID_COL not in df.columns:
        df[ID_COL] = [f"ROW-{i:05d}" for i in range(len(df))]

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]
    )

    # nur relevante Spalten sichern (plus optional category)
    cols = [ID_COL, LABEL_COL, TITLE_COL, DESC_COL, "text"]
    if "category" in df.columns:
        cols.append("category")

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TEST), exist_ok=True)

    train_df[cols].to_csv(OUT_TRAIN, index=False)
    test_df[cols].to_csv(OUT_TEST, index=False)

    print(f"✅ Saved train split: {OUT_TRAIN} ({len(train_df)} rows)")
    print(f"✅ Saved test split : {OUT_TEST} ({len(test_df)} rows)")

if __name__ == "__main__":
    main()
