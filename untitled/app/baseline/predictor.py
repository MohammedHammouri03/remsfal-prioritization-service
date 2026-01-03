import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "baseline")

VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")

TITLE = "Heizung komplett ausgefallen"
DESCRIPTION = "Seit heute Morgen funktioniert die Heizung in der gesamten Wohnung nicht mehr."

TEXT = f"{TITLE}\n{DESCRIPTION}".strip()

def main():
    # --- Check Artefakte ---
    if not os.path.exists(VEC_PATH):
        raise FileNotFoundError(f"Vectorizer not found: {VEC_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # --- Laden ---
    print("🔹 Loading vectorizer...")
    vectorizer = joblib.load(VEC_PATH)

    print("🔹 Loading model...")
    model = joblib.load(MODEL_PATH)

    # --- Transformieren ---
    X = vectorizer.transform([TEXT])

    # --- Vorhersage ---
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())

    predicted_label = model.classes_[idx]
    confidence = float(proba[idx])

    # --- Ausgabe ---
    print("\n🧪 BASELINE PREDICTION RESULT")
    print("--------------------------------")
    print(f"Text:\n{TEXT}\n")
    print(f"Predicted priority: {predicted_label}")
    print(f"Confidence score:   {confidence:.4f}")
    print(f"All class probs:")
    for label, p in zip(model.classes_, proba):
        print(f"  {label:6s}: {p:.4f}")

if __name__ == "__main__":
    main()
