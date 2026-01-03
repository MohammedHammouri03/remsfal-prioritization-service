from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.baseline.predictor import BaselinePredictor

app = FastAPI(title="REMSFAL Inference Service", version="0.1.0")

baseline = BaselinePredictor()
baseline.load()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "baselineReady": baseline.is_ready(),
        "modelVersion": baseline.model_version
    }

@app.post("/predict/baseline", response_model=PredictResponse)
def predict_baseline(req: PredictRequest):
    text = (req.title or "").strip()
    if req.description:
        text = f"{text}\n{req.description.strip()}".strip()

    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Text too short for classification")

    try:
        label, score, version = baseline.predict(text)
        return PredictResponse(priority=label, score=score, modelVersion=version)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
