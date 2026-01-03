from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import PredictRequest, PredictResponse
from app.baseline.predictor import BaselinePredictor
import logging
import traceback


baseline = BaselinePredictor()
logger = logging.getLogger("inference")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    baseline.load()
    yield


app = FastAPI(
    title="REMSFAL Inference Service",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "baselineReady": baseline.is_ready(),
        "modelVersion": baseline.model_version
    }


@app.post("/predict/baseline", response_model=PredictResponse)
def predict_baseline(req: PredictRequest):
    title = (req.title or "").strip()
    desc = (req.description or "").strip() if req.description else ""
    text = f"{title}\n{desc}".strip()

    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Text too short for classification")

    try:
        label, score, version = baseline.predict(text)
        return PredictResponse(priority=label, score=score, modelVersion=version)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Unexpected inference error: %s", str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected inference error")

