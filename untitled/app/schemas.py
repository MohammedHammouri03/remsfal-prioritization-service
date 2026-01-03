from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)

class PredictResponse(BaseModel):
    priority: str
    score: float
    modelVersion: str
