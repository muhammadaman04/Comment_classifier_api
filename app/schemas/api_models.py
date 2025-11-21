# app/schemas/api_models.py
from pydantic import BaseModel, Field
from typing import List

class CommentRequest(BaseModel):
    comment: str = Field(..., description="The comment text to classify", min_length=1)

class PredictionResult(BaseModel):
    label: str
    probability: float
    is_positive: bool

class CommentResponse(BaseModel):
    comment: str
    predictions: List[PredictionResult]
    is_toxic: bool
    explanation: str   # 🔹 Added text explanation field

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class BatchRequest(BaseModel):
    comments: List[str] = Field(..., min_items=1, max_items=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
