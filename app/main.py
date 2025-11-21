# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import torch
# from app.core.config import DEVICE, LABEL_COLS
from typing import List
from app.schemas.api_models import (
    CommentRequest,
    CommentResponse,
    PredictionResult,
    HealthResponse,
    BatchRequest,
)
from app.utils.text_cleaning import clean_text

# Initialize FastAPI app
app = FastAPI(
    title="Toxic Comment Classification API",
    description="Multi-label classification API for detecting toxic comments",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Define label-wise thresholds based on evaluation metrics
LABEL_THRESHOLDS = {
    'toxic': 0.5,
    'severe_toxic': 0.6,
    'obscene': 0.6,
    'threat': 0.6,
    'insult': 0.5,
    'identity_hate': 0.6
}

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup"""
    global model, tokenizer, pad_idx
    
    try:
        # Define model paths
        model_path = Path("models/gru_model.pt")
        tokenizer_vocab_path = Path("models/bpe_tokenizer-vocab.json")
        tokenizer_merges_path = Path("models/bpe_tokenizer-merges.txt")
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not tokenizer_vocab_path.exists():
            raise FileNotFoundError(f"Tokenizer vocab file not found: {tokenizer_vocab_path}")
        if not tokenizer_merges_path.exists():
            raise FileNotFoundError(f"Tokenizer merges file not found: {tokenizer_merges_path}")
        
        # Load tokenizer
        tokenizer = ByteLevelBPETokenizer(
            str(tokenizer_vocab_path),
            str(tokenizer_merges_path)
        )
        pad_idx = tokenizer.token_to_id("<PAD>")
        vocab_size = tokenizer.get_vocab_size()
        
        # Load the scripted model directly
        model = torch.jit.load(str(model_path), map_location=DEVICE)
        model.eval()
        
        print(f"✓ Model loaded successfully on {DEVICE}")
        print(f"✓ Vocabulary size: {vocab_size}")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "device": str(DEVICE),
    }


def build_explanation(predictions: List[PredictionResult], is_toxic: bool) -> str:
    """Build sentence explanation using labels above threshold."""
    strong_labels = [p.label.replace("_", " ") for p in predictions if p.is_positive]

    if not strong_labels:
        return "The comment appears non-toxic. No significant toxic categories detected."

    if len(strong_labels) == 1:
        return f"This comment is predicted to be '{strong_labels[0]}'."
    else:
        sentence = ", ".join(strong_labels[:-1]) + " and " + strong_labels[-1]
        return f"This comment is predicted to exhibit multiple toxic behaviors including {sentence}."



@app.post("/predict", response_model=CommentResponse)
async def predict_toxicity(request: CommentRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleaned_comment = clean_text(request.comment)
    if not cleaned_comment:
        raise HTTPException(status_code=400, detail="Comment is empty after cleaning")

    encoded = tokenizer.encode(cleaned_comment)
    input_tensor = torch.tensor(encoded.ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    predictions = []
    is_toxic = False

    for label, prob in zip(LABEL_COLS, probs):
        threshold = LABEL_THRESHOLDS.get(label, 0.5)
        is_positive = float(prob) >= threshold
        if is_positive:
            is_toxic = True
        predictions.append(PredictionResult(
            label=label,
            probability=round(float(prob), 4),
            is_positive=is_positive
        ))

    explanation = build_explanation(predictions, is_toxic)

    return CommentResponse(
        comment = request.comment,
        predictions=predictions,
        is_toxic=is_toxic,
        explanation=explanation
    )

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for comment in request.comments:
        cleaned_comment = clean_text(comment)
        if not cleaned_comment:
            results.append({
                "original_comment": comment,
                "error": "Empty after cleaning"
            })
            continue

        encoded = tokenizer.encode(cleaned_comment)
        input_tensor = torch.tensor(encoded.ids).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        predictions = []
        is_toxic = False

        for label, prob in zip(LABEL_COLS, probs):
            threshold = LABEL_THRESHOLDS.get(label, 0.5)
            is_positive = float(prob) >= threshold
            if is_positive:
                is_toxic = True
            predictions.append({
                "label": label,
                "probability": round(float(prob), 4),
                "is_positive": is_positive
            })

        preds_objects = [PredictionResult(**p) for p in predictions]
        explanation = build_explanation(preds_objects, is_toxic)

        results.append({
            "original_comment": comment,
            "predictions": predictions,
            "is_toxic": is_toxic,
            "explanation": explanation
        })

    return {
        "total_comments": len(request.comments),
        "results": results
    }

@app.get("/model/info")
async def model_info():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "GRU (TorchScript)",
        "vocabulary_size": tokenizer.get_vocab_size(),
        "labels": LABEL_COLS,
        "device": str(DEVICE),
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": False,
    }

# --------------------- RUN ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
