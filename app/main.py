# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
import json
from typing import List
import asyncio

# Import your schemas
from app.schemas.api_models import (
    CommentRequest,
    CommentResponse,
    PredictionResult,
    HealthResponse,
    BatchRequest,
)
from app.utils.text_cleaning import clean_text

# IMPORT MONITORING ROUTES
from app.monitoring.routes import router as monitoring_router

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

# INCLUDE MONITORING ROUTES
app.include_router(monitoring_router)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Define label-wise thresholds based on evaluation metrics
LABEL_THRESHOLDS = {
    'toxic': 0.5,        
    'severe_toxic': 0.4, 
    'obscene': 0.6,      
    'threat': 0.4,       
    'insult': 0.5,     
    'identity_hate': 0.4 
}

# Global variables
model = None
vocab = None
pad_idx = 0

# Import monitoring function
from app.monitoring.monitoring import track_prediction_for_monitoring

# Text preprocessing functions
def tokenize(text):
    return text.split()

def numericalize(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in tokenize(text)]

@app.on_event("startup")
async def load_model():
    """Load the model and vocabulary on startup"""
    global model, vocab, pad_idx
    
    try:
        # Define model paths
        model_path = Path("models/gru_model_pe.pt")
        vocab_path = Path("models/vocab.json")
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        # Load vocabulary from JSON
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        pad_idx = vocab.get("<PAD>", 0)
        
        # Load the scripted model directly
        model = torch.jit.load(str(model_path), map_location=DEVICE)
        model.eval()
        
        print(f"✓ Model loaded successfully on {DEVICE}")
        print(f"✓ Vocabulary size: {len(vocab)}")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and vocab is not None,
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
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleaned_comment = clean_text(request.comment)
    if not cleaned_comment:
        raise HTTPException(status_code=400, detail="Comment is empty after cleaning")

    # Track for monitoring (non-blocking)
    asyncio.create_task(track_prediction_for_monitoring(request.comment, vocab))
    
    tokens = numericalize(cleaned_comment, vocab)
    if not tokens:
        raise HTTPException(status_code=400, detail="No valid tokens after processing")

    input_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

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
        comment=request.comment,
        predictions=predictions,
        is_toxic=is_toxic,
        explanation=explanation
    )

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    if model is None or vocab is None:
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

        # Track for monitoring
        asyncio.create_task(track_prediction_for_monitoring(comment, vocab))
        
        tokens = numericalize(cleaned_comment, vocab)
        if not tokens:
            results.append({
                "original_comment": comment,
                "error": "No valid tokens after processing"
            })
            continue

        input_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

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
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Import here to avoid circular imports
    from app.monitoring.monitoring import vocab_monitor
    
    monitoring_stats = vocab_monitor.get_stats()
    
    return {
        "model_type": "GRU with Pretrained Embeddings",
        "vocabulary_size": len(vocab),
        "labels": LABEL_COLS,
        "device": str(DEVICE),
        "embedding_dim": 300,
        "monitoring": {
            "total_predictions": monitoring_stats["daily_stats"]["total_predictions"],
            "new_word_ratio": monitoring_stats["daily_stats"]["new_word_ratio"],
            "unknown_words_count": monitoring_stats["daily_stats"]["unknown_vocabulary_size"],
            "status": monitoring_stats["alert_status"]
        }
    }

# --------------------- RUN ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)