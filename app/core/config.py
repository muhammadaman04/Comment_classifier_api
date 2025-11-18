# app/core/config.py
from pathlib import Path
import torch

# Label categories
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for model and tokenizer
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "gru_model.pt"
TOKENIZER_VOCAB_PATH = MODELS_DIR / "bpe_tokenizer-vocab.json"
TOKENIZER_MERGES_PATH = MODELS_DIR / "bpe_tokenizer-merges.txt"
