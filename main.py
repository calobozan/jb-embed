"""
Text embedding service using sentence-transformers.
Persistent mode - model stays loaded between calls.
"""

import json
import os
import logging

# Suppress progress bars and verbose logging during model load
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from sentence_transformers import SentenceTransformer

# Global state - model loaded once, reused for all calls
_model = None
_model_name = "all-MiniLM-L6-v2"


def _ensure_model():
    """Load model if not already loaded."""
    global _model
    if _model is None:
        # show_progress_bar=False suppresses tqdm during encoding
        _model = SentenceTransformer(_model_name)
    return _model


def embed(text: str) -> str:
    """Generate embedding for a single text."""
    model = _ensure_model()
    embedding = model.encode(text, convert_to_numpy=True)
    
    return json.dumps({
        "embedding": embedding.tolist(),
        "dimensions": len(embedding)
    })


def embed_batch(texts: list) -> str:
    """Generate embeddings for multiple texts."""
    model = _ensure_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    return json.dumps({
        "embeddings": embeddings.tolist(),
        "count": len(embeddings)
    })


def info() -> str:
    """Get model information."""
    model = _ensure_model()
    dim = model.get_sentence_embedding_dimension()
    
    return json.dumps({
        "model": _model_name,
        "dimensions": dim
    })


def health() -> str:
    """Health check - verifies model is loaded and working."""
    try:
        model = _ensure_model()
        # Quick test embedding
        _ = model.encode("health check", convert_to_numpy=True)
        return json.dumps({"status": "ok"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
