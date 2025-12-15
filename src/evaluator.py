
import numpy as np
from fastembed import TextEmbedding

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model

def embed_text(text: str) -> np.ndarray:
    if not isinstance(text, str):
        text = str(text)
    
    if not text or not text.strip():
        text = "empty response"
    
    model = get_embedding_model()
    embeddings = list(model.embed([text]))
    vec = np.array(embeddings[0])
    return vec / (np.linalg.norm(vec) + 1e-8)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
