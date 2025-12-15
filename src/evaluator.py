
import os
import numpy as np
from openai import OpenAI

def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str, client=None) -> np.ndarray:
    if client is None:
        client = get_openai_client()

    if not isinstance(text, str):
        text = str(text)
    
    if not text or not text.strip():
        text = "empty response"
    
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    vec = np.array(resp.data[0].embedding)
    return vec / (np.linalg.norm(vec) + 1e-8)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
