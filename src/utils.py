
import os
import base64

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_image_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

def infer_provider(model_name: str) -> str:
    model_lower = model_name.lower()
    
    if "gemini" in model_lower:
        return "gemini"
    
    openai_prefixes = ["gpt", "o1", "o3", "text-", "davinci", "curie", "babbage", "ada"]
    for prefix in openai_prefixes:
        if model_lower.startswith(prefix):
            return "openai"
