import os
import json
import base64
import numpy as np
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_text(text: str):
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

def cosine(a: np.ndarray, b: np.ndarray):
    return float(np.dot(a, b))

def optimize_ocr_prompt(
    initial_prompt: str,
    target_json_output,
    input_images: list[str],
    iterations: int = 10,
    test_model: str = "gpt-4o-mini",
    improve_model: str = "gpt-5",
    test_model_provider: str = "openai"
):
    if isinstance(target_json_output, dict):
        target_json_output = json.dumps(target_json_output, indent=2)

    target_vec = embed_text(target_json_output)
    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_score = -1.0
    best_output = ""
    feedback_history = []

    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it} ---")
        if it > 1:
            feedback_instruction = f"""
                    You are an OCR prompt analysis expert.
                    
                    Compare the TARGET JSON with the MODEL OUTPUT and provide specific feedback on:
                    1. What was missing or incorrect in the output
                    2. What was done correctly
                    3. Specific recommendations for prompt improvements
                    
                    TARGET JSON:
                    {target_json_output}
                    
                    MODEL OUTPUT:
                    {previous_output}
                    
                    SCORE: {previous_score:.4f}
                    
                    Provide concise, actionable feedback (2-3 bullet points).
                    """
            
            feedback_resp = client.chat.completions.create(
                model=improve_model,
                messages=[
                    {"role": "system", "content": feedback_instruction}
                ]
            )
            current_feedback = feedback_resp.choices[0].message.content.strip()
            feedback_history.append(f"Iteration {it-1} Feedback:\n{current_feedback}")
            
            all_feedback = "\n\n".join(feedback_history)
            instruction = f"""
                    You are an OCR prompt optimization expert.

                    Goal:
                    Improve the prompt so a weaker OCR-model produces JSON exactly matching:

                    TARGET JSON:
                    {target_json_output}

                    CURRENT PROMPT:
                    {current_prompt}

                    MODEL OUTPUT (from previous iteration):
                    {previous_output}

                    PREVIOUS SCORE: {previous_score:.4f}
                    
                    FEEDBACK FROM ANALYSIS:
                    {all_feedback}

                    Based on the feedback, improve the prompt by:
                    - addressing specific issues identified in the feedback
                    - making instructions clearer and more precise
                    - enforcing JSON schema and structure
                    - improving extraction rules for text, numbers, layout
                    - reducing chance of hallucination

                    Return ONLY the improved prompt (no commentary).
                    """
            improved_resp = client.chat.completions.create(
                model=improve_model,
                messages=[
                    {"role": "system", "content": instruction}
                ]
            )
            current_prompt = improved_resp.choices[0].message.content.strip()
            print("improved prompt")

        if test_model_provider.lower() == "openai":
            content_parts = [
                {"type": "text", "text": current_prompt}
            ]
            for img_path in input_images:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                ext = os.path.splitext(img_path)[1].lower()
                img_format = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
                
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{img_base64}"
                    }
                })

            response = client.chat.completions.create(
                model=test_model,
                messages=[
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
            )
            raw_output = response.choices[0].message.content
        
        elif test_model_provider.lower() == "gemini":
            content_parts = []
            
            for img_path in input_images:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                mime_type = 'image/jpeg' if img_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                content_parts.append(
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type=mime_type
                    )
                )
            
            content_parts.append(current_prompt)
            
            response = gemini_client.models.generate_content(
                model=test_model,
                contents=content_parts
            )
            raw_output = response.text
        
        else:
            raise ValueError(f"Unsupported model provider: {test_model_provider}. Use 'openai' or 'gemini'.")
        
        if not raw_output or not raw_output.strip():
            raw_output = '{"error": "empty response"}'

        try:
            parsed = json.loads(raw_output)
            normalized_output = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            print("response was not a json")
            normalized_output = raw_output

        out_vec = embed_text(normalized_output)
        score = cosine(out_vec, target_vec)
        print(f"Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_prompt = current_prompt
            best_output = normalized_output
            print(f"new best score: {best_score:.4f}")

        print(f"best Score So Far: {best_score:.4f}")

        previous_output = normalized_output
        previous_score = score

    return best_prompt, best_score, best_output

def load_dataset_and_optimize(dataset_path="Dataset", iterations=10, test_model="gpt-4o-mini", test_model_provider="openai"):
    prompts_folder = os.path.join(dataset_path, "prompts")
    outputs_folder = os.path.join(dataset_path, "outputs")
    images_folder = os.path.join(dataset_path, "images")
    os.makedirs(os.path.join("Results", "optimized_prompts"), exist_ok=True)
    os.makedirs(os.path.join("Results", "best_outputs"), exist_ok=True)

    results = {}

    for fname in os.listdir(prompts_folder):
        if not fname.endswith(".txt"):
            continue
        sample_id = fname[:-4]
        print(f"processing: {sample_id}")
        with open(os.path.join(prompts_folder, fname), "r", encoding="utf-8") as f:
            initial_prompt = f.read().strip()

        with open(os.path.join(outputs_folder, f"{sample_id}.json"), "r", encoding="utf-8") as f:
            target_json = json.load(f)

        input_images = sorted([
            os.path.join(images_folder, img)
            for img in os.listdir(images_folder)
            if img.startswith(f"{sample_id}-")
        ])

        best_prompt, best_score, best_output = optimize_ocr_prompt(
            initial_prompt=initial_prompt,
            target_json_output=target_json,
            input_images=input_images,
            iterations=iterations,
            test_model=test_model,
            test_model_provider=test_model_provider
        )
        
        with open(os.path.join("Results", "optimized_prompts", f"{sample_id}_optimized_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(best_prompt)
        
        with open(os.path.join("Results", "best_outputs", f"{sample_id}_best_output.json"), "w", encoding="utf-8") as f:
            f.write(best_output)
        
        results[sample_id] = {
            "best_prompt": best_prompt,
            "best_score": best_score,
            "best_output": best_output
        }

    return results

if __name__ == "__main__":
    final = load_dataset_and_optimize(
        dataset_path="Dataset", 
        iterations=2,
        test_model="gemini-2.5-flash-lite",
        test_model_provider="gemini"
    )

    print(json.dumps(final, indent=2))