
import os
import json
import base64
import numpy as np
from openai import OpenAI
from google import genai
from google.genai import types
from .evaluator import embed_text, cosine
from .utils import encode_image_to_base64, get_image_mime_type

class PromptOptimizer:
    def __init__(self, openai_api_key=None, gemini_api_key=None):
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.gemini_client = genai.Client(api_key=gemini_api_key or os.getenv("GEMINI_API_KEY"))

    def optimize(
        self,
        initial_prompt: str,
        target_json_output,
        input_images: list[str],
        iterations: int = 10,
        test_model: str = "gpt-4o-mini",
        improve_model: str = "gpt-5",
        test_model_provider: str = "openai",
        improve_model_provider: str = "openai"
    ):
        if isinstance(target_json_output, dict):
            target_json_output = json.dumps(target_json_output, indent=2)

        target_vec = embed_text(target_json_output, client=self.openai_client)
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = -1.0
        best_output = ""
        feedback_history = []
        previous_output = ""
        previous_score = 0.0

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
                
                current_feedback = self._generate_text(
                    model=improve_model,
                    provider=improve_model_provider,
                    system_instruction="You are an helpful AI assistant.",
                    prompt=feedback_instruction
                )
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
                current_prompt = self._generate_text(
                    model=improve_model,
                    provider=improve_model_provider,
                    system_instruction="You are an helpful AI assistant.",
                    prompt=instruction
                )
                print("improved prompt")

            raw_output = self._test_prompt(
                prompt=current_prompt,
                input_images=input_images,
                model=test_model,
                provider=test_model_provider
            )

            try:
                parsed = json.loads(raw_output)
                normalized_output = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                print("response was not a json")
                normalized_output = raw_output

            out_vec = embed_text(normalized_output, client=self.openai_client)
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

    def _generate_text(self, model: str, provider: str, system_instruction: str, prompt: str) -> str:
        if provider.lower() == "openai":
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        
        elif provider.lower() == "gemini":
            # Gemini Python SDK usually takes system instruction in generation config or just as context, 
            # but simpler is to just prepend/append or use the system_instruction arg if available.
            # Using the genai.Client (Google Gen AI SDK v1beta/v1)
            # The client.models.generate_content signature supports 'config'
            
            # For simplicity with the unified client used earlier:
            response = self.gemini_client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
                contents=[prompt]
            )
            return response.text.strip() if response.text else ""
            
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _test_prompt(self, prompt: str, input_images: list[str], model: str, provider: str) -> str:
        if provider.lower() == "openai":
            content_parts = [{"type": "text", "text": prompt}]
            for img_path in input_images:
                img_base64 = encode_image_to_base64(img_path)
                img_format = "jpeg" if img_path.lower().endswith(('.jpg', '.jpeg')) else "png"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{img_base64}"
                    }
                })

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
            )
            return response.choices[0].message.content or '{"error": "empty response"}'
        
        elif provider.lower() == "gemini":
            content_parts = []
            for img_path in input_images:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                mime_type = get_image_mime_type(img_path) 
                content_parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                )
            content_parts.append(prompt)
            
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=content_parts
            )
            # Handle cases where response might be blocked or empty
            if not response.text:
                 return '{"error": "empty response"}'
            return response.text
        
        else:
            raise ValueError(f"Unsupported model provider: {provider}. Use 'openai' or 'gemini'.")
