
import os
import json
import re
import time
import threading
import numpy as np
from openai import OpenAI
from google import genai
from google.genai import types
from .evaluator import embed_text, cosine
from .utils import encode_image_to_base64, get_image_mime_type


class GeminiRateLimiter:
    """Rate limiter for Gemini API calls (25 requests/minute limit)."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Block until a request can be made within rate limits."""
        with self.lock:
            now = time.time()
            # Remove requests outside the window
            self.requests = [t for t in self.requests if now - t < self.window_seconds]

            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request falls outside the window
                wait_time = self.window_seconds - (now - self.requests[0]) + 0.5
                if wait_time > 0:
                    print(f"Rate limit reached, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.requests = [t for t in self.requests if now - t < self.window_seconds]

            self.requests.append(time.time())


# Global rate limiter for Gemini
_gemini_rate_limiter = GeminiRateLimiter()


def clean_json_response(text: str) -> str:
    if not text:
        return text
    
    text = text.strip()
    
    code_fence_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(code_fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
    
    if text.endswith('```'):
        text = text[:-3]
    
    text = text.strip()
    
    if not text.startswith(('[', '{')):
        array_start = text.find('[')
        object_start = text.find('{')
        
        if array_start != -1 and (object_start == -1 or array_start < object_start):
            text = text[array_start:]
        elif object_start != -1:
            text = text[object_start:]
    
    if text.startswith('['):
        bracket_count = 0
        for i, char in enumerate(text):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    text = text[:i + 1]
                    break
    elif text.startswith('{'):
        brace_count = 0
        for i, char in enumerate(text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    text = text[:i + 1]
                    break
    
    return text


class PromptOptimizer:
    def __init__(self, openai_api_key=None, gemini_api_key=None, openrouter_api_key=None):
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.gemini_client = genai.Client(api_key=gemini_api_key or os.getenv("GEMINI_API_KEY"))
        
        # OpenRouter uses OpenAI-compatible API
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        )

    def _call_gemini_with_retry(self, api_call, extract_text, max_retries: int = 5):
        """Call Gemini API with rate limiting and retry on 429 errors."""
        for attempt in range(max_retries):
            _gemini_rate_limiter.wait_if_needed()
            try:
                response = api_call()
                return extract_text(response)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Extract retry delay from error message if available
                    wait_time = 10  # default wait
                    if "retry in" in error_str.lower():
                        match = re.search(r'retry in (\d+\.?\d*)', error_str.lower())
                        if match:
                            wait_time = float(match.group(1)) + 1

                    if attempt < max_retries - 1:
                        print(f"Rate limited, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                raise

    def optimize(
        self,
        initial_prompt: str,
        target_json_output,
        input_images: list[str],
        iterations: int = 10,
        test_model: str = "gemini-2.0-flash",
        improve_model: str = "gemini-3-pro-preview",
        test_model_provider: str = 'gemini',
        improve_model_provider: str = 'gemini'
    ):
        if isinstance(target_json_output, dict) or isinstance(target_json_output, list):
            target_json_output_str = json.dumps(target_json_output, indent=2)
        else:
            target_json_output_str = str(target_json_output)

        target_vec = embed_text(target_json_output_str)
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = -1.0
        best_output = ""
        feedback_history = []
        previous_output = ""
        previous_score = 0.0

        print("\n--- Pre-evaluating Worksheet Context ---")
        context_eval_instruction = """
        You are an expert educational analyst. Look at the provided worksheet images and its target JSON output.
        Identify the following key factors to help write better OCR instructions:
        1. **Question Type:** (e.g., addition, sequence, matching, word problem)
        2. **Examples Present:** Are there any solved examples on the page? How are they formatted?
        3. **Answer Format:** How does the student answer? (e.g., fill in the blanks, circling, drawing lines, writing in boxes)
        4. **Structural Quirks:** Any unique layout features (e.g., vertical columns, scattered text, specific numbering format like 'Q1.')
        
        Keep your assessment concise, accurate, and bulleted.
        """
        
        worksheet_context = self._test_prompt(
            prompt=context_eval_instruction + f"\n\nTARGET JSON:\n{target_json_output_str}",
            input_images=input_images,
            model=improve_model, # Use the bigger model for evaluation
            provider=improve_model_provider
        )
        print("Worksheet context evaluated.")
        print("-" * 30)
        print(worksheet_context)
        print("-" * 30)

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
                        {target_json_output_str}
                        
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
                        {target_json_output_str}
                        
                        ---
                        WORKSHEET CONTEXT & FEATURES (Identified by an Analyst):
                        {worksheet_context}
                        ---
    
                        CURRENT PROMPT:
                        {current_prompt}
    
                        MODEL OUTPUT (from previous iteration):
                        {previous_output}
    
                        PREVIOUS SCORE: {previous_score:.4f}
                        
                        FEEDBACK FROM ANALYSIS:
                        {all_feedback}
    
                        Based on the feedback and the worksheet context, improve the prompt by:
                        - addressing specific issues identified in the feedback
                        - making instructions clearer and more precise based on the Worksheet Context (e.g. explicitly mentioning how to handle 'fill in the blanks' or 'examples' if present)
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
                print("Improved prompt generated")

            raw_output = self._test_prompt(
                prompt=current_prompt,
                input_images=input_images,
                model=test_model,
                provider=test_model_provider
            )

            cleaned_output = clean_json_response(raw_output)

            try:
                parsed = json.loads(cleaned_output)
                normalized_output = json.dumps(parsed, indent=2)
                print("Valid JSON response")
            except json.JSONDecodeError as e:
                print(f"Response was not valid JSON: {str(e)[:50]}")
                # Still use cleaned output for scoring, but mark it
                normalized_output = cleaned_output

            out_vec = embed_text(normalized_output)
            score = cosine(out_vec, target_vec)
            print(f"Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_prompt = current_prompt
                best_output = normalized_output
                print(f"New best score: {best_score:.4f}")

            print(f"Best score so far: {best_score:.4f}")

            if score >= 1.0:
                print("Perfect score achieved! Stopping early.")
                break

            previous_output = normalized_output
            previous_score = score

        final_optimized_prompt = f"{best_prompt}\n\n=== WORKSHEET CONTEXT NOTES ===\n{worksheet_context}\n===============================\n"
        
        return final_optimized_prompt, best_score, best_output

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
            return self._call_gemini_with_retry(
                lambda: self.gemini_client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                    contents=[prompt]
                ),
                extract_text=lambda r: r.text.strip() if r.text else ""
            )

        elif provider.lower() == "openrouter":
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        
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

            return self._call_gemini_with_retry(
                lambda: self.gemini_client.models.generate_content(
                    model=model,
                    contents=content_parts
                ),
                extract_text=lambda r: r.text if r.text else '{"error": "empty response"}'
            )

        elif provider.lower() == "openrouter":
            # OpenRouter uses same format as OpenAI for vision
            content_parts = [{"type": "text", "text": prompt}]
            for img_path in input_images:
                img_base64 = encode_image_to_base64(img_path)
                mime_type = get_image_mime_type(img_path)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_base64}"
                    }
                })

            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
            )
            return response.choices[0].message.content or '{"error": "empty response"}'
        
        else:
            raise ValueError(f"Unsupported model provider: {provider}. Use 'openai', 'gemini', or 'openrouter'.")
