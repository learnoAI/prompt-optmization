"""
Script to generate missing output JSONs using Gemini 3 Pro with thinking.
- Scans Dataset folders for empty/placeholder outputs
- Uses Gemini 3 Pro with max thinking to generate ground truth JSONs
- Saves generated outputs to the outputs folder
- Runs in PARALLEL for faster processing
"""

import os
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def get_folders_needing_output(dataset_dir: Path) -> list[Path]:
    """Find all folders that have empty or placeholder output JSONs"""
    folders_needing_output = []
    
    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue
        
        outputs_folder = folder / "outputs"
        images_folder = folder / "images"
        prompts_folder = folder / "prompts"
        
        # Skip if missing required structure
        if not images_folder.exists() or not prompts_folder.exists():
            continue
        
        # Check if output is empty or just placeholder
        needs_output = True
        if outputs_folder.exists():
            for output_file in outputs_folder.glob("*.json"):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content and content != "[]" and content != "{}":
                            # Has valid content
                            data = json.loads(content)
                            if data and len(data) > 0:
                                needs_output = False
                                break
                except (json.JSONDecodeError, Exception):
                    pass
        
        if needs_output:
            folders_needing_output.append(folder)
    
    return folders_needing_output


def load_images(images_folder: Path) -> list[tuple[bytes, str]]:
    """Load all images from folder as bytes with mime types"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    images = []
    
    for img_file in sorted(images_folder.iterdir()):
        if img_file.suffix.lower() in image_extensions:
            with open(img_file, 'rb') as f:
                img_bytes = f.read()
            
            ext = img_file.suffix.lower()
            mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else f"image/{ext[1:]}"
            images.append((img_bytes, mime_type))
    
    return images


def load_prompt(prompts_folder: Path) -> str:
    """Load the first prompt file from prompts folder"""
    for prompt_file in prompts_folder.glob("*.txt"):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def generate_output_with_gemini(client: genai.Client, images: list[tuple[bytes, str]], prompt: str) -> str:
    """
    Generate output JSON using Gemini 3 Pro with maximum thinking.
    Includes rate limit handling and exponential backoff.
    """
    content_parts = []
    for img_bytes, mime_type in images:
        content_parts.append(
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        )
    content_parts.append(prompt)
    
    max_retries = 6
    base_delay = 5  # Start with a 5 second delay for 429s
    
    for attempt in range(max_retries):
        try:
            # Generate with thinking enabled
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=content_parts,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="HIGH"
                    )
                )
            )
            return response.text if response.text else ""
            
        except APIError as e:
            if e.code == 429:
                if attempt == max_retries - 1:
                    safe_print(f"  ❌ Max retries reached for 429 Resource Exhausted.")
                    raise e
                
                # Exponential backoff with jitter
                delay = (base_delay * (2 ** attempt)) + random.uniform(0, 2)
                safe_print(f"  ⏳ Rate limit hit (429). Retrying in {delay:.1f} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                # Re-raise other API errors immediately
                raise e
        except Exception as e:
             raise e


def clean_json_response(text: str) -> str:
    """Clean markdown code fences and extract JSON"""
    if not text:
        return text
    
    text = text.strip()
    
    # Remove code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Find JSON start
    if not text.startswith(('[', '{')):
        array_start = text.find('[')
        object_start = text.find('{')
        
        if array_start != -1 and (object_start == -1 or array_start < object_start):
            text = text[array_start:]
        elif object_start != -1:
            text = text[object_start:]
    
    return text


def process_folder(folder: Path, client: genai.Client) -> tuple[str, bool, str]:
    """
    Process a single folder - returns (folder_name, success, message)
    """
    folder_name = folder.name
    
    try:
        # Load images
        images_folder = folder / "images"
        images = load_images(images_folder)
        
        if not images:
            return (folder_name, False, "No images found")
        
        # Load prompt
        prompts_folder = folder / "prompts"
        prompt = load_prompt(prompts_folder)
        
        if not prompt:
            return (folder_name, False, "No prompt found")
        
        # Generate output with Gemini
        raw_output = generate_output_with_gemini(client, images, prompt)
        
        # Clean and validate JSON
        cleaned_output = clean_json_response(raw_output)
        
        item_count = 0
        try:
            parsed = json.loads(cleaned_output)
            formatted_output = json.dumps(parsed, indent=2)
            item_count = len(parsed) if isinstance(parsed, list) else 1
        except json.JSONDecodeError:
            formatted_output = cleaned_output
        
        # Save output
        outputs_folder = folder / "outputs"
        outputs_folder.mkdir(exist_ok=True)
        
        output_file = outputs_folder / "expected.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        return (folder_name, True, f"Generated {item_count} items")
        
    except Exception as e:
        return (folder_name, False, str(e))


def main():
    # Configuration
    # Reduced max workers to avoid hitting limits too quickly
    MAX_WORKERS = 5  
    
    # Setup
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Find folders needing output
    folders = get_folders_needing_output(dataset_dir)
    print(f"Found {len(folders)} folders needing output generation")
    print(f"Processing with {MAX_WORKERS} parallel workers")
    print("=" * 60)
    
    if not folders:
        print("All folders have valid outputs. Nothing to do!")
        return
    
    success_count = 0
    error_count = 0
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_folder = {
            executor.submit(process_folder, folder, client): folder 
            for folder in folders
        }
        
        # Process results as they complete
        for future in as_completed(future_to_folder):
            folder_name, success, message = future.result()
            
            if success:
                safe_print(f"✅ {folder_name}: {message}")
                success_count += 1
            else:
                safe_print(f"❌ {folder_name}: {message}")
                error_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully generated: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📂 Outputs saved in Dataset/*/outputs/expected.json")


if __name__ == "__main__":
    main()
