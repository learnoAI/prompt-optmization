"""
Script to create Dataset folders from worksheets_to_process.json
- Creates folder for each worksheet type
- Downloads images from S3 URLs
- Copies a base prompt to each folder
"""

import os
import json
import requests
import threading
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def download_image(url: str, save_path: Path) -> bool:
    """Download an image from URL and save to path"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        safe_print(f"  ✓ Downloaded: {save_path.name}")
        return True
    except Exception as e:
        safe_print(f"  ✗ Failed to download {url}: {e}")
        return False

def get_filename_from_url(url: str, index: int) -> str:
    """Extract filename from URL or generate one"""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename or not filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        filename = f"image_{index}.jpg"
    return filename

def process_worksheet(worksheet_number: str, info: dict, dataset_dir: Path, base_prompt: str) -> str:
    """Process a single worksheet (create folders, download images, write prompt). Returns status string."""
    folder_name = str(worksheet_number).strip()
    folder_path = dataset_dir / folder_name
    
    # Skip if folder already exists
    if folder_path.exists():
        return f"⏭ Skipped (exists): {folder_name}"
    
    safe_print(f"\n📁 Creating: {folder_name}")
    
    # Create folder structure
    images_folder = folder_path / "images"
    prompts_folder = folder_path / "prompts"
    outputs_folder = folder_path / "outputs"
    
    images_folder.mkdir(parents=True, exist_ok=True)
    prompts_folder.mkdir(parents=True, exist_ok=True)
    outputs_folder.mkdir(parents=True, exist_ok=True)
    
    # Download images
    s3_urls = info.get("s3_urls", [])
    for idx, url in enumerate(s3_urls):
        filename = get_filename_from_url(url, idx)
        save_path = images_folder / filename
        download_image(url, save_path)
    
    # Write the base prompt
    prompt_file = prompts_folder / "prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(base_prompt)
    safe_print(f"  ✓ Created prompt.txt for {folder_name}")
    
    # Create empty placeholder output JSON
    output_file = outputs_folder / "expected.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("[]")
    safe_print(f"  ✓ Created expected.json placeholder for {folder_name}")
    
    return f"✅ Created: {folder_name}"

def main():
    # Options
    MAX_WORKERS = 25
    
    # Paths
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    json_file = base_dir / "worksheets_to_process.json"
    dataset_dir = project_root / "Dataset"
    
    # Base prompt to use for all folders
    base_prompt = """You are a specialized OCR system designed to extract handwritten math worksheet data into a strict JSON format.

**TASK:**
Extract every question and its corresponding student answer from the image provided.

**JSON SCHEMA ENFORCEMENT:**
Respond in the following JSON format, providing a list of all questions and their answers:
{format_instructions}

**DO NOT** use dynamic keys (like "Q1", "A1", "Q_2", etc.).
**DO NOT** nest the objects.
**KEYS MUST BE:** `"question_number"`, `"question"`, `"student_answer"`.

**FORMATTING RULES:**
1.  **Question Field:**
    - Must include the label (e.g., "Q1.", "Q2.") at the start.
    - Must flatten vertical text into a single horizontal line.
    - **NO** newline characters in the string.
2.  **Student Answer Field:**
    - Extract exactly what the student wrote in pencil.
    - If the answer slot is blank, use an empty string `""`.
    - Return the answer as a string.

**HANDWRITING EXTRACTION GUIDELINES:**
- Look closely at digit formation.
- Distinguish **9** vs **8**, **6** vs **0**, **1** vs **7** vs **4**.
- If a number is messy, select the most likely digit based on visual stroke evidence.

**PROCESSING ORDER:**
- Sort the output by `question_number` ascending.
- Handle multiple columns correctly by tracking the Question IDs.
"""
    
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    worksheets = data.get("worksheets", {})
    
    print(f"Found {len(worksheets)} worksheets to process")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Processing in parallel with {MAX_WORKERS} workers")
    print("=" * 60)
    
    created_count = 0
    skipped_count = 0
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ws = {
            executor.submit(process_worksheet, ws_num, info, dataset_dir, base_prompt): ws_num
            for ws_num, info in worksheets.items()
        }
        
        for future in as_completed(future_to_ws):
            try:
                result = future.result()
                if "Skipped" in result:
                    skipped_count += 1
                elif "✅ Created" in result:
                    created_count += 1
            except Exception as e:
                safe_print(f"❌ Exception processing a worksheet: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Created: {created_count} folders")
    print(f"⏭ Skipped: {skipped_count} folders (already exist)")
    print(f"📂 All folders saved to: {dataset_dir}")

if __name__ == "__main__":
    main()
