"""
Script to copy optimized prompts from Results folder to extract_worksheets/prompts
- Maps folder names to worksheet numbers from worksheets_to_process.json
- Appends JSON format instructions to each prompt
- Saves as {worksheet_number}.txt
"""

import json
from pathlib import Path


def sanitize_folder_name(name: str) -> str:
    """Apply same sanitization as used when creating Dataset folders"""
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(char, '_')
    return name.strip()


def main():
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    
    results_dir = project_root / "Results"
    prompts_output_dir = base_dir / "prompts"
    json_file = base_dir / "worksheets_to_process.json"
    
    # Create prompts folder if needed
    prompts_output_dir.mkdir(exist_ok=True)
    
    # Load worksheet mappings
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    worksheets = data.get("worksheets", {})
    print(f"Loaded reference to {len(worksheets)} worksheets")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {prompts_output_dir}")
    print("=" * 60)
    
    # JSON format instruction to append
    format_instruction = '''

Respond in the following JSON format, providing a list of all questions and their answers:
{format_instructions}'''
    
    copied_count = 0
    skipped_count = 0
    not_found_count = 0
    
    # Process each result folder
    for result_folder in results_dir.iterdir():
        if not result_folder.is_dir():
            continue
        
        folder_name = result_folder.name
        optimized_prompt_path = result_folder / "optimized_prompt.txt"
        
        # Skip if no optimized prompt
        if not optimized_prompt_path.exists():
            print(f"  No prompt: {folder_name}")
            skipped_count += 1
            continue
        
        # Folder name IS the worksheet number
        worksheet_number = folder_name
        
        # Read optimized prompt
        with open(optimized_prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        
        # Append format instruction
        final_content = prompt_content + format_instruction
        
        # Save to prompts folder
        output_file = prompts_output_dir / f"{worksheet_number}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"  {folder_name} -> {worksheet_number}.txt")
        copied_count += 1
    
    print("\n" + "=" * 60)
    print(f"Copied: {copied_count}")
    print(f"Skipped (no prompt): {skipped_count}")
    print(f"Not found (no mapping): {not_found_count}")
    print(f"\nPrompts saved to: {prompts_output_dir}")


if __name__ == "__main__":
    main()
