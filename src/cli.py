
import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .optimizer import PromptOptimizer
from .utils import infer_provider

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def discover_samples(dataset_path: str, results_path: str = "Results", folder_filters: set = None) -> list[dict]:
    samples = []
    skipped_count = 0

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return samples

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            continue
            
        # If filters are provided, only pick matching folders
        if folder_filters and folder_name not in folder_filters:
            continue

        # Skip already optimized folders
        optimized_prompt_path = os.path.join(results_path, folder_name, "optimized_prompt.txt")
        if os.path.exists(optimized_prompt_path):
            skipped_count += 1
            continue
        
        images_folder = os.path.join(folder_path, "images")
        outputs_folder = os.path.join(folder_path, "outputs")
        prompts_folder = os.path.join(folder_path, "prompts")
        
        if not os.path.exists(images_folder):
            continue
        if not os.path.exists(outputs_folder):
            continue
        if not os.path.exists(prompts_folder):
            continue
        
        prompt_files = [f for f in os.listdir(prompts_folder) if f.endswith('.txt')]
        if not prompt_files:
            print(f"Warning: No prompt files found in {prompts_folder}. Skipping.")
            continue
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        images = sorted([
            os.path.join(images_folder, img)
            for img in os.listdir(images_folder)
            if os.path.splitext(img)[1].lower() in image_extensions
        ])
        
        if not images:
            print(f"Warning: No images found in {images_folder}. Skipping.")
            continue
        
        output_files = [f for f in os.listdir(outputs_folder) if f.endswith('.json')]
        if not output_files:
            print(f"Warning: No output JSON files found in {outputs_folder}. Skipping.")
            continue
        
        prompt_file = os.path.join(prompts_folder, prompt_files[0])
        output_file = os.path.join(outputs_folder, output_files[0])
        
        samples.append({
            "name": folder_name,
            "folder_path": folder_path,
            "prompt_file": prompt_file,
            "output_file": output_file,
            "images": images
        })

    if skipped_count > 0:
        print(f"Skipped {skipped_count} already optimized folder(s)")

    return samples


def process_sample(sample: dict, args, test_provider: str, improve_provider: str) -> dict:
    sample_name = sample["name"]
    
    try:
        safe_print(f"\n Starting: {sample_name}")
        
        optimizer = PromptOptimizer()
        
        with open(sample["prompt_file"], "r", encoding="utf-8") as f:
            initial_prompt = f.read().strip()
        
        with open(sample["output_file"], "r", encoding="utf-8") as f:
            target_json = json.load(f)
        
        safe_print(f"  📷 {sample_name}: {len(sample['images'])} images, {len(initial_prompt)} chars prompt")
        
        best_prompt, best_score, best_output = optimizer.optimize(
            initial_prompt=initial_prompt,
            target_json_output=target_json,
            input_images=sample["images"],
            iterations=args.iterations,
            test_model=args.test_model,
            improve_model=args.improve_model,
            test_model_provider=test_provider,
            improve_model_provider=improve_provider
        )
        
        results_folder = os.path.join("Results", sample_name)
        os.makedirs(results_folder, exist_ok=True)
        
        prompt_output_path = os.path.join(results_folder, "optimized_prompt.txt")
        with open(prompt_output_path, "w", encoding="utf-8") as f:
            f.write(best_prompt)
        
        output_path = os.path.join(results_folder, "best_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(best_output)
        
        safe_print(f"✅ {sample_name}: score={best_score:.4f}")
        
        return {
            "name": sample_name,
            "success": True,
            "best_score": best_score,
            "results_folder": results_folder
        }
        
    except Exception as e:
        safe_print(f"❌ {sample_name}: Error - {e}")
        return {
            "name": sample_name,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="OCR Prompt Optimizer & Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Usage examples:
              python main.py --iterations 5
              python main.py --1123 --1124           (Process specific folders)
              python main.py --range-1120-1125       (Process a range of folders)
              python main.py --1100 --range-1200-1205 (Mix and match)

            Dataset Structure:
            Dataset/
            ├── sample1/
            │   ├── images/      (input images)
            │   ├── outputs/     (target JSON files)
            │   └── prompts/     (initial prompt text files)
            ├── sample2/
            │   ├── images/
            │   ├── outputs/
            │   └── prompts/
            └── ...

            Results are saved to:
            Results/
            ├── sample1/
            │   ├── best_output.json
            │   └── optimized_prompt.txt
            └── sample2/
                └── ...
        """
    )
    
    parser.add_argument("--dataset", type=str, default="Dataset", 
                        help="Path to dataset directory containing sample folders")
    parser.add_argument("--iterations", type=int, default=10, 
                        help="Number of optimization iterations per sample")
    parser.add_argument("--test-model", type=str, default="gpt-4o-mini", 
                        help="Student model for testing prompts (e.g., gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--improve-model", type=str, default="gpt-4o", 
                        help="Teacher model for improving prompts (e.g., gpt-4o, gemini-1.5-pro)")
    parser.add_argument("--workers", type=int, default=25, 
                        help="Number of parallel workers (default: 25)")
    
    # Parse known args, leaving dynamic folder flags in 'unknown'
    args, unknown = parser.parse_known_args()
    
    # Process dynamic folder flags like --1123 or --range-1120-1125
    selected_folders = set()
    for arg in unknown:
        if arg.startswith("--range-"):
            parts = arg.replace("--range-", "").split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start = int(parts[0])
                end = int(parts[1])
                for i in range(start, end + 1):
                    selected_folders.add(str(i))
            else:
                print(f"Warning: Ignoring invalid range format '{arg}'. Use --range-START-END")
        elif arg.startswith("--") and arg[2:].isdigit():
            selected_folders.add(arg[2:])
        else:
            print(f"Warning: Unrecognized argument ignored: {arg}")
    
    test_provider = infer_provider(args.test_model)
    improve_provider = infer_provider(args.improve_model)
    
    print(f"Test model: {args.test_model} (provider: {test_provider})")
    print(f"Improve model: {args.improve_model} (provider: {improve_provider})")
    print(f"Iterations per sample: {args.iterations}")
    print(f"Parallel workers: {args.workers}")
    if selected_folders:
        print(f"Specific folders selected: {sorted(list(selected_folders))}")
    print("-" * 50)
    
    # Pass the set of selected folders (if any) down to discover_samples
    samples = discover_samples(args.dataset, folder_filters=selected_folders if selected_folders else None)
    
    if not samples:
        print(f"No valid sample folders found in {args.dataset}")
        print("\nExpected structure:")
        print("  Dataset/")
        print("  └── your_sample_name/")
        print("      ├── images/    (put your images here)")
        print("      ├── outputs/   (put target JSON here)")
        print("      └── prompts/   (put initial prompt.txt here)")
        return
    
    print(f"Found {len(samples)} sample(s)")
    print("=" * 50)
    
    all_results = {}
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample = {
            executor.submit(process_sample, sample, args, test_provider, improve_provider): sample
            for sample in samples
        }
        
        for future in as_completed(future_to_sample):
            result = future.result()
            
            if result["success"]:
                all_results[result["name"]] = {
                    "best_score": result["best_score"],
                    "results_folder": result["results_folder"]
                }
                success_count += 1
            else:
                error_count += 1
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    
    for name, result in sorted(all_results.items(), key=lambda x: x[1]['best_score'], reverse=True):
        print(f"  {name}: score={result['best_score']:.4f}")
    
    print(f"\nSuccessful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"\nAll results saved in Results/ folder")


if __name__ == "__main__":
    main()
