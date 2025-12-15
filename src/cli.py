
import os
import json
import argparse
from .optimizer import PromptOptimizer
from .utils import infer_provider


def discover_samples(dataset_path: str) -> list[dict]:
    samples = []
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return samples
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.isdir(folder_path):
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
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="OCR Prompt Optimizer & Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    
    args = parser.parse_args()
    
    test_provider = infer_provider(args.test_model)
    improve_provider = infer_provider(args.improve_model)
    
    print(f"Test model: {args.test_model} (provider: {test_provider})")
    print(f"Improve model: {args.improve_model} (provider: {improve_provider})")
    print(f"Iterations per sample: {args.iterations}")
    print("-" * 50)
    
    samples = discover_samples(args.dataset)
    
    if not samples:
        print(f"No valid sample folders found in {args.dataset}")
        print("\nExpected structure:")
        print("  Dataset/")
        print("  └── your_sample_name/")
        print("      ├── images/    (put your images here)")
        print("      ├── outputs/   (put target JSON here)")
        print("      └── prompts/   (put initial prompt.txt here)")
        return
    
    print(f"Found {len(samples)} sample(s): {[s['name'] for s in samples]}")
    print("=" * 50)
    
    optimizer = PromptOptimizer()
    all_results = {}
    
    for sample in samples:
        sample_name = sample["name"]
        print(f"\n{'='*50}")
        print(f"Processing: {sample_name}")
        print(f"{'='*50}")
        
        # Load prompt
        with open(sample["prompt_file"], "r", encoding="utf-8") as f:
            initial_prompt = f.read().strip()
        
        # Load target output
        with open(sample["output_file"], "r", encoding="utf-8") as f:
            target_json = json.load(f)
        
        print(f"Images: {len(sample['images'])}")
        print(f"Prompt length: {len(initial_prompt)} chars")
        
        # Run optimization
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
        
        # Create results folder for this sample
        results_folder = os.path.join("Results", sample_name)
        os.makedirs(results_folder, exist_ok=True)
        
        # Save optimized prompt
        prompt_output_path = os.path.join(results_folder, "optimized_prompt.txt")
        with open(prompt_output_path, "w", encoding="utf-8") as f:
            f.write(best_prompt)
        
        # Save best output
        output_path = os.path.join(results_folder, "best_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(best_output)
        
        all_results[sample_name] = {
            "best_score": best_score,
            "results_folder": results_folder
        }
        
        print(f"\nResults saved to: {results_folder}/")
        print(f"  - optimized_prompt.txt")
        print(f"  - best_output.json")
        print(f"Final score: {best_score:.4f}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    
    for name, result in all_results.items():
        print(f"  {name}: score={result['best_score']:.4f}")
    
    print(f"\nAll results saved in Results/ folder")


if __name__ == "__main__":
    main()
