
import os
import json
import argparse
from .optimizer import PromptOptimizer
from .utils import infer_provider

def main():
    parser = argparse.ArgumentParser(description="OCR Prompt Optimizer & Evaluator")
    
    parser.add_argument("--dataset", type=str, default="Dataset", help="Path to dataset directory")
    parser.add_argument("--iterations", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--test-model", type=str, default="gpt-4o-mini", help="Student model used for testing prompts (e.g., gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--improve-model", type=str, default="gpt-4o", help="Teacher model used for improving prompts (e.g., gpt-4o, gemini-1.5-pro)")
    
    args = parser.parse_args()
    
    # Auto-detect providers from model names
    test_provider = infer_provider(args.test_model)
    improve_provider = infer_provider(args.improve_model)
    
    print(f"Using test model: {args.test_model} (provider: {test_provider})")
    print(f"Using improve model: {args.improve_model} (provider: {improve_provider})")
    
    dataset_path = args.dataset
    prompts_folder = os.path.join(dataset_path, "prompts")
    outputs_folder = os.path.join(dataset_path, "outputs")
    images_folder = os.path.join(dataset_path, "images")
    
    # Ensure results directory exists
    os.makedirs(os.path.join("Results", "optimized_prompts"), exist_ok=True)
    os.makedirs(os.path.join("Results", "best_outputs"), exist_ok=True)
    
    optimizer = PromptOptimizer()
    results = {}
    
    if not os.path.exists(prompts_folder):
        print(f"Error: Prompts folder not found at {prompts_folder}")
        return

    for fname in os.listdir(prompts_folder):
        if not fname.endswith(".txt"):
            continue
            
        sample_id = fname[:-4]
        print(f"Processing sample: {sample_id}")
        
        prompt_path = os.path.join(prompts_folder, fname)
        target_output_path = os.path.join(outputs_folder, f"{sample_id}.json")
        
        if not os.path.exists(target_output_path):
            print(f"Warning: Target output not found for {sample_id}. Skipping.")
            continue
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            initial_prompt = f.read().strip()
            
        with open(target_output_path, "r", encoding="utf-8") as f:
            target_json = json.load(f)
            
        input_images = sorted([
            os.path.join(images_folder, img)
            for img in os.listdir(images_folder)
            if img.startswith(f"{sample_id}-")
        ])
        
        if not input_images:
            print(f"Warning: No images found for {sample_id}. Skipping.")
            continue
            
        best_prompt, best_score, best_output = optimizer.optimize(
            initial_prompt=initial_prompt,
            target_json_output=target_json,
            input_images=input_images,
            iterations=args.iterations,
            test_model=args.test_model,
            improve_model=args.improve_model,
            test_model_provider=test_provider,
            improve_model_provider=improve_provider
        )
        
        # Save results
        with open(os.path.join("Results", "optimized_prompts", f"{sample_id}_optimized_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(best_prompt)
        
        with open(os.path.join("Results", "best_outputs", f"{sample_id}_best_output.json"), "w", encoding="utf-8") as f:
            f.write(best_output)
            
        results[sample_id] = {
            "best_prompt": best_prompt,
            "best_score": best_score,
            "best_output": best_output
        }
        
    print("\nFinal Results Summary:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
