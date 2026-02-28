import os
import json
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_existing_prompts(prompts_folder: Path) -> set:
    """Get worksheet numbers that already have prompts"""
    existing = set()
    for file in prompts_folder.glob("*.txt"):
        # Extract the worksheet number from filename (e.g., "1344.txt" -> "1344")
        worksheet_num = file.stem
        existing.add(worksheet_num)
    return existing

def main():
    # Paths
    base_dir = Path(__file__).parent
    context_file = base_dir / "context.json"
    prompts_folder = base_dir / "prompts"
    output_file = base_dir / "worksheets_to_process.json"
    
    # Load context.json (worksheet types and their worksheet numbers)
    with open(context_file, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    # Get existing prompts (to skip these)
    existing_prompts = get_existing_prompts(prompts_folder)
    print(f"Found {len(existing_prompts)} existing prompts: {existing_prompts}")
    
    # Connect to MongoDB
    mongo_url = os.getenv("MONGO_URL")
    if not mongo_url:
        print("Error: MONGO_URL not found in .env")
        return
    
    client = MongoClient(mongo_url)
    db = client["saarthiEd"]
    collection = db["worksheets"]
    
    # Find ALL unprocessed worksheets
    results = {}  # {worksheet_number: {"worksheet_type": type, "s3_urls": urls}}
    found_count = 0
    missing_from_db = []
    missing_s3_urls = []
    
    for worksheet_type, worksheet_nums in context.items():
        for num in worksheet_nums:
            num_str = str(num)
            
            # Skip if this worksheet already has a prompt
            if num_str in existing_prompts:
                continue
            
            # Query by worksheet_name field
            doc = collection.find_one({"worksheet_name": num_str})
            
            if not doc:
                missing_from_db.append(num_str)
            elif "s3_urls" not in doc or not doc["s3_urls"]:
                missing_s3_urls.append(num_str)
            else:
                results[num_str] = {
                    "worksheet_type": worksheet_type,
                    "s3_urls": doc["s3_urls"]
                }
                found_count += 1
                print(f"Found worksheet {num_str} (Type: {worksheet_type})")
    
    print(f"\n--- Summary ---")
    print(f"Worksheets found and ready: {found_count}")
    print(f"Worksheets missing from DB: {len(missing_from_db)}")
    print(f"Worksheets lacking S3 URLs: {len(missing_s3_urls)}")
    
    # Save results to JSON
    output_data = {
        "total_worksheets_ready": found_count,
        "worksheets": results,
        "missing_from_db": missing_from_db,
        "missing_s3_urls": missing_s3_urls
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Close MongoDB connection
    client.close()

if __name__ == "__main__":
    main()
