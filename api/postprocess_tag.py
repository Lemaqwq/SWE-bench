import json
import re
import sys
import os

def parse_llm_xml(raw_text):
    """
    Parses the XML-like string from the LLM.
    structure: 
    <step_N>
        <thought>...</thought>
        <abilities>...</abilities>
    </step_N>
    """
    if not raw_text:
        return []

    parsed_steps = []
    
    # 1. Regex to find all step blocks: <step_1> ... </step_1>
    step_pattern = re.compile(r"<step_(\d+)>(.*?)</step_\1>", re.DOTALL)
    
    # Find all steps
    step_matches = step_pattern.findall(raw_text)
    
    for step_num, content in step_matches:
        step_data = {
            "step_number": int(step_num)
        }
        
        # 2. Extract <thought>
        thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
        step_data["thought"] = thought_match.group(1).strip() if thought_match else None
        
        # 3. Extract <abilities>
        # Handles potential typo </abilities> at the start tag
        abilities_match = re.search(r"<(?:/)?abilities>(.*?)</abilities>", content, re.DOTALL)
        step_data["abilities"] = abilities_match.group(1).strip() if abilities_match else None
        
        parsed_steps.append(step_data)
        
    return parsed_steps

def postprocess_to_directory(input_file, output_dir):
    print(f"Reading from: {input_file}")
    print(f"Saving individual JSONs to: {output_dir}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Load the raw record
                    record = json.loads(line)
                    
                    # Get relevant fields
                    instance_id = record.get("instance_id", f"unknown_{processed_count}")
                    llm_output = record.get('llm_result', '')
                    llm_error = record.get('llm_error') # Capture error if present
                    
                    # Parse the analysis
                    parsed_steps = parse_llm_xml(llm_output)
                    
                    # Create the new clean structure
                    new_record = {
                        "instance_id": instance_id,
                        "model_name": record.get("model_name"),
                        "llm_error": llm_error,
                        "analysis_steps": parsed_steps
                    }
                    
                    # --- SAVE AS INDIVIDUAL JSON FILE ---
                    # Sanitize filename (replace slashes to avoid directory errors)
                    safe_filename = str(instance_id).replace('/', '_').replace('\\', '_') + ".json"
                    file_path = os.path.join(output_dir, safe_filename)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        # indent=4 makes it highly readable
                        json.dump(new_record, f, indent=4, ensure_ascii=False)
                        
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print("Skipping invalid JSON line")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    print(f"Done. Processed {processed_count} records.")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "./llm_results.jsonl"
    OUTPUT_DIRECTORY = "./final_trajectories_json"

    # Allow command line args
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIRECTORY = sys.argv[2]

    postprocess_to_directory(INPUT_FILE, OUTPUT_DIRECTORY)