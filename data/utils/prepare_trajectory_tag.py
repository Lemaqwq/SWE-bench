import json
import sys
import os

def process_single_file(input_file):
    """
    Reads a single input JSON, formats the messages into a sequential XML-style string,
    and returns the dictionary object.
    Returns None if processing fails.
    """
    # 1. Load the Input Data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Skipping] File not found: {input_file}")
        return None
    except json.JSONDecodeError:
        print(f"[Skipping] Invalid JSON: {input_file}")
        return None

    # --- Extract Metadata based on provided template ---

    # 1. Instance ID
    # Priority: root 'instance_id' -> root 'id' -> filename
    instance_id = data.get('instance_id')
    if not instance_id:
        instance_id = data.get('id')
    if not instance_id:
        instance_id = os.path.splitext(os.path.basename(input_file))[0]

    # 2. Model Name
    # Path: info -> config -> model -> model_name
    model_name = None
    try:
        # Use .get() chains to safely navigate potential NoneTypes or missing keys
        info = data.get('info', {}) or {}
        config = info.get('config', {}) or {}
        model_config = config.get('model', {}) or {}
        model_name = model_config.get('model_name')
    except (AttributeError, TypeError):
        pass

    # Fallback: check root or default to unknown
    if not model_name:
        model_name = data.get('model', 'unknown_model')

    messages = data.get('messages', [])
    
    # This string will hold the concatenated XML steps
    full_content_str = ""
    
    # Buffer to hold the user message until the assistant responds
    current_user_content = None
    step_count = 1

    # 3. Process Messages
    for msg in messages:
        # Determine if message is User (no 'extra') or Assistant (has 'extra')
        has_extra_data = msg.get('extra') is not None

        if not has_extra_data:
            # --- Capture User Content ---
            current_user_content = msg.get('content')
        
        else:
            # --- Capture Assistant Content ---
            assistant_content = msg.get('content')
            
            # Navigate deep into extra -> response -> choices -> message
            try:
                response_payload = msg['extra'].get('response', {})
                # Handle cases where response might be None
                if response_payload:
                    choices = response_payload.get('choices', [])
                    if choices:
                        inner_message = choices[0].get('message', {})
                        # Fallback: if top-level content is null, check inner content
                        if assistant_content is None:
                            assistant_content = inner_message.get('content')
            except (AttributeError, TypeError):
                pass

            # Ensure content is string and handle None values
            u_text = str(current_user_content) if current_user_content is not None else ""
            a_text = str(assistant_content) if assistant_content is not None else ""

            # --- Format the Step String ---
            step_str = (
                f"<step_{step_count}>\n"
                f"<user_content>{u_text}</user_content>\n"
                f"<assistant_contents>{a_text}</assistant_contents>\n"
                f"</step_{step_count}>\n\n"
            )
            
            # Append to the main content string
            full_content_str += step_str
            
            # Reset for next turn
            current_user_content = None
            step_count += 1

    # 4. Return Final Object
    return {
        "model_name": model_name,
        "instance_id": instance_id,
        "content": full_content_str
    }

def process_directory_recursively(input_dir, output_dir):
    """
    Walks through input_dir, processes all .json files, and appends them
    to a single JSONL file named after the input directory (reponame).
    """
    
    # Determine the "reponame" from the input directory path
    # os.path.normpath handles trailing slashes (e.g., "folder/" becomes "folder")
    repo_name = os.path.basename(os.path.normpath(input_dir))
    
    # If the folder name is empty (e.g. input was just root), default to 'output'
    if not repo_name: 
        repo_name = "output"
        
    output_filename = f"{repo_name}.jsonl"
    final_output_path = os.path.join(output_dir, output_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    files_processed = 0
    errors = 0

    print(f"Starting processing: '{input_dir}'")
    print(f"Target Output File: '{final_output_path}'\n")

    try:
        # Open the single output file ONCE in write mode
        with open(final_output_path, 'w', encoding='utf-8') as out_f:
            
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.json'):
                        input_path = os.path.join(root, file)
                        
                        # Process the file in memory
                        result_object = process_single_file(input_path)
                        
                        if result_object:
                            # Write to the single JSONL file
                            json.dump(result_object, out_f, ensure_ascii=False)
                            out_f.write('\n')
                            
                            # Optional: Print progress (can get noisy if too many files)
                            # print(f"[OK] Processed: {file}") 
                            files_processed += 1
                        else:
                            errors += 1
                            
    except IOError as e:
        print(f"Critical Error opening output file {final_output_path}: {e}")
        return

    print(f"\n--- Complete ---")
    print(f"Total Files Merged: {files_processed}")
    print(f"Errors/Skipped: {errors}")
    print(f"Saved to: {os.path.abspath(final_output_path)}")

if __name__ == "__main__":
    # Default directories
    input_directory = './input_data'
    output_directory = './processed_output'

    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
    if len(sys.argv) > 2:
        output_directory = sys.argv[2]

    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        sys.exit(1)

    process_directory_recursively(input_directory, output_directory)