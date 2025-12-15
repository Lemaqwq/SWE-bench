import json
import sys
import os
import argparse

def extract_trajectory_from_messages(messages):
    """
    Core logic to convert a list of raw messages into a list of trajectory steps.
    Consolidates logic for both Live-SWE-Agent (using 'extra') and standard formats.
    """
    extracted_trajectory = []
    
    # Buffer to hold the user message until the assistant responds
    current_user_content = None
    step_count = 1

    for msg in messages:
        # Determine if message is Assistant or User
        has_extra_data = msg.get('extra') is not None
        role = msg.get('role')
        
        is_assistant = has_extra_data or (role == 'assistant')

        if not is_assistant:
            # --- Capture User Content ---
            current_user_content = msg.get('content')
        
        else:
            # --- Capture Assistant Content & Reasoning ---
            assistant_content = msg.get('content')
            reasoning_content = msg.get('reasoning_content')
            
            # Live-SWE-Agent specific 'extra' structure
            if has_extra_data:
                try:
                    response_payload = msg['extra'].get('response', {})
                    choices = response_payload.get('choices', [])
                    if choices:
                        inner_message = choices[0].get('message', {})
                        if assistant_content is None:
                            assistant_content = inner_message.get('content')
                        if reasoning_content is None:
                            reasoning_content = inner_message.get('reasoning_content')
                except (AttributeError, TypeError):
                    pass

            entry = {
                "step": step_count,
                "user_content": current_user_content,
                "assistant_reasoning_content": reasoning_content,
                "assistant_content": assistant_content
            }
            
            extracted_trajectory.append(entry)
            
            # Reset for next turn
            current_user_content = None
            step_count += 1
            
    return extracted_trajectory

def create_output_object(instance_id, trajectory, mode='visualize'):
    """
    Wraps the trajectory in the output format based on the selected mode.
    """
    if mode == 'tagging':
        # Construct the single string content for tagging mode
        full_content_str = ""
        for entry in trajectory:
            step_num = entry['step']
            u_text = entry['user_content'] or ""
            
            # Combine reasoning and content for a_text if reasoning exists, 
            # otherwise just use content.
            r_text = entry.get('assistant_reasoning_content')
            a_content = entry.get('assistant_content') or ""
            
            if r_text:
                a_text = f"{r_text}\n{a_content}"
            else:
                a_text = a_content

            # Strict formatting as requested
            step_str = (
                f"<step_{step_num}>\n"
                f"<user_content>{u_text}</user_content>\n"
                f"<assistant_contents>{a_text}</assistant_contents>\n"
                f"</step_{step_num}>\n\n"
            )
            full_content_str += step_str
            
        return {
            "instance_id": instance_id,
            "content": full_content_str
        }

    else:
        # Default 'visualize' mode (original structure)
        return {
            "instance_id": instance_id,
            "trajectory": trajectory,
            "trajectory_length": len(trajectory)
        }

def process_live_swe_agent_file(input_file, output_file, mode):
    """
    Reads a single JSON file, converts it to the unified format, 
    and saves it as a single-line JSONL file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[Skipping] Error reading {input_file}: {e}")
        return False

    # Extract ID (default to filename if missing)
    instance_id = data.get('instance_id')
    if not instance_id:
        print(f"[Warning] 'instance_id' not detected in {input_file}. Using filename as ID.")
        instance_id = os.path.splitext(os.path.basename(input_file))[0]

    messages = data.get('messages', [])
    trajectory = extract_trajectory_from_messages(messages)
    
    final_output = create_output_object(instance_id, trajectory, mode)

    # Save to File (as a single line for JSONL consistency)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(final_output, ensure_ascii=False) + '\n')
        return True
    except IOError as e:
        print(f"Error saving {output_file}: {e}")
        return False

def process_swe_smith_file(input_file, output_file, mode):
    """
    Reads a JSONL file, processes each line into the unified format, 
    and writes to an output JSONL file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Invalid JSON at line {line_idx+1} in {input_file}")
                    continue
                
                # Extract ID
                instance_id = data.get('instance_id') or data.get('task_id')
                if not instance_id:
                    print(f"[Warning] 'instance_id' not detected at line {line_idx+1} in {input_file}. Using generated ID.")
                    instance_id = f"unknown_id_line_{line_idx+1}"

                # Extract Messages
                messages = []
                if isinstance(data, dict):
                    messages = data.get('messages', [])
                elif isinstance(data, list):
                    messages = data
                
                trajectory = extract_trajectory_from_messages(messages)
                
                final_output = create_output_object(instance_id, trajectory, mode)
                
                # Write unified line
                f_out.write(json.dumps(final_output, ensure_ascii=False) + '\n')
                
        return True

    except IOError as e:
        print(f"Error processing {input_file} -> {output_file}: {e}")
        return False

def process_single_file(input_path, output_path, dataset_type, mode):
    if dataset_type == 'live-swe-agent':
        return process_live_swe_agent_file(input_path, output_path, mode)
    elif dataset_type == 'swe-smith':
        return process_swe_smith_file(input_path, output_path, mode)
    return False

def process_directory_recursively(input_dir, output_dir, dataset_type, mode):
    files_processed = 0
    errors = 0

    print(f"Starting processing ({dataset_type}) in mode '{mode}':\n'{input_dir}' -> '{output_dir}'\n")

    if dataset_type == 'live-swe-agent':
        for root, _, files in os.walk(input_dir):
            for file in files:
                is_target_file = False
                if file.lower().endswith('.json'):
                    is_target_file = True

                if is_target_file:
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, relative_path)
                    
                    # Enforce .jsonl extension for unified output format
                    if not output_path.endswith('.jsonl'):
                        output_path = os.path.splitext(output_path)[0] + '.jsonl'

                    success = process_single_file(input_path, output_path, dataset_type, mode)
                    
                    if success:
                        print(f"[OK] {relative_path} -> {os.path.basename(output_path)}")
                        files_processed += 1
                    else:
                        errors += 1
    elif dataset_type == 'swe-smith':
        success = process_single_file(input_dir, output_dir, dataset_type, mode)
        if success:
            print(f"[OK] {input_dir} -> {output_dir}")
        else:
            errors += 1


    print(f"\n--- Complete ---")
    print(f"Files processed: {files_processed}")
    print(f"Errors/Skipped: {errors}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trajectory files into a unified JSONL format.")
    parser.add_argument("input_directory", nargs='?', default='./input_data', help="Input directory path")
    parser.add_argument("output_directory", nargs='?', default='./processed_output', help="Output directory path")
    parser.add_argument("--dataset", type=str, default="live-swe-agent", 
                        choices=["live-swe-agent", "swe-smith"],
                        help="The dataset type to process")
    parser.add_argument("--mode", type=str, default="visualize",
                        choices=["visualize", "tagging"],
                        help="Output format: 'visualize' for structured JSON, 'tagging' for XML-tagged string content.")

    args = parser.parse_args()

    if not os.path.exists(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' does not exist.")
        sys.exit(1)

    process_directory_recursively(args.input_directory, args.output_directory, args.dataset, args.mode)