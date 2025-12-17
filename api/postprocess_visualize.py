import json
import re
import os
import ast
import argparse

# Configuration for fixed fields
FIXED_METADATA = {
    "task_source": "SWE-Bench-Verified",
    "trajectory_source": "mini-swe-agent-1",
    "in_domain": "1",
    "agent_metadata": {
        "producer": "Kimi-K2-Thinking",
        "model_name": "moonshot/kimi-k2-thinking",
    },
    "critic_agent_metadata": {
        "producer": "Claude-4.5-Sonnet",
        "model_name": "claude-sonnet-4-5-20250929",
    }
}

def parse_llm_xml(raw_text):
    """
    Parses the critic output (LLM result).
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
        abilities_match = re.search(r"<(?:/)?abilities>(.*?)</abilities>", content, re.DOTALL)
        step_data["abilities"] = abilities_match.group(1).strip() if abilities_match else None
        
        parsed_steps.append(step_data)
        
    return parsed_steps

def parse_content_xml(raw_text):
    """
    Parses the 'content' field containing the interaction history.
    """
    if not raw_text:
        return []

    parsed_steps = []
    
    # Regex to find <step_N> blocks
    step_pattern = re.compile(r"<step_(\d+)>(.*?)</step_\1>", re.DOTALL)
    step_matches = step_pattern.findall(raw_text)

    for step_num, content in step_matches:
        step_data = {
            "step_number": int(step_num)
        }

        # Extract User Content (Observation)
        user_match = re.search(r"<user_content>(.*?)</user_content>", content, re.DOTALL)
        step_data["observation"] = user_match.group(1).strip() if user_match else ""

        # Extract Assistant Contents (Raw Response)
        assist_match = re.search(r"<assistant_contents>(.*?)</assistant_contents>", content, re.DOTALL)
        step_data["raw_response"] = assist_match.group(1).strip() if assist_match else ""

        parsed_steps.append(step_data)
    
    return parsed_steps

def parse_assistant_response(raw_response):
    """
    Splits raw response into thought and action.
    Format: Thought text followed by <function=name>...</function>
    """
    thought = ""
    action = ""
    
    # Regex to find the function block
    # Matches <function=...> ... </function> (DOTALL allows matching across newlines)
    action_match = re.search(r"(<function=[\s\S]*?</function>)", raw_response)
    
    if action_match:
        action = action_match.group(1)
        # Thought is everything before the function block
        thought_part = raw_response.split(action)[0]
        thought = thought_part.strip()
    else:
        # If no function block found, assume it is entirely thought/reasoning
        thought = raw_response.strip()
        action = ""

    return {
        "thought": thought,
        "action": action,
        "agent_reasoning": thought 
    }

def process_dataset(input_file_path, output_dir_path):
    """
    Reads the input JSONL and writes reformatted JSON files to the output directory.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            print(f"Created output directory: {output_dir_path}")
        except OSError as e:
            print(f"Error creating directory {output_dir_path}: {e}")
            return

    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found.")
        return

    print(f"Processing {input_file_path}...")
    
    count = 0
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_num}")
                continue

            # 1. Extract IDs from input
            trace_id = data.get("trace_id")
            instance_id = data.get("instance_id", "unknown")

            if not trace_id:
                print(f"Skipping line {line_num}: missing trace_id")
                continue
            
            # 2. Parse Content (Trajectory)
            content_steps = parse_content_xml(data.get("content", ""))
            
            # 3. Parse LLM Result (Critics)
            critic_steps = parse_llm_xml(data.get("llm_result", ""))
            # Index critic steps by step_number for easy lookup
            critic_map = {item['step_number']: item for item in critic_steps}

            # 4. Construct Trajectory List
            trajectory = []
            instruction = ""

            for i, step in enumerate(content_steps):
                step_idx = step["step_number"]
                
                # Assume Step 1 user_content is the instruction
                if step_idx == 1:
                    instruction = step["observation"]

                # Parse internal thought/action using the function XML format
                parsed_response = parse_assistant_response(step["raw_response"])

                # Get critic info for this step
                critic_info = critic_map.get(step_idx, {})
                critic_abilities = critic_info.get("abilities", "[]")
                
                # Convert string representation of list to actual list
                try:
                    if critic_abilities and critic_abilities.startswith("["):
                         critic_abilities_list = ast.literal_eval(critic_abilities)
                    else:
                        critic_abilities_list = [critic_abilities] if critic_abilities else []
                except:
                    critic_abilities_list = [critic_abilities]

                trajectory_step = {
                    "step_index": i, 
                    "observation": step["observation"],
                    "raw_response": step["raw_response"],
                    "agent_reasoning": parsed_response["agent_reasoning"],
                    "thought": parsed_response["thought"],
                    "action": parsed_response["action"],
                    "critics": {
                        "abilities": critic_abilities_list,
                        "critic_thought": critic_info.get("thought", "")
                    }
                }
                trajectory.append(trajectory_step)

            # 5. Assemble Final Object
            final_output = {
                "trace_id": trace_id,
                "task_id": instance_id,
                **FIXED_METADATA,
                "instruction": instruction,
                "trajectory": trajectory,
                "trajectory_length": len(trajectory)
            }

            # 6. Save to individual file
            output_filename = f"{trace_id}.json"
            output_path = os.path.join(output_dir_path, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(final_output, out_f, indent=2, ensure_ascii=False)
            
            count += 1

    print(f"Successfully processed {count} traces to '{output_dir_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat trajectory JSONL data into individual JSON files.")
    
    # Input argument
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True, 
        help="Path to the input .jsonl file"
    )
    
    # Output argument
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        required=True, 
        help="Path to the output directory"
    )

    args = parser.parse_args()
    
    process_dataset(args.input, args.output)