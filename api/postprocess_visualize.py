import json
import re
import os
import ast
import argparse
import sys
from collections import Counter

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
FIXED_METADATA = {
    "task_source": "SWE-Smith",
    "trajectory_source": "SWE-agent-LM-32B_train_5016_trajectories.json",
    "in_domain": "1",
    "agent_metadata": {
        "producer": "",
        "model_name": "",
    },
    "critic_agent_metadata": {
        "producer": "Claude-4.5-Sonnet",
        "model_name": "claude-sonnet-4-5-20250929",
    }
}

# -------------------------------------------------------------------------
# VALIDATION LOGIC
# -------------------------------------------------------------------------

def parse_input_steps(content):
    """
    Scans the input content to find all unique step numbers.
    Returns a set of integers representing the steps found.
    """
    if not content:
        return set()
    step_pattern = re.compile(r'<step_(\d+)>')
    steps = set()
    for match in step_pattern.finditer(content):
        steps.add(int(match.group(1)))
    return steps

def validate_response_structure(llm_result, expected_steps):
    """
    Validates that the LLM result contains all expected steps,
    and that each step has <thought> and <abilities> tags.
    """
    if not llm_result:
        return False, "Empty or Null LLM Result"

    missing_steps = []
    malformed_steps = []

    # If no steps were found in input (e.g. raw text), we assume at least step 1 is required
    steps_to_check = sorted(expected_steps) if expected_steps else [1]

    for step_num in steps_to_check:
        step_regex = re.compile(
            fr"<step_{step_num}>(.*?)</step_{step_num}>",
            re.DOTALL | re.IGNORECASE
        )
        match = step_regex.search(llm_result)

        if not match:
            # Check if tag exists but unclosed or just missing
            if f"<step_{step_num}>" not in llm_result:
                missing_steps.append(step_num)
            else:
                malformed_steps.append(f"Step {step_num} unclosed tag")
            continue

        step_content = match.group(1)

        has_thought = re.search(r"<thought>.*?</thought>", step_content, re.DOTALL | re.IGNORECASE)
        has_abilities = re.search(r"<abilities>.*?</abilities>", step_content, re.DOTALL | re.IGNORECASE)

        if not has_thought:
            malformed_steps.append(f"Step {step_num} missing <thought>")
        elif not has_abilities:
            malformed_steps.append(f"Step {step_num} missing <abilities>")

    if missing_steps:
        return False, f"Missing steps: {missing_steps}"

    if malformed_steps:
        return False, f"Malformed steps: {', '.join(malformed_steps)}"

    return True, "Valid"

# -------------------------------------------------------------------------
# PARSING & REFORMATTING LOGIC
# -------------------------------------------------------------------------

def parse_llm_xml(raw_text):
    """Parses the critic output (LLM result)."""
    if not raw_text:
        return []

    parsed_steps = []
    step_pattern = re.compile(r"<step_(\d+)>(.*?)</step_\1>", re.DOTALL)
    step_matches = step_pattern.findall(raw_text)
    
    for step_num, content in step_matches:
        step_data = {"step_number": int(step_num)}
        
        # Extract <thought>
        thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
        step_data["thought"] = thought_match.group(1).strip() if thought_match else None
        
        # Extract <abilities>
        abilities_match = re.search(r"<(?:/)?abilities>(.*?)</abilities>", content, re.DOTALL)
        step_data["abilities"] = abilities_match.group(1).strip() if abilities_match else None
        
        parsed_steps.append(step_data)
        
    return parsed_steps

def parse_content_xml(raw_text):
    """Parses the 'content' field containing the interaction history."""
    if not raw_text:
        return []

    parsed_steps = []
    step_pattern = re.compile(r"<step_(\d+)>(.*?)</step_\1>", re.DOTALL)
    step_matches = step_pattern.findall(raw_text)

    for step_num, content in step_matches:
        step_data = {"step_number": int(step_num)}

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
    
    # Matches <function=...> ... </function>
    action_match = re.search(r"(<function=[\s\S]*?</function>)", raw_response)
    
    if action_match:
        action = action_match.group(1)
        thought_part = raw_response.split(action)[0]
        thought = thought_part.strip()
    else:
        thought = raw_response.strip()
        action = ""

    return {
        "thought": thought,
        "action": action,
        "agent_reasoning": thought 
    }

# -------------------------------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------------------------------

def process_dataset(input_file_path, output_dir_path, error_report_path=None):
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found.")
        return

    print(f"1. Validating file: {input_file_path} ...\n")
    
    valid_data_lines = []
    error_log = []
    stats = {"total": 0, "valid": 0, "failed": 0}

    # --- Step 1: Validation Pass ---
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                stats["failed"] += 1
                error_log.append({"line": line_num, "id": "unknown", "reason": "Invalid JSON line"})
                continue

            instance_id = data.get("instance_id", f"line_{line_num}")
            content = data.get("content", "")
            llm_result = data.get("llm_result")
            llm_error = data.get("llm_error")

            is_failed = False
            fail_reason = ""

            # Check 1: API Errors
            if llm_error and str(llm_error).lower() not in ["null", "none", ""]:
                is_failed = True
                fail_reason = f"API Error: {str(llm_error)[:100]}..."
            
            # Check 2: Empty Result
            elif not llm_result:
                is_failed = True
                fail_reason = "Empty Result"
            
            # Check 3: Structure (Steps match, tags exist)
            else:
                expected_steps = parse_input_steps(content)
                is_valid, reason = validate_response_structure(llm_result, expected_steps)
                if not is_valid:
                    is_failed = True
                    fail_reason = f"Structure Error: {reason}"

            if is_failed:
                stats["failed"] += 1
                error_log.append({"id": instance_id, "reason": fail_reason})
            else:
                stats["valid"] += 1
                valid_data_lines.append(data)

    # --- Step 2: Print Report ---
    print("-" * 50)
    print("VALIDATION REPORT")
    print("-" * 50)
    print(f"Total Processed: {stats['total']}")
    print(f"✅ Valid:         {stats['valid']}")
    print(f"❌ Failed:        {stats['failed']}")
    print("-" * 50)

    if error_log:
        print("\n--- Top 5 Errors ---")
        for err in error_log[:5]:
            print(f"ID: {err['id']:<25} | {err['reason']}")
        if len(error_log) > 5:
            print(f"... and {len(error_log) - 5} more.")
        
        # Save full error report
        if error_report_path:
            os.makedirs(os.path.dirname(os.path.abspath(error_report_path)), exist_ok=True)
            with open(error_report_path, "w", encoding="utf-8") as ef:
                json.dump({"stats": stats, "errors": error_log}, ef, indent=2)
            print(f"\nFull error report saved to: {error_report_path}")

    if stats["valid"] == 0:
        print("\nNo valid traces found. Aborting reformat.")
        return

    # --- Step 3: Reformatting Pass ---
    print(f"\n2. Reformatting {len(valid_data_lines)} valid traces to '{output_dir_path}'...")
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    processed_count = 0
    
    for data in valid_data_lines:
        trace_id = data.get("trace_id")
        instance_id = data.get("instance_id")

        if not trace_id:
            print(f"Warning: Missing trace_id for instance {instance_id}. Skipping.")
            continue

        # Parse Content
        content_steps = parse_content_xml(data.get("content", ""))
        
        # Parse Critics
        critic_steps = parse_llm_xml(data.get("llm_result", ""))
        critic_map = {item['step_number']: item for item in critic_steps}

        trajectory = []
        instruction = ""

        for i, step in enumerate(content_steps):
            step_idx = step["step_number"]
            
            if step_idx == 1:
                instruction = step["observation"]

            parsed_response = parse_assistant_response(step["raw_response"])
            critic_info = critic_map.get(step_idx, {})
            critic_abilities = critic_info.get("abilities", "[]")

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
                "agent_reasoning": "",
                "thought": parsed_response["thought"],
                "action": parsed_response["action"],
                "critics": {
                    "abilities": critic_abilities_list,
                    "critic_thought": critic_info.get("thought", "")
                }
            }
            trajectory.append(trajectory_step)

        final_output = {
            "trace_id": trace_id,
            "task_id": instance_id,
            **FIXED_METADATA,
            "instruction": instruction,
            "trajectory": trajectory,
            "trajectory_length": len(trajectory)
        }

        output_filename = f"{trace_id}.json"
        output_path = os.path.join(output_dir_path, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_output, out_f, indent=2, ensure_ascii=False)
        
        processed_count += 1

    print(f"Successfully reformatted {processed_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and reformat trajectory JSONL data.")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output directory")
    parser.add_argument("--error_report", "-e", type=str, default="validation_errors.json", help="Path to save error report")

    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.error_report)