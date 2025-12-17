import json
import re
import argparse
import sys
import os
from collections import Counter

import matplotlib.pyplot as plt


def parse_input_steps(content):
    """
    Scans the input content to find all unique step numbers.
    Returns a set of integers representing the steps found (e.g., {1, 2, 3}).
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


# -------------------------
# Abilities extraction
# -------------------------

ABILITIES_BLOCK_RE = re.compile(r"<abilities>(.*?)</abilities>", re.DOTALL | re.IGNORECASE)


def normalize_ability_token(tok: str) -> str:
    tok = tok.strip().strip('"').strip("'")
    tok = re.sub(r"^\s*[-*•]+\s*", "", tok)  # remove bullets
    tok = tok.strip()
    return tok


def parse_abilities_categories(abilities_text: str):
    """
    Returns a list of normalized ability category strings.
    Handles:
      - JSON list: ["a","b"]
      - comma-separated: a, b
      - newline/bullet-separated
    """
    if not abilities_text:
        return []

    text = abilities_text.strip()

    # Try JSON list first
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                out = []
                for x in arr:
                    if isinstance(x, str):
                        t = normalize_ability_token(x)
                        if t:
                            out.append(t)
                return out
        except Exception:
            pass  # fall through

    # Otherwise split on commas and/or newlines
    parts = re.split(r"[,\n\r]+", text)
    out = []
    for p in parts:
        t = normalize_ability_token(p)
        if t:
            out.append(t)
    return out


def extract_abilities_from_llm_result(llm_result: str):
    """
    Extract abilities across the entire llm_result (all steps).
    Returns list of ability categories.
    """
    cats = []
    for m in ABILITIES_BLOCK_RE.finditer(llm_result or ""):
        block = m.group(1)
        cats.extend(parse_abilities_categories(block))
    return cats


def save_histogram(counter: Counter, out_path: str, top_k: int = 30):
    """
    Saves a bar chart (histogram) of the top_k categories.
    """
    if not counter:
        print("No abilities found; histogram not generated.")
        return

    most_common = counter.most_common(top_k)
    labels = [k for k, _ in most_common]
    values = [v for _, v in most_common]

    plt.figure(figsize=(max(10, len(labels) * 0.4), 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"Top {min(top_k, len(labels))} Abilities Categories")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Histogram saved to: {out_path}")


def default_path_in_input_dir(input_filepath: str, filename: str) -> str:
    """
    Returns a path that lives in the same directory as input_filepath.
    If input_filepath has no directory component, uses current directory.
    """
    in_dir = os.path.dirname(os.path.abspath(input_filepath))
    return os.path.join(in_dir, filename)


def process_file(
    input_filepath,
    retry_output_path=None,
    abilities_summary_path=None,
    histogram_path=None,
    top_k=30,
    error_report_path=None,
    count_abilities_only_on_valid=True,
):
    print(f"Analyzing file: {input_filepath} ...\n")

    stats = {
        "total": 0,
        "valid": 0,
        "failed": 0,  # Aggregate of api_error, structural_error, empty
    }

    failed_instances = []  # Store full objects to regenerate retry file
    error_log = []         # Store error messages for display + JSON
    abilities_counter = Counter()

    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                stats["total"] += 1

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    err_id = f"unknown_line_{line_num}"
                    reason = "Invalid JSON line"
                    stats["failed"] += 1
                    error_log.append({"id": err_id, "reason": reason})
                    failed_instances.append({"instance_id": err_id, "content": ""})
                    print(f"[Line {line_num}] Error: Invalid JSON line.")
                    continue

                instance_id = data.get("instance_id", f"unknown_line_{line_num}")
                content = data.get("content", "")
                llm_result = data.get("llm_result")
                llm_error = data.get("llm_error")

                is_failed = False
                fail_reason = ""

                # 1. Check for API/System Errors
                if llm_error and str(llm_error).lower() not in ["null", "none", ""]:
                    is_failed = True
                    fail_reason = f"API Error: {str(llm_error)[:200]}"

                # 2. Check for Empty Result
                elif not llm_result:
                    is_failed = True
                    fail_reason = "Empty Result"

                # 3. Check Structure
                else:
                    expected_steps = parse_input_steps(content)
                    is_valid, reason = validate_response_structure(llm_result, expected_steps)

                    if not is_valid:
                        is_failed = True
                        fail_reason = f"Structure Error: {reason}"

                # 4. Count abilities (either only on valid, or on any non-empty llm_result)
                if llm_result and (not count_abilities_only_on_valid or not is_failed):
                    cats = extract_abilities_from_llm_result(llm_result)
                    abilities_counter.update(cats)

                # 5. Categorize
                if is_failed:
                    stats["failed"] += 1
                    error_log.append({"id": instance_id, "reason": fail_reason})
                    failed_instances.append({"instance_id": instance_id, "content": content})
                else:
                    stats["valid"] += 1

    except FileNotFoundError:
        print(f"Error: File '{input_filepath}' not found.")
        sys.exit(1)

    # --- Print Report ---
    print("-" * 50)
    print("SUMMARY REPORT")
    print("-" * 50)
    print(f"Total Processed: {stats['total']}")
    print(f"✅ Valid:         {stats['valid']}")
    print(f"❌ Failed:        {stats['failed']}")
    print("-" * 50)

    # --- Abilities summary ---
    print("\nABILITIES SUMMARY (Top 20):")
    if abilities_counter:
        for k, v in abilities_counter.most_common(20):
            print(f"{k:<30} {v}")
    else:
        print("(none found)")

    # --- Generate Retry File ---
    if retry_output_path:
        if stats["failed"] > 0:
            print(f"\nGenerating retry file for {stats['failed']} instances...")
            try:
                os.makedirs(os.path.dirname(retry_output_path) or ".", exist_ok=True)
                with open(retry_output_path, "w", encoding="utf-8") as f:
                    for instance in failed_instances:
                        json.dump(instance, f, ensure_ascii=False)
                        f.write("\n")
                print(f"✅ Retry file saved to: {retry_output_path}")
            except Exception as e:
                print(f"Error writing retry file: {e}")
        else:
            print("\nNo failed instances found. No retry file generated.")

    # --- Abilities counts JSON ---
    if abilities_summary_path:
        try:
            os.makedirs(os.path.dirname(abilities_summary_path) or ".", exist_ok=True)
            with open(abilities_summary_path, "w", encoding="utf-8") as f:
                json.dump(dict(abilities_counter.most_common()), f, ensure_ascii=False, indent=2)
            print(f"✅ Abilities summary saved to: {abilities_summary_path}")
        except Exception as e:
            print(f"Error writing abilities summary: {e}")

    # --- Error report JSON (SAVED IN SAME DIR AS INPUT by default) ---
    if error_report_path:
        try:
            os.makedirs(os.path.dirname(error_report_path) or ".", exist_ok=True)
            report = {"stats": stats, "errors": error_log}
            with open(error_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ Error report saved to: {error_report_path}")
        except Exception as e:
            print(f"Error writing error report: {e}")

    # --- Histogram ---
    if histogram_path:
        save_histogram(abilities_counter, histogram_path, top_k=top_k)

    # --- Print Detailed Errors (Optional View) ---
    if error_log:
        print("\n--- ERROR DETAILS (First 10) ---")
        for err in error_log[:10]:
            print(f"ID: {err['id']:<30} | {err['reason']}")
        if len(error_log) > 10:
            print(f"... and {len(error_log) - 10} more.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze results, generate retry file, count abilities categories, and create a histogram + error report."
    )

    parser.add_argument("input_file", help="Path to the .jsonl results file to analyze")
    parser.add_argument("--retry_output", "-r", help="Path to save the retry input file (optional)")

    parser.add_argument("--abilities_summary", "-a", help="Path to save abilities counts JSON (optional)")
    parser.add_argument("--histogram", "-g", help="Path to save histogram PNG (optional)")
    parser.add_argument("--top_k", type=int, default=30, help="Top K abilities to plot in histogram (default 30)")

    # Error report: if omitted, we will auto-save into input dir with a default name.
    parser.add_argument(
        "--error_report",
        "-e",
        nargs="?",
        const="__AUTO__",
        default="__AUTO__",
        help=(
            "Path to save error report JSON (optional). "
            "If provided without a value, or omitted, defaults to same directory as input."
        ),
    )

    args = parser.parse_args()

    # Default error report path = same directory as input file
    if args.error_report == "__AUTO__":
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        error_report_path = default_path_in_input_dir(args.input_file, f"{base}_error_report.json")
    else:
        # If user explicitly provides a path, respect it.
        # If they give only a filename, it still goes to CWD (as standard argparse behavior).
        error_report_path = args.error_report

    process_file(
        args.input_file,
        retry_output_path=args.retry_output,
        abilities_summary_path=args.abilities_summary,
        histogram_path=args.histogram,
        top_k=args.top_k,
        error_report_path=error_report_path,
    )
