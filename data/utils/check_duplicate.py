import json
from collections import Counter
import sys

def find_duplicate_instance_ids(jsonl_path):
    instance_ids = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {line_num}: {e}")
                continue

            instance_id = data.get("instance_id")
            if instance_id is not None:
                instance_ids.append(instance_id)
            else:
                print(f"Warning: Missing instance_id on line {line_num}")

    counts = Counter(instance_ids)
    unique_count = len(counts)
    total_count = len(instance_ids)

    duplicates = {k: v for k, v in counts.items() if v > 1}

    print(f"Total entries          : {total_count}")
    print(f"Unique instance_id(s)  : {unique_count}")

    if duplicates:
        print("\nDuplicate instance_id(s) found:\n")
        for instance_id, count in duplicates.items():
            print(f"{instance_id}: {count} occurrences")
    else:
        print("\nNo duplicate instance_id found.")

    return {
        "total": total_count,
        "unique": unique_count,
        "duplicates": duplicates,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_duplicates.py <path_to_jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    find_duplicate_instance_ids(jsonl_path)
