import json
import sys

def strip_values(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if key == "type":
                new_obj[key] = value        # preserve
            else:
                new_obj[key] = strip_values(value)
        return new_obj

    elif isinstance(obj, list):
        return [strip_values(item) for item in obj]

    else:
        return None   # omit value for non-"type" entries

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    result = strip_values(data)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python strip_json_values.py input.json output.json")
    else:
        main(sys.argv[1], sys.argv[2])
