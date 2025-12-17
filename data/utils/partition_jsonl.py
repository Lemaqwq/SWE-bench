import argparse
import os

def count_lines(filepath):
    """Counts total lines in the file efficiently."""
    print(f"Counting lines in {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        # Generator expression is memory efficient
        return sum(1 for _ in f)

def split_jsonl(input_file, num_partitions):
    # 1. Validate Input
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return

    # 2. Get total line count
    total_lines = count_lines(input_file)
    print(f"Total lines: {total_lines}")

    if total_lines == 0:
        print("File is empty.")
        return

    # 3. Calculate partition sizes
    # Basic size per file
    chunk_size = total_lines // num_partitions
    # Remainder to distribute among the first few files
    remainder = total_lines % num_partitions

    base_name, ext = os.path.splitext(input_file)
    
    current_line_idx = 0
    
    # 4. Process the file
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for i in range(num_partitions):
            # Calculate how many lines go into THIS partition
            # The first 'remainder' files get 1 extra line to balance perfectly
            lines_in_this_part = chunk_size + (1 if i < remainder else 0)
            
            # Generate output filename (e.g., data_part_1.jsonl)
            output_filename = f"{base_name}_part_{i+1}{ext}"
            
            print(f"Writing {lines_in_this_part} lines to {output_filename}...")
            
            # Write the specific number of lines to the output file
            with open(output_filename, 'w', encoding='utf-8') as f_out:
                for _ in range(lines_in_this_part):
                    line = f_in.readline()
                    if not line: 
                        break # Should not happen if logic is correct
                    f_out.write(line)
                    
    print("\nPartitioning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSONL file into N partitions.")
    parser.add_argument("input_file", help="Path to the input .jsonl file")
    parser.add_argument("n", type=int, help="Number of partitions to create")

    args = parser.parse_args()

    split_jsonl(args.input_file, args.n)