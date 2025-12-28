import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# --- Configuration ---
INPUT_DIR = 'format_output/SWE-agent-LM-32B_train'
BASE_OUTPUT_DIR = 'stat_output'

# Strict whitelist of allowed tags
VALID_TAGS = {
    "localization",
    "code_generation", 
    "verification", 
    "tool_use", 
    "error", 
    "unknown"
}

def create_output_dir(base_dir):
    """Creates a unique timestamped directory for this analysis run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Created output directory: {output_dir}")
    return output_dir

def analyze_and_export_abilities(directory_path, output_dir):
    data_list = []
    
    # --- 1. Aggregators ---
    total_ability_counter = Counter()
    invalid_tag_counter = Counter()
    
    ability_unique_map = defaultdict(set)
    ability_block_durations = defaultdict(list)
    ability_blocks_per_traj = defaultdict(list)
    ability_relative_positions = defaultdict(list)
    transition_counts = defaultdict(Counter)

    if not os.path.exists(directory_path):
        print(f"❌ Directory '{directory_path}' not found.")
        return

    # --- 2. Process Files ---
    files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    print(f"📊 Processing {len(files)} trajectory files...")

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                instance_id = data.get("task_id", "unknown")
                trajectory = data.get("trajectory", [])
                traj_len = len(trajectory)
                
                data_list.append({
                    "instance_id": instance_id,
                    "trajectory_length": traj_len
                })

                # State tracking per file
                current_run_lengths = defaultdict(int)
                blocks_in_this_traj = defaultdict(int)
                seen_abilities_in_file = set()
                previous_abilities = set()

                for step_index, step in enumerate(trajectory):
                    raw_abilities = step.get("critics", {}).get("abilities", [])
                    step_valid_abilities = set()
                    
                    # --- NORMALIZE AND SPLIT ---
                    raw_set = set()
                    
                    if isinstance(raw_abilities, str):
                        raw_list = [raw_abilities]
                    elif isinstance(raw_abilities, list):
                        raw_list = raw_abilities
                    else:
                        raw_list = []

                    for item in raw_list:
                        if item:
                            # Split by comma to handle "localization, tool_use"
                            parts = str(item).split(',')
                            for part in parts:
                                cleaned = part.strip()
                                cleaned = cleaned.replace('<', '').replace('>', '').strip()
                                if cleaned:
                                    raw_set.add(cleaned)

                    # --- VALIDATE TAGS ---
                    for tag in raw_set:
                        if tag in VALID_TAGS:
                            step_valid_abilities.add(tag)
                        else:
                            invalid_tag_counter[tag] += 1 

                    # --- 3. STATS COLLECTION (Valid Only) ---
                    for ability in step_valid_abilities:
                        total_ability_counter[ability] += 1
                        seen_abilities_in_file.add(ability)
                        ability_unique_map[ability].add(instance_id)
                        
                        if traj_len > 0:
                            ability_relative_positions[ability].append(step_index / traj_len)

                    # Transitions
                    if step_index > 0:
                        for prev in previous_abilities:
                            for curr in step_valid_abilities:
                                transition_counts[prev][curr] += 1
                    previous_abilities = step_valid_abilities

                    # Consecutive Logic
                    for ability in step_valid_abilities:
                        if current_run_lengths[ability] == 0:
                            blocks_in_this_traj[ability] += 1
                        current_run_lengths[ability] += 1
                    
                    active_abilities = list(current_run_lengths.keys())
                    for ability in active_abilities:
                        if ability not in step_valid_abilities:
                            ability_block_durations[ability].append(current_run_lengths[ability])
                            del current_run_lengths[ability]

                # End of Trajectory Cleanup
                for ability, duration in current_run_lengths.items():
                    ability_block_durations[ability].append(duration)
                
                for ability in seen_abilities_in_file:
                    ability_blocks_per_traj[ability].append(blocks_in_this_traj[ability])

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

    # --- 4. Compile Valid Statistics ---
    stats = []
    for ability, count in total_ability_counter.most_common():
        unique_instances = len(ability_unique_map[ability])
        durations = ability_block_durations.get(ability, [])
        avg_duration = np.mean(durations) if durations else 0
        blocks_counts = ability_blocks_per_traj.get(ability, [])
        avg_blocks_per_traj = np.mean(blocks_counts) if blocks_counts else 0

        stats.append({
            "Ability": ability,
            "Total_Steps": count,
            "Unique_Instances": unique_instances,
            "Avg_Consecutive_Steps": round(avg_duration, 2),
            "Avg_Blocks_Per_Traj": round(avg_blocks_per_traj, 2)
        })

    stats_df = pd.DataFrame(stats)
    if not stats_df.empty:
        stats_df = stats_df.sort_values(by="Total_Steps", ascending=False)

    # --- 5. Generate Report (Markdown) ---
    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 📊 Agent Ability Analysis Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 1. High-Level Summary\n")
        f.write(f"- **Total Traces Parsed:** {len(data_list)}\n")
        if not pd.DataFrame(data_list).empty:
            f.write(f"- **Avg Trajectory Length:** {pd.DataFrame(data_list)['trajectory_length'].mean():.2f} steps\n")
        f.write(f"- **Valid Tags Count:** {sum(total_ability_counter.values())}\n")
        f.write(f"- **Invalid Tags Count:** {sum(invalid_tag_counter.values())}\n\n")
        
        f.write("## 2. Valid Ability Statistics\n")
        if not stats_df.empty:
            f.write(stats_df.to_markdown(index=False))
        else:
            f.write("No valid abilities found in the dataset.\n")

        f.write("\n\n## 3. Invalid / Screened-Out Tags\n")
        if invalid_tag_counter:
            invalid_df = pd.DataFrame(invalid_tag_counter.most_common(), columns=["Invalid Tag", "Count"])
            f.write(invalid_df.to_markdown(index=False))
        else:
            f.write("✅ No invalid tags found!\n")

    # --- 6. Generate Visualizations (Valid Data Only) ---
    if not stats_df.empty:
        sns.set_theme(style="whitegrid")
        
        # --- MODIFIED: Plot A with Labels ---
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(stats_df))
        width = 0.35
        
        # Create bars and capture them in variables (rects1, rects2)
        rects1 = ax.bar(x - width/2, stats_df["Total_Steps"], width, label='Total Steps', color='skyblue', edgecolor='black')
        rects2 = ax.bar(x + width/2, stats_df["Unique_Instances"], width, label='Unique Instances', color='salmon', edgecolor='black')
        
        # Add data labels
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        
        # Set formatting
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df["Ability"], rotation=45, ha='right')
        ax.set_title("Valid Abilities: Volume vs Breadth")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_1_counts.png"), dpi=300)
        plt.close()

        # Plot B: Temporal Distribution
        sorted_abilities = sorted(
            [k for k in ability_relative_positions.keys() if k in VALID_TAGS and len(ability_relative_positions[k]) > 5], 
            key=lambda k: np.median(ability_relative_positions[k])
        )
        if sorted_abilities:
            plot_data = [ability_relative_positions[k] for k in sorted_abilities]
            plt.figure(figsize=(12, 8))
            plt.boxplot(plot_data, labels=sorted_abilities, patch_artist=True, showfliers=False)
            plt.title("Temporal Distribution of Valid Abilities")
            plt.ylabel("Trajectory Progress (0=Start, 1=End)")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plot_2_temporal.png"), dpi=300)
            plt.close()

        # Plot C: Transition Matrix
        if transition_counts:
            matrix_labels = sorted(list(VALID_TAGS.intersection(set(stats_df["Ability"]))))
            if matrix_labels:
                matrix = pd.DataFrame(0, index=matrix_labels, columns=matrix_labels)
                for prev, next_counts in transition_counts.items():
                    if prev in matrix_labels:
                        for curr, count in next_counts.items():
                            if curr in matrix_labels:
                                matrix.loc[prev, curr] = count
                
                row_sums = matrix.sum(axis=1)
                matrix_norm = matrix.div(row_sums, axis=0).fillna(0)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(matrix_norm, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Prob'})
                plt.title("Transition Probabilities (Valid Tags Only)")
                plt.ylabel("From")
                plt.xlabel("To")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "plot_3_transitions.png"), dpi=300)
                plt.close()

    # --- 7. Save Raw Data ---
    raw_json_path = os.path.join(output_dir, "raw_stats.json")
    with open(raw_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_df.to_dict(orient='records'), f, indent=4)
        
    print(f"✅ Analysis Complete!")
    print(f"📄 Report: {report_path}")
    print(f"📈 Plots saved in: {output_dir}")

if __name__ == "__main__":
    out_dir = create_output_dir(BASE_OUTPUT_DIR)
    analyze_and_export_abilities(INPUT_DIR, out_dir)