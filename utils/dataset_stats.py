import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def compute_dataset_stats(answers_path='data/answers.csv'):
    """Compute statistics for the dataset"""
    print("Loading data...")
    df = pd.read_csv(answers_path)
    
    # Compute all statistics
    stats = {}
    
    # Observation period - use ISO8601 format
    dates = pd.to_datetime(df['created_at'], format='ISO8601')
    start_date = dates.min().strftime('%b. %d')
    end_date = dates.max().strftime('%d, %Y')
    stats['period'] = f"{start_date}-{end_date}"
    
    # Number of unique annotators
    stats['annotators'] = df['user_id'].nunique()
    
    # Number of unique images
    stats['images'] = df['image_path'].nunique()
    
    # Number of unique crops
    stats['crops'] = df['crop_id'].nunique()
    
    # Tasks per crop (assuming constant)
    stats['tasks_per_crop'] = len(df.groupby(['crop_id', 'question']))
    
    # Distribution of repeats per task
    task_repeats = df.groupby(['crop_id', 'question']).size()
    stats['min_repeats'] = task_repeats.min()
    stats['max_repeats'] = task_repeats.max()
    
    # Total number of repeats
    stats['total_repeats'] = len(df)
    
    # Average time per task
    stats['avg_time'] = df['duration_ms'].mean() / 1000  # Convert to seconds
    
    # Disagreement rate
    # First compute majority vote for each task
    print("\nComputing disagreement rate...")
    disagreements = []
    for _, group in tqdm(df.groupby(['crop_id', 'question'])):
        majority = group['answer'].mode().iloc[0]
        disagreements.extend(group['answer'] != majority)
    stats['disagreement_rate'] = 100 * np.mean(disagreements)
    
    return stats

def print_text_table(stats):
    """Print statistics in text format"""
    print("\nDataset Statistics")
    print("=" * 50)
    print(f"Observation period:           {stats['period']}")
    print(f"Number of annotators:         {stats['annotators']}")
    print(f"Number of images:             {stats['images']:,}")
    print(f"Number of crops:              {stats['crops']:,}")
    print(f"Tasks per crop:               {stats['tasks_per_crop']}")
    print(f"Repeats per task:            {stats['min_repeats']}-{stats['max_repeats']}")
    print(f"Total repeats:                {stats['total_repeats']:,}")
    print(f"Average time per task (sec):  {stats['avg_time']:.2f}")
    print(f"Average disagreement rate (%): {stats['disagreement_rate']:.2f}")

def print_latex_table(stats):
    """Print statistics in LaTeX format"""
    print("\n% LaTeX table")
    print("\\begin{table}")
    print("")
    print("\\centering")
    print("\\begin{tabular}{l cc}")
    print("\\toprule")
    print("")
    print("& \\multicolumn{2}{c}{Dataset}   \\\\ \\cmidrule{2-3}")
    print("& ECPD & ZOD \\\\ \\midrule")
    print("")
    print(f"Observation period & {stats['period']} & Jul. 16-17, 2024 \\\\")
    print(f"Annotators & {stats['annotators']} & 18 \\\\")
    print(f"Images & {stats['images']:,} & 8,488 \\\\")
    print(f"Crops & {stats['crops']:,} & 16,000 \\\\")
    print(f"Tasks per crop & {stats['tasks_per_crop']} & 1 \\\\")
    print(f"Repeats per task & {stats['min_repeats']}-{stats['max_repeats']} & 11 \\\\")
    print(f"Total repeats & {stats['total_repeats']:,} & 167,876 \\\\")
    print(f"Average time spent on task (sec) & {stats['avg_time']:.2f} & \\RRM{{X}} \\\\")
    print(f"Average disagreement rate (\\%) & {stats['disagreement_rate']:.2f} & 5.17")
    print("")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("")
    print("\\caption{Dataset Statistics}")
    print("\\label{tab:dataset-stats}")
    print("")
    print("\\end{table}")
    print("")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=['text', 'latex'], default='text',
                       help='Output format (text or latex)')
    args = parser.parse_args()
    
    stats = compute_dataset_stats()
    
    if args.format == 'latex':
        print_latex_table(stats)
    else:
        print_text_table(stats)

if __name__ == "__main__":
    main() 