import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tabulate import tabulate
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze simulation results across multiple experiments.'
    )
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], required=True,
                        help='Dataset to analyze')
    return parser.parse_args()

def find_simulation_experiments(dataset_path):
    """Find all simulation experiment directories and validate required files exist"""
    simulation_path = dataset_path / 'simulation'
    if not simulation_path.exists():
        print(f"Error: Simulation directory not found at {simulation_path}")
        sys.exit(1)
    
    valid_experiments = []
    skipped_experiments = []
    
    for exp_dir in simulation_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        config_file = exp_dir / 'config.json'
        summary_file = exp_dir / 'tasks_summaries.csv'
        
        if config_file.exists() and summary_file.exists():
            valid_experiments.append(exp_dir)
        else:
            missing_files = []
            if not config_file.exists():
                missing_files.append('config.json')
            if not summary_file.exists():
                missing_files.append('tasks_summaries.csv')
            skipped_experiments.append((exp_dir.name, missing_files))
    
    return valid_experiments, skipped_experiments

def load_experiment_data(experiment_path):
    """Load and validate all data for a single experiment"""
    try:
        # Load config
        with open(experiment_path / 'config.json', 'r') as f:
            config = json.load(f)
            
        # Skip if debug mode
        if config.get('debug_mode', False):
            return None
        
        # Load tasks summary
        tasks_df = pd.read_csv(experiment_path / 'tasks_summaries.csv')
        print("\nAvailable columns:", tasks_df.columns.tolist())
        
        # Print summary statistics for key columns
        print("\nSummary statistics:")
        for col in ['num_pruned', 'num_left', 'is_qualified']:
            if col in tasks_df.columns:
                print(f"\n{col}:")
                print(f"  Sum: {tasks_df[col].sum()}")
                print(f"  Mean: {tasks_df[col].mean():.2f}")
                print(f"  Non-zero: {(tasks_df[col] > 0).sum()}")
        
        # Clean up config
        config.pop('dataset', None)  # Remove dataset
        config.pop('experiment_name', None)  # Remove experiment_name
        config.pop('debug_mode', None)  # Remove debug_mode
        
        # Rename config keys to be shorter
        key_mapping = {
            'initial_training_hours': 'init_hours',
            'recalibration_frequency_hours': 'recalib_hours',
            'prune_threshold': 'threshold',
            'pruning_strategy': 'strategy'
        }
        config = {key_mapping.get(k, k): v for k, v in config.items()}
        
        return config, tasks_df
    except Exception as e:
        print(f"Error loading data for {experiment_path.name}: {str(e)}")
        return None

def calculate_quality_metrics(tasks_df):
    """Calculate all quality metrics from a tasks_summaries DataFrame"""
    # Validate required columns
    required_cols = ['num_pruned', 'num_left', 'is_qualified', 'sim_mv', 'gt_mv']
    missing_cols = [col for col in required_cols if col not in tasks_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Task counts
    total_tasks = len(tasks_df)
    affected_tasks = tasks_df[tasks_df['num_pruned'] > 0]
    num_affected_tasks = len(affected_tasks)
    
    # Instance counts
    total_instances = tasks_df['num_pruned'].sum() + tasks_df['num_left'].sum()  # Total is sum of pruned and remaining
    total_pruned = tasks_df['num_pruned'].sum()
    total_remaining = tasks_df['num_left'].sum()
    
    # Calculate qualified instances using is_qualified
    total_disqualified = tasks_df['is_qualified'].sum()  # is_qualified is 1 for disqualified instances
    total_qualified = total_instances - total_disqualified
    remaining_qualified = total_remaining - tasks_df[tasks_df['num_left'] > 0]['is_qualified'].sum()
    
    # Accuracy metrics
    total_accuracy = (tasks_df['sim_mv'] == tasks_df['gt_mv']).mean() * 100
    affected_tasks_accuracy = (affected_tasks['sim_mv'] == affected_tasks['gt_mv']).mean() * 100 if num_affected_tasks > 0 else 0
    
    # Reduction rates
    total_reduction = (1 - total_remaining / total_instances) * 100
    qualified_reduction = (1 - remaining_qualified / total_qualified) * 100 if total_qualified > 0 else 100
    
    return {
        # Task-level counts
        'total_tasks': total_tasks,
        'affected_tasks': num_affected_tasks,
        'pct_tasks_affected': (num_affected_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        
        # Instance-level counts
        'total_instances': total_instances,
        'total_pruned': total_pruned,
        'total_remaining': total_remaining,
        'total_disqualified': total_disqualified,
        'total_qualified': total_qualified,
        'remaining_qualified': remaining_qualified,
        
        # Quality metrics
        'accuracy': total_accuracy,
        'affected_acc': affected_tasks_accuracy,
        'reduction': total_reduction,
        'qual_reduction': qualified_reduction
    }

def create_summary_dataframe(experiments_data):
    """Create DataFrame combining config and metrics for all experiments"""
    summaries = []
    
    for exp_name, (config, metrics) in experiments_data.items():
        # Add all config parameters and metrics
        summary = {**config, **metrics}
        summaries.append(summary)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summaries)
    
    # Sort by threshold
    summary_df = summary_df.sort_values('threshold')
    
    return summary_df

def format_and_save_results(summary_df, output_path):
    """Format results and save to files"""
    # Format float columns to 2 decimal places
    float_cols = ['accuracy', 'affected_acc', 'reduction', 'qual_reduction', 'pct_tasks_affected']
    for col in float_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(2)
    
    # Format integer columns with commas
    int_cols = ['total_tasks', 'affected_tasks', 'total_instances', 'total_pruned', 
                'total_remaining', 'total_disqualified', 'total_qualified', 'remaining_qualified']
    for col in int_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:,}'.format)
    
    # Save CSV
    summary_df.to_csv(output_path / 'simulation_summaries.csv', index=False)
    print(f"\nSaved results to {output_path / 'simulation_summaries.csv'}")
    
    # Print pretty table
    print("\nSimulation Results Summary:")
    print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))

def main():
    args = parse_args()
    
    # Set up paths
    results_dir = Path('results')
    dataset_dir = results_dir / args.dataset
    
    print(f"\nAnalyzing simulations for dataset: {args.dataset}")
    
    # Find experiments
    valid_experiments, skipped_experiments = find_simulation_experiments(dataset_dir)
    
    if not valid_experiments:
        print("No valid experiments found!")
        sys.exit(1)
    
    print(f"\nFound {len(valid_experiments)} valid experiments")
    
    # Process each experiment
    experiments_data = {}
    for exp_dir in valid_experiments:
        print(f"\nProcessing experiment: {exp_dir.name}")
        
        result = load_experiment_data(exp_dir)
        if result is None:
            skipped_experiments.append((exp_dir.name, ['data loading error']))
            continue
            
        config, tasks_df = result
        metrics = calculate_quality_metrics(tasks_df)
        experiments_data[exp_dir.name] = (config, metrics)
    
    # Create summary
    summary_df = create_summary_dataframe(experiments_data)
    
    # Save and display results
    format_and_save_results(summary_df, dataset_dir)
    
    # Report skipped experiments
    if skipped_experiments:
        print("\nSkipped Experiments:")
        for exp_name, missing in skipped_experiments:
            print(f"  - {exp_name}: Missing {', '.join(missing)}")

if __name__ == "__main__":
    main() 