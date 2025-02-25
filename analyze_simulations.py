import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils.plot_style import setup_plot_style, style_axis
from ast import literal_eval
from simulation.analysis import plot_before_after_histograms, plot_cycle_roc_curves

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
        
        # Print summary statistics for key columns
        print("\nSummary statistics:")
        for col in ['total_instances', 'pruned_instances']:
            if col in tasks_df.columns:
                print(f"\n{col}:")
                print(f"  Sum: {tasks_df[col].sum()}")
                print(f"  Mean: {tasks_df[col].mean():.2f}")
                print(f"  Non-zero: {(tasks_df[col] > 0).sum()}")
        
        # Print majority vote statistics
        print("\nMajority vote statistics:")
        print("\nGround truth majority votes:")
        print(tasks_df['gt_mv'].value_counts())
        print("\nSimulation majority votes:")
        print(tasks_df[tasks_df['sim_mv'].notna()]['sim_mv'].value_counts())
        
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
    required_cols = ['task_id', 'gt_mv', 'sim_mv', 'pruned_instances', 'total_instances']
    missing_cols = [col for col in required_cols if col not in tasks_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Task-level statistics
    total_tasks = len(tasks_df)
    tasks_with_mv = tasks_df['sim_mv'].notna()
    tasks_with_valid_mv = (tasks_df['gt_mv'].isin(['yes', 'no'])) & (tasks_df['sim_mv'].isin(['yes', 'no']))
    matching_mv = (tasks_df['gt_mv'] == tasks_df['sim_mv']) & tasks_with_mv
    
    # Binary classification metrics (treating 'yes' as positive class)
    valid_tasks = tasks_df[tasks_with_valid_mv].copy()
    y_true = (valid_tasks['gt_mv'] == 'yes').astype(int)
    y_pred = (valid_tasks['sim_mv'] == 'yes').astype(int)
    
    # Calculate confusion matrix
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    
    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Response counts
    total_responses = tasks_df['total_instances'].sum()
    pruned_responses = tasks_df['pruned_instances'].sum()
    
    return {
        # Task-level statistics
        'total_tasks': total_tasks,
        'tasks_with_mv': tasks_with_mv.sum(),
        'tasks_with_valid_mv': tasks_with_valid_mv.sum(),
        'matching_mv': matching_mv.sum(),
        'raw_accuracy': (matching_mv.sum() / tasks_with_mv.sum() * 100) if tasks_with_mv.sum() > 0 else 0,
        
        # Binary classification metrics
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        
        # Confusion matrix
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        
        # Response counts
        'total_responses': total_responses,
        'pruned_responses': pruned_responses,
        'prune_rate': (pruned_responses / total_responses * 100) if total_responses > 0 else 0
    }

def create_summary_dataframe(experiments_data):
    """Create DataFrame combining config and metrics for all experiments"""
    summary_df = pd.DataFrame()
    
    for exp_name, (config, metrics) in experiments_data.items():
        # Extract experiment parameters from name and config
        exp_dict = {
            'experiment': exp_name,
            'strategy': config.get('strategy', 'default'),
            'threshold': config.get('threshold', 0.0),
            'total_instances': metrics.get('total_responses', 0),
            'pruned_instances': metrics.get('pruned_responses', 0),
            'raw_accuracy': metrics.get('raw_accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'prune_rate': metrics.get('prune_rate', 0),
            'true_positives': metrics.get('true_positives', 0),
            'false_positives': metrics.get('false_positives', 0),
            'true_negatives': metrics.get('true_negatives', 0),
            'false_negatives': metrics.get('false_negatives', 0)
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([exp_dict])], ignore_index=True)
    
    # Sort by threshold
    if not summary_df.empty:
        summary_df = summary_df.sort_values('threshold')
    
    return summary_df

def format_and_save_results(summary_df, output_path):
    """Format results and save to files"""
    # Format float columns to 2 decimal places
    float_cols = [
        'threshold', 'raw_accuracy', 'precision', 'recall', 'f1_score', 'prune_rate'
    ]
    for col in float_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(2)
    
    # Format integer columns with commas
    int_cols = [
        'total_instances', 'pruned_instances', 'true_positives', 'false_positives',
        'true_negatives', 'false_negatives'
    ]
    for col in int_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:,}'.format)
    
    # Save full results to CSV
    summary_df.to_csv(output_path / 'simulation_summaries.csv', index=False)
    print(f"\nSaved full results to {output_path / 'simulation_summaries.csv'}")
    
    # Create a simplified table for display
    display_cols = [
        'experiment', 'strategy', 'threshold',
        'raw_accuracy', 'precision', 'recall', 'f1_score',
        'total_instances', 'pruned_instances', 'prune_rate'
    ]
    display_df = summary_df[display_cols].copy()
    
    # Print summary table
    print("\nSimulation Results Summary:")
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Print confusion matrix summary for each experiment
    print("\nConfusion Matrix Summary:")
    for idx, row in summary_df.iterrows():
        print(f"\nExperiment: {row['experiment']}")
        print(f"Strategy: {row['strategy']}, Threshold: {row['threshold']}")
        cm_data = [
            ['', 'Predicted No', 'Predicted Yes'],
            ['Actual No', row['true_negatives'], row['false_positives']],
            ['Actual Yes', row['false_negatives'], row['true_positives']]
        ]
        print(tabulate(cm_data, headers='firstrow', tablefmt='grid'))

def load_gold_standard_data(dataset):
    """Load the gold standard dataset, caching it for reuse"""
    if not hasattr(load_gold_standard_data, 'cache'):
        load_gold_standard_data.cache = {}
    
    if dataset not in load_gold_standard_data.cache:
        data_path = f'data/{dataset}/answers_logreg_all.csv'
        print(f"\nLoading gold standard dataset from {data_path}...")
        df = pd.read_csv(data_path, parse_dates=['created_at'])
        load_gold_standard_data.cache[dataset] = df
    
    return load_gold_standard_data.cache[dataset]

def generate_distribution_plots(full_data, tasks_df, output_dir, experiment_name):
    """Generate before/after histograms comparing distributions"""
    # Calculate remaining responses for each task
    tasks_df = tasks_df.copy()
    tasks_df['remaining_responses'] = tasks_df['total_instances'] - tasks_df['pruned_instances']
    
    # Create a mapping of task_id to remaining_responses
    task_remaining = tasks_df.set_index('task_id')['remaining_responses']
    
    # Add task_id to full_data if not present
    if 'task_id' not in full_data.columns:
        full_data = full_data.copy()
        full_data['task_id'] = full_data['crop_id'].astype(str) + '_' + full_data['question']
    
    # Mark responses as pruned based on task_id
    full_data_with_pruning = full_data.copy()
    full_data_with_pruning['remaining_count'] = full_data_with_pruning['task_id'].map(task_remaining)
    
    # For each task_id, mark the last N responses as not pruned, where N is remaining_count
    full_data_with_pruning['was_pruned'] = True
    for task_id, group in full_data_with_pruning.groupby('task_id'):
        remaining = int(task_remaining.get(task_id, 0))
        if remaining > 0:
            # Sort by created_at and mark the last 'remaining' responses as not pruned
            not_pruned_indices = group.sort_values('created_at').tail(remaining).index
            full_data_with_pruning.loc[not_pruned_indices, 'was_pruned'] = False
    
    # Process worker counts
    worker_counts = full_data_with_pruning['user_id'].value_counts()
    remaining_worker_counts = full_data_with_pruning[~full_data_with_pruning['was_pruned']]['user_id'].value_counts()
    
    # Plot worker repeats
    plot_before_after_histograms(
        worker_counts,
        remaining_worker_counts,
        'Worker',
        str(output_dir),
        experiment_name
    )
    
    # Process task counts
    before_task_counts = tasks_df['total_instances']
    after_task_counts = tasks_df['remaining_responses']
    
    # Plot task repeats
    plot_before_after_histograms(
        before_task_counts,
        after_task_counts,
        'Task',
        str(output_dir),
        experiment_name
    )

def plot_simulation_roc_curves(exp_dir, plots_dir, experiment_name):
    """Generate ROC curves for each cycle's predictions"""
    # Load cycle files
    cycle_files = []
    cycle_num = 1
    while True:
        cycle_file = exp_dir / f'cycle_{cycle_num}.csv'
        if not cycle_file.exists():
            break
        cycle_files.append(pd.read_csv(cycle_file))
        cycle_num += 1
    
    # Call the plotting function from simulation.analysis
    try:
        plot_cycle_roc_curves(cycle_files, str(plots_dir), experiment_name)
    except Exception as e:
        print(f"Warning: Error generating cycle ROC curves: {str(e)}")

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
    
    # Load gold standard data once
    full_data = load_gold_standard_data(args.dataset)
    
    # Process each experiment
    experiments_data = {}
    for exp_dir in valid_experiments:
        print(f"\nProcessing experiment: {exp_dir.name}")
        
        # Create plots directory for this experiment
        plots_dir = exp_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        result = load_experiment_data(exp_dir)
        if result is None:
            skipped_experiments.append((exp_dir.name, ['data loading error']))
            continue
            
        config, tasks_df = result
        metrics = calculate_quality_metrics(tasks_df)
        experiments_data[exp_dir.name] = (config, metrics)
        
        # Generate plots for this experiment
        print(f"\nGenerating plots for {exp_dir.name}...")
        
        try:
            # Before/after histograms
            generate_distribution_plots(full_data, tasks_df, plots_dir, exp_dir.name)
            print("  - Generated before/after histograms")
            
            # Cycle ROC curves
            plot_simulation_roc_curves(exp_dir, plots_dir, exp_dir.name)
            print("  - Generated cycle ROC curves")
        except Exception as e:
            print(f"Warning: Error generating plots for {exp_dir.name}: {str(e)}")
    
    # Create summary DataFrame
    summary_df = create_summary_dataframe(experiments_data)
    
    # Save and display results
    format_and_save_results(summary_df, dataset_dir)
    
    # Report skipped experiments
    if skipped_experiments:
        print("\nSkipped Experiments:")
        for exp_name, missing in skipped_experiments:
            print(f"  - {exp_name}: Missing {', '.join(missing)}")

if __name__ == '__main__':
    main() 