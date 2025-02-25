import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import json
from tqdm import tqdm
# Import our simulation helper modules and our simulation model
from simulation import data_utils, pruning, analysis
from models import disagreement_simulation
from sklearn.metrics import confusion_matrix
from utils.metrics_utils import calculate_confusion_metrics, calculate_binary_metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simulate task removal based on disagreement predictions.'
    )
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], required=True,
                        help='Dataset to process')
    parser.add_argument('--initial_training_hours', type=int, default=24,
                        help='Number of hours to use for initial training')
    parser.add_argument('--recalibration_frequency_hours', type=int, default=6,
                        help='Hours between model recalibrations')
    parser.add_argument('--prune_threshold', type=float, default=0.1,
                        help='Probability threshold for pruning')
    parser.add_argument('--pruning_strategy', 
                        choices=['default', 'allow_workers'],
                        default='default',
                        help='Strategy for pruning tasks: "default" protects first instance of each worker, '
                             '"allow_workers" allows pruning new workers')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name for this experiment run')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with reduced dataset')
    return parser.parse_args()

def debug_majority_vote_errors(training_data, full_data, max_errors=5):
    """
    Debug function to analyze tasks where current majority vote differs from ground truth.
    
    Args:
        training_data: Current training dataset with majority votes
        full_data: Complete dataset with ground truth majority votes
        max_errors: Maximum number of errors to report (default: 5)
    
    Returns:
        tuple: (total_tasks, wrong_tasks, error_details)
            - total_tasks: Total number of tasks evaluated
            - wrong_tasks: Number of tasks with incorrect majority votes
            - error_details: List of dicts with detailed error information
    """
    # Get gold standard majority votes directly from full_data
    gold_majority = full_data.groupby('task_id')['majority_vote'].first()
    
    # Get current majority votes by using update_votes on training data
    training_data_with_votes = data_utils.update_votes(training_data.copy(), verbose=False)
    current_majority = training_data_with_votes.groupby('task_id')['majority_vote'].first()
    
    # Find common tasks and compare votes
    common_tasks = current_majority.index.intersection(gold_majority.index)
    total_tasks = len(common_tasks)
    
    # Find tasks with incorrect majority votes
    incorrect_tasks = common_tasks[current_majority[common_tasks] != gold_majority[common_tasks]]
    wrong_tasks = len(incorrect_tasks)
    
    # Get detailed information for incorrect tasks
    error_details = []
    if wrong_tasks > 0:
        for task_id in incorrect_tasks[:max_errors]:
            task_data = training_data_with_votes[training_data_with_votes['task_id'] == task_id]
            gold_vote = gold_majority[task_id]
            current_vote = current_majority[task_id]
            
            # Get question type if it exists (ECPD specific)
            q_cols = [col for col in task_data.columns if col.startswith('q_')]
            question_type = task_data[q_cols].iloc[0].idxmax() if q_cols else "N/A"
            
            # Get current response distribution
            response_dist = task_data['response'].value_counts().to_dict()
            
            # Get full data responses for comparison
            full_task_data = full_data[full_data['task_id'] == task_id]
            full_response_dist = full_task_data['response'].value_counts().to_dict()
            
            error_details.append({
                'task_id': task_id,
                'question_type': question_type,
                'gold_vote': gold_vote,
                'current_vote': current_vote,
                'current_responses': response_dist,
                'full_responses': full_response_dist
            })
    
    # Add debug info about overall state
    print(f"Total instances: {len(full_data)}")
    print(f"Pruned instances: {full_data['was_pruned'].sum()}")
    print(f"Remaining instances: {len(full_data[~full_data['was_pruned']])}")
    
    return total_tasks, wrong_tasks, error_details

def format_error_details(error_details):
    """Format error details for printing"""
    if not error_details:
        return []
    
    output = [
        "\nTop Majority Vote Errors:",
        "----------------------------------------"
    ]
    
    for error in error_details:
        output.extend([
            f"\nTask {error['task_id']} ({error['question_type']}):",
            f"  Ground truth majority: {error['gold_vote']}",
            f"  Current majority: {error['current_vote']}",
            f"  Current responses: {error['current_responses']}",
            f"  Full dataset responses: {error['full_responses']}"
        ])
    
    return output

def calculate_cycle_metrics(test_data, pruned_data, cycle_num, 
                        cumulative_stats, training_data, threshold, 
                        full_data):
    """Calculate and format metrics for the current cycle."""
    # Get candidate rows (those eligible for pruning)
    candidate_mask = test_data['predicted_proba'] >= 0
    candidates = test_data[candidate_mask]
    
    # Update cumulative stats
    cumulative_stats['total_processed'] += len(test_data)
    cumulative_stats['total_pruned'] += len(pruned_data)
    cumulative_stats['total_disqualified'] += len(test_data) - len(candidates)
    cumulative_stats['total_qualified'] += len(candidates)
    
    # Calculate TPR/FPR for model predictions on qualified points only
    y_true = candidates['disagree']
    y_pred = (candidates['predicted_proba'] >= threshold).astype(int)
    tpr, fpr = calculate_confusion_metrics(y_true, y_pred)
    
    # Calculate dataset quality metrics and get error details
    total_tasks, wrong_tasks, error_details = debug_majority_vote_errors(
        training_data, full_data)
    error_rate = (wrong_tasks/total_tasks*100) if total_tasks > 0 else 0
    cumulative_stats['current_error_rate'] = error_rate
    
    # Calculate prune rates
    prune_rate = (len(pruned_data) / len(candidates) * 100) if len(candidates) > 0 else 0
    overall_prune_rate = (cumulative_stats['total_pruned'] / cumulative_stats['total_qualified'] * 100) if cumulative_stats['total_qualified'] > 0 else 0
    
    # Format the output
    output = [
        "\n" + "*"*80,
        f"CYCLE {cycle_num} SUMMARY",
        "*"*80,
        f"\nCurrent Cycle Statistics:",
        f"  - Total rows: {len(test_data)}",
        f"  - Candidate rows: {len(candidates)}",
        f"  - Non-candidate rows: {len(test_data) - len(candidates)}",
        f"  - Pruned rows: {len(pruned_data)}",
        f"  - Prune rate (of candidates): {prune_rate:.1f}%",
        f"  - Model TPR: {tpr:.3f}",
        f"  - Model FPR: {fpr:.3f}",
        f"\nDataset Quality:",
        f"  - Total unique tasks: {total_tasks}",
        f"  - Tasks with incorrect majority: {wrong_tasks}",
        f"  - Error rate: {error_rate:.1f}%"
    ]
    
    # Add error details
    # output.extend(format_error_details(error_details))
    
    output.extend([
        f"\nCumulative Statistics:",
        f"  - Total processed: {cumulative_stats['total_processed']}",
        f"  - Total qualified: {cumulative_stats['total_qualified']}",
        f"  - Total disqualified: {cumulative_stats['total_disqualified']}",
        f"  - Total pruned: {cumulative_stats['total_pruned']}",
        f"  - Overall prune rate: {overall_prune_rate:.1f}%",
        f"  - Current error rate: {cumulative_stats['current_error_rate']:.1f}%",
        "*"*80 + "\n"
    ])
    
    return "\n".join(output)

def main():
    args = parse_args()
    dataset = args.dataset
    experiment_name = args.experiment_name
    
    # Set up simulation output directories
    sim_results_dir = Path(f'results/{dataset}/simulation/{experiment_name}')
    sim_results_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir = sim_results_dir / 'model_parameters'
    parameters_dir.mkdir(exist_ok=True)
    plots_dir = sim_results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Save simulation configuration
    config = {
        'dataset': dataset,
        'experiment_name': experiment_name,
        'initial_training_hours': args.initial_training_hours,
        'recalibration_frequency_hours': args.recalibration_frequency_hours,
        'prune_threshold': args.prune_threshold,
        'pruning_strategy': args.pruning_strategy,
        'debug_mode': args.debug
    }
    with open(sim_results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("\nSaved simulation configuration to:", sim_results_dir / 'config.json')
    
    # Load gold standard file. 
    # data/ECPD/answers_logreg_all.csv is assumed to be invariant.
    if dataset == 'ECPD':
        gt_path = 'data/ECPD/answers_logreg_all.csv'
    else:
        gt_path = 'data/ZOD/answers_logreg_all.csv'
    full_data = pd.read_csv(gt_path, parse_dates=['created_at'])
    # Force all created_at entries to be Timestamps
    full_data['created_at'] = pd.to_datetime(full_data['created_at'],
                                             errors='coerce',
                                             utc=True)
    
    # Recalculate majority votes in gold standard dataset using consistent logic
    print("\nRecalculating majority votes in gold standard dataset...")
    full_data = data_utils.update_votes(full_data)
    print(f"Updated {len(full_data)} rows with new majority votes")
    
    # Initialize state tracking columns
    print("\nInitializing state tracking columns...")
    full_data['is_qualified'] = False  # Was instance eligible for pruning?
    full_data['was_pruned'] = False    # Did it get pruned?
    full_data['sim_mv'] = None         # Majority vote at test time
    full_data['cycle_num'] = -1        # Which cycle it was tested in (-1 = not tested yet)
    full_data['predicted_proba'] = -1  # Model prediction (-1 = not predicted yet)
    full_data['write_count'] = 0       # Track number of times each row is written
    
    if args.debug:
        print("DEBUG MODE: Using 50% of full dataset")
        full_data = full_data.sample(frac=0.5, random_state=42)
    
    # Sort by created_at. 
    # Note: more realistic work might use created_at+duration_ms.
    full_data.sort_values('created_at', inplace=True)
    
    # Set start time based on dataset
    if dataset == 'ECPD':
        # For ECPD: Use midnight of the first day
        first_date = pd.to_datetime(full_data['created_at'].min()).date()
        start_time = pd.Timestamp(first_date).tz_localize('UTC').replace(
            hour=0, minute=0, second=0, microsecond=0)
    else:
        # For ZOD: Use 1 minute before first observation. This is because there 
        # is not enough data in ZOD to use midnight of the first day.
        first_observation = pd.to_datetime(full_data['created_at'].min())
        start_time = first_observation - pd.Timedelta(minutes=1)
    
    initial_end_time = start_time + timedelta(hours=args.initial_training_hours)
    training_data = full_data[ 
        full_data['created_at'] < initial_end_time ].copy()
    remaining_data = full_data[ 
        full_data['created_at'] >= initial_end_time ].copy()
    
    print("\n" + "*"*80)
    print("INITIAL TRAINING PERIOD")
    print("*"*80)
    print(f"Start time: {start_time}")
    print(f"End time: {initial_end_time}")
    print(f"Duration: {args.initial_training_hours} hours")
    print(f"Number of annotations: {len(training_data)}")
    print("*"*80 + "\n")
    
    # Update training_data votes and disagreement indicators
    training_data = data_utils.update_votes(training_data)
    
    # Initialize cumulative statistics
    cumulative_stats = {
        'total_processed': 0,
        'total_pruned': 0,
        'total_disqualified': 0,
        'total_qualified': 0,
        'current_error_rate': 0
    }
    
    simulation_metrics = []  # collect metrics per cycle
    current_time = initial_end_time
    cycle_num = 1
    
    while True:
        next_time = current_time + \
            timedelta(hours=args.recalibration_frequency_hours)
        # If no remaining data in the next window, break simulation loop
        if remaining_data.empty:
            print("\nNo more data to process. Simulation complete.")
            break
        
        print("\n" + "*"*80)
        print(f"CYCLE {cycle_num} TIME WINDOW")
        print("*"*80)
        print(f"Start time: {current_time}")
        print(f"End time: {next_time}")
        print(f"Duration: {args.recalibration_frequency_hours} hours")
        print("*"*80)
        
        # Select testing period rows between current_time and next_time
        test_mask = remaining_data['created_at'].between(
            current_time, next_time, inclusive='left')
        test_data = remaining_data[test_mask].copy()
        test_data_indices = test_data.index

        # If no test data in this window, continue to next cycle
        if len(test_data) == 0:
            print("\nNo data in current time window. Moving to next cycle.")
            current_time = next_time
            continue
        
        # Remove testing period rows from remaining_data
        remaining_data = remaining_data[~test_mask].copy()
        
        # Combine test data with training data to obtain ground truth votes
        combined_data = pd.concat([training_data, test_data])  # Keep original indices
        combined_data = data_utils.update_votes(combined_data, verbose=False)
        
        # Extract updated test data with votes computed on all available data
        test_data = combined_data.loc[test_data_indices].copy()
        
        # Recalculate majority vote/disagree for current training set
        training_data = data_utils.update_votes(training_data)
        
        # Fit simulation mixed-effects model
        cycle_parameters_dir = parameters_dir / f'cycle_{cycle_num}'
        cycle_parameters_dir.mkdir(exist_ok=True)
        sim_model = disagreement_simulation.fit_simulation_model(
            training_data, 
            dataset=dataset, 
            output_dir=str(cycle_parameters_dir))
        
        # Make predictions on test data
        test_data['predicted_proba'] = disagreement_simulation.predict_simulation_proba(test_data, sim_model)
        
        # Verify predictions are valid
        if test_data['predicted_proba'].isna().any():
            raise ValueError("Model produced NaN predictions. Model fitting likely failed.")
        
        if len(test_data['predicted_proba'].unique()) <= 1:
            raise ValueError("Model produced constant predictions. Model fitting likely failed.")
            
        if not (0 <= test_data['predicted_proba'].min() <= test_data['predicted_proba'].max() <= 1):
            raise ValueError(f"Predictions outside valid probability range [0,1]. "
                           f"Range: [{test_data['predicted_proba'].min():.3f}, "
                           f"{test_data['predicted_proba'].max():.3f}]")
        
        # Handle new entities
        seen_mask = data_utils.is_seen(test_data, training_data,
                                      strategy=args.pruning_strategy,
                                      dataset=args.dataset)
        test_data.loc[~seen_mask, 'predicted_proba'] = -1.0
        
        # Prune test data
        pruned_data, cycle_stats = pruning.prune_data(
            test_data, threshold=args.prune_threshold,
            strategy=args.pruning_strategy)
        
        # Increment write count for affected rows
        full_data.loc[test_data_indices, 'write_count'] += 1
        
        # Check for multiple writes and warn if found
        multiple_writes = full_data[full_data['write_count'] > 1]
        if len(multiple_writes) > 0:
            print("\nWARNING: Found rows written multiple times:")
            print(f"Total rows with multiple writes: {len(multiple_writes)}")
            print("\nSample of affected rows:")
            sample = multiple_writes.head()
            for idx, row in sample.iterrows():
                print(f"\nRow {idx}:")
                print(f"  Write count: {row['write_count']}")
                print(f"  Created at: {row['created_at']}")
                print(f"  Task ID: {row['task_id']}")
                print(f"  Cycle number: {row['cycle_num']}")
        
        # Update state tracking columns
        full_data.loc[test_data_indices, 'is_qualified'] = seen_mask
        full_data.loc[test_data_indices, 'cycle_num'] = cycle_num
        full_data.loc[test_data_indices, 'predicted_proba'] = test_data['predicted_proba']
        pruned_indices = pruned_data.index
        full_data.loc[pruned_indices, 'was_pruned'] = True
        
        if len(pruned_data) == 0:
            print("\nNo assignments were pruned in this cycle.")
        
        # The surviving testing data (i.e. not pruned) are added to training dataset for next cycle
        kept_data = test_data.drop(pruned_data.index)
        training_data = pd.concat([training_data, kept_data])  # Remove ignore_index=True to maintain indices
        training_data = data_utils.update_votes(training_data, verbose=False)
        
        # Update simulation majority votes for current data using original indices
        kept_indices = kept_data.index  # These are the original indices we want to update
        full_data.loc[kept_indices, 'sim_mv'] = training_data.loc[kept_indices, 'majority_vote']
        
        # Optionally write out cycle testing data for record
        cycle_file = sim_results_dir / f'cycle_{cycle_num}.csv'
        pruned_data.to_csv(cycle_file, index=False)  # Save only pruned data
        
        # Calculate and print cycle metrics
        cycle_summary = calculate_cycle_metrics(
            test_data, pruned_data, cycle_num, cumulative_stats, training_data, 
            args.prune_threshold, full_data)
        print(cycle_summary)

        cycle_stats['cycle_num'] = cycle_num
        cycle_stats['cycle_start'] = current_time
        cycle_stats['cycle_end'] = next_time
        simulation_metrics.append(cycle_stats)
        
        cycle_num += 1
        current_time = next_time

    # Process any remaining data in the final cycle
    if not remaining_data.empty:
        print("*"*80)
        print("*"*80)
        print("*"*80)
        print("Is there any reason why we should ever get here?")
        print("*"*80)
        print("*"*80)
        print("*"*80)
        final_test_data = remaining_data.copy()
        # Combine with training data for vote computation
        combined_data = pd.concat([training_data, final_test_data], ignore_index=True)
        combined_data = data_utils.update_votes(combined_data)
        final_test_data = combined_data[~combined_data['created_at'].isin(training_data['created_at'])].copy()
        
        # Predict and prune final test data
        sim_model = disagreement_simulation.fit_simulation_model(
            training_data, dataset=dataset, output_dir=str(cycle_parameters_dir))
        final_test_data['predicted_proba'] = disagreement_simulation.predict_simulation_proba(
            final_test_data, sim_model)
        
        # Handle new entities in final cycle
        seen_mask = data_utils.is_seen(final_test_data, training_data,
                                      strategy=args.pruning_strategy,
                                      dataset=args.dataset)
        final_test_data.loc[~seen_mask, 'predicted_proba'] = -1.0
        
        # Prune final test data
        final_pruned_data, final_cycle_stats = pruning.prune_data(
            final_test_data, threshold=args.prune_threshold,
            strategy=args.pruning_strategy)
        
        # Add kept data to training set
        final_kept_data = final_test_data.drop(final_pruned_data.index)
        training_data = pd.concat([training_data, final_kept_data], ignore_index=True)
        
        # Update cumulative stats
        final_cycle_stats['cycle_num'] = cycle_num
        simulation_metrics.append(final_cycle_stats)

    # Save overall simulation metrics for post-hoc analysis
    metrics_df = pd.DataFrame(simulation_metrics)
    metrics_df.to_csv(sim_results_dir / f'cycle_summaries.csv', index=False)
    
    # Generate combined plots (e.g., lines by day) summarizing key metrics.
    # analysis.generate_simulation_plots(
    #     metrics_df, output_dir=str(plots_dir), experiment_name=experiment_name)
    
    # Final dataset comparison
    print("\n" + "*"*80)
    print("FINAL DATASET COMPARISON")
    print("*"*80)
    
    # Generate task-level summary
    print("\nGenerating task-level summary...")
    
    # Initialize task summary data
    task_summaries = []
    
    # Process each unique task
    print("\nProcessing tasks...")
    for task_id in tqdm(full_data['task_id'].unique()):
        task_data = full_data[full_data['task_id'] == task_id]
        remaining_data = task_data[~task_data['was_pruned']]
        
        # Initialize task summary
        task_summary = {
            'task_id': task_id,
            'total_instances': len(task_data),
            'qualified_instances': task_data['is_qualified'].sum(),
            'pruned_instances': task_data['was_pruned'].sum(),
            'gt_mv': task_data['majority_vote'].iloc[0]  # Ground truth from full data
        }

        # Check if more instances were pruned than qualified
        if task_summary['pruned_instances'] > task_summary['qualified_instances']:
            print(f"\nWARNING: Task {task_id} has more pruned instances than qualified instances!")
            print(f"  Total instances: {task_summary['total_instances']}")
            print(f"  Qualified instances: {task_summary['qualified_instances']}")
            print(f"  Pruned instances: {task_summary['pruned_instances']}")
            
            # Get cycle information for this task
            cycle_info = task_data[task_data['cycle_num'] >= 0].groupby('cycle_num').agg({
                'is_qualified': 'sum',
                'was_pruned': 'sum',
                'predicted_proba': ['count', 'mean']
            })
            cycle_info.columns = ['qualified_count', 'pruned_count', 'total_count', 'mean_prob']
            
            print("\n  Cycle-by-cycle breakdown:")
            for cycle, stats in cycle_info.iterrows():
                print(f"    Cycle {cycle}:")
                print(f"      Total instances: {stats['total_count']}")
                print(f"      Qualified instances: {stats['qualified_count']}")
                print(f"      Pruned instances: {stats['pruned_count']}")
                print(f"      Mean prediction probability: {stats['mean_prob']:.3f}")
            
            # Show response distribution
            print("\n  Response distribution:")
            response_dist = task_data['response'].value_counts()
            for response, count in response_dist.items():
                print(f"    {response}: {count}")

        # Get simulation majority vote from non-pruned instances
        if len(remaining_data) > 0:
            response_counts = remaining_data['response'].value_counts()
            if len(response_counts) > 0:
                task_summary['sim_mv'] = response_counts.index[0]
            else:
                task_summary['sim_mv'] = None
        else:
            task_summary['sim_mv'] = None
            print(f"\nWARNING: Task {task_id} has no remaining responses after pruning!")
            print(f"  Total instances: {task_summary['total_instances']}")
            print(f"  Qualified instances: {task_summary['qualified_instances']}")
            print(f"  Pruned instances: {task_summary['pruned_instances']}")
            
            # Show original response distribution
            print("\n  Original response distribution:")
            response_dist = task_data['response'].value_counts()
            for response, count in response_dist.items():
                print(f"    {response}: {count}")
            
            # Show cycle information
            cycle_info = task_data[task_data['cycle_num'] >= 0].groupby('cycle_num').agg({
                'was_pruned': 'sum',
                'predicted_proba': ['count', 'mean']
            })
            cycle_info.columns = ['pruned_count', 'total_count', 'mean_prob']
            
            print("\n  Pruning by cycle:")
            for cycle, stats in cycle_info.iterrows():
                print(f"    Cycle {cycle}:")
                print(f"      Total instances: {stats['total_count']}")
                print(f"      Pruned instances: {stats['pruned_count']}")
                print(f"      Mean prediction probability: {stats['mean_prob']:.3f}")
        
        task_summaries.append(task_summary)
    
    # Create DataFrame and save
    tasks_df = pd.DataFrame(task_summaries)
    tasks_df.to_csv(sim_results_dir / f'tasks_summaries.csv', index=False)
    print(f"Saved task summaries to tasks_summaries.csv")
    
    # Calculate final metrics
    # Accuracy: percentage of tasks where simulation majority vote matches ground truth
    # Only consider tasks that were actually affected by pruning
    affected_tasks = tasks_df[tasks_df['pruned_instances'] > 0]
    total_accuracy = (affected_tasks['sim_mv'] == affected_tasks['gt_mv']).mean() * 100 if len(affected_tasks) > 0 else 100
    affected_tasks_accuracy = total_accuracy  # For affected tasks, this is the same
    
    # Task instance reduction: percentage of original task instances that were pruned
    total_instances = tasks_df['total_instances'].sum()
    total_pruned = tasks_df['pruned_instances'].sum()
    total_reduction = (total_pruned / total_instances) * 100
    
    # Qualified reduction: reduction rate among qualified instances
    total_qualified = tasks_df['qualified_instances'].sum()
    qualified_reduction = (total_pruned / total_qualified * 100) if total_qualified > 0 else 0
    
    print("\nRetention Rates:")
    for entity in ['user_id', 'crop_id', 'task_id']:
        retention = (training_data[entity].nunique() / full_data[entity].nunique() * 100)
        print(f"  - {entity.replace('_id', 's')} retained: {retention:.1f}%")
    
    print("\nQuality Metrics:")
    print(f"  - Overall dataset quality (accuracy): {total_accuracy:.1f}%")
    print(f"  - Affected dataset quality (accuracy): {affected_tasks_accuracy:.1f}% ({len(affected_tasks)} tasks)")
    print(f"  - Overall task instance reduction: {total_reduction:.1f}%")
    print(f"  - Qualified task instance reduction: {qualified_reduction:.1f}%")
    
    # Calculate binary classification metrics
    valid_tasks = (tasks_df['gt_mv'].isin(['yes', 'no'])) & (tasks_df['sim_mv'].notna())
    binary_metrics = calculate_binary_metrics(
        tasks_df.loc[valid_tasks, 'gt_mv'],
        tasks_df.loc[valid_tasks, 'sim_mv']
    )
    
    print("\nBinary Classification Metrics (treating 'yes' as positive class):")
    print(f"  - True Positive Rate (Recall): {100*binary_metrics['true_positive_rate']:.2f}%")
    print(f"  - False Positive Rate: {100*binary_metrics['false_positive_rate']:.2f}%")
    print(f"  - Precision: {100*binary_metrics['precision']:.2f}%")
    print(f"  - F1 Score: {100*binary_metrics['f1']:.2f}%")
    print("\nConfusion Matrix:")
    cm = binary_metrics['confusion_matrix']
    print(f"  True Negative: {cm['tn']}, False Positive: {cm['fp']}")
    print(f"  False Negative: {cm['fn']}, True Positive: {cm['tp']}")
    
    # Print diagnostic information
    print("\nDiagnostic Statistics:")
    print("Ground Truth Majority Vote Distribution:")
    gt_dist = tasks_df['gt_mv'].value_counts()
    for vote, count in gt_dist.items():
        print(f"  {vote}: {count} tasks ({count/len(tasks_df)*100:.1f}%)")
    
    print("\nSimulation Majority Vote Distribution (non-pruned tasks):")
    sim_dist = tasks_df[tasks_df['sim_mv'].notna()]['sim_mv'].value_counts()
    for vote, count in sim_dist.items():
        print(f"  {vote}: {count} tasks ({count/len(tasks_df)*100:.1f}%)")
    
    print("\nResponse Distribution in Original Dataset:")
    response_dist = full_data['response'].value_counts()
    for response, count in response_dist.items():
        print(f"  {response}: {count} instances ({count/len(full_data)*100:.1f}%)")
    
    print("\nResponse Distribution in Final Dataset:")
    final_response_dist = full_data[~full_data['was_pruned']]['response'].value_counts()
    for response, count in final_response_dist.items():
        print(f"  {response}: {count} instances ({count/len(full_data[~full_data['was_pruned']])*100:.1f}%)")
    
    # Print top majority vote errors
    print("\nTop 5 Majority Vote Errors:")
    print("-" * 80)
    error_tasks = tasks_df[
        (tasks_df['sim_mv'] != tasks_df['gt_mv']) & 
        (tasks_df['sim_mv'].notna()) & 
        (tasks_df['pruned_instances'] > 0)
    ].nlargest(5, 'pruned_instances')
    
    for _, task in error_tasks.iterrows():
        task_data = full_data[full_data['task_id'] == task['task_id']]
        original_responses = task_data['response'].value_counts().to_dict()
        remaining_responses = task_data[~task_data['was_pruned']]['response'].value_counts().to_dict()
        
        # Get question type if it's ECPD
        if dataset == 'ECPD':
            q_cols = [col for col in task_data.columns if col.startswith('q_')]
            question_type = task_data[q_cols].iloc[0].idxmax()
        else:
            question_type = "N/A"
        
        print(f"\nTask {task['task_id']} ({question_type}):")
        print(f"  Ground truth majority: {task['gt_mv']}")
        print(f"  Simulation majority: {task['sim_mv']}")
        print(f"  Original responses: {original_responses}")
        print(f"  Remaining responses: {remaining_responses}")
        print(f"  Total instances: {task['total_instances']}")
        print(f"  Pruned instances: {task['pruned_instances']}")
        print(f"  Qualified instances: {task['qualified_instances']}")

    # After processing all tasks, add summary debug
    print("\nDEBUG: Final Task Summary Statistics:")
    print(f"Total tasks processed: {len(tasks_df)}")
    print(f"Tasks with sim_mv set: {tasks_df['sim_mv'].notna().sum()}")
    print(f"Tasks with gt_mv set: {tasks_df['gt_mv'].notna().sum()}")
    print("\nMajority Vote Distribution:")
    print(tasks_df['sim_mv'].value_counts())
    print("\nGround Truth Distribution:")
    print(tasks_df['gt_mv'].value_counts())

if __name__ == "__main__":
    main() 