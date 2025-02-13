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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simulate removal of tasks predicted to be disagreement points.'
    )
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Dataset to use')
    parser.add_argument('--initial_training_hours', type=float, default=24,
                        help='Initial training period (hours)')
    parser.add_argument('--recalibration_frequency_hours', type=float, default=6,
                        help='Frequency of recalibration (hours)')
    parser.add_argument('--prune_threshold', type=float, default=0.1,
                        help='Probability threshold for pruning')
    parser.add_argument('--pruning_strategy', type=str, default='default',
                        help='Pruning strategy identifier')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment (used in output files)')
    parser.add_argument('--debug', action='store_true', help='Run simulation in debug mode (reduce dataset size by 50%)')
    return parser.parse_args()

def calculate_cycle_metrics(test_data, pruned_data, cycle_num, 
                            cumulative_stats, training_data, threshold, 
                            full_data):
    """Calculate and format metrics for the current cycle."""
    def calculate_confusion_metrics(y_true, y_pred):
        """Calculate TPR and FPR from binary predictions"""
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        true_neg = np.sum((y_true == 0) & (y_pred == 0))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
        return tpr, fpr
    
    def evaluate_majority_vote_quality(df):
        """Compare training set majority votes with gold standard majority votes"""
        # Get gold standard majority votes for each task
        gold_majority = full_data.groupby('task_id')['response'].agg(
            lambda x: pd.Series.mode(x)[0]).reset_index()
        gold_majority = gold_majority.set_index('task_id')['response']
        
        # Get current majority votes
        current_majority = df.groupby('task_id')['majority_vote'].first()
        
        # Find common tasks and compare votes
        common_tasks = current_majority.index.intersection(gold_majority.index)
        total_tasks = len(common_tasks)
        disagreements = np.sum(current_majority[common_tasks] != gold_majority[common_tasks])
        
        return total_tasks, disagreements
    
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
    
    # Calculate dataset quality metrics (using cumulative training data)
    total_tasks, wrong_tasks = evaluate_majority_vote_quality(training_data)
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
        f"  - Error rate: {error_rate:.1f}%",
        f"\nCumulative Statistics:",
        f"  - Total processed: {cumulative_stats['total_processed']}",
        f"  - Total qualified: {cumulative_stats['total_qualified']}",
        f"  - Total disqualified: {cumulative_stats['total_disqualified']}",
        f"  - Total pruned: {cumulative_stats['total_pruned']}",
        f"  - Overall prune rate: {overall_prune_rate:.1f}%",
        f"  - Current error rate: {cumulative_stats['current_error_rate']:.1f}%",
        "*"*80 + "\n"
    ]
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
    
    if args.debug:
        print("DEBUG MODE: Using 50% of full dataset")
        full_data = full_data.sample(frac=0.5, random_state=42)
    
    # Sort by created_at. 
    # Note: more realistic work might use created_at+duration_ms.
    full_data.sort_values('created_at', inplace=True)
    
    # Use initial training period (e.g., first 24 hours)
    # Get the date of the first annotation and set time to midnight
    first_date = pd.to_datetime(full_data['created_at'].min()).date()
    # Create timezone-aware timestamp at midnight
    start_time = pd.Timestamp(first_date).tz_localize('UTC').replace(
        hour=0, minute=0, second=0, microsecond=0)
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
        
        # If no test data in this window, continue to next cycle
        if len(test_data) == 0:
            print("\nNo data in current time window. Moving to next cycle.")
            current_time = next_time
            continue
            
        # Remove testing period rows from remaining_data
        remaining_data = remaining_data[~test_mask].copy()
        
        # Combine test data with training data to obtain ground truth votes at 
        # current time. This is only needed to measure accuracy of this cycle's 
        # model predictions.
        combined_data = pd.concat([training_data, test_data], ignore_index=True)
        combined_data = data_utils.update_votes(combined_data)
        
        # Extract updated test data with votes computed on all available data
        test_data = combined_data[
            combined_data['created_at'].between(current_time, next_time)].copy()
        
        # Debug: Check for new entities in test data
        # *****************************************************
        # *****************************************************
        print(f"\nDEBUG (Cycle {cycle_num}):")
        # Get unique entities
        entities = {
            'workers': {'test': set(test_data['user_id'].unique().tolist()),
                       'train': set(training_data['user_id'].unique().tolist())},
            'crops': {'test': set(test_data['crop_id'].unique().tolist()),
                     'train': set(training_data['crop_id'].unique().tolist())},
            'tasks': {'test': set(test_data['task_id'].unique().tolist()),
                     'train': set(training_data['task_id'].unique().tolist())}
        }
        
        # Calculate new entities
        new_entities = {
            'workers': entities['workers']['test'] - entities['workers']['train'],
            'crops': entities['crops']['test'] - entities['crops']['train'],
            'tasks': entities['tasks']['test'] - entities['tasks']['train']
        }
        
        # Print summary
        for entity_type in ['workers', 'crops', 'tasks']:
            print(f"\n{entity_type.title()}:")
            print(f"  Test set unique: {len(entities[entity_type]['test'])}")
            print(f"  Training set unique: {len(entities[entity_type]['train'])}")
            print(f"  New in test: {len(new_entities[entity_type])} " +
                  f"{list(new_entities[entity_type])[:5] if new_entities[entity_type] else 'None'}")
        
        # Analyze question distribution in testing data
        print("\nQuestion Distribution in Testing Data:")
        q_cols = [col for col in test_data.columns if col.startswith('q_')]
        
        # For each task, find which question it corresponds to
        task_questions = test_data.groupby('task_id')[q_cols].first()
        question_counts = task_questions.idxmax(axis=1).value_counts()  # type: ignore
        total_tasks = len(task_questions)
        
        # Print distribution
        print("\nTask distribution across questions:")
        for question, count in question_counts.items():
            percentage = (count / total_tasks) * 100
            print(f"  {question}: {count} tasks ({percentage:.1f}%)")
        # *****************************************************
        # *****************************************************
        
        # Recalculate majority vote/disagree for current training set
        training_data = data_utils.update_votes(training_data)
        
        # Fit simulation mixed-effects model (using only q_ predictors + 
        # worker and crop REs)
        cycle_parameters_dir = parameters_dir / f'cycle_{cycle_num}'
        cycle_parameters_dir.mkdir(exist_ok=True)
        sim_model = disagreement_simulation.fit_simulation_model(
            training_data, 
            dataset=dataset, 
            output_dir=str(cycle_parameters_dir))
        
        # Predict for testing data (perform logistic transformation on linear 
        # predictor)
        test_data['predicted_proba'] = disagreement_simulation.predict_simulation_proba(test_data, sim_model)
        
        # For rows with new worker/crop/question, force predicted probability 
        # to -1 so that they will not be pruned.
        seen_mask = data_utils.is_seen(test_data, training_data)
        test_data.loc[~seen_mask, 'predicted_proba'] = -1.0
        
        # Track qualification status and cycle number
        test_data['is_qualified'] = seen_mask
        test_data['cycle_num'] = cycle_num
        
        # Prune test data using the default pruning strategy
        pruned_data, cycle_stats = pruning.prune_data(
            test_data, threshold=args.prune_threshold,
            strategy=args.pruning_strategy)
        
        # Analyze pruned data distribution
        print("\nPRUNED DATA ANALYSIS:")
        
        if len(pruned_data) == 0:
            print("\nNo assignments were pruned in this cycle.")
            print("*"*80)
        else:
            # Question distribution in pruned data
            q_cols = [col for col in pruned_data.columns if col.startswith('q_')]
            pruned_questions = pruned_data[q_cols].idxmax(axis=1).value_counts()  # type: ignore
            total_pruned = len(pruned_data)
            
            print("\nPruned assignments by question type:")
            for question, count in pruned_questions.items():
                percentage = (count / total_pruned) * 100
                # Calculate what percentage of all assignments for this question type were pruned
                question_total = test_data[
                    test_data[q_cols].idxmax(axis=1) == question].shape[0]
                prune_rate = (count / question_total * 100) if question_total > 0 else 0
                print(f"  {question}: {count} assignments ({percentage:.1f}% of pruned, {prune_rate:.1f}% of this question's assignments)")
            
            # Worker distribution in pruned data
            pruned_workers = pruned_data['user_id'].value_counts()
            print("\nTop 10 workers in pruned data:")
            for worker_id, count in pruned_workers.head(10).items():
                worker_total = test_data[test_data['user_id'] == worker_id].shape[0]
                prune_rate = (count / worker_total * 100)
                print(f"  Worker {worker_id}: {count} assignments pruned ({prune_rate:.1f}% of their assignments)")
            
            # Task distribution in pruned data
            pruned_tasks = pruned_data['task_id'].value_counts()
            print(f"\nUnique tasks affected: {len(pruned_tasks)}")
            print("Top 10 most pruned tasks:")
            for task_id, count in pruned_tasks.head(10).items():
                task_total = test_data[test_data['task_id'] == task_id].shape[0]
                prune_rate = (count / task_total * 100)
                # Get the question type for this task
                task_question = pruned_data[pruned_data['task_id'] == task_id][q_cols].iloc[0].idxmax()
                print(f"  Task {task_id} ({task_question}): {count} assignments pruned ({prune_rate:.1f}% of task's assignments)")
            
            print("\nSummary statistics for pruned assignments:")
            print(f"  Total assignments pruned: {total_pruned}")
            print(f"  Unique workers affected: {len(pruned_workers)}")
            print(f"  Unique tasks affected: {len(pruned_tasks)}")
            print(f"  Average assignments pruned per affected worker: {total_pruned/len(pruned_workers):.1f}")
            print(f"  Average assignments pruned per affected task: {total_pruned/len(pruned_tasks):.1f}")
            print("*"*80)
        
        cycle_stats['cycle_num'] = cycle_num
        cycle_stats['start_time'] = current_time
        cycle_stats['end_time'] = next_time
        simulation_metrics.append(cycle_stats)
        
        # The surviving testing data (i.e. not pruned) are added to training 
        # dataset for next cycle.
        kept_test_data = test_data.drop(pruned_data.index)
        training_data = pd.concat([training_data, kept_test_data], ignore_index=True)
        
        # Optionally write out cycle testing data for record
        cycle_file = sim_results_dir / f'cycle_{cycle_num}.csv'
        test_data.to_csv(cycle_file, index=False)
        
        # Calculate and print cycle metrics
        cycle_summary = calculate_cycle_metrics(
            test_data, pruned_data, cycle_num, cumulative_stats, training_data, 
            args.prune_threshold, full_data)
        print(cycle_summary)
        
        cycle_num += 1
        current_time = next_time

    # Process any remaining data in the final cycle
    if not remaining_data.empty:
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
        
        # Handle new entities
        seen_mask = data_utils.is_seen(final_test_data, training_data)
        final_test_data.loc[~seen_mask, 'predicted_proba'] = -1.0
        
        # Prune final test data
        final_pruned_data, final_cycle_stats = pruning.prune_data(
            final_test_data, threshold=args.prune_threshold,
            strategy=args.pruning_strategy)
        
        # Add kept data to training set
        final_kept_data = final_test_data.drop(final_pruned_data.index)
        training_data = pd.concat([training_data, final_kept_data], ignore_index=True)
        
        # Update cumulative stats
        cycle_stats['cycle_num'] = cycle_num
        simulation_metrics.append(cycle_stats)

    # Save overall simulation metrics for post-hoc analysis
    metrics_df = pd.DataFrame(simulation_metrics)
    metrics_df.to_csv(sim_results_dir / f'cycle_summaries.csv', index=False)
    
    # Generate combined plots (e.g., lines by day) summarizing key metrics.
    analysis.generate_simulation_plots(metrics_df, output_dir=str(plots_dir), 
                                     experiment_name=experiment_name)
    
    # Final dataset comparison
    print("\n" + "*"*80)
    print("FINAL DATASET COMPARISON")
    print("*"*80)
    
    # Generate task-level summary
    print("\nGenerating task-level summary...")
    
    # Get question columns
    q_cols = [col for col in full_data.columns if col.startswith('q_')]
    
    # Read and pre-process cycle files
    print("\nReading cycle files...")
    cycle_files = []
    for i in range(1, cycle_num + 1):
        cycle_file = sim_results_dir / f'cycle_{i}.csv'
        if cycle_file.exists():
            # Only read necessary columns to save memory
            cycle_files.append(pd.read_csv(cycle_file, 
                usecols=['task_id', 'user_id', 'is_qualified', 'predicted_proba'] + q_cols))
    
    all_pruned_data = pd.concat(cycle_files, ignore_index=True) if cycle_files else pd.DataFrame()
    print(f"Loaded {len(cycle_files)} cycle files with {len(all_pruned_data)} total pruned assignments")
    
    # Pre-compute pruning mask
    print("\nPre-computing task statistics...")
    pruned_mask = all_pruned_data['predicted_proba'] >= args.prune_threshold
    
    # Merge full data with pruning data in one go
    merged_data = full_data.merge(
        all_pruned_data[['task_id', 'user_id', 'is_qualified']], 
        on=['task_id', 'user_id'], 
        how='left'
    )
    
    # Pre-compute task-level statistics
    task_stats = merged_data.groupby('task_id').agg({
        'crop_id': 'first',
        'user_id': 'count',  # num_repeats
        'is_qualified': lambda x: x.fillna(False).eq(False).sum()  # num_disqualified
    }).reset_index()
    
    # Add question columns
    task_meta = merged_data.groupby('task_id')[q_cols].first()
    task_stats = task_stats.join(task_meta, on='task_id')
    
    # Calculate pruned counts per task
    pruned_counts = all_pruned_data[pruned_mask].groupby('task_id').size()
    task_stats['num_pruned'] = task_stats['task_id'].map(pruned_counts).fillna(0)
    
    # Calculate remaining instances
    task_stats['num_left'] = task_stats['user_id'] - task_stats['num_pruned']
    
    # Calculate majority votes
    print("\nComputing majority votes...")
    # Ground truth majority votes
    gt_votes = merged_data.groupby('task_id')['response'].agg(lambda x: x.mode().iloc[0])
    task_stats['gt_mv'] = task_stats['task_id'].map(gt_votes)
    
    # Simulation majority votes (from non-pruned instances)
    pruned_users = all_pruned_data[pruned_mask].groupby('task_id')['user_id'].agg(set)
    
    def get_sim_mv(group):
        task_id = group.name
        pruned = pruned_users.get(task_id, set())
        remaining = group[~group['user_id'].isin(pruned)]
        return remaining['response'].mode().iloc[0] if not remaining.empty else None
    
    sim_votes = merged_data.groupby('task_id').apply(get_sim_mv)
    task_stats['sim_mv'] = task_stats['task_id'].map(sim_votes)
    
    # Save results
    task_stats.to_csv(sim_results_dir / f'tasks_summaries.csv', index=False)
    print(f"Saved task summaries to tasks_summaries.csv")
    
    # Calculate final metrics
    print("\nCalculating final metrics...")
    total_accuracy = (task_stats['sim_mv'] == task_stats['gt_mv']).mean() * 100
    
    affected_tasks = task_stats[task_stats['num_pruned'] > 0]
    affected_tasks_accuracy = (affected_tasks['sim_mv'] == affected_tasks['gt_mv']).mean() * 100 if len(affected_tasks) > 0 else 0
    
    total_instances = task_stats['user_id'].sum()
    remaining_instances = task_stats['num_left'].sum()
    total_reduction = (1 - remaining_instances / total_instances) * 100
    
    total_qualified = total_instances - task_stats['is_qualified'].sum()
    remaining_qualified = remaining_instances - task_stats['is_qualified'].sum()
    qualified_reduction = (1 - remaining_qualified / total_qualified) * 100 if total_qualified > 0 else 100
    
    # Compare statistics
    print("\nRetention Rates:")
    for entity in ['user_id', 'crop_id', 'task_id']:
        retention = (training_data[entity].nunique() / full_data[entity].nunique() * 100)
        print(f"  - {entity.replace('_id', 's')} retained: {retention:.1f}%")
    
    print("\nQuality Metrics:")
    print(f"  - Overall dataset quality (accuracy): {total_accuracy:.1f}%")
    print(f"  - Affected dataset quality (accuracy): {affected_tasks_accuracy:.1f}% ({len(affected_tasks)} tasks)")
    print(f"  - Overall task instance reduction: {total_reduction:.1f}%")
    print(f"  - Qualified task instance reduction: {qualified_reduction:.1f}%")

if __name__ == "__main__":
    main() 