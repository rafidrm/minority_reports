import pandas as pd
import numpy as np


def print_no_disagreement_diagnostics(df):
    """
    Print detailed diagnostic information when no disagreements are found.
    Called when a time window shows perfect agreement among workers.
    """
    print("\n" + "*"*80)
    print("DIAGNOSTIC INFORMATION FOR NO DISAGREEMENTS WINDOW")
    print("*"*80)
    
    # Basic dataset stats
    print("\nBasic Statistics:")
    print(f"Total annotations: {len(df)}")
    print(f"Unique workers: {df['user_id'].nunique()}")
    print(f"Unique tasks: {df['task_id'].nunique()}")
    print(f"Unique crops: {df['crop_id'].nunique()}")
    
    # Worker statistics
    worker_tasks = df.groupby('user_id').size()
    print("\nWorker Activity:")
    print(f"Tasks per worker:")
    print(f"  Mean: {worker_tasks.mean():.1f}")
    print(f"  Median: {worker_tasks.median():.1f}")
    print(f"  Min: {worker_tasks.min()}")
    print(f"  Max: {worker_tasks.max()}")
    print("\nMost active workers (top 5):")
    for user_id, count in worker_tasks.nlargest(5).items():
        print(f"  Worker {user_id}: {count} tasks")
    
    # Task statistics
    task_repeats = df.groupby('task_id').size()
    print("\nTask Repeats:")
    print(f"Repeats per task:")
    print(f"  Mean: {task_repeats.mean():.1f}")
    print(f"  Median: {task_repeats.median():.1f}")
    print(f"  Min: {task_repeats.min()}")
    print(f"  Max: {task_repeats.max()}")
    
    # Response distribution
    print("\nResponse Distribution:")
    response_dist = df['response'].value_counts()
    for response, count in response_dist.items():
        print(f"  {response}: {count} ({count/len(df)*100:.1f}%)")
    
    # Question type distribution (if ECPD)
    q_cols = [col for col in df.columns if col.startswith('q_')]
    if q_cols:
        print("\nQuestion Type Distribution:")
        question_dist = df[q_cols].idxmax(axis=1).value_counts()
        for question, count in question_dist.items():
            print(f"  {question}: {count} ({count/len(df)*100:.1f}%)")
    
    # Time window analysis
    df['created_at'] = pd.to_datetime(df['created_at'])
    print("\nTime Window:")
    print(f"Start: {df['created_at'].min()}")
    print(f"End: {df['created_at'].max()}")
    print(f"Duration: {(df['created_at'].max() - df['created_at'].min()).total_seconds()/3600:.1f} hours")
    
    # Hourly annotation rate
    df['hour'] = df['created_at'].dt.hour
    hourly_rate = df.groupby('hour').size()
    print("\nHourly Annotation Rate:")
    print(f"  Mean: {hourly_rate.mean():.1f}")
    print(f"  Max: {hourly_rate.max()} (Hour {hourly_rate.idxmax()})")
    print(f"  Min: {hourly_rate.min()} (Hour {hourly_rate.idxmin()})")
    
    print("\n" + "*"*80 + "\n")

def majority_vote(series):
    """
    Calculate the majority vote from a series of responses.
    
    Rules:
    1. If all responses are 'cant-solve', return 'cant-solve'
    2. Otherwise, exclude 'cant-solve' responses and find the most common response
    3. For a response to be majority, it must have more votes than any other response
    4. In case of ties between 'yes' and 'no', always return 'no'
    
    Note: The returned majority vote will never be 'cant-solve' unless ALL responses 
    were 'cant-solve', since 'cant-solve' responses are filtered out before calculating
    the majority among valid responses.
    
    Args:
        series: pandas.Series containing responses for a single task
        
    Returns:
        str: The majority vote response. Will be either:
            - 'cant-solve' if all responses were 'cant-solve'
            - 'no' if there's a tie between 'yes' and 'no'
            - The response with the most votes otherwise
    """
    # Count all responses including cant-solve
    total_counts = series.value_counts()
    
    # If all responses are cant-solve, return cant-solve
    if len(total_counts) == 1 and total_counts.index[0] == 'cant-solve':
        return 'cant-solve'
    
    # Filter out cant-solve for majority calculation
    valid_responses = series[series != 'cant-solve']
    if len(valid_responses) == 0:
        return 'cant-solve'
    
    # Get counts of valid responses
    counts = valid_responses.value_counts()
    max_count = counts.max()
    
    # Get responses with maximum count
    modes = counts[counts == max_count].index.tolist()
    
    # If there's only one mode, that's our majority
    if len(modes) == 1:
        return modes[0]
    # If 'no' is one of the modes, return 'no'
    elif 'no' in modes:
        return 'no'
    # Otherwise return the first mode (alphabetically)
    else:
        return modes[0]

def update_votes(df, verbose=True):
    """
    Given a cumulative dataset, update the majority vote and disagree columns.
    We use the task_id column which uniquely identifies each task.
    
    For ties between 'yes' and 'no', we always choose 'no' as the majority vote.
    Cant-solve responses are automatically treated as disagreements.
    
    Args:
        df: DataFrame containing task responses
        verbose: If True, print diagnostics when no disagreements are found
        
    Returns:
        DataFrame with updated majority_vote and disagree columns
    """
    # Ensure there is a 'response' column for later processing.
    if 'response' not in df.columns:
        raise KeyError("'response' column not found in the dataset.")

    df = df.copy()
    
    # Apply majority vote calculation to each task
    majority = df.groupby('task_id')['response'].transform(majority_vote)
    df['majority_vote'] = majority
    
    # Set disagree to 1 for cant-solve responses and normal disagreements
    df['disagree'] = ((df['response'] != df['majority_vote']) | 
                      (df['response'] == 'cant-solve')).astype(int)
    
    # Verify we created some disagreements
    if verbose == True:
        if df['disagree'].nunique() <= 1:
            print("Warning: No disagreements found in this time window")
            print_no_disagreement_diagnostics(df)
    
    return df

def is_seen(test_df, training_df, strategy='default', dataset='ECPD'):
    """
    Determine which instances are qualified for pruning based on strategy.
    Each strategy defines its own complete set of conditions.
    
    Args:
        test_df: DataFrame of instances to check
        training_df: DataFrame of previously seen instances
        strategy: Pruning strategy to use ('default' or 'allow_workers')
        dataset: Dataset being processed ('ECPD' or 'ZOD')
    
    Returns:
        Series of booleans indicating which instances are qualified for pruning
    """
    # Common sets that will be needed by most strategies
    seen_workers = set(training_df['user_id'].unique())
    seen_crops = set(training_df['crop_id'].unique())
    seen_tasks = set(training_df['task_id'].unique())
    
    test_workers = test_df['user_id']
    test_crops = test_df['crop_id']
    test_tasks = test_df['task_id']

    if dataset == 'ECPD':
        # For ECPD, also check question types
        q_cols = [col for col in training_df.columns if col.startswith('q_')]
        training_q_types = set(training_df[q_cols].idxmax(axis=1).unique())
        test_q_types = test_df[q_cols].idxmax(axis=1)
        
        if strategy == 'default':
            # Default strategy requires all entities to be seen
            return (test_workers.isin(seen_workers) & 
                    test_crops.isin(seen_crops) & 
                    test_tasks.isin(seen_tasks) &
                    test_q_types.isin(training_q_types))
                    
        elif strategy == 'allow_workers':
            # Allow workers strategy ignores worker history
            return (test_crops.isin(seen_crops) & 
                    test_tasks.isin(seen_tasks) &
                    test_q_types.isin(training_q_types))
    else:  # ZOD
        # For ZOD, only check workers, crops, and tasks
        if strategy == 'default':
            # Default strategy requires all entities to be seen
            return (test_workers.isin(seen_workers) & 
                    test_crops.isin(seen_crops) & 
                    test_tasks.isin(seen_tasks))
                    
        elif strategy == 'allow_workers':
            # Allow workers strategy ignores worker history
            return (test_crops.isin(seen_crops) & 
                    test_tasks.isin(seen_tasks))
    
    raise ValueError(f"Unknown pruning strategy: {strategy}") 