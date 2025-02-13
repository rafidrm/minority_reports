import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process answers data')
    
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Choose the dataset to process (ECPD or ZOD)')
    
    parser.add_argument('--unfilter', action='store_true', 
                        help='Do not filter workers or tasks')
    
    parser.add_argument('--gap_threshold', type=int, default=10,
                        help='Gap threshold in minutes for continuous activity')
    
    parser.add_argument('--filename', type=str, default='answers_revised.csv',
                        help='Filename to save the processed data')
    
    return parser.parse_args()

def load_data(dataset='ECPD'):
    """Load the answers CSV file based on the selected dataset"""
    if dataset == 'ECPD':
        filepath = 'data/ECPD/answers.csv'
    else:
        filepath = 'data/ZOD/answers.csv'
    
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def save_data(df, 
              dataset='ECPD', 
              unfilter=False, 
              filename='answers_revised.csv'):
    """Save the processed DataFrame in the appropriate directory"""
    if dataset == 'ECPD':
        if unfilter:
            filepath = 'data/ECPD/answers_unfiltered.csv'
        else:
            filepath = f'data/ECPD/{filename}'
    else:
        if unfilter:
            filepath = 'data/ZOD/answers_unfiltered.csv'
        else:
            filepath = f'data/ZOD/{filename}'
    
    print(f"Saving processed data to {filepath}...")
    df.to_csv(filepath, index=False)
    print("Save complete!")

def create_task_ids(df):
    """
    Create unique task_ids for each crop_id x question combination
    Returns DataFrame with new task_id column
    """
    print("Creating task IDs...")
    
    # Create a dictionary to store unique task_ids
    task_id_map = {}
    current_task_id = 0
    
    # Create a new column for task_ids
    df['task_id'] = -1
    
    # Use tqdm to show progress
    for idx in tqdm(df.index, desc="Assigning task IDs"):
        crop_id = df.at[idx, 'crop_id']
        question = df.at[idx, 'question']
        
        # Create tuple key
        key = (crop_id, question)
        
        # Assign task_id
        if key not in task_id_map:
            task_id_map[key] = current_task_id
            current_task_id += 1
        
        df.at[idx, 'task_id'] = task_id_map[key]
    
    return df

def remove_low_activity_workers(df, min_tasks=100):
    """
    Remove workers who have completed fewer than min_tasks different tasks
    
    Args:
        df: DataFrame containing the answers data
        min_tasks: Minimum number of tasks a worker must have completed to be kept
    
    Returns:
        DataFrame with only high-activity workers
    """
    print(f"Removing workers with fewer than {min_tasks} tasks...")
    
    # Count tasks per worker
    worker_tasks = df['user_id'].value_counts()
    
    # Get list of workers who meet the threshold
    active_workers = worker_tasks[worker_tasks >= min_tasks].index
    
    # Filter the dataframe
    df_filtered = df[df['user_id'].isin(active_workers)].copy()
    
    # Print statistics
    print(f"Removed {len(worker_tasks) - len(active_workers)} workers")
    print(f"Kept {len(active_workers)} workers")
    print(f"Reduced from {len(df)} to {len(df_filtered)} annotations")
    
    return df_filtered

def add_agreement_columns(df):
    """
    Add majority_vote and disagree columns to the DataFrame
    
    majority_vote: mode of answers for each task_id
    disagree: 1 if answer differs from majority or cant_solve is True, 0 otherwise
    
    Args:
        df: DataFrame containing the answers data with task_ids
    
    Returns:
        DataFrame with new columns
    """
    print("Adding majority vote and disagreement columns...")
    
    # Get majority vote for each task_id
    majority_votes = df.groupby('task_id').agg({
        'answer': lambda x: x.mode()[0]  # Get the most common answer
    }).reset_index()
    
    # Create majority_vote column by mapping from task_ids
    df['majority_vote'] = df['task_id'].map(majority_votes.set_index('task_id')['answer'])
    
    # Create disagree column
    df['disagree'] = ((df['answer'] != df['majority_vote']) | (df['cant_solve'] == True)).astype(int)
    
    # Print some statistics
    total_annotations = len(df)
    disagreements = df['disagree'].sum()
    print(f"Total disagreements: {disagreements} ({disagreements/total_annotations*100:.1f}% of annotations)")
    
    return df

def add_image_paths(df, boxes_path='data/ECPD/boxes.csv'):
    """
    Add image_path column by merging with boxes.csv on crop_id.
    
    Args:
        df: DataFrame containing the answers data
        boxes_path: Path to boxes.csv file
    
    Returns:
        DataFrame with new image_path column
    """
    print("Adding image paths from boxes.csv...")
    
    # Load boxes data
    boxes_df = pd.read_csv(boxes_path)
    
    # Keep only needed columns to make merge efficient
    boxes_df = boxes_df[['crop_id', 'image_path']]
    
    # Merge with answers data
    df = df.merge(boxes_df, on='crop_id', how='left')
    
    # Check for any missing paths
    missing_paths = df['image_path'].isna().sum()
    if missing_paths > 0:
        print(f"WARNING: Found {missing_paths} annotations without matching image paths")
    
    print(f"Added paths for {len(boxes_df['crop_id'].unique())} unique crops")
    
    return df

def check_task_ids(df):
    """Check statistics after creating task IDs"""
    print("\nChecking task ID creation:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of unique task_ids: {df['task_id'].nunique()}")
    print(f"Number of unique crop_id x question pairs: {len(df.groupby(['crop_id', 'question']))}")
    print(f"Average time per task: {df['duration_ms'].mean() / 1000:.2f} seconds")
    
    # Verify each task_id maps to exactly one crop_id x question pair
    task_combos = df.groupby('task_id').agg({
        'crop_id': 'nunique',
        'question': 'nunique'
    })
    invalid = task_combos[(task_combos['crop_id'] > 1) | (task_combos['question'] > 1)]
    if len(invalid) > 0:
        print("WARNING: Found task_ids mapping to multiple crop_id x question pairs!")
        print(invalid)

def check_worker_filtering(df_original, df_filtered):
    """Check statistics after filtering low-activity workers"""
    print("\nChecking worker filtering:")
    print(f"Original number of rows: {len(df_original)}")
    print(f"Filtered number of rows: {len(df_filtered)}")
    print(f"Original number of workers: {df_original['user_id'].nunique()}")
    print(f"Filtered number of workers: {df_filtered['user_id'].nunique()}")
    
    # Check that no new workers were introduced
    new_workers = set(df_filtered['user_id'].unique()) - set(df_original['user_id'].unique())
    if new_workers:
        print("WARNING: Found new workers after filtering!")
        print(new_workers)

def check_agreement_columns(df):
    """Check statistics after adding agreement columns"""
    print("\nChecking agreement columns:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of rows with majority_vote: {df['majority_vote'].count()}")
    print(f"Number of rows with disagree: {df['disagree'].count()}")
    
    # Check for any null values
    nulls = df[['majority_vote', 'disagree']].isnull().sum()
    if nulls.any():
        print("WARNING: Found null values in agreement columns!")
        print(nulls)
    
    # Verify disagree is binary
    invalid_disagree = df[~df['disagree'].isin([0, 1])]
    if len(invalid_disagree) > 0:
        print("WARNING: Found non-binary values in disagree column!")
        print(invalid_disagree['disagree'].unique())

def check_image_paths(df_before, df_after):
    """Check statistics after adding image paths"""
    print("\nChecking image paths:")
    print(f"Number of rows before: {len(df_before)}")
    print(f"Number of rows after: {len(df_after)}")
    print(f"Number of unique crop_ids: {df_after['crop_id'].nunique()}")
    print(f"Number of unique image_paths: {df_after['image_path'].nunique()}")
    
    # Check for missing paths
    missing = df_after['image_path'].isnull().sum()
    if missing > 0:
        print(f"WARNING: Found {missing} rows with missing image paths")
    
    # Check crop_id to image_path mapping
    path_per_crop = df_after.groupby('crop_id')['image_path'].nunique()
    multiple_paths = path_per_crop[path_per_crop > 1]
    if len(multiple_paths) > 0:
        print("WARNING: Found crop_ids mapping to multiple image paths!")
        print(multiple_paths)

def add_continuous_activity(df, gap_threshold=10):
    """
    Add columns tracking continuous activity time for each task.
    Continuous activity is reset when there's a break of 10+ minutes or a new day starts.
    
    Args:
        df: DataFrame with created_at timestamps
    
    Returns:
        DataFrame with new continuous_activity_seconds and continuous_activity_hours columns
    """
    print("Adding continuous activity tracking...")
    
    # Convert timestamps to datetime if not already
    df = df.copy()
    # Use format='ISO8601' to handle various ISO formats including timezone offsets
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
    
    # Sort by user_id and timestamp
    df = df.sort_values(['user_id', 'created_at'])
    
    # Initialize new columns
    df['continuous_activity_seconds'] = 0
    df['segment_id'] = -1  # For testing/verification
    
    # Process each user separately
    for user_id in tqdm(df['user_id'].unique(), desc="Processing users"):
        user_data = df[df['user_id'] == user_id].copy()
        
        # Get time differences between consecutive tasks
        time_diffs = user_data['created_at'].diff()
        
        # Start new segment when:
        # 1. First task of the sequence
        # 2. Gap of gap_threshold+ minutes
        # 3. New day
        new_segment = (
            (time_diffs.isna()) |  # First task
            (time_diffs > pd.Timedelta(minutes=gap_threshold)) |
            (user_data['created_at'].dt.date != user_data['created_at'].shift().dt.date)  # New day
        )
        
        # Assign segment IDs
        current_segment = 0
        segment_ids = []
        for is_new in new_segment:
            if is_new:
                current_segment += 1
            segment_ids.append(current_segment)
        user_data['segment_id'] = segment_ids
        
        # Calculate continuous activity time for each task
        for segment_id in user_data['segment_id'].unique():
            segment = user_data[user_data['segment_id'] == segment_id]
            segment_start = segment['created_at'].min()
            
            # Calculate time since segment start for each task
            activity_time = (segment['created_at'] - segment_start).dt.total_seconds()
            
            # Update the main dataframe
            df.loc[segment.index, 'continuous_activity_seconds'] = activity_time
            df.loc[segment.index, 'segment_id'] = segment_id
    
    # Convert to hours
    df['continuous_activity_hours'] = df['continuous_activity_seconds'] / 3600
    
    return df

def test_continuous_activity(df, gap_threshold=10):
    """
    Run tests to verify continuous activity calculations
    """
    print("\nTesting continuous activity calculations:")
    
    # Test 1: Check for negative values
    neg_times = df[df['continuous_activity_seconds'] < 0]
    if len(neg_times) > 0:
        print("WARNING: Found negative continuous activity times!")
        print(f"Number of negative values: {len(neg_times)}")
    else:
        print("✓ No negative activity times found")
    
    # Test 2: Check that times reset each day
    df['date'] = df['created_at'].dt.date
    max_times_per_day = df.groupby(['user_id', 'date'])['continuous_activity_seconds'].max()
    too_long = max_times_per_day[max_times_per_day > 24*3600]  # More than 24 hours
    if len(too_long) > 0:
        print("WARNING: Found continuous activity times longer than 24 hours!")
        print(f"Number of instances: {len(too_long)}")
    else:
        print("✓ All daily activity times are within 24 hours")
    
    # Test 3: Check for proper segmentation
    def check_segment(group):
        time_diffs = group['created_at'].diff()
        invalid_gaps = time_diffs[
            (time_diffs > pd.Timedelta(minutes=gap_threshold)) & 
                (group['segment_id'].shift() == group['segment_id'])
        ]
        return len(invalid_gaps)
    
    invalid_segments = df.groupby('user_id').apply(check_segment).sum()
    if invalid_segments > 0:
        print(f"WARNING: Found {invalid_segments} cases where {gap_threshold}+ minute gaps didn't start new segments!")
    else:
        print(f"✓ All {gap_threshold}+ minute gaps properly start new segments")
    
    # Test 4: Verify hours conversion
    hour_diffs = df['continuous_activity_hours'] - (df['continuous_activity_seconds'] / 3600)
    if not np.allclose(hour_diffs, 0):
        print("WARNING: Hours conversion is incorrect!")
    else:
        print("✓ Hours conversion is correct")
    
    print("\nExample segments for one user:")
    example_user = df['user_id'].iloc[0]
    example_data = df[df['user_id'] == example_user].sort_values('created_at').head(10)
    print(example_data[['created_at', 'segment_id', 'continuous_activity_seconds', 'continuous_activity_hours']])

def filter_low_repeat_tasks(df, min_repeats=5):
    """
    Remove tasks that have been completed by fewer than min_repeats different workers
    
    Args:
        df: DataFrame containing the answers data with task_ids
        min_repeats: Minimum number of different workers needed per task
    
    Returns:
        DataFrame with only tasks that have sufficient repeats
    """
    print(f"\nFiltering tasks with fewer than {min_repeats} workers...")
    print(f"Tasks before filtering: {df['task_id'].nunique()}")
    print(f"Rows before filtering: {len(df)}")
    
    # Count unique workers per task
    task_user_counts = df.groupby('task_id')['user_id'].nunique()
    valid_tasks = task_user_counts[task_user_counts >= min_repeats].index
    
    # Filter to keep only tasks with sufficient repeats
    df_filtered = df[df['task_id'].isin(valid_tasks)].copy()
    
    # Print statistics
    print(f"Tasks after filtering (≥{min_repeats} workers): {df_filtered['task_id'].nunique()}")
    print(f"Rows after filtering: {len(df_filtered)}")
    print("\nRepeats per task after filtering:")
    print(df_filtered.groupby('task_id')['user_id'].nunique().describe())
    
    return df_filtered

def check_task_filtering(df_before, df_after):
    """Check statistics after filtering low-repeat tasks"""
    print("\nChecking task filtering:")
    print(f"Tasks before filtering: {df_before['task_id'].nunique()}")
    print(f"Tasks after filtering: {df_after['task_id'].nunique()}")
    print(f"Rows before filtering: {len(df_before)}")
    print(f"Rows after filtering: {len(df_after)}")
    
    # Check repeats distribution
    repeats_before = df_before.groupby('task_id')['user_id'].nunique()
    repeats_after = df_after.groupby('task_id')['user_id'].nunique()
    
    print("\nRepeats per task before filtering:")
    print(repeats_before.describe())
    print("\nRepeats per task after filtering:")
    print(repeats_after.describe())
    
    # Verify minimum repeats
    min_repeats = repeats_after.min()
    if min_repeats < 5:
        print(f"WARNING: Found tasks with fewer than 5 repeats! Minimum is {min_repeats}")
    else:
        print("✓ All tasks have at least 5 repeats")
    
    # Check that no new tasks were introduced
    new_tasks = set(df_after['task_id'].unique()) - set(df_before['task_id'].unique())
    if new_tasks:
        print("WARNING: Found new tasks after filtering!")
        print(new_tasks)

def main():
    args = parse_args()
    
    # Load the data
    df = load_data(args.dataset)
    original_df = df.copy()
    
    # Create task IDs and check
    df = create_task_ids(df)
    check_task_ids(df)
    
    # Add image paths only if ECPD dataset
    if args.dataset == 'ECPD':
        df_before_paths = df.copy()
        df = add_image_paths(df)  # Call without dataset check
        check_image_paths(df_before_paths, df)
    
    if not args.unfilter:
        # Filter workers and check
        df_before_filter = df.copy()  # Store state before filtering
        df = remove_low_activity_workers(df)
        check_worker_filtering(df_before_filter, df)
    
    # Add agreement columns and check
    df = add_agreement_columns(df)
    check_agreement_columns(df)
    
    if not args.unfilter:
        # Filter tasks with insufficient repeats and check
        df_before_task_filter = df.copy()  # Store state before filtering
        df = filter_low_repeat_tasks(df)
        check_task_filtering(df_before_task_filter, df)
    
    # Add continuous activity tracking and check
    df = add_continuous_activity(df, args.gap_threshold)
    test_continuous_activity(df, args.gap_threshold)
    
    # Save the processed data
    save_data(df, args.dataset, args.unfilter, args.filename)
    
    print("\nFinal Statistics:")
    print(f"Original rows: {len(original_df)}")
    print(f"Final rows: {len(df)}")
    print(f"Original workers: {original_df['user_id'].nunique()}")
    print(f"Final workers: {df['user_id'].nunique()}")

if __name__ == "__main__":
    help_message = """
    This script processes the answers CSV file (answers.csv) to create a new CSV file (answers_revised.csv) with standard pre-processing to prepare for data exploration.

    The following columns are created:
    - task_id: A unique identifier for each task
    - crop_id: The ID of the crop associated with the task
    - question: The question associated with the task
    - answer: The answer provided by the worker
    - majority_vote: The majority vote for the task
    - disagree: 1 if the answer differs from the majority vote, 0 otherwise
    - image_path: The path to the image associated with the crop
    - continuous_activity_seconds: The number of seconds the worker has been active on the task
    - continuous_activity_hours: The number of hours the worker has been active on the task
    """
    print(help_message)
    main()
