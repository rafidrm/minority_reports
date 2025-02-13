import pandas as pd
import numpy as np


def update_votes(df):
    """
    Given a cumulative dataset, update the majority vote and disagree columns.
    We use the task_id column which uniquely identifies each task.
    
    Ties are resolved by a random decision.
    """
    # Ensure there is a 'response' column for later processing.
    if 'response' not in df.columns:
        raise KeyError("'response' column not found in the dataset.")

    df = df.copy()
    
    # For each task_id compute the majority vote from the 'response' column.
    def majority_vote(series):
        counts = series.value_counts()
        max_count = counts.max()
        modes = counts[counts == max_count].index.tolist()
        return np.random.choice(modes)
    
    majority = df.groupby('task_id')['response'].transform(majority_vote)
    df['majority_vote'] = majority
    df['disagree'] = (df['response'] != df['majority_vote']).astype(int)
    
    # Verify we created some disagreements
    if df['disagree'].nunique() <= 1:
        print("Warning: No disagreements found in this time window")
    
    return df

def is_seen(test_df, training_df):
    """
    For each row in test_df, return True if ALL of the following are true:
    1. Its user_id has been seen in training_df
    2. Its crop_id has been seen in training_df
    3. Its task_id has been seen in training_df
    4. Its question type has been seen in training_df
    Return False if ANY of these conditions are not met.
    """
    seen_workers = set(training_df['user_id'].unique())
    seen_crops = set(training_df['crop_id'].unique())
    seen_tasks = set(training_df['task_id'].unique())
    q_cols = [col for col in training_df.columns if col.startswith('q_')]
    training_q_types = set(training_df[q_cols].idxmax(axis=1).unique())
    
    test_workers = test_df['user_id']
    test_crops = test_df['crop_id']
    test_tasks = test_df['task_id']
    test_q_types = test_df[q_cols].idxmax(axis=1)
    
    # Return True only if ALL entities have been seen before
    is_seen = (test_workers.isin(seen_workers) & 
               test_crops.isin(seen_crops) & 
               test_tasks.isin(seen_tasks) &
               test_q_types.isin(training_q_types))
    return is_seen 