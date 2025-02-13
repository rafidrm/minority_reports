import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for logistic regression')
    
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Choose the dataset to process (ECPD or ZOD)')
    
    parser.add_argument('--filename', type=str, default='answers_revised.csv',
                        help='Filename to load the processed data')
    
    parser.add_argument('--save_filename', type=str, 
                        default='answers_logreg_all.csv',
                        help='Filename to save the preprocessed data')
    
    return parser.parse_args()

def load_data(dataset='ECPD', filename='answers_revised.csv'):
    """Load the revised answers CSV file based on the selected dataset"""
    if dataset == 'ECPD':
        filepath = f'data/ECPD/{filename}'
    else:
        filepath = f'data/ZOD/{filename}'
    
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def save_data(df, dataset='ECPD', save_filename='answers_logreg_all.csv'):
    """Save the preprocessed DataFrame in the appropriate directory"""
    if dataset == 'ECPD':
        filepath = f'data/ECPD/{save_filename}'
    else:
        filepath = f'data/ZOD/{save_filename}'
    
    print(f"\nSaving preprocessed data to {filepath}...")
    df.to_csv(filepath, index=False)
    print("Save complete!")

def encode_categorical_columns(df, dataset='ECPD'):
    """
    Process categorical columns and dates for logistic regression
    """
    print("Processing columns...")
    df = df.copy()
    
    # Convert timestamps to datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
    df['date'] = df['created_at'].dt.date  # Ensure we have a date column
    
    # Ensure disagree column is int (1/0)
    df['disagree'] = df['disagree'].astype(int)
    
    # Define ID_COLS based on dataset
    if dataset == 'ECPD':
        ID_COLS = ['id', 'crop_id', 'user_id', 'image_path']
        
        # Create mapping for cleaner question names
        question_mapping = {
            'reflection of a person': 'reflection',
            'on poster/picture/billboard': 'poster',
            'in mirror/reflection': 'mirror',
            'on tv/screen': 'screen',
            # Add more mappings as needed
        }
        
        # Store original questions and their mappings
        original_questions = df['question'].unique()
        cleaned_questions = {}
        
        # Apply mapping and clean question names
        df['question_clean'] = df['question'].map(question_mapping).fillna(df['question'])
        df['question_clean'] = df['question_clean'].str.lower().str.replace(' ', '_').str.replace('/', '_')
        
        # Store the mappings before dropping columns
        for orig in original_questions:
            cleaned_questions[orig] = df.loc[df['question'] == orig, 'question_clean'].iloc[0]
        
        # Create one-hot encoding for question column with cleaner names and convert to int (1/0)
        question_dummies = pd.get_dummies(df['question_clean'], prefix='q').astype(int)
        
        # Add encoded question columns
        df = pd.concat([df, question_dummies], axis=1)
        
    else:  # For ZOD
        ID_COLS = ['id', 'crop_id', 'user_id']
        
        # Remove question column
        df = df.drop('question', axis=1)
    
    # Extract date components and create one-hot encoding for date
    date_dummies = pd.get_dummies(df['date'], prefix='date').astype(int)
    print(f"Created {len(date_dummies.columns)} date dummy variables")
    
    # Combine answer and cant_solve into single column
    df['response'] = df['answer']
    df.loc[df['cant_solve'] == True, 'response'] = 'cant-solve'
    
    # Round continuous activity
    df['continuous_activity_seconds'] = df['continuous_activity_seconds'].round(4)
    df['continuous_activity_hours'] = df['continuous_activity_hours'].round(8)
    
    # Add encoded date columns before dropping the original date column
    df = pd.concat([df, date_dummies], axis=1)
    
    # Remove unwanted columns, but keep 'created_at'
    columns_to_drop = ['question', 'question_clean', 'date', 'segment_id', 'duration_ms', 
                      'answer', 'cant_solve']
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Reorder columns
    date_cols = [col for col in df.columns if col.startswith('date_')]
    other_cols = [col for col in df.columns if col not in ID_COLS + date_cols]
    df = df[ID_COLS + other_cols + date_cols]  # Put date columns at the end
    
    return df, ID_COLS

def check_encoding(df_original, df_encoded, ID_COLS):
    """
    Verify the categorical encoding and data formatting
    """
    print("\nChecking data processing:")
    
    # Check row counts
    print(f"Original rows: {len(df_original)}")
    print(f"Processed rows: {len(df_encoded)}")
    
    if len(df_original) != len(df_encoded):
        print("WARNING: Row count changed during processing!")
    
    # Check question encoding
    original_questions = df_original['question'].unique()
    question_columns = [col for col in df_encoded.columns if col.startswith('q_')]
    print(f"\nOriginal question categories: {len(original_questions)}")
    print(f"Encoded question columns: {len(question_columns)}")
    
    # Verify each row has exactly one question encoded
    question_sums = df_encoded[question_columns].sum(axis=1)
    invalid_questions = question_sums[question_sums != 1]
    if len(invalid_questions) > 0:
        print(f"WARNING: Found {len(invalid_questions)} rows with invalid question encoding!")
    
    # Check response categories
    print("\nResponse categories:", sorted(df_encoded['response'].unique()))
    
    # Verify continuous activity precision
    max_decimals = df_encoded['continuous_activity_seconds'].astype(str).str.split('.').str[1].str.len().max()
    if max_decimals > 4:
        print(f"WARNING: Found continuous_activity_seconds with more than 4 decimal places!")
    
    # Print sample rows for verification
    print("\nSample of processed data:")
    sample_cols = ID_COLS + ['response', 'continuous_activity_seconds'] + question_columns[:2]
    print(df_encoded[sample_cols].head())

def main():
    args = parse_args()
    
    # Load the data
    df = load_data(args.dataset, args.filename)
    df_original = df.copy()
    
    # Apply preprocessing steps
    df, ID_COLS = encode_categorical_columns(df, args.dataset)
    check_encoding(df_original, df, ID_COLS)
    
    # Save the preprocessed data
    save_data(df, args.dataset, args.save_filename)
    
    print("\nFinal Statistics:")
    print(f"Original columns: {len(df_original.columns)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Number of rows: {len(df)}")

if __name__ == "__main__":
    help_message = """
    This script processes the data-preprocessed CSV file (answers_revised.csv) to create a new CSV file (answers_logreg_all.csv) with logistic regression-specific pre-processing. 

    Usage instructions:
    - First, run process_answers.py to create the answers_revised.csv file.
    - Then, run this script to create the answers_logreg_all.csv file.

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
