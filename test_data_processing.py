import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate # type: ignore
import textwrap

def load_datasets():
    """Load both original and processed datasets"""
    print("Loading datasets...")
    df_orig = pd.read_csv('data/answers_ids_fixed.csv')
    df_proc = pd.read_csv('data/answers_logreg_all.csv')
    return df_orig, df_proc

def print_section_header(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def check_missing_ids(df_orig, df_proc, id_col, name):
    """Check for missing IDs between datasets"""
    print_section_header(f"{name} ID Analysis")
    
    orig_ids = set(df_orig[id_col].unique())
    proc_ids = set(df_proc[id_col].unique())
    missing_ids = orig_ids - proc_ids
    
    # Create summary table
    summary_data = [
        ["Original unique IDs", len(orig_ids)],
        ["Processed unique IDs", len(proc_ids)],
        ["Missing IDs", len(missing_ids)]
    ]
    print(tabulate(summary_data, headers=["Metric", "Count"], tablefmt="grid"))
    
    if missing_ids:
        # Sample of missing IDs
        sample_ids = sorted(list(missing_ids))[:5]
        print(f"\nSample of missing {name} IDs:")
        print(tabulate([[id_] for id_ in sample_ids], headers=[f"{name} ID"], tablefmt="grid"))
        
        # Sample rows for missing IDs
        sample_rows = df_orig[df_orig[id_col].isin(sample_ids)]
        print(f"\nSample rows from original data for missing {name}s:")
        print(tabulate(sample_rows[['user_id', 'crop_id', 'question', 'answer', 'cant_solve']], 
                      headers='keys', tablefmt="grid", showindex=False))
    
    return missing_ids

def check_na_values(df_proc):
    """Check for NA values in processed dataset"""
    print_section_header("NA Value Analysis")
    
    na_counts = df_proc.isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    
    if len(cols_with_na) > 0:
        na_data = [[col, count, f"{count/len(df_proc)*100:.2f}%"] 
                  for col, count in cols_with_na.items()]
        print(tabulate(na_data, 
                      headers=["Column", "NA Count", "Percentage"], 
                      tablefmt="grid"))
    else:
        print("No NA values found in processed dataset")

def compare_column_stats(df_orig, df_proc):
    """Compare summary statistics for common numerical columns"""
    print_section_header("Column Statistics Comparison")
    
    orig_num = df_orig.select_dtypes(include=[np.number]).columns
    proc_num = df_proc.select_dtypes(include=[np.number]).columns
    common_cols = list(set(orig_num) & set(proc_num))
    
    for col in common_cols:
        print(f"\nColumn: {col}")
        orig_stats = df_orig[col].describe()
        proc_stats = df_proc[col].describe()
        stats_diff = proc_stats - orig_stats
        
        comparison = pd.DataFrame({
            'Original': orig_stats,
            'Processed': proc_stats,
            'Difference': stats_diff
        })
        print(tabulate(comparison, headers='keys', tablefmt="grid", floatfmt=".3f"))

def check_response_distribution(df_orig, df_proc):
    """Compare response distributions between datasets"""
    print_section_header("Response Distribution Analysis")
    
    # Original distributions
    orig_answers = df_orig['answer'].value_counts()
    orig_cant_solve = df_orig['cant_solve'].value_counts()
    proc_response = df_proc['response'].value_counts()
    
    print("Original Answer Distribution:")
    print(tabulate(orig_answers.reset_index(), 
                  headers=['Answer', 'Count'], 
                  tablefmt="grid"))
    
    print("\nOriginal Cant_solve Distribution:")
    print(tabulate(orig_cant_solve.reset_index(), 
                  headers=['Cant_solve', 'Count'], 
                  tablefmt="grid"))
    
    print("\nProcessed Response Distribution:")
    print(tabulate(proc_response.reset_index(), 
                  headers=['Response', 'Count'], 
                  tablefmt="grid"))

def check_question_distribution(df_orig, df_proc):
    """Compare question distributions between datasets"""
    print_section_header("Question Distribution Analysis")
    
    # Original question distribution
    orig_questions = df_orig['question'].value_counts()
    
    # Get processed question columns
    question_cols = [col for col in df_proc.columns if col.startswith('q_')]
    proc_questions = df_proc[question_cols].sum()
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Original': orig_questions,
        'Processed': proc_questions
    }).fillna(0)
    comparison['Difference'] = comparison['Processed'] - comparison['Original']
    
    print(tabulate(comparison, headers='keys', tablefmt="grid", floatfmt=".0f"))

def check_continuous_activity(df_proc):
    """Check continuous activity variables"""
    print_section_header("Continuous Activity Analysis")
    
    seconds = df_proc['continuous_activity_seconds']
    hours = df_proc['continuous_activity_hours']
    
    conversion_diff = abs(seconds/3600 - hours)
    max_diff = conversion_diff.max()
    
    print("Hours/Seconds Conversion Check:")
    print(tabulate([["Maximum difference between seconds/3600 and hours", f"{max_diff:.6f}"]], 
                  headers=['Metric', 'Value'], 
                  tablefmt="grid"))
    
    print("\nValue Ranges:")
    stats = pd.DataFrame({
        'Seconds': seconds.describe(),
        'Hours': hours.describe()
    })
    print(tabulate(stats, headers='keys', tablefmt="grid", floatfmt=".3f"))

def main():
    # Load datasets
    df_orig, df_proc = load_datasets()
    
    print_section_header("Dataset Overview")
    overview = [
        ["Original Dataset", f"{df_orig.shape[0]:,}", f"{df_orig.shape[1]:,}"],
        ["Processed Dataset", f"{df_proc.shape[0]:,}", f"{df_proc.shape[1]:,}"]
    ]
    print(tabulate(overview, headers=["Dataset", "Rows", "Columns"], tablefmt="grid"))
    
    # Run checks
    missing_users = check_missing_ids(df_orig, df_proc, 'user_id', 'User')
    missing_crops = check_missing_ids(df_orig, df_proc, 'crop_id', 'Crop')
    
    check_na_values(df_proc)
    compare_column_stats(df_orig, df_proc)
    check_response_distribution(df_orig, df_proc)
    check_question_distribution(df_orig, df_proc)
    check_continuous_activity(df_proc)
    
    # Save detailed results
    Path('results/tests').mkdir(parents=True, exist_ok=True)
    
    if missing_users:
        pd.Series(list(missing_users)).to_csv('results/tests/missing_users.csv', index=False)
    if missing_crops:
        pd.Series(list(missing_crops)).to_csv('results/tests/missing_crops.csv', index=False)

if __name__ == "__main__":
    main() 