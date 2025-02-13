import numpy as np
import pandas as pd
from pathlib import Path
from utils.plot_style import setup_plot_style
import subprocess
import os
from utils.model_utils import (
    save_model_files, plot_random_effects_dist, 
    plot_extreme_effects, plot_roc_curve
)

def fit_disagreement_meworkerscrops_model(df, model_name='disagreement_meworkerscrops_r', dataset='ECPD'):
    """
    Fit mixed-effects logistic regression using R's lme4 with both worker and crop random effects
    """
    # Get feature columns
    question_cols = [col for col in df.columns if col.startswith('q_')]
    date_cols = [col for col in df.columns if col.startswith('date_')]
    
    # Remove reference categories
    if question_cols:  # Handle case where there are no question columns (ZOD)
        question_cols = [col for col in question_cols if col != 'q_reflection']
    
    sorted_dates = sorted(date_cols)
    date_cols = sorted_dates[:-1]  # Remove last date
    
    # Prepare features
    feature_cols = question_cols + date_cols
    if 'continuous_activity_hours' in df.columns:
        df = df.copy()
        df['activity_hours_sq'] = df['continuous_activity_hours'] ** 2
        feature_cols.extend(['continuous_activity_hours', 'activity_hours_sq'])
    
    print(f"\nFitting mixed-effects model with random worker and crop effects...")
    print(f"Number of workers: {df['user_id'].nunique()}")
    print(f"Number of crops: {df['crop_id'].nunique()}")
    print(f"Number of observations: {len(df)}")
    print("\nObservations per worker:")
    print(df.groupby('user_id').size().describe())
    print("\nObservations per crop:")
    print(df.groupby('crop_id').size().describe())
    
    # Create tmp directory if it doesn't exist
    Path('tmp').mkdir(exist_ok=True)
    
    # Clean column names for R compatibility and ensure we include both random effect columns
    required_cols = ['disagree', 'user_id', 'crop_id'] + feature_cols
    print("\nRequired columns:", required_cols)
    print("Available columns:", df.columns.tolist())
    
    # Assert all required columns are available
    assert all(col in df.columns for col in required_cols), "Missing required columns"
    
    model_df = df[required_cols].copy()
    # Replace hyphens with underscores in all column names
    model_df.columns = [col.replace('-', '_') for col in model_df.columns]
    # Also clean the feature_cols for R formula
    feature_cols = [col.replace('-', '_') for col in feature_cols]
    
    # Save data for R with model-specific prefix
    model_df.to_csv('tmp/meworkerscrops_model_data.csv', index=False)
    
    # Run R script with arguments
    try:
        print("\nFitting models in R...")
        r_script_path = Path(__file__).parent / 'fit_mixed_workerscrops_model.R'
        subprocess.run([
            'Rscript',
            str(r_script_path),
            'tmp/meworkerscrops_model_data.csv',
            ','.join(feature_cols)
        ], check=True)
        
        # Read results with updated paths
        fit_stats = pd.read_csv('tmp/meworkerscrops_fit_stats.csv')
        worker_effects = pd.read_csv('tmp/meworkerscrops_worker_effects.csv')
        crop_effects = pd.read_csv('tmp/meworkerscrops_crop_effects.csv')
        predictions = pd.read_csv('tmp/meworkerscrops_predictions.csv')  # Read predictions early
        
        # Create results object with necessary attributes for plotting
        class Results:
            def __init__(self):
                self.llf = fit_stats['loglik'].iloc[0]
                self.aic = fit_stats['aic'].iloc[0]
                self.model = type('Model', (), {'endog': df['disagree'].values})
                self.worker_effects = worker_effects
                self.crop_effects = crop_effects
                self.mcfadden_r2 = fit_stats['mcfadden_r2'].iloc[0]
                self.mcfadden_r2_simple = fit_stats['mcfadden_r2_simple'].iloc[0]
                self.predictions = predictions['pred'].values  # Store predictions
        
        results = Results()
        
        print("\nModel Comparison:")
        print(f"Full model log-likelihood: {results.llf:.4f}")
        print(f"Null model (with RE) log-likelihood: {fit_stats['null_loglik'].iloc[0]:.4f}")
        print(f"Null model (simple) log-likelihood: {fit_stats['simple_null_loglik'].iloc[0]:.4f}")
        print(f"McFadden's pseudo R² (vs RE null): {results.mcfadden_r2:.4f}")
        print(f"McFadden's pseudo R² (vs simple null): {results.mcfadden_r2_simple:.4f}")
        
        # Save results and generate plots
        save_model_results(results, model_name=model_name, dataset=dataset)
        
        return results
        
    except subprocess.CalledProcessError as e:
        print(f"R model fitting failed: {str(e)}")
        raise
    finally:
        # Clean up tmp files with updated prefixes
        tmp_files = [
            'meworkerscrops_model_data.csv',
            'meworkerscrops_fit_stats.csv',
            'meworkerscrops_fixed_effects.csv',
            'meworkerscrops_worker_effects.csv',
            'meworkerscrops_crop_effects.csv',
            'meworkerscrops_predictions.csv',
            'meworkerscrops_model_summary.txt'
        ]

def save_model_results(results, model_name, dataset='ECPD'):
    """Save mixed model results and generate plots"""
    setup_plot_style()
    
    # Create dataset-specific directories
    results_dir = Path(f'results/{dataset}')
    param_dir = results_dir / 'parameters'
    plot_dir = results_dir / 'plots'
    for d in [param_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save all model files with correct prefixed filenames
    save_model_files(model_name, {
        'meworkerscrops_model_summary.txt': 'summary.txt',
        'meworkerscrops_predictions.csv': 'predictions.csv',
        'meworkerscrops_fixed_effects.csv': 'fixed_effects.csv',
        'meworkerscrops_worker_effects.csv': 'worker_effects.csv',
        'meworkerscrops_crop_effects.csv': 'crop_effects.csv'
    }, dataset=dataset)
    
    # Plot worker random effects
    plot_random_effects_dist(
        results.worker_effects, 
        'Worker',
        str(plot_dir / f'{model_name}_worker_effects_hist.png')
    )
    
    plot_extreme_effects(
        results.worker_effects,
        'worker_id',
        'Worker',
        20,
        str(plot_dir / f'{model_name}_extreme_workers.png')
    )
    
    # Plot crop random effects
    plot_random_effects_dist(
        results.crop_effects, 
        'Crop',
        str(plot_dir / f'{model_name}_crop_effects_hist.png')
    )
    
    plot_extreme_effects(
        results.crop_effects,
        'crop_id',
        'Crop',
        20,
        str(plot_dir / f'{model_name}_extreme_crops.png')
    )
    
    # Save model fit statistics
    with open(param_dir / f'{model_name}_fit_stats.txt', 'w') as f:
        f.write(f"Log-likelihood: {results.llf:.4f}\n")
        f.write(f"McFadden's pseudo R² (vs RE null): {results.mcfadden_r2:.4f}\n")
        f.write(f"McFadden's pseudo R² (vs simple null): {results.mcfadden_r2_simple:.4f}\n")
    
    # Plot ROC curve
    try:
        print("\nComputing ROC curve...")
        y_true = results.model.endog
        y_pred = results.predictions
        plot_roc_curve(
            y_true, y_pred, model_name,
            str(plot_dir / f'{model_name}_roc.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate ROC curve: {str(e)}") 