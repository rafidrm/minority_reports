import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import os
from scipy.special import expit  # Import logistic function

class SimulationModel:
    def __init__(self, fixed_effects, worker_effects, crop_effects):
        self.fixed_effects = fixed_effects  # dict: parameter -> value (including 'Intercept')
        self.worker_effects = worker_effects  # dict: worker_id -> effect
        self.crop_effects = crop_effects      # dict: crop_id -> effect

def undersample_with_correction(df, minority_col='disagree', random_state=42):
    """
    Undersample the majority class while ensuring all workers, crops, and active questions
    are represented in the final dataset.
    
    Steps:
    1. Separate majority and minority classes
    2. Undersample majority class to match minority class size
    3. Check for missing workers, crops, and inactive questions
    4. Add back examples to maintain representation of all entities
    """
    # Separate majority and minority classes
    minority_class = df[df[minority_col] == 1]
    majority_class = df[df[minority_col] == 0]
    
    # Initial undersampling of majority class
    undersampled_majority = majority_class.sample(
        n=len(minority_class), 
        random_state=random_state
    )
    
    # Combine minority class with undersampled majority
    sampled_df = pd.concat([minority_class, undersampled_majority])
    
    # Find missing entities
    missing_workers = set(df['user_id'].unique().tolist()) - \
        set(sampled_df['user_id'].unique().tolist())
    missing_crops = set(df['crop_id'].unique().tolist()) - \
        set(sampled_df['crop_id'].unique().tolist())
    
    # Find questions that were active in original but inactive in sampled
    q_cols = [col for col in df.columns if col.startswith('q_')]
    orig_active_q = [col for col in q_cols if df[col].sum() > 0]
    sampled_active_q = [col for col in q_cols if sampled_df[col].sum() > 0]
    missing_active_q = set(orig_active_q) - set(sampled_active_q)
    
    # Print diagnostic information
    # print("\nCorrecting undersampled dataset:")
    # print(f"Missing workers: {len(missing_workers)}")
    # print(f"Missing crops: {len(missing_crops)}")
    # print(f"Questions that lost activation: {len(missing_active_q)}")
    
    rows_to_add = []
    
    # Add examples for missing workers
    for worker in missing_workers:
        worker_rows = majority_class[majority_class['user_id'] == worker]
        if len(worker_rows) > 0:
            rows_to_add.append(worker_rows.iloc[0])
    
    # Add examples for missing crops
    for crop in missing_crops:
        crop_rows = majority_class[majority_class['crop_id'] == crop]
        if len(crop_rows) > 0:
            rows_to_add.append(crop_rows.iloc[0])
    
    # Add examples for questions that lost activation
    for q_col in missing_active_q:
        q_rows = majority_class[majority_class[q_col] == 1]
        if len(q_rows) > 0:
            rows_to_add.append(q_rows.iloc[0])
    
    # Add correction rows if any were found
    if rows_to_add:
        correction_df = pd.DataFrame(rows_to_add)
        final_df = pd.concat([sampled_df, correction_df])
        print(f"Added {len(rows_to_add)} rows to maintain representation")
    else:
        final_df = sampled_df
        print("No correction rows needed to be added")
    
    return final_df.sample(frac=1, random_state=random_state)  # Shuffle the final dataset

def fit_simulation_model(training_df, dataset='ECPD', output_dir='results/ECPD/simulation/parameters'):
    """
    Fit the simulation model by calling an R script.
    Uses only the one-hot encoded question columns (q_*) along with user_id and crop_id.
    Saves the model outputs in output_dir and returns a SimulationModel instance.
    """
    # Create experiment-specific temp directory
    temp_dir = Path(output_dir).parent / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    # Identify all q_ columns and which ones are active/inactive
    q_cols = [col for col in training_df.columns if col.startswith('q_')]
    active_q_cols = [col for col in q_cols if training_df[col].sum() > 0]
    inactive_q_cols = list(set(q_cols) - set(active_q_cols))
    
    # Keep all columns for consistency in predictions, but only use active ones for fitting
    cols_to_keep = ['disagree', 'user_id', 'crop_id'] + q_cols
    
    # Create a copy for model fitting that will be undersampled with correction
    model_training_data = training_df[cols_to_keep].copy()
    model_training_data = undersample_with_correction(model_training_data)
    
    # Print comparative statistics
    print("\nTraining Data Summary (Original → Corrected):")
    print(f"Total rows: {len(training_df)} → {len(model_training_data)}")
    
    orig_disagree = training_df['disagree'].value_counts()
    corr_disagree = model_training_data['disagree'].value_counts()
    orig_active = len([col for col in q_cols if training_df[col].sum() > 0])
    corr_active = len([col for col in q_cols if model_training_data[col].sum() > 0])
    
    print(f"Class balance (0:1): {orig_disagree[0]}:{orig_disagree[1]} → {corr_disagree[0]}:{corr_disagree[1]}")
    print(f"Unique workers: {training_df['user_id'].nunique()} → {model_training_data['user_id'].nunique()}")
    print(f"Unique crops: {training_df['crop_id'].nunique()} → {model_training_data['crop_id'].nunique()}")
    print(f"Active questions: {orig_active} → {corr_active}")
    
    # Verify we have variation in the response
    if model_training_data['disagree'].nunique() <= 1:
        raise ValueError("Error: 'disagree' column has no variation (all values are the same)")
    
    # Use experiment-specific temp file
    temp_csv = temp_dir / 'simulation_training.csv'
    model_training_data.to_csv(temp_csv, index=False)
    
    # Build predictor string as comma-separated list of active q_columns only
    predictors = ",".join(active_q_cols)
    
    # Call the R script (located in models/)
    r_script = os.path.join(os.path.dirname(__file__), 'fit_simulation_model.R')
    cmd = ['Rscript', r_script, str(temp_csv), predictors, output_dir, str(temp_dir)]
    subprocess.run(cmd, check=True)
    
    # Read outputs produced by the R script
    out_dir = Path(output_dir)
    fixed_effects_file = out_dir / 'fixed_effects_simulation.csv'
    worker_effects_file = out_dir / 'worker_effects_simulation.csv'
    crop_effects_file = out_dir / 'crop_effects_simulation.csv'
    
    # Read the fixed effects and add zero coefficients for inactive columns
    fixed_effects_df = pd.read_csv(fixed_effects_file, index_col=0)
    fixed_effects = fixed_effects_df['coef'].to_dict()
    
    # Add zero coefficients for inactive question columns
    for col in inactive_q_cols:
        fixed_effects[col] = 0.0
        print(f"Added zero coefficient for inactive column: {col}")
    
    worker_effects_df = pd.read_csv(worker_effects_file)
    worker_effects = dict(zip(worker_effects_df['worker_id'], worker_effects_df['effect']))
    
    crop_effects_df = pd.read_csv(crop_effects_file)
    crop_effects = dict(zip(crop_effects_df['crop_id'], crop_effects_df['effect']))
    
    # Clean up temp file
    os.remove(temp_csv)
    
    return SimulationModel(fixed_effects, worker_effects, crop_effects)

def predict_simulation_proba(test_df, sim_model):
    """
    Predict probabilities for test_df using the fitted simulation model.
    Computes the linear predictor:
       lp = Intercept + sum_{q in q_cols} (beta_q * q_value) + (worker_RE) + (crop_RE)
    Then applies the logistic transformation.
    
    For any new worker or crop (not found in the model), the corresponding RE is assumed 0.
    Returns a pandas Series of predicted probabilities.
    """
    q_cols = [col for col in test_df.columns if col.startswith('q_')]
    intercept = sim_model.fixed_effects.get('Intercept', 0)
    
    lin_preds = []
    for idx, row in test_df.iterrows():
        fixed_sum = 0
        for col in q_cols:
            beta = sim_model.fixed_effects.get(col, 0)
            fixed_sum += beta * row.get(col, 0)
        lp = intercept + fixed_sum
        worker = row['user_id']
        crop = row['crop_id']
        lp += sim_model.worker_effects.get(worker, 0)
        lp += sim_model.crop_effects.get(crop, 0)
        lin_preds.append(lp)
    
    lin_preds = np.array(lin_preds)
    proba = expit(lin_preds)  # logistic function 1 / (1 + exp(-x))
    return pd.Series(proba, index=test_df.index) 