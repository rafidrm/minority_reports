from .base_model import LogisticModel
import numpy as np
import statsmodels.api as sm # type: ignore

def remove_reference_categories(df, feature_cols):
    """
    Remove reference categories to avoid dummy variable trap:
    - Remove 'q_reflection' from question dummies
    - Remove last date from date dummies
    """
    # Get question and date columns
    question_cols = [col for col in feature_cols if col.startswith('q_')]
    date_cols = [col for col in feature_cols if col.startswith('date_')]
    
    # Remove reference categories
    if question_cols:
        cols_to_keep = [col for col in question_cols if col != 'q_reflection']
    else:
        cols_to_keep = []
        
    if date_cols:
        # Sort date columns to ensure consistent reference category
        sorted_dates = sorted(date_cols)
        cols_to_keep.extend(sorted_dates[:-1])  # Remove last date
    
    # Get other columns that aren't questions or dates
    other_cols = [col for col in feature_cols if col not in question_cols + date_cols]
    cols_to_keep.extend(other_cols)
    
    return cols_to_keep

def fit_disagreement_model(df, dataset='ECPD', model_name='disagreement_base'):
    """Fit base disagreement model with questions and dates"""
    model = LogisticModel(model_name, dataset=dataset)
    # Get feature columns
    question_cols = [col for col in df.columns if col.startswith('q_')]
    date_cols = [col for col in df.columns if col.startswith('date_')]
    
    # Remove reference categories
    feature_cols = remove_reference_categories(df, question_cols + date_cols)
    print(f"Fitting model with {len(feature_cols)} features after removing reference categories")
    
    # Prepare features and target
    X = df[feature_cols]
    y = df['disagree']
    
    # Fit null model first (intercept only)
    X_null = sm.add_constant(np.ones((len(df), 0)))
    null_model = sm.Logit(y, X_null)
    null_results = null_model.fit(disp=0)
    print(f"\nNull model log-likelihood: {null_results.llf:.4f}")
    
    # Fit full model
    results = model.fit(X, y, plot_roc=True)
    print(f"Full model log-likelihood: {results.llf:.4f}")
    print(f"McFadden's pseudo R-squared: {results.prsquared:.4f}")
    
    return results

def fit_disagreement_model_with_activity(df, dataset='ECPD', model_name='disagreement_activity'):
    """Fit disagreement model with activity hours"""
    model = LogisticModel(model_name, dataset=dataset)
    # Get feature columns
    question_cols = [col for col in df.columns if col.startswith('q_')]
    date_cols = [col for col in df.columns if col.startswith('date_')]
    
    # Create quadratic term
    df['activity_hours_sq'] = df['continuous_activity_hours'] ** 2
    
    # Remove reference categories
    all_features = ['continuous_activity_hours', 'activity_hours_sq'] + question_cols + date_cols
    feature_cols = remove_reference_categories(df, all_features)
    print(f"Fitting model with {len(feature_cols)} features after removing reference categories")
    
    # Prepare features and target
    X = df[feature_cols]
    y = df['disagree']
    
    # Fit model
    results = model.fit(X, y, plot_roc=True)
    
    return results 