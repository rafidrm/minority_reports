import pandas as pd

def prune_data(test_data, threshold, strategy='default'):
    """
    Prune test_data based on the specified pruning strategy.
    Default strategy:
      - Consider only rows with predicted_proba >= 0 (i.e. those with seen worker, crop, and question).
      - Prune the row if predicted_proba >= threshold.
    
    Returns:
      pruned_data: DataFrame of rows to be pruned
      cycle_stats: dict with summary statistics such as total test rows, candidate rows, and counts pruned.
    """
    # Total rows in test data
    total_rows = len(test_data)
    
    # Candidates are rows with predicted_proba >= 0 (new ones have -1)
    candidate_mask = test_data['predicted_proba'] >= 0
    candidates = test_data[candidate_mask]
    prune_mask = candidates['predicted_proba'] >= threshold
    pruned = candidates[prune_mask]
    
    # Calculate statistics
    stats = {
        'total_test': total_rows,
        'num_candidates': len(candidates),
        'num_pruned': len(pruned),
        'prune_rate': len(pruned) / len(candidates) if len(candidates) > 0 else 0,
        'prune_rate_total': len(pruned) / total_rows if total_rows > 0 else 0
    }
    
    # Add question and worker distributions if available
    if 'question_type' in test_data.columns:
        stats['pruned_by_question'] = pruned['question_type'].value_counts().to_dict()
    else:
        stats['pruned_by_question'] = {}
    stats['pruned_by_worker'] = pruned['user_id'].value_counts().to_dict()
    
    return pruned, stats 