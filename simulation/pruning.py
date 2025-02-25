import pandas as pd

def prune_data(test_data, threshold, strategy='default'):
    """
    Prune test_data based on the specified pruning strategy.
    
    Strategies:
      - default: First instance of worker_id, task_id, crop_id, or question is never pruned
      - allow_workers: First instance of task_id, crop_id, or question is never pruned
                      (but new workers can be pruned)
    
    For qualified instances (based on strategy):
      - Prune if predicted_proba >= threshold
    
    Returns:
      pruned_data: DataFrame of rows to be pruned
      cycle_stats: dict with summary statistics
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
    
    # Add question and worker distributions
    if 'question_type' in test_data.columns:
        stats['pruned_by_question'] = pruned['question_type'].value_counts().to_dict()
    else:
        stats['pruned_by_question'] = {}
    stats['pruned_by_worker'] = pruned['user_id'].value_counts().to_dict()
    
    return pruned, stats 