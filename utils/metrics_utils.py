import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_confusion_metrics(y_true, y_pred):
    """Calculate TPR and FPR from binary predictions
    
    Args:
        y_true: Array-like of true binary labels (0 or 1)
        y_pred: Array-like of predicted binary labels (0 or 1)
        
    Returns:
        tuple: (tpr, fpr)
            - tpr: True Positive Rate (sensitivity/recall)
            - fpr: False Positive Rate (1 - specificity)
    """
    true_pos = np.sum((y_true == 1) & (y_pred == 1))
    false_pos = np.sum((y_true == 0) & (y_pred == 1))
    true_neg = np.sum((y_true == 0) & (y_pred == 0))
    false_neg = np.sum((y_true == 1) & (y_pred == 0))
    
    tpr = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
    return tpr, fpr

def calculate_binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics treating 'yes' as positive class
    
    Args:
        y_true: Array-like of true labels ('yes' or 'no')
        y_pred: Array-like of predicted labels ('yes' or 'no')
        
    Returns:
        dict: Dictionary containing:
            - true_positive_rate: TPR (sensitivity/recall)
            - false_positive_rate: FPR (1 - specificity)
            - precision: Precision (positive predictive value)
            - recall: Same as TPR
            - f1: F1 score
            - confusion_matrix: Dict with tn, fp, fn, tp counts
    """
    # Convert to binary (yes=1, no=0)
    y_true_binary = np.array(y_true == 'yes').astype(int)
    y_pred_binary = np.array(y_pred == 'yes').astype(int)
    
    # Get TPR/FPR from base metrics function
    tpr, fpr = calculate_confusion_metrics(y_true_binary, y_pred_binary)
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positive_rate': tpr,
        'false_positive_rate': fpr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'tn': tn, 'fp': fp,
            'fn': fn, 'tp': tp
        }
    } 