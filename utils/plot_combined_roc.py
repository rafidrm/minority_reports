import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
from utils.plot_style import setup_plot_style, style_axis

def parse_args():
    parser = argparse.ArgumentParser(description='Plot combined ROC curves')
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Choose the dataset to process (ECPD or ZOD)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up paths based on dataset
    results_dir = Path(f'results/{args.dataset}')
    param_dir = results_dir / 'parameters'
    plot_dir = results_dir / 'plots'
    
    # Ensure plot directory exists
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load true values from dataset
    data_path = Path(f'data/{args.dataset}/answers_logreg_all.csv')
    y_true = pd.read_csv(data_path)['disagree'].values
    
    models = [
        ('disagreement_base', 'Base'),
        ('disagreement_activity', 'Model A'),
        ('disagreement_meworkers_r', 'Model A + W'),
        ('disagreement_mecrops_r', 'Model A + C'),
        ('disagreement_meworkerscrops_r', 'Model A + W + C')
    ]
    
    plt.figure(figsize=(10, 8))
    
    for model_name, label in models:
        try:
            # Update path to use dataset-specific directory
            pred_path = param_dir / f'{model_name}_predictions.csv'
            if not pred_path.exists():
                print(f"Skipping {model_name} - predictions not found")
                continue
                
            preds = pd.read_csv(pred_path)
            fpr, tpr, _ = roc_curve(y_true, preds['pred'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curves Comparison - {args.dataset}')
    plt.legend(loc="lower right")
    style_axis(plt.gca())
    
    # Save with dataset-specific filename
    output_path = plot_dir / f'{args.dataset}_combined_roc.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined ROC plot to {output_path}")

if __name__ == '__main__':
    setup_plot_style()
    main() 