import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils.plot_style import setup_plot_style, style_axis
import os
import logging

logger = logging.getLogger(__name__)

def save_model_files(model_name, file_mapping, dataset='ECPD', subdir='parameters'):
    """
    Move files from tmp directory to final location with standardized naming
    
    Args:
        model_name: Name of model for file prefixes
        file_mapping: Dict mapping source filenames to destination filenames
        dataset: Dataset being processed (ECPD or ZOD)
        subdir: Subdirectory within results/dataset/ to save files (e.g. 'parameters')
    """
    # Construct destination directory
    dst_dir = Path(f'results/{dataset}/{subdir}')
    
    # Check that destination directory exists
    assert dst_dir.exists(), f"Destination directory {dst_dir} does not exist. It should be created by LogisticModel."
    
    # Move each file
    for src_file, dst_file in file_mapping.items():
        src_path = Path('tmp') / src_file
        if not src_path.exists():
            logger.warning(f"Source file not found: {src_path}")
            continue
            
        dst_path = dst_dir / f'{model_name}_{dst_file}'
        src_path.rename(dst_path)
        logger.info(f"Moved {src_path} to {dst_path}")

def plot_random_effects_dist(effects_df, effect_name, output_path):
    """Plot distribution of random effects"""
    setup_plot_style()

    fig, ax = plt.subplots()
    ax.hist(effects_df['effect'], 
            bins=30, 
            edgecolor='black', 
            color='#2E86C1', 
            alpha=0.7,
            linewidth=1)
    # ax.axvline(x=0, color='red', linestyle='--', label='Population mean')
    ax.set_xlabel(f'{effect_name} Random Effect')
    ax.set_ylabel('Count')
    # ax.set_title(f'Distribution of {effect_name} Random Effects')
    # ax.legend()
    style_axis(ax)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_extreme_effects(effects_df, id_col, effect_name, n_extreme, output_path):
    """Plot extreme random effects with error bars"""
    extremes = pd.concat([
        effects_df.nlargest(n_extreme, 'effect'),
        effects_df.nsmallest(n_extreme, 'effect')
    ])
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.errorbar(
        x=extremes['effect'],
        y=range(len(extremes)),
        xerr=1.96 * extremes['std_err'],
        fmt='o',
        capsize=5
    )
    # ax.axvline(x=0, color='red', linestyle='--', label='Population mean')
    ax.set_yticks(range(len(extremes)))
    ax.set_yticklabels(extremes[id_col])
    ax.set_xlabel(f'{effect_name} Random Effect')
    # ax.set_title(f'Top and Bottom {n_extreme} {effect_name} Effects')
    # ax.legend()
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred, model_name, output_path):
    """Plot ROC curve for a model"""
    setup_plot_style()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    style_axis(ax)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() 