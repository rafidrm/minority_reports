import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from utils.plot_style import setup_plot_style, style_axis

def generate_simulation_plots(metrics_df, output_dir, experiment_name):
    """
    Generate combined plots summarizing simulation metrics.
    Expects metrics_df to contain columns from cycle_summaries.csv such as:
    - cycle_num: The cycle number
    - num_pruned: Number of rows pruned in this cycle
    - num_candidates: Number of candidate rows in this cycle
    - prune_rate: Pruning rate for this cycle
    
    Saves the plot as '{experiment_name}_summary.png' in the specified output directory.
    """
    setup_plot_style()  # use common plot styling from the project
    
    plt.figure(figsize=(12, 8))
    
    # First subplot: Counts per cycle
    ax1 = plt.subplot(2, 1, 1)
    # Use colors consistent with other plots (darkorange and navy)
    ax1.plot(metrics_df['cycle_num'], metrics_df['num_pruned'], marker='o', color='darkorange', label='Num Pruned')
    ax1.plot(metrics_df['cycle_num'], metrics_df['num_candidates'], marker='s', color='navy', label='Num Candidates')
    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Count")
    ax1.set_title("Pruning Counts per Cycle")
    ax1.legend()
    style_axis(ax1)
    
    # Second subplot: Prune rate per cycle
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(metrics_df['cycle_num'], metrics_df['prune_rate'], marker='o', color='purple', label='Prune Rate')
    ax2.set_xlabel("Cycle Number")
    ax2.set_ylabel("Prune Rate")
    ax2.legend()
    style_axis(ax2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment_name}_summary.png", dpi=300)
    plt.close()

def plot_cycle_roc_curves(cycle_files, output_dir, experiment_name):
    """
    Generate ROC curves for each cycle's predictions on qualified instances.
    
    Args:
        cycle_files: List of DataFrames containing cycle data
        output_dir: Directory to save the plot
        experiment_name: Name of the experiment for the filename
    """
    setup_plot_style()
    plt.figure(figsize=(10, 8))
    
    for i, cycle_df in enumerate(cycle_files, 1):
        # Only consider qualified instances (predicted_proba >= 0)
        qualified = cycle_df[cycle_df['predicted_proba'] >= 0]
        
        if len(qualified) == 0:
            print(f"Warning: Cycle {i} has no qualified instances")
            continue
            
        # Check if we have any variation in the true labels
        if qualified['disagree'].nunique() < 2:
            print(f"Warning: Cycle {i} has no variation in true labels")
            continue
            
        try:
            fpr, tpr, _ = roc_curve(qualified['disagree'], qualified['predicted_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, alpha=0.7,
                    label=f'Cycle {i} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Warning: Could not compute ROC curve for cycle {i}: {str(e)}")
            continue
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0., 1.0])
    plt.ylim([0., 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves by Cycle')
    plt.legend(loc="lower right", fontsize=8)
    style_axis(plt.gca())
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment_name}_cycle_roc.png", dpi=300)
    plt.close()

def plot_before_after_histograms(before_counts, after_counts, entity_type, output_dir, experiment_name):
    """
    Generate before/after histograms comparing repeat counts on the same plot.
    
    Args:
        before_counts: Series of counts from original dataset
        after_counts: Series of counts from final dataset
        entity_type: String indicating what's being counted ('Worker' or 'Task')
        output_dir: Directory to save the plot
        experiment_name: Name of the experiment for the filename
    """
    setup_plot_style()
    plt.figure(figsize=(10, 6))
    
    # Calculate bin edges that work for both distributions
    all_counts = np.concatenate([before_counts.values, after_counts.values])
    max_count = np.percentile(all_counts, 99)  # Use 99th percentile to avoid extreme outliers
    bins = np.linspace(0, max_count, 50)
    
    # Plot both histograms on the same axes
    plt.hist(before_counts.values, bins=bins, alpha=0.6, color='#2E86C1',
            edgecolor='black', linewidth=1, label='Original')
    plt.hist(after_counts.values, bins=bins, alpha=0.6, color='#E67E22',
            edgecolor='black', linewidth=1, label='After Pruning')
    
    plt.xlabel(f'Number of {entity_type} Repeats')
    plt.ylabel('Count')
    # plt.title(f'{entity_type} Repeat Distribution')
    plt.legend()
    style_axis(plt.gca())
    
    # Add summary statistics as text
    # stats_text = (
    #     f"Original: mean={before_counts.mean():.1f}, "
    #     f"median={before_counts.median():.1f}, "
    #     f"std={before_counts.std():.1f}\n"
    #     f"After: mean={after_counts.mean():.1f}, "
    #     f"median={after_counts.median():.1f}, "
    #     f"std={after_counts.std():.1f}"
    # )
    # plt.figtext(0.1, 0.02, stats_text, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment_name}_{entity_type.lower()}_repeats.png",
                dpi=300, bbox_inches='tight')
    plt.close() 