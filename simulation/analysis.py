import matplotlib.pyplot as plt
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