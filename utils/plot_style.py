import matplotlib.pyplot as plt

def setup_plot_style():
    """Set up the common plotting style for the project"""
    plt.rcParams.update({
        # Font sizes
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        # 'font.size': 11,
        # 'axes.labelsize': 12,
        # 'axes.titlesize': 12,
        # 'xtick.labelsize': 10,
        # 'ytick.labelsize': 10,
        # 'legend.fontsize': 10,
        
        # Figure size
        'figure.figsize': (8, 6),
        
        # Axis styling
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # Grid styling
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'grid.color': '#CCCCCC',
        
        # Font families
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        
        # Border width
        'axes.linewidth': 2.0,
    })

def style_axis(ax):
    """Apply consistent styling to a single axis"""
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC')
    
    # Move ticks inside
    ax.tick_params(direction='in')
    
    return ax 