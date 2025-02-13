import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.plot_style import setup_plot_style
from pathlib import Path
import argparse

def plot_worker_task_distribution(df, output_path='results/exploratory_plots/worker_task_distribution.png', show_stats=True, dataset='ECPD'):
    """Plot distribution of tasks per worker"""
    setup_plot_style()
    tasks_per_worker = df.groupby('user_id').size()
    
    plt.figure(figsize=(10, 6))
    plt.hist(tasks_per_worker, bins=30)
    plt.xlabel('Number of Tasks')
    plt.ylabel('Number of Workers')
    # plt.title('Distribution of Tasks per Worker')
    
    # Add summary statistics as text
    if show_stats:
        stats_text = f'Mean: {tasks_per_worker.mean():.1f}\n'
        stats_text += f'Median: {tasks_per_worker.median():.1f}\n'
        stats_text += f'Total Workers: {len(tasks_per_worker)}'
        plt.text(0.95, 0.95, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_worker_task_bars(df, output_path='results/exploratory_plots/worker_task_bars.png', show_stats=True, dataset='ECPD'):
    """
    Create a bar plot showing number of tasks completed by each worker,
    sorted from most to least tasks, with log scale y-axis
    
    Args:
        df: DataFrame containing the answers data
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots()
    
    # Count tasks per worker and sort
    worker_tasks = df['user_id'].value_counts().sort_values(ascending=False)
    
    # Create bar plot
    bars = ax.bar(range(len(worker_tasks)), 
                  worker_tasks.values,
                  color='#2E86C1',
                  alpha=0.7,
                  edgecolor='black',
                  linewidth=1)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Customize plot
    ax.set_xlabel('Worker Rank')
    ax.set_ylabel('Number of Tasks Completed')
    # ax.set_title('Tasks Completed by Each Worker')
    
    # Add summary statistics as text
    stats_text = f'Mean: {worker_tasks.mean():.1f}\n'
    stats_text += f'Median: {worker_tasks.median():.1f}\n'
    stats_text += f'Total Workers: {len(worker_tasks)}'
    
    if show_stats:
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_worker_disagreement_histogram(df, output_path='results/exploratory_plots/worker_disagreement_histogram.png', show_stats=True, dataset='ECPD'):
    """
    Create a histogram of disagreement rates per worker
    """
    fig, ax = plt.subplots()
    
    # Calculate disagreement rate for each worker
    worker_disagreement = df.groupby('user_id').agg({
        'disagree': 'mean'  # mean of 0s and 1s gives us the proportion
    }) * 100  # Convert to percentage
    
    # Create histogram
    n, bins, patches = ax.hist(worker_disagreement['disagree'], 
                              bins=30, 
                              edgecolor='black', 
                              color='#2E86C1', 
                              alpha=0.7,
                              linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Disagreement Rate (%)')
    ax.set_ylabel('Number of Workers')
    # ax.set_title('Distribution of Worker Disagreement Rates')
    
    # Add summary statistics as text
    stats_text = f'Mean: {worker_disagreement["disagree"].mean():.1f}%\n'
    stats_text += f'Median: {worker_disagreement["disagree"].median():.1f}%\n'
    stats_text += f'Total Workers: {len(worker_disagreement)}'
    
    if show_stats:
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_worker_disagreement_bars(df, output_path='results/exploratory_plots/worker_disagreement_bars.png', show_stats=True, dataset='ECPD'):
    """
    Create a bar plot of disagreement rates per worker, sorted from highest to lowest
    """
    fig, ax = plt.subplots()
    
    # Calculate and sort disagreement rates
    worker_disagreement = df.groupby('user_id').agg({
        'disagree': 'mean'
    }) * 100  # Convert to percentage
    worker_disagreement = worker_disagreement.sort_values('disagree', ascending=False)
    
    # Create bar plot
    bars = ax.bar(range(len(worker_disagreement)), 
                  worker_disagreement['disagree'],
                  color='#2E86C1',
                  alpha=0.7,
                  edgecolor='black',
                  linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Worker Rank')
    ax.set_ylabel('Disagreement Rate (%)')
    # ax.set_title('Worker Disagreement Rates')
    
    # Add summary statistics as text
    stats_text = f'Mean: {worker_disagreement["disagree"].mean():.1f}%\n'
    stats_text += f'Median: {worker_disagreement["disagree"].median():.1f}%\n'
    stats_text += f'Total Workers: {len(worker_disagreement)}'
    
    if show_stats:
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_crop_disagreement_histogram(df, output_path='results/exploratory_plots/crop_disagreement_histogram.png', show_stats=True, dataset='ECPD'):
    """
    Create a histogram of disagreement rates per crop_id
    """
    fig, ax = plt.subplots()
    
    # Calculate disagreement rate for each crop_id
    crop_disagreement = df.groupby('crop_id').agg({
        'disagree': 'mean'  # mean of 0s and 1s gives us the proportion
    }) * 100  # Convert to percentage
    
    # Create histogram
    n, bins, patches = ax.hist(crop_disagreement['disagree'], 
                              bins=30, 
                              edgecolor='black', 
                              color='#2E86C1', 
                              alpha=0.7,
                              linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Disagreement Rate (%)')
    ax.set_ylabel('Number of Crops')
    # ax.set_title('Distribution of Crop Disagreement Rates')
    
    # Add summary statistics as text
    stats_text = f'Mean: {crop_disagreement["disagree"].mean():.1f}%\n'
    stats_text += f'Median: {crop_disagreement["disagree"].median():.1f}%\n'
    stats_text += f'Total Images: {len(crop_disagreement)}'
    
    if show_stats:
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_image_disagreement_histogram(df, output_path='results/exploratory_plots/image_disagreement_histogram.png', show_stats=True, dataset='ECPD'):
    """
    Create a histogram of disagreement rates per unique image_path
    (averaging over all crops from each image)
    """
    fig, ax = plt.subplots()
    
    # First group by image_path and crop_id to get crop-level disagreement
    crop_disagreement = df.groupby(['image_path', 'crop_id'])['disagree'].mean()
    
    # Then average over all crops from the same image
    image_disagreement = crop_disagreement.groupby('image_path').mean() * 100  # Convert to percentage
    
    # Create histogram
    n, bins, patches = ax.hist(image_disagreement, 
                              bins=30, 
                              edgecolor='black', 
                              color='#2E86C1', 
                              alpha=0.7,
                              linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Disagreement Rate (%)')
    ax.set_ylabel('Number of Images')
    # ax.set_title('Distribution of Image-Level Disagreement Rates')
    
    # Add summary statistics as text
    stats_text = f'Mean: {image_disagreement.mean():.1f}%\n'
    stats_text += f'Median: {image_disagreement.median():.1f}%\n'
    stats_text += f'Total Images: {len(image_disagreement)}'
    
    if show_stats:
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_daily_hourly_disagreements(df, output_path='results/exploratory_plots/daily_hourly_disagreements.png', show_stats=True, dataset='ECPD'):
    """Plot disagreement rates by day and hour"""
    setup_plot_style()
    
    # Convert created_at to datetime and extract date and hour
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    
    # Create daily_patterns subdirectory
    daily_patterns_dir = Path(output_path).parent / 'daily_patterns'
    daily_patterns_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each day separately
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        
        # Calculate hourly disagreement rates for all workers
        hourly_stats = day_data.groupby(['hour', 'user_id']).agg({
            'disagree': 'mean',
            'created_at': 'count'  # Use created_at instead of user_id for count
        }).reset_index()
        hourly_stats = hourly_stats.rename(columns={'created_at': 'annotations'})
        hourly_stats['disagree'] *= 100  # Convert to percentage
        
        # Calculate average across workers
        avg_stats = day_data.groupby('hour').agg({
            'disagree': ['mean', 'size'],
            'user_id': 'nunique'
        }).reset_index()
        avg_stats.columns = ['hour', 'mean_disagree', 'total_annotations', 'active_workers']
        avg_stats['mean_disagree'] *= 100  # Convert to percentage
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot individual worker lines
        for worker in hourly_stats['user_id'].unique():
            worker_data = hourly_stats[hourly_stats['user_id'] == worker]
            plt.plot(worker_data['hour'], worker_data['disagree'], 
                    color='lightblue', alpha=0.5, linewidth=1)
        
        # Plot average line
        plt.plot(avg_stats['hour'], avg_stats['mean_disagree'], 
                color='black', linewidth=2, marker='o')
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Disagreement Rate (%)')
        # plt.title(f'Worker Disagreement Rates on {date}')
        
        # Set x-axis ticks for each hour
        plt.xticks(range(24))
        plt.ylim(0, 60)  # Set y-axis limit to match example
        plt.grid(True, alpha=0.3)
        
        # Add summary statistics
        plt.text(0.98, 0.98, 
                f'Active Workers: {avg_stats["active_workers"].max()}\n'
                f'Total Annotations: {avg_stats["total_annotations"].sum():,}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right')
        
        # Save plot
        plt.savefig(daily_patterns_dir / f'disagreement_rate_{date}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_hourly_disagreement_all_days(df, output_path='results/exploratory_plots/hourly_disagreement_all_days.png', show_stats=True, dataset='ECPD'):
    """Plot hourly disagreement rates for each day in subplots"""
    setup_plot_style()
    
    # Convert created_at to datetime and extract date and hour
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    
    # Get unique dates and sort them
    unique_dates = sorted(df['date'].unique())
    
    # Set up the figure and axes
    if dataset == 'ECPD':
        n_rows, n_cols = 2, 3
        figsize = (18, 12)  # Adjusted for ECPD
    else:  # For ZOD
        n_rows, n_cols = 2, 1  # 2 rows, 1 column
        figsize = (6, 12)  # Adjusted for ZOD
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Set a consistent y-axis limit
    y_limit = 60  # Adjust as needed
    
    # Set the super title
    # fig.suptitle('Hourly Disagreement Rate per Day-of-Study', fontsize=16)
    
    for i, date in enumerate(unique_dates):
        if i >= n_rows * n_cols:
            break  # Limit to available plots
        
        day_data = df[df['date'] == date]
        
        # Calculate hourly disagreement rates for all workers
        hourly_disagreement = day_data.groupby('hour')['disagree'].mean() * 100
        
        # Plot individual worker lines
        for worker in day_data['user_id'].unique():
            worker_data = day_data[day_data['user_id'] == worker]
            worker_hourly_disagreement = worker_data.groupby('hour')['disagree'].mean() * 100
            axes[i].plot(worker_hourly_disagreement.index, worker_hourly_disagreement.values, 
                        color='lightblue', alpha=0.5, linewidth=1)
        
        # Plot average line
        axes[i].plot(hourly_disagreement.index, hourly_disagreement.values, marker='o', color='black')
        
        # Format the date for the title
        formatted_date = date.strftime("%B %d, %Y")
        axes[i].set_title(f'{formatted_date}')
        axes[i].set_xlabel('Hour of Day')
        # axes[i].set_xlabel(f'{formatted_date} (Hour of Day)')
        axes[i].set_ylabel('Disagreement Rate (%)')
        axes[i].set_ylim(0, y_limit)
        axes[i].set_xticks(range(0, 24, 2))  # Set x-axis ticks every 2 hours
        axes[i].grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis('off')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_worker_task_thresholds(df, output_path='results/exploratory_plots/worker_task_thresholds.png', show_stats=True, dataset='ECPD'):
    """
    Create bar plot showing number of workers above different task count thresholds
    
    Args:
        df: DataFrame with user_id and task counts
        output_dir: Directory to save plot
    """
    setup_plot_style()
    
    # Calculate tasks per worker
    tasks_per_worker = df.groupby('user_id').size()
    
    # Define thresholds and calculate counts
    thresholds = [10, 100, 1000, 10000]
    worker_counts = [sum(tasks_per_worker > t) for t in thresholds]
    
    # Create labels
    labels = [f'> {t:,}' for t in thresholds]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bars
    bars = ax.bar(range(len(thresholds)), worker_counts, 
                  color='#2E86C1', alpha=0.7,
                  edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of Tasks Completed')
    ax.set_ylabel('Number of Workers')
    # ax.set_title('Worker Task Completion Thresholds')
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate preprocessing analysis plots')
    parser.add_argument('plot', choices=['worker_dist', 'worker_bars', 'worker_disagree_hist', 
                                       'worker_disagree_bars', 'crop_disagree', 'image_disagree', 
                                       'daily_hourly', 'all', 'hourly_disagreement_all',
                                       'worker_thresholds'],
                       help='Which plot to generate')
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Choose the dataset to process (ECPD or ZOD)')
    parser.add_argument('--show-stats', action='store_true', 
                        help='Show statistics text on plots')
    args = parser.parse_args()
    
    # Determine output directory based on dataset
    output_dir = Path(f'results/{args.dataset}/exploratory_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = f'data/{args.dataset}/answers_unfiltered.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Map plot choices to functions
    plot_funcs = {
        'worker_dist': (plot_worker_task_distribution, 'worker_task_distribution.png'),
        'worker_bars': (plot_worker_task_bars, 'worker_task_bars.png'),
        'worker_disagree_hist': (plot_worker_disagreement_histogram, 'worker_disagreement_histogram.png'),
        'worker_disagree_bars': (plot_worker_disagreement_bars, 'worker_disagreement_bars.png'),
        'crop_disagree': (plot_crop_disagreement_histogram, 'crop_disagreement_histogram.png'),
        'daily_hourly': (plot_daily_hourly_disagreements, 'daily_hourly_disagreements.png'),
        'hourly_disagreement_all': (plot_hourly_disagreement_all_days, 'hourly_disagreement_all_days.png'),
        'worker_thresholds': (plot_worker_task_thresholds, 'worker_task_thresholds.png')
    }
    
    # Remove image-related plots for ZOD
    if args.dataset == 'ZOD':
        plot_funcs.pop('image_disagree', None)
    
    if args.plot == 'all':
        for func, filename in plot_funcs.values():
            output_path = output_dir / filename
            func(df, str(output_path), show_stats=args.show_stats, dataset=args.dataset)
    else:
        func, filename = plot_funcs[args.plot]
        output_path = output_dir / filename
        func(df, str(output_path), show_stats=args.show_stats, dataset=args.dataset) 