# Disagreement Analysis Project

This project analyzes patterns of disagreement in image annotation tasks, using various logistic regression models to understand the factors that contribute to annotator disagreements.

## Project Structure 

```
├── data/                      # Data directory
│   ├── ECPD/                 # ECPD dataset
│   │   ├── answers_revised.csv
│   │   └── answers_logreg_all.csv
│   └── ZOD/                  # ZOD dataset
├── models/                    # Model implementations
│   ├── base_model.py         # Base logistic regression model
│   ├── disagreement.py       # Basic disagreement models
│   ├── disagreement_meimages_r.py      # Mixed effects models (R implementation)
│   ├── disagreement_meworkers_r.py
│   ├── disagreement_mecrops_r.py
│   ├── disagreement_meworkerscrops_r.py
│   └── disagreement_simulation.py       # Simulation model
├── simulation/               # Simulation framework
│   ├── data_utils.py        # Data processing utilities
│   ├── pruning.py          # Task pruning strategies
│   └── analysis.py         # Analysis and visualization
├── utils/                   # Utility functions
│   ├── model_utils.py      # Common model utilities
│   ├── plot_style.py       # Plotting configuration
│   └── dataset_stats.py    # Dataset statistics
└── results/                 # Results directory
    ├── ECPD/               # ECPD results
    │   ├── parameters/     # Model parameters
    │   ├── plots/         # Generated plots
    │   └── simulation/    # Simulation results
    └── ZOD/               # ZOD results
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install R and required packages:
```R
install.packages(c("lme4", "Matrix"))
```

## Data Preprocessing

Before training models, the raw data needs to be processed. Starting with `answers_revised.csv` and `boxes.csv`, follow these steps:

1. Process the raw answers file:
```bash
python process_answers.py --dataset ECPD
```
This script:
- Adds task IDs for each unique crop_id x question pair
- Filters low-activity workers (< 100 tasks)
- Adds agreement/disagreement columns
- Adds image paths from boxes.csv
- Adds continuous activity tracking

2. Prepare data for logistic regression:
```bash
python preprocess_data_for_logreg.py --dataset ECPD
```
This script:
- Encodes categorical variables (questions, dates)
- Creates one-hot encoded features
- Prepares the final dataset for model training

The processed data will be saved as:
- `data/ECPD/answers_revised.csv` (after step 1)
- `data/ECPD/answers_logreg_all.csv` (after step 2)

## Usage

### Training Models

Train various disagreement prediction models:
```bash
python train_logreg.py --dataset ECPD --all
```

Options:
- `--dataset`: Choose dataset (ECPD or ZOD)
- `--base`: Train base model
- `--activity`: Include activity hours
- `--me_workers_r`: Mixed effects with worker random effects
- `--me_images_r`: Mixed effects with image random effects
- `--me_crops_r`: Mixed effects with crop random effects
- `--me_workerscrops_r`: Mixed effects with both worker and crop random effects
- `--all`: Train all model variants

### [WIP] Running Simulations

Simulate task pruning based on disagreement predictions:
```bash
python simulate_removal.py --dataset ECPD --experiment_name v1 --prune_threshold 0.1
```

Options:
- `--dataset`: Choose dataset (ECPD or ZOD)
- `--initial_training_hours`: Initial training period length (default: 24)
- `--recalibration_frequency_hours`: How often to recalibrate model (default: 6)
- `--prune_threshold`: Probability threshold for pruning (default: 0.1)
- `--pruning_strategy`: Pruning strategy to use (default: 'default')
- `--experiment_name`: Name for the experiment
- `--debug`: Run in debug mode with reduced dataset

### Analyzing Results

1. Generate exploratory data visualizations:
```bash
python plot_analysis.py --dataset ECPD
```
This generates several plots in `results/ECPD/exploratory_plots/`:
- Worker task distribution and activity patterns
- Worker and crop disagreement histograms
- Daily and hourly disagreement patterns
- Worker task threshold analysis

2. Generate dataset statistics:
```bash
python utils/dataset_stats.py --format latex
```
This produces summary statistics of the dataset including:
- Number of workers, tasks, and crops
- Average disagreement rates
- Task completion times
- Worker activity patterns

3. Compare model performance:
```bash
python utils/compare_fixed_effects.py --format latex
```
This analyzes:
- Fixed effects across different model variants
- Random effects distributions
- Model fit statistics and comparisons

All plots and analysis results are saved in the `results/` directory, organized by dataset and analysis type:
```
results/
├── ECPD/
│   ├── exploratory_plots/    # Data visualization plots
│   ├── parameters/           # Model parameters and effects
│   ├── plots/               # Model performance plots
│   └── simulation/          # Simulation results
└── ZOD/
    └── ...
```



##  [WIP] Known Issues

### Simulation Framework

- Majority vote calculations have inconsistent tie-breaking across different stages
- Edge cases in fully pruned tasks need better handling
- Accuracy calculations could be improved to consider vote confidence
- Quality metrics need refinement to better reflect dataset state


Configuration options:
- `initial_training_hours`: Initial training period length (default: 24)
- `recalibration_frequency_hours`: Model recalibration frequency (default: 6)
- `prune_threshold`: Probability threshold for pruning (default: 0.1)
- `pruning_strategy`: Strategy for task pruning (default: 'default')
