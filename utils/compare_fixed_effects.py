import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.stats import chi2
import argparse

# Number of random effects for each model type
N_WORKERS = 88  # Number of unique workers
N_CROPS = 32711  # Number of unique crops

def parse_args():
    parser = argparse.ArgumentParser(description='Compare fixed effects across models')
    
    parser.add_argument('--odds_ratios', action='store_true',
                        help='Show odds ratios instead of coefficients')
    
    parser.add_argument('--format', choices=['text', 'latex'], default='text',
                        help='Output format (text or latex)')
    
    parser.add_argument('--lrt', action='store_true',
                        help='Perform Likelihood Ratio Tests')
    
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Choose the dataset to process (ECPD or ZOD)')
    
    return parser.parse_args()

def compare_fixed_effects(use_odds_ratio=False, format='text', base_path=Path('results/parameters')):
    """
    Create a comparison table of fixed effects across all models.
    
    Args:
        use_odds_ratio: If True, show odds ratios instead of parameter estimates
        format: 'text' or 'latex' for output format
        base_path: Path to the parameters directory
    """
    # Find all fixed effects files
    fixed_effects_files = list(base_path.glob('*_fixed_effects.csv'))
    
    # Load all fixed effects into a dictionary
    model_effects = {}
    for file_path in fixed_effects_files:
        model_name = file_path.stem.replace('_fixed_effects', '')
        df = pd.read_csv(file_path, index_col=0)  # Read index column as row names
        
        # Extract coefficients, standard errors, and calculate p-values
        effects_dict = {}
        for param_name, row in df.iterrows():
            # Normalize parameter names
            if param_name.lower() in ['cost', '(intercept)', 'intercept', 'const']:
                param_name = '(Intercept)'
            elif 'date' in param_name.lower():
                # Ensure consistent date formatting across models
                param_name = param_name.replace('date.', 'date_')
                param_name = param_name.replace('date-', 'date_')
                param_name = param_name.replace('-', '_')
            
            coef = np.exp(row['coef']) if use_odds_ratio else row['coef']
            p_value = 2 * (1 - norm.cdf(abs(row['z'])))
            effects_dict[param_name] = {
                'estimate': coef,
                'se': row['std_err'],
                'p_value': p_value
            }
        
        # Load random effects
        random_effects = load_random_effects(model_name, base_path)
        # Add random effects with special prefix to distinguish them
        for param, values in random_effects.items():
            effects_dict[f"RE_{param}"] = values
        
        model_effects[model_name] = effects_dict
    
    if not model_effects:
        print("No fixed effects files found in results/parameters/")
        return
    
    # Get all unique parameters across models
    all_params = set()
    for effects in model_effects.values():
        all_params.update(effects.keys())
    all_params = sorted(all_params)
    
    # Define desired order of models
    # model_order = [
    #     'disagreement_base',
    #     'disagreement_activity',
    #     'disagreement_meworkers_r',
    #     'disagreement_mecrops_r',
    #     'disagreement_meworkerscrops_r'
    # ]
    model_order = [
        'disagreement_activity',
        'ct1__activity',
        'ct30__activity',
        'ct60__activity',
    ]
    
    # Sort model_effects dictionary by desired order
    model_effects = {k: model_effects[k] for k in model_order if k in model_effects}
    
    # Create formatted strings for each cell
    def format_cell(estimate, se, p_value):
        """Format a cell with estimate, SE, and significance stars"""
        if estimate is None:
            return "---"
        
        stars = ''
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        
        if se is None:
            return f"{estimate:.3f}{stars}"
        else:
            return f"{estimate:.3f} ({se:.3f}){stars}"
    
    # Print header
    print("\nFixed Effects Comparison Table")
    print("=" * 80)
    header = "Parameter".ljust(30) + " | " + " | ".join(name.ljust(20) for name in model_effects.keys())
    print(header)
    print("-" * len(header))
    
    # Print each parameter row
    for param in all_params:
        row = [str(param).ljust(30)]
        for model_name in model_effects.keys():
            if param in model_effects[model_name]:
                effect = model_effects[model_name][param]
                cell = format_cell(
                    effect['estimate'],
                    effect['se'],
                    effect['p_value']
                )
            else:
                cell = "---"
            row.append(cell.ljust(20))
        print(" | ".join(row))
    
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
    if use_odds_ratio:
        print("Values shown as: Odds Ratio (Standard Error)")
    else:
        print("Values shown as: Estimate (Standard Error)")
    
    if format == 'latex':
        print_latex_table(model_effects, all_params, model_order, use_odds_ratio)
    
    if args.lrt:
        # Pass base_path to print_lrt_results
        print_lrt_results(base_path)

def format_latex_cell(estimate, se, p_value):
    """Format a cell for LaTeX with estimate, SE, and significance stars"""
    if estimate is None:
        return "---"
        
    stars = ''
    if p_value < 0.001:
        stars = '^{***}'
    elif p_value < 0.01:
        stars = '^{**}'
    elif p_value < 0.05:
        stars = '^{*}'
    
    if se is None:
        return f"${estimate:.3f}{stars}$"
    else:
        return f"${estimate:.3f}_{{{se:.3f}}}{stars}$"

def format_parameter_name(param):
    """Format parameter names for display in table"""
    # Handle random effects parameters
    if param.startswith('RE_'):
        effect_name = param[3:]  # Remove RE_ prefix
        if 'worker' in effect_name.lower():
            return 'Worker RE (SD)'
        elif 'crop' in effect_name.lower():
            return 'Crop RE (SD)'
        return f"Random Effect: {effect_name}"
    
    # Date formatting - handle both 2023 and 2024 dates
    if any(year in param for year in ['date_2023', 'date_2024']):
        try:
            date_parts = param.split('_')
            if len(date_parts) >= 3:  # Ensure we have enough parts
                year = date_parts[-1]
                day = date_parts[-2]
                month = 'Jan'  # Both datasets use January
                return f"{month} {int(day)}, {year}"
        except:
            return param  # Return original if parsing fails
    
    # Question formatting - only present in ECPD
    param_map = {
        'q_human_being': 'Human being',
        'q_on_bike': 'On bike',
        'q_on_wheels': 'On wheels',
        'q_poster': 'Poster',
        'q_statue_mannequin': 'Statue/mannequin',
        'continuous_activity_hours': 'Continuous activity',
        'activity_hours_sq': 'Continuous activity$^2$'
    }
    
    # Return mapped name if exists, otherwise return original
    return param_map.get(param, param)

def print_latex_table(model_effects, all_params, model_order, use_odds_ratio):
    """Print the table in LaTeX format with coefficients and SE in separate rows"""
    # Column headers
    display_names = {
        'disagreement_base': 'Base',
        'disagreement_activity': 'Activity',
        'disagreement_meworkers_r': 'Worker RE',
        'disagreement_mecrops_r': 'Crop RE',
        'disagreement_meworkerscrops_r': 'Worker+Crop RE',
        'ct1__activity': 'Activity (1 day)',
        'ct30__activity': 'Activity (30 days)',
        'ct60__activity': 'Activity (60 days)'
    }
    
    print("\n% LaTeX table")
    print("\\begin{table}")
    print("")
    print("\\centering")
    print("\\begin{tabular}{l" + "c" * len(model_order) + "}")
    print("\\toprule")
    print("")
    
    # Header row
    headers = ["Parameter"] + [display_names[m] for m in model_order if m in model_effects]
    print(" & ".join(headers) + " \\\\")
    print("")
    print("\\midrule")
    print("")
    
    # First print fixed effects
    fixed_params = [p for p in all_params if not p.startswith('RE_')]
    random_params = [p for p in all_params if p.startswith('RE_')]
    
    # Print fixed effects
    for param in fixed_params:
        if param.lower() in ['(intercept)', 'intercept', 'const']:
            continue
        
        # First row: Parameter name and coefficients
        coef_row = [format_parameter_name(param)]  # Use formatted parameter name
        se_row = [""]  # Empty cell under parameter name
        
        for model_name in model_effects.keys():
            if param in model_effects[model_name]:
                effect = model_effects[model_name][param]
                # Add stars to coefficient
                stars = ''
                if effect['p_value'] < 0.001:
                    stars = '^{***}'
                elif effect['p_value'] < 0.01:
                    stars = '^{**}'
                elif effect['p_value'] < 0.05:
                    stars = '^{*}'
                
                coef_row.append(f"${effect['estimate']:.3f}{stars}$")
                se_row.append(f"$(${effect['se']:.3f}$)$")
            else:
                coef_row.append("---")
                se_row.append("")
        
        print(" & ".join(coef_row) + " \\\\")
        print(" & ".join(se_row) + " \\\\[0.5em]")  # Add some vertical space between parameters
        print("")
    
    # Add a midrule before random effects
    if random_params:
        print("\\midrule")
        print("")
        
        # Print random effects
        for param in random_params:
            coef_row = [format_parameter_name(param)]
            se_row = [""]
            
            for model_name in model_effects.keys():
                if param in model_effects[model_name]:
                    effect = model_effects[model_name][param]
                    coef_row.append(f"${effect['estimate']:.3f}$")
                    se_row.append("")  # No SE for random effects
                else:
                    coef_row.append("---")
                    se_row.append("")
            
            print(" & ".join(coef_row) + " \\\\")
            print(" & ".join(se_row) + " \\\\[0.5em]")
            print("")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("")
    print("\\caption{Fixed Effects Comparison}")
    print("\\label{tab:fixed-effects}")
    print("")
    print("\\begin{tablenotes}")
    print("\\small")
    print("\\item Note: Standard errors shown in parentheses below estimates.")
    print("\\item Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    if use_odds_ratio:
        print("\\item Coefficients are shown as odds ratios")
    print("\\end{tablenotes}")
    print("")
    print("\\end{table}")
    print("")

def perform_lrt(model1_name, model2_name, base_path):
    """
    Perform Likelihood Ratio Test between two models.
    
    Args:
        model1_name: Name of first (simpler) model
        model2_name: Name of second (more complex) model
        base_path: Path to the parameters directory
    """
    try:
        # Get log-likelihoods from fit stats files
        with open(base_path / f'{model1_name}_fit_stats.txt') as f:
            for line in f:
                if 'Log-likelihood:' in line:
                    ll1 = float(line.split(':')[1])
                    break
        
        with open(base_path / f'{model2_name}_fit_stats.txt') as f:
            for line in f:
                if 'Log-likelihood:' in line:
                    ll2 = float(line.split(':')[1])
                    break
        
        # Calculate test statistic
        test_stat = -2 * (ll1 - ll2)
        
        # Hardcode degrees of freedom for each comparison
        df_map = {
            ('disagreement_base', 'disagreement_activity'): 2,
            ('disagreement_activity', 'disagreement_meworkers_r'): 1,
            ('disagreement_activity', 'disagreement_mecrops_r'): 1,
            ('disagreement_activity', 'disagreement_meworkerscrops_r'): 2,
            ('disagreement_meworkers_r', 'disagreement_meworkerscrops_r'): 1,
            ('disagreement_mecrops_r', 'disagreement_meworkerscrops_r'): 1
        }
        
        df = df_map.get((model1_name, model2_name))
        if df is None:
            raise ValueError(f"Unknown model comparison: {model1_name} vs {model2_name}")
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(test_stat, df)
        
        return {
            'test_stat': test_stat,
            'df': df,
            'p_value': p_value
        }
    except Exception as e:
        print(f"Warning: Could not perform LRT for {model1_name} vs {model2_name}: {str(e)}")
        return None

def print_lrt_results(base_path):
    """Print results of all Likelihood Ratio Tests"""
    print("\nLikelihood Ratio Tests")
    print("=" * 80)
    
    # Define comparisons
    comparisons = [
        ('disagreement_base', 'disagreement_activity', 'Activity vs Base'),
        ('disagreement_activity', 'disagreement_meworkers_r', 'Worker RE vs Activity'),
        ('disagreement_activity', 'disagreement_mecrops_r', 'Crop RE vs Activity'),
        ('disagreement_activity', 'disagreement_meworkerscrops_r', 'Worker+Crop RE vs Activity'),
        ('disagreement_meworkers_r', 'disagreement_meworkerscrops_r', 'Worker+Crop RE vs Worker RE'),
        ('disagreement_mecrops_r', 'disagreement_meworkerscrops_r', 'Worker+Crop RE vs Crop RE')
    ]
    
    print(f"{'Model Comparison':<40} {'Test Statistic':>15} {'df':>6} {'p-value':>12}")
    print("-" * 80)
    
    for model1, model2, label in comparisons:
        result = perform_lrt(model1, model2, base_path)  # Pass base_path here
        if result:
            print(f"{label:<40} {result['test_stat']:>15.2f} {result['df']:>6d} {result['p_value']:>12.2e}")
            if result['p_value'] < 0.001:
                print(f"{'=> Highly significant improvement':>80}")
            elif result['p_value'] < 0.01:
                print(f"{'=> Very significant improvement':>80}")
            elif result['p_value'] < 0.05:
                print(f"{'=> Significant improvement':>80}")
            else:
                print(f"{'=> No significant improvement':>80}")
            print()

def load_random_effects(model_name, base_path):
    """Load random effects parameters from summary file"""
    try:
        with open(base_path / f'{model_name}_summary.txt') as f:
            lines = f.readlines()
            
        random_effects = {}
        in_random_section = False
        for i, line in enumerate(lines):
            if 'Random effects:' in line:
                in_random_section = True
                continue
            if in_random_section and line.strip():
                if 'Number of obs:' in line:  # End of random effects section
                    break
                if 'Groups' in line:  # Skip header
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:  # Should have: group_id, name, variance, std.dev
                    group_name = parts[0]  # e.g., 'worker_id' or 'crop_id'
                    std_dev = float(parts[3])  # Standard deviation is the last number
                    random_effects[group_name] = {
                        'estimate': std_dev,
                        'se': None,  # SE not available for random effects
                        'p_value': 0  # Always show stars for random effects
                    }
        return random_effects
    except Exception as e:
        print(f"Warning: Could not load random effects for {model_name}: {str(e)}")
        return {}

if __name__ == "__main__":
    args = parse_args()
    base_path = Path(f'results/{args.dataset}/parameters')
    compare_fixed_effects(
        use_odds_ratio=args.odds_ratios, 
        format=args.format,
        base_path=base_path
    ) 