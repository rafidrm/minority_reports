import pandas as pd
import argparse
from models.disagreement import fit_disagreement_model, fit_disagreement_model_with_activity
from models.disagreement_meworkers_r import fit_disagreement_meworkers_model as fit_disagreement_meworkers_model_r
from models.disagreement_meimages_r import fit_disagreement_meimages_model as fit_disagreement_meimages_model_r
from models.disagreement_mecrops_r import fit_disagreement_mecrops_model as fit_disagreement_mecrops_model_r
from models.disagreement_meworkerscrops_r import fit_disagreement_meworkerscrops_model as fit_disagreement_meworkerscrops_model_r


def load_data(dataset='ECPD', dataset_file='answers_logreg_all.csv'):
    """Load the preprocessed data"""
    filepath = f'data/{dataset}/{dataset_file}'
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def parse_args():
    parser = argparse.ArgumentParser(description='Train logistic regression models for disagreement prediction')
    
    parser.add_argument('--dataset', choices=['ECPD', 'ZOD'], default='ECPD',
                        help='Choose the dataset to process (ECPD or ZOD)')
    
    parser.add_argument('--dataset_file', type=str, 
                        default='answers_logreg_all.csv',
                        help='Path to the dataset file')
    
    parser.add_argument('--model_name', type=str, default='',
                        help='Name of the model to fit')
    
    parser.add_argument('--base', action='store_true',
                        help='Fit base model (questions and dates only)')
    
    parser.add_argument('--activity', action='store_true',
                        help='Fit model with activity hours (linear + quadratic)')
    
    parser.add_argument('--me_workers_r', action='store_true',
                        help='Fit mixed-effects model with worker random effects using R')
    
    parser.add_argument('--me_images_r', action='store_true',
                       help='Fit mixed-effects model with image random effects using R')
    
    parser.add_argument('--me_crops_r', action='store_true',
                       help='Fit mixed-effects model with crop random effects using R')
    
    parser.add_argument('--me_workerscrops_r', action='store_true',
                       help='Fit mixed-effects model with both worker and crop random effects using R')
    
    parser.add_argument('--all', action='store_true',
                        help='Fit all available models')
    
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with 10% of data')
    
    args = parser.parse_args()
    
    if not any([args.base,
                args.activity, 
                args.me_workers_r,
                args.me_images_r,
                args.me_crops_r, 
                args.me_workerscrops_r,
                args.all]):
        parser.print_help()
        exit(1)
        
    return args

def main():
    args = parse_args()
    df = load_data(args.dataset, args.dataset_file)
    
    if args.debug:
        print("\nRunning in debug mode with 0.1% of data...")
        df = df.sample(frac=0.01, random_state=42)
    
    fit_all = args.all
    
    if args.base or fit_all:
        print("\nFitting base disagreement model...")
        
        model_name = f'{args.model_name}_base'
        results_base = fit_disagreement_model(
            df, dataset=args.dataset, model_name=model_name)
        
        print(f"Base model pseudo R-squared: {results_base.prsquared:.4f}")
    
    if args.activity or fit_all:
        print("\nFitting disagreement model with activity hours...")
        
        model_name = f'{args.model_name}_activity'
        results_activity = fit_disagreement_model_with_activity(
            df, dataset=args.dataset, model_name=model_name)
        
        print(f"Activity model pseudo R-squared: {results_activity.prsquared:.4f}")
    
    if args.me_workers_r or fit_all:
        print("\nFitting mixed-effects model with worker random effects using R...")
        
        model_name = f'{args.model_name}_meworkers_r'
        results_me_workers_r = fit_disagreement_meworkers_model_r(
            df, dataset=args.dataset, model_name=model_name)
        
        print(f"Worker ME model log-likelihood: {results_me_workers_r.llf:.4f}")
        print(f"Worker ME model AIC: {results_me_workers_r.aic:.4f}")
    
    if args.me_images_r or fit_all:
        print("\nFitting mixed-effects model with image random effects using R...")
        
        model_name = f'{args.model_name}_meimages_r'
        results_me_images_r = fit_disagreement_meimages_model_r(
            df, dataset=args.dataset, model_name=model_name)
        
        print(f"Image ME model log-likelihood: {results_me_images_r.llf:.4f}")
        print(f"Image ME model AIC: {results_me_images_r.aic:.4f}")
    
    if args.me_crops_r or fit_all:
        print("\nFitting mixed-effects model with crop random effects using R...")
        
        model_name = f'{args.model_name}_mecrops_r'
        results_me_crops_r = fit_disagreement_mecrops_model_r(
            df, dataset=args.dataset, model_name=model_name)
        
        print(f"Crop ME model log-likelihood: {results_me_crops_r.llf:.4f}")
        print(f"Crop ME model AIC: {results_me_crops_r.aic:.4f}")
    
    if args.me_workerscrops_r or fit_all:
        print("\nFitting mixed-effects model with worker and crop random effects using R...")
        
        model_name = f'{args.model_name}_meworkerscrops_r'
        results_me_workerscrops_r = fit_disagreement_meworkerscrops_model_r(
            df, dataset=args.dataset, model_name=model_name)
        
        print(f"Worker+Crop ME model log-likelihood: {results_me_workerscrops_r.llf:.4f}")
        print(f"Worker+Crop ME model AIC: {results_me_workerscrops_r.aic:.4f}")
    
    print(f"\nModel results saved in results/ directory")

if __name__ == "__main__":
    main() 