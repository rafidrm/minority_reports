import pandas as pd
import numpy as np
import statsmodels.api as sm # type: ignore
from pathlib import Path
from utils.model_utils import plot_roc_curve, save_model_files
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogisticModel:
    def __init__(self, model_name, dataset='ECPD'):
        self.model_name = model_name
        self.dataset = dataset
        self.results_dir = Path(f'results/{dataset}')
        self.param_dir = self.results_dir / 'parameters'
        self.plot_dir = self.results_dir / 'plots'
        self.test_dir = self.results_dir / 'test'
        
        # Create directories if they don't exist
        for d in [self.param_dir, self.plot_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def fit(self, X, y, plot_roc=True):
        """Fit logistic regression and save results"""
        # Create tmp directory if it doesn't exist
        tmp_dir = Path('tmp')
        tmp_dir.mkdir(exist_ok=True)
        logger.info(f"Using temporary directory: {tmp_dir}")
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit model with modified parameters
        logger.info("Fitting logistic regression model...")
        model = sm.Logit(y, X)
        self.results = model.fit(
            method='lbfgs',          # Try a different optimization method
            maxiter=10000,           # Increase max iterations
            tol=1e-8,               # Adjust tolerance
            disp=True               # Show convergence messages
        )
        
        # Save intermediate files to tmp
        logger.info("Saving intermediate files to tmp directory...")
        self._save_summary()
        self._save_parameters()
        
        # Get predictions
        logger.info("Generating predictions...")
        y_pred = self.results.predict(X)
        
        # Save predictions in same format as R models
        predictions = pd.DataFrame({'pred': y_pred})
        pred_path = tmp_dir / 'predictions.csv'
        predictions.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to: {pred_path}")
        
        # Save outputs using utility function
        logger.info("Moving files to final location...")
        save_model_files(
            self.model_name, 
            {
                'summary.txt': 'summary.txt',
                'predictions.csv': 'predictions.csv',
                'parameters.csv': 'fixed_effects.csv'  # Rename for consistency
            },
            dataset=self.dataset,
            subdir='parameters'
        )
        
        # Verify predictions file was moved correctly
        final_pred_path = self.param_dir / f'{self.model_name}_predictions.csv'
        if final_pred_path.exists():
            logger.info(f"Successfully moved predictions to: {final_pred_path}")
        else:
            logger.warning(f"Failed to find predictions at: {final_pred_path}")
        
        # Save additional stats
        self._save_pseudor2()
        
        if plot_roc:
            self._plot_roc_curve(y, y_pred)  # Pass predictions directly
        
        return self.results
    
    def _save_summary(self):
        """Save full model summary"""
        summary_path = Path('tmp/summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self.results.summary().as_text())
        logger.info(f"Saved model summary to: {summary_path}")
    
    def _save_parameters(self):
        """Save parameter estimates and confidence intervals"""
        params_df = pd.DataFrame({
            'coef': self.results.params,
            'std_err': self.results.bse,
            'z': self.results.tvalues,
            'P>|z|': self.results.pvalues,
            '[0.025': self.results.conf_int()[0],
            '0.975]': self.results.conf_int()[1]
        })
        params_path = Path('tmp/parameters.csv')
        params_df.to_csv(params_path)
        logger.info(f"Saved model parameters to: {params_path}")
    
    def _save_pseudor2(self):
        """Save pseudo R-squared and log-likelihood statistics"""
        stats_path = self.param_dir / f'{self.model_name}_fit_stats.txt'
        with open(stats_path, 'w') as f:
            f.write(f"Log-likelihood: {self.results.llf:.4f}\n")
            f.write(f"McFadden's pseudo R²: {self.results.prsquared:.4f}\n")
        logger.info(f"Saved model fit statistics to: {stats_path}")
    
    def _plot_roc_curve(self, y, y_pred):
        """Generate and save ROC curve using utility function"""
        plot_path = self.plot_dir / f'{self.model_name}_roc.png'
        plot_roc_curve(
            y, y_pred, self.model_name,
            str(plot_path)
        )
        logger.info(f"Saved ROC curve plot to: {plot_path}") 