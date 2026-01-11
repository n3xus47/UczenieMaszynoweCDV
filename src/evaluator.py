"""
ModelEvaluator class for model evaluation and metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Optional
import os


class ModelEvaluator:
    """
    Class responsible for evaluating models and generating metrics and visualizations.
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results and plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: Dict[str, Dict[str, float]] = {}
    
    def evaluate(self, 
                model_name: str,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate a model and calculate all metrics.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision_Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Precision_Weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall_Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall_Weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'F1_Weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_all(self, 
                    models: Dict[str, Any],
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with results for all models
        """
        all_results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            y_pred = model.predict(X_test)
            
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            metrics = self.evaluate(model_name, y_test, y_pred, y_pred_proba)
            all_results.append({
                'Model': model_name,
                **metrics
            })
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        return results_df
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str,
                             class_names: Optional[list] = None):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            class_names: List of class names for labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {filename}")
    
    def plot_feature_importance(self, 
                               model: Any,
                               feature_names: list,
                               model_name: str,
                               top_n: int = 15):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance.")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {filename}")
    
    def plot_results_comparison(self, results_df: pd.DataFrame):
        """
        Plot comparison of all models.
        
        Args:
            results_df: DataFrame with evaluation results
        """
        metrics_to_plot = ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in results_df.columns:
                results_df.plot(x='Model', y=metric, kind='barh', ax=axes[idx], legend=False)
                axes[idx].set_title(f'{metric} Comparison')
                axes[idx].set_xlabel(metric)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'models_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Models comparison plot saved to {filename}")
    
    def generate_classification_report(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      model_name: str,
                                      class_names: Optional[list] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            class_names: List of class names
            
        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        filename = os.path.join(self.output_dir, f'classification_report_{model_name}.txt')
        with open(filename, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        print(f"Classification report saved to {filename}")
        return report
    
    def save_results(self, results_df: pd.DataFrame, filename: str = 'evaluation_results.csv'):
        """
        Save evaluation results to CSV.
        
        Args:
            results_df: DataFrame with results
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
