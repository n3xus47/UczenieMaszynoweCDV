"""
Unit tests for ModelEvaluator class
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator(output_dir='test_results')
        
        # Create sample data
        np.random.seed(42)
        self.y_true = np.random.randint(0, 3, 100)
        self.y_pred = np.random.randint(0, 3, 100)
        self.y_pred_proba = np.random.rand(100, 3)
        self.y_pred_proba = self.y_pred_proba / self.y_pred_proba.sum(axis=1, keepdims=1)
    
    def test_evaluate(self):
        """Test evaluating a model."""
        metrics = self.evaluator.evaluate('TestModel', self.y_true, self.y_pred)
        
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision_Macro', metrics)
        self.assertIn('Recall_Macro', metrics)
        self.assertIn('F1_Macro', metrics)
        self.assertIsInstance(metrics['Accuracy'], (int, float))
    
    def test_evaluate_all(self):
        """Test evaluating multiple models."""
        # Create mock models
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 3, len(X))
        
        models = {
            'Model1': MockModel(),
            'Model2': MockModel()
        }
        
        X_test = pd.DataFrame(np.random.randn(100, 3))
        y_test = pd.Series(np.random.randint(0, 3, 100))
        
        results_df = self.evaluator.evaluate_all(models, X_test, y_test)
        
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertGreater(len(results_df), 0)
        self.assertIn('Model', results_df.columns)
        self.assertIn('Accuracy', results_df.columns)
    
    def test_plot_confusion_matrix(self):
        """Test plotting confusion matrix."""
        try:
            self.evaluator.plot_confusion_matrix(
                self.y_true, 
                self.y_pred, 
                'TestModel'
            )
            # If no exception is raised, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised {type(e).__name__}: {e}")
    
    def test_generate_classification_report(self):
        """Test generating classification report."""
        report = self.evaluator.generate_classification_report(
            self.y_true,
            self.y_pred,
            'TestModel'
        )
        
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    
    def test_save_results(self):
        """Test saving results to CSV."""
        results_df = pd.DataFrame({
            'Model': ['Model1', 'Model2'],
            'Accuracy': [0.8, 0.9],
            'F1_Macro': [0.75, 0.85]
        })
        
        try:
            self.evaluator.save_results(results_df, 'test_results.csv')
            # If no exception is raised, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"save_results raised {type(e).__name__}: {e}")


if __name__ == '__main__':
    unittest.main()
