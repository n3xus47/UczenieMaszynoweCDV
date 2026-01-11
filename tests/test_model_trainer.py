"""
Unit tests for ModelTrainer class
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer(random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y_train = pd.Series(np.random.randint(0, 3, 100))
        self.X_test = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20)
        })
    
    def test_initialize_models(self):
        """Test initializing models."""
        self.trainer.initialize_models()
        self.assertGreater(len(self.trainer.models), 0)
        self.assertIn('RandomForest', self.trainer.models)
        self.assertIn('LogisticRegression', self.trainer.models)
    
    def test_train_all(self):
        """Test training all models."""
        self.trainer.initialize_models()
        trained_models = self.trainer.train_all(self.X_train, self.y_train)
        
        self.assertIsInstance(trained_models, dict)
        self.assertGreater(len(trained_models), 0)
    
    def test_train_single(self):
        """Test training a single model."""
        self.trainer.initialize_models()
        model = self.trainer.train_single('RandomForest', self.X_train, self.y_train)
        
        self.assertIsNotNone(model)
        self.assertIn('RandomForest', self.trainer.trained_models)
    
    def test_predict(self):
        """Test making predictions."""
        self.trainer.initialize_models()
        self.trainer.train_single('RandomForest', self.X_train, self.y_train)
        
        predictions = self.trainer.predict('RandomForest', self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_predict_proba(self):
        """Test getting prediction probabilities."""
        self.trainer.initialize_models()
        self.trainer.train_single('RandomForest', self.X_train, self.y_train)
        
        probabilities = self.trainer.predict_proba('RandomForest', self.X_test)
        self.assertEqual(len(probabilities), len(self.X_test))
        self.assertIsInstance(probabilities, np.ndarray)
    
    def test_get_model(self):
        """Test getting a trained model."""
        self.trainer.initialize_models()
        self.trainer.train_single('RandomForest', self.X_train, self.y_train)
        
        model = self.trainer.get_model('RandomForest')
        self.assertIsNotNone(model)
    
    def test_get_all_models(self):
        """Test getting all trained models."""
        self.trainer.initialize_models()
        self.trainer.train_all(self.X_train, self.y_train)
        
        all_models = self.trainer.get_all_models()
        self.assertIsInstance(all_models, dict)
        self.assertGreater(len(all_models), 0)


if __name__ == '__main__':
    unittest.main()
