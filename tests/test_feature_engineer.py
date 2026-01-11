"""
Unit tests for FeatureEngineer class
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineer import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
        
        # Create sample data
        self.sample_X = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Gender': [0, 1, 0, 1, 0],
            'Condition': [0, 1, 2, 0, 1],
            'Dosage_mg': [100, 200, 150, 250, 300],
            'Treatment_Duration_days': [10, 20, 15, 25, 30]
        })
        
        self.sample_y = pd.Series([0, 1, 0, 1, 2])
    
    def test_create_features(self):
        """Test creating new features."""
        X_new = self.feature_engineer.create_features(self.sample_X, self.sample_y)
        
        self.assertIsInstance(X_new, pd.DataFrame)
        self.assertGreaterEqual(len(X_new.columns), len(self.sample_X.columns))
    
    def test_select_features_mutual_info(self):
        """Test feature selection using mutual information."""
        X_with_features = self.feature_engineer.create_features(self.sample_X, self.sample_y)
        X_selected = self.feature_engineer.select_features(
            X_with_features, 
            self.sample_y, 
            method='mutual_info',
            k=5
        )
        
        self.assertIsInstance(X_selected, pd.DataFrame)
        self.assertLessEqual(len(X_selected.columns), 5)
        self.assertIsNotNone(self.feature_engineer.selected_features)
    
    def test_select_features_rf_importance(self):
        """Test feature selection using Random Forest importance."""
        X_with_features = self.feature_engineer.create_features(self.sample_X, self.sample_y)
        X_selected = self.feature_engineer.select_features(
            X_with_features, 
            self.sample_y, 
            method='rf_importance',
            k=5
        )
        
        self.assertIsInstance(X_selected, pd.DataFrame)
        self.assertLessEqual(len(X_selected.columns), 5)
        self.assertIsNotNone(self.feature_engineer.selected_features)
    
    def test_select_features_all(self):
        """Test selecting all features."""
        X_with_features = self.feature_engineer.create_features(self.sample_X, self.sample_y)
        X_selected = self.feature_engineer.select_features(
            X_with_features, 
            self.sample_y, 
            method='all'
        )
        
        self.assertEqual(len(X_selected.columns), len(X_with_features.columns))
    
    def test_transform(self):
        """Test transforming new data with selected features."""
        X_with_features = self.feature_engineer.create_features(self.sample_X, self.sample_y)
        self.feature_engineer.select_features(
            X_with_features, 
            self.sample_y, 
            method='mutual_info',
            k=5
        )
        
        # Transform new data
        X_new = self.feature_engineer.create_features(self.sample_X, self.sample_y)
        X_transformed = self.feature_engineer.transform(X_new)
        
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(len(X_transformed.columns), len(self.feature_engineer.selected_features))


if __name__ == '__main__':
    unittest.main()
