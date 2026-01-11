"""
Unit tests for DataPreprocessor class
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Patient_ID': ['P001', 'P002', 'P003'],
            'Age': [25, 30, 35],
            'Gender': ['Male', 'Female', 'Male'],
            'Condition': ['Infection', 'Hypertension', 'Diabetes'],
            'Drug_Name': ['DrugA', 'DrugB', 'DrugA'],
            'Dosage_mg': [100, 200, 150],
            'Treatment_Duration_days': [10, 20, 15],
            'Side_Effects': ['Nausea', 'Tiredness', 'Dizziness']
        })
    
    def test_preprocess(self):
        """Test preprocessing of data."""
        X, y = self.preprocessor.preprocess(
            self.sample_data,
            target_column='Drug_Name',
            drop_columns=['Patient_ID']
        )
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(self.sample_data))
        self.assertEqual(len(y), len(self.sample_data))
        self.assertNotIn('Patient_ID', X.columns)
        self.assertNotIn('Drug_Name', X.columns)
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'Age'] = np.nan
        data_with_missing.loc[1, 'Gender'] = None
        
        X, y = self.preprocessor.preprocess(
            data_with_missing,
            target_column='Drug_Name',
            drop_columns=['Patient_ID']
        )
        
        # Check that missing values are handled
        self.assertEqual(X.isnull().sum().sum(), 0)
    
    def test_encode_categorical(self):
        """Test encoding of categorical variables."""
        X, y = self.preprocessor.preprocess(
            self.sample_data,
            target_column='Drug_Name',
            drop_columns=['Patient_ID']
        )
        
        # Check that categorical columns are encoded (numeric)
        categorical_cols = ['Gender', 'Condition', 'Side_Effects']
        for col in categorical_cols:
            if col in X.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(X[col]))
    
    def test_normalize_numerical(self):
        """Test normalization of numerical variables."""
        X, y = self.preprocessor.preprocess(
            self.sample_data,
            target_column='Drug_Name',
            drop_columns=['Patient_ID']
        )
        
        # Check that numerical columns exist and are normalized
        numerical_cols = ['Age', 'Dosage_mg', 'Treatment_Duration_days']
        for col in numerical_cols:
            if col in X.columns:
                # Normalized data should have mean close to 0
                self.assertAlmostEqual(X[col].mean(), 0, places=1)
    
    def test_transform(self):
        """Test transforming new data."""
        # First preprocess training data
        X_train, y_train = self.preprocessor.preprocess(
            self.sample_data,
            target_column='Drug_Name',
            drop_columns=['Patient_ID']
        )
        
        # Transform new data
        new_data = self.sample_data.copy()
        new_data = new_data.drop(columns=['Drug_Name', 'Patient_ID'])
        X_transformed = self.preprocessor.transform(new_data)
        
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape[0], len(new_data))


if __name__ == '__main__':
    unittest.main()
