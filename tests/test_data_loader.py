"""
Unit tests for DataLoader class
"""

import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_path = 'data/real_drug_dataset.csv'
        self.loader = DataLoader(self.data_path)
    
    def test_load_data(self):
        """Test loading data from CSV file."""
        data = self.loader.load()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
    
    def test_get_data(self):
        """Test getting loaded data."""
        self.loader.load()
        data = self.loader.get_data()
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_get_data_without_load(self):
        """Test that get_data raises error if data not loaded."""
        with self.assertRaises(ValueError):
            self.loader.get_data()
    
    def test_get_info(self):
        """Test getting data information."""
        self.loader.load()
        info = self.loader.get_info()
        self.assertIn('shape', info)
        self.assertIn('columns', info)
        self.assertIn('missing_values', info)
        self.assertIn('dtypes', info)
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        loader = DataLoader('nonexistent_file.csv')
        with self.assertRaises(FileNotFoundError):
            loader.load()


if __name__ == '__main__':
    unittest.main()
