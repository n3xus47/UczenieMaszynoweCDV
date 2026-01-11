"""
DataLoader class for loading CSV data files
"""

import pandas as pd
import os
from typing import Optional


class DataLoader:
    """
    Class responsible for loading data from CSV files.
    Handles encoding issues (BOM) and basic data validation.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize DataLoader with file path.
        
        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file with proper encoding handling.
        
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Try different encodings to handle BOM
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                self.data = pd.read_csv(self.file_path, encoding=encoding)
                # Remove BOM if present in column names
                if self.data.columns[0].startswith('\ufeff'):
                    self.data.columns = [col.replace('\ufeff', '') for col in self.data.columns]
                break
            except UnicodeDecodeError:
                continue
        
        if self.data is None:
            raise ValueError(f"Could not read file with any of the tried encodings: {encodings}")
        
        if self.data.empty:
            raise ValueError("Loaded file is empty")
        
        return self.data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get loaded data.
        
        Returns:
            DataFrame with loaded data
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data
    
    def get_info(self) -> dict:
        """
        Get basic information about loaded data.
        
        Returns:
            Dictionary with data info (shape, columns, missing values)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict()
        }
