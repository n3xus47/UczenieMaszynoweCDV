"""
DataPreprocessor class for data cleaning and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, Dict


class DataPreprocessor:
    """
    Class responsible for data preprocessing:
    - Handling missing values
    - Encoding categorical variables
    - Normalizing numerical variables
    - Separating features and target
    """
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: Optional[list] = None
        self.target_column: Optional[str] = None
    
    def preprocess(self, 
                   data: pd.DataFrame, 
                   target_column: str = 'Drug_Name',
                   drop_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data: handle missing values, encode, normalize.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            drop_columns: List of columns to drop (e.g., ['Patient_ID'])
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = data.copy()
        
        # Drop specified columns (e.g., Patient_ID)
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
        
        # Separate target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical variables
        if categorical_cols:
            X = self._encode_categorical(X, categorical_cols)
        
        # Normalize numerical variables
        if numerical_cols:
            X = self._normalize_numerical(X, numerical_cols)
        
        # Store column information
        self.feature_columns = list(X.columns)
        self.target_column = target_column
        
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values
        missing = X.isnull().sum()
        
        if missing.sum() > 0:
            print(f"Found missing values:\n{missing[missing > 0]}")
            
            # For numerical columns: fill with median
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if X[col].isnull().sum() > 0:
                    X[col].fillna(X[col].median(), inplace=True)
            
            # For categorical columns: fill with mode
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    mode_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                    X[col].fillna(mode_value, inplace=True)
        
        return X
    
    def _encode_categorical(self, X: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            X: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # For test data, use existing encoder
                # Handle unseen categories
                known_classes = set(self.label_encoders[col].classes_)
                X_encoded[col] = X[col].astype(str).apply(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in known_classes else -1
                )
        
        return X_encoded
    
    def _normalize_numerical(self, X: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """
        Normalize numerical variables using StandardScaler.
        
        Args:
            X: Input DataFrame
            numerical_cols: List of numerical column names
            
        Returns:
            DataFrame with normalized numerical variables
        """
        X_normalized = X.copy()
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_normalized[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            # For test data, use existing scaler
            X_normalized[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X_normalized
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders and scalers.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.label_encoders and self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call preprocess() first.")
        
        X_transformed = X.copy()
        
        # Handle missing values
        X_transformed = self._handle_missing_values(X_transformed)
        
        # Encode categorical
        categorical_cols = X_transformed.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X_transformed = self._encode_categorical(X_transformed, categorical_cols)
        
        # Normalize numerical
        numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            X_transformed = self._normalize_numerical(X_transformed, numerical_cols)
        
        return X_transformed
