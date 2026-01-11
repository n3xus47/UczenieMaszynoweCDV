"""
FeatureEngineer class for feature engineering and selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, List


class FeatureEngineer:
    """
    Class responsible for feature engineering:
    - Creating new features
    - Feature selection
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.selected_features: Optional[List[str]] = None
        self.feature_selector: Optional[SelectKBest] = None
    
    def create_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            X: Input features DataFrame
            y: Target Series
            
        Returns:
            DataFrame with original and new features
        """
        X_new = X.copy()
        
        # Get column names for reference
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Age groups (if Age column exists)
        if 'Age' in X_new.columns:
            X_new['Age_Group'] = pd.cut(
                X_new['Age'], 
                bins=[0, 30, 50, 70, 100], 
                labels=[0, 1, 2, 3]
            ).astype(int)
        
        # 2. Dosage per day (if both Dosage_mg and Treatment_Duration_days exist)
        if 'Dosage_mg' in X_new.columns and 'Treatment_Duration_days' in X_new.columns:
            X_new['Dosage_per_Day'] = X_new['Dosage_mg'] / (X_new['Treatment_Duration_days'] + 1)
            X_new['Total_Dosage'] = X_new['Dosage_mg'] * X_new['Treatment_Duration_days']
        
        # 3. Log transformations for skewed numerical features
        for col in ['Dosage_mg', 'Treatment_Duration_days']:
            if col in X_new.columns:
                # Add small value to avoid log(0)
                X_new[f'{col}_log'] = np.log1p(X_new[col])
        
        # 4. Interaction features (if both columns exist)
        if 'Age' in X_new.columns and 'Condition' in X_new.columns:
            # Age * Condition interaction (encoded)
            X_new['Age_Condition'] = X_new['Age'] * X_new['Condition']
        
        if 'Gender' in X_new.columns and 'Age' in X_new.columns:
            # Gender * Age interaction
            X_new['Gender_Age'] = X_new['Gender'] * X_new['Age']
        
        # 5. Polynomial features for key numerical columns
        if 'Age' in X_new.columns:
            X_new['Age_squared'] = X_new['Age'] ** 2
        
        if 'Dosage_mg' in X_new.columns:
            X_new['Dosage_squared'] = X_new['Dosage_mg'] ** 2
        
        return X_new
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       method: str = 'mutual_info',
                       k: Optional[int] = None) -> pd.DataFrame:
        """
        Select best features using various methods.
        
        Args:
            X: Input features DataFrame
            y: Target Series
            method: Selection method ('mutual_info', 'rf_importance', 'all')
            k: Number of features to select (None = select all)
            
        Returns:
            DataFrame with selected features
        """
        if method == 'mutual_info':
            return self._select_by_mutual_info(X, y, k)
        elif method == 'rf_importance':
            return self._select_by_rf_importance(X, y, k)
        elif method == 'all':
            # Return all features
            self.selected_features = list(X.columns)
            return X
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _select_by_mutual_info(self, 
                               X: pd.DataFrame, 
                               y: pd.Series, 
                               k: Optional[int]) -> pd.DataFrame:
        """
        Select features using mutual information.
        
        Args:
            X: Input features DataFrame
            y: Target Series
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if k is None:
            k = min(20, X.shape[1])  # Default to 20 or all features if less
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Get top k features
        top_k_indices = np.argsort(mi_scores)[-k:]
        self.selected_features = [X.columns[i] for i in top_k_indices]
        
        return X[self.selected_features]
    
    def _select_by_rf_importance(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series, 
                                 k: Optional[int]) -> pd.DataFrame:
        """
        Select features using Random Forest importance.
        
        Args:
            X: Input features DataFrame
            y: Target Series
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if k is None:
            k = min(20, X.shape[1])  # Default to 20 or all features if less
        
        # Train Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importances = rf.feature_importances_
        
        # Get top k features
        top_k_indices = np.argsort(importances)[-k:]
        self.selected_features = [X.columns[i] for i in top_k_indices]
        
        return X[self.selected_features]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using selected features.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            DataFrame with selected features only
        """
        if self.selected_features is None:
            raise ValueError("Features not selected. Call select_features() first.")
        
        # Check if all selected features exist
        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        return X[self.selected_features]
