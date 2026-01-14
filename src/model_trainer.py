"""
ModelTrainer class for training multiple ML models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Any
import pickle


class ModelTrainer:
    """
    Class responsible for training multiple machine learning models.
    Supports: Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.trained_models: Dict[str, Any] = {}
    
    def initialize_models(self):
        """Initialize all models with default parameters."""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            )
        }
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
        
        self.trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"{name} trained successfully.")
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return self.trained_models
    
    def train_single(self, 
                    model_name: str, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        if not self.models:
            self.initialize_models()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        return model
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the trained model
            X: Features for prediction
            
        Returns:
            Prediction probabilities array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"Model {model_name} does not support predict_proba")
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model object
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        return self.trained_models[model_name]
    
    def get_all_models(self) -> Dict[str, Any]:
        """
        Get all trained models.
        
        Returns:
            Dictionary of all trained models
        """
        return self.trained_models.copy()
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the model file
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        if not self.trained_models:
            self.trained_models = {}
        
        self.trained_models[model_name] = model
        print(f"Model {model_name} loaded from {filepath}")
