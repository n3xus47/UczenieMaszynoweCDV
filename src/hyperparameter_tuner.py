"""
HyperparameterTuner class for hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Class responsible for hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    """
    
    def __init__(self, cv: int = 5, scoring: str = 'accuracy', n_jobs: int = -1, random_state: int = 42):
        """
        Initialize HyperparameterTuner.
        
        Args:
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.best_scores: Dict[str, float] = {}
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Get parameter grids for different models.
        
        Returns:
            Dictionary of parameter grids for each model type
        """
        return {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
    
    def tune_model(self, 
                   model_type: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   search_method: str = 'grid',
                   n_iter: Optional[int] = None) -> Any:
        """
        Tune hyperparameters for a specific model type.
        
        Args:
            model_type: Type of model ('RandomForest', 'GradientBoosting', 'SVM', etc.)
            X_train: Training features
            y_train: Training target
            search_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            Best model with optimized hyperparameters
        """
        param_grids = self.get_param_grids()
        
        if model_type not in param_grids:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(param_grids.keys())}")
        
        # Initialize base model
        base_models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
            'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=self.n_jobs),
            'KNN': KNeighborsClassifier()
        }
        
        base_model = base_models[model_type]
        param_grid = param_grids[model_type]
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif search_method == 'random':
            if n_iter is None:
                n_iter = 20  # Default for RandomizedSearchCV
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown search method: {search_method}. Use 'grid' or 'random'")
        
        print(f"Tuning {model_type} using {search_method} search...")
        search.fit(X_train, y_train)
        
        # Store results
        self.best_models[model_type] = search.best_estimator_
        self.best_params[model_type] = search.best_params_
        self.best_scores[model_type] = search.best_score_
        
        print(f"Best parameters for {model_type}: {search.best_params_}")
        print(f"Best {self.scoring} score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def tune_all_models(self, 
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       search_method: str = 'random',
                       n_iter: Optional[int] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for all available models.
        
        Args:
            X_train: Training features
            y_train: Training target
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            Dictionary of best models for each type
        """
        param_grids = self.get_param_grids()
        best_models = {}
        
        for model_type in param_grids.keys():
            try:
                best_model = self.tune_model(model_type, X_train, y_train, search_method, n_iter)
                best_models[model_type] = best_model
            except Exception as e:
                print(f"Error tuning {model_type}: {str(e)}")
                continue
        
        return best_models
    
    def get_best_model(self) -> tuple:
        """
        Get the best model based on cross-validation score.
        
        Returns:
            Tuple of (model_type, model, score)
        """
        if not self.best_scores:
            raise ValueError("No models have been tuned yet.")
        
        best_type = max(self.best_scores, key=self.best_scores.get)
        return best_type, self.best_models[best_type], self.best_scores[best_type]
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all tuning results.
        
        Returns:
            DataFrame with model types, best parameters, and scores
        """
        if not self.best_scores:
            return pd.DataFrame()
        
        results = []
        for model_type in self.best_scores.keys():
            results.append({
                'Model': model_type,
                'Best_Score': self.best_scores[model_type],
                'Best_Params': str(self.best_params[model_type])
            })
        
        return pd.DataFrame(results).sort_values('Best_Score', ascending=False)
