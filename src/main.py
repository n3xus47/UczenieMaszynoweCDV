"""
Main pipeline script for drug classification project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from hyperparameter_tuner import HyperparameterTuner
from evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def main():
    """
    Main pipeline function that orchestrates the entire ML workflow.
    """
    print("=" * 60)
    print("Drug Classification Project - Main Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    data_loader = DataLoader('data/real_drug_dataset.csv')
    data = data_loader.load()
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Columns: {list(data.columns)}")
    
    # Display basic info
    info = data_loader.get_info()
    print(f"\nData Info:")
    print(f"  Shape: {info['shape']}")
    print(f"  Missing values: {sum(info['missing_values'].values())} total")
    
    # Step 2: Preprocess data
    print("\n[2/7] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(
        data, 
        target_column='Drug_Name',
        drop_columns=['Patient_ID']
    )
    print(f"Features shape: {X.shape}")
    print(f"Target classes: {len(y.unique())}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Step 3: Feature Engineering
    print("\n[3/7] Feature engineering...")
    feature_engineer = FeatureEngineer()
    X = feature_engineer.create_features(X, y)
    print(f"Features after engineering: {X.shape[1]} features")
    
    # Feature selection
    X = feature_engineer.select_features(X, y, method='mutual_info', k=20)
    print(f"Features after selection: {X.shape[1]} features")
    
    # Step 4: Split data
    print("\n[4/7] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Train models
    print("\n[5/7] Training models...")
    model_trainer = ModelTrainer(random_state=42)
    model_trainer.initialize_models()
    trained_models = model_trainer.train_all(X_train, y_train)
    print(f"Trained {len(trained_models)} models")
    
    # Step 6: Hyperparameter tuning
    print("\n[6/7] Hyperparameter tuning...")
    tuner = HyperparameterTuner(cv=5, scoring='accuracy', random_state=42)
    
    # Tune all models (using random search for speed)
    best_models = tuner.tune_all_models(
        X_train, y_train, 
        search_method='random', 
        n_iter=10  # Reduced for faster execution
    )
    
    # Get best model
    best_type, best_model, best_score = tuner.get_best_model()
    print(f"\nBest model: {best_type} with CV score: {best_score:.4f}")
    
    # Display tuning results
    tuning_summary = tuner.get_results_summary()
    print("\nTuning Results Summary:")
    print(tuning_summary.to_string(index=False))
    
    # Step 7: Evaluation
    print("\n[7/7] Evaluating models...")
    evaluator = ModelEvaluator(output_dir='results')
    
    # Evaluate all tuned models
    results_df = evaluator.evaluate_all(best_models, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Save results
    evaluator.save_results(results_df, 'evaluation_results.csv')
    
    # Generate visualizations for best model
    best_model_name = results_df.iloc[0]['Model']
    best_model_obj = best_models[best_model_name]
    
    y_pred = best_model_obj.predict(X_test)
    y_pred_proba = None
    if hasattr(best_model_obj, 'predict_proba'):
        y_pred_proba = best_model_obj.predict_proba(X_test)
    
    # Get class names
    class_names = sorted(y.unique().tolist())
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(y_test, y_pred, best_model_name, class_names)
    
    # Feature importance (if applicable)
    if hasattr(best_model_obj, 'feature_importances_'):
        evaluator.plot_feature_importance(
            best_model_obj, 
            list(X.columns), 
            best_model_name
        )
    
    # Classification report
    evaluator.generate_classification_report(
        y_test, y_pred, best_model_name, class_names
    )
    
    # Comparison plot
    evaluator.plot_results_comparison(results_df)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nBest model: {best_model_name}")
    print(f"Test Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"Test F1-Score (Macro): {results_df.iloc[0]['F1_Macro']:.4f}")
    print(f"\nResults saved in 'results/' directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
