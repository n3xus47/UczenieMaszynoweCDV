import pytest
import pandas as pd
from src.data_processor import DataProcessor

# Fixture tworzy mały, tymczasowy plik CSV do testów
@pytest.fixture
def mock_csv(tmp_path):
    df = pd.DataFrame({
        'Patient_ID': ['P1', 'P2'],
        'Age': [30, 40],
        'Gender': ['Male', 'Female'],
        'Condition': ['Flu', 'Cold'],
        'Drug_Name': ['A', 'B'],
        'Dosage_mg': [100, 200],
        'Treatment_Duration_days': [5, 10],
        'Side_Effects': ['Nausea', 'Headache'],
        'Improvement_Score': [8.0, 7.0]
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_load_and_clean(mock_csv):
    """Testuje czy dane ładują się i czy usuwane jest ID"""
    proc = DataProcessor(mock_csv)
    df = proc.load_data()
    assert 'Patient_ID' not in df.columns
    assert len(df) == 2

def test_preprocessing_shape(mock_csv):
    """Testuje czy podział na X i y działa poprawnie"""
    proc = DataProcessor(mock_csv)
    df = proc.load_data()
    X_train, X_test, y_train, y_test = proc.prepare_data(df)
    
    # Sprawdzamy czy suma rekordów się zgadza
    assert len(X_train) + len(X_test) == 2
    # Sprawdzamy czy mamy kolumny po One-Hot Encoding
    assert X_train.shape[1] > 0
