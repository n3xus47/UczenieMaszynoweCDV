# Projekt: Klasyfikacja leków na podstawie cech pacjenta

## Opis projektu

Projekt realizuje system klasyfikacji leków wykorzystujący uczenie maszynowe. Celem jest przewidywanie odpowiedniego leku (Drug_Name) na podstawie cech pacjenta takich jak wiek, płeć, schorzenie, dawka i czas trwania leczenia.

## Struktura projektu

```
UczenieMaszynoweCDV/
├── data/
│   └── real_drug_dataset.csv          # Zbiór danych
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Klasa do ładowania danych
│   ├── preprocessor.py                  # Klasa do preprocessing
│   ├── feature_engineer.py              # Klasa do feature engineering
│   ├── model_trainer.py                # Klasa do trenowania modeli
│   ├── hyperparameter_tuner.py         # Klasa do optymalizacji hiperparametrów
│   ├── evaluator.py                    # Klasa do ewaluacji
│   └── main.py                         # Główny pipeline
├── notebooks/
│   └── analysis.ipynb                  # Notebook z analizą EDA
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_feature_engineer.py
│   ├── test_model_trainer.py
│   └── test_evaluator.py
├── results/                            # Wyniki i wizualizacje
├── requirements.txt                    # Zależności
├── README.md                           # Ten plik
└── report.md                           # Raport końcowy
```

## Wymagania

- Python 3.8+
- Biblioteki wymienione w `requirements.txt`

## Instalacja

1. Sklonuj repozytorium lub pobierz pliki projektu.

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## Użycie

### Uruchomienie głównego pipeline'u

```bash
python src/main.py
```

Pipeline automatycznie:
1. Wczytuje dane z pliku CSV
2. Przeprowadza preprocessing
3. Tworzy nowe cechy
4. Trenuje wiele modeli ML
5. Optymalizuje hiperparametry
6. Ewaluuje wyniki
7. Generuje wizualizacje i raporty

### Uruchomienie analizy EDA

Otwórz notebook Jupyter:
```bash
jupyter notebook notebooks/analysis.ipynb
```

### Uruchomienie testów jednostkowych

```bash
pytest tests/
```

lub

```bash
python -m pytest tests/
```

## Architektura OOP

Projekt wykorzystuje programowanie obiektowe z następującymi klasami:

- **DataLoader**: Ładowanie danych z plików CSV z obsługą encoding
- **DataPreprocessor**: Preprocessing danych (encoding, normalizacja, obsługa brakujących wartości)
- **FeatureEngineer**: Tworzenie nowych cech i selekcja zmiennych
- **ModelTrainer**: Trenowanie wielu modeli ML (Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN)
- **HyperparameterTuner**: Optymalizacja hiperparametrów używając GridSearchCV/RandomizedSearchCV
- **ModelEvaluator**: Ewaluacja modeli z metrykami i wizualizacjami

## Funkcjonalności

### Preprocessing
- Obsługa brakujących wartości
- Label encoding dla zmiennych kategorycznych
- Standaryzacja zmiennych numerycznych
- Usuwanie niepotrzebnych kolumn

### Feature Engineering
- Tworzenie grup wiekowych
- Interakcje między zmiennymi
- Przekształcenia logarytmiczne
- Feature selection (mutual information, Random Forest importance)

### Modelowanie
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)

### Ewaluacja
- Accuracy, Precision, Recall, F1-score (macro i weighted)
- Confusion Matrix
- Feature Importance
- Porównanie modeli

## Wyniki

Wyniki są zapisywane w katalogu `results/`:
- `evaluation_results.csv` - Tabela z metrykami wszystkich modeli
- `confusion_matrix_*.png` - Macierze pomyłek
- `feature_importance_*.png` - Wykresy ważności cech
- `models_comparison.png` - Porównanie modeli
- `classification_report_*.txt` - Szczegółowe raporty klasyfikacji

## Autorzy

Projekt wykonany w ramach przedmiotu Uczenie Maszynowe.

## Licencja

Projekt edukacyjny.
