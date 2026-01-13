# Projekt ML - Przewidywanie Intencji Zakupowych Online Shoppers

## Opis Projektu

Projekt klasyfikacji binarnej przewidującej, czy użytkownik dokona zakupu (kolumna `Revenue`) na podstawie danych o zachowaniu na stronie e-commerce.

## Struktura Projektu

- `main.ipynb` - Główny notebook z pełną implementacją i analizą
- `ml_classes.py` - Klasy OOP (używane przez testy jednostkowe)
- `requirements.txt` - Zależności projektu
- `test_*.py` - Testy jednostkowe dla każdej klasy
- `RAPORT.md` - Szablon raportu/sprawozdania
- `online_shoppers_intention.csv` - Zbiór danych

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie

### Jupyter Notebook
Otwórz `main.ipynb` w Jupyter Notebook/Lab i wykonaj komórki sekwencyjnie.

### Testy jednostkowe
```bash
# Uruchom wszystkie testy
pytest test_*.py -v

# Lub pojedyncze testy
python -m pytest test_data_loader.py -v
python -m pytest test_preprocessor.py -v
python -m pytest test_feature_engineer.py -v
python -m pytest test_model_trainer.py -v
```

## Architektura OOP

Projekt wykorzystuje 6 głównych klas:

1. **DataLoader** - Wczytanie danych z pliku CSV
2. **DataPreprocessor** - Preprocessing (obsługa brakujących wartości, kodowanie, normalizacja)
3. **DataAnalyzer** - Analiza eksploracyjna i wizualizacje
4. **FeatureEngineer** - Feature engineering i selekcja zmiennych
5. **ModelTrainer** - Trenowanie i porównywanie modeli
6. **HyperparameterTuner** - Optymalizacja hiperparametrów (Grid Search, Random Search)

## Pipeline ML

1. **Pobranie danych** - Wczytanie z pliku CSV
2. **Preprocessing** - Normalizacja, kodowanie, obsługa brakujących wartości
3. **EDA** - Statystyki opisowe, korelacje, wizualizacje
4. **Feature Engineering** - Tworzenie nowych cech, selekcja zmiennych
5. **Przygotowanie zbiorów** - Podział train/test z stratyfikacją
6. **Trenowanie modeli** - Logistic Regression, Random Forest, SVM, XGBoost
7. **Fine-tuning** - Optymalizacja hiperparametrów
8. **Ewaluacja** - Metryki (Accuracy, Precision, Recall, F1, ROC-AUC), Confusion Matrix, ROC Curve

## Autorzy

[Imię i nazwisko członków grupy]

## Data

2026
