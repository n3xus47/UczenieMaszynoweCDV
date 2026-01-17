# Raport - Przewidywanie Intencji Zakupowych

### Opis problemu
Projekt dotyczy klasyfikacji binarnej przewidującej, czy użytkownik dokona zakupu w sklepie internetowym na podstawie danych o jego zachowaniu na stronie.

### Cel projektu
Na podstawie danych o zachowaniu użytkownika (liczba odwiedzonych stron, czas spędzony na stronie, wskaźniki odrzuceń itp.) przewidzi, czy użytkownik dokona zakupu (kolumna `Revenue`).

### Opis zbioru danych
Zbiór danych `online_shoppers_intention.csv` zawiera informacje o sesjach użytkowników w sklepie internetowym.

- **Liczba obserwacji**: 12,330
- **Liczba zmiennych**: 18 (17 cech + 1 zmienna docelowa)

### Opis zmiennych

#### Zmienne numeryczne:
- `Administrative`, `Informational`, `ProductRelated` - liczba odwiedzonych stron danego typu
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` - czas spędzony na stronach danego typu
- `BounceRates` - wskaźnik odrzuceń
- `ExitRates` - wskaźnik wyjść
- `PageValues` - wartość stron
- `SpecialDay` - bliskość specjalnego dnia (np. święta)

#### Zmienne kategoryczne:
- `Month` - miesiąc sesji
- `OperatingSystems` - system operacyjny
- `Browser` - przeglądarka
- `Region` - region
- `TrafficType` - typ ruchu
- `VisitorType` - typ odwiedzającego (Returning_Visitor, New_Visitor, Other)
- `Weekend` - czy sesja odbyła się w weekend

#### Zmienna docelowa:
- `Revenue` - czy użytkownik dokonał zakupu (TRUE/FALSE)

## Preprocessing

### Wykryte problemy
- **Brakujące wartości**: [W naszym przypadku nie znaleźliśmy żadnych brakujących wartości]
- **Typy danych**: Wszystkie zmienne kategoryczne wymagały kodowania
- **Niezbalansowanie klas**: [Procentowy rozkład: False    84.525547
                                                  True     15.474453]

### Zastosowane transformacje

1. **Obsługa brakujących wartości**: 
   - Strategia: [drop/mean/median/mode]
   - Uzasadnienie: [Wyjaśnij wybór]

2. **Kodowanie zmiennych kategorycznych**:
   - Metoda: Label Encoding
   - Zakodowane zmienne: Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType
   - Uzasadnienie: Label Encoding został wybrany, aby zachować strukturę danych i uniknąć zbyt wielu kolumn (w przeciwieństwie do One-Hot Encoding)

3. **Normalizacja**:
   - Metoda: StandardScaler (standaryzacja do średniej=0, std=1)
   - Uzasadnienie: Normalizacja jest ważna dla algorytmów wrażliwych na skalę (np. SVM, Logistic Regression)

## 4. Analiza Eksploracyjna (EDA)

### Statystyki opisowe
[Wstaw tutaj kluczowe statystyki - średnie, mediany, odchylenia standardowe dla najważniejszych zmiennych]

### Wnioski z wizualizacji
1. **Rozkłady zmiennych**: [Opisz główne obserwacje]
2. **Korelacje**: [Wymień najsilniejsze korelacje ze zmienną docelową]
3. **Balans klas**: [Opisz rozkład klas docelowych]

### Analiza korelacji
Najsilniejsze korelacje ze zmienną docelową `Revenue`:
- [Wpisz top 5-10 najsilniejszych korelacji]

## 5. Feature Engineering

### Utworzone nowe cechy

1. **TotalPages** = Administrative + Informational + ProductRelated
   - Uzasadnienie: Łączna liczba odwiedzonych stron może być lepszym predyktorem niż poszczególne wartości

2. **TotalDuration** = Administrative_Duration + Informational_Duration + ProductRelated_Duration
   - Uzasadnienie: Całkowity czas spędzony na stronie jest ważnym wskaźnikiem zaangażowania

3. **AvgPageDuration** = TotalDuration / TotalPages
   - Uzasadnienie: Średni czas na stronę może wskazywać na jakość interakcji

4. **BounceExitRatio** = BounceRates / ExitRates
   - Uzasadnienie: Stosunek odrzuceń do wyjść może być wskaźnikiem jakości sesji

5. **ProductRelatedRatio** = ProductRelated / TotalPages
   - Uzasadnienie: Proporcja stron produktowych może być silnym predyktorem intencji zakupowej

### Selekcja zmiennych
- **Metoda**: Feature Importance (Random Forest)
- **Próg**: 0.01
- **Liczba wybranych cech**: [Wpisz liczbę]
- **Uzasadnienie**: Random Forest pozwala na określenie ważności cech, co pomaga w usunięciu zmiennych o niskiej wartości predykcyjnej

## 6. Modelowanie

### Porównanie modeli

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] | [ ] | [ ] |
| SVM | [ ] | [ ] | [ ] | [ ] | [ ] |
| XGBoost | [ ] | [ ] | [ ] | [ ] | [ ] |

### Wybór najlepszego modelu
**Najlepszy model przed tuningiem**: [Nazwa modelu]
- **Uzasadnienie**: [Wyjaśnij dlaczego ten model osiągnął najlepsze wyniki]

### Optymalizacja hiperparametrów

**Model**: Random Forest

**Metoda**: Grid Search CV (5-fold)

**Hiperparametry testowane**:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

**Najlepsze parametry**: [Wpisz najlepsze parametry]
- `n_estimators`: [ ]
- `max_depth`: [ ]
- `min_samples_split`: [ ]
- `min_samples_leaf`: [ ]

**Uzasadnienie wyborów**: Grid Search pozwala na systematyczne przeszukanie przestrzeni parametrów, a cross-validation zapewnia niezawodną ocenę wydajności.

## 7. Wyniki i Ewaluacja

### Metryki najlepszego modelu (Random Forest - Tuned)

- **Accuracy**: [Wpisz wartość]
- **Precision**: [Wpisz wartość]
- **Recall**: [Wpisz wartość]
- **F1-score**: [Wpisz wartość]
- **ROC-AUC**: [Wpisz wartość]

### Analiza Confusion Matrix

| | Przewidziano: No Purchase | Przewidziano: Purchase |
|--|---------------------------|------------------------|
| **Rzeczywistość: No Purchase** | TN: [ ] | FP: [ ] |
| **Rzeczywistość: Purchase** | FN: [ ] | TP: [ ] |

**Interpretacja**:
- **True Positives (TP)**: [Liczba] - Poprawnie przewidziane zakupy
- **False Positives (FP)**: [Liczba] - Błędnie przewidziane zakupy (użytkownik nie kupił)
- **False Negatives (FN)**: [Liczba] - Przeoczone zakupy (użytkownik kupił, ale model nie przewidział)
- **True Negatives (TN)**: [Liczba] - Poprawnie przewidziane braki zakupów

### Wnioski z ROC Curve
- **AUC Score**: [Wartość]
- **Interpretacja**: [Opisz jakość modelu na podstawie AUC]

### Cross-Validation Results
- **Średni F1-score (5-fold CV)**: [Wartość] (+/- [odchylenie])
- **Interpretacja**: [Opisz stabilność modelu]

## 8. Konkluzja

### Podsumowanie wyników
[Napisz podsumowanie osiągniętych wyników, czy model spełnia oczekiwania, jakie są jego mocne i słabe strony]

### Analiza błędów
1. **False Positives**: [Liczba] - Model przewiduje zakup, ale użytkownik nie kupił
   - **Możliwe przyczyny**: [Przeanalizuj przykłady]
   
2. **False Negatives**: [Liczba] - Model nie przewiduje zakupu, ale użytkownik kupił
   - **Możliwe przyczyny**: [Przeanalizuj przykłady]

### Pomysły na dalszy rozwój

1. **Przetestowanie innych algorytmów**:
   - Gradient Boosting (LightGBM, CatBoost)
   - Neural Networks (Deep Learning)
   - Ensemble methods (Voting, Stacking)

2. **Zbalansowanie zbioru danych**:
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Undersampling większościowej klasy
   - Wagi klas w modelu

3. **Głębsza analiza feature engineering**:
   - Interakcje między zmiennymi
   - Transformacje nieliniowe
   - Analiza czasowa (jeśli dostępne dane czasowe)

4. **Optymalizacja metryk biznesowych**:
   - Dostosowanie progu decyzyjnego (threshold tuning)
   - Koszt-funkcja uwzględniająca koszty błędnych predykcji
   - Analiza ROI dla różnych strategii

5. **Analiza błędnych predykcji**:
   - Głęboka analiza przypadków False Positives i False Negatives
   - Identyfikacja wzorców w błędnych predykcjach
   - Segmentacja użytkowników

### Ograniczenia modelu

1. **Jakość danych**: [Opisz ograniczenia związane z danymi]
2. **Niezbalansowanie klas**: [Jeśli występuje, opisz wpływ]
3. **Brak danych kontekstowych**: [Czy brakuje ważnych informacji?]
4. **Overfitting**: [Czy model może być przeuczony?]

### Wnioski końcowe
[Napisz końcowe wnioski, czy projekt osiągnął cel, jakie są główne odkrycia, co można poprawić]

---

**Data utworzenia**: [Data]
**Autorzy**: [Imiona i nazwiska członków grupy]
