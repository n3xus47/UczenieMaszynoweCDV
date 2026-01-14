"""
Klasy OOP dla projektu ML - Online Shoppers Intention
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report


class DataLoader:
    """Klasa odpowiedzialna za wczytanie danych z pliku CSV"""
    
    def __init__(self):
        self.data = None
        self.path = None
    
    def load_data(self, path: str) -> pd.DataFrame:
        """
        Wczytuje dane z pliku CSV
        
        Parameters:
        -----------
        path : str
            Ścieżka do pliku CSV
            
        Returns:
        --------
        pd.DataFrame
            Wczytane dane
        """
        try:
            self.data = pd.read_csv(path)
            self.path = path
            print(f"Dane wczytane pomyślnie z: {path}")
            print(f"Kształt danych: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Błąd podczas wczytania danych: {e}")
            return None
    
    def get_info(self) -> dict:
        """
        Zwraca podstawowe informacje o zbiorze danych
        
        Returns:
        --------
        dict
            Słownik z informacjami o danych
        """
        if self.data is None:
            return {"error": "Dane nie zostały wczytane"}
        
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum()
        }
        return info


class DataPreprocessor:
    """Klasa odpowiedzialna za preprocessing danych"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Obsługuje brakujące wartości
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do przetworzenia
        strategy : str
            Strategia obsługi ('drop', 'mean', 'median', 'mode')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame po obsłudze brakujących wartości
        """
        df_processed = df.copy()
        
        if strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'mean':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif strategy == 'mode':
            for col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0)
        
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, columns: list = None, method: str = 'label') -> pd.DataFrame:
        """
        Koduje zmienne kategoryczne
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do przetworzenia
        columns : list
            Lista kolumn do zakodowania (None = automatyczne wykrycie)
        method : str
            Metoda kodowania ('label' lub 'onehot')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame z zakodowanymi zmiennymi
        """
        df_encoded = df.copy()
        
        if columns is None:
            # Automatyczne wykrycie kolumn kategorycznych
            categorical_cols = df_encoded.select_dtypes(include=['object', 'bool']).columns.tolist()
        else:
            categorical_cols = columns
        
        if method == 'label':
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, columns: list = None, fit: bool = True) -> pd.DataFrame:
        """
        Normalizuje zmienne numeryczne
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do przetworzenia
        columns : list
            Lista kolumn do normalizacji (None = wszystkie numeryczne)
        fit : bool
            Czy dopasować scaler (True dla train, False dla test)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame z znormalizowanymi zmiennymi
        """
        df_normalized = df.copy()
        
        if columns is None:
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = columns
        
        if fit:
            df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler nie został dopasowany. Użyj fit=True dla danych treningowych.")
            df_normalized[numeric_cols] = self.scaler.transform(df_normalized[numeric_cols])
        
        return df_normalized
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str = None, 
                           handle_missing: str = 'drop', encode_method: str = 'label',
                           normalize: bool = True, fit: bool = True) -> tuple:
        """
        Pełny pipeline preprocessingu
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do przetworzenia
        target_col : str
            Nazwa kolumny docelowej (jeśli None, nie separuje)
        handle_missing : str
            Strategia obsługi brakujących wartości
        encode_method : str
            Metoda kodowania kategorycznych
        normalize : bool
            Czy normalizować zmienne
        fit : bool
            Czy dopasować transformatory (True dla train)
            
        Returns:
        --------
        tuple
            (X, y) lub (df_processed, None) jeśli target_col=None
        """
        df_processed = df.copy()
        
        # Obsługa brakujących wartości
        df_processed = self.handle_missing_values(df_processed, strategy=handle_missing)
        
        # Separacja targetu jeśli podany
        if target_col and target_col in df_processed.columns:
            y = df_processed[target_col].copy()
            X = df_processed.drop(columns=[target_col])
        else:
            X = df_processed
            y = None
        
        # Kodowanie kategorycznych
        X = self.encode_categorical(X, method=encode_method)
        
        # Normalizacja
        if normalize:
            X = self.normalize_features(X, fit=fit)
        
        if y is not None:
            return X, y
        else:
            return X, None


class DataAnalyzer:
    """Klasa odpowiedzialna za analizę danych i wizualizacje"""
    
    def __init__(self):
        pass
    
    def descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zwraca statystyki opisowe
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do analizy
            
        Returns:
        --------
        pd.DataFrame
            Statystyki opisowe
        """
        return df.describe()
    
    def correlation_analysis(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """
        Analiza korelacji
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do analizy
        target : str
            Nazwa kolumny docelowej
            
        Returns:
        --------
        pd.DataFrame
            Macierz korelacji lub korelacje z targetem
        """
        # Wybierz tylko kolumny numeryczne dla korelacji
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target and target in df.columns:
            # Sprawdź czy target jest numeryczny
            if target in numeric_df.columns:
                correlations = numeric_df.corr()[target].sort_values(ascending=False)
                return correlations
            else:
                # Jeśli target nie jest numeryczny, zwróć korelacje wszystkich numerycznych zmiennych
                print(f"Uwaga: Kolumna '{target}' nie jest numeryczna. Zwracam korelacje wszystkich zmiennych numerycznych.")
                return numeric_df.corr()
        else:
            return numeric_df.corr()
    
    def visualize_distributions(self, df: pd.DataFrame, columns: list = None, figsize: tuple = (15, 10)):
        """
        Wizualizuje rozkłady zmiennych numerycznych
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do wizualizacji
        columns : list
            Lista kolumn do wizualizacji (None = wszystkie numeryczne)
        figsize : tuple
            Rozmiar wykresu
        """
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = columns
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Rozkład {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Częstość')
        
        # Ukryj puste subploty
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_correlations(self, df: pd.DataFrame, figsize: tuple = (12, 10)):
        """
        Wizualizuje macierz korelacji
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do wizualizacji
        figsize : tuple
            Rozmiar wykresu
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Macierz korelacji')
        plt.tight_layout()
        plt.show()
    
    def class_balance_analysis(self, y: pd.Series) -> dict:
        """
        Analiza balansu klas
        
        Parameters:
        -----------
        y : pd.Series
            Zmienna docelowa
            
        Returns:
        --------
        dict
            Słownik z informacjami o balansie klas
        """
        value_counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100
        
        analysis = {
            "counts": value_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "is_balanced": (percentages.min() > 40) and (percentages.max() < 60)
        }
        
        # Wizualizacja
        plt.figure(figsize=(8, 5))
        value_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Rozkład klas docelowych')
        plt.xlabel('Klasa')
        plt.ylabel('Liczba obserwacji')
        plt.xticks(rotation=0)
        for i, v in enumerate(value_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        
        return analysis


class FeatureEngineer:
    """Klasa odpowiedzialna za feature engineering"""
    
    def __init__(self):
        self.feature_importance = None
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tworzy cechy interakcyjne
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do przetworzenia
            
        Returns:
        --------
        pd.DataFrame
            DataFrame z nowymi cechami
        """
        df_new = df.copy()
        
        # TotalPages = suma wszystkich stron
        if all(col in df_new.columns for col in ['Administrative', 'Informational', 'ProductRelated']):
            df_new['TotalPages'] = (df_new['Administrative'] + 
                                   df_new['Informational'] + 
                                   df_new['ProductRelated'])
        
        # TotalDuration = suma wszystkich czasów
        if all(col in df_new.columns for col in ['Administrative_Duration', 
                                                  'Informational_Duration', 
                                                  'ProductRelated_Duration']):
            df_new['TotalDuration'] = (df_new['Administrative_Duration'] + 
                                       df_new['Informational_Duration'] + 
                                       df_new['ProductRelated_Duration'])
        
        # AvgPageDuration = średni czas na stronę
        if 'TotalDuration' in df_new.columns and 'TotalPages' in df_new.columns:
            df_new['AvgPageDuration'] = df_new['TotalDuration'] / (df_new['TotalPages'] + 1e-6)
        
        # BounceExitRatio = stosunek bounce do exit
        if all(col in df_new.columns for col in ['BounceRates', 'ExitRates']):
            df_new['BounceExitRatio'] = df_new['BounceRates'] / (df_new['ExitRates'] + 1e-6)
        
        # ProductRelatedRatio = stosunek stron produktowych do wszystkich
        if 'ProductRelated' in df_new.columns and 'TotalPages' in df_new.columns:
            df_new['ProductRelatedRatio'] = df_new['ProductRelated'] / (df_new['TotalPages'] + 1e-6)
        
        return df_new
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tworzy cechy zagregowane
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame do przetworzenia
            
        Returns:
        --------
        pd.DataFrame
            DataFrame z nowymi cechami
        """
        df_new = df.copy()
        
        # Można dodać więcej cech zagregowanych tutaj
        # Na przykład: średnie, mediany, itp.
        
        return df_new
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'correlation', threshold: float = 0.01) -> tuple:
        """
        Selekcja zmiennych
        
        Parameters:
        -----------
        X : pd.DataFrame
            Cechy
        y : pd.Series
            Zmienna docelowa
        method : str
            Metoda selekcji ('correlation', 'importance')
        threshold : float
            Próg dla selekcji
            
        Returns:
        --------
        tuple
            (X_selected, selected_features)
        """
        if method == 'correlation':
            correlations = X.corrwith(y).abs()
            selected_features = correlations[correlations >= threshold].index.tolist()
            X_selected = X[selected_features]
        
        elif method == 'importance':
            # Użyj Random Forest do określenia ważności cech
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            self.feature_importance = feature_importance.sort_values(ascending=False)
            
            selected_features = feature_importance[feature_importance >= threshold].index.tolist()
            X_selected = X[selected_features]
        
        else:
            X_selected = X
            selected_features = X.columns.tolist()
        
        return X_selected, selected_features


class ModelTrainer:
    """Klasa odpowiedzialna za trenowanie modeli"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_model(self, X_train, y_train, model_type: str, **kwargs):
        """
        Trenuje model
        
        Parameters:
        -----------
        X_train : array-like
            Cechy treningowe
        y_train : array-like
            Zmienna docelowa treningowa
        model_type : str
            Typ modelu ('logistic', 'random_forest', 'svm', 'xgboost')
        **kwargs
            Dodatkowe parametry modelu
            
        Returns:
        --------
        model
            Wytrenowany model
        """
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
        elif model_type == 'svm':
            model = SVC(random_state=42, probability=True, **kwargs)
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(random_state=42, **kwargs)
            except ImportError:
                print("XGBoost nie jest zainstalowany. Używam Random Forest zamiast tego.")
                model = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
        else:
            raise ValueError(f"Nieznany typ modelu: {model_type}")
        
        model.fit(X_train, y_train)
        self.models[model_type] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test) -> dict:
        """
        Ewaluuje model (obsługuje zarówno zwykłe modele jak i Pipeline)
        
        Parameters:
        -----------
        model : model lub Pipeline
            Model do ewaluacji (może być Pipeline z sklearn)
        X_test : array-like
            Cechy testowe (jeśli model to Pipeline, dane powinny być przed skalowaniem)
        y_test : array-like
            Zmienna docelowa testowa
            
        Returns:
        --------
        dict
            Słownik z metrykami
        """
        # Pipeline automatycznie zastosuje skalowanie podczas predict
        y_pred = model.predict(X_test)
        
        # Sprawdź czy model ma predict_proba (obsługuje Pipeline)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'predict_proba'):
            # Pipeline z modelem mającym predict_proba
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def compare_models(self, models: dict, X_test, y_test) -> pd.DataFrame:
        """
        Porównuje wiele modeli
        
        Parameters:
        -----------
        models : dict
            Słownik modeli {nazwa: model}
        X_test : array-like
            Cechy testowe
        y_test : array-like
            Zmienna docelowa testowa
            
        Returns:
        --------
        pd.DataFrame
            DataFrame z wynikami porównania
        """
        results = []
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics['model'] = name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('model')
        
        return results_df


class HyperparameterTuner:
    """Klasa odpowiedzialna za optymalizację hiperparametrów"""
    
    def __init__(self):
        self.best_params = {}
        self.best_scores = {}
    
    def grid_search(self, model, param_grid: dict, X_train, y_train, 
                   cv: int = 5, scoring: str = 'f1', n_jobs: int = -1):
        """
        Grid Search dla optymalizacji hiperparametrów z użyciem Pipeline
        
        Tworzy Pipeline składający się ze StandardScaler i modelu, dzięki czemu
        skalowanie odbywa się osobno dla każdego foldu walidacji krzyżowej
        (uniknięcie data leakage).
        
        Parameters:
        -----------
        model : model
            Model do optymalizacji
        param_grid : dict
            Siatka parametrów (musi używać prefiksu 'model__' dla parametrów modelu)
            Przykład: {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]}
        X_train : array-like
            Cechy treningowe (przed skalowaniem - surowe dane po feature engineering)
        y_train : array-like
            Zmienna docelowa treningowa
        cv : int
            Liczba foldów cross-validation
        scoring : str
            Metryka do optymalizacji
        n_jobs : int
            Liczba równoległych zadań
            
        Returns:
        --------
        Pipeline
            Najlepszy Pipeline (scaler + model)
        """
        # Tworzenie Pipeline ze StandardScaler i modelem
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params[type(model).__name__] = grid_search.best_params_
        self.best_scores[type(model).__name__] = grid_search.best_score_
        
        print(f"Najlepsze parametry: {grid_search.best_params_}")
        print(f"Najlepszy wynik CV: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def random_search(self, model, param_distributions: dict, X_train, y_train,
                     n_iter: int = 50, cv: int = 5, scoring: str = 'f1', n_jobs: int = -1):
        """
        Random Search dla optymalizacji hiperparametrów z użyciem Pipeline
        
        Tworzy Pipeline składający się ze StandardScaler i modelu, dzięki czemu
        skalowanie odbywa się osobno dla każdego foldu walidacji krzyżowej
        (uniknięcie data leakage).
        
        Parameters:
        -----------
        model : model
            Model do optymalizacji
        param_distributions : dict
            Rozkłady parametrów (musi używać prefiksu 'model__' dla parametrów modelu)
            Przykład: {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]}
        X_train : array-like
            Cechy treningowe (przed skalowaniem - surowe dane po feature engineering)
        y_train : array-like
            Zmienna docelowa treningowa
        n_iter : int
            Liczba iteracji
        cv : int
            Liczba foldów cross-validation
        scoring : str
            Metryka do optymalizacji
        n_jobs : int
            Liczba równoległych zadań
            
        Returns:
        --------
        Pipeline
            Najlepszy Pipeline (scaler + model)
        """
        # Tworzenie Pipeline ze StandardScaler i modelem
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params[type(model).__name__] = random_search.best_params_
        self.best_scores[type(model).__name__] = random_search.best_score_
        
        print(f"Najlepsze parametry: {random_search.best_params_}")
        print(f"Najlepszy wynik CV: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
