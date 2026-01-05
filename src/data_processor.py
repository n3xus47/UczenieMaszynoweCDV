import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.le_target = LabelEncoder()
        self.scaler = StandardScaler()
        # Kolumny kategoryczne (tekstowe) do zamiany na liczby (One-Hot)
        self.categorical_cols = ['Gender', 'Condition', 'Drug_Name']

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath)
            # Usuwamy ID pacjenta, bo to nie jest cecha medyczna
            if 'Patient_ID' in df.columns:
                df = df.drop(columns=['Patient_ID'])
            return df
        except FileNotFoundError:
            raise Exception(f"Nie znaleziono pliku: {self.filepath}")

    def prepare_data(self, df):
        # X to dane wejściowe, y to cel (Side_Effects)
        X = df.drop(columns=['Side_Effects', 'Improvement_Score'])
        y = df['Side_Effects']

        # Zamiana tekstu na liczby (One-Hot Encoding dla cech wejściowych)
        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)

        # Kodowanie celu (Label Encoding dla Side_Effects)
        y = self.le_target.fit_transform(y)

        # Podział na trening i test (80% trening, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Skalowanie danych liczbowych (Wiek, Dawka, Czas trwania)
        numeric_cols = ['Age', 'Dosage_mg', 'Treatment_Duration_days']
        
        # Ważne: fit_transform na treningowym, transform na testowym (żeby nie było wycieku danych)
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train, X_test, y_train, y_test

    def get_class_names(self):
        # Zwraca nazwy skutków ubocznych (np. 'Nausea', 'Headache')
        return self.le_target.classes_
