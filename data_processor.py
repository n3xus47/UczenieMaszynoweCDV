import pandas as pd
import re
import string

class MovieDataProcessor:
    def __init__(self, file_path):
        """
        Inicjalizacja klasy ścieżką do pliku.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Wczytuje dane z pliku CSV i usuwa zbędną kolumnę indeksu.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            # Usunięcie kolumny 'Unnamed: 0', jeśli istnieje (częsty śmieć w CSV)
            if 'Unnamed: 0' in self.df.columns:
                self.df.drop(columns=['Unnamed: 0'], inplace=True)
            print("Dane wczytane poprawnie.")
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku {self.file_path}")

    def clean_data(self):
        """
        Czyszczenie danych: usuwanie pustych wierszy i wstępna obróbka tekstu.
        """
        if self.df is not None:
            # 1. Usuń wiersze, gdzie nie ma opisu (overview) lub oceny
            initial_count = len(self.df)
            self.df.dropna(subset=['overview', 'vote_average'], inplace=True)
            print(f"Usunięto {initial_count - len(self.df)} pustych wierszy.")

            # 2. Funkcja do czyszczenia tekstu (usuwanie znaków specjalnych, małe litery)
            def clean_text(text):
                text = str(text).lower()  # Zamiana na małe litery
                text = re.sub(f'[{string.punctuation}]', '', text)  # Usuwanie interpunkcji
                return text

            # Zastosowanie funkcji do kolumny overview
            self.df['clean_overview'] = self.df['overview'].apply(clean_text)
            print("Tekst został wyczyszczony.")
        else:
            print("Najpierw wczytaj dane metodą load_data()")

    def feature_engineering(self):
        """
        Tworzenie zmiennej celu (target) i przygotowanie danych do modelu.
        """
        if self.df is not None:
            # Tworzymy klasę: 1 jeśli film jest dobry (ocena > 6.5), 0 jeśli słabszy
            # 6.5 to mediana dla Twojego zbioru danych
            threshold = 6.5
            self.df['target'] = (self.df['vote_average'] > threshold).astype(int)
            
            print("Stworzono kolumnę 'target' (klasyfikacja: Hit vs Flop).")
            
            # Zwracamy podział w formacie X (tekst) i y (etykieta)
            return self.df['clean_overview'], self.df['target']
        else:
            print("Brak danych do przetworzenia.")
            return None, None
