from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
import os

def main():
    # Ścieżka do pliku z danymi
    DATA_FILE = 'data/real_drug_dataset.csv'
    
    # Sprawdzenie czy plik istnieje
    if not os.path.exists(DATA_FILE):
        print(f"BŁĄD: Brak pliku {DATA_FILE}.")
        print("Upewnij się, że skopiowałeś plik real_drug_dataset.csv do folderu data/")
        return

    try:
        # 1. Przetwarzanie danych
        print("--- KROK 1: Ładowanie i przetwarzanie danych ---")
        processor = DataProcessor(DATA_FILE)
        df = processor.load_data()
        print(f"Załadowano {len(df)} rekordów.")
        
        X_train, X_test, y_train, y_test = processor.prepare_data(df)
        print(f"Rozmiar zbioru treningowego: {X_train.shape}")
        
        # 2. Trenowanie modelu
        print("\n--- KROK 2: Trenowanie modelu ---")
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        
        # 3. Ewaluacja
        print("\n--- KROK 3: Wyniki ---")
        class_names = processor.get_class_names()
        trainer.evaluate(X_test, y_test, class_names)
        
    except Exception as e:
        print(f"Wystąpił błąd krytyczny: {e}")

if __name__ == "__main__":
    main()
