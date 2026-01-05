from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self):
        # Random Forest świetnie radzi sobie z danymi tabelarycznymi
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        print("Trenowanie modelu Random Forest...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, class_names):
        print("Ewaluacja modelu...")
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\nDokładność modelu (Accuracy): {acc:.4f}")
        
        # zero_division=0 zapobiega ostrzeżeniom, gdy jakiejś klasy brakuje w zbiorze testowym
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
