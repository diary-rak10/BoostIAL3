import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os
from pathlib import Path

# Étape 1 : Vérifier et charger les données
def load_data(file_path="diabetes.csv"):    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Le fichier {file_path} est introuvable. Assurez-vous qu'il est dans {os.getcwd()}."
        )
    try:
        data = pd.read_csv(file_path)
        required_columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Le fichier CSV ne contient pas toutes les colonnes requises.")
        return data
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du fichier CSV : {e}")

# Étape 2 : Préparer les données
def prepare_data(data):
    try:
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, scaler
    except Exception as e:
        raise Exception(f"Erreur lors de la préparation des données : {e}")

# Étape 3 : Créer ou charger le modèle
def build_or_load_model(X_train, y_train, X_test, y_test, model_path="model.h5"):
    try:
        if os.path.exists(model_path):
            print(" Chargement du modèle existant...")
            return load_model(model_path)
        
        print("🛠️ Création d'un nouveau modèle...")
        model = Sequential([
            Dense(6, input_dim=8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Entraînement du modèle en cours...")
        model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Précision sur les données de test : {accuracy * 100:.2f}%")
        
        model.save(model_path)
        print(f"Modèle sauvegardé sous {model_path}")
        
        return model
    except Exception as e:
        raise Exception(f"Erreur lors de la création ou du chargement du modèle : {e}")

# Étape 4 : Interface console pour saisir un patient
def saisir_patient():
    print("\n Entrez les informations du patient :")
    champs = [
        ("Nombre de grossesses", "grossesses (ex. 0 à 17)"),
        ("Taux de glucose (mg/dL)", "glucose (ex. 0 à 200)"),
        ("Pression artérielle (mm Hg)", "pression (ex. 0 à 122)"),
        ("Épaisseur de la peau (mm)", "épaisseur (ex. 0 à 99)"),
        ("Insuline (µU/mL)", "insuline (ex. 0 à 846)"),
        ("IMC (kg/m²)", "IMC (ex. 0 à 67.1)"),
        ("Antécédents familiaux du diabète (0 à 2)", "antécédents (ex. 0.078 à 2.42)"),
        ("Âge (ans)", "âge (ex. 21 à 81)")
    ]
    valeurs = []
    for champ, desc in champs:
        while True:
            try:
                valeur = float(input(f"{champ} ({desc}) : "))
                if valeur < 0:
                    print(" La valeur ne peut pas être négative.")
                    continue
                valeurs.append(valeur)
                break
            except ValueError:
                print(" Valeur invalide, veuillez entrer un nombre.")
    return np.array([valeurs])

# Étape 5 : Prédire pour un patient
def predire_diabete(model, scaler):
    try:
        patient = saisir_patient()
        patient_normalise = scaler.transform(patient)
        prediction = model.predict(patient_normalise, verbose=0)
        proba = prediction[0][0] * 100
        
        print("\n Résultats de la prédiction :")
        if prediction[0][0] > 0.5:
            print(f" Ce patient est probablement **DIABÉTIQUE** ({proba:.2f}% de probabilité).")
        else:
            print(f" Ce patient est probablement **NON DIABÉTIQUE** ({100 - proba:.2f}% de probabilité).")
    except Exception as e:
        print(f" Erreur lors de la prédiction : {e}")

# Étape 6 : Programme principal
def main():
    try:
        print(" Lancement du programme de prédiction du diabète")
        
        # Charger et préparer les données
        data = load_data()
        X_train, X_test, y_train, y_test, scaler = prepare_data(data)
        
        # Créer ou charger le modèle
        model = build_or_load_model(X_train, y_train, X_test, y_test)
        
        # Boucle pour prédire
        while True:
            predire_diabete(model, scaler)
            choix = input("\n Souhaitez-vous tester un autre patient ? (o/n) : ").lower()
            if choix != 'o':
                print(" Fin du programme.")
                break
    except Exception as e:
        print(f" Erreur dans le programme : {e}")
        print("Vérifiez que diabetes.csv est présent et que les dépendances sont installées.")

if __name__ == "__main__":
    main()