import pandas as pd
import os

BASE_PATH = r"C:\Users\HP\Desktop\NAFA_données"
file_path = os.path.join(r"C:\Users\HP\Desktop\NAFA_données", "football_clean.csv")

df = pd.read_csv(r"C:\Users\HP\Desktop\NAFA_données\football_clean.csv")

print("---  VÉRIFICATION FINALE DU DATASET ---")
print(f"Dimensions actuelles : {df.shape} (lignes, colonnes)")

# 1. Détecter les colonnes 100% vides
colonnes_vides = [col for col in df.columns if df[col].isnull().all()]
print(f"\n• Colonnes 100% vides détectées : {len(colonnes_vides)}")
if colonnes_vides:
    print(f"  --> Elles vont être supprimées : {colonnes_vides}")
    df = df.drop(columns=colonnes_vides)

# 2. Compter le reste des valeurs manquantes par colonne
manquants_par_colonne = df.isnull().sum()
colonnes_avec_nan = manquants_par_colonne[manquants_par_colonne > 0]

print(f"\n• Colonnes contenant encore des valeurs manquantes (NaN) : {len(colonnes_avec_nan)}")
if not colonnes_avec_nan.empty:
    print(colonnes_avec_nan)
    # Sécurité ultime : on remplace les NaN restants par 0 pour les chiffres, ou "Inconnu" pour le texte
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Inconnu")
        else:
            df[col] = df[col].fillna(0)
else:
    print(" Aucune valeur manquante restante")

# 3. Sauvegarde finale si des modifications ont eu lieu
df.to_csv(r"C:\Users\HP\Desktop\NAFA_données\football_clean.csv", index=False)
print("\n Dataset validé et paré pour l'entraînement du modèle ! ")
