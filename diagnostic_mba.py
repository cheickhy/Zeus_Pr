import os
import pandas as pd

# Le chemin vers ton dossier de données brutes basket
DOSSIER_BRUT = r"C:\Users\HP\Desktop\NAFA_données\Données\brutes\nba_data\nba_data"

chemin_schedule = os.path.join(DOSSIER_BRUT, "schedule.csv")
chemin_quarters = os.path.join(DOSSIER_BRUT, "quarters.csv")

print("--- EXÉCUTION DU DIAGNOSTIC DE COLONNES ---")

if not os.path.exists(chemin_schedule):
    print(f"Erreur : Le fichier schedule est introuvable au chemin : {chemin_schedule}")
else:
    try:
        # On charge juste 1 ligne pour voir les en-têtes
        df_schedule = pd.read_csv(chemin_schedule, nrows=1)
        print("\n Colonnes réelles trouvées dans SCHEDULE.CSV :")
        print(df_schedule.columns.tolist())
        
        if os.path.exists(chemin_quarters):
            df_quarters = pd.read_csv(chemin_quarters, nrows=1)
            print("\n Colonnes réelles trouvées dans QUARTERS.CSV :")
            print(df_quarters.columns.tolist())
            
    except Exception as e:
        print(f"Erreur lors du diagnostic : {e}")
