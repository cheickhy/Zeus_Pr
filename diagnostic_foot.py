import os
import pandas as pd

# Chemin absolu vers votre fichier de football racine
foot_file = r"C:\Users\HP\Desktop\NAFA_données\_sofascore-all-match-id-data.csv"

print("---  SCRIPT DE DIAGNOSTIC FOOTBALL ---")

if not os.path.exists(foot_file):
    print(f" Erreur : Le fichier est introuvable au chemin : {foot_file}")
else:
    try:
        # On charge seulement 2 lignes pour aller à la vitesse de l'éclair
        df = pd.read_csv(foot_file, nrows=2)
        
        liste_colonnes = df.columns.tolist()
        total_cols = len(liste_colonnes)
        
        print(f" Fichier lu avec succès !")
        print(f"Nombre total de colonnes trouvées : {total_cols}")
        print("\nVoici les colonnes disponibles dans votre fichier :")
        
        # On les affiche proprement ligne par ligne
        for i, col in enumerate(liste_colonnes):
            print(f" [{i}] -> {col}")
            
    except Exception as e:
        print(f" Une erreur est survenue lors de la lecture : {e}")
