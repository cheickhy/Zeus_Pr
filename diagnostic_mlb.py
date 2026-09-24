import os
import pandas as pd

# Configuration du chemin vers le dossier du baseball (avec la correction de la liste ligne 14)
DOSSIER_BASEBALL = r"C:\Users\HP\Desktop\NAFA_données\Données\brutes\mlb_data\mlb_data"

print("--- DIAGNOSTIC AUTOMATIQUE BASEBALL ---")

if not os.path.exists(DOSSIER_BASEBALL):
    print(f"Erreur : Le dossier de base est introuvable : {DOSSIER_BASEBALL}")
    print("Verifie bien l'orthographe de tes dossiers (Majuscules/Minuscules).")
else:
    print("Fouille du dossier en cours...")
    
    # CORRECTION : Initialisation correcte de la liste vide
    fichiers_trouves = []
    
    # Exploration automatique de tous les sous-dossiers
    for racine, dossiers, fichiers in os.walk(DOSSIER_BASEBALL):
        for fichier in fichiers:
            if fichier.endswith('.csv'):
                chemin_complet = os.path.join(racine, fichier)
                fichiers_trouves.append((fichier, chemin_complet))

    if not fichiers_trouves:
        print("Aucun fichier CSV n'a ete trouve dans le dossier mlb_data ou ses sous-dossiers.")
        print("Verifie que les fichiers ne sont pas restes dans l'archive zip d'origine.")
    else:
        print(f"Succes : {len(fichiers_trouves)} fichiers CSV detectes !")
        print("\n--- ANALYSE DES COLONNES DES FICHIERS ---")
        
        # Lecture de la premiere ligne de chaque fichier pour lister les en-tetes
        for nom_fichier, chemin in fichiers_trouves:
            try:
                df_temp = pd.read_csv(chemin, nrows=1)
                print(f"\nFichier trouve : {nom_fichier}")
                print(f"Chemin reel : {chemin}")
                print(f"Colonnes : {df_temp.columns.tolist()}")
            except Exception as e:
                print(f"Impossible de lire le fichier {nom_fichier} : {e}")
