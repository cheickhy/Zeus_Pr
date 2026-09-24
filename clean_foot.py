import os
import pandas as pd

# Chemin absolu vers votre fichier de football racine
foot_file = r"C:\Users\HP\Desktop\NAFA_données\_sofascore-all-match-id-data.csv"

print("---  DÉMARRAGE DU NETTOYAGE FOOTBALL ---")

if not os.path.exists(foot_file):
    print(f" Erreur : Le fichier est introuvable au chemin : {foot_file}")
else:
    try:
        # 1. Chargement de l'intégralité du dataset
        df = pd.read_csv(foot_file)
        print(f"Fichier chargé. Nombre de matchs bruts : {len(df)}")

        # 2. Conversion forcée des colonnes clés en numérique (sécurité anti-texte)
        cols_cles = [
            'home_score', 'away_score',
            'Corner_kicks_home', 'Corner_kicks_away',
            'Yellow_cards_home', 'Yellow_cards_away'
        ]
        
        for col in cols_cles:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Suppression des lignes où les scores ou stats indispensables sont absents
        df = df.dropna(subset=['home_score', 'away_score'])

        # 4. CALCUL DES VARIABLES CIBLES (Vos marchés de paris Over/Under)
        print("Calcul des totaux cibles (Buts, Corners, Cartons)...")
        
        # Total de buts (Match)
        df['total_buts_match'] = df['home_score'] + df['away_score']
        
        # Total de corners (Match)
        df['total_corners_match'] = df['Corner_kicks_home'].fillna(0) + df['Corner_kicks_away'].fillna(0)
        
        # Total de cartons jaunes (Match)
        df['total_cartons_jaunes_match'] = df['Yellow_cards_home'].fillna(0) + df['Yellow_cards_away'].fillna(0)

        # 5. NETTOYAGE ET REMPLISSAGE DES COMPOSANTES DE JEU (Features)
        # S'il y a des valeurs vides dans les statistiques avancées, on met 0 (ex: pas de tirs sur le poteau = 0)
        df = df.fillna(0)

        # 6. SAUVEGARDE DU DATASET PROPRE
        output_file = r"C:\Users\HP\Desktop\NAFA_données\football_clean.csv"
        df.to_csv(output_file, index=False)

        print(f"\n Analyse et uniformisation Réussies !")
        print(f"Structure du fichier final Football : {df.shape}")
        print(f"Fichier sauvegardé ici : {output_file}")

    except Exception as e:
        print(f" Une erreur est survenue lors du traitement : {e}")
