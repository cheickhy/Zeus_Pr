import os
import pandas as pd

# Chemin vers vos données de basket
path_nba = r"C:\Users\HP\Desktop\NAFA_données\nba_data\nba_data"

print("--- STARTING CLEANING PROCESS ---")

# 1. Charger l'intégralité des 3 fichiers
schedule = pd.read_csv(os.path.join(path_nba, "schedule.csv"))
quarters = pd.read_csv(os.path.join(path_nba, "quarters.csv"))
players = pd.read_csv(os.path.join(path_nba, "players.csv"))

# 2. Agrégation du fichier 'players' par match et par équipe (team_id)
# On calcule la performance collective des joueurs sur le match
print("Agrégation des statistiques des joueurs...")
players_resume = (
    players.groupby(["game_id", "team_id"])
    .agg(
        total_passes_match=("ast", "sum"),     # 'ast' pour assists
        total_rebonds_match=("reb", "sum"),    # 'reb' pour rebounds
        total_fautes_match=("pf", "sum")        # 'pf' pour personal fouls
    )
    .reset_index()
)

# 3. Fusion des deux premiers fichiers (Schedule + Quarters)
# On fusionne sur les clés communes : le match ET l'équipe
print("Fusion des calendriers et des quart-temps...")
df_total = pd.merge(schedule, quarters, on=["game_id", "team_id", "season"], how="inner")

# 4. Fusion finale avec le résumé des joueurs
print("Intégration des statistiques des joueurs au dataset principal...")
df_basket_complet = pd.merge(df_total, players_resume, on=["game_id", "team_id"], how="left")

# 5. CALCUL DES VARIABLES CIBLES POUR VOS PARIS (Over/Under)
print("Calcul des totaux cibles par équipe...")
# Total 1er quart-temps de l'équipe
df_basket_complet["equipe_total_q1"] = pd.to_numeric(df_basket_complet["pts_q1"], errors="coerce")

# Total 1ère mi-temps de l'équipe (Q1 + Q2)
df_basket_complet["equipe_total_mi_temps"] = (
    pd.to_numeric(df_basket_complet["pts_q1"], errors="coerce") + 
    pd.to_numeric(df_basket_complet["pts_q2"], errors="coerce")
)

# Total du match de l'équipe
df_basket_complet["equipe_total_match"] = pd.to_numeric(df_basket_complet["pts_total"], errors="coerce")

# Nettoyage des lignes sans score
df_basket_complet = df_basket_complet.dropna(subset=["equipe_total_match"])

# 6. Sauvegarde du fichier propre
output_path = r"C:\Users\HP\Desktop\NAFA_données\basketball_clean.csv"
df_basket_complet.to_csv(output_path, index=False)

print(f"\n Analyse réussie !")
print(f"Structure finale du fichier Basket : {df_basket_complet.shape}")
print(f"Fichier sauvegardé ici : {output_path}")
