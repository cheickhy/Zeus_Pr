import os
import pandas as pd

path_mlb = r"C:\Users\HP\Desktop\NAFA_données\mlb_data\mlb_data"
print("---  DÉMARRAGE DU NETTOYAGE BASEBALL (MLB) ---")

# 1. Chargement des données
schedule = pd.read_csv(os.path.join(path_mlb, "schedule.csv"))
linescore = pd.read_csv(os.path.join(path_mlb, "linescore.csv"))
pitching = pd.read_csv(os.path.join(path_mlb, "pitching.csv"))
batting = pd.read_csv(os.path.join(path_mlb, "batting.csv"))

# 2. Agrégation de linescore.csv (Calcul des totaux par manche)
print("Traitement des scores par manche (Innings)...")
# On s'assure que les scores des manches sont numériques
linescore["home_runs"] = pd.to_numeric(linescore["home_runs"], errors="coerce")
linescore["away_runs"] = pd.to_numeric(linescore["away_runs"], errors="coerce")

# Extraction des points de la 1ère manche
m1 = linescore[linescore["inning"] == 1][["gamePk", "home_runs", "away_runs"]].rename(
    columns={"home_runs": "home_m1", "away_runs": "away_m1"}
)

# Extraction des points cumulés des manches 1, 2 et 3
m1_3 = linescore[linescore["inning"].isin([1, 2, 3])].groupby("gamePk").agg(
    home_m1_3=("home_runs", "sum"),
    away_m1_3=("away_runs", "sum")
).reset_index()

# 3. Agrégation de pitching.csv (Statistiques des lanceurs par camp)
print("Agrégation des statistiques des lanceurs (Pitching)...")
pitching_resume = pitching.groupby(["gamePk", "side"]).agg(
    pitcher_moy_era=("era", "mean"),
    pitcher_total_so=("so", "sum"),        # Strikeouts cumulés
    pitcher_total_whip=("whip", "mean")    # WHIP moyen
).unstack(fill_value=0) # Aligne les stats home/away sur une seule ligne

# Nettoyage des noms de colonnes après le unstack
pitching_resume.columns = [f"{col[0]}_{col[1]}" for col in pitching_resume.columns]
pitching_resume = pitching_resume.reset_index()

# 4. Agrégation de batting.csv (Puissance offensive par camp)
print("Agrégation de la puissance offensive (Batting)...")
batting_resume = batting.groupby(["gamePk", "side"]).agg(
    team_moy_ops=("ops", "mean"),          # On-base plus slugging
    team_moy_avg=("avg", "mean")           # Moyenne au bâton
).unstack(fill_value=0)

batting_resume.columns = [f"{col[0]}_{col[1]}" for col in batting_resume.columns]
batting_resume = batting_resume.reset_index()

# 5. FUSION FINALE DE TOUS LES BLOCS
print("Fusion de toutes les composantes MLB...")
df_mlb = pd.merge(schedule, m1, on="gamePk", how="inner")
df_mlb = pd.merge(df_mlb, m1_3, on="gamePk", how="inner")
df_mlb = pd.merge(df_mlb, pitching_resume, on="gamePk", how="left")
df_mlb = pd.merge(df_mlb, batting_resume, on="gamePk", how="left")

# 6. CALCUL DES SÉRIES DE TOTAUX (Over/Under Cibles)
print("Création des variables cibles pour l'application de pari...")
# Total 1ère manche
df_mlb["total_manche_1"] = df_mlb["home_m1"] + df_mlb["away_m1"]

# Total des 3 premières manches
df_mlb["total_3_prem_manches"] = df_mlb["home_m1_3"] + df_mlb["away_m1_3"]

# Total de tout le match
df_mlb["total_match"] = pd.to_numeric(df_mlb["home_score"], errors="coerce") + pd.to_numeric(df_mlb["away_score"], errors="coerce")

# Suppression des lignes sans score final
df_mlb = df_mlb.dropna(subset=["total_match"])

# 7. SAUVEGARDE
output_file = r"C:\Users\HP\Desktop\NAFA_données\baseball_clean.csv"
df_mlb.to_csv(output_file, index=False)

print(f"\n Analyse MLB réussie !")
print(f"Structure du fichier final Baseball : {df_mlb.shape}")
print(f"Fichier sauvegardé ici : {output_file}")
