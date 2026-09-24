


import os
import re
import pandas as pd
import numpy as np

FICHIER_BRUT = r"C:\Users\HP\Desktop\NAFA_données\données\brutes\_sofascore-all-match-id-data.csv"
DOSSIER_SORTIE = r"C:\Users\HP\Desktop\NAFA_données\données\traitées"
FICHIER_PROPRE = os.path.join(DOSSIER_SORTIE, "football_propre.csv")
RAPPORT_QUALITE = os.path.join(DOSSIER_SORTIE, "rapport_qualite.csv")

# Colonnes au format "X/Y (Z%)" -> on en tire 3 colonnes : _reussi, _tentes, _pct
COLONNES_RATIO = [
    "Final_third_phase", "Long_balls", "Crosses",
    "Ground_duels", "Aerial_duels", "Dribbles",
]

# Colonnes au format "Z%" seul -> une colonne _pct
COLONNES_POURCENTAGE = [
    "Ball_possession", "Duels", "Tackles_won",
]

CIBLES = ["total_buts_match", "total_corners_match", "total_cartons_jaunes_match"]
IDENTIFIANTS = ["match_id", "home_team", "away_team", "home_score", "away_score"]


def parse_ratio(valeur):
    """'34/47 (72%)' -> (34.0, 47.0, 72.0) ; NaN si vide ou format inattendu."""
    if pd.isna(valeur):
        return (np.nan, np.nan, np.nan)
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*\((\d+)%\)\s*", str(valeur))
    if not m:
        return (np.nan, np.nan, np.nan)
    reussi, tentes, pct = m.groups()
    return (float(reussi), float(tentes), float(pct))


def parse_pourcentage(valeur):
    """'58%' -> 58.0 ; NaN si vide ou format inattendu."""
    if pd.isna(valeur):
        return np.nan
    m = re.match(r"\s*(\d+)%\s*", str(valeur))
    return float(m.group(1)) if m else np.nan


def nettoyer_colonnes_texte(df):
    nouvelles_colonnes = {}
    colonnes_ratio_presentes = []

    # Colonnes "X/Y (Z%)" -> 3 nouvelles colonnes numériques chacune
    for base in COLONNES_RATIO:
        for suffixe in ["_home", "_away"]:
            col = base + suffixe
            if col not in df.columns:
                continue
            parsed = df[col].apply(parse_ratio)
            nouvelles_colonnes[f"{col}_reussi"] = parsed.apply(lambda t: t[0])
            nouvelles_colonnes[f"{col}_tentes"] = parsed.apply(lambda t: t[1])
            nouvelles_colonnes[f"{col}_pct"] = parsed.apply(lambda t: t[2])
            colonnes_ratio_presentes.append(col)

    df = df.drop(columns=colonnes_ratio_presentes)
    df = pd.concat([df, pd.DataFrame(nouvelles_colonnes, index=df.index)], axis=1)

    # Colonnes "Z%" seules -> même nom, converties en place (pas de fragmentation,
    # une seule colonne réécrite, pas d'ajout)
    for base in COLONNES_POURCENTAGE:
        for suffixe in ["_home", "_away"]:
            col = base + suffixe
            if col not in df.columns:
                continue
            df[col] = df[col].apply(parse_pourcentage)

    return df


def calculer_cibles(df):
    # home_score / away_score déjà numériques dans le fichier source SofaScore
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["total_buts_match"] = df["home_score"] + df["away_score"]

    for col in ["Corner_kicks_home", "Corner_kicks_away",
                "Yellow_cards_home", "Yellow_cards_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # IMPORTANT : on garde NaN si un des deux camps manque -> pas de faux 0
    df["total_corners_match"] = np.where(
        df["Corner_kicks_home"].isna() | df["Corner_kicks_away"].isna(),
        np.nan,
        df["Corner_kicks_home"] + df["Corner_kicks_away"],
    )
    df["total_cartons_jaunes_match"] = np.where(
        df["Yellow_cards_home"].isna() | df["Yellow_cards_away"].isna(),
        np.nan,
        df["Yellow_cards_home"] + df["Yellow_cards_away"],
    )
    return df


def rapport_qualite(df):
    taux_completude = (1 - df.isna().mean()) * 100
    rapport = taux_completude.sort_values().rename("pct_complet").to_frame()
    rapport.index.name = "colonne"
    return rapport.reset_index()


def main():
    print("DÉMARRAGE DU NETTOYAGE")

    if not os.path.exists(FICHIER_BRUT):
        print(f"Erreur : fichier brut introuvable : {FICHIER_BRUT}")
        return

    df = pd.read_csv(FICHIER_BRUT)
    print(f"Fichier chargé. Matchs bruts : {len(df)}")

    df = calculer_cibles(df)
    print(f"Matchs conservés après filtrage score manquant : {len(df)}")

    df = nettoyer_colonnes_texte(df)
    print("Colonnes texte (ratios/pourcentages) converties en numérique.")

    # Cartons rouges : 0 est un vrai défaut logique (pas de carton = 0 carton, pas "inconnu")
    for col in ["Red_cards_home", "Red_cards_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    df.to_csv(FICHIER_PROPRE, index=False)
    print(f"Fichier propre sauvegardé : {FICHIER_PROPRE}")

    rapport = rapport_qualite(df)
    rapport.to_csv(RAPPORT_QUALITE, index=False)
    print(f"Rapport de qualité sauvegardé : {RAPPORT_QUALITE}")
    print("\nColonnes les moins complètes :")
    print(rapport.head(10).to_string(index=False))

    print("\n⚠ Rappel : aucune colonne date dans les données. Les colonnes "
          "'in-match' (possession, tirs, corners du match, etc.) ne doivent "
          "PAS servir de features pour prédire CE match — seulement comme "
          "matière première pour des moyennes glissantes 'avant-match' une "
          "fois l'historique temporel reconstitué.")


if __name__ == "__main__":
    main()