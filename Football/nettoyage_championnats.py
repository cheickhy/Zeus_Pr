
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_BRUTES, etape, sauvegarder  # noqa: E402
from mise_a_jour import CHAMPIONNATS  # noqa: E402

DOSSIER = DOSSIER_BRUTES / "football_data"

# Colonnes du site -> noms du projet
COLONNES = {
    "Date": "date", "Time": "heure", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_buts", "FTAG": "away_buts", "HTHG": "home_buts_mt", "HTAG": "away_buts_mt",
    "HS": "home_tirs", "AS": "away_tirs", "HST": "home_tirs_cadres", "AST": "away_tirs_cadres",
    "HF": "home_fautes", "AF": "away_fautes", "HC": "home_corners", "AC": "away_corners",
    "HY": "home_jaunes", "AY": "away_jaunes", "HR": "home_rouges", "AR": "away_rouges",
    "Referee": "arbitre",
}
# Anciens fichiers : noms de colonnes différents
ALIAS = {"HT": "HomeTeam", "AT": "AwayTeam", "HG": "FTHG", "AG": "FTAG"}

# Valeur maximale plausible pour UNE équipe sur un match (au-delà : erreur de saisie)
MAXIMUM = {"buts": 15, "buts_mt": 10, "tirs": 50, "tirs_cadres": 30, "fautes": 45,
           "corners": 25, "jaunes": 10, "rouges": 5}

# Cible -> statistique additionnée pour les deux équipes
CIBLES = {"total_buts": "buts", "total_corners": "corners", "total_fautes": "fautes",
          "total_cartons_jaunes": "jaunes"}


def lire_fichier(chemin):
    code, saison = chemin.stem.rsplit("_", 1)
    for encodage in ["utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(chemin, encoding=encodage, on_bad_lines="skip", low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df = df.rename(columns=ALIAS)
    df = df[[c for c in COLONNES if c in df.columns]].rename(columns=COLONNES)
    df["code"] = code
    df["championnat"] = CHAMPIONNATS.get(code, code)
    df["saison"] = f"20{saison[:2]}-{saison[2:]}"
    return df


def main():
    print("=== NETTOYAGE FOOTBALL : CHAMPIONNATS EUROPÉENS ===")
    fichiers = sorted(DOSSIER.glob("*.csv"))
    df = pd.concat([lire_fichier(f) for f in fichiers], ignore_index=True)
    n0 = len(df)
    print(f"{len(fichiers)} fichiers lus, {n0} lignes")

    etape("Lignes vides, dates et doublons")
    df = df.dropna(subset=["home_team", "away_team", "date"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(["code", "date", "home_team", "away_team"])
    for col in ["home_team", "away_team", "arbitre"]:
        df[col] = df[col].astype("string").str.strip()

    etape("Scores : matchs sans score (reportés, pas encore joués) retirés")
    for camp in ["home", "away"]:
        df[f"{camp}_buts"] = pd.to_numeric(df[f"{camp}_buts"], errors="coerce")
    df = df.dropna(subset=["home_buts", "away_buts"])

    etape("Statistiques : conversion en nombres et valeurs impossibles mises à vide")
    aberrants = 0
    for stat, maxi in MAXIMUM.items():
        for camp in ["home", "away"]:
            col = f"{camp}_{stat}"
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = pd.to_numeric(df[col], errors="coerce")
            faux = (df[col] < 0) | (df[col] > maxi)
            aberrants += int(faux.sum())
            df.loc[faux, col] = pd.NA
    # Plus de tirs cadrés que de tirs : l'une des deux valeurs est fausse, on vide les deux
    for camp in ["home", "away"]:
        faux = df[f"{camp}_tirs_cadres"] > df[f"{camp}_tirs"]
        aberrants += 2 * int(faux.sum())
        df.loc[faux, [f"{camp}_tirs", f"{camp}_tirs_cadres"]] = pd.NA
    print(f"Valeurs impossibles mises à vide : {aberrants}")

    etape("Cibles des paris (vides si une équipe n'a pas la statistique)")
    for cible, stat in CIBLES.items():
        df[cible] = df[f"home_{stat}"] + df[f"away_{stat}"]   # vide si un camp manque
    df["total_buts_mt"] = df["home_buts_mt"] + df["away_buts_mt"]

    colonnes = (["code", "championnat", "saison", "date", "heure", "home_team", "away_team", "arbitre"]
                + [f"{c}_{s}" for s in MAXIMUM for c in ["home", "away"]]
                + list(CIBLES) + ["total_buts_mt"])
    df = df[colonnes].sort_values(["date", "code", "home_team"]).reset_index(drop=True)
    print(f"{n0} lignes brutes -> {len(df)} matchs")

    etape("Disponibilité des statistiques par championnat (% de matchs)")
    dispo = df.groupby("championnat")[list(CIBLES)].apply(lambda g: g.notna().mean() * 100)
    dispo.insert(0, "matchs", df.groupby("championnat").size())
    print(dispo.round(0).astype(int).to_string())

    sauvegarder(df, "football_championnats.csv")


if __name__ == "__main__":
    main()
