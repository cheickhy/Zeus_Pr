
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_BRUTES, etape, sauvegarder  # noqa: E402

DOSSIER = DOSSIER_BRUTES / "selections"
PREMIERE_ANNEE = 1990        # le football d'avant 1990 apporte peu pour prédire aujourd'hui
COMPETITIONS_AFRICAINES = ["African Cup of Nations", "African Cup of Nations qualification"]


def main():
    print("=== NETTOYAGE FOOTBALL : SÉLECTIONS NATIONALES ===")
    df = pd.read_csv(DOSSIER / "results.csv")
    n0 = len(df)

    etape("Dates, scores et doublons")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])   # matchs pas encore joués
    df = df.drop_duplicates(["date", "home_team", "away_team"])
    df = df[df["date"].dt.year >= PREMIERE_ANNEE]
    print(f"{n0} lignes brutes -> {len(df)} matchs joués depuis {PREMIERE_ANNEE}")

    etape("Tirs au but : vainqueur des matchs nuls à élimination directe")
    tab = pd.read_csv(DOSSIER / "shootouts.csv", parse_dates=["date"])
    tab = tab[["date", "home_team", "away_team", "winner"]].rename(columns={"winner": "vainqueur_tab"})
    df = df.merge(tab.drop_duplicates(["date", "home_team", "away_team"]),
                  on=["date", "home_team", "away_team"], how="left")
    print(f"Matchs décidés aux tirs au but : {df['vainqueur_tab'].notna().sum()}")

    etape("Sélections africaines (ont déjà joué la CAN ou ses qualifications)")
    matchs_caf = df[df["tournament"].isin(COMPETITIONS_AFRICAINES)]
    africaines = set(matchs_caf["home_team"]) | set(matchs_caf["away_team"])
    df["home_africain"] = df["home_team"].isin(africaines).astype(int)
    df["away_africain"] = df["away_team"].isin(africaines).astype(int)
    df["can"] = (df["tournament"] == "African Cup of Nations").astype(int)
    df["amical"] = (df["tournament"] == "Friendly").astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    print(f"{len(africaines)} sélections africaines identifiées")

    etape("Cibles")
    df["total_buts"] = df["home_score"] + df["away_score"]
    df["resultat"] = (df["home_score"] > df["away_score"]).map({True: "domicile"})
    df.loc[df["home_score"] < df["away_score"], "resultat"] = "exterieur"
    df.loc[df["home_score"] == df["away_score"], "resultat"] = "nul"

    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    etape("Résumé")
    afr = df[(df["home_africain"] == 1) | (df["away_africain"] == 1)]
    print(f"Matchs impliquant une sélection africaine : {len(afr)}")
    print(f"Matchs de CAN : {df['can'].sum()} | dernier match : {df['date'].max().date()}")
    print(f"Buts par match : {df['total_buts'].mean():.2f} (tous) | "
          f"{df.loc[df['can'] == 1, 'total_buts'].mean():.2f} (CAN)")
    sauvegarder(df, "football_selections.csv")


if __name__ == "__main__":
    main()
