"""Retrouve la date des matchs SofaScore en les reliant aux matchs de football-data.co.uk."""
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, etape  # noqa: E402

FICHIER_SOFASCORE = DOSSIER_TRAITEES / "sofascore" / "sofascore_matchs.csv"
SORTIE = DOSSIER_TRAITEES / "sofascore" / "sofascore_dates.csv"
SIGNATURE = ["hg", "ag", "hc", "ac", "hy", "ay"]   # buts, corners et cartons jaunes de chaque équipe
ECART_SUSPECT_JOURS = 180


def mots(nom):
    nom = unicodedata.normalize("NFKD", str(nom)).encode("ascii", "ignore").decode().lower()
    nom = re.sub(r"\b(fc|cf|afc|sc|ac|ssc|as|ud|cd|rc|sv|vfb|vfl|tsg|1\.|fk|sk|club|calcio|de|of)\b", " ", nom)
    return re.sub(r"[^a-z0-9 ]", " ", nom).split()


def ressemblance(a, b):
    a, b = mots(a), mots(b)
    if not a or not b:
        return 0
    commun = len(set(a) & set(b)) / min(len(set(a)), len(set(b)))
    return max(commun, SequenceMatcher(None, " ".join(a), " ".join(b)).ratio())


def main():
    print("=== DATES DES MATCHS SOFASCORE ===")
    noms = {"Corner_kicks_home": "hc", "Corner_kicks_away": "ac", "Yellow_cards_home": "hy",
            "Yellow_cards_away": "ay", "home_score": "hg", "away_score": "ag"}
    s = pd.read_csv(FICHIER_SOFASCORE, low_memory=False).rename(columns=noms)
    f = pd.read_csv(DOSSIER_TRAITEES / "football_championnats.csv", parse_dates=["date"], low_memory=False).rename(
        columns={"home_corners": "hc", "away_corners": "ac", "home_jaunes": "hy", "away_jaunes": "ay",
                 "home_buts": "hg", "away_buts": "ag"})
    n_total = len(s)
    s, fs = s.dropna(subset=SIGNATURE).copy(), f.dropna(subset=SIGNATURE).copy()
    for d in (s, fs):
        d[SIGNATURE] = d[SIGNATURE].astype(int)

    etape("1. Même signature de match et noms d'équipes ressemblants")
    c = s[["match_id", "home_team", "away_team"] + SIGNATURE].merge(
        fs[["home_team", "away_team"] + SIGNATURE], on=SIGNATURE, suffixes=("_s", "_f"))
    c["sim"] = [min(ressemblance(a, b), ressemblance(x, y)) for a, b, x, y in
                zip(c.home_team_s, c.home_team_f, c.away_team_s, c.away_team_f)]
    surs = c[c["sim"] >= 0.6].sort_values("sim", ascending=False).drop_duplicates("match_id")
    print(f"{len(surs)} matchs reliés")

    etape("2. Correspondance des noms d'équipes (vote majoritaire, au moins 3 matchs)")
    votes = pd.concat([surs[["home_team_s", "home_team_f"]].set_axis(["s", "f"], axis=1),
                       surs[["away_team_s", "away_team_f"]].set_axis(["s", "f"], axis=1)])
    compte = votes.value_counts().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("s")
    correspondance = dict(zip(compte.loc[compte["n"] >= 3, "s"], compte.loc[compte["n"] >= 3, "f"]))
    print(f"{len(correspondance)} équipes associées")

    etape("3. Tous les matchs : mêmes équipes et même score (les corners départagent)")
    s3 = s.assign(home_team=s["home_team"].map(correspondance), away_team=s["away_team"].map(correspondance))
    m = s3.dropna(subset=["home_team", "away_team"])[["match_id", "home_team", "away_team", "hg", "ag", "hc", "ac"]].merge(
        f[["date", "code", "home_team", "away_team", "hg", "ag", "hc", "ac"]],
        on=["home_team", "away_team", "hg", "ag"], suffixes=("", "_f"))
    m["ecart_corners"] = (m["hc"] - m["hc_f"]).abs().fillna(99) + (m["ac"] - m["ac_f"]).abs().fillna(99)
    m = m.sort_values("ecart_corners")
    nets = m.groupby("match_id").filter(lambda g: len(g) == 1 or g["ecart_corners"].iloc[0] < g["ecart_corners"].iloc[1])
    dates = nets.drop_duplicates("match_id").sort_values("match_id").reset_index(drop=True)

    etape("4. Contrôle : les identifiants SofaScore augmentent avec le temps")
    # Une date très éloignée de celles des identifiants voisins vient sans doute d'un autre match
    # entre les mêmes équipes, avec le même score, une autre saison : on l'écarte.
    jours = (dates["date"] - pd.Timestamp("2000-01-01")).dt.days
    voisins = jours.rolling(51, center=True, min_periods=10).median()
    suspects = (jours - voisins).abs() > ECART_SUSPECT_JOURS
    dates = dates[~suspects]
    print(f"Dates suspectes écartées : {int(suspects.sum())}")

    dates[["match_id", "date", "code"]].to_csv(SORTIE, index=False)
    print(f"\n{len(dates)} matchs datés sur {n_total} -> {SORTIE}")


if __name__ == "__main__":
    main()
