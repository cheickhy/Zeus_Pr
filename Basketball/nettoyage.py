"""Nettoyage des données NBA : une ligne par match avec les deux équipes.

Corrections par rapport à l'ancien script :
- le calendrier brut contient chaque match 3 à 5 fois : dédoublonné ;
- le calendrier ne liste qu'une équipe par match : on part du fichier des
  quart-temps, qui contient les deux, et on place domicile et extérieur sur une ligne ;
- matchs sans score ou annulés (0 point) retirés ;
- points des prolongations au-delà de la 2e récupérés à partir du total.

ATTENTION : les statistiques des joueurs sont celles DU match.
Elles ne doivent servir qu'à calculer des moyennes sur les matchs PRÉCÉDENTS.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import BRUT_NBA, sauvegarder, etape  # noqa: E402

QT = ["pts_q1", "pts_q2", "pts_q3", "pts_q4", "pts_ot1", "pts_ot2"]
STATS_JOUEURS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
                 "reb", "ast", "stl", "blk", "to", "pf"]


DOSSIER_MAJ = BRUT_NBA.parent / "mise_a_jour"   # rempli par Basketball/mise_a_jour.py


def lire_brut(nom, **options):
    """Lit un fichier brut d'origine et y ajoute les matchs téléchargés par la mise à jour.

    Un match présent dans la mise à jour remplace toutes ses lignes d'origine.
    """
    brut = pd.read_csv(BRUT_NBA / f"{nom}.csv", **options)
    chemin_maj = DOSSIER_MAJ / f"{nom}.csv"
    if not chemin_maj.exists():
        return brut
    maj = pd.read_csv(chemin_maj)
    return pd.concat([brut[~brut["game_id"].isin(maj["game_id"])], maj], ignore_index=True)


def charger_calendrier():
    etape("Calendrier : suppression des doublons, identification de l'équipe à domicile")
    sch = lire_brut("schedule").drop_duplicates()
    assert sch["game_id"].is_unique, "Un match a plusieurs lignes différentes"
    sch["date"] = pd.to_datetime(sch["date"])
    # "BOS vs. CLE" = BOS reçoit ; "BOS @ CLE" = BOS se déplace
    a_domicile = sch["matchup"].str.contains(" vs. ", regex=False)
    print(f"{len(sch)} matchs uniques ; lignes « @ » (équipe à l'extérieur) : {(~a_domicile).sum()}")
    sch["team_id_ligne_domicile"] = a_domicile
    return sch[["game_id", "season", "date", "team_id", "team_id_ligne_domicile"]]


def charger_quart_temps():
    etape("Quart-temps : doublons, matchs vides ou annulés")
    q = lire_brut("quarters")
    n0 = q["game_id"].nunique()
    q = q.dropna(subset=["team_id"]).drop_duplicates(["game_id", "team_id"])
    q["team_id"] = q["team_id"].astype("int64")
    q = q.dropna(subset=["pts_q1", "pts_q2", "pts_q3", "pts_q4", "pts_total"])
    q = q[q["pts_total"] > 0]
    # Prolongations au-delà de la 2e : non détaillées, on les déduit du total
    q["pts_ot_suppl"] = q["pts_total"] - q[QT].fillna(0).sum(axis=1)
    incoherents = q.loc[q["pts_ot_suppl"] < 0, "game_id"]
    q = q[~q["game_id"].isin(incoherents)]
    # On ne garde que les matchs où les deux équipes sont présentes
    deux = q.groupby("game_id")["team_id"].transform("size") == 2
    q = q[deux]
    print(f"{n0} matchs bruts -> {q['game_id'].nunique()} matchs complets "
          f"(dont {(q['pts_ot_suppl'] > 0).sum() // 2} avec 3 prolongations ou plus)")
    return q


def resumer_joueurs():
    etape("Joueurs : totaux par équipe et par match")
    p = pd.read_csv(BRUT_NBA / "players.csv", usecols=["game_id", "team_id", "player_id"] + STATS_JOUEURS)
    p = p.drop_duplicates(["game_id", "team_id", "player_id"])
    # Joueurs sans statistiques = n'ont pas joué : ignorés par la somme (min_count=1)
    equipes = p.groupby(["game_id", "team_id"])[STATS_JOUEURS].sum(min_count=1).reset_index()
    # Matchs de la mise à jour : statistiques d'équipe déjà additionnées par la NBA
    chemin_maj = DOSSIER_MAJ / "equipes.csv"
    if chemin_maj.exists():
        maj = pd.read_csv(chemin_maj)
        equipes = pd.concat([equipes[~equipes["game_id"].isin(maj["game_id"])],
                             maj[["game_id", "team_id"] + STATS_JOUEURS]], ignore_index=True)
    return equipes


def main():
    print("=== NETTOYAGE BASKETBALL (NBA) ===")
    sch = charger_calendrier()
    q = charger_quart_temps().merge(resumer_joueurs(), on=["game_id", "team_id"], how="left")

    etape("Assemblage : une ligne par match, domicile / extérieur")
    q = q.merge(sch, on="game_id", how="inner", suffixes=("", "_sch"))
    ligne_calendrier = q["team_id"] == q["team_id_sch"]
    q["domicile"] = ligne_calendrier == q["team_id_ligne_domicile"]
    q = q.drop(columns=["team_id_sch", "team_id_ligne_domicile", "season_sch"], errors="ignore")
    ok = q.groupby("game_id")["domicile"].transform("sum") == 1
    print(f"Matchs où le domicile n'a pas pu être déterminé : {q.loc[~ok, 'game_id'].nunique()}")
    q = q[ok]

    cles = ["game_id", "season", "date"]
    stats = [c for c in q.columns if c not in cles + ["domicile"]]
    home = q[q["domicile"]].set_index(cles)[stats].add_prefix("home_")
    away = q[~q["domicile"]].set_index(cles)[stats].add_prefix("away_")
    df = home.join(away, how="inner").reset_index()

    etape("Cibles des paris Over/Under (total des deux équipes)")
    df["total_q1"] = df["home_pts_q1"] + df["away_pts_q1"]
    df["total_mi_temps"] = (df["home_pts_q1"] + df["home_pts_q2"]
                            + df["away_pts_q1"] + df["away_pts_q2"])
    df["total_match"] = df["home_pts_total"] + df["away_pts_total"]
    df["prolongation"] = (df["home_pts_q1"] + df["home_pts_q2"] + df["home_pts_q3"] + df["home_pts_q4"]
                          == df["away_pts_q1"] + df["away_pts_q2"] + df["away_pts_q3"] + df["away_pts_q4"])

    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    assert df["game_id"].is_unique, "Des matchs sont encore en double"
    sauvegarder(df, "basketball_matchs.csv")


if __name__ == "__main__":
    main()
