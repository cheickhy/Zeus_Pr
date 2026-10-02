"""Nettoyage des données MLB : une ligne par match, prête pour le calcul des features.


"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import BRUT_MLB, sauvegarder, etape  # noqa: E402

COLS_LANCEURS = ["outs", "hits", "runs", "earned_runs", "bb", "so", "hr",
                 "pitch_count", "batters_faced"]
COLS_FRAPPEURS = ["ab", "hits", "doubles", "triples", "hr", "bb", "so", "hbp",
                  "sac_fly", "total_bases", "left_on_base"]


DOSSIER_MAJ = BRUT_MLB.parent / "mise_a_jour"   # rempli par Baseball/mise_a_jour.py


def lire_brut(nom, **options):
    """Lit un fichier brut d'origine et y ajoute les matchs téléchargés par la mise à jour.

    Un match présent dans la mise à jour remplace sa ligne d'origine (par exemple
    un match encore « Scheduled » dans les données brutes, joué depuis).
    """
    brut = pd.read_csv(BRUT_MLB / f"{nom}.csv", **options)
    chemin_maj = DOSSIER_MAJ / f"{nom}.csv"
    if not chemin_maj.exists():
        return brut
    maj = pd.read_csv(chemin_maj, **options)
    return pd.concat([brut[~brut["gamePk"].isin(maj["gamePk"])], maj], ignore_index=True)


def manches_en_retraits(ip):
    """5.2 manches lancées = 5 manches et 2/3 = 17 retraits (et non 5,2)."""
    entier = ip.fillna(0).astype(int)
    tiers = ((ip.fillna(0) - entier) * 10).round().astype(int)
    return (entier * 3 + tiers).where(ip.notna())


def charger_calendrier():
    etape("Calendrier : matchs terminés, un seul exemplaire par match")
    sch = lire_brut("schedule")
    n0 = len(sch)
    sch = sch[sch["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    sch["date"] = pd.to_datetime(sch["date"])
    # Match suspendu = même gamePk à deux dates (interrompu puis repris plus tard).
    # On les retire : leur fin se joue parfois des semaines après le début, ce qui
    # fausserait les moyennes sur les matchs précédents. Ils représentent ~0,1 % des matchs.
    suspendus = sch["gamePk"].duplicated(keep=False)
    print(f"Matchs suspendus retirés : {sch.loc[suspendus, 'gamePk'].nunique()}")
    sch = sch[~suspendus]
    print(f"{n0} lignes brutes -> {len(sch)} matchs terminés uniques")
    return sch


def resumer_manches():
    etape("Scores par manche : cibles 1ère manche et 3 premières manches")
    ls = lire_brut("linescore")
    ls = ls.drop_duplicates(["gamePk", "inning"])
    g = ls.groupby("gamePk")
    res = pd.DataFrame({
        "home_m1": ls[ls["inning"] == 1].set_index("gamePk")["home_runs"],
        "away_m1": ls[ls["inning"] == 1].set_index("gamePk")["away_runs"],
        "home_m1_3": ls[ls["inning"] <= 3].groupby("gamePk")["home_runs"].sum(),
        "away_m1_3": ls[ls["inning"] <= 3].groupby("gamePk")["away_runs"].sum(),
        "home_runs_total_manches": g["home_runs"].sum(),
        "away_runs_total_manches": g["away_runs"].sum(),
        "home_hits": g["home_hits"].sum(),
        "away_hits": g["away_hits"].sum(),
        "home_errors": g["home_errors"].sum(),
        "away_errors": g["away_errors"].sum(),
        "nb_manches": g["inning"].max(),
    })
    # Un match doit avoir ses 3 premières manches pour les cibles
    nb_m1_3 = ls[ls["inning"] <= 3].groupby("gamePk").size()
    res = res[res.index.isin(nb_m1_3[nb_m1_3 == 3].index)]
    return res.reset_index()


def a_plat(df, prefixe_col=""):
    """Passe de (gamePk, side) en lignes à une ligne par match : home_xxx / away_xxx."""
    large = df.unstack("side")
    large.columns = [f"{side}_{prefixe_col}{col}" for col, side in large.columns]
    return large.reset_index()


def resumer_lanceurs():
    etape("Lanceurs : totaux par équipe et lanceur principal")
    p = lire_brut("pitching")
    # Même ligne en double pour les matchs suspendus : on retire les doublons (hors date)
    p = p.drop_duplicates([c for c in p.columns if c not in ("date", "season")])
    p["outs"] = manches_en_retraits(p["innings_pitched"])

    equipe = p.groupby(["gamePk", "side"])[COLS_LANCEURS].sum(min_count=1)
    # Les fichiers bruts d'origine n'indiquent pas le partant et ne respectent pas l'ordre
    # d'entrée : on prend le lanceur ayant affronté le plus de frappeurs (c'est le partant
    # dans la grande majorité des matchs). Les matchs de la mise à jour indiquent le vrai
    # partant (colonne « partant ») : il est alors prioritaire.
    if "partant" not in p.columns:
        p["partant"] = 0
    p["partant"] = p["partant"].fillna(0)
    idx = (p.sort_values(["partant", "batters_faced"], ascending=False)
           .groupby(["gamePk", "side"]).head(1).index)
    partant = (p.loc[idx]
               .set_index(["gamePk", "side"])
               [["player_id", "player_name", "outs", "earned_runs", "hits", "bb", "so",
                 "pitch_count"]]
               .rename(columns={"player_id": "id", "player_name": "nom"}))
    return a_plat(equipe, "lanc_").merge(a_plat(partant, "principal_"), on="gamePk")


def resumer_frappeurs():
    etape("Frappeurs : totaux offensifs par équipe (fichier volumineux)")
    b = lire_brut("batting",
                  usecols=["gamePk", "side", "player_id"] + COLS_FRAPPEURS)
    b = b.drop_duplicates()
    equipe = b.groupby(["gamePk", "side"])[COLS_FRAPPEURS].sum(min_count=1)
    return a_plat(equipe, "frap_")


def main():
    print("=== NETTOYAGE BASEBALL (MLB) ===")
    sch = charger_calendrier()
    df = sch.merge(resumer_manches(), on="gamePk", how="inner")

    etape("Contrôle : la somme des manches doit égaler le score final")
    ok = ((df["home_runs_total_manches"] == df["home_score"])
          & (df["away_runs_total_manches"] == df["away_score"]))
    print(f"Matchs incohérents retirés : {(~ok).sum()}")
    df = df[ok].drop(columns=["home_runs_total_manches", "away_runs_total_manches"])

    df = df.merge(resumer_lanceurs(), on="gamePk", how="left")
    df = df.merge(resumer_frappeurs(), on="gamePk", how="left")

    etape("Cibles des paris Over/Under")
    df["total_manche_1"] = df["home_m1"] + df["away_m1"]
    df["total_3_prem_manches"] = df["home_m1_3"] + df["away_m1_3"]
    df["total_match"] = df["home_score"] + df["away_score"]

    df = df.sort_values(["date", "gamePk"]).reset_index(drop=True)
    assert df["gamePk"].is_unique, "Des matchs sont encore en double"
    sauvegarder(df, "baseball_matchs.csv")


if __name__ == "__main__":
    main()
