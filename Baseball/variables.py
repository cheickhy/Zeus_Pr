"""Variables d'entrée (features) du baseball : uniquement des informations connues AVANT le match."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import (DOSSIER_TRAITEES, etape, moyenne_precedente,  # noqa: E402
                            sauvegarder, somme_precedente)

FENETRES_EQUIPE = [10, 30]      # nombre de matchs précédents de l'équipe
FENETRES_LANCEUR = [5, 15]      # nombre de matchs précédents du lanceur principal

STATS_EQUIPE = ["pts", "pts_encaisses", "pts_m1", "enc_m1", "marque_m1", "encaisse_m1",
                "pts_m1_3", "enc_m1_3", "coups_surs", "bb", "so", "hr", "bases"]

CIBLES = ["total_manche_1", "point_en_manche_1", "total_3_prem_manches", "total_match"]


def table_equipes(df):
    """Une ligne par équipe et par match (domicile et extérieur réunis)."""
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        lignes.append(pd.DataFrame({
            "gamePk": df["gamePk"], "date": df["date"], "season": df["season"],
            "equipe": df[f"{camp}_id"],
            "pts": df[f"{camp}_score"], "pts_encaisses": df[f"{adv}_score"],
            "pts_m1": df[f"{camp}_m1"], "enc_m1": df[f"{adv}_m1"],
            "marque_m1": (df[f"{camp}_m1"] > 0).astype(int),
            "encaisse_m1": (df[f"{adv}_m1"] > 0).astype(int),
            "pts_m1_3": df[f"{camp}_m1_3"], "enc_m1_3": df[f"{adv}_m1_3"],
            "coups_surs": df[f"{camp}_frap_hits"], "bb": df[f"{camp}_frap_bb"],
            "so": df[f"{camp}_frap_so"], "hr": df[f"{camp}_frap_hr"],
            "bases": df[f"{camp}_frap_total_bases"],
        }))
    return pd.concat(lignes).sort_values(["equipe", "date", "gamePk"]).reset_index(drop=True)


def variables_equipes(eq):
    etape("Forme récente des équipes (10 et 30 derniers matchs)")
    res = eq[["gamePk", "equipe"]].copy()
    for n in FENETRES_EQUIPE:
        for col in STATS_EQUIPE:
            res[f"{col}_moy{n}"] = moyenne_precedente(eq, "equipe", col, n)

    etape("Moyennes depuis le début de la saison et jours de repos")
    eq["equipe_saison"] = eq["equipe"].astype(str) + "_" + eq["season"].astype(str)
    for col in ["pts", "pts_encaisses", "pts_m1", "enc_m1"]:
        res[f"{col}_saison"] = moyenne_precedente(eq, "equipe_saison", col, 200, min_matchs=5)
    res["matchs_joues_saison"] = eq.groupby("equipe_saison").cumcount()
    res["jours_repos"] = (eq["date"] - eq.groupby("equipe")["date"].shift(1)).dt.days.clip(upper=10)
    return res


def variables_lanceurs(df):
    etape("Forme récente du lanceur principal (5 et 15 derniers matchs)")
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        lignes.append(pd.DataFrame({
            "gamePk": df["gamePk"], "date": df["date"],
            "lanceur": df[f"{camp}_principal_id"],
            "outs": df[f"{camp}_principal_outs"], "er": df[f"{camp}_principal_earned_runs"],
            "h": df[f"{camp}_principal_hits"], "bb": df[f"{camp}_principal_bb"],
            "so": df[f"{camp}_principal_so"],
            # Le lanceur principal lance presque toujours la 1ère manche
            "enc_m1": df[f"{adv}_m1"], "encaisse_m1": (df[f"{adv}_m1"] > 0).astype(int),
        }))
    lc = pd.concat(lignes).sort_values(["lanceur", "date", "gamePk"]).reset_index(drop=True)

    res = lc[["gamePk", "lanceur"]].copy()
    res["lanc_nb_matchs"] = lc.groupby("lanceur").cumcount()
    for n in FENETRES_LANCEUR:
        s = {c: somme_precedente(lc, "lanceur", c, n) for c in ["outs", "er", "h", "bb", "so"]}
        outs = s["outs"].where(s["outs"] > 0)
        res[f"lanc_era{n}"] = 27 * s["er"] / outs            # points mérités par 9 manches
        res[f"lanc_whip{n}"] = 3 * (s["h"] + s["bb"]) / outs  # frappeurs sur base par manche
        res[f"lanc_so_manche{n}"] = 3 * s["so"] / outs
        res[f"lanc_outs_moy{n}"] = moyenne_precedente(lc, "lanceur", "outs", n, min_matchs=1)
        res[f"lanc_enc_m1_moy{n}"] = moyenne_precedente(lc, "lanceur", "enc_m1", n, min_matchs=1)
        res[f"lanc_encaisse_m1_taux{n}"] = moyenne_precedente(lc, "lanceur", "encaisse_m1", n,
                                                              min_matchs=1)
    return res


def effet_stade(df, nb_saisons=3, prudence=150):
    """Le stade favorise-t-il les points ? Calculé sur les saisons PRÉCÉDENTES uniquement.

    Facteur = moyenne du stade / moyenne de la ligue, sur les `nb_saisons` d'avant.
    Un stade avec peu de matchs est rapproché de 1 (neutre) : `prudence` matchs
    « virtuels » moyens sont ajoutés, pour ne pas croire un petit échantillon.
    """
    etape("Effet du stade (saisons précédentes uniquement)")
    par_saison = df.groupby(["venue", "season"]).agg(
        pts=("total_match", "sum"), m1=("total_manche_1", "sum"), n=("gamePk", "size"))
    ligue = df.groupby("season").agg(pts=("total_match", "sum"), m1=("total_manche_1", "sum"),
                                     n=("gamePk", "size"))
    lignes = []
    for saison in sorted(df["season"].unique()):
        avant = list(range(saison - nb_saisons, saison))
        stade = par_saison[par_saison.index.get_level_values("season").isin(avant)].groupby("venue").sum()
        lig = ligue[ligue.index.isin(avant)].sum()
        if lig["n"] == 0:
            continue
        for col in ["pts", "m1"]:
            moy_ligue = lig[col] / lig["n"]
            stade[f"stade_facteur_{col}"] = ((stade[col] + prudence * moy_ligue)
                                             / (stade["n"] + prudence)) / moy_ligue
        stade["season"] = saison
        lignes.append(stade[["season", "stade_facteur_pts", "stade_facteur_m1"]].reset_index())
    return pd.concat(lignes)


def fatigue_releveurs(df, jours=3):
    """Lancers des releveurs (tous sauf le lanceur principal) sur les `jours` précédents."""
    etape(f"Fatigue des releveurs ({jours} jours précédents)")
    lignes = []
    for camp in ["home", "away"]:
        lignes.append(pd.DataFrame({
            "gamePk": df["gamePk"], "date": df["date"], "equipe": df[f"{camp}_id"],
            "lancers_releve": df[f"{camp}_lanc_pitch_count"] - df[f"{camp}_principal_pitch_count"],
        }))
    t = pd.concat(lignes).sort_values(["equipe", "date", "gamePk"]).reset_index(drop=True)
    # closed="left" : fenêtre de `jours` jours qui s'arrête AVANT le jour du match
    somme = (t.set_index("date").groupby("equipe")["lancers_releve"]
             .rolling(f"{jours}D", closed="left").sum())
    t["releve_lancers_3j"] = somme.to_numpy()
    t["releve_lancers_3j"] = t["releve_lancers_3j"].fillna(0)
    return t[["gamePk", "equipe", "releve_lancers_3j"]]


def contexte_ligue(df):
    etape("Niveau moyen de la ligue sur les 30 jours précédents")
    jour = df.groupby("date")[["total_match", "total_manche_1"]].agg(["sum", "count"])
    jour.columns = ["_".join(c) for c in jour.columns]
    glisse = jour.rolling("30D", closed="left").sum()   # closed="left" exclut le jour même
    return pd.DataFrame({
        "ligue_total_match_30j": glisse["total_match_sum"] / glisse["total_match_count"],
        "ligue_total_m1_30j": glisse["total_manche_1_sum"] / glisse["total_manche_1_count"],
    }).reset_index()


def rattacher(df, feats, cle_df, cle_feats, prefixe):
    """Ajoute les variables `feats` (une ligne par match et par équipe/lanceur) à `df`."""
    f = feats.rename(columns={c: prefixe + c for c in feats.columns if c not in ("gamePk", cle_feats)})
    return df.merge(f.rename(columns={cle_feats: cle_df}), on=["gamePk", cle_df], how="left")


def construire(df):
    """Calcule toutes les variables d'entrée à partir des matchs (une ligne par match).

    Utilisé par main() pour l'entraînement, et par Baseball/prediction.py pour les
    matchs à venir (lignes sans résultat, ajoutées à la fin de l'historique).
    """
    df = df.sort_values(["date", "gamePk"]).reset_index(drop=True)
    df["point_en_manche_1"] = (df["total_manche_1"] > 0).astype(int).where(df["total_manche_1"].notna())

    eq = variables_equipes(table_equipes(df))
    lc = variables_lanceurs(df)

    etape("Assemblage : une ligne par match")
    sortie = df[["gamePk", "date", "season", "home_team", "away_team", "home_id", "away_id",
                 "home_principal_id", "away_principal_id", "venue"] + CIBLES].copy()
    sortie = rattacher(sortie, eq, "home_id", "equipe", "home_")
    sortie = rattacher(sortie, eq, "away_id", "equipe", "away_")
    sortie = rattacher(sortie, lc, "home_principal_id", "lanceur", "home_")
    sortie = rattacher(sortie, lc, "away_principal_id", "lanceur", "away_")
    sortie = sortie.merge(contexte_ligue(df), on="date", how="left")
    sortie = sortie.merge(effet_stade(df), on=["venue", "season"], how="left")
    fatigue = fatigue_releveurs(df)
    sortie = rattacher(sortie, fatigue, "home_id", "equipe", "home_")
    sortie = rattacher(sortie, fatigue, "away_id", "equipe", "away_")

    etape("Contexte du match (connu à l'avance)")
    sortie = sortie.copy()   # regroupe les colonnes en mémoire (évite un avertissement de lenteur)
    sortie["mois"] = sortie["date"].dt.month
    sortie["nuit"] = (df["day_night"] == "night").astype(int)
    sortie["temperature"] = df["weather_temp"]
    # Le vent est stocké en texte (« 9 mph, L To R ») : on extrait la vitesse
    sortie["vent"] = pd.to_numeric(df["wind_speed"].astype(str).str.extract(r"(\d+)\s*mph")[0],
                                   errors="coerce")
    sortie["toit_ferme"] = df["weather_cond"].isin(["Dome", "Roof Closed"]).astype(int)

    assert sortie["gamePk"].is_unique and len(sortie) == len(df)
    return sortie


def main():
    print("=== VARIABLES D'ENTRÉE BASEBALL ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "baseball_matchs.csv", parse_dates=["date"])
    sauvegarder(construire(df), "baseball_variables.csv")


if __name__ == "__main__":
    main()
