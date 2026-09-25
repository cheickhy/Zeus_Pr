"""Variables d'entrée (features) du basket : uniquement des informations connues AVANT le match.

Règle : chaque statistique est calculée sur les matchs PRÉCÉDENTS (fonctions de
Sources/outils.py, qui décalent d'un match). Le match à prédire n'y entre jamais.

Entrée  : Données/Traitées/basketball_matchs.csv (produit par Basketball/nettoyage.py)
Sortie  : Données/Traitées/basketball_variables.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import (DOSSIER_TRAITEES, etape, moyenne_precedente,  # noqa: E402
                            sauvegarder)

FENETRES = [5, 10, 20]   # nombre de matchs précédents de l'équipe

STATS_EQUIPE = ["pts", "pts_encaisses", "pts_q1", "enc_q1", "pts_mt", "enc_mt",
                "possessions", "fga", "fg3a", "fta", "oreb", "dreb", "ast", "to", "pf",
                "fga_adv", "fg3a_adv", "fta_adv", "pf_adv"]

CIBLES = ["total_q1", "total_mi_temps", "total_match"]


def possessions(df, camp):
    """Estimation classique du nombre de possessions : mesure le rythme de jeu."""
    return (df[f"{camp}_fga"] - df[f"{camp}_oreb"] + df[f"{camp}_to"] + 0.44 * df[f"{camp}_fta"])


def table_equipes(df):
    """Une ligne par équipe et par match (domicile et extérieur réunis)."""
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        lignes.append(pd.DataFrame({
            "game_id": df["game_id"], "date": df["date"], "season": df["season"],
            "equipe": df[f"{camp}_team_id"], "domicile": int(camp == "home"),
            "pts": df[f"{camp}_pts_total"], "pts_encaisses": df[f"{adv}_pts_total"],
            "pts_q1": df[f"{camp}_pts_q1"], "enc_q1": df[f"{adv}_pts_q1"],
            "pts_mt": df[f"{camp}_pts_q1"] + df[f"{camp}_pts_q2"],
            "enc_mt": df[f"{adv}_pts_q1"] + df[f"{adv}_pts_q2"],
            "possessions": (possessions(df, camp) + possessions(df, adv)) / 2,
            "fga": df[f"{camp}_fga"], "fg3a": df[f"{camp}_fg3a"], "fta": df[f"{camp}_fta"],
            "oreb": df[f"{camp}_oreb"], "dreb": df[f"{camp}_dreb"], "ast": df[f"{camp}_ast"],
            "to": df[f"{camp}_to"], "pf": df[f"{camp}_pf"],
            # Ce que l'équipe laisse faire à ses adversaires (qualité défensive)
            "fga_adv": df[f"{adv}_fga"], "fg3a_adv": df[f"{adv}_fg3a"],
            "fta_adv": df[f"{adv}_fta"], "pf_adv": df[f"{adv}_pf"],
        }))
    return pd.concat(lignes).sort_values(["equipe", "date", "game_id"]).reset_index(drop=True)


def variables_equipes(eq):
    etape("Forme récente des équipes (5, 10 et 20 derniers matchs)")
    cols = {}
    for n in FENETRES:
        for col in STATS_EQUIPE:
            cols[f"{col}_moy{n}"] = moyenne_precedente(eq, "equipe", col, n)

    etape("Moyennes depuis le début de la saison, repos et fatigue")
    eq["equipe_saison"] = eq["equipe"].astype(str) + "_" + eq["season"].astype(str)
    for col in ["pts", "pts_encaisses", "pts_q1", "enc_q1", "pts_mt", "enc_mt", "possessions"]:
        cols[f"{col}_saison"] = moyenne_precedente(eq, "equipe_saison", col, 100, min_matchs=3)
    cols["matchs_joues_saison"] = eq.groupby("equipe_saison").cumcount()
    ecart = (eq["date"] - eq.groupby("equipe")["date"].shift(1)).dt.days
    cols["jours_repos"] = ecart.clip(upper=7)
    cols["dos_a_dos"] = (ecart == 1).astype(int)   # a joué la veille
    # Toutes les colonnes ajoutées d'un coup (plus rapide que une par une)
    return pd.concat([eq[["game_id", "equipe"]], pd.DataFrame(cols)], axis=1)


def contexte_ligue(df):
    etape("Niveau moyen de la ligue sur les 30 jours précédents")
    jour = df.groupby("date")[CIBLES].agg(["sum", "count"])
    glisse = jour.rolling("30D", closed="left").sum()   # closed="left" exclut le jour même
    res = pd.DataFrame(index=jour.index)
    for c in CIBLES:
        res[f"ligue_{c}_30j"] = glisse[(c, "sum")] / glisse[(c, "count")]
    return res.reset_index()


def rattacher(df, feats, cle_df, prefixe):
    """Ajoute les variables d'une équipe (domicile ou extérieur) à chaque match."""
    f = feats.rename(columns={c: prefixe + c for c in feats.columns if c not in ("game_id", "equipe")})
    return df.merge(f.rename(columns={"equipe": cle_df}), on=["game_id", cle_df], how="left")


def main():
    print("=== VARIABLES D'ENTRÉE BASKET ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "basketball_matchs.csv", parse_dates=["date"])
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    eq = variables_equipes(table_equipes(df))

    etape("Assemblage : une ligne par match")
    sortie = df[["game_id", "date", "season", "home_team_id", "away_team_id",
                 "home_team_abbr", "away_team_abbr"] + CIBLES].copy()
    sortie = rattacher(sortie, eq, "home_team_id", "home_")
    sortie = rattacher(sortie, eq, "away_team_id", "away_")
    sortie = sortie.merge(contexte_ligue(df), on="date", how="left").copy()
    sortie["mois"] = sortie["date"].dt.month
    # Première année de la saison ("2008-09" -> 2008), utile pour découper dans le temps
    sortie["annee_saison"] = sortie["season"].str[:4].astype(int)

    assert sortie["game_id"].is_unique and len(sortie) == len(df)
    sauvegarder(sortie, "basketball_variables.csv")


if __name__ == "__main__":
    main()
