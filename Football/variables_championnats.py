"""Variables d'entrée des championnats européens : uniquement des informations connues AVANT le match."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import (DOSSIER_TRAITEES, etape, moyenne_precedente,  # noqa: E402
                            sauvegarder)

FENETRES = [5, 10, 20]   # nombre de matchs précédents de l'équipe

# Statistique de l'équipe -> (colonne de l'équipe, colonne de l'adversaire)
STATS = {"buts": "buts", "buts_mt": "buts_mt", "tirs": "tirs", "tirs_cadres": "tirs_cadres",
         "corners": "corners", "fautes": "fautes", "jaunes": "jaunes", "rouges": "rouges"}

CIBLES = ["total_buts", "total_buts_mt", "total_corners", "total_fautes", "total_cartons_jaunes"]


def pays(code):
    """E0, E1, EC -> "E" ; SC0 -> "SC" ; D1 -> "D". Une équipe change de division, pas de pays."""
    return "E" if code == "EC" else re.sub(r"\d+$", "", code)


def table_equipes(df):
    """Une ligne par équipe et par match : ce qu'elle a fait (pour) et subi (contre)."""
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        t = pd.DataFrame({"match": df["match"], "date": df["date"], "saison": df["saison"],
                          "equipe": df["pays"] + "|" + df[f"{camp}_team"],
                          "domicile": int(camp == "home")})
        for nom, col in STATS.items():
            t[f"{nom}_pour"] = df[f"{camp}_{col}"]
            t[f"{nom}_contre"] = df[f"{adv}_{col}"]
        lignes.append(t)
    return pd.concat(lignes).sort_values(["equipe", "date", "match"]).reset_index(drop=True)


def variables_equipes(eq):
    etape("Forme récente des équipes (5, 10 et 20 derniers matchs)")
    cols = {}
    stats = [c for c in eq.columns if c.endswith(("_pour", "_contre"))]
    for n in FENETRES:
        for col in stats:
            cols[f"{col}_moy{n}"] = moyenne_precedente(eq, "equipe", col, n)

    etape("Moyennes de la saison en cours et jours de repos")
    eq["equipe_saison"] = eq["equipe"] + "_" + eq["saison"]
    for col in ["buts_pour", "buts_contre", "corners_pour", "corners_contre",
                "fautes_pour", "fautes_contre", "jaunes_pour", "jaunes_contre"]:
        cols[f"{col}_saison"] = moyenne_precedente(eq, "equipe_saison", col, 60, min_matchs=3)
    cols["matchs_joues_saison"] = eq.groupby("equipe_saison").cumcount()
    cols["jours_repos"] = (eq["date"] - eq.groupby("equipe")["date"].shift(1)).dt.days.clip(upper=30)
    return pd.concat([eq[["match", "equipe"]], pd.DataFrame(cols)], axis=1)


def variables_arbitres(df):
    """Cartons et fautes sifflés par l'arbitre sur ses 20 matchs précédents (Angleterre surtout)."""
    etape("Habitudes des arbitres (20 matchs précédents)")
    a = df[["match", "date", "arbitre", "total_cartons_jaunes", "total_fautes"]].dropna(subset=["arbitre"])
    a = a.sort_values(["arbitre", "date", "match"]).reset_index(drop=True)
    res = a[["match"]].copy()
    res["arbitre_jaunes_moy20"] = moyenne_precedente(a, "arbitre", "total_cartons_jaunes", 20, min_matchs=5)
    res["arbitre_fautes_moy20"] = moyenne_precedente(a, "arbitre", "total_fautes", 20, min_matchs=5)
    return res


def contexte_championnat(df):
    """Moyennes du championnat sur les 60 jours précédents (le jour même exclu)."""
    etape("Niveau moyen du championnat sur les 60 jours précédents")
    res = []
    for code, g in df.groupby("code"):
        jour = g.groupby("date")[CIBLES].agg(["sum", "count"])
        glisse = jour.rolling("60D", closed="left").sum()
        r = pd.DataFrame({"date": jour.index, "code": code})
        for c in CIBLES:
            r[f"champ_{c}_60j"] = (glisse[(c, "sum")] / glisse[(c, "count")]).to_numpy()
        res.append(r)
    return pd.concat(res)


def rattacher(df, feats, cle, prefixe):
    f = feats.rename(columns={c: prefixe + c for c in feats.columns if c not in ("match", "equipe")})
    return df.merge(f.rename(columns={"equipe": cle}), on=["match", cle], how="left")


def construire(df):
    """Calcule toutes les variables à partir des matchs (une ligne par match).

    Réutilisable pour des matchs à venir : lignes sans résultat ajoutées à la fin.
    """
    df = df.sort_values(["date", "code", "home_team"]).reset_index(drop=True)
    df["match"] = range(len(df))
    df["pays"] = df["code"].map(pays)
    df["cle_dom"] = df["pays"] + "|" + df["home_team"]
    df["cle_ext"] = df["pays"] + "|" + df["away_team"]

    eq = variables_equipes(table_equipes(df))

    etape("Assemblage : une ligne par match")
    sortie = df[["match", "code", "championnat", "saison", "date", "home_team", "away_team",
                 "arbitre", "cle_dom", "cle_ext"] + CIBLES].copy()
    sortie = rattacher(sortie, eq, "cle_dom", "home_")
    sortie = rattacher(sortie, eq, "cle_ext", "away_")
    sortie = sortie.merge(variables_arbitres(df), on="match", how="left")
    sortie = sortie.merge(contexte_championnat(df), on=["date", "code"], how="left").copy()
    sortie["mois"] = sortie["date"].dt.month
    sortie["annee_saison"] = sortie["saison"].str[:4].astype(int)
    # Numéro de division (0 = 1re division ; National League anglaise = 4)
    sortie["division"] = sortie["code"].map(lambda c: 4 if c == "EC" else int(re.findall(r"\d+$", c)[0]))
    assert len(sortie) == len(df)
    return sortie.drop(columns=["cle_dom", "cle_ext"])


def main():
    print("=== VARIABLES D'ENTRÉE FOOTBALL : CHAMPIONNATS ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "football_championnats.csv", parse_dates=["date"], low_memory=False)
    sauvegarder(construire(df), "football_championnats_variables.csv")


if __name__ == "__main__":
    main()
