"""Variables d'entrée des sélections nationales (CAN comprise) : informations connues AVANT le match.

Deux familles de variables :
1. Classement Elo : une note de force par sélection, mise à jour après chaque match.
   La note gagnée dépend de l'importance du match (CAN > qualifications > amical),
   de l'écart de buts et de la force de l'adversaire. On utilise la note AVANT le match.
2. Forme récente : buts marqués / encaissés et points pris sur les matchs PRÉCÉDENTS.

Entrée  : Données/Traitées/football_selections.csv (produit par nettoyage_selections.py)
Sortie  : Données/Traitées/football_selections_variables.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import (DOSSIER_TRAITEES, etape, moyenne_precedente,  # noqa: E402
                            sauvegarder)

ELO_DEPART = 1500
AVANTAGE_DOMICILE = 100      # points Elo ajoutés à l'équipe qui reçoit (sauf terrain neutre)
FENETRES = [5, 10]

# Importance du match : plus K est grand, plus la note bouge (méthode eloratings.net)
GRANDS_TOURNOIS = {"African Cup of Nations", "UEFA Euro", "Copa América", "AFC Asian Cup",
                   "Gold Cup", "CONCACAF Championship"}


def importance(tournoi):
    if tournoi == "FIFA World Cup":
        return 60
    if tournoi in GRANDS_TOURNOIS:
        return 50
    if "qualification" in tournoi or "Nations League" in tournoi:
        return 40
    if tournoi == "Friendly":
        return 20
    return 30


def classement_elo(df):
    """Note Elo de chaque équipe AVANT chaque match (les matchs doivent être triés par date)."""
    etape("Classement Elo (note de force avant chaque match)")
    notes = {}
    avant_dom, avant_ext = np.empty(len(df)), np.empty(len(df))
    for i, m in enumerate(df.itertuples(index=False)):
        rd, re_ = notes.get(m.home_team, ELO_DEPART), notes.get(m.away_team, ELO_DEPART)
        avant_dom[i], avant_ext[i] = rd, re_
        if np.isnan(m.home_score):          # match à venir : pas de mise à jour
            continue
        ecart = rd - re_ + (0 if m.neutral else AVANTAGE_DOMICILE)
        attendu = 1 / (10 ** (-ecart / 400) + 1)
        reel = 1.0 if m.home_score > m.away_score else 0.5 if m.home_score == m.away_score else 0.0
        diff = abs(m.home_score - m.away_score)
        multiplicateur = 1 if diff <= 1 else 1.5 if diff == 2 else (11 + diff) / 8
        delta = importance(m.tournament) * multiplicateur * (reel - attendu)
        notes[m.home_team], notes[m.away_team] = rd + delta, re_ - delta
    return avant_dom, avant_ext


def table_equipes(df):
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        t = pd.DataFrame({"match": df["match"], "date": df["date"], "equipe": df[f"{camp}_team"],
                          "buts_pour": df[f"{camp}_score"], "buts_contre": df[f"{adv}_score"]})
        t["points"] = np.select([t.buts_pour > t.buts_contre, t.buts_pour == t.buts_contre], [3, 1], 0)
        t.loc[t["buts_pour"].isna(), "points"] = np.nan
        lignes.append(t)
    return pd.concat(lignes).sort_values(["equipe", "date", "match"]).reset_index(drop=True)


def variables_equipes(eq):
    etape("Forme récente des sélections (5 et 10 derniers matchs)")
    cols = {}
    for n in FENETRES:
        for col in ["buts_pour", "buts_contre", "points"]:
            cols[f"{col}_moy{n}"] = moyenne_precedente(eq, "equipe", col, n, min_matchs=3)
    cols["jours_depuis_dernier_match"] = (eq["date"] - eq.groupby("equipe")["date"].shift(1)).dt.days.clip(upper=365)
    cols["nb_matchs_joues"] = eq.groupby("equipe").cumcount()
    return pd.concat([eq[["match", "equipe"]], pd.DataFrame(cols)], axis=1)


def construire(df):
    """Calcule toutes les variables (une ligne par match). Les matchs à venir ont un score vide."""
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    df["match"] = range(len(df))
    df["home_elo"], df["away_elo"] = classement_elo(df)

    eq = variables_equipes(table_equipes(df))
    etape("Assemblage : une ligne par match")
    sortie = df.copy()
    for camp in ["home", "away"]:
        f = eq.rename(columns={c: f"{camp}_{c}" for c in eq.columns if c not in ("match", "equipe")})
        sortie = sortie.merge(f.rename(columns={"equipe": f"{camp}_team"}), on=["match", f"{camp}_team"], how="left")
    sortie["ecart_elo"] = (sortie["home_elo"] - sortie["away_elo"]
                           + np.where(sortie["neutral"] == 1, 0, AVANTAGE_DOMICILE))
    sortie["importance"] = sortie["tournament"].map(importance)
    sortie["qualification"] = sortie["tournament"].str.contains("qualification").astype(int)
    sortie["equipe_hote"] = (sortie["country"] == sortie["home_team"]).astype(int)
    sortie["annee"] = sortie["date"].dt.year
    return sortie


def main():
    print("=== VARIABLES D'ENTRÉE FOOTBALL : SÉLECTIONS ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "football_selections.csv", parse_dates=["date"])
    sauvegarder(construire(df), "football_selections_variables.csv")


if __name__ == "__main__":
    main()
