"""Mise à jour des données NBA : télécharge les saisons récentes depuis stats.nba.com.

Les fichiers bruts d'origine ne sont JAMAIS modifiés. Les données téléchargées sont
écrites dans Données/brutes/nba_data/mise_a_jour/ et Basketball/nettoyage.py les lit
en plus des fichiers d'origine.

Chaque saison récente est retéléchargée entièrement (environ 8 demandes par saison,
moins d'une minute) : c'est simple et ça corrige d'éventuels scores modifiés.

Fichiers produits :
- schedule.csv : une ligne par match (équipe à domicile), même format que l'origine
- quarters.csv : points par quart-temps et par équipe, même format que l'origine
- equipes.csv  : statistiques d'équipe par match (tirs, rebonds, fautes...),
                 qui remplacent le détail joueur par joueur

Utilisation : python Basketball/mise_a_jour.py
"""
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import BRUT_NBA, etape  # noqa: E402

API = "https://stats.nba.com/stats"
DOSSIER_MAJ = BRUT_NBA.parent / "mise_a_jour"
# Le site refuse les demandes qui ne ressemblent pas à celles d'un navigateur
ENTETES = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/",
           "Origin": "https://www.nba.com", "Accept": "application/json"}
# Période -> colonne (5 et 6 = 1ère et 2e prolongation, comme les données d'origine)
PERIODES = {1: "pts_q1", 2: "pts_q2", 3: "pts_q3", 4: "pts_q4", 5: "pts_ot1", 6: "pts_ot2"}
STATS = {"FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm", "FTA": "fta",
         "OREB": "oreb", "DREB": "dreb", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk",
         "TOV": "to", "PF": "pf"}
# Première saison dont les données d'origine sont incomplètes (2024-25 : 38 matchs manquants)
PREMIERE_SAISON_A_METTRE_A_JOUR = 2024


def telecharger(url, essais=3):
    for i in range(essais):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=ENTETES), timeout=60) as r:
                return json.load(r)
        except Exception:  # site lent ou coupure passagère : on réessaie
            if i == essais - 1:
                raise
            time.sleep(5 * (i + 1))


def tableau(url):
    """Convertit la réponse de stats.nba.com en tableau pandas."""
    res = telecharger(url)["resultSets"][0]
    return pd.DataFrame(res["rowSet"], columns=res["headers"])


def saisons_a_mettre_a_jour():
    """De 2025-26 jusqu'à la saison en cours (une saison NBA commence en octobre)."""
    aujourd_hui = date.today()
    derniere = aujourd_hui.year if aujourd_hui.month >= 10 else aujourd_hui.year - 1
    return [f"{a}-{(a + 1) % 100:02d}" for a in range(PREMIERE_SAISON_A_METTRE_A_JOUR, derniere + 1)]


def telecharger_saison(saison):
    etape(f"Saison {saison}")
    base = f"LeagueID=00&Season={saison}&SeasonType=Regular%20Season"
    totaux = tableau(f"{API}/leaguegamelog?Counter=0&Direction=ASC&PlayerOrTeam=T&Sorter=DATE&{base}")
    if totaux.empty:
        print("Aucun match joué pour l'instant.")
        return None
    totaux["game_id"] = totaux["GAME_ID"].astype(int)
    print(f"{totaux['game_id'].nunique()} matchs joués")

    quarts = totaux[["game_id", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].rename(columns={
        "TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team_abbr", "PTS": "pts_total"})
    for periode, col in PERIODES.items():
        p = tableau(f"{API}/teamgamelogs?Period={periode}&{base}")
        if p.empty:
            quarts[col] = 0.0
            continue
        p["game_id"] = p["GAME_ID"].astype(int)
        p = p[["game_id", "TEAM_ID", "PTS"]].rename(columns={"TEAM_ID": "team_id", "PTS": col})
        quarts = quarts.merge(p, on=["game_id", "team_id"], how="left")
        time.sleep(1)   # on ménage le site
    # Pas de ligne de prolongation = pas de prolongation jouée = 0 point
    quarts[["pts_ot1", "pts_ot2"]] = quarts[["pts_ot1", "pts_ot2"]].fillna(0)
    quarts.insert(1, "season", saison)

    # Une ligne par match : celle de l'équipe à domicile (« vs. »). Pour les matchs sur
    # terrain neutre, les deux lignes portent « @ » : on garde la première, et le
    # nettoyage considère alors l'autre équipe comme « à domicile ».
    totaux["_vs"] = totaux["MATCHUP"].str.contains(" vs. ", regex=False)
    domicile = totaux.sort_values("_vs", ascending=False).drop_duplicates("game_id")
    calendrier = pd.DataFrame({
        "game_id": domicile["game_id"], "season": saison,
        "date": pd.to_datetime(domicile["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
        "matchup": domicile["MATCHUP"], "team_id": domicile["TEAM_ID"],
        "team_name": domicile["TEAM_NAME"], "wl": domicile["WL"], "pts": domicile["PTS"]})

    equipes = totaux[["game_id", "TEAM_ID"] + list(STATS)].rename(columns={"TEAM_ID": "team_id", **STATS})
    return calendrier, quarts, equipes


def main():
    print("=== MISE À JOUR DES DONNÉES NBA ===")
    resultats = [r for r in map(telecharger_saison, saisons_a_mettre_a_jour()) if r is not None]
    if not resultats:
        print("Rien à mettre à jour.")
        return
    DOSSIER_MAJ.mkdir(parents=True, exist_ok=True)
    for i, nom in enumerate(["schedule", "quarters", "equipes"]):
        df = pd.concat([r[i] for r in resultats], ignore_index=True)
        df.to_csv(DOSSIER_MAJ / f"{nom}.csv", index=False)
        print(f"{nom}.csv : {len(df)} lignes")
    print(f"\nDernier match téléchargé : {resultats[-1][0]['date'].max()}")


if __name__ == "__main__":
    main()
