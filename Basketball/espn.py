"""Source de secours pour la NBA (ESPN), utilisée quand stats.nba.com ne répond pas."""
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import BRUT_NBA, etape

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
DOSSIER = BRUT_NBA.parent / "espn"
DOSSIER_MAJ = BRUT_NBA.parent / "mise_a_jour"
ABREVIATIONS = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
QUARTS = ["pts_q1", "pts_q2", "pts_q3", "pts_q4", "pts_ot1", "pts_ot2"]
# Vérifié sur des matchs présents dans les deux sources : mêmes valeurs que la NBA
STATS = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
         "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
         "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
         "offensiveRebounds": "oreb", "defensiveRebounds": "dreb", "totalRebounds": "reb",
         "assists": "ast", "steals": "stl", "blocks": "blk", "totalTurnovers": "to", "fouls": "pf"}


def telecharger(url, essais=3):
    for i in range(essais):
        try:
            requete = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(requete, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == essais - 1:
                raise
            time.sleep(3 * (i + 1))


def abreviation(equipe):
    a = equipe["team"]["abbreviation"]
    return ABREVIATIONS.get(a, a)


def identifiants():
    """Abréviation NBA de chaque équipe -> identifiant NBA."""
    q = pd.read_csv(DOSSIER_MAJ / "quarters.csv", usecols=["team_id", "team_abbr"])
    return q.drop_duplicates("team_abbr").set_index("team_abbr")["team_id"].to_dict()


def saison_de(jour):
    annee = jour.year if jour.month >= 10 else jour.year - 1
    return f"{annee}-{(annee + 1) % 100:02d}"


def matchs_du_jour(jour):
    """Matchs de saison régulière d'un jour (date américaine), sans les matchs reportés ou annulés."""
    matchs = []
    for e in telecharger(f"{API}/scoreboard?dates={jour:%Y%m%d}").get("events", []):
        if e["season"]["type"] != 2 or e["status"]["type"]["name"] in ("STATUS_POSTPONED", "STATUS_CANCELED"):
            continue
        matchs.append((e, {x["homeAway"]: x for x in e["competitions"][0]["competitors"]}))
    return matchs


def a_venir(jour):
    """Même format que prediction.matchs_a_venir."""
    ids = identifiants()
    lignes = []
    for e, eq in matchs_du_jour(jour):
        dom, ext = abreviation(eq["home"]), abreviation(eq["away"])
        lignes.append({"game_id": int(e["id"]), "date": pd.Timestamp(jour), "season": saison_de(jour),
                       "home_team_id": ids[dom], "away_team_id": ids[ext],
                       "home_team_abbr": dom, "away_team_abbr": ext,
                       "coup_envoi_utc": e["date"][:16].replace("T", " ")})
    return pd.DataFrame(lignes)


def resultats(debut, fin):
    """Matchs terminés entre deux dates, au format des fichiers de mise à jour de la NBA."""
    ids = identifiants()
    calendrier, quarts, equipes = [], [], []
    jour = debut
    while jour <= fin:
        for e, eq in matchs_du_jour(jour):
            if not e["status"]["type"]["completed"]:
                continue
            gid, saison = int(e["id"]), saison_de(jour)
            boite = {abreviation(t): {s["name"]: s["displayValue"] for s in t["statistics"]}
                     for t in telecharger(f"{API}/summary?event={e['id']}")["boxscore"]["teams"]}
            dom = eq["home"]
            calendrier.append({"game_id": gid, "season": saison, "date": jour.isoformat(),
                               "matchup": f"{abreviation(dom)} vs. {abreviation(eq['away'])}",
                               "team_id": ids[abreviation(dom)], "team_name": dom["team"]["displayName"],
                               "wl": "W" if dom.get("winner") else "L", "pts": int(dom["score"])})
            for x in eq.values():
                a = abreviation(x)
                points = [l["value"] for l in x.get("linescores", [])]
                quarts.append({"game_id": gid, "season": saison, "team_id": ids[a], "team_abbr": a,
                               "pts_total": int(x["score"]),
                               **{c: points[i] if i < len(points) else 0.0 for i, c in enumerate(QUARTS)}})
                ligne = {"game_id": gid, "team_id": ids[a]}
                for nom, col in STATS.items():
                    if isinstance(col, tuple):
                        ligne[col[0]], ligne[col[1]] = map(int, boite[a][nom].split("-"))
                    else:
                        ligne[col] = int(boite[a][nom])
                equipes.append(ligne)
            time.sleep(0.5)
        jour += timedelta(days=1)
    return calendrier, quarts, equipes


def main():
    """Complète avec ESPN les matchs joués depuis le dernier match obtenu auprès de la NBA."""
    etape("Source de secours : ESPN")
    debut = date.fromisoformat(pd.read_csv(DOSSIER_MAJ / "schedule.csv")["date"].max()) + timedelta(days=1)
    fichiers = resultats(debut, date.today())
    DOSSIER.mkdir(parents=True, exist_ok=True)
    for nom, lignes in zip(["schedule", "quarters", "equipes"], fichiers):
        pd.DataFrame(lignes).to_csv(DOSSIER / f"{nom}.csv", index=False)
    print(f"{len(fichiers[0])} matchs récupérés chez ESPN depuis le {debut}")


def effacer():
    """La NBA répond de nouveau : ses données remplacent celles d'ESPN."""
    for f in DOSSIER.glob("*.csv"):
        f.unlink()
