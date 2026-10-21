"""Matchs et statistiques de la Ligue des champions depuis ESPN (passés et à venir)."""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_BRUTES, etape  # noqa: E402

API = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}"
COMPETITIONS = {"uefa.champions": "Ligue des champions"}
PREMIERE_ANNEE = 2016
FICHIER = DOSSIER_BRUTES / "coupes" / "coupes_matchs.csv"
STATS = {"wonCorners": "corners", "foulsCommitted": "fautes", "yellowCards": "jaunes", "redCards": "rouges",
         "totalShots": "tirs", "shotsOnTarget": "tirs_cadres", "possessionPct": "possession"}


def telecharger(url, essais=3, delai_max=60):
    for i in range(essais):
        try:
            debut, morceaux = time.time(), []
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as r:
                while morceau := r.read(65536):
                    morceaux.append(morceau)
                    if time.time() - debut > delai_max:
                        raise TimeoutError("téléchargement trop lent")
            return json.loads(b"".join(morceaux))
        except Exception:
            if i == essais - 1:
                raise
            time.sleep(2 * (i + 1))


def matchs_de_l_annee(code, annee):
    lignes = []
    for e in telecharger(API.format(code) + f"/scoreboard?dates={annee}&limit=1000").get("events", []):
        c = e["competitions"][0]
        eq = {x["homeAway"]: x for x in c["competitors"]}
        lignes.append({"espn_id": str(e["id"]), "code": code, "competition": COMPETITIONS[code],
                       "date": e["date"][:10], "heure_utc": e["date"][11:16],
                       "home_team": eq["home"]["team"]["displayName"], "away_team": eq["away"]["team"]["displayName"],
                       "termine": e["status"]["type"]["completed"], "statut": e["status"]["type"]["description"],
                       "home_buts": float(eq["home"]["score"]) if e["status"]["type"]["completed"] else None,
                       "away_buts": float(eq["away"]["score"]) if e["status"]["type"]["completed"] else None})
    return lignes


def statistiques(ligne):
    """Corners, fautes, cartons... des deux équipes d'un match terminé."""
    resume = telecharger(API.format(ligne["code"]) + f"/summary?event={ligne['espn_id']}")
    res = {"espn_id": ligne["espn_id"]}
    for t in resume.get("boxscore", {}).get("teams", []):
        camp = "home" if t["team"]["displayName"] == ligne["home_team"] else "away"
        for s in t.get("statistics", []):
            if s["name"] in STATS:
                res[f"{camp}_{STATS[s['name']]}"] = pd.to_numeric(s["displayValue"], errors="coerce")
    return res


def main():
    print("=== MISE À JOUR DES COUPES D'EUROPE (ESPN) ===")
    anciens = pd.read_csv(FICHIER, dtype={"espn_id": str}) if FICHIER.exists() else pd.DataFrame()
    deja = set(anciens.loc[anciens["termine"] & anciens["home_corners"].notna(), "espn_id"]) if len(anciens) else set()
    premiere = PREMIERE_ANNEE if (not deja or "--complet" in sys.argv) else date.today().year - 1
    lignes = []
    for code, nom in COMPETITIONS.items():
        etape(f"{nom} : liste des matchs de {premiere} à {date.today().year + 1}")
        for annee in range(premiere, date.today().year + 2):
            lignes += matchs_de_l_annee(code, annee)
    matchs = pd.DataFrame(lignes).drop_duplicates("espn_id", keep="last")
    print(f"{len(matchs)} matchs trouvés | terminés : {matchs['termine'].sum()} | à venir : {(~matchs['termine']).sum()}")

    a_completer = matchs[matchs["termine"] & ~matchs["espn_id"].isin(deja)]
    etape(f"Statistiques de {len(a_completer)} matchs terminés")
    with ThreadPoolExecutor(max_workers=6) as pool:
        stats = pd.DataFrame(list(pool.map(statistiques, a_completer.to_dict("records"))))
    nouveaux = matchs.merge(stats, on="espn_id", how="left") if len(stats) else matchs

    # On garde les anciens matchs non retéléchargés (années précédentes) et ceux qui avaient déjà
    # leurs statistiques ; le reste vient du nouveau téléchargement
    garder = (~anciens["espn_id"].isin(matchs["espn_id"]) | anciens["espn_id"].isin(deja)) if len(anciens) else []
    tout = pd.concat([anciens[garder] if len(anciens) else anciens,
                      nouveaux[~nouveaux["espn_id"].isin(deja)]], ignore_index=True)
    tout = tout.sort_values(["date", "heure_utc"])
    FICHIER.parent.mkdir(parents=True, exist_ok=True)
    tout.to_csv(FICHIER, index=False)
    print(f"Total enregistré : {len(tout)} matchs -> {FICHIER}")


if __name__ == "__main__":
    main()
