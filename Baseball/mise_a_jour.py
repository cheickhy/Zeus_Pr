
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import BRUT_MLB, etape  # noqa: E402

API = "https://statsapi.mlb.com/api/v1"
DOSSIER_MAJ = BRUT_MLB.parent / "mise_a_jour"
STATUTS_TERMINES = {"Final", "Completed Early", "Game Over"}


def telecharger(url, essais=3):
    for i in range(essais):
        try:
            requete = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(requete, timeout=30) as r:
                return json.load(r)
        except Exception:  # coupure réseau passagère : on réessaie
            if i == essais - 1:
                raise
            time.sleep(2 * (i + 1))


def lire(nom):
    chemin = DOSSIER_MAJ / f"{nom}.csv"
    return pd.read_csv(chemin) if chemin.exists() else pd.DataFrame()


def matchs_deja_connus():
    """gamePk des matchs terminés présents dans les données brutes ou déjà mis à jour."""
    brut = pd.read_csv(BRUT_MLB / "schedule.csv", usecols=["gamePk", "status"])
    connus = set(brut.loc[brut["status"].isin(STATUTS_TERMINES), "gamePk"])
    maj = lire("schedule")
    if len(maj):
        connus |= set(maj["gamePk"])
    return connus


def date_depart():
    """Quelques jours avant le dernier match terminé connu, par sécurité."""
    brut = pd.read_csv(BRUT_MLB / "schedule.csv", usecols=["date", "status"])
    dates = [brut.loc[brut["status"].isin(STATUTS_TERMINES), "date"].max()]
    maj = lire("schedule")
    if len(maj):
        dates.append(maj["date"].max())
    return date.fromisoformat(max(dates)) - timedelta(days=3)


def ligne_calendrier(g):
    meteo = g.get("weather", {})
    return {
        "gamePk": g["gamePk"], "season": int(g["season"]), "date": g["officialDate"],
        "status": g["status"]["detailedState"],
        "home_team": g["teams"]["home"]["team"]["name"], "home_id": g["teams"]["home"]["team"]["id"],
        "away_team": g["teams"]["away"]["team"]["name"], "away_id": g["teams"]["away"]["team"]["id"],
        "home_score": g["teams"]["home"].get("score"), "away_score": g["teams"]["away"].get("score"),
        "venue": g["venue"]["name"], "day_night": g.get("dayNight"),
        "weather_temp": pd.to_numeric(meteo.get("temp"), errors="coerce"),
        "weather_cond": meteo.get("condition"), "wind_speed": meteo.get("wind"),
    }


def lignes_manches(g):
    lignes = []
    for m in g.get("linescore", {}).get("innings", []):
        h, a = m.get("home", {}), m.get("away", {})
        lignes.append({
            "gamePk": g["gamePk"], "season": int(g["season"]), "date": g["officialDate"],
            "inning": m["num"],
            "home_runs": h.get("runs", 0), "away_runs": a.get("runs", 0),
            "home_hits": h.get("hits", 0), "away_hits": a.get("hits", 0),
            "home_errors": h.get("errors", 0), "away_errors": a.get("errors", 0),
        })
    return lignes


def lignes_feuille_de_match(g):
    """Statistiques des lanceurs et des frappeurs, au format des fichiers bruts."""
    boite = telecharger(f"{API}/game/{g['gamePk']}/boxscore")
    base = {"gamePk": g["gamePk"], "season": int(g["season"]), "date": g["officialDate"]}
    lanceurs, frappeurs = [], []
    for camp in ["home", "away"]:
        equipe = boite["teams"][camp]
        nom_equipe = equipe["team"]["name"]
        # L'API donne les lanceurs dans l'ordre d'entrée en jeu : le 1er est le partant
        for ordre, pid in enumerate(equipe["pitchers"]):
            j = equipe["players"][f"ID{pid}"]
            s = j["stats"]["pitching"]
            lanceurs.append({**base, "side": camp, "team": nom_equipe, "player_id": pid,
                             "player_name": j["person"]["fullName"],
                             "innings_pitched": float(s.get("inningsPitched", "0")),
                             "hits": s.get("hits"), "runs": s.get("runs"),
                             "earned_runs": s.get("earnedRuns"), "bb": s.get("baseOnBalls"),
                             "so": s.get("strikeOuts"), "hr": s.get("homeRuns"), "era": None,
                             "pitch_count": s.get("numberOfPitches"), "strikes": s.get("strikes"),
                             "balls": s.get("balls"), "batters_faced": s.get("battersFaced"),
                             "whip": None, "partant": int(ordre == 0)})
        for pid in equipe["batters"]:
            j = equipe["players"][f"ID{pid}"]
            s = j["stats"].get("batting", {})
            if not s:
                continue
            frappeurs.append({**base, "side": camp, "team": nom_equipe, "player_id": pid,
                              "player_name": j["person"]["fullName"],
                              "position": j.get("position", {}).get("abbreviation"),
                              "bat_order": pd.to_numeric(j.get("battingOrder"), errors="coerce"),
                              "ab": s.get("atBats"), "runs": s.get("runs"), "hits": s.get("hits"),
                              "doubles": s.get("doubles"), "triples": s.get("triples"),
                              "hr": s.get("homeRuns"), "rbi": s.get("rbi"),
                              "bb": s.get("baseOnBalls"), "ibb": s.get("intentionalWalks"),
                              "so": s.get("strikeOuts"), "sb": s.get("stolenBases"),
                              "cs": s.get("caughtStealing"), "avg": None, "obp": None,
                              "slg": None, "ops": None, "total_bases": s.get("totalBases"),
                              "left_on_base": s.get("leftOnBase"),
                              "gidp": s.get("groundIntoDoublePlay"), "hbp": s.get("hitByPitch"),
                              "sac_fly": s.get("sacFlies"), "sac_bunt": s.get("sacBunts")})
    return lanceurs, frappeurs


def main():
    print("=== MISE À JOUR DES DONNÉES MLB ===")
    debut, fin = date_depart(), date.today() - timedelta(days=1)
    etape(f"Calendrier du {debut} au {fin}")
    cal = telecharger(f"{API}/schedule?sportId=1&gameType=R,F,D,L,W&startDate={debut}"
                      f"&endDate={fin}&hydrate=linescore,weather,venue")
    connus = matchs_deja_connus()
    nouveaux = [g for d in cal["dates"] for g in d["games"]
                if g["status"]["detailedState"] in STATUTS_TERMINES and g["gamePk"] not in connus]
    print(f"Nouveaux matchs terminés à télécharger : {len(nouveaux)}")
    if not nouveaux:
        print("Données déjà à jour.")
        return

    etape("Téléchargement des feuilles de match (lanceurs et frappeurs)")
    with ThreadPoolExecutor(max_workers=6) as pool:
        feuilles = list(pool.map(lignes_feuille_de_match, nouveaux))

    nouvelles = {
        "schedule": pd.DataFrame([ligne_calendrier(g) for g in nouveaux]),
        "linescore": pd.DataFrame([m for g in nouveaux for m in lignes_manches(g)]),
        "pitching": pd.DataFrame([l for lanc, _ in feuilles for l in lanc]),
        "batting": pd.DataFrame([f for _, frap in feuilles for f in frap]),
    }
    DOSSIER_MAJ.mkdir(parents=True, exist_ok=True)
    for nom, df in nouvelles.items():
        total = pd.concat([lire(nom), df], ignore_index=True)
        total.to_csv(DOSSIER_MAJ / f"{nom}.csv", index=False)
        print(f"{nom}.csv : +{len(df)} lignes (total {len(total)})")
    print(f"\nDernier match téléchargé : {nouvelles['schedule']['date'].max()}")


if __name__ == "__main__":
    main()
