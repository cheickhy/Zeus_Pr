"""Matchs récents et à venir des sélections (CAN, qualifications, amicaux) depuis ESPN."""
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_BRUTES, etape  # noqa: E402

API = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/scoreboard?dates={}"
FICHIER = DOSSIER_BRUTES / "selections" / "espn_matchs.csv"
JOURS_A_VENIR = 14

# Compétition ESPN -> nom utilisé dans les données historiques
COMPETITIONS = {
    "caf.nations": "African Cup of Nations",
    "caf.nations_qual": "African Cup of Nations qualification",
    "fifa.worldq.caf": "FIFA World Cup qualification",
    "fifa.world": "FIFA World Cup",
    "fifa.friendly": "Friendly",
}


def telecharger(url, essais=3):
    for i in range(essais):
        try:
            requete = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(requete, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == essais - 1:
                raise
            time.sleep(2 * (i + 1))


def matchs_du_jour(code, jour):
    lignes = []
    for e in telecharger(API.format(code, jour.strftime("%Y%m%d"))).get("events", []):
        c = e["competitions"][0]
        equipes = {x["homeAway"]: x for x in c["competitors"]}
        termine = e["status"]["type"]["completed"]
        lieu = c.get("venue", {}).get("address", {})
        lignes.append({
            "date": e["date"][:10], "home_team": equipes["home"]["team"]["displayName"],
            "away_team": equipes["away"]["team"]["displayName"],
            # ESPN affiche 0-0 pour un match à venir : le score n'est gardé que si le match est fini
            "home_score": float(equipes["home"]["score"]) if termine else None,
            "away_score": float(equipes["away"]["score"]) if termine else None,
            "tournament": COMPETITIONS[code], "city": lieu.get("city"), "country": lieu.get("country"),
            "neutral": bool(c.get("neutralSite", False)), "espn_id": e["id"],
            "statut": e["status"]["type"]["description"],
        })
    return lignes


def main(premier_jour=None):
    """premier_jour : date à partir de laquelle compléter (par défaut, reprise là où on s'était arrêté)."""
    print("=== MISE À JOUR DES SÉLECTIONS (ESPN) ===")
    anciens = pd.read_csv(FICHIER) if FICHIER.exists() else pd.DataFrame()
    if premier_jour is None:
        termines = anciens.dropna(subset=["home_score"]) if len(anciens) else anciens
        premier_jour = (date.fromisoformat(termines["date"].max()) - timedelta(days=3)
                        if len(termines) else date.today() - timedelta(days=60))
    dernier_jour = date.today() + timedelta(days=JOURS_A_VENIR)
    etape(f"Du {premier_jour} au {dernier_jour} ({len(COMPETITIONS)} compétitions)")

    nouveaux = []
    jour = premier_jour
    while jour <= dernier_jour:
        for code in COMPETITIONS:
            nouveaux += matchs_du_jour(code, jour)
        jour += timedelta(days=1)
    nouveaux = pd.DataFrame(nouveaux)

    # Les données fraîches remplacent les anciennes (un match à venir devient un match joué).
    # Identifiants en texte des deux côtés, sinon les doublons ne sont pas reconnus.
    tout = pd.concat([anciens, nouveaux], ignore_index=True)
    tout["espn_id"] = tout["espn_id"].astype(str)
    tout = tout.drop_duplicates("espn_id", keep="last").sort_values(["date", "home_team"])
    FICHIER.parent.mkdir(parents=True, exist_ok=True)
    tout.to_csv(FICHIER, index=False)
    print(f"{len(nouveaux)} matchs récupérés | joués : {nouveaux['home_score'].notna().sum()} | "
          f"à venir : {nouveaux['home_score'].isna().sum()} | total enregistré : {len(tout)}")


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None)
