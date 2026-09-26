"""Téléchargement des championnats européens depuis football-data.co.uk (gratuit).

Chaque fichier contient une saison d'un championnat : date, équipes, buts, tirs,
fautes, corners, cartons (et les cotes des bookmakers, non utilisées pour l'instant).

Les fichiers sont enregistrés tels quels dans Données/brutes/football_data/.
- Saisons terminées : téléchargées une seule fois.
- Saison en cours : retéléchargée à chaque lancement (nouveaux matchs de la semaine).

Utilisation : python Football/mise_a_jour.py
"""
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_BRUTES, etape  # noqa: E402

URL = "https://www.football-data.co.uk/mmz4281/{saison}/{code}.csv"
DOSSIER = DOSSIER_BRUTES / "football_data"
PREMIERE_SAISON = 2005

CHAMPIONNATS = {
    "E0": "Angleterre - Premier League", "E1": "Angleterre - Championship",
    "E2": "Angleterre - League One", "E3": "Angleterre - League Two",
    "EC": "Angleterre - National League",
    "SC0": "Écosse - Premiership", "SC1": "Écosse - Championship",
    "SC2": "Écosse - League One", "SC3": "Écosse - League Two",
    "D1": "Allemagne - Bundesliga", "D2": "Allemagne - 2. Bundesliga",
    "I1": "Italie - Serie A", "I2": "Italie - Serie B",
    "SP1": "Espagne - La Liga", "SP2": "Espagne - Segunda División",
    "F1": "France - Ligue 1", "F2": "France - Ligue 2",
    "N1": "Pays-Bas - Eredivisie", "B1": "Belgique - Pro League",
    "P1": "Portugal - Primeira Liga", "T1": "Turquie - Süper Lig", "G1": "Grèce - Super League",
}


def saison_en_cours():
    """Une saison européenne commence en juillet-août : 2026-09 -> 2026."""
    aujourd_hui = date.today()
    return aujourd_hui.year if aujourd_hui.month >= 7 else aujourd_hui.year - 1


def code_saison(annee):
    """2025 -> "2526" (format utilisé par le site)."""
    return f"{annee % 100:02d}{(annee + 1) % 100:02d}"


def telecharger(url, essais=3):
    for i in range(essais):
        try:
            requete = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(requete, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:        # saison ou championnat absent du site
                return None
            if i == essais - 1:
                raise
        except Exception:
            if i == essais - 1:
                raise
        time.sleep(2 * (i + 1))


def main():
    print("=== TÉLÉCHARGEMENT FOOTBALL (football-data.co.uk) ===")
    DOSSIER.mkdir(parents=True, exist_ok=True)
    en_cours = saison_en_cours()
    nouveaux, absents = 0, 0
    for code, nom in CHAMPIONNATS.items():
        etape(nom)
        for annee in range(PREMIERE_SAISON, en_cours + 1):
            fichier = DOSSIER / f"{code}_{code_saison(annee)}.csv"
            if fichier.exists() and annee < en_cours:
                continue                     # saison terminée déjà téléchargée
            contenu = telecharger(URL.format(saison=code_saison(annee), code=code))
            if not contenu:
                absents += 1
                continue
            fichier.write_bytes(contenu)
            nouveaux += 1
            time.sleep(0.3)                  # on ménage le site
        print(f"{len(list(DOSSIER.glob(f'{code}_*.csv')))} saisons disponibles")
    print(f"\nFichiers téléchargés ou mis à jour : {nouveaux} | saisons absentes du site : {absents}")


if __name__ == "__main__":
    main()
