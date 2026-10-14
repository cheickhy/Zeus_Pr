"""Probabilité de dépasser une ligne de bookmaker, à partir des prédictions du basket enregistrées."""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import RACINE  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "basketball_predictions.csv"
MARCHES = {"q1": ("total_q1", "1er quart-temps"), "mt": ("total_mi_temps", "mi-temps"),
           "match": ("total_match", "match complet"),
           # Points de l'équipe nommée seulement (pas le total des deux équipes)
           "equipe-q1": ("q1", "1er quart-temps, points de l'équipe"),
           "equipe-mt": ("mi_temps", "mi-temps, points de l'équipe"),
           "equipe-match": ("match", "match complet, points de l'équipe")}

UTILISATION = """Utilisation : python Basketball/proba_ligne.py ÉQUIPE MARCHÉ LIGNE
  ÉQUIPE : abréviation de l'une des deux équipes (ex. BOS, LAL, NYK)
  MARCHÉ : q1 (1er quart-temps), mt (mi-temps) ou match : total des deux équipes
           equipe-q1, equipe-mt ou equipe-match : points de l'équipe nommée seulement
  LIGNE  : ligne du bookmaker (ex. 220.5)
Exemple : python Basketball/proba_ligne.py BOS match 220.5"""


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in MARCHES:
        print(UTILISATION)
        return
    equipe, (cible, nom), ligne = sys.argv[1].upper(), MARCHES[sys.argv[2]], float(sys.argv[3].replace(",", "."))
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée. Lance d'abord : python Basketball/prediction.py")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
    match = pred[(pred["home_team_abbr"] == equipe) | (pred["away_team_abbr"] == equipe)].sort_values("date")
    if match.empty:
        print(f"Aucune prédiction pour {equipe}.")
        return
    m = match.iloc[-1]   # match le plus récent prédit pour cette équipe
    if sys.argv[2].startswith("equipe-"):
        camp = "dom" if m["home_team_abbr"] == equipe else "ext"
        cible = f"equipe_{camp}_{cible}"
        nom = f"{nom} ({equipe})"
        if f"prevu_{cible}" not in m or pd.isna(m[f"prevu_{cible}"]):
            print("Ce match a été prédit avant l'ajout des points par équipe. Relance le programme de prédiction.")
            return
    prevu, sigma = m[f"prevu_{cible}"], m[f"sigma_{cible}"]
    plus = 1 - stats.norm.cdf((ligne - prevu) / sigma)
    print(f"{m['home_team_abbr']} - {m['away_team_abbr']} du {m['date'].date()}, {nom}")
    print(f"Total prévu : {prevu:.1f} points (erreur habituelle : ±{sigma:.0f})")
    print(f"Plus de {ligne}  : {plus:.0%}")
    print(f"Moins de {ligne} : {1 - plus:.0%}")
    if m.get("debut_saison") is True or str(m.get("debut_saison")) == "True":
        print("Prudence : début de saison, une des équipes a joué moins de 5 matchs. Probabilité peu fiable.")


if __name__ == "__main__":
    main()
