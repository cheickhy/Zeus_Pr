"""Probabilité de dépasser une ligne de bookmaker (buts, corners, fautes, cartons jaunes) pour un match prédit."""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import RACINE  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "football_predictions.csv"
MARCHES = {"buts": "buts", "corners": "corners", "fautes": "fautes", "jaunes": "cartons jaunes"}

UTILISATION = """Utilisation : python Football/proba_ligne.py ÉQUIPE MARCHÉ LIGNE
  ÉQUIPE : une partie du nom de l'une des deux équipes (ex. Barcelona, Man City, Inter)
  MARCHÉ : buts, corners, fautes ou jaunes
  LIGNE  : ligne du bookmaker (ex. 10.5)
Exemple : python Football/proba_ligne.py Barcelona corners 10.5"""


def main():
    if len(sys.argv) < 4 or sys.argv[-2] not in MARCHES:
        print(UTILISATION)
        return
    equipe, marche = " ".join(sys.argv[1:-2]).lower(), sys.argv[-2]
    ligne = float(sys.argv[-1].replace(",", "."))
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée. Lance d'abord : python Football/prediction_championnats.py")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
    if f"mu_{marche}" not in pred:
        print("Les prédictions enregistrées ne contiennent pas encore les modèles de comptage. Relance le programme de prédiction.")
        return
    pred = pred.dropna(subset=[f"mu_{marche}"])
    trouve = pred[pred["home_team"].str.lower().str.contains(equipe, regex=False) |
                  pred["away_team"].str.lower().str.contains(equipe, regex=False)].sort_values("date")
    if trouve.empty:
        print(f"Aucune prédiction pour « {equipe} » avec les modèles de comptage (ajoutés le 14 octobre 2026). "
              "Relance le programme de prédiction avant ses prochains matchs.")
        return
    m = trouve.iloc[-1]   # match le plus récent prédit pour cette équipe
    mu, r = m[f"mu_{marche}"], m[f"r_{marche}"]
    plus = stats.nbinom.sf(int(ligne), r, r / (r + mu))   # P(total > ligne) pour une ligne en x,5
    print(f"{m['home_team']} - {m['away_team']} du {m['date'].date()} ({m['championnat']}), {MARCHES[marche]}")
    print(f"Nombre attendu : {mu:.1f}")
    print(f"Plus de {ligne}  : {plus:.0%}")
    print(f"Moins de {ligne} : {1 - plus:.0%}")


if __name__ == "__main__":
    main()
