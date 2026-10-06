"""Probabilité de dépasser une ligne sur les 3 premières manches, à partir des prédictions enregistrées."""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import RACINE  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "baseball_predictions.csv"

UTILISATION = """Utilisation : python Baseball/proba_ligne.py ÉQUIPE LIGNE
  ÉQUIPE : une partie du nom de l'une des deux équipes (ex. Yankees, Dodgers, Red Sox)
  LIGNE  : ligne du bookmaker pour les 3 premières manches (ex. 3.5)
Exemple : python Baseball/proba_ligne.py Yankees 3.5"""


def main():
    if len(sys.argv) < 3:
        print(UTILISATION)
        return
    equipe, ligne = " ".join(sys.argv[1:-1]).lower(), float(sys.argv[-1].replace(",", "."))
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée. Lance d'abord : python Baseball/prediction.py")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
    if "mu_3_manches" not in pred:
        print("Les prédictions enregistrées ne contiennent pas encore les 3 premières manches.")
        return
    pred = pred.dropna(subset=["mu_3_manches"])
    trouve = pred[pred["home_team"].str.lower().str.contains(equipe) |
                  pred["away_team"].str.lower().str.contains(equipe)].sort_values("date")
    if trouve.empty:
        print(f"Aucune prédiction pour « {equipe} ».")
        return
    m = trouve.iloc[-1]   # match le plus récent prédit pour cette équipe
    mu, r = m["mu_3_manches"], m["r_3_manches"]
    plus = stats.nbinom.sf(int(ligne), r, r / (r + mu))   # P(total > ligne) pour une ligne en x,5
    print(f"{m['home_team']} - {m['away_team']} du {m['date'].date()}, 3 premières manches")
    print(f"Points attendus : {mu:.1f}")
    print(f"Plus de {ligne}  : {plus:.0%}")
    print(f"Moins de {ligne} : {1 - plus:.0%}")


if __name__ == "__main__":
    main()
