"""Entraînement et évaluation des premiers modèles basket (totaux des deux équipes).

Découpage dans le temps (jamais au hasard, sinon le modèle « voit le futur ») :
- apprentissage : saisons 2008-09 à 2021-22
- réglage       : saison 2022-23 (arrêt de l'apprentissage au bon moment)
- test final    : saisons 2023-24 et suivantes (matchs jamais vus)

Les lignes des bookmakers varient d'un match à l'autre : on mesure donc l'écart
moyen (en points) entre le total prédit et le total réel, et on compare à deux
prédictions naïves. On mesure aussi si le modèle se place du bon côté d'une
« ligne naïve » calculée à partir des moyennes des équipes (imitation grossière
d'une ligne de bookmaker, en attendant les vraies cotes).

Entrée : Données/Traitées/basketball_variables.csv (produit par Basketball/variables.py)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, etape  # noqa: E402

FIN_APPRENTISSAGE = 2021   # saison 2021-22
SAISON_REGLAGE = 2022      # saison 2022-23

NON_VARIABLES = {"game_id", "home_team_id", "away_team_id", "annee_saison",
                 "total_q1", "total_mi_temps", "total_match"}

# Cible -> suffixes des colonnes « points marqués / encaissés » correspondantes
CIBLES = {
    "total_q1": ("pts_q1", "enc_q1", "1er quart-temps"),
    "total_mi_temps": ("pts_mt", "enc_mt", "Mi-temps"),
    "total_match": ("pts", "pts_encaisses", "Match complet"),
}


def ligne_naive(d, marques, encaisses, n=10):
    """Total attendu = (attaque domicile + défense extérieur)/2 + (attaque extérieur + défense domicile)/2."""
    return ((d[f"home_{marques}_moy{n}"] + d[f"away_{encaisses}_moy{n}"]) / 2
            + (d[f"away_{marques}_moy{n}"] + d[f"home_{encaisses}_moy{n}"]) / 2)


def main():
    print("=== ENTRAÎNEMENT BASKET ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "basketball_variables.csv", parse_dates=["date"])
    variables = [c for c in df.select_dtypes("number").columns if c not in NON_VARIABLES]
    app = df[df["annee_saison"] <= FIN_APPRENTISSAGE]
    reg = df[df["annee_saison"] == SAISON_REGLAGE]
    test = df[df["annee_saison"] > SAISON_REGLAGE]
    print(f"{len(variables)} variables | apprentissage {len(app)} matchs | "
          f"réglage {len(reg)} | test {len(test)}")

    for cible, (marques, encaisses, nom) in CIBLES.items():
        etape(f"Cible : total {nom}")
        y_test = test[cible]
        ligne = ligne_naive(test, marques, encaisses)
        predictions = {
            "Moyenne de la ligue (30 j)": test[f"ligue_{cible}_30j"],
            "Ligne naïve des équipes": ligne,
        }

        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
        ridge.fit(app[variables], app[cible])
        predictions["Régression linéaire"] = ridge.predict(test[variables])

        gbm = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=15,
                                min_child_samples=100, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.5, reg_lambda=5, verbose=-1)
        gbm.fit(app[variables], app[cible], eval_X=(reg[variables],), eval_y=(reg[cible],),
                eval_metric="l1", callbacks=[lgb.early_stopping(100, verbose=False)])
        predictions["LightGBM"] = gbm.predict(test[variables])

        lignes = []
        valide = ligne.notna() & (y_test != ligne)   # on ignore les « push » (total = ligne)
        for modele, pred in predictions.items():
            pred = pd.Series(np.asarray(pred), index=test.index)
            ok = pred.notna()
            bon_cote = ((pred > ligne) == (y_test > ligne))[valide & ok & (pred != ligne)]
            lignes.append({"modèle": modele,
                           "écart moyen (points)": mean_absolute_error(y_test[ok], pred[ok]),
                           "bon côté de la ligne naïve": bon_cote.mean() if len(bon_cote) else np.nan})
        print(f"Total réel moyen pendant le test : {y_test.mean():.1f} points")
        print(pd.DataFrame(lignes).set_index("modèle").round(3).to_string())

        imp = pd.Series(gbm.booster_.feature_importance("gain"), index=variables)
        print("Variables les plus utiles :", ", ".join(imp.sort_values(ascending=False).head(6).index))


if __name__ == "__main__":
    main()
