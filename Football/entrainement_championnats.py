"""Entraînement et évaluation des premiers modèles football (championnats européens).

Découpage dans le temps (jamais au hasard) :
- apprentissage : saisons 2005-06 à 2021-22
- réglage       : saison 2022-23 (arrêt de l'apprentissage au bon moment)
- test final    : saisons 2023-24 et suivantes (matchs jamais vus)

Un marché = une ligne classique des bookmakers (ex. plus de 2,5 buts).
Chaque modèle est comparé à une référence naïve (toujours le même taux).

Entrée : Données/Traitées/football_championnats_variables.csv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, etape  # noqa: E402

FIN_APPRENTISSAGE = 2021   # saison 2021-22
SAISON_REGLAGE = 2022      # saison 2022-23

CIBLES = ["total_buts", "total_buts_mt", "total_corners", "total_fautes", "total_cartons_jaunes"]
NON_VARIABLES = set(CIBLES) | {"match", "annee_saison"}

# Nom affiché -> (colonne, ligne)
MARCHES = {
    "Plus de 2,5 buts": ("total_buts", 2.5),
    "Plus de 9,5 corners": ("total_corners", 9.5),
    "Plus de 24,5 fautes": ("total_fautes", 24.5),
    "Plus de 3,5 cartons jaunes": ("total_cartons_jaunes", 3.5),
}


def evaluer(nom, y, proba):
    return {"modèle": nom, "log_loss": log_loss(y, proba),
            "auc": roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5,
            "précision": accuracy_score(y, proba > 0.5)}


def main():
    print("=== ENTRAÎNEMENT FOOTBALL : CHAMPIONNATS ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "football_championnats_variables.csv",
                     parse_dates=["date"], low_memory=False)
    variables = [c for c in df.select_dtypes("number").columns if c not in NON_VARIABLES]
    print(f"{len(variables)} variables")

    for marche, (cible, ligne) in MARCHES.items():
        etape(f"Marché : {marche}")
        d = df.dropna(subset=[cible])
        app = d[d["annee_saison"] <= FIN_APPRENTISSAGE]
        reg = d[d["annee_saison"] == SAISON_REGLAGE]
        test = d[d["annee_saison"] > SAISON_REGLAGE]
        y_app, y_reg, y_test = [(x[cible] > ligne).astype(int) for x in (app, reg, test)]
        print(f"Matchs : apprentissage {len(app)} | réglage {len(reg)} | test {len(test)} | "
              f"taux réel pendant le test : {y_test.mean():.1%}")

        resultats = [evaluer("Référence naïve", y_test, np.full(len(y_test), y_app.mean()))]

        logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.05, max_iter=3000))
        logit.fit(app[variables], y_app)
        resultats.append(evaluer("Régression logistique", y_test,
                                 logit.predict_proba(test[variables])[:, 1]))

        gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=31,
                                 min_child_samples=200, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, reg_lambda=5, verbose=-1)
        gbm.fit(app[variables], y_app, eval_X=(reg[variables],), eval_y=(y_reg,),
                callbacks=[lgb.early_stopping(100, verbose=False)])
        resultats.append(evaluer("LightGBM", y_test, gbm.predict_proba(test[variables])[:, 1]))

        print(pd.DataFrame(resultats).set_index("modèle").round(4).to_string())
        imp = pd.Series(gbm.booster_.feature_importance("gain"), index=variables)
        print("Variables les plus utiles :", ", ".join(imp.sort_values(ascending=False).head(6).index))


if __name__ == "__main__":
    main()
