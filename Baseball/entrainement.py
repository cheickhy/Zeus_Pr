"""Entraînement et évaluation des premiers modèles baseball.

Découpage dans le temps 
- apprentissage : 2008 à 2022
- réglage       : 2023 
- test final    : 2024 à 2026




Entrée : Données/Traitées/baseball_variables.csv (produit par Baseball/variables.py)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, etape  # noqa: E402

FIN_APPRENTISSAGE = 2022
SAISON_REGLAGE = 2023

NON_VARIABLES = {"gamePk", "season", "home_id", "away_id", "home_principal_id",
                 "away_principal_id", "total_manche_1", "point_en_manche_1",
                 "total_3_prem_manches", "total_match"}

# Nom affiché -> (colonne cible, fonction qui crée la cible 0/1)
MARCHES = {
    "Au moins 1 point en 1ère manche": lambda d: (d["total_manche_1"] > 0).astype(int),
    "Plus de 8,5 points sur le match": lambda d: (d["total_match"] > 8.5).astype(int),
}


def evaluer(nom, y, proba):
    return {
        "modèle": nom,
        "log_loss": log_loss(y, proba),
        "brier": brier_score_loss(y, proba),
        "auc": roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5,
        "précision": accuracy_score(y, proba > 0.5),
    }


def calibration(y, proba, nb=5):
    """Compare les probabilités annoncées aux fréquences réellement observées."""
    tranches = pd.qcut(proba, nb, duplicates="drop")
    return (pd.DataFrame({"annoncé": proba, "observé": y})
            .groupby(tranches, observed=True).agg(["mean", "size"])
            .iloc[:, [0, 2, 3]].set_axis(["proba_annoncée", "fréquence_réelle", "nb_matchs"], axis=1)
            .round(3))


def main():
    print("=== ENTRAÎNEMENT BASEBALL ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "baseball_variables.csv", parse_dates=["date"])
    variables = [c for c in df.select_dtypes("number").columns if c not in NON_VARIABLES]
    app = df[df["season"] <= FIN_APPRENTISSAGE]
    reg = df[df["season"] == SAISON_REGLAGE]
    test = df[df["season"] > SAISON_REGLAGE]
    print(f"{len(variables)} variables | apprentissage {len(app)} matchs | "
          f"réglage {len(reg)} | test {len(test)}")

    for marche, faire_cible in MARCHES.items():
        etape(f"Marché : {marche}")
        y_app, y_reg, y_test = faire_cible(app), faire_cible(reg), faire_cible(test)
        resultats = []

        # 1. Référence naïve : toujours le même taux
        taux = y_app.mean()
        resultats.append(evaluer("Référence naïve", y_test, np.full(len(y_test), taux)))

        # 2. Régression logistique : modèle simple et robuste
        logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.05, max_iter=2000))
        logit.fit(app[variables], y_app)
        resultats.append(evaluer("Régression logistique", y_test,
                                 logit.predict_proba(test[variables])[:, 1]))

        # 3. LightGBM : capte les interactions entre variables
        gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                                 min_child_samples=200, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, reg_lambda=5, verbose=-1)
        gbm.fit(app[variables], y_app, eval_X=(reg[variables],), eval_y=(y_reg,),
                callbacks=[lgb.early_stopping(100, verbose=False)])
        proba_gbm = gbm.predict_proba(test[variables])[:, 1]
        resultats.append(evaluer("LightGBM", y_test, proba_gbm))

        print(f"Taux réel pendant l'apprentissage : {taux:.1%} | pendant le test : {y_test.mean():.1%}")
        print(pd.DataFrame(resultats).set_index("modèle").round(4).to_string())

        print("\nCalibration LightGBM (la proba annoncée doit ≈ la fréquence réelle) :")
        print(calibration(y_test.to_numpy(), proba_gbm).to_string())

        imp = pd.Series(gbm.booster_.feature_importance("gain"), index=variables)
        print("\nVariables les plus utiles :", ", ".join(imp.sort_values(ascending=False).head(8).index))


if __name__ == "__main__":
    main()
