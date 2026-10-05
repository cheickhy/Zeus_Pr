"""Entraînement et évaluation des modèles des sélections nationales (résultat et buts).

Le modèle apprend sur TOUS les matchs internationaux (plus de données), puis on mesure
sa fiabilité sur les matchs qui intéressent le projet : ceux des sélections africaines,
la CAN, et les qualifications de la CAN jouées pendant la trêve de septembre-octobre 2026.

Découpage dans le temps :
- apprentissage : 2000 à 2019 (le classement Elo démarre en 1990 et se stabilise d'abord)
- réglage       : 2020 et 2021
- test final    : 2022 à aujourd'hui (dont la CAN 2024 et la CAN 2025 au Maroc)

Entrée : Données/Traitées/football_selections_variables.csv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, etape  # noqa: E402

DEBUT, FIN_APPRENTISSAGE, FIN_REGLAGE = 2000, 2019, 2021
NON_VARIABLES = {"home_score", "away_score", "total_buts", "match", "annee"}
RESULTATS = ["domicile", "nul", "exterieur"]   # domicile = première équipe nommée

# Nom affiché -> fonction qui crée la cible
MARCHES = {
    "Résultat (1 N 2)": lambda d: d["resultat"].map({r: i for i, r in enumerate(RESULTATS)}),
    "Plus de 1,5 buts": lambda d: (d["total_buts"] > 1.5).astype(int),
    "Plus de 2,5 buts": lambda d: (d["total_buts"] > 2.5).astype(int),
}


def groupes_de_test(test):
    africain = (test["home_africain"] == 1) | (test["away_africain"] == 1)
    return {
        "Matchs des sélections africaines": africain,
        "Matchs de CAN": test["can"] == 1,
        "Qualifs CAN sept.-oct. 2026": (test["tournament"] == "African Cup of Nations qualification")
                                       & (test["date"] >= "2026-09-01"),
    }


def main():
    print("=== ENTRAÎNEMENT FOOTBALL : SÉLECTIONS ===")
    df = pd.read_csv(DOSSIER_TRAITEES / "football_selections_variables.csv", parse_dates=["date"])
    df = df[df["annee"] >= DEBUT]
    variables = [c for c in df.select_dtypes("number").columns if c not in NON_VARIABLES]
    app = df[df["annee"] <= FIN_APPRENTISSAGE]
    reg = df[(df["annee"] > FIN_APPRENTISSAGE) & (df["annee"] <= FIN_REGLAGE)]
    test = df[df["annee"] > FIN_REGLAGE]
    print(f"{len(variables)} variables | apprentissage {len(app)} | réglage {len(reg)} | test {len(test)} matchs")

    for marche, cible in MARCHES.items():
        etape(f"Marché : {marche}")
        y_app, y_reg, y_test = cible(app), cible(reg), cible(test)
        multi = marche.startswith("Résultat")

        frequences = np.bincount(y_app, minlength=3 if multi else 2) / len(y_app)
        naif = np.tile(frequences if multi else frequences[1], (len(test), 1) if multi else len(test))

        logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=3000))
        logit.fit(app[variables], y_app)

        gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                                 min_child_samples=100, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, reg_lambda=5, verbose=-1)
        gbm.fit(app[variables], y_app, eval_X=(reg[variables],), eval_y=(y_reg,),
                callbacks=[lgb.early_stopping(100, verbose=False)])

        probas = {"Référence naïve": naif,
                  "Régression logistique": logit.predict_proba(test[variables]),
                  "LightGBM": gbm.predict_proba(test[variables])}
        if not multi:
            probas = {k: (v if v.ndim == 1 else v[:, 1]) for k, v in probas.items()}

        lignes = []
        for groupe, masque in groupes_de_test(test).items():
            m = masque.to_numpy()
            if m.sum() == 0:
                continue
            for modele, p in probas.items():
                pg, yg = p[m], y_test[m]
                choix = pg.argmax(axis=1) if multi else (pg > 0.5).astype(int)
                lignes.append({"groupe": groupe, "matchs": int(m.sum()), "modèle": modele,
                               "bonnes réponses": accuracy_score(yg, choix),
                               "score d'erreur": log_loss(yg, pg, labels=[0, 1, 2] if multi else [0, 1])})
        print(pd.DataFrame(lignes).set_index(["groupe", "matchs", "modèle"]).round(3).to_string())


if __name__ == "__main__":
    main()
