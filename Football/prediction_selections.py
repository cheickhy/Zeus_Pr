"""Prédiction des matchs à venir des sélections (CAN, qualifications, amicaux) et bilan des prédictions passées."""
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mise_a_jour_espn  # noqa: E402
import mise_a_jour_selections  # noqa: E402
import nettoyage_selections  # noqa: E402
import variables_selections as variables  # noqa: E402
from Sources.outils import DOSSIER_TRAITEES, RACINE, arguments, etape, modele_sauvegarde  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "selections_predictions.csv"
CLE = ["date", "home_team", "away_team"]
DEBUT_APPRENTISSAGE = 2000
NON_VARIABLES = {"home_score", "away_score", "total_buts", "match", "annee"}
RESULTATS = ["domicile", "nul", "exterieur"]


def entrainer_resultat(histo, colonnes):
    y = histo["resultat"].map({r: i for i, r in enumerate(RESULTATS)})
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000)).fit(histo[colonnes], y)


def entrainer_buts(histo, colonnes):
    y = (histo["total_buts"] > 2.5).astype(int)
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                              verbose=-1).fit(histo[colonnes], y)


def bilan(matchs):
    etape("Bilan des prédictions passées")
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée pour l'instant.")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
    joues = pred.merge(matchs[CLE + ["resultat", "total_buts"]], on=CLE, how="inner")
    print(f"Prédictions enregistrées : {len(pred)} | matchs déjà joués : {len(joues)}")
    if len(joues) == 0:
        return
    p = joues[[f"proba_{r}" for r in RESULTATS]].to_numpy()
    y = joues["resultat"].map({r: i for i, r in enumerate(RESULTATS)})
    yb = (joues["total_buts"] > 2.5).astype(int)
    print(f"Résultat (1 N 2) : {accuracy_score(y, p.argmax(axis=1)):.1%} de bonnes réponses "
          f"| score d'erreur {log_loss(y, p, labels=[0, 1, 2]):.3f}")
    print(f"+2,5 buts        : {accuracy_score(yb, joues['proba_plus_2_5_buts'] > 0.5):.1%} de bonnes réponses "
          f"| score d'erreur {log_loss(yb, joues['proba_plus_2_5_buts'], labels=[0, 1]):.3f}")
    print("(En test sur les qualifs CAN de sept.-oct. 2026 : ~68 % pour le résultat, ~60 % pour +2,5 buts.)")


def main():
    _, reentrainer = arguments()
    print("=== PRÉDICTIONS FOOTBALL : SÉLECTIONS ===")
    mise_a_jour_selections.main()
    mise_a_jour_espn.main()
    nettoyage_selections.main()
    matchs = pd.read_csv(DOSSIER_TRAITEES / "football_selections.csv", parse_dates=["date"])
    bilan(matchs)

    etape("Matchs à venir")
    chemin = DOSSIER_TRAITEES / "football_selections_a_venir.csv"
    futurs = pd.read_csv(chemin, parse_dates=["date"]) if chemin.exists() else pd.DataFrame()
    if len(futurs):
        futurs = futurs[futurs["date"] >= pd.Timestamp.today().normalize()]
    if futurs.empty:
        print("Aucun match à venir connu pour l'instant.")
        return
    print(f"{len(futurs)} matchs, du {futurs['date'].min().date()} au {futurs['date'].max().date()}")

    etape("Calcul des indices (classement Elo et forme, matchs passés uniquement)")
    feats = variables.construire(pd.concat([matchs, futurs], ignore_index=True))
    colonnes = [c for c in feats.select_dtypes("number").columns if c not in NON_VARIABLES]
    histo = feats[feats["resultat"].notna() & (feats["annee"] >= DEBUT_APPRENTISSAGE)]
    a_predire = futurs[CLE + ["tournament"]].merge(feats, on=CLE + ["tournament"], how="left")

    etape("Modèles (réutilisés s'ils ont moins de 7 jours), puis prédiction")
    reutiliser = not reentrainer
    m_res = modele_sauvegarde("selections_resultat", colonnes, lambda: entrainer_resultat(histo, colonnes), reutiliser)
    m_buts = modele_sauvegarde("selections_plus_2_5_buts", colonnes, lambda: entrainer_buts(histo, colonnes), reutiliser)
    sortie = a_predire[CLE + ["tournament"]].copy()
    probas = m_res.predict_proba(a_predire[colonnes])
    for i, r in enumerate(RESULTATS):
        sortie[f"proba_{r}"] = probas[:, i].round(3)
    sortie["proba_plus_2_5_buts"] = m_buts.predict_proba(a_predire[colonnes])[:, 1].round(3)
    sortie["predit_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    affichage = sortie.rename(columns={"proba_domicile": "victoire 1", "proba_nul": "nul",
                                       "proba_exterieur": "victoire 2", "proba_plus_2_5_buts": "+2,5 buts"})
    affichage["date"] = affichage["date"].dt.strftime("%d/%m")
    print(affichage.drop(columns=["tournament", "predit_le"]).to_string(index=False))

    FICHIER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    if FICHIER_PREDICTIONS.exists():
        anciennes = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
        deja = anciennes.set_index(CLE).index.isin(sortie.set_index(CLE).index)
        sortie = pd.concat([anciennes[~deja], sortie])
    sortie.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédictions enregistrées dans {FICHIER_PREDICTIONS}")


if __name__ == "__main__":
    main()
