"""Prédiction des matchs de Ligue des champions à venir (résultat et cartons jaunes) et bilan des prédictions passées."""
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mise_a_jour_coupes  # noqa: E402
import variables_coupes  # noqa: E402
from entrainement_coupes import dispersion, variables_de  # noqa: E402
from Sources.outils import RACINE, arguments, etape, modele_sauvegarde  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "coupes_predictions.csv"
RESULTATS = ["domicile", "nul", "exterieur"]
# Seuls marchés fiables en test (saisons 2024-2026) : résultat (60 % contre 51 %) et cartons jaunes.
# Buts, corners et fautes ne battent pas la moyenne de la compétition : non prédits.


def entrainer_resultat(histo):
    y = histo["resultat"].map({r: i for i, r in enumerate(RESULTATS)})
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000)).fit(histo[["ldc_elo_ecart"]], y)


def entrainer_jaunes(histo, cols):
    d = histo.dropna(subset=["total_cartons_jaunes"])
    derniere = d["annee_saison"].max() - 1
    app, reg = d[d["annee_saison"] < derniere], d[d["annee_saison"] == derniere]
    params = dict(objective="poisson", learning_rate=0.02, num_leaves=7, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, verbose=-1)
    essai = lgb.LGBMRegressor(n_estimators=2000, **params).fit(
        app[cols], app["total_cartons_jaunes"], eval_X=(reg[cols],), eval_y=(reg["total_cartons_jaunes"],),
        callbacks=[lgb.early_stopping(100, verbose=False)])
    r = dispersion(reg["total_cartons_jaunes"], essai.predict(reg[cols]))
    final = lgb.LGBMRegressor(n_estimators=max(essai.best_iteration_, 30), **params).fit(d[cols], d["total_cartons_jaunes"])
    return {"modele": final, "r": r}


def bilan(feats):
    etape("Bilan des prédictions passées")
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée pour l'instant.")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS, dtype={"espn_id": str})
    joues = pred.merge(feats[["espn_id", "resultat", "total_cartons_jaunes"]].dropna(subset=["resultat"]), on="espn_id")
    print(f"Prédictions enregistrées : {len(pred)} | matchs déjà joués : {len(joues)}")
    if len(joues):
        p = joues[[f"proba_{r}" for r in RESULTATS]].to_numpy()
        y = joues["resultat"].map({r: i for i, r in enumerate(RESULTATS)})
        print(f"Résultat : {accuracy_score(y, p.argmax(axis=1)):.1%} de bons résultats "
              f"(en test : 60 %) | score d'erreur {log_loss(y, p, labels=[0, 1, 2]):.3f}")


def main():
    _, reentrainer = arguments()
    print("=== PRÉDICTIONS : LIGUE DES CHAMPIONS ===")
    mise_a_jour_coupes.main()
    coupes, champ = variables_coupes.charger()
    feats, _ = variables_coupes.construire(coupes, champ)
    bilan(feats)

    etape("Matchs à venir (pas encore commencés)")
    coup_envoi = pd.to_datetime(feats["date"].dt.strftime("%Y-%m-%d") + " " + feats["heure_utc"].fillna("23:59")).dt.tz_localize("UTC")
    futurs = feats[~feats["termine"].astype(bool) & (coup_envoi > pd.Timestamp.now(tz="UTC"))
                   & (coup_envoi < pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=8))]
    if futurs.empty:
        print("Aucun match dans les 8 prochains jours.")
        return
    print(f"{len(futurs)} matchs, du {futurs['date'].min().date()} au {futurs['date'].max().date()}")

    histo = feats[feats["resultat"].notna()]
    cols = variables_de(feats)
    m_res = modele_sauvegarde("coupes_resultat", ["ldc_elo_ecart"], lambda: entrainer_resultat(histo), not reentrainer)
    m_j = modele_sauvegarde("coupes_jaunes", cols, lambda: entrainer_jaunes(histo, cols), not reentrainer)

    sortie = futurs[["espn_id", "date", "heure_utc", "home_team", "away_team", "ldc_elo_dom", "ldc_elo_ext"]].copy()
    probas = m_res.predict_proba(futurs[["ldc_elo_ecart"]])
    for i, r in enumerate(RESULTATS):
        sortie[f"proba_{r}"] = probas[:, i].round(3)
    sortie["mu_jaunes"] = m_j["modele"].predict(futurs[cols]).round(3)
    sortie["r_jaunes"] = round(m_j["r"], 2)
    sortie["predit_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    affichage = sortie.assign(date=sortie["date"].dt.strftime("%d/%m"),
                              victoire_1=(sortie["proba_domicile"] * 100).round().astype(int).astype(str) + " %",
                              nul=(sortie["proba_nul"] * 100).round().astype(int).astype(str) + " %",
                              victoire_2=(sortie["proba_exterieur"] * 100).round().astype(int).astype(str) + " %",
                              jaunes_attendus=sortie["mu_jaunes"].round(1))
    print(affichage[["date", "heure_utc", "home_team", "away_team", "victoire_1", "nul", "victoire_2",
                     "jaunes_attendus"]].to_string(index=False))

    FICHIER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    if FICHIER_PREDICTIONS.exists():
        anciennes = pd.read_csv(FICHIER_PREDICTIONS, dtype={"espn_id": str}, parse_dates=["date"])
        sortie = pd.concat([anciennes[~anciennes["espn_id"].isin(sortie["espn_id"])], sortie])
    sortie.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédictions enregistrées dans {FICHIER_PREDICTIONS}")


if __name__ == "__main__":
    main()
