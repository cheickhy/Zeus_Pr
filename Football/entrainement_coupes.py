"""Test des modèles de comptage de la Ligue des champions (buts, corners, fautes, cartons jaunes)."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, etape  # noqa: E402

FIN_APPRENTISSAGE, SAISON_REGLAGE = 2022, 2023      # test : saisons 2024-25 et suivantes
LIGNES = {"total_buts": [1.5, 2.5, 3.5], "total_corners": [8.5, 9.5, 10.5],
          "total_fautes": [19.5, 22.5, 25.5], "total_cartons_jaunes": [2.5, 3.5, 4.5]}


def variables_de(df):
    return [c for c in df.columns if c.startswith(("home_champ_", "away_champ_", "home_ldc_", "away_ldc_", "ldc_tendance_",
                                                   "ldc_elo_")) or c == "mois"]


def dispersion(y, mu):
    nll = lambda lr: -stats.nbinom.logpmf(y, np.exp(lr), np.exp(lr) / (np.exp(lr) + mu)).sum()
    return float(np.exp(optimize.minimize_scalar(nll, bounds=(-2, 9), method="bounded").x))


def test_resultat(v, cols):
    """Victoire domicile / nul / victoire extérieur : référence naïve, Elo seul, Elo + forme."""
    from sklearn.linear_model import LogisticRegression
    ordre = {"domicile": 0, "nul": 1, "exterieur": 2}
    d = v.dropna(subset=["resultat"])
    app, test = d[d.annee_saison <= SAISON_REGLAGE], d[d.annee_saison > SAISON_REGLAGE]
    y_app, y = app["resultat"].map(ordre), test["resultat"].map(ordre)
    etape(f"Résultat du match : apprentissage {len(app)} | test {len(test)} matchs")
    freq = np.bincount(y_app, minlength=3) / len(y_app)
    print(f"  Référence naïve : {accuracy_score(y, np.full(len(y), freq.argmax())):.1%} / "
          f"{log_loss(y, np.tile(freq, (len(y), 1)), labels=[0, 1, 2]):.3f}")
    for nom, c in [("Elo seul", ["ldc_elo_ecart"]), ("Elo + forme", cols)]:
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=3000)).fit(app[c], y_app)
        p = m.predict_proba(test[c])
        print(f"  {nom:15s} : {accuracy_score(y, p.argmax(axis=1)):.1%} / {log_loss(y, p, labels=[0, 1, 2]):.3f}")


def main():
    print("=== TEST DES MODÈLES : LIGUE DES CHAMPIONS ===")
    v = pd.read_csv(DOSSIER_TRAITEES / "coupes_variables.csv", low_memory=False)
    cols = variables_de(v)
    test_resultat(v, cols)
    for cible, lignes in LIGNES.items():
        d = v.dropna(subset=[cible])
        app, reg, test = (d[d.annee_saison <= FIN_APPRENTISSAGE], d[d.annee_saison == SAISON_REGLAGE],
                          d[d.annee_saison > SAISON_REGLAGE])
        etape(f"{cible} : apprentissage {len(app)} | réglage {len(reg)} | test {len(test)} matchs")
        glm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            PoissonRegressor(alpha=1.0, max_iter=2000)).fit(app[cols], app[cible])
        gbm = lgb.LGBMRegressor(objective="poisson", n_estimators=2000, learning_rate=0.02, num_leaves=7,
                                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                reg_lambda=10, verbose=-1).fit(app[cols], app[cible], eval_X=(reg[cols],), eval_y=(reg[cible],),
                                                               callbacks=[lgb.early_stopping(100, verbose=False)])
        y = test[cible].to_numpy()
        modeles = {"Modèle simple": glm, "LightGBM": gbm}
        for L in lignes:
            yb = (y > L).astype(int)
            naif = np.clip(np.full(len(yb), (app[cible] > L).mean()), 1e-4, 1 - 1e-4)
            texte = f"  +{L} (réel {yb.mean():.0%}) | naïf {accuracy_score(yb, naif > .5):.1%} / {log_loss(yb, naif):.3f}"
            for nom, m in modeles.items():
                r = dispersion(reg[cible], m.predict(reg[cols]))
                p = np.clip(stats.nbinom.sf(int(L), r, r / (r + m.predict(test[cols]))), 1e-4, 1 - 1e-4)
                texte += f" | {nom} {accuracy_score(yb, p > .5):.1%} / {log_loss(yb, p):.3f}"
            print(texte)


if __name__ == "__main__":
    main()
