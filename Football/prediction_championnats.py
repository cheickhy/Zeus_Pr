"""Programme de prédiction des championnats européens : prochains matchs + bilan des prédictions passées.
"""
import io
import sys
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mise_a_jour  # noqa: E402
import nettoyage_championnats  # noqa: E402
import variables_championnats as variables  # noqa: E402
from Sources.outils import DOSSIER_TRAITEES, RACINE, arguments, etape, modele_sauvegarde  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "football_predictions.csv"
URL_PROCHAINS_MATCHS = "https://www.football-data.co.uk/fixtures.csv"
CLE = ["code", "date", "home_team", "away_team"]

# Colonne de sortie -> (nom affiché, cible, ligne)
MARCHES = {
    "proba_plus_2_5_buts": ("+2,5 buts", "total_buts", 2.5),
    "proba_plus_9_5_corners": ("+9,5 corners", "total_corners", 9.5),
    "proba_plus_24_5_fautes": ("+24,5 fautes", "total_fautes", 24.5),
    "proba_plus_3_5_jaunes": ("+3,5 jaunes", "total_cartons_jaunes", 3.5),
}
# Modèles de comptage : nombre attendu + dispersion -> probabilité pour n'importe quelle ligne
COMPTAGES = {"buts": "total_buts", "corners": "total_corners", "fautes": "total_fautes",
             "jaunes": "total_cartons_jaunes"}
NON_VARIABLES = set(variables.CIBLES) | {"match", "annee_saison"}
PARAMS = dict(learning_rate=0.02, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1)


def prochains_matchs():
    """Matchs à venir des championnats suivis, avec l'arbitre quand il est connu."""
    f = pd.read_csv(io.BytesIO(mise_a_jour.telecharger(URL_PROCHAINS_MATCHS)), encoding="utf-8-sig")
    f = f[f["Div"].isin(mise_a_jour.CHAMPIONNATS)]
    return pd.DataFrame({
        "code": f["Div"], "championnat": f["Div"].map(mise_a_jour.CHAMPIONNATS),
        "saison": f"{mise_a_jour.saison_en_cours()}-{(mise_a_jour.saison_en_cours() + 1) % 100:02d}",
        "date": pd.to_datetime(f["Date"], dayfirst=True), "heure": f.get("Time"),
        "home_team": f["HomeTeam"].str.strip(), "away_team": f["AwayTeam"].str.strip(),
        "arbitre": f.get("Referee"),
    })


def entrainer(histo, colonnes, cible, ligne):
    """Règle le nombre d'arbres sur la dernière saison complète, puis réentraîne sur tout."""
    d = histo.dropna(subset=[cible])
    y = (d[cible] > ligne).astype(int)
    derniere = d["annee_saison"].max() - 1
    app, reg = d["annee_saison"] < derniere, d["annee_saison"] == derniere
    essai = lgb.LGBMClassifier(n_estimators=2000, **PARAMS).fit(
        d.loc[app, colonnes], y[app], eval_X=(d.loc[reg, colonnes],), eval_y=(y[reg],),
        callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.LGBMClassifier(n_estimators=max(essai.best_iteration_, 50), **PARAMS).fit(d[colonnes], y)


def entrainer_comptage(histo, colonnes, cible):
    """Nombre moyen attendu (LightGBM, objectif Poisson) + dispersion r de la binomiale négative,
    réglés sur la dernière saison complète, puis réentraînement sur tout l'historique."""
    d = histo.dropna(subset=[cible])
    derniere = d["annee_saison"].max() - 1
    app, reg = d[d["annee_saison"] < derniere], d[d["annee_saison"] == derniere]
    params = {**PARAMS, "objective": "poisson"}
    essai = lgb.LGBMRegressor(n_estimators=3000, **params).fit(
        app[colonnes], app[cible], eval_X=(reg[colonnes],), eval_y=(reg[cible],),
        callbacks=[lgb.early_stopping(100, verbose=False)])
    mu = essai.predict(reg[colonnes])
    nll = lambda lr: -stats.nbinom.logpmf(reg[cible], np.exp(lr), np.exp(lr) / (np.exp(lr) + mu)).sum()
    r = float(np.exp(optimize.minimize_scalar(nll, bounds=(-2, 9), method="bounded").x))
    final = lgb.LGBMRegressor(n_estimators=max(essai.best_iteration_, 50), **params).fit(d[colonnes], d[cible])
    return {"modele": final, "r": r}


def comparer(tableau, titre):
    print(titre)
    lignes = []
    for col, (nom, cible, ligne) in MARCHES.items():
        t = tableau.dropna(subset=[cible, col])
        if len(t) == 0:
            continue
        y = (t[cible] > ligne).astype(int)
        lignes.append({"marché": nom, "matchs": len(t),
                       "bonnes réponses": accuracy_score(y, t[col] > 0.5),
                       "score d'erreur": log_loss(y, t[col], labels=[0, 1])})
    print(pd.DataFrame(lignes).set_index("marché").round(3).to_string())


def bilan(matchs):
    etape("Bilan des prédictions passées")
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée pour l'instant.")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
    cibles = [c for _, c, _ in MARCHES.values()]
    joues = pred.merge(matchs[CLE + cibles], on=CLE, how="inner")
    print(f"Prédictions enregistrées : {len(pred)} | matchs déjà joués : {len(joues)}")
    if len(joues):
        comparer(joues, "Résultats :")
        print("(Pendant les tests 2023-2026 : ~56 % buts, ~55 % corners, ~67 % fautes, ~62 % jaunes. "
              "Il faut plusieurs centaines de matchs pour conclure.)")


def main():
    args, reentrainer = arguments()
    jour = date.fromisoformat(args[0]) if args else date.today()
    demonstration = jour < date.today()
    print(f"=== PRÉDICTIONS FOOTBALL (championnats){' - démonstration du ' + str(jour) if demonstration else ''} ===")

    if not demonstration:
        mise_a_jour.main()
        nettoyage_championnats.main()
    matchs = pd.read_csv(DOSSIER_TRAITEES / "football_championnats.csv", parse_dates=["date"], low_memory=False)
    bilan(matchs)

    etape("Matchs à prédire")
    if demonstration:
        reels = matchs[matchs["date"] == pd.Timestamp(jour)]
        futurs = reels[CLE + ["championnat", "saison", "heure", "arbitre"]]
    else:
        futurs = prochains_matchs()
        futurs = futurs[futurs["date"] >= pd.Timestamp(jour)]
        # Un match déjà commencé ne compte pas comme test réel : on l'écarte
        # (heures du site = heure de Londres ; sans heure connue, on garde le match)
        coup_envoi = pd.to_datetime(futurs["date"].dt.strftime("%Y-%m-%d") + " " + futurs["heure"].fillna("23:59"),
                                    errors="coerce").dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
        commence = coup_envoi <= pd.Timestamp.now(tz="Europe/London")
        if commence.any():
            print(f"Matchs déjà commencés, écartés : {int(commence.sum())}")
        futurs = futurs[~commence]
    if futurs.empty:
        print("Aucun match à prédire.")
        return
    print(f"{len(futurs)} matchs, du {futurs['date'].min().date()} au {futurs['date'].max().date()}")

    etape("Calcul des indices (matchs passés uniquement)")
    premier_jour = futurs["date"].min()
    tout = pd.concat([matchs[matchs["date"] < premier_jour], futurs], ignore_index=True)
    feats = variables.construire(tout)
    colonnes = [c for c in feats.select_dtypes("number").columns if c not in NON_VARIABLES]
    histo = feats[feats["total_buts"].notna()]
    a_predire = futurs[CLE].merge(feats, on=CLE, how="left")

    etape("Modèles (réutilisés s'ils ont moins de 7 jours), puis prédiction")
    sortie = a_predire[CLE + ["championnat"]].copy()
    for col, (nom, cible, ligne) in MARCHES.items():
        modele = modele_sauvegarde(f"football_{col}", colonnes,
                                   lambda cible=cible, ligne=ligne: entrainer(histo, colonnes, cible, ligne),
                                   reutiliser=not (demonstration or reentrainer))
        sortie[col] = modele.predict_proba(a_predire[colonnes])[:, 1].round(3)
    for nom, cible in COMPTAGES.items():
        comptage = modele_sauvegarde(f"football_comptage_{nom}", colonnes,
                                     lambda cible=cible: entrainer_comptage(histo, colonnes, cible),
                                     reutiliser=not (demonstration or reentrainer))
        sortie[f"mu_{nom}"] = comptage["modele"].predict(a_predire[colonnes]).round(3)
        sortie[f"r_{nom}"] = round(comptage["r"], 2)
    sortie["predit_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    affichage = sortie.rename(columns={c: n for c, (n, _, _) in MARCHES.items()})
    affichage["date"] = affichage["date"].dt.strftime("%d/%m")
    print(affichage[["date", "home_team", "away_team"] + [n for n, _, _ in MARCHES.values()]].to_string(index=False))
    print("Probabilité pour une autre ligne : python Football/proba_ligne.py ÉQUIPE MARCHÉ LIGNE"
          "  (ex. Barcelona corners 10.5)")

    if demonstration:
        print()
        comparer(sortie.merge(reels[CLE + [c for _, c, _ in MARCHES.values()]], on=CLE),
                 "Comparaison avec les vrais résultats de ce jour-là :")
        print("\nDémonstration : rien n'a été enregistré.")
        return

    FICHIER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    if FICHIER_PREDICTIONS.exists():
        anciennes = pd.read_csv(FICHIER_PREDICTIONS, parse_dates=["date"])
        deja = anciennes.set_index(CLE).index.isin(sortie.set_index(CLE).index)
        sortie = pd.concat([anciennes[~deja], sortie])
    sortie.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédictions enregistrées dans {FICHIER_PREDICTIONS}")


if __name__ == "__main__":
    main()
