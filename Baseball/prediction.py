"""Programme quotidien : prédit les matchs MLB du jour et vérifie les prédictions passées."""
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mise_a_jour  
import nettoyage  
import variables  
from Sources.outils import DOSSIER_TRAITEES, RACINE, arguments, etape, modele_sauvegarde  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "baseball_predictions.csv"

NON_VARIABLES = {"gamePk", "season", "home_id", "away_id", "home_principal_id",
                 "away_principal_id", "total_manche_1", "point_en_manche_1",
                 "total_3_prem_manches", "total_match"}

# Nom de colonne -> (description, fonction qui crée la cible 0/1 à partir des matchs)
MARCHES = {
    "proba_point_manche_1": ("Au moins 1 point en 1ère manche",
                             lambda d: (d["total_manche_1"] > 0).astype(int)),
    "proba_plus_8_5": ("Plus de 8,5 points sur le match",
                       lambda d: (d["total_match"] > 8.5).astype(int)),
}
# Marché calculé par le modèle de comptage (binomiale négative), utilisé aussi pour le bilan
MARCHE_3_MANCHES = {"proba_3m_plus_2_5": ("Plus de 2,5 points sur les 3 premières manches",
                                          lambda d: (d["total_3_prem_manches"] > 2.5).astype(int))}


def entrainer_3_manches(histo, colonnes):
    """Nombre moyen de points attendu (Poisson) + dispersion r réglée sur la dernière saison.

    La binomiale négative tient compte du fait que les points varient plus que la moyenne :
    elle donne la probabilité de dépasser N'IMPORTE QUELLE ligne avec un seul modèle.
    """
    pipe = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 PoissonRegressor(alpha=1e-3, max_iter=1000))
    cible = "total_3_prem_manches"
    derniere = histo["season"].max()
    app, reg = histo[histo["season"] < derniere], histo[histo["season"] == derniere]
    mu = pipe().fit(app[colonnes], app[cible]).predict(reg[colonnes])
    nll = lambda lr: -stats.nbinom.logpmf(reg[cible], np.exp(lr), np.exp(lr) / (np.exp(lr) + mu)).sum()
    r = float(np.exp(optimize.minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))
    return {"modele": pipe().fit(histo[colonnes], histo[cible]), "r": r}


def matchs_a_venir(jour):
    """Matchs du jour avec les lanceurs partants annoncés, au format de baseball_matchs.csv."""
    cal = mise_a_jour.telecharger(
        f"{mise_a_jour.API}/schedule?sportId=1&date={jour}"
        "&hydrate=probablePitcher,venue,weather")
    lignes = []
    for d in cal["dates"]:
        for g in d["games"]:
            if g["status"]["abstractGameState"] == "Final" or g["status"]["detailedState"] == "Postponed":
                continue
            ligne = mise_a_jour.ligne_calendrier(g)
            for camp in ["home", "away"]:
                lanceur = g["teams"][camp].get("probablePitcher", {})
                ligne[f"{camp}_principal_id"] = lanceur.get("id")
                ligne[f"{camp}_principal_nom"] = lanceur.get("fullName", "inconnu")
            lignes.append(ligne)
    futurs = pd.DataFrame(lignes)
    if len(futurs):
        futurs["date"] = pd.to_datetime(futurs["date"])
        futurs[["home_score", "away_score"]] = np.nan
    return futurs


def bilan(matchs):
    """Compare les prédictions enregistrées aux résultats désormais connus."""
    etape("Bilan des prédictions passées")
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée pour l'instant.")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS)
    resultats = matchs[["gamePk", "total_manche_1", "total_3_prem_manches", "total_match"]]
    joues = pred.merge(resultats, on="gamePk", how="inner")
    print(f"Prédictions enregistrées : {len(pred)} | matchs déjà joués : {len(joues)}")
    if len(joues) == 0:
        return
    lignes = []
    for col, (nom, cible) in {**MARCHES, **MARCHE_3_MANCHES}.items():
        if col not in joues:          # prédictions enregistrées avant l'ajout de ce marché
            continue
        t = joues.dropna(subset=[col])
        y, p = cible(t), t[col]
        lignes.append({"marché": nom, "matchs": len(y),
                       "bonnes réponses": accuracy_score(y, p > 0.5),
                       "score d'erreur": log_loss(y, p, labels=[0, 1])})
    print(pd.DataFrame(lignes).set_index("marché").round(3).to_string())
    print("(Pendant les tests 2024-2026 : ~53 % en 1ère manche et ~55 % sur +8,5 ; "
          "score d'erreur ~0,69 et ~0,68. Il faut plusieurs centaines de matchs pour conclure.)")


def main():
    args, reentrainer = arguments()
    jour = args[0] if args else date.today().isoformat()
    reutiliser = not reentrainer and jour >= date.today().isoformat()
    print(f"=== PRÉDICTIONS BASEBALL DU {jour} ===")

    mise_a_jour.main()
    nettoyage.main()
    matchs = pd.read_csv(DOSSIER_TRAITEES / "baseball_matchs.csv", parse_dates=["date"])
    bilan(matchs)

    etape(f"Matchs prévus le {jour}")
    futurs = matchs_a_venir(jour)
    if futurs.empty:
        print("Aucun match à prédire à cette date.")
        return
    print(f"{len(futurs)} matchs trouvés")

    etape("Calcul des indices (matchs passés uniquement)")
    tout = pd.concat([matchs[matchs["date"] < pd.Timestamp(jour)],
                      futurs.drop(columns=["home_principal_nom", "away_principal_nom"])],
                     ignore_index=True)
    feats = variables.construire(tout)
    colonnes = [c for c in feats.select_dtypes("number").columns if c not in NON_VARIABLES]
    histo = feats[feats["total_match"].notna()]
    a_predire = feats[feats["gamePk"].isin(futurs["gamePk"])]

    etape("Modèles (réutilisés s'ils ont moins de 7 jours), puis prédiction")
    sortie = futurs[["gamePk", "date", "home_team", "away_team",
                     "home_principal_nom", "away_principal_nom"]].copy()
    for col, (nom, cible) in MARCHES.items():
        def entrainer(cible=cible):
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=0.05, max_iter=2000)).fit(histo[colonnes], cible(histo))
        modele = modele_sauvegarde(f"baseball_{col}", colonnes, entrainer, reutiliser)
        proba = pd.Series(modele.predict_proba(a_predire[colonnes])[:, 1], index=a_predire["gamePk"])
        sortie[col] = sortie["gamePk"].map(proba).round(3)
    nb = modele_sauvegarde("baseball_3_manches", colonnes, lambda: entrainer_3_manches(histo, colonnes), reutiliser)
    mu = pd.Series(nb["modele"].predict(a_predire[colonnes]), index=a_predire["gamePk"])
    sortie["mu_3_manches"] = sortie["gamePk"].map(mu).round(3)
    sortie["r_3_manches"] = round(nb["r"], 3)
    r = nb["r"]
    sortie["proba_3m_plus_2_5"] = stats.nbinom.sf(2, r, r / (r + sortie["mu_3_manches"])).round(3)
    sortie["date"] = sortie["date"].dt.date
    sortie["predit_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    affichage = sortie.rename(columns={
        "home_team": "domicile", "away_team": "extérieur",
        "home_principal_nom": "lanceur dom.", "away_principal_nom": "lanceur ext.",
        "proba_point_manche_1": "1 pt manche 1", "proba_plus_8_5": "+8,5 pts",
        "proba_3m_plus_2_5": "3 manches +2,5"})
    print(affichage.drop(columns=["gamePk", "date", "predit_le", "mu_3_manches", "r_3_manches"]).to_string(index=False))
    print("Probabilité pour une autre ligne des 3 premières manches : "
          "python Baseball/proba_ligne.py ÉQUIPE LIGNE  (ex. Yankees 3.5)")

    # Enregistrement : une prédiction par match (la plus récente, faite avant le match)
    FICHIER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    if FICHIER_PREDICTIONS.exists():
        anciennes = pd.read_csv(FICHIER_PREDICTIONS)
        sortie = pd.concat([anciennes[~anciennes["gamePk"].isin(sortie["gamePk"])], sortie])
    sortie.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédictions enregistrées dans {FICHIER_PREDICTIONS}")


if __name__ == "__main__":
    main()
