"""Programme quotidien : prédit les matchs MLB du jour et vérifie les prédictions passées.

Une seule commande, à lancer chaque matin (avant les matchs) :
    python Baseball/prediction.py              -> matchs d'aujourd'hui
    python Baseball/prediction.py 2026-09-26   -> matchs d'une autre date

Étapes :
1. Mise à jour des données (matchs joués depuis la dernière fois).
2. Nettoyage.
3. Bilan : les prédictions déjà enregistrées sont comparées aux vrais résultats.
4. Entraînement sur tous les matchs connus, puis prédiction des matchs du jour,
   avec les lanceurs partants annoncés par la MLB.

Les prédictions sont enregistrées AVANT les matchs dans
Données/Prédictions/baseball_predictions.csv : c'est la preuve qu'elles ont été
faites sans connaître le résultat.
"""
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mise_a_jour  # noqa: E402
import nettoyage  # noqa: E402
import variables  # noqa: E402
from Sources.outils import DOSSIER_TRAITEES, RACINE, etape  # noqa: E402

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
    resultats = matchs[["gamePk", "total_manche_1", "total_match"]]
    joues = pred.merge(resultats, on="gamePk", how="inner")
    print(f"Prédictions enregistrées : {len(pred)} | matchs déjà joués : {len(joues)}")
    if len(joues) == 0:
        return
    lignes = []
    for col, (nom, cible) in MARCHES.items():
        y, p = cible(joues), joues[col]
        lignes.append({"marché": nom, "matchs": len(y),
                       "bonnes réponses": accuracy_score(y, p > 0.5),
                       "score d'erreur": log_loss(y, p, labels=[0, 1])})
    print(pd.DataFrame(lignes).set_index("marché").round(3).to_string())
    print("(Pendant les tests 2024-2026 : ~53 % en 1ère manche et ~55 % sur +8,5 ; "
          "score d'erreur ~0,69 et ~0,68. Il faut plusieurs centaines de matchs pour conclure.)")


def main():
    jour = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
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

    etape("Entraînement sur tous les matchs connus, puis prédiction")
    sortie = futurs[["gamePk", "date", "home_team", "away_team",
                     "home_principal_nom", "away_principal_nom"]].copy()
    for col, (nom, cible) in MARCHES.items():
        modele = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=0.05, max_iter=2000))
        modele.fit(histo[colonnes], cible(histo))
        proba = pd.Series(modele.predict_proba(a_predire[colonnes])[:, 1], index=a_predire["gamePk"])
        sortie[col] = sortie["gamePk"].map(proba).round(3)
    sortie["date"] = sortie["date"].dt.date
    sortie["predit_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    affichage = sortie.rename(columns={
        "home_team": "domicile", "away_team": "extérieur",
        "home_principal_nom": "lanceur dom.", "away_principal_nom": "lanceur ext.",
        "proba_point_manche_1": "1 pt manche 1", "proba_plus_8_5": "+8,5 pts"})
    print(affichage.drop(columns=["gamePk", "date", "predit_le"]).to_string(index=False))

    # Enregistrement : une prédiction par match (la plus récente, faite avant le match)
    FICHIER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    if FICHIER_PREDICTIONS.exists():
        anciennes = pd.read_csv(FICHIER_PREDICTIONS)
        sortie = pd.concat([anciennes[~anciennes["gamePk"].isin(sortie["gamePk"])], sortie])
    sortie.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédictions enregistrées dans {FICHIER_PREDICTIONS}")


if __name__ == "__main__":
    main()
