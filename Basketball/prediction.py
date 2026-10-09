import sys
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mise_a_jour  # noqa: E402
import nettoyage  # noqa: E402
import variables  # noqa: E402
from Sources.outils import DOSSIER_TRAITEES, RACINE, arguments, etape, modele_sauvegarde  # noqa: E402

FICHIER_PREDICTIONS = RACINE / "Données" / "Prédictions" / "basketball_predictions.csv"
CIBLES = {"total_q1": "1er quart-temps", "total_mi_temps": "Mi-temps", "total_match": "Match"}
MATCHS_AVANT_FIABILITE = 5
NON_VARIABLES = {"game_id", "home_team_id", "away_team_id", "annee_saison"} | set(CIBLES)
PARAMS = dict(learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1)


def saison_de(jour):
    """2026-10-21 -> "2026-27" ; 2026-03-01 -> "2025-26"."""
    annee = jour.year if jour.month >= 10 else jour.year - 1
    return f"{annee}-{(annee + 1) % 100:02d}"


def matchs_a_venir(jour):
    """Matchs de saison régulière prévus ce jour-là, d'après le calendrier officiel."""
    saison = saison_de(jour)
    cal = mise_a_jour.telecharger(f"{mise_a_jour.API}/scheduleleaguev2?LeagueID=00&Season={saison}")
    lignes = []
    for jour_cal in cal["leagueSchedule"]["gameDates"]:
        if pd.to_datetime(jour_cal["gameDate"]).date() != jour:
            continue
        for g in jour_cal["games"]:
            if not g["gameId"].startswith("002"):   # 002 = saison régulière
                continue
            lignes.append({"game_id": int(g["gameId"]), "date": pd.Timestamp(jour), "season": saison,
                           "home_team_id": g["homeTeam"]["teamId"], "away_team_id": g["awayTeam"]["teamId"],
                           "home_team_abbr": g["homeTeam"]["teamTricode"],
                           "away_team_abbr": g["awayTeam"]["teamTricode"]})
    return pd.DataFrame(lignes)


def entrainer(histo, colonnes, cible):
    """Règle le nombre d'arbres sur la dernière saison, puis réentraîne sur tout l'historique."""
    derniere = histo["annee_saison"].max()
    app, reg = histo[histo["annee_saison"] < derniere], histo[histo["annee_saison"] == derniere]
    essai = lgb.LGBMRegressor(n_estimators=3000, **PARAMS)
    essai.fit(app[colonnes], app[cible], eval_X=(reg[colonnes],), eval_y=(reg[cible],),
              eval_metric="l1", callbacks=[lgb.early_stopping(100, verbose=False)])
    # Erreur habituelle du modèle, mesurée sur la saison qu'il n'a pas vue : sert à calculer
    # la probabilité de dépasser n'importe quelle ligne (les erreurs suivent une loi normale)
    sigma = float((reg[cible] - essai.predict(reg[colonnes])).std())
    final = lgb.LGBMRegressor(n_estimators=max(essai.best_iteration_, 50), **PARAMS)
    return {"modele": final.fit(histo[colonnes], histo[cible]), "sigma": sigma}


def comparer(tableau, titre):
    """Écart moyen entre prédictions et résultats réels."""
    print(titre)
    for cible, nom in CIBLES.items():
        ecart = (tableau[f"prevu_{cible}"] - tableau[cible]).abs().mean()
        print(f"  {nom:16s} : écart moyen de {ecart:.1f} points sur {len(tableau)} matchs")


def bilan(matchs):
    etape("Bilan des prédictions passées")
    if not FICHIER_PREDICTIONS.exists():
        print("Aucune prédiction enregistrée pour l'instant.")
        return
    pred = pd.read_csv(FICHIER_PREDICTIONS)
    joues = pred.merge(matchs[["game_id"] + list(CIBLES)], on="game_id", how="inner")
    print(f"Prédictions enregistrées : {len(pred)} | matchs déjà joués : {len(joues)}")
    if len(joues):
        comparer(joues, "Résultats :")
        print("(Pendant les tests 2023-2026 : environ 6,6 / 9,8 / 15,1 points d'écart.)")


def main():
    args, reentrainer = arguments()
    jour = date.fromisoformat(args[0]) if args else date.today()
    demonstration = jour < date.today()
    print(f"=== PRÉDICTIONS BASKET DU {jour}{' (démonstration)' if demonstration else ''} ===")

    mise_a_jour.main()
    nettoyage.main()
    matchs = pd.read_csv(DOSSIER_TRAITEES / "basketball_matchs.csv", parse_dates=["date"])
    bilan(matchs)

    etape(f"Matchs prévus le {jour}")
    if demonstration:
        reels = matchs[matchs["date"] == pd.Timestamp(jour)]
        futurs = reels[["game_id", "date", "season", "home_team_id", "away_team_id",
                        "home_team_abbr", "away_team_abbr"]]
    else:
        futurs = matchs_a_venir(jour)
    if futurs.empty:
        print("Aucun match de saison régulière à cette date.")
        return
    print(f"{len(futurs)} matchs trouvés")

    etape("Calcul des indices (matchs passés uniquement)")
    tout = pd.concat([matchs[matchs["date"] < pd.Timestamp(jour)], futurs], ignore_index=True)
    feats = variables.construire(tout)
    colonnes = [c for c in feats.select_dtypes("number").columns if c not in NON_VARIABLES]
    histo = feats[feats["total_match"].notna()]
    a_predire = feats[feats["game_id"].isin(futurs["game_id"])].set_index("game_id")

    etape("Modèles (réutilisés s'ils ont moins de 7 jours), puis prédiction")
    sortie = futurs[["game_id", "date", "home_team_abbr", "away_team_abbr"]].copy()
    for cible in CIBLES:
        modele = modele_sauvegarde(f"basket_{cible}", colonnes,
                                   lambda cible=cible: entrainer(histo, colonnes, cible),
                                   reutiliser=not (demonstration or reentrainer))
        prevu = pd.Series(modele["modele"].predict(a_predire[colonnes]), index=a_predire.index)
        sortie[f"prevu_{cible}"] = sortie["game_id"].map(prevu).round(1)
        sortie[f"sigma_{cible}"] = round(modele["sigma"], 1)
    # Mesuré sur 2023-2026 : pendant les 4 premiers matchs d'une équipe, le modèle ne fait
    # presque pas mieux que la moyenne de la ligue (les effectifs ont changé pendant l'été)
    deja_joues = a_predire[["home_matchs_joues_saison", "away_matchs_joues_saison"]].min(axis=1)
    sortie["debut_saison"] = sortie["game_id"].map(deja_joues < MATCHS_AVANT_FIABILITE).astype(bool)
    sortie["date"] = pd.to_datetime(sortie["date"]).dt.date
    sortie["predit_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    affichage = sortie.rename(columns={
        "home_team_abbr": "domicile", "away_team_abbr": "extérieur",
        "prevu_total_q1": "1er QT", "prevu_total_mi_temps": "Mi-temps", "prevu_total_match": "Match"})
    affichage["fiabilité"] = affichage["debut_saison"].map({True: "prudence", False: ""})
    print(affichage[["domicile", "extérieur", "1er QT", "Mi-temps", "Match", "fiabilité"]].to_string(index=False))
    if sortie["debut_saison"].any():
        print(f"« prudence » : une des équipes a joué moins de {MATCHS_AVANT_FIABILITE} matchs cette saison, "
              "le modèle connaît mal sa forme actuelle.")
    print("Probabilité pour une ligne de bookmaker : python Basketball/proba_ligne.py ÉQUIPE MARCHÉ LIGNE"
          "  (ex. BOS match 220.5)")

    if demonstration:
        print()
        comparer(sortie.merge(reels[["game_id"] + list(CIBLES)], on="game_id"),
                 "Comparaison avec les vrais résultats de ce jour-là :")
        print("\nDémonstration : rien n'a été enregistré.")
        return

    FICHIER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    if FICHIER_PREDICTIONS.exists():
        anciennes = pd.read_csv(FICHIER_PREDICTIONS)
        sortie = pd.concat([anciennes[~anciennes["game_id"].isin(sortie["game_id"])], sortie])
    sortie.to_csv(FICHIER_PREDICTIONS, index=False)
    print(f"\nPrédictions enregistrées dans {FICHIER_PREDICTIONS}")


if __name__ == "__main__":
    main()
