"""Chemins et fonctions communes au projet."""
import sys
from datetime import datetime
from pathlib import Path

import joblib

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_BRUTES = RACINE / "Données" / "brutes"
DOSSIER_TRAITEES = RACINE / "Données" / "Traitées"
DOSSIER_MODELES = RACINE / "Modeles"
VALIDITE_MODELE_JOURS = 7

BRUT_MLB = DOSSIER_BRUTES / "mlb_data" / "mlb_data"
BRUT_NBA = DOSSIER_BRUTES / "nba_data" / "nba_data"
# Ancien fichier SofaScore (sans dates) : isolé, seule source des touches
BRUT_SOFASCORE = DOSSIER_BRUTES / "sofascore" / "_sofascore-all-match-id-data.csv"


def sauvegarder(df, nom_fichier):
    """Enregistre un DataFrame dans Données/Traitées et affiche un résumé."""
    DOSSIER_TRAITEES.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER_TRAITEES / nom_fichier
    df.to_csv(chemin, index=False)
    print(f"Fichier sauvegardé : {chemin}  ({len(df)} lignes, {df.shape[1]} colonnes)")
    return chemin


def etape(message):
    print(f"\n--> {message}")


def arguments():
    """Renvoie (arguments sans option, True si l'option --reentrainer est donnée)."""
    return [a for a in sys.argv[1:] if not a.startswith("--")], "--reentrainer" in sys.argv


def modele_sauvegarde(nom, colonnes, entrainer, reutiliser=True):
    """Recharge le modèle s'il a moins de 7 jours et les mêmes variables, sinon l'entraîne et le sauvegarde.

    reutiliser=False pour une date passée : un modèle sauvegardé aurait appris sur des
    matchs postérieurs à cette date (triche involontaire).
    """
    chemin = DOSSIER_MODELES / f"{nom}.joblib"
    if reutiliser and chemin.exists():
        sauve = joblib.load(chemin)
        age = (datetime.now() - sauve["date"]).days
        if age < VALIDITE_MODELE_JOURS and sauve["colonnes"] == list(colonnes):
            print(f"Modèle « {nom} » réutilisé (entraîné il y a {age} jour(s))")
            return sauve["modele"]
    modele = entrainer()
    if reutiliser:
        DOSSIER_MODELES.mkdir(exist_ok=True)
        joblib.dump({"modele": modele, "colonnes": list(colonnes), "date": datetime.now()}, chemin)
        print(f"Modèle « {nom} » entraîné et sauvegardé")
    return modele


def moyenne_precedente(df, groupe, colonne, fenetre, min_matchs=None):
    """Moyenne de `colonne` sur les `fenetre` lignes PRÉCÉDENTES de chaque groupe.

    Le décalage shift(1) exclut la ligne en cours : le match à prédire
    n'entre jamais dans sa propre moyenne (pas de fuite d'information).
    `df` doit être trié par date à l'intérieur de chaque groupe.
    """
    min_matchs = min_matchs or max(3, fenetre // 3)
    decale = df.groupby(groupe)[colonne].shift(1)
    return (decale.groupby(df[groupe]).rolling(fenetre, min_periods=min_matchs).mean()
            .reset_index(level=0, drop=True))


def somme_precedente(df, groupe, colonne, fenetre, min_matchs=1):
    """Somme de `colonne` sur les `fenetre` lignes PRÉCÉDENTES de chaque groupe."""
    decale = df.groupby(groupe)[colonne].shift(1)
    return (decale.groupby(df[groupe]).rolling(fenetre, min_periods=min_matchs).sum()
            .reset_index(level=0, drop=True))
