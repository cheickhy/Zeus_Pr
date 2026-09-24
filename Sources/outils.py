"""Chemins et fonctions communes au projet.

Les chemins sont calculés à partir de l'emplacement du projet :
le code fonctionne sur n'importe quel poste, sans chemin écrit en dur.
"""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_BRUTES = RACINE / "Données" / "brutes"
DOSSIER_TRAITEES = RACINE / "Données" / "Traitées"

BRUT_MLB = DOSSIER_BRUTES / "mlb_data" / "mlb_data"
BRUT_NBA = DOSSIER_BRUTES / "nba_data" / "nba_data"
BRUT_FOOT = DOSSIER_BRUTES / "_sofascore-all-match-id-data.csv"


def sauvegarder(df, nom_fichier):
    """Enregistre un DataFrame dans Données/Traitées et affiche un résumé."""
    DOSSIER_TRAITEES.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER_TRAITEES / nom_fichier
    df.to_csv(chemin, index=False)
    print(f"Fichier sauvegardé : {chemin}  ({len(df)} lignes, {df.shape[1]} colonnes)")
    return chemin


def etape(message):
    print(f"\n--> {message}")
