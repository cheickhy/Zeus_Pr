"""Commande quotidienne unique : lance les prédictions de tous les sports, puis l'export pour l'application."""
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import RACINE  # noqa: E402

JOURNAL = RACINE / "Données" / "Prédictions" / "journal"

# (nom affiché, script, mois où la compétition se joue)
ETAPES = [
    ("Basket (NBA)", "Basketball/prediction.py", {10, 11, 12, 1, 2, 3, 4, 5, 6}),
    ("Football (championnats)", "Football/prediction_championnats.py", set(range(1, 13))),
    ("Ligue des champions", "Football/prediction_coupes.py", {9, 10, 11, 12, 1, 2, 3, 4, 5}),
    ("Sélections (CAN, qualifications)", "Football/prediction_selections.py", set(range(1, 13))),
    ("Baseball (MLB)", "Baseball/prediction.py", {3, 4, 5, 6, 7, 8, 9, 10}),
]
# Lignes de la sortie de chaque programme reprises dans le résumé
IMPORTANT = re.compile(r"matchs trouvés|matchs, du|Aucun match|Échec|ÉCHECS|commencés|bons résultats|"
                       r"écart moyen de .* Match|Prédictions enregistrées :|exportés")


def lancer(nom, script, fichier_journal):
    debut = time.time()
    res = subprocess.run([sys.executable, "-u", str(RACINE / script)], cwd=RACINE, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", env={**__import__("os").environ,
                                                                             "PYTHONIOENCODING": "utf-8"})
    with open(fichier_journal, "a", encoding="utf-8") as j:
        j.write(f"\n\n===== {nom} ({script}) =====\n{res.stdout}\n{res.stderr}")
    lignes = [l.strip() for l in res.stdout.splitlines() if IMPORTANT.search(l)]
    erreur = None
    if res.returncode != 0:
        derniere = [l for l in res.stderr.strip().splitlines() if l.strip()]
        erreur = derniere[-1] if derniere else "erreur inconnue"
        if "getaddrinfo" in erreur or "urlopen error" in erreur or "timed out" in erreur:
            erreur = "problème de connexion internet"
    return res.returncode == 0, lignes, erreur, time.time() - debut


def main():
    JOURNAL.mkdir(parents=True, exist_ok=True)
    fichier_journal = JOURNAL / f"journal_{datetime.now():%Y-%m-%d_%H-%M}.txt"
    mois = date.today().month
    print(f"=== LANCEMENT QUOTIDIEN DU {date.today():%d/%m/%Y} ===")
    bilan = []
    for nom, script, saison in ETAPES:
        if mois not in saison:
            bilan.append((nom, "hors saison", []))
            continue
        print(f"\n{nom} en cours...", flush=True)
        ok, lignes, erreur, duree = lancer(nom, script, fichier_journal)
        for l in lignes:
            print(f"   {l}")
        etat = f"OK ({duree / 60:.0f} min)" if ok else f"ÉCHEC : {erreur}"
        print(f"   -> {etat}")
        bilan.append((nom, etat, lignes))

    print("\nExport pour l'application en cours...", flush=True)
    ok, lignes, erreur, _ = lancer("Export", "Sources/export_application.py", fichier_journal)
    for l in lignes:
        print(f"   {l}")
    bilan.append(("Export pour l'application", "OK" if ok else f"ÉCHEC : {erreur}", lignes))

    print("\n=== RÉSUMÉ ===")
    for nom, etat, _ in bilan:
        print(f"{'✓' if etat.startswith(('OK', 'hors')) else '✗'} {nom} : {etat}")
    a_relancer = [nom for nom, etat, _ in bilan if etat.startswith("ÉCHEC")]
    if a_relancer:
        print("\nÀ relancer quand la connexion est revenue : python Sources/tout_lancer.py")
    print(f"Détail complet : {fichier_journal}")


if __name__ == "__main__":
    main()
