
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_BRUTES  # noqa: E402

URL = "https://raw.githubusercontent.com/martj42/international_results/master/{}"
DOSSIER = DOSSIER_BRUTES / "selections"
FICHIERS = ["results.csv",      # un match par ligne
            "shootouts.csv"]    # vainqueur des séances de tirs au but


def main():
    print("=== TÉLÉCHARGEMENT DES MATCHS INTERNATIONAUX ===")
    DOSSIER.mkdir(parents=True, exist_ok=True)
    for nom in FICHIERS:
        requete = urllib.request.Request(URL.format(nom), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(requete, timeout=60) as r:
            contenu = r.read()
        (DOSSIER / nom).write_bytes(contenu)
        print(f"{nom} : {contenu.count(b'\n') - 1} lignes")


if __name__ == "__main__":
    main()
