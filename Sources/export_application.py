"""Export des prédictions des trois sports dans un fichier JSON destiné à l'application."""
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import RACINE  # noqa: E402

DOSSIER = RACINE / "Données" / "Prédictions"
SORTIE = DOSSIER / "predictions_application.json"

# Lignes proposées à l'application pour chaque marché de comptage du football
LIGNES_FOOT = {"buts": np.arange(0.5, 6, 1), "corners": np.arange(6.5, 14, 1),
               "fautes": np.arange(18.5, 31, 1), "jaunes": np.arange(1.5, 8, 1)}
LIGNES_3_MANCHES = np.arange(0.5, 7, 1)
ECART_LIGNES_BASKET = 15        # lignes de (prévu - 15) à (prévu + 15), de 1 point en 1 point


def arrondi(x):
    return None if pd.isna(x) else round(float(x), 3)


def lignes_nbinom(mu, r, lignes):
    """P(total > ligne) avec la binomiale négative (lignes en x,5)."""
    return [{"ligne": float(l), "proba_plus": arrondi(stats.nbinom.sf(int(l), r, r / (r + mu)))} for l in lignes]


def lignes_normales(prevu, sigma):
    """P(total > ligne) avec la loi normale, lignes centrées sur le total prévu."""
    centre = np.floor(prevu) + 0.5
    lignes = np.arange(centre - ECART_LIGNES_BASKET, centre + ECART_LIGNES_BASKET + 1, 1)
    return [{"ligne": float(l), "proba_plus": arrondi(1 - stats.norm.cdf((l - prevu) / sigma))} for l in lignes]


def lire(nom, depuis):
    chemin = DOSSIER / nom
    if not chemin.exists():
        return pd.DataFrame()
    p = pd.read_csv(chemin, parse_dates=["date"])
    if depuis == "dernier_jour":        # exemple : la dernière journée prédite de chaque sport
        return p[p["date"] == p["date"].max()]
    return p[p["date"] >= pd.Timestamp(depuis)]


def football(depuis):
    fiches = []
    for _, m in lire("football_predictions.csv", depuis).iterrows():
        marches = {}
        for nom, lignes in LIGNES_FOOT.items():
            if f"mu_{nom}" in m and pd.notna(m[f"mu_{nom}"]):
                marches[nom] = {"nombre_attendu": arrondi(m[f"mu_{nom}"]),
                                "lignes": lignes_nbinom(m[f"mu_{nom}"], m[f"r_{nom}"], lignes)}
        if not marches:   # anciennes prédictions : une seule ligne par marché
            for col, nom, ligne in [("proba_plus_2_5_buts", "buts", 2.5), ("proba_plus_9_5_corners", "corners", 9.5),
                                    ("proba_plus_24_5_fautes", "fautes", 24.5), ("proba_plus_3_5_jaunes", "jaunes", 3.5)]:
                marches[nom] = {"nombre_attendu": None, "lignes": [{"ligne": ligne, "proba_plus": arrondi(m[col])}]}
        fiches.append({"sport": "football", "competition": m["championnat"], "date": m["date"].date().isoformat(),
                       "domicile": m["home_team"], "exterieur": m["away_team"], "marches": marches,
                       "avertissement": None, "predit_le": m["predit_le"]})
    return fiches


def selections(depuis):
    fiches = []
    for _, m in lire("selections_predictions.csv", depuis).iterrows():
        fiches.append({"sport": "football_selections", "competition": m["tournament"],
                       "date": m["date"].date().isoformat(), "domicile": m["home_team"], "exterieur": m["away_team"],
                       "marches": {"resultat": {"victoire_domicile": arrondi(m["proba_domicile"]),
                                                "nul": arrondi(m["proba_nul"]),
                                                "victoire_exterieur": arrondi(m["proba_exterieur"])},
                                   "buts": {"nombre_attendu": None,
                                            "lignes": [{"ligne": 2.5, "proba_plus": arrondi(m["proba_plus_2_5_buts"])}]}},
                       "avertissement": None, "predit_le": m["predit_le"]})
    return fiches


def basket(depuis):
    cibles = {"total_q1": "total_1er_quart_temps", "total_mi_temps": "total_mi_temps", "total_match": "total_match",
              "equipe_dom_q1": "domicile_1er_quart_temps", "equipe_dom_mi_temps": "domicile_mi_temps",
              "equipe_dom_match": "domicile_match", "equipe_ext_q1": "exterieur_1er_quart_temps",
              "equipe_ext_mi_temps": "exterieur_mi_temps", "equipe_ext_match": "exterieur_match"}
    fiches = []
    for _, m in lire("basketball_predictions.csv", depuis).iterrows():
        marches = {nom: {"points_attendus": arrondi(m[f"prevu_{c}"]), "lignes": lignes_normales(m[f"prevu_{c}"], m[f"sigma_{c}"])}
                   for c, nom in cibles.items() if f"prevu_{c}" in m and pd.notna(m[f"prevu_{c}"])}
        prudence = str(m.get("debut_saison")) == "True"
        fiches.append({"sport": "basket", "competition": "NBA", "date": m["date"].date().isoformat(),
                       "domicile": m["home_team_abbr"], "exterieur": m["away_team_abbr"], "marches": marches,
                       "avertissement": "debut_de_saison" if prudence else None, "predit_le": m["predit_le"]})
    return fiches


def baseball(depuis):
    fiches = []
    for _, m in lire("baseball_predictions.csv", depuis).iterrows():
        marches = {"point_en_1ere_manche": {"lignes": [{"ligne": 0.5, "proba_plus": arrondi(m["proba_point_manche_1"])}]},
                   "total_match": {"lignes": [{"ligne": 8.5, "proba_plus": arrondi(m["proba_plus_8_5"])}]}}
        if "mu_3_manches" in m and pd.notna(m["mu_3_manches"]):
            marches["total_3_premieres_manches"] = {"points_attendus": arrondi(m["mu_3_manches"]),
                                                    "lignes": lignes_nbinom(m["mu_3_manches"], m["r_3_manches"], LIGNES_3_MANCHES)}
        fiches.append({"sport": "baseball", "competition": "MLB", "date": pd.Timestamp(m["date"]).date().isoformat(),
                       "domicile": m["home_team"], "exterieur": m["away_team"],
                       "lanceurs": {"domicile": m["home_principal_nom"], "exterieur": m["away_principal_nom"]},
                       "marches": marches, "avertissement": None, "predit_le": m["predit_le"]})
    return fiches


def main():
    """Par défaut : matchs d'aujourd'hui et à venir. Option --exemple : dernière journée prédite de chaque sport."""
    depuis = "dernier_jour" if "--exemple" in sys.argv else date.today().isoformat()
    matchs = football(depuis) + selections(depuis) + basket(depuis) + baseball(depuis)
    contenu = {"genere_le": datetime.now().strftime("%Y-%m-%d %H:%M"), "version_format": 1,
               "explication": "proba_plus = probabilité que le total dépasse la ligne (ex. ligne 2.5 = 3 buts ou plus). "
                              "La probabilité de « moins » vaut 1 - proba_plus.",
               "nombre_de_matchs": len(matchs), "matchs": matchs}
    SORTIE.write_text(json.dumps(contenu, ensure_ascii=False, indent=2), encoding="utf-8")
    par_sport = pd.Series([m["sport"] for m in matchs]).value_counts().to_dict() if matchs else {}
    print(f"{len(matchs)} matchs exportés {par_sport} -> {SORTIE}")


if __name__ == "__main__":
    main()
