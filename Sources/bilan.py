"""Bilan détaillé : pour chaque match joué, la prédiction, le résultat réel et ✓ / ✗ par marché."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Sources.outils import DOSSIER_TRAITEES, RACINE  # noqa: E402

PREDICTIONS = RACINE / "Données" / "Prédictions"
SORTIE = PREDICTIONS / "bilan_detaille.csv"
JOURS_AFFICHES = 7
UTILISATION = """Utilisation : python Sources/bilan.py [SPORT] [JOURS]
  SPORT : football, coupes, selections, basket, baseball ou tous (par défaut)
  JOURS : nombre de jours affichés à l'écran (par défaut 7). Le fichier contient tous les matchs.
Exemple : python Sources/bilan.py football 3"""


def signe(ok):
    return "" if pd.isna(ok) else ("✓" if ok else "✗")


def certitude(proba):
    """Probabilité du côté choisi : 30 % de « plus » = 70 % de « moins »."""
    return max(proba, 1 - proba)


def plus_ou_moins(proba, reel, ligne):
    """Choix du modèle (plus / moins de la ligne), et s'il avait raison."""
    if pd.isna(proba) or pd.isna(reel):
        return None, None
    choix_plus = proba > 0.5
    return ("plus" if choix_plus else "moins"), choix_plus == (reel > ligne)


def football():
    p = PREDICTIONS / "football_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_csv(p, parse_dates=["date"])
    reel = pd.read_csv(DOSSIER_TRAITEES / "football_championnats.csv", parse_dates=["date"], low_memory=False)
    m = pred.merge(reel[["code", "date", "home_team", "away_team", "home_buts", "away_buts", "total_buts",
                         "total_corners", "total_fautes", "total_cartons_jaunes"]],
                   on=["code", "date", "home_team", "away_team"])
    lignes = []
    for _, r in m.iterrows():
        ligne = {"sport": "football", "date": r["date"].date(), "match": f"{r['home_team']} - {r['away_team']}",
                 "score": f"{int(r['home_buts'])}-{int(r['away_buts'])}"}
        for col, nom, cible, seuil in [("proba_plus_2_5_buts", "buts +2,5", "total_buts", 2.5),
                                       ("proba_plus_9_5_corners", "corners +9,5", "total_corners", 9.5),
                                       ("proba_plus_24_5_fautes", "fautes +24,5", "total_fautes", 24.5),
                                       ("proba_plus_3_5_jaunes", "jaunes +3,5", "total_cartons_jaunes", 3.5)]:
            choix, ok = plus_ou_moins(r[col], r[cible], seuil)
            reel_txt = "" if pd.isna(r[cible]) else int(r[cible])
            ligne[nom] = f"{choix} {certitude(r[col]):.0%} → {reel_txt} {signe(ok)}" if choix else ""
            ligne[f"_{nom}"] = ok
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def resultat_1n2(pred, reel_col, sport):
    noms = {"domicile": "victoire 1", "nul": "nul", "exterieur": "victoire 2"}
    probas = pred[["proba_domicile", "proba_nul", "proba_exterieur"]].to_numpy()
    choix = [list(noms)[i] for i in probas.argmax(axis=1)]
    return [f"{noms[c]} {p.max():.0%} → {noms.get(r, '?')} {signe(c == r)}" for c, p, r in zip(choix, probas, reel_col)], \
           [c == r for c, r in zip(choix, reel_col)]


def coupes():
    p = PREDICTIONS / "coupes_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_csv(p, dtype={"espn_id": str}, parse_dates=["date"])
    reel = pd.read_csv(RACINE / "Données" / "brutes" / "coupes" / "coupes_matchs.csv", dtype={"espn_id": str})
    m = pred.merge(reel[["espn_id", "home_buts", "away_buts", "home_jaunes", "away_jaunes"]].dropna(subset=["home_buts"]),
                   on="espn_id")
    if m.empty:
        return pd.DataFrame()
    r = m["home_buts"] - m["away_buts"]
    vrai = pd.Series("nul", index=m.index).mask(r > 0, "domicile").mask(r < 0, "exterieur")
    textes, oks = resultat_1n2(m, vrai, "coupes")
    return pd.DataFrame({"sport": "ligue des champions", "date": m["date"].dt.date,
                         "match": m["home_team"] + " - " + m["away_team"],
                         "score": m["home_buts"].astype(int).astype(str) + "-" + m["away_buts"].astype(int).astype(str),
                         "résultat": textes, "_résultat": oks,
                         "jaunes réels": (m["home_jaunes"] + m["away_jaunes"]).astype("Int64"),
                         "jaunes prévus": m["mu_jaunes"].round(1)})


def selections():
    p = PREDICTIONS / "selections_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_csv(p, parse_dates=["date"])
    reel = pd.read_csv(DOSSIER_TRAITEES / "football_selections.csv", parse_dates=["date"])
    m = pred.merge(reel[["date", "home_team", "away_team", "home_score", "away_score", "resultat", "total_buts"]],
                   on=["date", "home_team", "away_team"])
    if m.empty:
        return pd.DataFrame()
    textes, oks = resultat_1n2(m, m["resultat"], "selections")
    buts = [plus_ou_moins(p, t, 2.5) for p, t in zip(m["proba_plus_2_5_buts"], m["total_buts"])]
    return pd.DataFrame({"sport": "sélections", "date": m["date"].dt.date, "match": m["home_team"] + " - " + m["away_team"],
                         "score": m["home_score"].astype(int).astype(str) + "-" + m["away_score"].astype(int).astype(str),
                         "résultat": textes, "_résultat": oks,
                         "buts +2,5": [f"{c} {certitude(p):.0%} → {int(t)} {signe(o)}" for (c, o), p, t in
                                       zip(buts, m["proba_plus_2_5_buts"], m["total_buts"])],
                         "_buts +2,5": [o for _, o in buts]})


def basket():
    p = PREDICTIONS / "basketball_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_csv(p, parse_dates=["date"])
    reel = pd.read_csv(DOSSIER_TRAITEES / "basketball_matchs.csv")
    m = pred.merge(reel[["game_id", "home_pts_total", "away_pts_total", "total_q1", "total_mi_temps", "total_match"]],
                   on="game_id")
    if m.empty:
        return pd.DataFrame()
    res = pd.DataFrame({"sport": "basket", "date": m["date"].dt.date,
                        "match": m["home_team_abbr"] + " - " + m["away_team_abbr"],
                        "score": m["home_pts_total"].astype(int).astype(str) + "-" + m["away_pts_total"].astype(int).astype(str)})
    for nom, prevu, vrai in [("1er QT", "prevu_total_q1", "total_q1"), ("mi-temps", "prevu_total_mi_temps", "total_mi_temps"),
                             ("match", "prevu_total_match", "total_match")]:
        res[nom] = [f"prévu {a:.0f} → {b:.0f} (écart {b - a:+.0f})" for a, b in zip(m[prevu], m[vrai])]
    return res


def baseball():
    p = PREDICTIONS / "baseball_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_csv(p)
    reel = pd.read_csv(DOSSIER_TRAITEES / "baseball_matchs.csv")
    m = pred.merge(reel[["gamePk", "home_score", "away_score", "total_manche_1", "total_match"]], on="gamePk")
    if m.empty:
        return pd.DataFrame()
    res = pd.DataFrame({"sport": "baseball", "date": pd.to_datetime(m["date"]).dt.date,
                        "match": m["home_team"] + " - " + m["away_team"],
                        "score": m["home_score"].astype(int).astype(str) + "-" + m["away_score"].astype(int).astype(str)})
    for nom, col, cible, seuil in [("point manche 1", "proba_point_manche_1", "total_manche_1", 0.5),
                                   ("match +8,5", "proba_plus_8_5", "total_match", 8.5)]:
        choix = [plus_ou_moins(p, t, seuil) for p, t in zip(m[col], m[cible])]
        res[nom] = [f"{c} {certitude(p):.0%} → {int(t)} {signe(o)}" for (c, o), p, t in zip(choix, m[col], m[cible])]
        res[f"_{nom}"] = [o for _, o in choix]
    return res


SPORTS = {"football": football, "coupes": coupes, "selections": selections, "basket": basket, "baseball": baseball}


def main():
    args = sys.argv[1:]
    if args and args[0] not in SPORTS and args[0] != "tous" and not args[0].isdigit():
        print(UTILISATION)
        return
    sport = args[0] if args and not args[0].isdigit() else "tous"
    jours = int(next((a for a in args if a.isdigit()), JOURS_AFFICHES))
    choisis = SPORTS if sport == "tous" else {sport: SPORTS[sport]}
    tout = []
    for nom, fonction in choisis.items():
        d = fonction()
        if d.empty:
            continue
        tout.append(d)
        recents = d[pd.to_datetime(d["date"]) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=jours)]
        print(f"\n=== {nom.upper()} : {len(recents)} matchs joués ces {jours} derniers jours (sur {len(d)} au total) ===")
        if len(recents):
            visibles = [c for c in recents.columns if not c.startswith("_") and c != "sport"]
            print(recents.sort_values("date")[visibles].to_string(index=False))
        reussite = {c[1:]: d[c].dropna().astype(bool).mean() for c in d.columns if c.startswith("_")}
        if reussite:
            print("Bonnes réponses sur tous les matchs joués : " + " | ".join(f"{k} {v:.0%}" for k, v in reussite.items()))
    if tout:
        complet = pd.concat(tout, ignore_index=True)
        complet[[c for c in complet.columns if not c.startswith("_")]].to_csv(SORTIE, index=False, encoding="utf-8-sig")
        print(f"\nDétail de tous les matchs enregistré dans {SORTIE} (s'ouvre avec Excel)")
    else:
        print("Aucun match prédit n'a encore été joué.")


if __name__ == "__main__":
    main()
