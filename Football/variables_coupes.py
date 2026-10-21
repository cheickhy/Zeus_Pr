"""Variables d'entrée de la Ligue des champions : forme en championnat et en coupe d'Europe, avant le match."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elo_clubs  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dates_sofascore import ressemblance  # noqa: E402
from Sources.outils import DOSSIER_BRUTES, DOSSIER_TRAITEES, etape, moyenne_precedente, sauvegarder  # noqa: E402

FICHIER_COUPES = DOSSIER_BRUTES / "coupes" / "coupes_matchs.csv"
STATS = ["buts", "tirs", "tirs_cadres", "corners", "fautes", "jaunes"]
CIBLES = {"total_buts": "buts", "total_corners": "corners", "total_fautes": "fautes", "total_cartons_jaunes": "jaunes"}
FENETRE_CHAMPIONNAT, FENETRE_COUPE = 10, 6

# Noms ESPN que la ressemblance automatique ne relie pas correctement à football-data
NOMS_MANUELS = {"Paris Saint-Germain": "Paris SG", "Atletico Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
                "Sporting CP": "Sp Lisbon", "Bayer Leverkusen": "Leverkusen", "Internazionale": "Inter",
                "Inter Milan": "Inter", "AC Milan": "Milan", "Borussia Dortmund": "Dortmund",
                "Borussia Monchengladbach": "M'gladbach", "Manchester City": "Man City",
                "Manchester United": "Man United", "Tottenham Hotspur": "Tottenham", "Newcastle United": "Newcastle",
                "Eintracht Frankfurt": "Ein Frankfurt", "RB Leipzig": "RB Leipzig", "PSV Eindhoven": "PSV Eindhoven",
                "Real Sociedad": "Sociedad", "Celta Vigo": "Celta", "Nottingham Forest": "Nott'm Forest",
                "Wolverhampton Wanderers": "Wolves", "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
                "AS Monaco": "Monaco", "Lille": "Lille", "Stade Rennais": "Rennes", "Bayern Munich": "Bayern Munich",
                "Atlético Madrid": "Ath Madrid", "Borussia Mönchengladbach": "M'gladbach",
                "Club Brugge": "Club Brugge", "Rangers": "Rangers"}
# Clubs de championnats non suivis dont le nom ressemble à un club suivi (ex. Sparta Prague / Sparta Rotterdam)
SANS_CORRESPONDANCE = {"Sparta Prague", "Red Star Belgrade", "Lincoln Red Imps", "Hibernians", "Flora"}


def correspondance_noms(espn, fd):
    """Nom ESPN -> nom football-data (absent si le club ne joue pas dans un championnat suivi)."""
    res = {}
    for nom in espn:
        if nom in SANS_CORRESPONDANCE:
            continue
        if nom in NOMS_MANUELS and NOMS_MANUELS[nom] in fd:
            res[nom] = NOMS_MANUELS[nom]
            continue
        meilleur = max(fd, key=lambda f: ressemblance(nom, f))
        if ressemblance(nom, meilleur) >= 0.8:
            res[nom] = meilleur
    return res


def forme_championnat(champ):
    """Moyennes des 10 derniers matchs de championnat de chaque club, après chaque match (pour/contre)."""
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        t = pd.DataFrame({"date": champ["date"], "equipe": champ[f"{camp}_team"]})
        for s in STATS:
            t[f"{s}_pour"], t[f"{s}_contre"] = champ[f"{camp}_{s}"], champ[f"{adv}_{s}"]
        t["code"] = champ["code"]
        lignes.append(t)
    t = pd.concat(lignes).sort_values(["equipe", "date"]).reset_index(drop=True)
    cols = [c for c in t.columns if c.endswith(("_pour", "_contre"))]
    moy = t.groupby("equipe")[cols].transform(lambda s: s.rolling(FENETRE_CHAMPIONNAT, min_periods=3).mean())
    # Même statistique, moyenne de tout le championnat sur ses 300 derniers matchs d'équipe :
    # sert à comparer chaque club à SON championnat (on siffle plus en Italie qu'en Angleterre)
    t = t.sort_values(["code", "date"])
    ligue = t.groupby("code")[cols].transform(lambda s: s.rolling(300, min_periods=50).mean()).loc[moy.index]
    relatif = (moy / ligue).add_prefix("rel_")
    return pd.concat([t.loc[moy.index, ["date", "equipe"]], moy.add_prefix("champ_"),
                      relatif.add_prefix("champ_")], axis=1)


def forme_coupe(c):
    """Moyennes des 6 matchs de coupe d'Europe PRÉCÉDENTS de chaque club (décalage d'un match)."""
    lignes = []
    for camp, adv in [("home", "away"), ("away", "home")]:
        t = pd.DataFrame({"espn_id": c["espn_id"], "date": c["date"], "equipe": c[f"{camp}_team"]})
        for s in ["buts", "corners", "fautes", "jaunes"]:
            t[f"{s}_pour"], t[f"{s}_contre"] = c[f"{camp}_{s}"], c[f"{adv}_{s}"]
        lignes.append(t)
    t = pd.concat(lignes).sort_values(["equipe", "date"]).reset_index(drop=True)
    res = t[["espn_id", "equipe"]].copy()
    for col in [x for x in t.columns if x.endswith(("_pour", "_contre"))]:
        res[f"ldc_{col}"] = moyenne_precedente(t, "equipe", col, FENETRE_COUPE, min_matchs=2)
    res["ldc_nb_matchs"] = t.groupby("equipe").cumcount()
    return res


def construire(coupes, champ):
    """Une ligne par match de coupe, avec les indices des deux clubs connus avant le match."""
    coupes = coupes.sort_values(["date", "heure_utc"]).reset_index(drop=True)
    noms = correspondance_noms(set(coupes["home_team"]) | set(coupes["away_team"]),
                               sorted(set(champ["home_team"]) | set(champ["away_team"])))
    forme = forme_championnat(champ).sort_values("date")
    sortie = coupes.copy()
    fc = forme_coupe(coupes)
    for camp in ["home", "away"]:
        sortie[f"{camp}_nom_championnat"] = sortie[f"{camp}_team"].map(noms)
        # Dernier état de forme en championnat STRICTEMENT avant le jour du match
        sortie = pd.merge_asof(sortie.sort_values("date"), forme.rename(columns={"equipe": f"{camp}_nom_championnat"}),
                               on="date", by=f"{camp}_nom_championnat", allow_exact_matches=False)
        sortie = sortie.rename(columns={c: f"{camp}_{c}" for c in forme.columns if c.startswith("champ_")})
        f = fc.rename(columns={c: f"{camp}_{c}" for c in fc.columns if c.startswith("ldc_")})
        sortie = sortie.merge(f.rename(columns={"equipe": f"{camp}_team"}), on=["espn_id", f"{camp}_team"], how="left")
    sortie = sortie.sort_values(["date", "heure_utc"]).reset_index(drop=True)
    flux, _ = elo_clubs.classement(elo_clubs.flux_de_matchs(champ, coupes, noms),
                                   set(champ["home_team"]) | set(champ["away_team"]))
    elo = flux.loc[flux["type"] == "coupe", ["espn_id", "home_elo", "away_elo"]]
    sortie = sortie.merge(elo.rename(columns={"home_elo": "ldc_elo_dom", "away_elo": "ldc_elo_ext"}), on="espn_id", how="left")
    sortie["ldc_elo_ecart"] = sortie["ldc_elo_dom"] - sortie["ldc_elo_ext"]
    sortie["resultat"] = np.select([sortie["home_buts"] > sortie["away_buts"], sortie["home_buts"] == sortie["away_buts"]],
                                   ["domicile", "nul"], "exterieur")
    sortie.loc[sortie["home_buts"].isna(), "resultat"] = None
    for cible, s in CIBLES.items():
        sortie[cible] = sortie[f"home_{s}"] + sortie[f"away_{s}"]
        # Tendance de la compétition : moyenne des 60 matchs PRÉCÉDENTS (il y a moins de fautes ces dernières saisons)
        sortie[f"ldc_tendance_{cible}"] = sortie[cible].shift(1).rolling(60, min_periods=20).mean()
    sortie["mois"] = pd.to_datetime(sortie["date"]).dt.month
    sortie["annee_saison"] = pd.to_datetime(sortie["date"]).apply(lambda d: d.year if d.month >= 7 else d.year - 1)
    return sortie.sort_values(["date", "heure_utc"]).reset_index(drop=True), noms


def charger():
    coupes = pd.read_csv(FICHIER_COUPES, dtype={"espn_id": str}, parse_dates=["date"])
    champ = pd.read_csv(DOSSIER_TRAITEES / "football_championnats.csv", parse_dates=["date"], low_memory=False)
    return coupes, champ


def main():
    print("=== VARIABLES D'ENTRÉE : LIGUE DES CHAMPIONS ===")
    coupes, champ = charger()
    sortie, noms = construire(coupes, champ)
    clubs = set(coupes["home_team"]) | set(coupes["away_team"])
    etape(f"Clubs reliés à un championnat suivi : {len(noms)} sur {len(clubs)}")
    print("Exemples :", [(k, v) for k, v in list(noms.items()) if k != v][:10])
    print("Non reliés (championnats non suivis) :", sorted(clubs - set(noms))[:25])
    sauvegarder(sortie, "coupes_variables.csv")


if __name__ == "__main__":
    main()
