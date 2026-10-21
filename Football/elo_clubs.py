"""Classement Elo des clubs européens, à partir des championnats suivis et de la Ligue des champions."""
import numpy as np
import pandas as pd

ELO_DEPART = 1500
ELO_DEPART_INCONNU = 1400      # club d'un championnat non suivi : on le suppose plus faible au départ
AVANTAGE_DOMICILE = 60
K = {"championnat": 20, "coupe": 30}


def flux_de_matchs(champ, coupes, noms):
    """Tous les matchs dans l'ordre chronologique, avec un nom de club commun aux deux sources."""
    c = champ[["date", "home_team", "away_team", "home_buts", "away_buts"]].assign(type="championnat", espn_id=None)
    e = coupes[["date", "home_team", "away_team", "home_buts", "away_buts", "espn_id"]].copy()
    e[["home_team", "away_team"]] = e[["home_team", "away_team"]].replace(noms)
    e["type"] = "coupe"
    # Le jour même, les matchs de championnat passent avant ceux de coupe (heure inconnue)
    return pd.concat([c, e], ignore_index=True).sort_values(["date", "type"], kind="stable").reset_index(drop=True)


def classement(flux, clubs_suivis):
    """Note de chaque club AVANT chaque match (les matchs à venir n'ont pas de score et ne changent rien)."""
    notes = {}
    avant_dom, avant_ext = np.empty(len(flux)), np.empty(len(flux))
    for i, m in enumerate(flux.itertuples(index=False)):
        for equipe in (m.home_team, m.away_team):
            if equipe not in notes:
                notes[equipe] = ELO_DEPART if equipe in clubs_suivis else ELO_DEPART_INCONNU
        rd, re_ = notes[m.home_team], notes[m.away_team]
        avant_dom[i], avant_ext[i] = rd, re_
        if pd.isna(m.home_buts) or pd.isna(m.away_buts):
            continue
        attendu = 1 / (10 ** (-(rd - re_ + AVANTAGE_DOMICILE) / 400) + 1)
        reel = 1.0 if m.home_buts > m.away_buts else 0.5 if m.home_buts == m.away_buts else 0.0
        diff = abs(m.home_buts - m.away_buts)
        mult = 1 if diff <= 1 else 1.5 if diff == 2 else (11 + diff) / 8
        delta = K[m.type] * mult * (reel - attendu)
        notes[m.home_team], notes[m.away_team] = rd + delta, re_ - delta
    flux = flux.copy()
    flux["home_elo"], flux["away_elo"] = avant_dom, avant_ext
    return flux, notes
