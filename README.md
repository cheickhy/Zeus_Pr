# Prédiction de paris sportifs

Prédictions de marchés de paris pour le football, le basket (NBA) et le baseball (MLB).
Chaque programme télécharge les derniers résultats, calcule la forme des équipes à partir des
matchs **précédents uniquement**, puis donne une probabilité pour les matchs à venir.

## Ce que le projet prédit

| Sport | Compétitions | Marchés |
|---|---|---|
| Football | 22 championnats européens | Buts, corners, fautes, cartons jaunes (n'importe quelle ligne) |
| Football | Ligue des champions | Résultat (1 N 2), cartons jaunes |
| Football | Sélections nationales, CAN | Résultat (1 N 2), plus de 2,5 buts |
| Basket | NBA | Totaux du 1er quart-temps, de la mi-temps et du match ; points de chaque équipe |
| Baseball | MLB | Point en 1ère manche, total des 3 premières manches, plus de 8,5 points |

## Fiabilité mesurée

Tests sur des matchs jamais vus par les modèles, puis en conditions réelles (prédictions
enregistrées **avant** les matchs).

| Marché | En test | En réel |
|---|---|---|
| Football : +2,5 buts / +9,5 corners / +24,5 fautes / +3,5 jaunes | 56 / 55 / 67 / 62 % | 62 / 57 / 63 / 62 % (284 matchs) |
| Ligue des champions : résultat | 60 % (référence naïve : 51 %) | en cours |
| Sélections, qualifications CAN : résultat | 68 % (référence naïve : 53 %) | en cours |
| Basket : erreur moyenne sur le total du match | 15 points | en cours |
| Baseball : +8,5 points | 54 % | peu fiable, à suivre |

## Installation

1. Installer Python 3.12, puis dans le dossier du projet :
   ```
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt
   ```
2. Placer les données brutes d'origine dans `Données/brutes/` (elles ne sont pas sur GitHub,
   trop volumineuses) :
   - `Données/brutes/mlb_data/mlb_data/` : schedule.csv, linescore.csv, pitching.csv, batting.csv
   - `Données/brutes/nba_data/nba_data/` : schedule.csv, quarters.csv, players.csv

   Les autres données (football, sélections, Ligue des champions, saisons récentes) sont
   téléchargées automatiquement par les programmes.
3. Préparer les données une première fois :
   ```
   python Baseball/nettoyage.py
   python Basketball/nettoyage.py
   python Football/mise_a_jour.py
   python Football/nettoyage_championnats.py
   ```

## Utilisation quotidienne

| Quand (heure GMT) | Commande | Rôle |
|---|---|---|
| Tous les jours, 8 h - 20 h | `python Basketball/prediction.py` | Matchs NBA de la nuit + bilan |
| Mardi et mercredi, avant 16 h 45 | `python Football/prediction_coupes.py` | Ligue des champions |
| Mardi | `python Football/prediction_championnats.py` | Bilan du week-end + matchs de semaine |
| Vendredi, 14 h - 17 h | `python Football/prediction_championnats.py` | Matchs du week-end |
| Avant une trêve internationale | `python Football/prediction_selections.py` | Sélections, qualifications CAN |
| Jours de match MLB | `python Baseball/prediction.py` | Baseball |

Chaque programme affiche d'abord le **bilan** des prédictions passées (bonnes réponses en réel),
puis les nouvelles prédictions. Les modèles sont réentraînés automatiquement tous les 7 jours
(option `--reentrainer` pour forcer). Une date passée en argument lance une **démonstration** :
`python Basketball/prediction.py 2026-03-15`.

## Probabilité pour une ligne de bookmaker

```
python Football/proba_ligne.py Barcelona corners 10.5
python Basketball/proba_ligne.py BOS match 220.5
python Basketball/proba_ligne.py BOS equipe-match 110.5
python Baseball/proba_ligne.py Yankees 3.5
```

## Export pour l'application

```
python Sources/export_application.py            # matchs d'aujourd'hui et à venir
python Sources/export_application.py --exemple  # dernière journée prédite de chaque sport
```

Produit `Données/Prédictions/predictions_application.json` : une fiche par match, avec les
probabilités déjà calculées pour les lignes courantes (`proba_plus` = probabilité de dépasser la ligne).

## Organisation des dossiers

```
Baseball/, Basketball/   nettoyage, variables, entrainement (tests), mise_a_jour, prediction, proba_ligne
Football/                championnats (football-data.co.uk), sélections (résultats internationaux + ESPN),
                         coupes (Ligue des champions, ESPN), sofascore (ancien fichier isolé : touches)
Sources/                 outils communs (chemins, calcul des moyennes sur les matchs précédents,
                         sauvegarde des modèles) et export pour l'application
Données/                 brutes, Traitées, Prédictions (non publiées sur GitHub)
Modeles/                 modèles sauvegardés (recréés automatiquement, non publiés)
```

## Règle essentielle

Une variable d'entrée n'utilise **que des matchs joués avant** le match à prédire
(fonctions `moyenne_precedente` et `somme_precedente` de `Sources/outils.py`). Sans cette règle,
le modèle « verrait » le résultat et paraîtrait excellent en test, puis échouerait en réel.

## Limites connues

- Touches (football) : non prédites, aucune source à jour ne les publie.
- Ligue des champions : buts, corners et fautes non prédits (pas mieux que la moyenne de la compétition).
- Basket : prédictions moins fiables pendant les 4 premiers matchs de chaque équipe (« prudence »).
- Les fichiers de `Données/Prédictions/` sont la preuve des tests réels : à sauvegarder régulièrement.
