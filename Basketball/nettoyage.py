import os
import pandas as pd

# 1. Configuration des chemins reels trouves par le diagnostic
DOSSIER_BRUT = r"C:\Users\HP\Desktop\NAFA_données\Données\brutes\mlb_data\mlb_data"
DOSSIER_SORTIE = r"C:\Users\HP\Desktop\NAFA_données\données\traitées"

print("--- NETTOYAGE INDIVIDUEL BASEBALL ---")

chemin_schedule = os.path.join(DOSSIER_BRUT, "schedule.csv")
chemin_linescore = os.path.join(DOSSIER_BRUT, "linescore.csv")

if not os.path.exists(chemin_schedule) or not os.path.exists(chemin_linescore):
    print("Erreur : Les fichiers schedule.csv ou linescore.csv sont introuvables.")
else:
    try:
        os.makedirs(DOSSIER_SORTIE, exist_ok=True)

        print("Nettoyage du fichier schedule...")
        df_schedule = pd.read_csv(chemin_schedule)
        
        # On supprime immediatement les matchs sans score final
        df_schedule = df_schedule.dropna(subset=['home_score', 'away_score'])
        
        # Securite numerique
        df_schedule['home_score'] = pd.to_numeric(df_schedule['home_score'], errors='coerce')
        df_schedule['away_score'] = pd.to_numeric(df_schedule['away_score'], errors='coerce')
        
        df_schedule.to_csv(os.path.join(DOSSIER_SORTIE, "mlb_schedule_nettoye.csv"), index=False)

       
        print("Nettoyage et filtrage du fichier linescore...")
        df_linescore = pd.read_csv(chemin_linescore)
        
        # Conversion forcee en numerique des manches et des runs
        df_linescore['inning'] = pd.to_numeric(df_linescore['inning'], errors='coerce')
        df_linescore['home_runs'] = pd.to_numeric(df_linescore['home_runs'], errors='coerce')
        df_linescore['away_runs'] = pd.to_numeric(df_linescore['away_runs'], errors='coerce')
        
        # On supprime les lignes ou le numero de manche est manquant
        df_linescore = df_linescore.dropna(subset=['inning'])
        
        # On ne garde que les 3 premieres manches indispensables pour tes calculs de paris
        df_manches_3 = df_linescore[df_linescore['inning'] <= 3].copy()
        
        # Calcul du total de runs par manche pour le match
        df_manches_3['total_runs_inning'] = df_manches_3['home_runs'].fillna(0) + df_manches_3['away_runs'].fillna(0)
        
        df_manches_3.to_csv(os.path.join(DOSSIER_SORTIE, "mlb_linescore_nettoye.csv"), index=False)

        print("Nettoyage individuel du baseball termine avec succes !")

    except Exception as e:
        print(f"Une erreur est survenue lors du nettoyage : {e}")
