import pandas as pd
import os
import numpy as np

# Liste de tes paires
pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "CHFJPY", "EURJPY", "EURAUD", "GBPAUD", "CADJPY", "NZDJPY", "EURCAD", "GBPCAD",
    "USDTRY", "USDINR", "USDBRL", "USDZAR", "USDSEK"
]

def create_master_dataframe(pairs):
    master_df = pd.DataFrame()
    
    print("--- Création de la matrice de prix ---")
    
    for pair in pairs:
        ticker = f"{pair}=X"
        filename = f"{ticker}.csv"
        
        if os.path.exists(filename):
            try:
                # 1. Lecture
                df = pd.read_csv(filename, index_col=0)
                
                # 2. Nettoyage Index
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
                df = df[df.index.notnull()]
                df.index = df.index.tz_localize(None)
                df = df[~df.index.duplicated(keep='last')]

                # 3. Extraction Prix
                col_name = 'Close'
                if 'Close' not in df.columns and 'Adj Close' in df.columns:
                    col_name = 'Adj Close'
                
                if col_name in df.columns:
                    series = df[col_name]
                    series.name = pair
                    
                    # On utilise join outer pour garder TOUTES les dates de TOUS les fichiers
                    if master_df.empty:
                        master_df = pd.DataFrame(series)
                    else:
                        master_df = master_df.join(series, how='outer')
            except Exception as e:
                print(f"Erreur sur {pair} : {e}")
        else:
            print(f"Fichier manquant : {filename}")

    # Conversion en numérique
    master_df = master_df.apply(pd.to_numeric, errors='coerce')

    # Tri par date
    master_df = master_df.sort_index()

    # --- CORRECTION ICI ---
    # 1. On coupe tout ce qui est avant l'an 2000 (pour éviter de remonter à 1970 inutilement)
    master_df = master_df[master_df.index >= '2000-01-01']

    # 2. On remplit les petits trous (jours fériés) avec la valeur précédente
    master_df = master_df.ffill()
    
    # 3. IMPORTANT : On NE FAIT PAS dropna() global.
    # On accepte que certaines colonnes aient des NaN au début (années 2000-2015)
    # si la devise n'existait pas encore.
    
    print(f"\nMatrice terminée : {master_df.shape[0]} jours x {master_df.shape[1]} devises")
    return master_df
# --- EXÉCUTION ---
if __name__ == "__main__":
    # 1. Récupération des PRIX
    df_prices = create_master_dataframe(pairs_list)
    
    if not df_prices.empty:
        # 2. Calcul des RENDEMENTS (Returns)
        # Maintenant que ce sont des nombres, le calcul va marcher
        df_returns = df_prices.pct_change()        
        print("\n--- Aperçu des Prix (df_prices) ---")
        print(df_prices.tail())
        
        print("\n--- Aperçu des Rendements (df_returns) ---")
        print(df_returns.tail())

        # Sauvegarde
        df_prices.to_csv("MASTER_PRICES.csv")
        df_returns.to_csv("MASTER_RETURNS.csv")
        print("\nSuccès : Fichiers 'MASTER_PRICES.csv' et 'MASTER_RETURNS.csv' créés.")
    else:
        print("Erreur : Le DataFrame est vide.")
