import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Liste de tes paires
pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "CHFJPY", "EURJPY", "EURAUD", "GBPAUD", "CADJPY", "NZDJPY", "EURCAD", "GBPCAD",
    "USDTRY", "USDINR", "USDBRL", "USDZAR", "USDSEK"
]

def update_forex_data(pair):
    ticker_symbol = f"{pair}=X"
    filename = f"{ticker_symbol}.csv"
    
    print(f"--- Traitement de {ticker_symbol} ---")
    
    if os.path.exists(filename):
        print(f"Fichier {filename} trouvé. Vérification des mises à jour...")
        try: 
            # a. Charger le fichier CSV existant
            df_existing = pd.read_csv(filename, index_col=0, parse_dates=True)
            
            # Nettoyage index
            df_existing.index = pd.to_datetime(df_existing.index, utc=True, errors='coerce')
            df_existing = df_existing[df_existing.index.notnull()]
            df_existing.index = df_existing.index.tz_localize(None) 
            
            if not df_existing.empty:
                last_date = df_existing.index.max()
                start_date = last_date + timedelta(days=1)
                
                if start_date > datetime.now():
                    print("Déjà à jour, pas de changements à faire.")
                    return
                
                print(f"Dernière date connue : {last_date.date()}. Téléchargement depuis {start_date.date()}...")
                
                # Téléchargement incrémental (ça c'était déjà bon)
                df_new = yf.download(ticker_symbol, start=start_date, progress=False)
                
                if not df_new.empty:
                    df_updated = pd.concat([df_existing, df_new])
                    df_updated.to_csv(filename)
                    print(f"Mise à jour réussie. {len(df_new)} nouvelles lignes ajoutées.")
                else:
                    print("Aucune nouvelle donnée disponible sur Yahoo Finance.")
            else:
                print("Fichier existant vide. Relance complète depuis 2000.")
                # CORRECTION 1 ICI : On force le start à 2000 au lieu de period="10y"
                df_full = yf.download(ticker_symbol, start="2000-01-01", progress=False)
                df_full.to_csv(filename)
                
        except Exception as e:
            print(f"Erreur lors de la lecture/mise à jour de {filename} : {e}")
            
    else:
        # Cas 1 : Le fichier n'existe pas
        print(f"Fichier {filename} non trouvé. Téléchargement depuis l'an 2000...")
        
        # CORRECTION 2 ICI : On remplace period="10y" par start="2000-01-01"
        # Note: Si la paire n'existait pas en 2000, Yahoo donnera la date la plus ancienne dispo (ex: 2003)
        df_full = yf.download(ticker_symbol, start="2000-01-01", progress=False)
        
        if not df_full.empty:
            df_full.to_csv(filename)
            print(f"Fichier créé avec succès ({len(df_full)} lignes).")
        else:
            print("Erreur : Aucune donnée récupérée depuis Yahoo Finance.")

if __name__ == "__main__":
    for pair in pairs_list:
        update_forex_data(pair)
    print("Toutes les paires ont été chargées.")