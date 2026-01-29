import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas_datareader.data as web
import warnings
import time
import sys

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PARAMÈTRES
# ==============================================================================   
TARGET_BUDGET = 1000000.0   
RISK_AVERSION = 2.5      
#TRANSACTION_COST = 0.0001 # 0.01% (1 basis point) de spread + comm
TRANSACTION_COST = 0.000001 # 0.01% (1 basis point) de spread + comm

# On teste quelques leviers clés pour ne pas surcharger l'affichage
# (Tu peux remettre range(1, 16) si tu veux tout, mais ça fera beaucoup de lignes)
LEVERAGES = [1, 3, 5, 10, 15] 

WINDOWS = {
    '2 Mois': 42,
    '6 Mois': 126,
    '1.5 Ans': 378,
    '3 Ans': 756
}

# Période de TEST (Simulation)
SIM_START = "2015-01-01"
SIM_END = "2024-01-01"

# Période de DATA (Pour avoir de l'historique avant 2020)
DATA_START = "2010-01-01" 

# UNIVERS
pairs_list = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", 
    "EURJPY", "EURGBP", "EURCHF", "AUDJPY", "NZDJPY", 
    "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDBRL", "USDTRY", "USDINR", "USDSGD" 
]
tickers = [f"{pair}=X" for pair in pairs_list]

print("█"*80)
print(f"🚀 BACKTEST CONTINU (CUMULÉ) 2020-2024")
print(f"💰 Capital Initial: {TARGET_BUDGET:,.0f} $")
print(f"💸 Frais inclus: {TRANSACTION_COST*100}% par rotation")
print("█"*80)

# ==============================================================================
# 2. DATA
# ==============================================================================
print("\n📥 1. Téléchargement des Données...")
raw_data = yf.download(tickers, start=DATA_START, end=SIM_END, progress=False)['Close']
data = raw_data.dropna(axis=1, how='all').ffill().dropna()
valid_tickers = list(data.columns)

print("📥 2. Taux d'Intérêt (Carry)...")
rates_daily = pd.DataFrame(0.0, index=data.index, columns=['USD'])
try:
    # On simplifie en prenant juste le taux USD pour l'exemple, ou taux fixes
    # Pour un backtest rapide mais réaliste, on peut utiliser des constantes approximatives 
    # si FRED bug, sinon on réactive FRED. Ici je mets des taux moyens pour aller vite.
    rates_daily['USD'] = 0.03 # Moyenne sur la période
except: pass

# Taux exotiques fixes (approx moyenne période)
exotic_rates = {'TRY': 0.20, 'BRL': 0.08, 'ZAR': 0.06, 'MXN': 0.07, 'INR': 0.05}

print("⚙️ Calcul Rendements (Prix + Carry)...")
returns_total = pd.DataFrame(index=data.index, columns=valid_tickers)
price_returns = data.pct_change()

for t in valid_tickers:
    clean_t = t.replace('=X', '')
    base = clean_t[:3]
    quote = clean_t[3:]
    
    r_price = price_returns[t]
    
    # Carry approximatif (Différentiel fixe pour l'exercice)
    # Dans ton code final, garde ta logique FRED complète
    r_base = rates_daily['USD'] if base == 'USD' else (exotic_rates.get(base, 0.01))
    r_quote = rates_daily['USD'] if quote == 'USD' else (exotic_rates.get(quote, 0.01))
    
    # Si c'est un dataframe (rates_daily) ou float
    if isinstance(r_base, pd.Series): r_base = r_base.reindex(data.index).ffill()
    if isinstance(r_quote, pd.Series): r_quote = r_quote.reindex(data.index).ffill()
    
    r_carry = (r_base - r_quote) / 252
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# ==============================================================================
# 3. OPTIMISEUR
# ==============================================================================
def get_weights(mu, sigma, risk_aversion, max_leverage):
    n = len(mu)
    w = cp.Variable(n)
    sigma_np = sigma.values + np.eye(n) * 1e-8
    ret = w @ mu.values
    risk = cp.quad_form(w, sigma_np)
    prob = cp.Problem(cp.Maximize((ret*252) - (risk_aversion/2)*(risk*252)), 
                      [cp.sum(w)==1, cp.norm(w, 1)<=max_leverage, cp.abs(w)<=max_leverage*0.6])
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        return w.value if w.value is not None else np.ones(n)/n
    except: return np.ones(n)/n

# ==============================================================================
# 4. SIMULATION CUMULATIVE
# ==============================================================================
# On récupère les indices des jours de simulation
sim_days = returns_total.loc[SIM_START:SIM_END].index

results_final = []

for win_name, win_days in WINDOWS.items():
    print(f"\n🔎 TRAITEMENT FENÊTRE : {win_name} ({win_days}j)")
    
    # Capitaux initiaux (On ne reset PAS chaque année)
    capitals = {lev: TARGET_BUDGET for lev in LEVERAGES}
    prev_weights = {lev: np.zeros(len(valid_tickers)) for lev in LEVERAGES}
    
    # Pour tracer la courbe
    equity_curve = {lev: [TARGET_BUDGET] for lev in LEVERAGES}
    dates_curve = [sim_days[0]]
    
    start_time = time.time()
    
    # --- BOUCLE CONTINUE ---
    for i in range(len(sim_days) - 1):
        curr_date = sim_days[i]
        next_date = sim_days[i+1]
        
        # Affichage progression et Bilan Annuel
        if curr_date.year != sim_days[i-1].year and i > 0:
            print(f"   📅 Bilan fin {sim_days[i-1].year} terminé.")
            # On pourrait print ici les capitaux intermédiaires
        
        if i % 50 == 0:
            sys.stdout.write(f"\r   ⏳ Jour {i}/{len(sim_days)}...")
            sys.stdout.flush()

        # 1. Data Historique
        hist = returns_total.loc[:curr_date].tail(win_days)
        if len(hist) < win_days * 0.95: continue
        
        mu = hist.mean()
        sigma = hist.cov()
        day_ret = returns_total.loc[next_date]
        
        # 2. Leviers
        for lev in LEVERAGES:
            # Optim
            w = get_weights(mu, sigma, RISK_AVERSION, lev)
            
            # Frais
            turnover = np.sum(np.abs(w - prev_weights[lev]))
            cost = turnover * TRANSACTION_COST
            
            # Perf
            perf_gross = np.dot(w, day_ret)
            perf_net = perf_gross - cost
            
            # Update Capital Cumulé
            capitals[lev] *= (1 + perf_net)
            prev_weights[lev] = w
            
            # Stockage courbe
            equity_curve[lev].append(capitals[lev])
        
        dates_curve.append(next_date)

    print(f"\n   ✅ Terminé.")
    
    # Affichage du résultat final pour cette fenêtre
    print(f"   💰 RÉSULTATS CUMULÉS (2020 -> 2024) pour {win_name}:")
    for lev in LEVERAGES:
        final = capitals[lev]
        total_ret = (final / TARGET_BUDGET) - 1
        # CAGR (Taux de croissance annuel moyen)
        nb_years = (dates_curve[-1] - dates_curve[0]).days / 365.25
        cagr = (final / TARGET_BUDGET)**(1/nb_years) - 1
        
        print(f"      - Levier {lev}x : {final:,.0f} $ ({total_ret:+.2%}) | CAGR: {cagr:+.2%}")
        
        results_final.append({
            'Fenetre': win_name,
            'Levier': lev,
            'Final_Capital': final,
            'Total_Return': total_ret,
            'CAGR': cagr
        })

# ==============================================================================
# 5. GRAPHIQUE COMPARATIF
# ==============================================================================
df_res = pd.DataFrame(results_final)
print("\n" + "="*80)
print("🏆 PODIUM FINAL (Qui a le plus d'argent en banque aujourd'hui ?)")
print(df_res.sort_values(by='Final_Capital', ascending=False).head(5)[['Fenetre', 'Levier', 'Final_Capital', 'Total_Return']])
print("="*80)

# Petit graphique barres pour visualiser les survivants
plt.figure(figsize=(12, 6))
import seaborn as sns
sns.barplot(data=df_res, x='Fenetre', y='Total_Return', hue='Levier')
plt.title("Performance Totale Cumulée (2020-2024) - Net de frais")
plt.ylabel("Rendement Total (%)")
plt.axhline(0, color='black')
plt.grid(True, alpha=0.3)
plt.show()