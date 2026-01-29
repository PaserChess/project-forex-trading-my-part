import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web
import warnings

warnings.filterwarnings("ignore")

# --- PARAMÈTRES ---
TARGET_BUDGET = 1000000.0   
RISK_AVERSION = 2.5     
# Liste des 7 Majeures + Cross (Sans exotiques)
pairs_list = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD" , 
    "EURJPY", "EURGBP", "EURCHF", "AUDJPY", "NZDJPY", # Cross
    "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDBRL", "USDTRY", "USDINR", "USDSGD" 
]

tickers = [f"{pair}=X" for pair in pairs_list]
benchmark_ticker = "DX-Y.NYB"

start_date = "2010-01-01"
end_date = "2025-01-01"

# --- 1. TÉLÉCHARGEMENT PRIX (YAHOO) ---
print("📥 1. Téléchargement des Prix (Yahoo)...")
# On garde 'data' comme variable principale des prix pour la suite du script
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
data = data.ffill().dropna()

# --- 2. TÉLÉCHARGEMENT BENCHMARK (DOLLAR INDEX) ---
print("📥 1b. Téléchargement du Benchmark (DXY)...")
bench_raw = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)['Close']
bench_data = bench_raw.ffill().dropna()

# ==============================================================================
# 3. TÉLÉCHARGEMENT TAUX (HYBRIDE : FRED + MANUEL)
# ==============================================================================
print("📥 3. Récupération Taux d'Intérêt...")

# A. Taux Majeurs via FRED (Fiable)
fred_codes = {
    'USD': 'IR3TIB01USM156N', 'EUR': 'IR3TIB01EZM156N', 'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N', 'CHF': 'IR3TIB01CHM156N', 'AUD': 'IR3TIB01AUM156N',
    'CAD': 'IR3TIB01CAM156N', 'NZD': 'IR3TIB01NZM156N'
}

# Initialisation avec des 0
rates_daily = pd.DataFrame(0.0, index=data.index, columns=fred_codes.keys())

try:
    df_fred = web.DataReader(list(fred_codes.values()), 'fred', start_date, end_date)
    df_fred.columns = list(fred_codes.keys())
    # Fusion propre avec les dates du Forex
    rates_daily.update((df_fred / 100.0).resample('D').ffill())
    print("   ✅ Taux Majeurs FRED récupérés.")
except:
    print("   ⚠️ Erreur FRED. Utilisation taux fixes majeurs.")
    # Fallback taux fixes approximatifs récents
    rates_daily['USD'] = 0.05; rates_daily['EUR'] = 0.04; rates_daily['JPY'] = 0.00
    rates_daily['GBP'] = 0.05; rates_daily['AUD'] = 0.04; rates_daily['NZD'] = 0.05

# B. Taux Exotiques (Fixés manuellement pour éviter le bug 0%)
# On ajoute les colonnes manquantes avec des taux moyens historiques prudents
exotic_rates = {
    'TRY': 0.40,  # Turquie (40% !) - Empêche le short gratuit
    'BRL': 0.11,  # Brésil (11%)
    'ZAR': 0.08,  # Afrique du Sud
    'MXN': 0.11,  # Mexique
    'INR': 0.06,  # Inde
    'NOK': 0.04,  'SEK': 0.04, 'SGD': 0.03
}

for curr, rate in exotic_rates.items():
    rates_daily[curr] = rate

print("   ✅ Taux Exotiques injectés manuellement.")

# --- 4. CALCUL DU RENDEMENT TOTAL (PRIX + CARRY) ---
print("⚙️ Calcul des rendements nets de frais (Swap)...")

def get_currency_pair_components(pair_ticker):
    symbol = pair_ticker.replace('=X', '')
    base = symbol[:3]
    quote = symbol[3:]
    return base, quote

returns_total = pd.DataFrame(index=data.index, columns=data.columns)
price_returns = data.pct_change()

for ticker in tickers:
    base, quote = get_currency_pair_components(ticker)
    
    # 1. Rendement du Prix
    r_price = price_returns[ticker]
    
    # 2. Rendement du Carry
    if base in rates_daily.columns and quote in rates_daily.columns:
        interest_differential = (rates_daily[base] - rates_daily[quote]) / 252 
        r_total = r_price + interest_differential
    else:
        r_total = r_price 
        
    returns_total[ticker] = r_total

returns_total = returns_total.dropna()

# Définition des sets d'entrainement/test
train_data = returns_total.loc[:"2020-01-01"]
test_data = returns_total.loc["2020-01-01":]

# --- 5. OPTIMISATION MARKOWITZ (MODIFIÉE : LEVIER DÉBRIDÉ) ---
# --- 5. OPTIMISATION MARKOWITZ (VERSION BRIDÉE / SÉCURISÉE) ---
def optimize_markowitz(mu, sigma, risk_aversion):
    num_assets = len(mu)
    # On part d'une allocation neutre pour aider l'optimiseur
    init_guess = np.repeat(1/num_assets, num_assets)
    
    def objective(weights):
        # Fonction d'utilité classique : Rendement - (Risque * Aversion)
        utility = (np.dot(weights, mu) * 252) - (risk_aversion / 2) * (np.dot(weights.T, np.dot(sigma, weights)) * 252)
        return -utility

    # Contrainte 1 : Somme nette = 100% (Tout le capital est utilisé, long ou short)
    cons_net = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    # Contrainte 2 : LEVIER GLOBAL MAX = 3.0x
    # (Somme des valeurs absolues <= 300%)
    GLOBAL_MAX_LEVERAGE = 3.0
    cons_lev = {'type': 'ineq', 'fun': lambda x: GLOBAL_MAX_LEVERAGE - np.sum(np.abs(x))}
    
    constraints = [cons_net, cons_lev]
    
    # Contrainte 3 : EXPOSITION INDIVIDUELLE MAX = +/- 50%
    # On interdit de mettre plus de 50% du budget sur une seule devise
    bounds = tuple((-0.5, 0.5) for _ in range(num_assets))
    
    try:
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except:
        # Fallback si l'optimiseur échoue : on retourne des poids équilibrés
        return init_guess

# --- 6. BACKTEST COMPARATIF (MODIFIÉ : AFFICHAGE HEBDOMADAIRE) ---
print("\n⚙️ Lancement du Backtest Hebdomadaire...")

# --- MODIFICATION : Passage en 'W-FRI' (Hebdomadaire Vendredi) ---
rebal_dates = test_data.resample('W-FRI').last().index
history_returns = train_data.copy()

capital_markowitz = TARGET_BUDGET
capital_equal = TARGET_BUDGET
values_markowitz = []
values_equal = []
dates_history = []

for i in range(len(rebal_dates) - 1):
    current_date = rebal_dates[i]
    next_date = rebal_dates[i+1]
    
    # Stratégie A: Markowitz
    mu = history_returns.mean()
    sigma = history_returns.cov()
    w_markowitz = optimize_markowitz(mu, sigma, RISK_AVERSION)
    
    # --- MODIFICATION : AFFICHAGE DES RATES ET ALLOCATION ---
    print(f"\n📅 Semaine du {current_date.date()}")
    
    # 1. Affichage des Taux
    if current_date in rates_daily.index:
        cur_rates = rates_daily.loc[current_date]
        rates_str = " | ".join([f"{k}:{v:.2%}" for k, v in cur_rates.items() if k in ['USD','EUR','JPY','GBP']])
        print(f"   📈 Taux 3M Clés: {rates_str}")
    
    # 2. Affichage Allocation
    alloc_str = " | ".join([f"{t.replace('=X','')}:{w:.2f}" for t, w in zip(tickers, w_markowitz) if abs(w) > 0.05])
    print(f"   ⚖️ Alloc: {alloc_str}")
    # --------------------------------------------------------

    # Stratégie B: Equal Weight
    n = len(tickers)
    w_equal = np.repeat(1/n, n) 
    
    # Forward Test
    mask = (test_data.index > current_date) & (test_data.index <= next_date)
    month_returns = test_data.loc[mask]
    
    if month_returns.empty: continue
        
    daily_ret_markowitz = month_returns.dot(w_markowitz)
    for ret in daily_ret_markowitz:
        capital_markowitz *= (1 + ret)
        values_markowitz.append(capital_markowitz)
        dates_history.append(month_returns.index[daily_ret_markowitz.values == ret][0])

    daily_ret_equal = month_returns.dot(w_equal)
    for ret in daily_ret_equal:
        capital_equal *= (1 + ret)
        values_equal.append(capital_equal)
    
    history_returns = pd.concat([history_returns, month_returns])

# Création DataFrame résultats
df_results = pd.DataFrame(index=dates_history)
df_results['Markowitz (Uncapped)'] = values_markowitz
df_results['Equal Weight'] = values_equal

# Ajout du Benchmark (Dollar Index) aligné
df_results['Dollar index'] = bench_data.reindex(df_results.index).ffill()
df_results = df_results.dropna()

# Rebaser le Dollar index à 1M$
df_results['Dollar index'] = df_results['Dollar index'] / df_results['Dollar index'].iloc[0] * TARGET_BUDGET

# --- 7. MÉTRIQUES ET AFFICHAGE ---
def calculate_metrics(series):
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    daily_ret = series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    dd = (series / series.cummax() - 1).min()
    return total_ret, sharpe, dd

print("\n" + "="*65)
print(f"{'STRATÉGIE':<25} | {'RENDEMENT':<10} | {'SHARPE':<8} | {'MAX DD':<10}")
print("-" * 65)

for col in df_results.columns:
    ret, sharpe, dd = calculate_metrics(df_results[col])
    print(f"{col:<25} | {ret:,.2%}    | {sharpe:.2f}     | {dd:.2%}")

print("="*65)

plt.figure(figsize=(12, 6))
plt.plot(df_results['Markowitz (Uncapped)'], label='Markowitz (Levier Max)', color='blue', linewidth=1.5)
plt.plot(df_results['Equal Weight'], label='Equal Weight', color='gray', linestyle='--', alpha=0.7)
plt.plot(df_results['Dollar index'], label='Dollar index', color='orange', alpha=0.6)
plt.axhline(y=TARGET_BUDGET, color='r', linestyle=':', label='Budget Initial')
plt.title('Performance Forex : Stratégie Hebdo (2020-2025)')
plt.ylabel('Valeur Portefeuille ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 8. GÉNÉRATION DES ORDRES POUR DEMAIN ---
print("\n" + "█"*80)
print(f"🔮 ORDRES À EXÉCUTER (Levier Débridé)")
print("█"*80)

# Optimisation sur toute la donnée disponible
full_mu = returns_total.mean()
full_sigma = returns_total.cov()
# Appel sans limite de levier
optimal_weights = optimize_markowitz(full_mu, full_sigma, RISK_AVERSION)

# Récupération derniers prix
last_prices = data.iloc[-1]

print(f"{'ACTION':<10} | {'PAIRE':<10} | {'PRIX':<10} | {'CARRY (An)':<10} | {'POIDS':<8} | {'MONTANT ($)'}")
print("-" * 90)

total_invested = 0
for i, ticker in enumerate(tickers):
    weight = optimal_weights[i]
    if abs(weight) < 0.01: continue
        
    pair_name = ticker.replace('=X', '')
    price = last_prices[ticker]
    amount = weight * TARGET_BUDGET
    
    # Calcul du Carry annuel estimé pour l'affichage
    base, quote = get_currency_pair_components(ticker)
    carry_pct = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        last_rates = rates_daily.iloc[-1]
        raw_diff = last_rates[base] - last_rates[quote]
        # Si on est Long, on gagne (base - quote). Si Short, on gagne (quote - base).
        carry_pct = raw_diff if weight > 0 else -raw_diff

    action = "🟢 ACHAT" if weight > 0 else "🔴 VENTE"
    color_carry = "+" if carry_pct > 0 else ""
    
    print(f"{action:<10} | {pair_name:<10} | {price:<10.4f} | {color_carry}{carry_pct:.2%}    | {weight*100:>6.1f} % | {amount:>12,.0f} $")
    total_invested += abs(amount)

print("-" * 90)
print(f"Levier utilisé : {total_invested / TARGET_BUDGET:.2f}x")
plt.show()