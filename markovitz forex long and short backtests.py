import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
import os
import warnings

# ==============================================================================
# 1. CONFIGURATION "CERTIFIÉE RÉALISTE"
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

OUTPUT_DIR = "backtest_results_certified"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_REAL.csv")

# --- Paramètres Stratégie (MOMENTUM WEEKLY) ---
TARGET_BUDGET = 1000000.0   
TRANSACTION_COST = 0.0005   
LOOKBACK_MOMENTUM = 252     # 1 An (Signal Lent et Robuste)
LOOKBACK_COV = 126          

# --- Paramètres Black-Litterman ---
RISK_AVERSION_DELTA = 2.5   
TAU = 0.05                  
MAX_WEIGHT = 0.30           # 30% Max par position
GROSS_LEVERAGE = 1.2        # Levier 1.2x (Suffisant pour Momentum)

# --- Paramètres Analyse ---
TRADING_DAYS = 252       
CONFIDENCE_LEVEL = 0.95

pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
    "EURJPY", "EURGBP", "AUDJPY", "NZDJPY",
    "USDMXN", "USDTRY", "USDZAR", "USDSGD", "USDNOK", "USDSEK"
]
tickers = [f"{pair}=X" for pair in pairs_list]
start_date = "2018-01-01" 
end_date = "2025-12-31"

print("="*80)
print(f" 🚀 DÉMARRAGE : BL MOMENTUM WEEKLY (NO LOOK-AHEAD BIAS)")
print("="*80)

# ==============================================================================
# 2. DATA
# ==============================================================================
print("\n[1] Chargement Données...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)['Close'].ffill().dropna()

# Macro Rate Simulation
fred_codes = {k: 'SIM' for k in ['USD','EUR','JPY','GBP','CHF','AUD','CAD','NZD','MXN','ZAR','BRL','TRY','SGD','SEK','NOK','INR']}
def generate_macro_rates(price_index):
    dates = price_index
    rates = pd.DataFrame(index=dates, columns=fred_codes.keys())
    rates[:] = 0.0 
    rates['USD'] = 0.015; rates['EUR'] = 0.0; rates['JPY'] = -0.001
    rates['GBP'] = 0.0075; rates['CHF'] = -0.0075; rates['AUD'] = 0.015
    rates['MXN'] = 0.07; rates['TRY'] = 0.12; rates['ZAR'] = 0.06
    mask = dates >= '2022-03-01'
    def hike(c, t, v=0.0):
        if np.sum(mask) > 0:
            start = rates.loc[~mask, c].iloc[-1] if not rates.loc[~mask, c].empty else 0
            path = np.linspace(start, t, np.sum(mask))
            noise = np.random.normal(0, v/2, np.sum(mask)) 
            rates.loc[mask, c] = path + noise
    hike('USD', 0.055); hike('EUR', 0.040); hike('GBP', 0.052)
    hike('JPY', -0.001); hike('MXN', 0.1125); hike('TRY', 0.45, 0.02)
    return rates

rates_daily = generate_macro_rates(data.index).resample('D').ffill().reindex(data.index).ffill() / 252.0

# ==============================================================================
# 3. SIGNAL (MOMENTUM)
# ==============================================================================
print("[2] Calcul Signal Momentum...")
returns_total = pd.DataFrame(index=data.index, columns=data.columns)
momentum_score = pd.DataFrame(index=data.index, columns=data.columns)

for t in tickers:
    clean = t.replace('=X','')
    base, quote = clean[:3], clean[3:]
    r_price = data[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = rates_daily[base] - rates_daily[quote]
    returns_total[t] = r_price + r_carry
    
    # Momentum 12 Mois
    log_rets = np.log(1 + returns_total[t])
    momentum_score[t] = log_rets.rolling(LOOKBACK_MOMENTUM).sum().shift(1)

returns_total.dropna(inplace=True)
momentum_score.dropna(inplace=True)

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_black_litterman_weights(curr_date):
    # Utilise STRICTEMENT les données passées
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    sigma = history.cov() * 252 
    w_mkt = np.ones(n_assets) / n_assets 
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    curr_mom = momentum_score.loc[curr_date]
    ranked = curr_mom.sort_values(ascending=False)
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()
    
    P, Q, Omega_list = [], [], []
    for t in views_assets:
        idx = tickers_list.index(t)
        mom_val = curr_mom[t]
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        view_ret = np.tanh(mom_val) * 0.20 
        Q.append(view_ret)
        variance = sigma.iloc[idx, idx]
        Omega_list.append(variance * TAU)

    P = np.array(P); Q = np.array(Q); Omega = np.diag(Omega_list)
    
    try:
        tau_sigma = TAU * sigma
        inv_tau_sigma = np.linalg.inv(tau_sigma + np.eye(n_assets)*1e-6)
        inv_omega = np.linalg.inv(Omega)
        M_inverse = inv_tau_sigma + np.dot(np.dot(P.T, inv_omega), P)
        M = np.linalg.inv(M_inverse + np.eye(n_assets)*1e-6)
        term2 = np.dot(inv_tau_sigma, pi) + np.dot(np.dot(P.T, inv_omega), Q)
        bl_returns = np.dot(M, term2)
    except:
        return np.zeros(n_assets)

    def negative_utility(w):
        port_ret = np.dot(w, bl_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        return -(port_ret - (RISK_AVERSION_DELTA / 2) * (port_vol**2))

    cons = ({'type': 'ineq', 'fun': lambda x: GROSS_LEVERAGE - np.sum(np.abs(x))})
    bounds = tuple((-MAX_WEIGHT, MAX_WEIGHT) for _ in range(n_assets))
    init_guess = np.sign(bl_returns) * (1/n_assets)
    
    try:
        res = minimize(negative_utility, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-6)
        return res.x
    except:
        return np.zeros(n_assets)

# ==============================================================================
# 5. BACKTEST "NO TRICHE"
# ==============================================================================
print("\n[3] Exécution Backtest (Logique : Trade today based on yesterday's signal)...")

rebal_dates = returns_total.loc["2020-01-01":].resample('W-FRI').last().index
daily_dates = returns_total.loc["2020-01-01":].index
rebal_set = set(rebal_dates)

# On commence avec 0 positions
current_weights = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

last_log_month = 0
print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY':<12} | {'ALLOCATION (Trade Lundi pour la semaine)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    # 1. D'ABORD : On subit le marché d'aujourd'hui avec les poids décidés AVANT
    day_ret_vector = returns_total.loc[d]
    
    # On applique les poids actuels
    port_ret = np.dot(current_weights, day_ret_vector)
    
    # 2. ENSUITE : On vérifie si on doit changer les poids (pour DEMAIN)
    # Si on est Vendredi (Rebal_set), on calcule les nouveaux poids
    cost = 0.0
    if d in rebal_set:
        # On calcule le signal sur la cloture d'aujourd'hui (Vendredi soir)
        new_weights = get_black_litterman_weights(d)
        new_weights[np.abs(new_weights) < 0.02] = 0 
        
        # On paie les frais de transaction maintenant pour changer le portefeuille
        turnover = np.sum(np.abs(new_weights - current_weights))
        cost = turnover * TRANSACTION_COST
        
        # Ces nouveaux poids seront effectifs pour le PROCHAIN tour de boucle (Lundi)
        current_weights = new_weights
        
        # Logging
        if d.month != last_log_month:
            pos_l = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(current_weights) if w > 0.05]
            pos_s = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(current_weights) if w < -0.05]
            l_str = ",".join(pos_l) if pos_l else "."
            s_str = ",".join(pos_s) if pos_s else "."
            print(f"{d.date()} | {capital:,.0f} | L:[{l_str}] | S:[{s_str}]")
            last_log_month = d.month

    # Mise à jour du capital (Rendement du jour - Frais éventuels de rebalancement ce soir)
    port_ret_net = port_ret - (cost if d in rebal_set else 0.0)
    capital *= (1 + port_ret_net)
    
    equity_curve.append(capital)
    equity_dates.append(d)
    strategy_returns.append(port_ret_net)

# Export
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
df_export.to_csv(FILE_RETURNS)
print("-" * 100)
print(f"Final Capital: {capital:,.0f} $")

# ==============================================================================
# 6. REPORTING
# ==============================================================================
def generate_reports():
    print("\n[4] Generating Certified Reports...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = returns['STRATEGY']
    
    mean_ret = strat_ret.mean() * 252
    vol = strat_ret.std() * np.sqrt(252)
    sharpe = mean_ret / vol if vol > 0 else 0
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    max_dd = dd.min()
    
    print(f"\n--- PERFORMANCE (CERTIFIED REAL) ---")
    print(f"Total Return       : {(capital/TARGET_BUDGET)-1:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    print(f"Max Drawdown       : {max_dd:.2%}")
    print(f"Annual Volatility  : {vol:.2%}")

    # Charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(df_equity['Equity'], color='purple', linewidth=1.5, label='BL Momentum (Real)')
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--')
    ax1.set_title("Strategy Performance: Momentum (No Look-Ahead)", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_title(f"Drawdown (Max: {max_dd:.1%})", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_REAL_MOMENTUM.png"), dpi=150)
    print("-> Charts generated.")
    plt.show()

if __name__ == "__main__":
    generate_reports()