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
# 1. CONFIGURATION "ARIMA-GARCH"
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

OUTPUT_DIR = "backtest_results_garch"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_BL.csv")

# --- Paramètres Stratégie ---
TARGET_BUDGET = 1000000.0   
TRANSACTION_COST = 0.0005   
LOOKBACK_COV = 126          

# --- Paramètres Black-Litterman (DYNAMIC LEVERAGE) ---
RISK_AVERSION_DELTA = 2.5   
TAU = 0.05                  
MAX_WEIGHT = 0.80           # On autorise de grosses convictions
GROSS_LEVERAGE = 4.0        # Levier Max autorisé (4x)
# Ce levier ne sera utilisé QUE si la volatilité est faible (GARCH)

# --- Paramètres Analyse ---
CONFIDENCE_LEVEL = 0.95  
TRADING_DAYS = 252       
MC_SIMULATIONS = 10000   

pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
    "EURJPY", "EURGBP", "AUDJPY", "NZDJPY",
    "USDMXN", "USDTRY", "USDZAR", "USDSGD", "USDNOK", "USDSEK"
]
tickers = [f"{pair}=X" for pair in pairs_list]
start_date = "2018-01-01" 
end_date = "2025-12-31"

print("="*80)
print(f" 🚀 DÉMARRAGE : HYBRID ARIMA-GARCH STRATEGY (SMART LEVERAGE)")
print("="*80)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
print("\n[1] Loading Data & Simulating Rates...")
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
# 3. SIGNAL GENERATION (ARIMA-GARCH PROXY)
# ==============================================================================
print("[2] Calculating ARIMA Direction + GARCH Volatility...")

returns_total = pd.DataFrame(index=data.index, columns=data.columns)
smart_score = pd.DataFrame(index=data.index, columns=data.columns) # Le score final

for t in tickers:
    clean = t.replace('=X','')
    base, quote = clean[:3], clean[3:]
    r_price = data[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = rates_daily[base] - rates_daily[quote]
    
    returns_total[t] = r_price + r_carry
    
    rets = returns_total[t]
    
    # --- A. ARIMA PROXY (Direction) ---
    # Autocorrélation glissante (Trend persistence)
    rolling_ar = rets.rolling(window=63).apply(lambda x: x.autocorr(lag=1), raw=False)
    # Prédiction brute = Autocorr * Rendement d'hier
    raw_forecast = rolling_ar * rets.shift(1)
    
    # --- B. GARCH PROXY (Volatilité Exponentielle) ---
    # On utilise une EWMA (Exponential Weighted Moving Average) de la variance
    # C'est la méthode standard de RiskMetrics (JP Morgan) pour imiter GARCH
    # Lambda = 0.94 est le standard industriel
    variance_garch = (rets**2).ewm(alpha=0.06).mean()
    vol_garch = np.sqrt(variance_garch)
    
    # --- C. SCORE FINAL (Sharpe Prédictif) ---
    # On divise la prédiction par la volatilité GARCH.
    # Si la Vol est haute, le Score baisse -> Levier baisse.
    # Si la Vol est basse, le Score monte -> Levier monte.
    # On multiplie par un scalaire (ex: 50) pour ajuster l'échelle pour Black-Litterman
    final_signal = (raw_forecast / (vol_garch + 1e-6)) * 50.0
    
    smart_score[t] = final_signal.rolling(5).mean() # Lissage final

returns_total.dropna(inplace=True)
smart_score.dropna(inplace=True)

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_black_litterman_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    sigma = history.cov() * 252 
    w_mkt = np.ones(n_assets) / n_assets 
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    curr_sig = smart_score.loc[curr_date]
    ranked = curr_sig.sort_values(ascending=False)
    
    # On prend des vues sur les extrêmes (Top 4 / Bottom 4)
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()
    
    P, Q, Omega_list = [], [], []
    
    for t in views_assets:
        idx = tickers_list.index(t)
        val = curr_sig[t]
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        
        # Le signal est déjà ajusté par la vol (GARCH).
        # On utilise tanh pour le borner entre -30% et +30% d'expected return
        view_ret = np.tanh(val) * 0.30
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
# 5. BACKTEST
# ==============================================================================
print("\n[3] Executing ARIMA-GARCH Backtest...")

rebal_dates = returns_total.loc["2020-01-01":].resample('D').last().index
daily_dates = returns_total.loc["2020-01-01":].index
rebal_set = set(rebal_dates)

current_weights_daily = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

last_log_month = 0
print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY':<12} | {'LEV':<6} | {'ALLOCATION (Vol Adjusted)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    if d in rebal_set:
        target_weights = get_black_litterman_weights(d)
        target_weights[np.abs(target_weights) < 0.02] = 0 
        
        turnover = np.sum(np.abs(target_weights - current_weights_daily))
        cost = turnover * TRANSACTION_COST
        current_weights_daily = target_weights
        
        if d.month != last_log_month:
            curr_lev = np.sum(np.abs(target_weights))
            pos_longs = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(target_weights) if w > 0.1]
            pos_shorts = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(target_weights) if w < -0.1]
            
            l_str = ", ".join(pos_longs) if pos_longs else "."
            s_str = ", ".join(pos_shorts) if pos_shorts else "."
            
            print(f"{d.date()} | {capital:,.0f} | {curr_lev:.1f}x | L:[{l_str}] | S:[{s_str}]")
            last_log_month = d.month
            
    else:
        cost = 0.0
        
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector) - cost
    
    capital *= (1 + port_ret)
    equity_curve.append(capital)
    equity_dates.append(d)
    strategy_returns.append(port_ret)

# Export
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
df_export.to_csv(FILE_RETURNS)

print("-" * 100)
print(f"Final Capital: {capital:,.0f} $")

# ==============================================================================
# 6. REPORTING & VISUALIZATION
# ==============================================================================
def generate_reports():
    print("\n[4] Generating GARCH Reports...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = returns['STRATEGY']
    
    # Metrics
    mean_ret = strat_ret.mean() * 252
    vol = strat_ret.std() * np.sqrt(252)
    sharpe = mean_ret / vol if vol > 0 else 0
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    max_dd = dd.min()
    
    print(f"\n--- PERFORMANCE (ARIMA-GARCH) ---")
    print(f"Total Return       : {(capital/TARGET_BUDGET)-1:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    print(f"Max Drawdown       : {max_dd:.2%}")
    print(f"Annual Volatility  : {vol:.2%}")

    # Charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Equity
    ax1.plot(df_equity['Equity'], color='#2ca02c', linewidth=1.5, label='ARIMA-GARCH Strategy')
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Strategy Performance: Dynamic Volatility Targeting", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2.plot(dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_title(f"Drawdown Profile (Max: {max_dd:.1%})", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_GARCH_PERFORMANCE.png"), dpi=150)
    
    # Risk/Return Scatter
    plt.figure(figsize=(12, 8))
    summary = pd.DataFrame()
    summary['Returns'] = returns.mean() * 252
    summary['Volatility'] = returns.std() * np.sqrt(252)
    
    subset = summary.drop('STRATEGY', errors='ignore')
    sns.scatterplot(data=subset, x='Volatility', y='Returns', s=100, color='gray', alpha=0.5)
    if 'STRATEGY' in summary.index:
        strat = summary.loc['STRATEGY']
        plt.scatter(strat['Volatility'], strat['Returns'], s=350, color='#2ca02c', edgecolors='black', label='ARIMA-GARCH')
        plt.text(strat['Volatility']+0.003, strat['Returns'], "  SMART LEVERAGE", fontsize=11, fontweight='bold', color='#2ca02c')
        
    for i in range(len(subset)):
        plt.text(subset.iloc[i]['Volatility'], subset.iloc[i]['Returns'], subset.index[i], fontsize=8, alpha=0.7)
        
    plt.axhline(0, color='black'); plt.axvline(0, color='black')
    plt.title('Risk/Return Profile: Does GARCH improve efficiency?', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_GARCH_SCATTER.png"), dpi=150)
    
    print("-> Charts generated.")
    plt.show()

if __name__ == "__main__":
    generate_reports()