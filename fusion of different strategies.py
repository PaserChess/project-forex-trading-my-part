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
# 1. CONFIGURATION & SETUP
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# Directory Management
OUTPUT_DIR = "backtest_results_final"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_BL.csv")

# --- Strategy Parameters (Optimized) ---
TARGET_BUDGET = 1000000.0   
TRANSACTION_COST = 0.0005   
LOOKBACK_MOMENTUM = 252     # 1 Year (Robust Signal)
LOOKBACK_COV = 126          # 6 Months (Risk Structure)

# --- Black-Litterman Parameters ---
RISK_AVERSION_DELTA = 2.5   
TAU = 0.05                  
MAX_WEIGHT = 0.30           # Max 30% per currency
GROSS_LEVERAGE = 1.2        # Max Total Leverage 120%

# --- Risk Analysis Parameters ---
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
print(f" STARTING: ULTIMATE BLACK-LITTERMAN FX STRATEGY")
print("="*80)

# ==============================================================================
# 2. DATA LOADING & RATE SIMULATION
# ==============================================================================
print("\n[1] Loading Data & Simulating Rates...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)['Close'].ffill().dropna()

# Macro Rate Simulation (for Swap/Carry calculation)
fred_codes = {k: 'SIM' for k in ['USD','EUR','JPY','GBP','CHF','AUD','CAD','NZD','MXN','ZAR','BRL','TRY','SGD','SEK','NOK','INR']}
def generate_macro_rates(price_index):
    dates = price_index
    rates = pd.DataFrame(index=dates, columns=fred_codes.keys())
    rates[:] = 0.0 
    rates['USD'] = 0.015; rates['EUR'] = 0.0; rates['JPY'] = -0.001
    rates['GBP'] = 0.0075; rates['CHF'] = -0.0075; rates['AUD'] = 0.015
    rates['MXN'] = 0.07; rates['TRY'] = 0.12; rates['ZAR'] = 0.06
    
    # Inflation shock scenario (2022+)
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
# 3. SIGNAL GENERATION (MOMENTUM + COVARIANCE)
# ==============================================================================
print("[2] Calculating Signals (12-Month Momentum)...")

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
    
    # 12-Month Momentum (The "View")
    log_rets = np.log(1 + returns_total[t])
    momentum_score[t] = log_rets.rolling(LOOKBACK_MOMENTUM).sum().shift(1)

returns_total.dropna(inplace=True)
momentum_score.dropna(inplace=True)

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_black_litterman_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    # Historical Data Window
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    # Covariance Matrix
    sigma = history.cov() * 252 
    # Equilibrium Weights (Equal Weight assumption)
    w_mkt = np.ones(n_assets) / n_assets 
    # Implied Equilibrium Returns
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    # Views (Based on Momentum)
    curr_mom = momentum_score.loc[curr_date]
    ranked = curr_mom.sort_values(ascending=False)
    # View on Top 4 and Bottom 4 assets
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()
    
    P, Q, Omega_list = [], [], []
    
    for t in views_assets:
        idx = tickers_list.index(t)
        mom_val = curr_mom[t]
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        
        # Scale view to expected return (max 20%)
        view_ret = np.tanh(mom_val) * 0.20 
        Q.append(view_ret)
        
        # Uncertainty is proportional to asset variance
        variance = sigma.iloc[idx, idx]
        Omega_list.append(variance * TAU)

    P = np.array(P); Q = np.array(Q); Omega = np.diag(Omega_list)
    
    # BL Master Formula
    try:
        tau_sigma = TAU * sigma
        inv_tau_sigma = np.linalg.inv(tau_sigma + np.eye(n_assets)*1e-6)
        inv_omega = np.linalg.inv(Omega)
        M_inverse = inv_tau_sigma + np.dot(np.dot(P.T, inv_omega), P)
        M = np.linalg.inv(M_inverse + np.eye(n_assets)*1e-6)
        term2 = np.dot(inv_tau_sigma, pi) + np.dot(np.dot(P.T, inv_omega), Q)
        bl_returns = np.dot(M, term2)
    except:
        # Fallback to zero if matrix inversion fails
        return np.zeros(n_assets)

    # Optimizer (Maximize Utility)
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
# 5. BACKTEST WITH TRADE LOGGING
# ==============================================================================
print("\n[3] Executing Backtest with Detailed Trade Logs...")

rebal_dates = returns_total.loc["2020-01-01":].resample('W-FRI').last().index
daily_dates = returns_total.loc["2020-01-01":].index
rebal_set = set(rebal_dates)

current_weights_daily = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

# Logging variable
last_log_month = 0

print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY ($)':<12} | {'ALLOCATION (Top Convictions)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    # --- REBALANCING ---
    if d in rebal_set:
        target_weights = get_black_litterman_weights(d)
        target_weights[np.abs(target_weights) < 0.02] = 0 # Cleanup small weights
        
        turnover = np.sum(np.abs(target_weights - current_weights_daily))
        cost = turnover * TRANSACTION_COST
        current_weights_daily = target_weights
        
        # LOGGING (Monthly or on major shifts)
        if d.month != last_log_month:
            # Extract positions
            pos_longs = []
            pos_shorts = []
            for idx, w in enumerate(target_weights):
                t_name = tickers[idx].replace('=X', '')
                if w > 0.05: pos_longs.append(f"{t_name}:{w:.0%}")
                elif w < -0.05: pos_shorts.append(f"{t_name}:{w:.0%}")
            
            l_str = ", ".join(pos_longs) if pos_longs else "None"
            s_str = ", ".join(pos_shorts) if pos_shorts else "None"
            
            print(f"{d.date()} | {capital:,.0f} | LONG: [{l_str}] | SHORT: [{s_str}]")
            last_log_month = d.month
            
    else:
        cost = 0.0
        
    # --- PnL Calculation ---
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector) - cost
    
    capital *= (1 + port_ret)
    equity_curve.append(capital)
    equity_dates.append(d)
    strategy_returns.append(port_ret)

# Export Data
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
df_export.to_csv(FILE_RETURNS)

print("-" * 100)
print(f"Backtest Complete. Final Capital: {capital:,.0f} $")

# ==============================================================================
# 6. RISK ANALYSIS MODULE
# ==============================================================================
def run_full_risk_analysis():
    print("\n" + "="*80)
    print(" [4] FULL RISK AUDIT & ANALYTICS")
    print("="*80)
    
    df_ret = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = df_ret['STRATEGY']
    
    # 1. Base Metrics
    vol_ann = strat_ret.std() * np.sqrt(TRADING_DAYS)
    mean_ret_ann = strat_ret.mean() * TRADING_DAYS
    sharpe = mean_ret_ann / vol_ann if vol_ann > 0 else 0
    
    # 2. Drawdowns
    nav = (1 + strat_ret).cumprod()
    dd = (nav - nav.cummax()) / nav.cummax()
    max_dd = dd.min()
    calmar = mean_ret_ann / abs(max_dd) if max_dd < 0 else 0
    
    # 3. Tail Risk (VaR 95%)
    var_95_hist = strat_ret.quantile(1 - CONFIDENCE_LEVEL)
    
    # Monte Carlo VaR
    mu_mc = strat_ret.mean()
    sigma_mc = strat_ret.std()
    sim_rets = np.random.normal(mu_mc, sigma_mc, MC_SIMULATIONS)
    var_95_mc = np.percentile(sim_rets, (1 - CONFIDENCE_LEVEL) * 100)
    
    # Skewness/Kurtosis
    skew_val = skew(strat_ret)
    kurt_val = kurtosis(strat_ret)

    print(f"\n--- PERFORMANCE ---")
    print(f"Annualized Return  : {mean_ret_ann:+.2%}")
    print(f"Total Return       : {(capital/TARGET_BUDGET)-1:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    print(f"Calmar Ratio       : {calmar:.2f}")
    
    print(f"\n--- EXTREME RISK ---")
    print(f"Max Drawdown       : {max_dd:.2%}")
    print(f"Annual Volatility  : {vol_ann:.2%}")
    print(f"Weekly VaR (95%)   : {var_95_hist:.2%}")
    print(f"MC VaR (95%)       : {var_95_mc:.2%}")
    print(f"Skewness           : {skew_val:.2f} " + ("(Crash Risk)" if skew_val < -0.5 else "(Normal)"))
    print(f"Kurtosis           : {kurt_val:.2f}")

    # --- CSV REPORT ---
    risk_report = pd.DataFrame({
        'Metric': ['Annual Return', 'Total Return', 'Sharpe', 'Calmar', 'Max DD', 'Volatility', 'VaR 95%', 'Skewness'],
        'Value': [mean_ret_ann, (capital/TARGET_BUDGET)-1, sharpe, calmar, max_dd, vol_ann, var_95_hist, skew_val]
    })
    risk_report.to_csv(os.path.join(OUTPUT_DIR, "RISK_METRICS_REPORT.csv"))
    print(f"\nRisk Report Saved: {os.path.join(OUTPUT_DIR, 'RISK_METRICS_REPORT.csv')}")

# ==============================================================================
# 7. VISUALIZATION
# ==============================================================================
def generate_charts():
    print("\n[5] Generating Charts...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    
    # 1. Equity Curve + Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Equity
    ax1.plot(df_equity.index, df_equity['Equity'], label='Black-Litterman Strategy', color='purple', linewidth=1.5)
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title("Strategy Performance: Black-Litterman (2020-2025)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Capital ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Yearly Returns Annotations
    years = df_equity.index.year.unique()
    for year in years:
        data_year = df_equity[df_equity.index.year == year]
        if data_year.empty: continue
        ret_year = (data_year['Equity'].iloc[-1] / data_year['Equity'].iloc[0]) - 1
        # Position text in the middle of the year
        mid_date = data_year.index[len(data_year)//2]
        ax1.text(mid_date, df_equity['Equity'].max()*0.95, f"{year}\n{ret_year:+.1%}", 
                 ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # Drawdown
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    ax2.plot(dd.index, dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown Profile", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_PERFORMANCE_FINAL.png"), dpi=150)
    
    # 2. Risk vs Return Scatter
    summary = pd.DataFrame()
    summary['Returns'] = returns.mean() * 252
    summary['Volatility'] = returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(12, 8))
    subset = summary.drop('STRATEGY', errors='ignore')
    sns.scatterplot(data=subset, x='Volatility', y='Returns', s=100, color='gray', alpha=0.5, label='Forex Pairs')
    
    if 'STRATEGY' in summary.index:
        strat = summary.loc['STRATEGY']
        plt.scatter(strat['Volatility'], strat['Returns'], s=300, color='green', edgecolors='black', label='MY STRATEGY')
        plt.text(strat['Volatility']+0.003, strat['Returns'], "  BL STRATEGY", fontsize=11, fontweight='bold', color='green')
        
    for i in range(len(subset)):
        plt.text(subset.iloc[i]['Volatility'], subset.iloc[i]['Returns'], subset.index[i], fontsize=8, alpha=0.7)
        
    plt.axhline(0, color='black'); plt.axvline(0, color='black')
    plt.title('Risk/Return Positioning (Efficient Frontier)', fontsize=14)
    plt.xlabel('Volatility (Risk)'); plt.ylabel('Total Return (Price + Carry)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_RISK_SCATTER.png"), dpi=150)
    
    print(f"-> Charts generated in folder '{OUTPUT_DIR}'.")
    plt.show()

# ==============================================================================
# 8. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    run_full_risk_analysis()
    generate_charts()