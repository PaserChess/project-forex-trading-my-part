import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os
import warnings

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

OUTPUT_DIR = "backtest_results_bl"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_BL.csv")

# --- STRATEGY PARAMETERS ---
TARGET_BUDGET = 1000000.0   
TRANSACTION_COST = 0.0005   
LOOKBACK_MOMENTUM = 252     # 1 Year (Signal)
LOOKBACK_COV = 126          # 6 Months (Risk Correlation)

# --- BLACK-LITTERMAN PARAMETERS ---
RISK_AVERSION_DELTA = 2.5   # Standard risk aversion
TAU = 0.05                  # Confidence in the prior (0.05 is standard)
MAX_WEIGHT = 0.30           # Max 30% per asset (Constraint)
GROSS_LEVERAGE = 1.2        # Max Total Leverage (120%)

pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
    "EURJPY", "EURGBP", "AUDJPY", "NZDJPY",
    "USDMXN", "USDTRY", "USDZAR", "USDSGD", "USDNOK", "USDSEK"
]
tickers = [f"{pair}=X" for pair in pairs_list]
start_date = "2018-01-01" 
end_date = "2025-12-31"

print(">>> INITIALIZING BLACK-LITTERMAN FX STRATEGY")

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
print("1. Loading Data & Simulating Rates...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)['Close'].ffill().dropna()

# Simulated Rates (Macro)
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
# 3. SIGNALS (MOMENTUM + COVARIANCE)
# ==============================================================================
print("2. Calculating Returns & Covariance...")

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
    
    # 12-Month Momentum (Our "View")
    log_rets = np.log(1 + returns_total[t])
    momentum_score[t] = log_rets.rolling(LOOKBACK_MOMENTUM).sum().shift(1)

returns_total.dropna(inplace=True)
momentum_score.dropna(inplace=True)

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================

def get_black_litterman_weights(curr_date):
    """
    Calculates weights using the Black-Litterman model.
    Prior: Equal Weights (Neutral)
    Views: Momentum Scores
    """
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    # 1. Historical Data Window (for Covariance)
    # We look back 6 months for correlation structure
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5) # slightly larger buffer
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: 
        return np.zeros(n_assets)

    # --- INPUTS ---
    # Sigma: Annualized Covariance Matrix
    sigma = history.cov() * 252 
    
    # Prior Weights (Market Equilibrium): We assume Equal Weight is "Neutral"
    # In equity, this would be Market Cap. In FX, Equal Weight is standard.
    w_mkt = np.ones(n_assets) / n_assets 
    
    # Pi: Implied Equilibrium Excess Returns
    # Formula: Pi = delta * Sigma * w_mkt
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    # --- VIEWS (P & Q) ---
    # We construct views based on Momentum
    # View: "Asset i will return X%" -> X derived from Momentum Score
    
    curr_mom = momentum_score.loc[curr_date]
    
    # We only take views on the Top 4 and Bottom 4 (Conviction)
    ranked = curr_mom.sort_values(ascending=False)
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()
    
    P = [] # Link matrix (identifies assets)
    Q = [] # Expected returns vector
    Omega_list = [] # Uncertainty of views
    
    for t in views_assets:
        idx = tickers_list.index(t)
        mom_val = curr_mom[t]
        
        # Construct P (1 for the asset)
        row = np.zeros(n_assets)
        row[idx] = 1
        P.append(row)
        
        # Construct Q (Expected Return)
        # We scale momentum to a realistic annual return expectation (e.g. max 20%)
        # Sigmoid function to cap views between -20% and +20%
        view_ret = np.tanh(mom_val) * 0.20 
        Q.append(view_ret)
        
        # Construct Omega (Uncertainty)
        # Standard BL heuristic: Omega = diag(P * (tau * Sigma) * P.T)
        # Basically, uncertainty is proportional to the asset's historical variance
        variance = sigma.iloc[idx, idx]
        Omega_list.append(variance * TAU)

    P = np.array(P)
    Q = np.array(Q)
    Omega = np.diag(Omega_list)
    
    # --- BL MASTER FORMULA ---
    # New Expected Returns (E[R])
    # E[R] = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1 * [(tau*Sigma)^-1 * Pi + P^T Omega^-1 Q]
    
    try:
        tau_sigma = TAU * sigma
        inv_tau_sigma = np.linalg.inv(tau_sigma + np.eye(n_assets)*1e-6) # Add jitter for stability
        inv_omega = np.linalg.inv(Omega)
        
        # Term 1 (Inverse Variance)
        M_inverse = inv_tau_sigma + np.dot(np.dot(P.T, inv_omega), P)
        M = np.linalg.inv(M_inverse + np.eye(n_assets)*1e-6)
        
        # Term 2 (Weighted Sum of Prior and Views)
        term2 = np.dot(inv_tau_sigma, pi) + np.dot(np.dot(P.T, inv_omega), Q)
        
        # BL Returns
        bl_returns = np.dot(M, term2)
        
    except np.linalg.LinAlgError:
        # Fallback to pure Momentum if matrix inversion fails
        return np.zeros(n_assets)

    # --- OPTIMIZATION (MEAN-VARIANCE) ---
    # Maximize: w.T * bl_returns - (delta/2) * w.T * sigma * w
    
    def negative_utility(w):
        port_ret = np.dot(w, bl_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        # Constraint penalty included in utility to avoid unstable solvers
        return -(port_ret - (RISK_AVERSION_DELTA / 2) * (port_vol**2))

    # Constraints
    # 1. Sum of absolute weights <= GROSS_LEVERAGE (e.g. 1.2)
    cons = ({'type': 'ineq', 'fun': lambda x: GROSS_LEVERAGE - np.sum(np.abs(x))})
    
    # Bounds: -30% to +30% per asset
    bounds = tuple((-MAX_WEIGHT, MAX_WEIGHT) for _ in range(n_assets))
    
    # Initial Guess (Equal Weight based on sign of BL returns)
    init_guess = np.sign(bl_returns) * (1/n_assets)
    
    try:
        res = minimize(negative_utility, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-6)
        return res.x
    except:
        return np.zeros(n_assets)

# ==============================================================================
# 5. BACKTEST
# ==============================================================================
print("3. Executing Backtest (2020-2025)...")

rebal_dates = returns_total.loc["2020-01-01":].resample('W-FRI').last().index
daily_dates = returns_total.loc["2020-01-01":].index
rebal_set = set(rebal_dates)

current_weights_daily = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

for i, d in enumerate(daily_dates[1:]):
    # Rebalance
    if d in rebal_set:
        target_weights = get_black_litterman_weights(d)
        
        # Clean small weights (Noise reduction)
        target_weights[np.abs(target_weights) < 0.02] = 0
        
        turnover = np.sum(np.abs(target_weights - current_weights_daily))
        cost = turnover * TRANSACTION_COST
        current_weights_daily = target_weights
    else:
        cost = 0.0
        
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector) - cost
    
    capital *= (1 + port_ret)
    equity_curve.append(capital)
    equity_dates.append(d)
    strategy_returns.append(port_ret)
    
    if i % 100 == 0:
        print(f"Date: {d.date()} | Equity: {capital:,.0f} $")

# Export for Risk Analysis
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
df_export.to_csv(FILE_RETURNS)

# ==============================================================================
# 6. RESULTS
# ==============================================================================
df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
total_ret = (capital / TARGET_BUDGET) - 1
dd_series = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
max_dd = dd_series.min()

print("\n" + "="*60)
print(f"FINAL RESULTS: BLACK-LITTERMAN FX")
print("="*60)
print(f"💰 Final Capital : {capital:,.0f} $")
print(f"📈 Total Return  : {total_ret:+.2%}")
print(f"📉 Max Drawdown  : {max_dd:.2%}")
print("="*60)

# Visualization
def generate_visuals():
    # Yearly Subplots
    years = df_equity.index.year.unique()
    num_years = len(years)
    fig, axes = plt.subplots(num_years, 1, figsize=(12, 3 * num_years), sharex=False)
    
    for i, year in enumerate(years):
        data_year = df_equity[df_equity.index.year == year]['Equity']
        if data_year.empty: continue
        normalized = (data_year / data_year.iloc[0]) - 1
        ax = axes[i]
        ax.plot(normalized.index, normalized.values, color='teal', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"{year}", loc='left', fontsize=10, fontweight='bold')
        final_val = normalized.iloc[-1]
        ax.text(normalized.index[-1], final_val, f"{final_val:+.2%}", color='black', fontweight='bold', ha='left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'CHART_BL_YEARLY.png'), dpi=150)
    
    # Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(df_equity['Equity'], label='Black-Litterman Strategy', color='purple')
    plt.axhline(TARGET_BUDGET, color='black', linestyle='--')
    plt.title("Black-Litterman Optimization (Prior: Equal, View: Momentum)")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'CHART_BL_EQUITY.png'), dpi=150)
    plt.show()

if __name__ == "__main__":
    generate_visuals()