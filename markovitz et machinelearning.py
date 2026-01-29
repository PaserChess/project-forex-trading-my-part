import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis
import os
import warnings

# ==============================================================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# --- FOLDER MANAGEMENT ---
OUTPUT_DIR = "backtest_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f">>> Created output directory: {OUTPUT_DIR}")

# Strategy Parameters
TARGET_BUDGET = 1000000.0   
LEVERAGE_PER_POS = 0.20     # 20% per position to manage drawdown
TRANSACTION_COST = 0.0005   # 5 bps
LOOKBACK_MOMENTUM = 252     # 1 Year Lookback
# File path is now inside the folder
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS.csv")

# Risk Parameters
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

print(">>> INITIALIZING GLOBAL MACRO MOMENTUM STRATEGY (WEEKLY REBALANCING)")

# ==============================================================================
# 2. DATA DOWNLOAD & RATE SIMULATION
# ==============================================================================
print("1. Downloading Price Data and Simulating Macro Rates...")

data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)['Close'].ffill().dropna()

# Simulated Macro Rates (to calculate Carry/Swaps)
fred_codes = {k: 'SIM' for k in ['USD','EUR','JPY','GBP','CHF','AUD','CAD','NZD','MXN','ZAR','BRL','TRY','SGD','SEK','NOK','INR']}

def generate_macro_rates(price_index):
    dates = price_index
    rates = pd.DataFrame(index=dates, columns=fred_codes.keys())
    rates[:] = 0.0 
    
    # Base rates
    rates['USD'] = 0.015; rates['EUR'] = 0.0; rates['JPY'] = -0.001
    rates['GBP'] = 0.0075; rates['CHF'] = -0.0075; rates['AUD'] = 0.015
    rates['MXN'] = 0.07; rates['TRY'] = 0.12; rates['ZAR'] = 0.06
    
    # Inflation shock (2022+)
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
# 3. TOTAL RETURN CALCULATION (PRICE + SWAP)
# ==============================================================================
print("2. Calculating Total Returns (Price + Swap) and Momentum Scores...")

returns_total = pd.DataFrame(index=data.index, columns=data.columns)
momentum_score = pd.DataFrame(index=data.index, columns=data.columns)

for t in tickers:
    clean = t.replace('=X','')
    base, quote = clean[:3], clean[3:]
    
    r_price = data[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = rates_daily[base] - rates_daily[quote]
    
    # CRITICAL: This benchmark data now includes the swap rate
    total_daily_ret = r_price + r_carry
    returns_total[t] = total_daily_ret
    
    # Momentum 12 Months
    log_rets = np.log(1 + total_daily_ret)
    momentum_score[t] = log_rets.rolling(LOOKBACK_MOMENTUM).sum().shift(1)

returns_total.dropna(inplace=True)
momentum_score.dropna(inplace=True)

# ==============================================================================
# 4. BACKTEST ENGINE
# ==============================================================================
def allocation_system(curr_date):
    if curr_date not in momentum_score.index: return np.zeros(len(tickers))
    
    scores = momentum_score.loc[curr_date]
    ranked = scores.sort_values(ascending=False)
    weights = np.zeros(len(tickers))
    
    # Top 3 Longs / Top 3 Shorts
    top_3 = ranked.head(3).index
    bottom_3 = ranked.tail(3).index
    
    for t in top_3:
        weights[tickers.index(t)] = LEVERAGE_PER_POS 
        
    for t in bottom_3:
        idx = tickers.index(t)
        # Short only if momentum is negative
        if scores[t] < 0: weights[idx] = -LEVERAGE_PER_POS
        else: weights[idx] = 0.0
            
    return weights

print("3. Executing Backtest (2020-2025)...")

rebal_dates = returns_total.loc["2020-01-01":].resample('W-FRI').last().index
strategy_returns = [] 
daily_dates = returns_total.loc["2020-01-01":].index
current_weights_daily = np.zeros(len(tickers))
rebal_set = set(rebal_dates)
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]

for i, d in enumerate(daily_dates[1:]):
    if d in rebal_set:
        target_weights = allocation_system(d)
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

# Align DataFrames
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
# Save to folder
df_export.to_csv(FILE_RETURNS)
print(f"-> Saved returns data to: {FILE_RETURNS}")

# Create Equity DataFrame for Reporting
df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
df_equity['Returns'] = df_equity['Equity'].pct_change()

# ==============================================================================
# 5. YEARLY PERFORMANCE REPORTING
# ==============================================================================
print("\n" + "="*60)
print("YEAR-OVER-YEAR PERFORMANCE REPORT")
print("="*60)

# Resample to Yearly
yearly_equity = df_equity['Equity'].resample('YE').last()
yearly_returns = yearly_equity.pct_change()
# Handle first year manually
first_year_ret = (yearly_equity.iloc[0] / TARGET_BUDGET) - 1
yearly_returns.iloc[0] = first_year_ret

for date, ret in yearly_returns.items():
    print(f"Year {date.year}: {ret:+.2%}")

total_ret = (capital / TARGET_BUDGET) - 1
print("-" * 30)
print(f"TOTAL RETURN: {total_ret:+.2%}")
print("="*60)

# ==============================================================================
# 6. VISUALIZATION (SUBPLOTS & RISK)
# ==============================================================================

def generate_visuals():
    if not os.path.exists(FILE_RETURNS): return
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    
    # --- CHART 1: YEARLY PERFORMANCE SUBPLOTS ---
    years = df_equity.index.year.unique()
    num_years = len(years)
    
    fig, axes = plt.subplots(num_years, 1, figsize=(12, 3 * num_years), sharex=False)
    
    for i, year in enumerate(years):
        # Extract data for the year
        data_year = df_equity[df_equity.index.year == year]['Equity']
        if data_year.empty: continue
        
        # Normalize to 0% start for visualization
        normalized = (data_year / data_year.iloc[0]) - 1
        
        ax = axes[i]
        ax.plot(normalized.index, normalized.values, color='tab:blue', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"{year} Performance", loc='left', fontsize=10, fontweight='bold')
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3)
        
        # Add final return annotation
        final_val = normalized.iloc[-1]
        color_text = 'green' if final_val > 0 else 'red'
        ax.text(normalized.index[-1], final_val, f"{final_val:+.2%}", color=color_text, fontweight='bold', ha='left')

    plt.tight_layout()
    plt.suptitle("Strategy Performance by Year", y=1.02, fontsize=14)
    # Save to folder
    save_path = os.path.join(OUTPUT_DIR, 'CHART_YEARLY_SUBPLOTS.png')
    plt.savefig(save_path, dpi=150)
    print(f"-> Generated {save_path}")

    # --- CHART 2: RISK VS RETURN (TOTAL RETURN) ---
    summary = pd.DataFrame()
    # Ensure we are using Total Return (Price + Swap)
    summary['Returns'] = returns.mean() * 252
    summary['Volatility'] = returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(12, 8))
    
    # Plot Assets
    subset = summary.drop('STRATEGY', errors='ignore')
    sns.scatterplot(data=subset, x='Volatility', y='Returns', s=100, color='gray', alpha=0.5, label='FX Pairs (Total Return)')
    
    # Plot Strategy
    if 'STRATEGY' in summary.index:
        strat = summary.loc['STRATEGY']
        plt.scatter(strat['Volatility'], strat['Returns'], s=250, color='red', edgecolors='black', label='STRATEGY')
        plt.text(strat['Volatility']+0.002, strat['Returns'], "  MY STRATEGY", fontsize=10, fontweight='bold', color='red')

    # Add labels
    for i in range(len(subset)):
        plt.text(subset.iloc[i]['Volatility'], subset.iloc[i]['Returns'], subset.index[i], fontsize=8, alpha=0.7)

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Risk/Return Profile (Including Swap Rates)', fontsize=14)
    plt.xlabel('Annualized Volatility', fontsize=12)
    plt.ylabel('Annualized Total Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Save to folder
    save_path = os.path.join(OUTPUT_DIR, 'CHART_RISK_RETURN.png')
    plt.savefig(save_path, dpi=150)
    print(f"-> Generated {save_path}")

    # --- CHART 3: DRAWDOWNS ---
    plt.figure(figsize=(12, 6))
    cum_ret = (1 + returns['STRATEGY']).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    
    plt.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.2)
    plt.title('Drawdown Analysis', fontsize=14)
    plt.ylabel('Drawdown', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Save to folder
    save_path = os.path.join(OUTPUT_DIR, 'CHART_DRAWDOWN.png')
    plt.savefig(save_path, dpi=150)
    print(f"-> Generated {save_path}")

    # Display Charts
    plt.show()

if __name__ == "__main__":
    generate_visuals()