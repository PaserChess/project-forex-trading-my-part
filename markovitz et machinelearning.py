import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis
import os
import warnings

# ==============================================================================
# 1. CONFIGURATION GLOBALE
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# --- Paramètres Stratégie ---
TARGET_BUDGET = 1000000.0   
LEVERAGE_PER_POS = 0.20     # Levier réduit à 20% par pos (pour réduire le DD)
TRANSACTION_COST = 0.0005   
LOOKBACK_MOMENTUM = 252     
FILE_RETURNS = "MASTER_RETURNS.csv" # Le fichier pont entre Stratégie et Risque

# --- Paramètres Risk Management ---
CONFIDENCE_LEVEL = 0.95  
RISK_FREE_RATE = 0.0     
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

print(f" PARTIE 1 : BACKTEST 'GLOBAL MACRO MOMENTUM' (WEEKLY)")

# ==============================================================================
# 2. DATA & TAUX
# ==============================================================================
print("\n📥 1. Téléchargement Données & Simulation Taux...")
# Threading False pour éviter les timeouts
data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)['Close'].ffill().dropna()

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
# 3. CALCUL MOMENTUM & RENDEMENTS
# ==============================================================================
print(" 2. Calcul des Rendements Totaux & Scores...")

returns_total = pd.DataFrame(index=data.index, columns=data.columns)
momentum_score = pd.DataFrame(index=data.index, columns=data.columns)

for t in tickers:
    clean = t.replace('=X','')
    base, quote = clean[:3], clean[3:]
    
    r_price = data[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = rates_daily[base] - rates_daily[quote]
    
    total_daily_ret = r_price + r_carry
    returns_total[t] = total_daily_ret
    
    # Momentum 12 Mois
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
        if scores[t] < 0: weights[idx] = -LEVERAGE_PER_POS
        else: weights[idx] = 0.0
            
    return weights

print("\n 3. Exécution Backtest (2020-2025)...")

rebal_dates = returns_total.loc["2020-01-01":].resample('W-FRI').last().index
curr_weights = np.zeros(len(tickers))
strategy_returns = [] # Pour stocker les rendements de la strat jour par jour

# On parcourt jour par jour pour avoir une série temporelle propre pour le Risk Management
daily_dates = returns_total.loc["2020-01-01":].index
current_weights_daily = np.zeros(len(tickers))

# Dictionnaire pour savoir quand rebalancer
rebal_set = set(rebal_dates)

capital = TARGET_BUDGET

for d in daily_dates:
    # 1. Rebalancement si on est un vendredi
    if d in rebal_set:
        target_weights = allocation_system(d)
        # Coûts
        turnover = np.sum(np.abs(target_weights - current_weights_daily))
        cost = turnover * TRANSACTION_COST
        # On applique le coût au rendement du jour (simplification)
        current_weights_daily = target_weights
    else:
        cost = 0.0
        
    # 2. Calcul PnL du jour
    # Rendement du jour * Poids de la veille
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector) - cost
    
    strategy_returns.append(port_ret)
    capital *= (1 + port_ret)

# Création du fichier MASTER_RETURNS.csv
# On combine les rendements des Paires + La Stratégie
df_export = returns_total.loc[daily_dates].copy()
df_export['STRATEGY'] = strategy_returns # On ajoute notre algo comme un actif
df_export.to_csv(FILE_RETURNS)

print(f" Backtest terminé. Capital Final: {capital:,.0f} $")
print(f" Données sauvegardées dans '{FILE_RETURNS}' pour l'analyse de risque.")


# ==============================================================================
# ==============================================================================
# PARTIE 2 : MODULE DE RISK MANAGEMENT (Ton Code Intégré)
# ==============================================================================
# ==============================================================================

def run_full_analysis():
    print(" PARTIE 2 : ANALYSE DE RISQUE (QUANTITATIVE)")
    
    # 1. CHARGEMENT
    if not os.path.exists(FILE_RETURNS):
        print(f"Erreur : {FILE_RETURNS} introuvable.")
        return
    
    df_ret = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    # Nettoyage des données infinies ou NaN
    df_ret = df_ret.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f" Analyse sur {df_ret.shape[1]} actifs (incluant 'STRATEGY') sur {df_ret.shape[0]} jours.")

    # ---------------------------------------------------------
    # BLOC 1 : VOLATILITÉ
    # ---------------------------------------------------------
    print("\n--- BLOC 1 : ANALYSE DE LA VOLATILITÉ ---")
    vol_ann = df_ret.std() * np.sqrt(TRADING_DAYS)
    
    downside_returns = df_ret[df_ret < 0]
    semi_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
    
    # EWMA (Volatilité pondérée)
    ewma_vol = df_ret.ewm(alpha=0.06).std().iloc[-1] * np.sqrt(TRADING_DAYS)

    df_block1 = pd.DataFrame({
        'Volatilité (Ann)': vol_ann,
        'Semi-Déviation': semi_dev,
        'EWMA (Actuelle)': ewma_vol,
        'Ratio Symétrie': vol_ann / semi_dev
    })

    # ---------------------------------------------------------
    # BLOC 2 : TAIL RISK (VaR & CVaR)
    # ---------------------------------------------------------
    print("--- BLOC 2 : ANALYSE DES QUEUES (VaR & CVaR) ---")
    
    # VaR Historique
    var_hist = df_ret.quantile(1 - CONFIDENCE_LEVEL)
    
    # VaR Paramétrique
    mu = df_ret.mean()
    sigma = df_ret.std()
    z_score = norm.ppf(1 - CONFIDENCE_LEVEL)
    var_param = mu + z_score * sigma
    
    # VaR Monte Carlo
    var_mc = []
    # On réduit les simus si trop lent, mais 10000 c'est bien
    for col in df_ret.columns:
        simulated_rets = np.random.normal(mu[col], sigma[col], MC_SIMULATIONS)
        var_mc.append(np.percentile(simulated_rets, (1 - CONFIDENCE_LEVEL) * 100))
    var_mc_series = pd.Series(var_mc, index=df_ret.columns)

    # CVaR
    cvar_list = []
    for col in df_ret.columns:
        cutoff = var_hist[col]
        tail_losses = df_ret[col][df_ret[col] <= cutoff]
        cvar_list.append(tail_losses.mean() if not tail_losses.empty else 0)
    cvar_series = pd.Series(cvar_list, index=df_ret.columns)

    df_block2 = pd.DataFrame({
        f'VaR Hist {CONFIDENCE_LEVEL:.0%}': var_hist,
        f'VaR Param': var_param,
        f'VaR MonteCarlo': var_mc_series,
        f'CVaR': cvar_series
    })

    # ---------------------------------------------------------
    # BLOC 3 : PERFORMANCE & CORRÉLATION
    # ---------------------------------------------------------
    print("--- BLOC 3 : PERFORMANCE & RATIOS ---")
    
    mean_ret_ann = df_ret.mean() * TRADING_DAYS
    sharpe = (mean_ret_ann - RISK_FREE_RATE) / vol_ann
    sortino = (mean_ret_ann - RISK_FREE_RATE) / semi_dev
    
    # Beta vs 'STRATEGY' (pour voir la contribution de chaque devise à la strat)
    market_index = df_ret['STRATEGY']
    market_var = market_index.var()
    betas = []
    for col in df_ret.columns:
        cov = df_ret[col].cov(market_index)
        beta = cov / market_var if market_var != 0 else 0
        betas.append(beta)
    beta_series = pd.Series(betas, index=df_ret.columns)

    df_block3 = pd.DataFrame({
        'Rendement (Ann)': mean_ret_ann,
        'Ratio Sharpe': sharpe,
        'Ratio Sortino': sortino,
        'Beta (vs Strategy)': beta_series
    })

    # ---------------------------------------------------------
    # BLOC 4 : DISTRIBUTION & DRAWDOWNS
    # ---------------------------------------------------------
    print("--- BLOC 4 : TRAJECTOIRES & DRAWDOWNS ---")
    
    skew_s = df_ret.apply(skew)
    kurt_s = df_ret.apply(kurtosis)
    
    nav = (1 + df_ret).cumprod() * 100
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar = mean_ret_ann / abs(max_drawdown)
    
    df_block4 = pd.DataFrame({
        'Skewness': skew_s,
        'Kurtosis': kurt_s,
        'Max Drawdown': max_drawdown,
        'Ratio Calmar': calmar
    })

    # ---------------------------------------------------------
    # SAUVEGARDE
    # ---------------------------------------------------------
    global_report = pd.concat([df_block1, df_block2, df_block3, df_block4], axis=1)
    global_report = global_report.sort_values(by='Ratio Sharpe', ascending=False)
    
    global_report.to_csv("RAPPORT_GLOBAL_RISQUE.csv")
    print(f"\n Rapport sauvegardé : RAPPORT_GLOBAL_RISQUE.csv")
    
    # Focus sur la Stratégie
    print("\n--- PERFORMANCE DE LA STRATÉGIE ---")
    print(global_report.loc[['STRATEGY']].T)

def generate_visuals():
    if not os.path.exists(FILE_RETURNS): return

    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    print("\n Génération des graphiques...")

    # 1. HEATMAP (Sans la Stratégie pour voir les devises entre elles)
    plt.figure(figsize=(12, 10))
    ret_assets = returns.drop(columns=['STRATEGY'], errors='ignore')
    corr = ret_assets.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".1f", annot_kws={"size": 7})
    plt.title('Corrélation Croisée des Devises', fontsize=16)
    plt.tight_layout()
    plt.savefig('CHART_1_CORRELATION.png', dpi=150)

    # 2. RISK vs RETURN
    summary = pd.DataFrame()
    summary['Returns'] = returns.mean() * 252
    summary['Volatility'] = returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(12, 8))
    # Toutes les devises en bleu
    sns.scatterplot(data=summary.drop('STRATEGY', errors='ignore'), x='Volatility', y='Returns', s=100, color='gray', alpha=0.5, label='Paires FX')
    # La Stratégie en ROUGE GROS
    if 'STRATEGY' in summary.index:
        strat = summary.loc['STRATEGY']
        plt.scatter(strat['Volatility'], strat['Returns'], s=300, color='red', label='TA STRATÉGIE', edgecolors='black')
        plt.text(strat['Volatility']+0.005, strat['Returns'], "STRATEGY", fontsize=12, fontweight='bold', color='red')

    for i in range(len(summary)):
        if summary.index[i] != 'STRATEGY':
            plt.text(summary.Volatility[i], summary.Returns[i], summary.index[i], fontsize=8, alpha=0.7)

    plt.axhline(0, color='black', linestyle='--')
    plt.title('Efficient Frontier : Ta Stratégie vs Le Marché', fontsize=16)
    plt.xlabel('Risque (Volatilité)', fontsize=12)
    plt.ylabel('Rendement', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('CHART_2_RISK_RETURN.png', dpi=150)

    # 3. DISTRIBUTION STRATÉGIE (Histogramme)
    plt.figure(figsize=(12, 6))
    if 'STRATEGY' in returns.columns:
        sns.histplot(returns['STRATEGY'], bins=50, kde=True, color='blue')
        plt.axvline(returns['STRATEGY'].mean(), color='red', linestyle='--', label='Moyenne')
        # VaR Line
        var_95 = returns['STRATEGY'].quantile(0.05)
        plt.axvline(var_95, color='orange', linestyle='--', label=f'VaR 95% ({var_95:.2%})')
        plt.title('Distribution des Rendements Hebdomadaires de la Stratégie', fontsize=16)
        plt.legend()
    plt.savefig('CHART_3_DISTRIBUTION.png', dpi=150)

    # 4. DRAWDOWNS
    plt.figure(figsize=(12, 6))
    cum_ret = (1 + returns['STRATEGY']).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    plt.plot(drawdown.index, drawdown, label='Drawdown Stratégie', color='red')
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.title('Analyse du Drawdown (Profondeur des pertes)', fontsize=16)
    plt.ylabel('Drawdown', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('CHART_4_DRAWDOWN.png', dpi=150)
    
    print("-> 4 Graphiques générés.")

if __name__ == "__main__":
    run_full_analysis()
    generate_visuals()
    plt.show()