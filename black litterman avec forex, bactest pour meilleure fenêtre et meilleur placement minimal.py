import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import itertools

TARGET_BUDGET = 1000000.0  
RISK_AVERSION = 2.5     
TAU = 0.05              
CONFIDENCE_LEVEL = 0.95 

pairs_list_raw = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY"#, "GBPJPY", "EURGBP"
]
FOREX_TICKERS = [f"{p}=X" for p in pairs_list_raw]
BENCHMARK_TICKER = "DX-Y.NYB" 

print(f"1. CONFIG : BL Trend-Following | Mode: FOREX MAJORS | Budget ~{TARGET_BUDGET}€")

# --- 2. DONNÉES HISTORIQUES ---
print("2. TÉLÉCHARGEMENT DONNÉES FOREX...")
all_tickers = FOREX_TICKERS + [BENCHMARK_TICKER]
try:
    data = yf.download(all_tickers, start="2002-01-01", progress=False)['Close'].ffill()
    data = data.dropna(axis=1, how='all')
    
    if BENCHMARK_TICKER in data.columns:
        prices_bench = data[BENCHMARK_TICKER]
        data = data.drop(columns=[BENCHMARK_TICKER])
    else:
        print("Warning: Dollar Index introuvable, utilisation Moyenne Simple.")
        prices_bench = data.mean(axis=1)

    valid_tickers = [t for t in FOREX_TICKERS if t in data.columns]
    data = data[valid_tickers]
    last_date = data.index[-1]
    print(f"   Données OK : {len(valid_tickers)} paires chargées jusqu'au {last_date.date()}")
except Exception as e:
    print(f"Erreur Download: {e}")
    exit()

# --- 3. FONCTIONS MATHÉMATIQUES ---

def get_covariance_sim(asset_returns, benchmark_returns):
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10: return None 
    y = asset_returns.loc[common_idx]
    x = benchmark_returns.loc[common_idx]
    var_mkt = x.var()
    if var_mkt < 1e-6: return asset_returns.cov() 
    covs_xy = y.apply(lambda col: col.cov(x))
    betas = covs_xy / var_mkt
    beta_outer = np.outer(betas, betas) * var_mkt
    var_assets = y.var()
    idio_vars = var_assets - (betas ** 2) * var_mkt
    idio_vars = np.maximum(idio_vars, 0) 
    sim_cov = beta_outer + np.diag(idio_vars)
    return pd.DataFrame(sim_cov, index=asset_returns.columns, columns=asset_returns.columns)
def optimize_black_litterman(prices_train, prices_bench_train):
    """
    Version RSI MEAN REVERSION (Contrarienne Pure) :
    - RSI > 70 (Surachat) -> VENTE (Short)
    - RSI < 30 (Survente) -> ACHAT (Long)
    - Parfait pour les marchés en range.
    """
    valid = prices_train.dropna(axis=1, how='any')
    if valid.shape[1] < 2: return None
    
    returns = valid.pct_change(fill_method=None).dropna()
    if returns.empty: return None

    # 1. Paramètres de Risque (Covariance)
    # On prend tout l'historique disponible pour la stabilité
    cov_mat_annual = returns.cov() * 252
    
    n_assets = len(returns.columns)
    risk_aversion = 2.5
    tau = 0.05
    
    weights_eq = np.zeros((n_assets, 1))
    pi = risk_aversion * cov_mat_annual.dot(weights_eq)

    # 2. VUES (RSI 14 Périodes)
    pick_matrix = []   
    view_returns = [] 
    view_confidences = []

    # Calcul RSI Vectorisé (plus rapide et sûr)
    delta = valid.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    for i, asset in enumerate(returns.columns):
        if pd.isna(rsi[asset]): continue
        
        rsi_val = rsi[asset]
        signal = 0.0
        
        # Logique Contrarienne RSI
        if rsi_val > 70: # Suroptimisme -> On Short
            # Plus on est haut, plus on short fort
            signal = -1.0 * (rsi_val - 70) / 30.0 
        elif rsi_val < 30: # Pessimisme -> On Achète
            signal = 1.0 * (30 - rsi_val) / 30.0
            
        # Filtre : On ne joue que si le signal est significatif
        if abs(signal) > 0.1:
            p_row = np.zeros(n_assets)
            p_row[i] = 1 
            pick_matrix.append(p_row)
            
            # Vue : 20% de retour espéré si le signal est max (1.0)
            expected_ret = np.clip(signal * 0.20, -0.20, 0.20)
            view_returns.append(expected_ret)
            
            asset_vol = np.sqrt(cov_mat_annual.iloc[i, i])
            view_confidences.append((asset_vol * tau) ** 2)

    # 3. Calcul BL
    if len(pick_matrix) > 0:
        P = np.array(pick_matrix)
        Q = np.array(view_returns).reshape(-1, 1)
        Omega = np.diag(view_confidences)
        
        try:
            tau_cov = tau * cov_mat_annual
            inv_tau_cov = np.linalg.inv(tau_cov + np.eye(n_assets)*1e-6)
            inv_omega = np.linalg.inv(Omega)
            
            left = inv_tau_cov + P.T.dot(inv_omega).dot(P)
            minv = np.linalg.inv(left + np.eye(n_assets)*1e-6)
            right = inv_tau_cov.dot(pi) + P.T.dot(inv_omega).dot(Q)
            
            bl_returns = minv.dot(right).flatten()
        except:
            bl_returns = pi.flatten()
    else:
        return None # Pas de signal extrême -> Cash

    # 4. Optimisation
    def neg_sharpe(w):
        p_ret = np.sum(bl_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_annual, w)))
        if p_vol < 1e-6: return 0
        return - (p_ret) / p_vol

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
    bounds = tuple((-1.0, 1.0) for _ in range(n_assets)) 
    
    try:
        init_guess = np.sign(bl_returns) * (1/n_assets)
        if np.all(init_guess == 0): init_guess = [1/n_assets] * n_assets
        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        return pd.Series(res.x, index=returns.columns).sort_values(ascending=False)
    except:
        return None
def clean_portfolio(allocations, threshold):
    if allocations is None: return None
    # On garde si la valeur ABSOLUE est supérieure au seuil
    clean = allocations[allocations.abs() >= threshold]
    if clean.empty: return None
    
    # Renormalisation pour que la somme des valeurs absolues soit 1
    # (Gross leverage = 1)
    total_exposure = clean.abs().sum()
    if total_exposure == 0: return None
    
    return clean / total_exposure

def calculate_risk_metrics(returns_series, confidence=0.95):
    if returns_series.empty: return np.nan, np.nan
    var = np.percentile(returns_series, (1 - confidence) * 100)
    cvar = returns_series[returns_series <= var].mean()
    if np.isnan(cvar): cvar = var 
    return var, cvar
# --- 4. GRID SEARCH HYPER-PARAMÈTRES (2010-2022) ---

def backtest_parameters(window_y, threshold_pct, start_y=2010, end_y=2020):
    """ Simule la stratégie avec prise en compte du Spread (Frais) """
    capital = 100.0
    current_year = start_y
    
    while current_year <= end_y:
        test_start = f"{current_year}-01-01"
        test_end = f"{current_year}-12-31"
        test_start_dt = pd.Timestamp(test_start)
        start_train = test_start_dt - pd.DateOffset(months=int(window_y * 12))
        end_train = test_start_dt - pd.DateOffset(days=1)
        
        train_slice = data.loc[start_train:end_train]
        bench_slice = prices_bench.loc[start_train:end_train]
        test_slice = data.loc[test_start:test_end]
        
        if test_slice.empty: break
        
        # Optimisation
        alloc = optimize_black_litterman(train_slice, bench_slice)
        alloc = clean_portfolio(alloc, threshold_pct)
        
        if alloc is not None:
            daily_returns = test_slice[alloc.index].pct_change(fill_method=None).dropna()
            port_ret = daily_returns.dot(alloc.values)
            
            # --- REALISME : COUT DE TRANSACTION (SPREAD) ---
            # On simule un coût de 0.05% à l'entrée en position (début d'année)
            if not port_ret.empty:
                port_ret.iloc[0] -= 0.0005 

            year_perf = (1 + port_ret).prod() - 1
            capital *= (1 + year_perf)
        
        current_year += 1
        
    return capital

print("\n" + "█"*80)
print("3. RECHERCHE DES MEILLEURS PARAMÈTRES (GRID SEARCH 2010-2022)")
print("   Objectif : Maximiser la performance Réaliste (Net de frais)")
print("█"*80)

# Nouvelle grille : Plus courte (0.5 an à 3 ans) car l'algo suit la tendance
windows_test = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,2.5, 2.75,3.0,3.25,3.5,3.75,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8] 
filters_test = [0.00] # On garde 0% pour laisser le BL décider

best_score = -999
best_params = (1.5, 0.00) 

print(f"{'WINDOW (Ans)':<15} | {'FILTRE (%)':<15} | {'CAPITAL FINAL (Base 100)':<25}")
print("-" * 65)

for w in windows_test:
    for f in filters_test:
        try:
            final_cap = backtest_parameters(w, f, 2010, 2022)
            print(f"{w:<15} | {f*100:<15.0f} | {final_cap:.2f} €")
            
            if final_cap > best_score:
                best_score = final_cap
                best_params = (w, f)
        except Exception as e:
            continue

print("-" * 65)
print(f"🏆 VAINQUEUR : Window={best_params[0]} ans | Filtre={best_params[1]*100:.0f}%")
WINDOW_YEARS = best_params[0]
SEUIL_FILTRE = best_params[1]

# --- 5. BACKTEST VISUEL ENGINE (Adapté) ---

def run_scenario(scenario_id, name, start_year, end_year):
    print(f"\n[{scenario_id}] {name} (Rebalancement Hebdomadaire)...")
    
    # --- PRÉPARATION TEMPORELLE ---
    # On génère les dates de rebalancement (Tous les Vendredis)
    # On prend large sur les dates pour couvrir la période
    full_dates = prices_bench.loc[f"{start_year}-01-01":f"{end_year}-12-31"].index
    if full_dates.empty: return
    
    # On crée une série temporelle hebdomadaire
    weekly_dates = prices_bench.loc[full_dates[0]:full_dates[-1]].resample('W-FRI').last().index
    
    # Capitaux (Base 100 USD pour simplifier la comparaison)
    cap_strat = 100.0
    cap_bench = 100.0
    
    # Pour le graphique
    dates = [weekly_dates[0]]
    strat_curve = [100.0]
    bench_curve = [100.0]
    
    current_alloc = None
    
    print(f"{'DATE':<12} | {'BENCH':<8} | {'STRAT':<8} | {'TOP POSITIONS (Long/Short)'}")
    print("-" * 100)

    # BOUCLE SEMAINE PAR SEMAINE
    for i in range(len(weekly_dates) - 1):
        curr_date = weekly_dates[i]
        next_date = weekly_dates[i+1]
        
        # --- A. OPTIMISATION (Le Cerveau) ---
        # On regarde le passé (Fenetre glissante définie par WINDOW_YEARS)
        train_start = curr_date - pd.DateOffset(months=int(WINDOW_YEARS * 12))
        
        # Données d'entraînement s'arrêtant STRICTEMENT à la date actuelle (pas de biais futur)
        train_data = data.loc[train_start:curr_date]
        bench_train = prices_bench.loc[train_start:curr_date]
        
        # On recalcule les poids optimaux pour la semaine à venir
        # On utilise ta fonction optimize_black_litterman existante
        new_alloc = optimize_black_litterman(train_data, bench_train)
        
        if new_alloc is not None:
            # On nettoie et on applique
            current_alloc = clean_portfolio(new_alloc, SEUIL_FILTRE)
        
        # --- B. SIMULATION (Le Réel) ---
        # On regarde ce qui se passe la semaine suivante
        week_prices = data.loc[curr_date:next_date]
        bench_prices = prices_bench.loc[curr_date:next_date]
        
        if week_prices.empty or len(week_prices) < 2: continue
        
        # Rendement de la semaine (Prix Fin / Prix Début - 1)
        # Benchmark
        b_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0]) - 1
        cap_bench *= (1 + b_ret)
        
        # Stratégie
        s_ret = 0.0
        pos_str = "CASH"
        
        if current_alloc is not None:
            # Rendement de chaque actif durant la semaine
            asset_rets = (week_prices.iloc[-1] / week_prices.iloc[0]) - 1
            
            # Alignement des index pour le produit scalaire
            common = asset_rets.index.intersection(current_alloc.index)
            if not common.empty:
                # Performance = Somme (Poids * Rendement)
                # Note: Si Poids est négatif (Short) et Rendement négatif (Baisse), on gagne (+)
                s_ret = asset_rets[common].dot(current_alloc[common])
                
                # --- COSMETIQUE (Affichage) ---
                longs = current_alloc[current_alloc > 0.1].index.tolist()
                shorts = current_alloc[current_alloc < -0.1].index.tolist()
                
                l_txt = " ".join([t.replace('=X','') for t in longs[:2]])
                s_txt = " ".join([t.replace('=X','') for t in shorts[:2]])
                
                if l_txt or s_txt:
                    pos_str = f"L:[{l_txt}] S:[{s_txt}]"
                else:
                    pos_str = "Div." # Diversifié

        # Mise à jour Capital Stratégie
        cap_strat *= (1 + s_ret)
        if 'high_water_mark' not in locals(): high_water_mark = 100.0
        
        if cap_strat > high_water_mark:
            high_water_mark = cap_strat
            
        drawdown = (cap_strat - high_water_mark) / high_water_mark
        
        # Si on perd plus de 5% depuis le sommet -> Sécurité CASH
        if drawdown < -0.05:
            pos_str += " [⚠️ DD > 5%]"
        # Stockage
        strat_curve.append(cap_strat)
        bench_curve.append(cap_bench)
        dates.append(next_date)
        
        # Affichage une fois par mois pour ne pas saturer la console (toutes les 4 semaines)
        if i % 4 == 0:
            print(f"{curr_date.date()} | {cap_bench:<8.1f} | {cap_strat:<8.1f} | {pos_str}")

    print("-" * 100)
    print(f"FINAL : Stratégie {cap_strat:.1f} $ vs Benchmark {cap_bench:.1f} $")
    
    # Graphique final
    plt.figure(figsize=(12, 6))
    plt.plot(dates, strat_curve, label='Stratégie (Weekly Rebal.)', color='blue', linewidth=1.5)
    plt.plot(dates, bench_curve, label='Benchmark (DXY)', color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Performance Hebdomadaire : {start_year}-{end_year}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- 6. OPTIMISEUR ENTIER (LOGIQUE CONSERVÉE) ---
def smart_integer_optimizer(allocation, current_prices, target_budget):
    # Note: Sur le Forex, "shares" représente le nombre d'unités (ex: 1000 EURUSD)
    results = []
    tickers = allocation.index.tolist()
    target_weights = allocation.values
    
    # On calcule les montants idéaux en Valeur Absolue pour respecter l'exposition
    # Si le poids est négatif, le montant sera négatif
    ideal_amounts = target_weights * target_budget
    ideal_shares = ideal_amounts / current_prices[tickers].values
    
    ranges = []
    for share_count in ideal_shares:
        floor_s = math.floor(share_count)
        ceil_s = math.ceil(share_count)
        # Gestion du cas zéro ou proche de zéro
        if abs(share_count) < 0.1:
             ranges.append([0])
        else:
            ranges.append(list(set([floor_s, ceil_s])))
    
    # Si trop de combinaisons, on arrondit simplement (fallback)
    if len(ranges) > 12:
         shares = np.round(ideal_shares)
         # Calcul de l'exposition brute (Somme des valeurs absolues)
         total_exposure = np.sum(np.abs(shares * current_prices[tickers].values))
         return {"shares": shares, "cost": total_exposure, "score": 0}

    for combination in itertools.product(*ranges):
        shares = np.array(combination)
        # C'EST ICI LA CORRECTION : On somme les valeurs ABSOLUES
        gross_exposure = np.sum(np.abs(shares * current_prices[tickers].values))
        
        # On vérifie qu'on est proche du budget en exposition (levier 1)
        if gross_exposure < target_budget * 0.95 or gross_exposure > target_budget * 1.05: continue
        
        # Poids réels basés sur l'exposition brute
        real_weights = (shares * current_prices[tickers].values) / gross_exposure
        
        # Erreur par rapport aux poids cibles (en valeur absolue aussi)
        weight_error = np.sum(np.abs(real_weights - target_weights))
        budget_error = abs(gross_exposure - target_budget) / target_budget
        
        score = (2.0 * weight_error) + (1.0 * budget_error)
        
        results.append({"shares": shares, "cost": gross_exposure, "score": score, "real_weights": real_weights})
        
    if not results: return None
    return sorted(results, key=lambda x: x["score"])[0]
# --- 7. EXÉCUTION ---
print("4. LANCEMENT DU BACKTEST AVEC PARAMÈTRES OPTIMISÉS...")

# On lance uniquement sur la période récente (2023-2025) car 2005-2022 a servi à l'entrainement des paramètres
scenarios_list = [
    {"id": 1, "years": (2021, 2025), "desc": f"FOREX | Window={WINDOW_YEARS}y | Filtre={SEUIL_FILTRE*100:.0f}%"}
]

for s in scenarios_list:
    run_scenario(s["id"], s["desc"], s["years"][0], s["years"][1])
# --- 8. PRÉVISIONS 2026 ---
print("\n" + "█"*80)
print(f"🔮 PRÉVISIONS 2026 : BL + SIM (Optimisé Forex)")
print("█"*80)

# Pour les prévisions live, on utilise les données jusqu'à la fin
end_train_26 = last_date
start_train_26 = last_date - pd.DateOffset(months=int(WINDOW_YEARS * 12))
train_data_26 = data.loc[start_train_26:end_train_26]
bench_train_26 = prices_bench.loc[start_train_26:end_train_26]

alloc_base = optimize_black_litterman(train_data_26, bench_train_26)

if alloc_base is not None:
    # On filtre un peu plus fort pour les ordres réels (éviter les micro-positions)
    alloc_conviction = clean_portfolio(alloc_base, 0.02) 
    
    if alloc_conviction is not None:
        current_prices = data.iloc[-1][alloc_conviction.index]
        
        # Calcul du plan en Dollars (USD) car c'est la monnaie de référence Forex
        best_plan = smart_integer_optimizer(alloc_conviction, current_prices, TARGET_BUDGET)
        
        print("\n" + "█"*80)
        print(f"C. PLAN D'ACHAT (Unités) 2026 (Budget ~{TARGET_BUDGET}$)")
        print("█"*80)
        
        if best_plan:
            print(f"   Exposition Totale : {best_plan['cost']:.2f} $")
            print("-" * 90)
            print(f"{'ACTION':<13} | {'PAIRE':<10} | {'COURS':<10} | {'UNITÉS':<10} | {'VALEUR ($)':<12} | {'POIDS'}")
            print("-" * 90)
            
            tickers = alloc_conviction.index.tolist()
            shares = best_plan["shares"]
            
            for i, t in enumerate(tickers):
                n = int(shares[i])
                if n == 0: continue
                
                p = current_prices[t]
                amt = n * p # Valeur nominale en devise de cotation (souvent USD)
                w_real = abs(amt) / best_plan['cost']
                
                action = "🟢 ACHAT" if n > 0 else "🔴 VENTE"
                
                # Note: On affiche € ou $ selon ton budget, mais ici on met $ pour la logique
                print(f"{action:<13} | {t.replace('=X',''):<10} | {p:<10.4f} | {n:<10} | {amt:<12.2f} $ | {w_real*100:.1f}%")
            print("-" * 90)
        else:
            print("Budget trop serré pour respecter les contraintes.")
else:
    print("Pas de tendance claire détectée (Cash).")
scenarios_list = [
    {"id": 1, "years": (2010, 2020), "desc": f"FOREX | Window={WINDOW_YEARS}y | Filtre={SEUIL_FILTRE*100:.0f}%"}
]
for s in scenarios_list:
    run_scenario(s["id"], s["desc"], s["years"][0], s["years"][1])
plt.show()