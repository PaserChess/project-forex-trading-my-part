import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
FILE_RETURNS = "MASTER_RETURNS.csv"
CONFIDENCE_LEVEL = 0.95  # Pour la VaR (95%)
RISK_FREE_RATE = 0.0     # Taux sans risque (0% pour simplifier sur le Forex)
TRADING_DAYS = 252       # Jours de trading par an
MC_SIMULATIONS = 10000   # Nombre de simulations pour Monte Carlo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10) 
def run_full_analysis():
    print(">>> DÉMARRAGE DE L'ANALYSE COMPLÈTE (4 BLOCS) <<<")
    
    # 1. CHARGEMENT
    if not os.path.exists(FILE_RETURNS):
        print(f"Erreur : {FILE_RETURNS} introuvable.")
        return
    
    df_ret = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    print(f"Données chargées : {df_ret.shape[1]} devises sur {df_ret.shape[0]} jours.")

    # ---------------------------------------------------------
    # BLOC 1 : VOLATILITÉ
    # ---------------------------------------------------------
    print("\n--- BLOC 1 : ANALYSE DE LA VOLATILITÉ ---")
    # Volatilité Annualisée
    vol_ann = df_ret.std() * np.sqrt(TRADING_DAYS)
    
    # Semi-Déviation (Risque à la baisse)
    downside_returns = df_ret[df_ret < 0]
    semi_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
    
    # EWMA (Volatilité pondérée, lambda=0.94)
    # alpha = 1 - lambda = 0.06
    ewma_vol = df_ret.ewm(alpha=0.06).std().iloc[-1] * np.sqrt(TRADING_DAYS)

    df_block1 = pd.DataFrame({
        'Volatilité (Ann)': vol_ann,
        'Semi-Déviation': semi_dev,
        'EWMA (Actuelle)': ewma_vol,
        'Ratio Symétrie (Vol/Semi)': vol_ann / semi_dev
    })

    # ---------------------------------------------------------
    # BLOC 2 : TAIL RISK (QUEUES DE DISTRIBUTION)
    # ---------------------------------------------------------
    print("--- BLOC 2 : ANALYSE DES QUEUES (VaR & CVaR) ---")
    
    # A. VaR Historique (95%)
    # On cherche le percentile 5%
    var_hist = df_ret.quantile(1 - CONFIDENCE_LEVEL)
    
    # B. VaR Paramétrique (Méthode Variance-Covariance / Gaussienne)
    # Formule : Mu - z * Sigma
    mu = df_ret.mean()
    sigma = df_ret.std()
    z_score = norm.ppf(1 - CONFIDENCE_LEVEL) # -1.645 pour 95%
    var_param = mu + z_score * sigma
    
    # C. VaR Monte Carlo (Simulation)
    var_mc = []
    for col in df_ret.columns:
        # On simule 10000 rendements possibles basés sur mu et sigma
        simulated_rets = np.random.normal(mu[col], sigma[col], MC_SIMULATIONS)
        var_mc.append(np.percentile(simulated_rets, (1 - CONFIDENCE_LEVEL) * 100))
    
    var_mc_series = pd.Series(var_mc, index=df_ret.columns)

    # D. Expected Shortfall (CVaR) - Historique
    # Moyenne des pertes pires que la VaR Historique
    cvar_list = []
    for col in df_ret.columns:
        cutoff = var_hist[col]
        tail_losses = df_ret[col][df_ret[col] <= cutoff]
        cvar_list.append(tail_losses.mean())
    cvar_series = pd.Series(cvar_list, index=df_ret.columns)

    df_block2 = pd.DataFrame({
        f'VaR Hist {CONFIDENCE_LEVEL:.0%}': var_hist,
        f'VaR Param {CONFIDENCE_LEVEL:.0%}': var_param,
        f'VaR MonteCarlo {CONFIDENCE_LEVEL:.0%}': var_mc_series,
        f'CVaR (Expected Shortfall)': cvar_series
    })

    # ---------------------------------------------------------
    # BLOC 3 : PERFORMANCE & CORRÉLATION
    # ---------------------------------------------------------
    print("--- BLOC 3 : PERFORMANCE & CORRÉLATIONS ---")
    
    # Rendement Annualisé Moyen (Géométrique c'est mieux, mais Arithmétique simple ici)
    mean_ret_ann = df_ret.mean() * TRADING_DAYS
    
    # Ratio de Sharpe
    sharpe = (mean_ret_ann - RISK_FREE_RATE) / vol_ann
    
    # Ratio de Sortino (Rendement / Semi-Déviation)
    sortino = (mean_ret_ann - RISK_FREE_RATE) / semi_dev
    
    # Beta
    # On crée un Benchmark "Synthétique" (Moyenne équipondérée de toutes les devises)
    market_index = df_ret.mean(axis=1)
    market_var = market_index.var()
    
    betas = []
    for col in df_ret.columns:
        cov = df_ret[col].cov(market_index)
        beta = cov / market_var
        betas.append(beta)
    
    beta_series = pd.Series(betas, index=df_ret.columns)

    df_block3 = pd.DataFrame({
        'Rendement Moyen (Ann)': mean_ret_ann,
        'Ratio Sharpe': sharpe,
        'Ratio Sortino': sortino,
        'Beta (vs Market Avg)': beta_series
    })

    # Matrice de Corrélation (sauvegardée à part)
    corr_matrix = df_ret.corr()

    # ---------------------------------------------------------
    # BLOC 4 : DISTRIBUTION & DRAWDOWNS
    # ---------------------------------------------------------
    print("--- BLOC 4 : TRAJECTOIRES & DRAWDOWNS ---")
    
    # Skewness (Asymétrie) et Kurtosis (Aplatissement)
    # Skew < 0 : Queue épaisse à gauche (risque de krach)
    # Kurtosis > 3 : Leptokurtique (événements extrêmes fréquents)
    skew_s = df_ret.apply(skew)
    kurt_s = df_ret.apply(kurtosis) # C'est l'Excess Kurtosis (Normal = 0 ici)
    
    # Calcul des Drawdowns
    # 1. NAV (Base 100)
    nav = (1 + df_ret).cumprod() * 100
    # 2. Pic Courant (Running Max)
    running_max = nav.cummax()
    # 3. Drawdown (%)
    drawdown = (nav - running_max) / running_max
    # 4. Max Drawdown
    max_drawdown = drawdown.min()
    
    # Ratio de Calmar (Rendement Ann / Abs(Max Drawdown))
    # Attention aux divisions par zéro
    calmar = mean_ret_ann / abs(max_drawdown)
    
    df_block4 = pd.DataFrame({
        'Skewness': skew_s,
        'Kurtosis (Excess)': kurt_s,
        'Max Drawdown': max_drawdown,
        'Ratio Calmar': calmar
    })

    # ---------------------------------------------------------
    # CONSOLIDATION & SAUVEGARDE
    # ---------------------------------------------------------
    print("\n>>> SAUVEGARDE DES RÉSULTATS... <<<")
    
    # On fusionne tout dans un énorme fichier Global
    global_report = pd.concat([df_block1, df_block2, df_block3, df_block4], axis=1)
    
    # Tri par Sharpe Ratio (les meilleurs en haut)
    global_report = global_report.sort_values(by='Ratio Sharpe', ascending=False)
    
    # Sauvegarde CSV
    global_report.to_csv("RAPPORT_GLOBAL_RISQUE.csv")
    corr_matrix.to_csv("MATRICE_CORRELATION.csv")
    
    print("\n--- RÉSUMÉ DES MEILLEURS ACTIFS (TOP 3 SHARPE) ---")
    print(global_report[['Ratio Sharpe', 'Volatilité (Ann)', 'Max Drawdown']].head(3))
    
    print("\n--- FICHIERS GÉNÉRÉS ---")
    print("1. RAPPORT_GLOBAL_RISQUE.csv (Toutes les métriques)")
    print("2. MATRICE_CORRELATION.csv (Corrélations croisées)")
    print("\nAnalyse terminée avec succès.")

def generate_visuals():
    if not os.path.exists(FILE_RETURNS):
        print(f"Erreur : {FILE_RETURNS} manquant. Lance l'analyse précédente d'abord.")
        return

    # Chargement
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    print("Données chargées. Génération des graphiques...")

    # ==============================================================================
    # 1. HEATMAP DE CORRÉLATION (La Matrice Colorée)
    # ==============================================================================
    plt.figure(figsize=(16, 14))
    
    # Calcul de la corrélation
    corr = returns.corr()
    
    # Masque pour cacher la moitié supérieure (car c'est symétrique, ça fait plus propre)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Création de la Heatmap
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .5}, annot=True, fmt=".2f", annot_kws={"size": 8})
    
    plt.title('Matrice de Corrélation des Devises', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('CHART_1_CORRELATION.png', dpi=300)
    print("-> CHART_1_CORRELATION.png généré.")

    # ==============================================================================
    # 2. SCATTER PLOT : RISQUE vs RENDEMENT (L'Efficient Frontier)
    # ==============================================================================
    # Calcul des métriques basiques
    summary = pd.DataFrame()
    summary['Returns'] = returns.mean() * 252 # Annualisé
    summary['Volatility'] = returns.std() * np.sqrt(252) # Annualisé
    
    plt.figure(figsize=(12, 8))
    
    # Le nuage de points
    sns.scatterplot(data=summary, x='Volatility', y='Returns', s=100, color='darkblue', alpha=0.7)
    
    # Ajouter les labels (noms des devises) sur les points
    for i in range(summary.shape[0]):
        plt.text(
            summary.Volatility[i]+0.002, 
            summary.Returns[i]+0.002, 
            summary.index[i], 
            fontsize=9
        )
        
    # Lignes médianes pour diviser en 4 quadrants
    plt.axhline(y=summary['Returns'].mean(), color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=summary['Volatility'].mean(), color='r', linestyle='--', alpha=0.3)
    
    plt.title('Profil Risque / Rendement (Risk-Return Profile)', fontsize=16)
    plt.xlabel('Volatilité (Risque)', fontsize=12)
    plt.ylabel('Rendement Annuel Moyen', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('CHART_2_RISK_RETURN.png', dpi=300)
    print("-> CHART_2_RISK_RETURN.png généré.")

    # ==============================================================================
    # 3. BAR CHART : RATIO DE SHARPE (Performance ajustée au risque)
    # ==============================================================================
    plt.figure(figsize=(14, 8))
    
    # Calcul Sharpe (Taux sans risque = 0 pour simplifier)
    sharpe = summary['Returns'] / summary['Volatility']
    sharpe = sharpe.sort_values(ascending=False)
    
    # Code couleur : Vert si positif, Rouge si négatif
    colors = ['green' if x > 0 else 'red' for x in sharpe.values]
    
    sns.barplot(x=sharpe.index, y=sharpe.values, palette=colors, hue=sharpe.index, legend=False)
    
    plt.title('Classement par Ratio de Sharpe (Rentabilité ajustée du Risque)', fontsize=16)
    plt.ylabel('Ratio de Sharpe', fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linewidth=1)
    plt.savefig('CHART_3_SHARPE.png', dpi=300)
    print("-> CHART_3_SHARPE.png généré.")

    # ==============================================================================
    # 4. UNDERWATER PLOT (Les Drawdowns des 5 pires actifs)
    # ==============================================================================
    plt.figure(figsize=(14, 8))
    
    # On prend les 5 actifs les plus volatils pour voir les gros crashs
    top_vol_assets = summary.sort_values(by='Volatility', ascending=False).head(5).index
    
    for asset in top_vol_assets:
        # Calcul du Drawdown
        cum_ret = (1 + returns[asset]).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        
        plt.plot(drawdown.index, drawdown, label=asset, linewidth=1.5)
        
        # Optionnel : Remplir la zone sous la courbe pour le pire actif
        if asset == top_vol_assets[0]:
            plt.fill_between(drawdown.index, drawdown, 0, alpha=0.1)

    plt.title(f'Drawdowns Historiques (Les 5 devises les plus volatiles)', fontsize=16)
    plt.ylabel('Perte depuis le sommet (Drawdown)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('CHART_4_DRAWDOWNS.png', dpi=300)
    print("-> CHART_4_DRAWDOWNS.png généré.")
    
    print("\n--- TERMINE : 4 IMAGES GÉNÉRÉES DANS LE DOSSIER ---")

if __name__ == "__main__":
    run_full_analysis()
    generate_visuals()
