Projet forex trading where I try to implement and compare different methods to create an algorithm efficient specifically for trading, here are the 3 main methods tested for now, with results available in their folder :
1. The "Naive" Approach (Baseline)
Strategy: Risk Parity / Equal Weight.
Result: Loss (-4%) or very low return.
Conclusion: "Being passive doesn't work in FX."

2. The "Predictive" Approach (Time Series / AI)
Strategy: ARIMA + GARCH or ARIMA or GARCH --------------->>>> what I'll be working on next
Result: High Return (+28%) but massive crash (-44% Drawdown).
Analysis: "Predicting the next day is possible, but the 'Signal-to-Noise' ratio is so low that we need 5x leverage to make money. This creates unacceptable tail risk."

3. The "Institutional" Approach (Winner)
Strategy: Global Macro Momentum (Black-Litterman) (The code before ARIMA).
Result: High Return (+30.19%) with Safe Risk (-12.61% Drawdown).
Analysis: "Instead of predicting tomorrow's noise, we capture long-term divergence in monetary policy (Interest Rates). This is robust, requires less leverage (1.2x), and protects capital."
