# Stock Return Predictor

A machine learning project that predicts whether a stock will outperform the S&P 500 (SPY) over the following month, and prices call options on the top picks using Black-Scholes and a Cox-Ross-Rubinstein binomial tree.

---

## Project Overview

This project combines quantitative finance and machine learning to build an end-to-end stock selection and options pricing pipeline. It was built as a personal portfolio project to explore momentum-based feature engineering, classification modeling, backtesting methodology, and derivatives pricing.

**Universe:** 20 stocks across 4 sectors (Technology, Finance, Energy, Healthcare)  
**Period:** January 2018 – December 2024  
**Prediction target:** Will this stock outperform SPY next month? (Binary: 1 = yes, 0 = no)

---

## Project Structure

```
stock-predictor/
├── data/
│   ├── prices.csv          # Daily adjusted closing prices
│   └── features.csv        # Engineered features and target variable
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_options_pricing.ipynb
├── src/
│   └── utils.py
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.13+, Homebrew (macOS)

```bash
# Install libomp (required for XGBoost on macOS)
brew install libomp

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install yfinance pandas numpy scikit-learn xgboost matplotlib seaborn jupyter ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name=stock-env --display-name "stock-env"
```

Open any notebook in VS Code and select **stock-env** as the kernel.

---

## Notebooks

### 01 — Data Collection
Pulls 7 years of daily adjusted closing prices for 20 stocks using `yfinance` and saves them to `data/prices.csv`.

- **Output:** `prices.csv` — shape (1760, 20)
- **Library:** `yfinance`

### 02 — Feature Engineering
Resamples daily prices to monthly frequency and engineers 6 features per stock per month. Constructs the binary target variable by comparing each stock's next-month return to SPY's.

**Features engineered:**
| Feature | Description |
|---|---|
| `ret_1m` | 1-month momentum |
| `ret_3m` | 3-month momentum |
| `ret_6m` | 6-month momentum |
| `ma_ratio_20` | Price relative to 20-day moving average |
| `ma_ratio_50` | Price relative to 50-day moving average |
| `volatility` | 20-day annualized historical volatility |

- **Output:** `features.csv` — shape (77, 140)
- **Target distribution:** ~47% of stock-months beat SPY (close to 50/50)

### 03 — Modeling
Stacks features into long format (one row per stock per month), splits by time (80/20), and trains three classifiers.

**Results:**
| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 51.9% | 0.470 |
| Ridge Classifier | 52.5% | — |
| XGBoost | 50.0% | 0.493 |

All three models perform near random, which is expected given the small sample size (77 months) and the difficulty of beating efficient markets with simple features. Ridge Classifier performed best and was carried forward.

### 04 — Backtesting
Simulates a rolling window trading strategy using the Ridge Classifier. Each month the model trains on all historical data, scores the current month, and selects the top 5 stocks to hold equally weighted for the next month.

**Results vs SPY (2020–2024):**
| Metric | Strategy | SPY |
|---|---|---|
| Total Return | 238.45% | 96.35% |
| Annualized Return | 32.49% | 16.85% |
| Sharpe Ratio | 1.275 | 1.035 |
| Max Drawdown | -17.73% | -23.93% |

> **Important caveat:** These results are likely overstated due to small sample size, survivorship bias in the stock universe, and the absence of transaction costs. They should be interpreted as a proof of concept, not a live trading signal.

### 05 — Options Pricing
Takes the Ridge model's top 5 picks and prices 30-day at-the-money European call options using two methods, then compares the results.

**Pricing methods:**
- **Black-Scholes** — closed-form solution for European calls
- **Cox-Ross-Rubinstein Binomial Tree** — discrete-time lattice model with 100 steps

**Results (latest signal date):**
| Ticker | Volatility | Black-Scholes | Binomial Tree | Difference |
|---|---|---|---|---|
| GS | 50.76% | $33.42 | $33.33 | $0.08 |
| META | 23.93% | $17.19 | $17.15 | $0.04 |
| WFC | 51.60% | $4.16 | $4.15 | $0.01 |
| AMZN | 37.47% | $9.88 | $9.85 | $0.02 |
| AAPL | 15.08% | $4.80 | $4.79 | $0.01 |

With 100 binomial steps both methods converge to within a few cents, confirming the theoretical result that the binomial tree approaches Black-Scholes as the number of steps increases. The remaining gap is due to the discretization of the tree.

> **Note:** These are theoretical premiums using historical realized volatility. Real options markets price using implied volatility, which reflects forward-looking market expectations and will differ from historical vol.

---

## Theory

### Why momentum features?
The momentum anomaly — documented by Jegadeesh and Titman (1993) — shows that past winners tend to continue outperforming short term. The intuition is that good news about a company takes time to fully get priced in as investors update their views slowly.

### Why Ridge Classifier?
Momentum features are correlated with each other (1m, 3m, and 6m returns all measure similar things). Ridge regression adds L2 regularization — a penalty for large coefficients — which prevents any single correlated feature from dominating and improves generalization on small datasets.

### Why does the binomial tree converge to Black-Scholes?
Both methods price the same underlying contract — a European call on a non-dividend-paying stock — under the same assumptions (log-normal price process, constant volatility, constant risk-free rate). As the number of binomial steps increases, the discrete lattice approximates the continuous log-normal distribution that Black-Scholes assumes, so the prices converge mathematically.

### Why is the model accuracy near random?
The Efficient Market Hypothesis (EMH) argues that all publicly available information is already reflected in stock prices. Simple price-based features like momentum and moving averages are widely known, so any edge they once had tends to get arbitraged away over time. The small sample size (77 months) also limits the model's ability to find reliable patterns.

---

## Limitations

- **Small sample size** — 77 monthly observations limits model reliability
- **Survivorship bias** — only stocks with clean data for the full period are included; bankrupt or delisted companies are excluded
- **No transaction costs** — real portfolios incur fees, slippage, and taxes
- **Historical volatility** — options are priced with realized vol, not implied vol
- **European options only** — binomial tree is configured for European exercise; American options would require early exercise checks at each node

---

## Planned Improvements

- Expand universe to full S&P 500 for more training data
- Add fundamental features: P/E ratio, earnings surprise, debt/equity
- Add macro features: VIX, interest rate regime, inflation
- Add 12-month momentum (strongest known momentum signal in literature)
- Include transaction costs in backtesting
- Use implied volatility from options chain instead of historical vol
- Test American option pricing with early exercise in the binomial tree

---

## Libraries

| Library | Purpose |
|---|---|
| `yfinance` | Market data retrieval |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | Machine learning models |
| `xgboost` | Gradient boosted trees |
| `matplotlib` | Visualization |
| `jupyter` | Notebook environment |
