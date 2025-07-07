# Portfolio Optimization for Sharpe Ratios

This project implements a portfolio optimization model for a set of stocks, aiming to **maximize the Sharpe ratio**. The Sharpe ratio is a risk-adjusted performance metric that evaluates how much excess return an investment portfolio generates for each unit of risk.

## 🔍 What the Model Does

The Jupyter notebook `Portfolio_Optimization.ipynb` performs the following steps:

1. **Loads historical stock price data** for a set of tickers (e.g., AAPL, MSFT, GOOG).
2. **Computes daily returns** from adjusted closing prices.
3. **Calculates the expected return, volatility, and Sharpe ratio** of various portfolio allocations.
4. **Uses optimization techniques** (typically from `scipy.optimize`) to identify the asset weightings that **maximize the Sharpe ratio**.
5. **Visualizes the efficient frontier**, optimal portfolio, and associated risk/return metrics.

## 🧰 Requirements

Dependencies are listed in `requirements.txt`. These include:

- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`
- `scipy`
- `seaborn`
- `jupyter`

## 🚀 Getting Started with Docker

### 1. Build the Docker Image

```bash
docker build -t portfolio-optimization .
