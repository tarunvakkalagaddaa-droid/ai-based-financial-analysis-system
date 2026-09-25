# AI-Based Financial Analysis System

A simple base project that analyzes company financial statements, computes
key financial ratios, uses a machine learning model to classify financial
health, forecasts next year's revenue, and generates a readable report with
trend charts.

## What it does
1. Generates synthetic financial statement data for multiple companies over
   several years (revenue, net income, assets, liabilities, equity, cash
   flow, etc.).
2. Computes key financial ratios:
   - Current Ratio & Quick Ratio (liquidity)
   - Debt-to-Equity (leverage)
   - ROE & ROA (profitability)
   - Net Profit Margin
   - Cash Flow to Debt
3. Labels each company-year as **Healthy** or **At Risk** using a rule-based
   scoring system (used as training data).
4. Trains a **Random Forest** classifier to predict financial health directly
   from the ratios.
5. Forecasts next year's revenue for a company using **Linear Regression**.
6. Generates a full text report + a trend chart (PNG) for a chosen company.

## Setup
```bash
pip install -r requirements_financial.txt
```

## Run
```bash
python financial_analysis.py
```

This will:
- Print model performance metrics (accuracy, precision, recall, F1)
- Save the trained model as `financial_health_model.pkl` and the scaler as
  `financial_scaler.pkl`
- Print a full financial report for a sample company
- Save a trend chart as `company_1_trends.png`

## Using your own data
Replace `generate_synthetic_financial_data()` with:
```python
df = pd.read_csv("your_financials.csv")
```
Your CSV needs these columns (per company, per year):
`company_id, year, revenue, net_income, total_assets, total_liabilities,
equity, current_assets, current_liabilities, inventory, operating_cash_flow`

## Analyzing a different company
```python
generate_report(labeled_df, company_id=42, model=model, scaler=scaler)
```

## Predicting health for a new set of ratios
```python
from financial_analysis import predict_health

ratios = {
    "current_ratio": 1.8, "quick_ratio": 1.2, "debt_to_equity": 0.9,
    "roe": 0.15, "roa": 0.08, "net_profit_margin": 0.10,
    "cash_flow_to_debt": 0.25,
}
label, probability = predict_health(model, scaler, ratios)
print(label, probability)
```

## Next steps to extend this base project
- Connect to a real financial data API (e.g. Alpha Vantage, Yahoo Finance)
  for live stock/company data.
- Add sentiment analysis on financial news headlines as an extra feature.
- Add more advanced forecasting (ARIMA, Prophet, LSTM) instead of linear
  regression.
- Add a dashboard (Streamlit/Flask) so users can pick a company and view
  ratios, predictions, and charts interactively.
- Add anomaly detection to flag unusual financial statement patterns
  (possible fraud indicators).
