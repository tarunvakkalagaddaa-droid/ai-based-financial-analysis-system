"""
AI-Based Financial Analysis System
====================================
A simple, self-contained base project that analyzes a company's financial
statements, computes key financial ratios, classifies financial health using
a machine learning model, forecasts next year's revenue, and generates a
readable financial report with charts.

Pipeline:
    1. Generate / load financial statement data (multiple companies, years)
    2. Compute financial ratios (liquidity, profitability, leverage)
    3. Label financial health (Healthy / At Risk) from the ratios
    4. Train an ML classifier to predict financial health from ratios
    5. Forecast next year's revenue with a simple regression model
    6. Generate a text report + trend chart for a chosen company

Run:
    python financial_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNG files
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. DATA GENERATION
# ---------------------------------------------------------------------
def generate_synthetic_financial_data(n_companies: int = 300, n_years: int = 5) -> pd.DataFrame:
    """
    Creates synthetic yearly financial statement data for multiple companies.
    Replace this with pd.read_csv("your_financials.csv") to use real data.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for company_id in range(1, n_companies + 1):
        base_revenue = rng.uniform(1_000_000, 50_000_000)
        growth_rate = rng.normal(0.06, 0.08)  # can be negative (declining company)

        revenue = base_revenue
        for year in range(1, n_years + 1):
            revenue *= (1 + growth_rate + rng.normal(0, 0.03))
            revenue = max(revenue, 50_000)

            net_margin = rng.normal(0.08, 0.06)
            net_income = revenue * net_margin

            total_assets = revenue * rng.uniform(0.8, 2.0)
            total_liabilities = total_assets * rng.uniform(0.2, 0.9)
            equity = total_assets - total_liabilities

            current_assets = total_assets * rng.uniform(0.3, 0.6)
            current_liabilities = total_liabilities * rng.uniform(0.3, 0.7)
            inventory = current_assets * rng.uniform(0.1, 0.4)

            operating_cash_flow = net_income * rng.uniform(0.7, 1.4)

            rows.append(
                {
                    "company_id": company_id,
                    "year": 2019 + year,
                    "revenue": round(revenue, 2),
                    "net_income": round(net_income, 2),
                    "total_assets": round(total_assets, 2),
                    "total_liabilities": round(total_liabilities, 2),
                    "equity": round(equity, 2),
                    "current_assets": round(current_assets, 2),
                    "current_liabilities": round(current_liabilities, 2),
                    "inventory": round(inventory, 2),
                    "operating_cash_flow": round(operating_cash_flow, 2),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. FINANCIAL RATIOS
# ---------------------------------------------------------------------
def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["current_ratio"] = df["current_assets"] / df["current_liabilities"]
    df["quick_ratio"] = (df["current_assets"] - df["inventory"]) / df["current_liabilities"]
    df["debt_to_equity"] = df["total_liabilities"] / df["equity"].replace(0, np.nan)
    df["roe"] = df["net_income"] / df["equity"].replace(0, np.nan)          # Return on Equity
    df["roa"] = df["net_income"] / df["total_assets"]                      # Return on Assets
    df["net_profit_margin"] = df["net_income"] / df["revenue"]
    df["cash_flow_to_debt"] = df["operating_cash_flow"] / df["total_liabilities"].replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ---------------------------------------------------------------------
# 3. FINANCIAL HEALTH LABEL (for training the classifier)
# ---------------------------------------------------------------------
def label_financial_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple rule-based health score used to create training labels.
    1 = At Risk, 0 = Healthy
    """
    df = df.copy()
    risk_points = (
        (df["current_ratio"] < 1.0).astype(int)
        + (df["debt_to_equity"] > 2.0).astype(int)
        + (df["net_profit_margin"] < 0).astype(int)
        + (df["roa"] < 0.02).astype(int)
        + (df["cash_flow_to_debt"] < 0.1).astype(int)
    )
    df["at_risk"] = (risk_points >= 2).astype(int)
    return df


# ---------------------------------------------------------------------
# 4. ML CLASSIFIER: PREDICT FINANCIAL HEALTH
# ---------------------------------------------------------------------
RATIO_FEATURES = [
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "roe",
    "roa",
    "net_profit_margin",
    "cash_flow_to_debt",
]


def train_health_classifier(df: pd.DataFrame):
    X = df[RATIO_FEATURES]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("\n" + "=" * 50)
    print("Financial Health Classifier Performance")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "At Risk"]))

    return model, scaler, metrics


def predict_health(model, scaler, ratios: dict):
    """ratios: dict with keys matching RATIO_FEATURES"""
    X_new = pd.DataFrame([ratios])[RATIO_FEATURES]
    X_scaled = scaler.transform(X_new)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    label = "AT RISK" if prediction == 1 else "HEALTHY"
    return label, round(probability, 4)


# ---------------------------------------------------------------------
# 5. REVENUE FORECASTING (simple regression per company)
# ---------------------------------------------------------------------
def forecast_next_year_revenue(company_df: pd.DataFrame):
    """
    company_df: rows of a single company sorted by year, with 'year' and 'revenue'.
    Fits a simple linear trend and forecasts the following year.
    """
    company_df = company_df.sort_values("year")
    X = company_df[["year"]].values
    y = company_df["revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    next_year = company_df["year"].max() + 1
    forecast = model.predict([[next_year]])[0]
    return next_year, round(float(forecast), 2)


# ---------------------------------------------------------------------
# 6. REPORTING
# ---------------------------------------------------------------------
def plot_company_trends(company_df: pd.DataFrame, company_id: int, out_path: str):
    company_df = company_df.sort_values("year")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(company_df["year"], company_df["revenue"], marker="o", label="Revenue")
    axes[0].plot(company_df["year"], company_df["net_income"], marker="o", label="Net Income")
    axes[0].set_title(f"Company {company_id}: Revenue & Net Income")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Amount")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(company_df["year"], company_df["current_ratio"], marker="o", label="Current Ratio")
    axes[1].plot(company_df["year"], company_df["debt_to_equity"], marker="o", label="Debt/Equity")
    axes[1].set_title(f"Company {company_id}: Key Ratios")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Ratio")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_report(df: pd.DataFrame, company_id: int, model, scaler):
    company_df = df[df["company_id"] == company_id]
    latest = company_df.sort_values("year").iloc[-1]

    print("\n" + "#" * 60)
    print(f"FINANCIAL ANALYSIS REPORT - Company {company_id}")
    print("#" * 60)
    print(f"Latest Year: {int(latest['year'])}")
    print(f"Revenue:            {latest['revenue']:,.2f}")
    print(f"Net Income:         {latest['net_income']:,.2f}")
    print(f"Total Assets:       {latest['total_assets']:,.2f}")
    print(f"Total Liabilities:  {latest['total_liabilities']:,.2f}")
    print(f"Equity:             {latest['equity']:,.2f}")

    print("\nKey Ratios:")
    for r in RATIO_FEATURES:
        print(f"  {r:<20}: {latest[r]:.3f}")

    label, probability = predict_health(model, scaler, latest[RATIO_FEATURES].to_dict())
    print(f"\nAI Health Prediction: {label} (probability at risk: {probability})")

    next_year, forecast = forecast_next_year_revenue(company_df)
    print(f"Revenue Forecast for {next_year}: {forecast:,.2f}")

    chart_path = f"company_{company_id}_trends.png"
    plot_company_trends(company_df, company_id, chart_path)
    print(f"Trend chart saved to: {chart_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Generating synthetic financial data...")
    raw_df = generate_synthetic_financial_data()
    print(f"Raw data shape: {raw_df.shape}")

    print("\nComputing financial ratios...")
    ratio_df = compute_ratios(raw_df)

    print("Labeling financial health (rule-based, used as training target)...")
    labeled_df = label_financial_health(ratio_df)
    print(f"At-risk rate in dataset: {labeled_df['at_risk'].mean():.2%}")

    print("\nTraining AI health classifier...")
    model, scaler, metrics = train_health_classifier(labeled_df)

    joblib.dump(model, "financial_health_model.pkl")
    joblib.dump(scaler, "financial_scaler.pkl")
    print("\nSaved model to 'financial_health_model.pkl' and scaler to 'financial_scaler.pkl'.")

    # Generate a report for a sample company
    sample_company_id = 1
    generate_report(labeled_df, sample_company_id, model, scaler)


if __name__ == "__main__":
    main()
