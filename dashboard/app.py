import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize


# ---------- UI ----------
st.title(" AI Portfolio Optimizer")
st.write("Optimize your investment using Machine Learning")
st.write(" Welcome! Click the button to generate portfolio.")

st.markdown("###  Enter details below")


# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)
    return df


# ---------- Feature Engineering ----------
def compute_features(df):
    returns = df.pct_change()
    momentum = df.pct_change(periods=10)
    volatility = returns.rolling(20).std()

    features = returns.copy()
    features["momentum"] = momentum.mean(axis=1)
    features["volatility"] = volatility.mean(axis=1)

    return features.dropna()


# ---------- Portfolio Functions ----------
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk


def negative_sharpe(weights, mean_returns, cov_matrix):
    returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -returns / risk


def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    init_guess = num_assets * [1.0 / num_assets]

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    })

    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x


# ---------- User Input ----------
investment = st.number_input(
    "Enter Investment Amount ($)",
    min_value=1000,
    value=10000
)

run = st.button(" Optimize Portfolio")

# Debug
st.write("Button clicked:", run)


# ---------- Main Logic ----------
if run:
    try:
        st.write(" Processing...")

        # Load data
        prices = load_data()

        # Returns + covariance
        returns = prices.pct_change().dropna()
        cov_matrix = returns.cov()

        # Load ML model
        model = joblib.load("models/model.pkl")

        # Generate features
        features = compute_features(prices)

        # Use latest row (keeps feature names → no warning)
        latest_features = features.iloc[[-1]]

        # Predict return
        prediction = model.predict(latest_features)[0]

        # Apply to all stocks
        predicted_returns = np.repeat(prediction, len(prices.columns))

        # Optimize
        weights = optimize_portfolio(predicted_returns, cov_matrix)

        portfolio_return, portfolio_risk = portfolio_performance(
            weights,
            predicted_returns,
            cov_matrix
        )

        # ---------- Output ----------
        st.write("## Optimal Allocation")

        allocation = dict(zip(prices.columns, weights))

        alloc_df = pd.DataFrame.from_dict(
            allocation,
            orient="index",
            columns=["Weight"]
        )

        # Chart
        st.bar_chart(alloc_df)

        # Breakdown
        st.write("### Investment Breakdown")
        for stock, weight in allocation.items():
            amount = weight * investment
            st.write(f"{stock}: {weight:.2%} → ${amount:,.2f}")

        # Metrics
        st.write("### Portfolio Metrics")
        st.write(f"Expected Return: {portfolio_return:.4f}")
        st.write(f"Risk (Volatility): {portfolio_risk:.4f}")

        st.success(" Optimization Complete!")

    except Exception as e:
        st.error(f" Error: {e}")

else:
    st.info(" Click the button to generate portfolio allocation")