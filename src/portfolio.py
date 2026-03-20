import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize


def load_data():
    df = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)
    return df


def compute_features(df):
    """
    Compute latest features for prediction
    """
    returns = df.pct_change()
    momentum = df.pct_change(periods=10)
    volatility = returns.rolling(20).std()

    features = returns.copy()
    features["momentum"] = momentum.mean(axis=1)
    features["volatility"] = volatility.mean(axis=1)

    features = features.dropna()

    return features


def get_predicted_returns(model, features):
    """
    Predict future returns using ML model
    """
    latest_features = features.iloc[-1].values.reshape(1, -1)

    prediction = model.predict(latest_features)

    # distribute prediction across assets
    num_assets = len(features.columns) - 2  # excluding added features
    predicted_returns = np.repeat(prediction[0], num_assets)

    return predicted_returns


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


if __name__ == "__main__":

    print("Loading data...\n")

    prices = load_data()

    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov()

    print("Loading ML model...\n")

    model = joblib.load("models/model.pkl")

    print("Generating features...\n")

    features = compute_features(prices)

    print("Predicting future returns...\n")

    predicted_returns = get_predicted_returns(model, features)

    print("Optimizing portfolio using ML predictions...\n")

    optimal_weights = optimize_portfolio(predicted_returns, cov_matrix)

    portfolio_return, portfolio_risk = portfolio_performance(
        optimal_weights,
        predicted_returns,
        cov_matrix
    )

    print("Optimal Portfolio Allocation (ML-based):\n")

    for stock, weight in zip(prices.columns, optimal_weights):
        print(f"{stock}: {weight:.2%}")

    print("\nExpected Return:", portfolio_return)
    print("Portfolio Risk:", portfolio_risk)