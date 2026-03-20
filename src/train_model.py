import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


def load_features():
    """
    Load and prepare features for ML model
    """
    prices = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)

    returns = prices.pct_change().dropna()
    momentum = prices.pct_change(periods=10)
    volatility = returns.rolling(20).std()

    # align datasets
    df = returns.copy()
    df["momentum"] = momentum.mean(axis=1)
    df["volatility"] = volatility.mean(axis=1)

    df = df.dropna()

    return df


def prepare_data(df):
    """
    Create features (X) and target (y)
    """
    # target = next day return
    df["target"] = df.mean(axis=1).shift(-1)

    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def train_model(X, y):
    """
    Train Random Forest model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    print(f"Model MSE: {mse}")

    return model


def save_model(model, path="models/model.pkl"):
    """
    Save trained model
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":

    print("Loading features...\n")

    df = load_features()

    print("Preparing data...\n")

    X, y = prepare_data(df)

    print("Training model...\n")

    model = train_model(X, y)

    save_model(model)