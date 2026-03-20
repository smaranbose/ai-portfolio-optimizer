import pandas as pd


def load_price_data(filepath="data/stock_prices.csv"):
    """
    Load stock price data from CSV
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def compute_daily_returns(df):
    """
    Compute daily percentage returns
    """
    returns = df.pct_change()
    returns = returns.dropna()
    return returns


def compute_moving_averages(df, windows=[20, 50]):
    """
    Compute moving averages for given windows
    """
    ma_data = {}

    for window in windows:
        ma = df.rolling(window).mean()
        ma_data[f"MA_{window}"] = ma

    return ma_data


def compute_volatility(returns, window=20):
    """
    Compute rolling volatility (risk)
    """
    volatility = returns.rolling(window).std()
    return volatility


def compute_momentum(df, window=10):
    """
    Compute momentum (price change over window)
    """
    momentum = df.pct_change(periods=window)
    return momentum


if __name__ == "__main__":

    print("Loading data...\n")

    prices = load_price_data()

    print("Computing features...\n")

    # returns = compute_daily_returns(prices)
    # ma = compute_moving_averages(prices)
    # volatility = compute_volatility(returns)
    # momentum = compute_momentum(prices)

    # print("Daily Returns:\n", returns.head())
    # print("\nMoving Average (20 days):\n", ma["MA_20"].head())
    # print("\nVolatility:\n", volatility.head())
    # print("\nMomentum:\n", momentum.head())

    returns = compute_daily_returns(prices)
ma = compute_moving_averages(prices)
volatility = compute_volatility(returns)
momentum = compute_momentum(prices)

# drop NaNs for cleaner output
ma_clean = ma["MA_20"].dropna()
vol_clean = volatility.dropna()
momentum_clean = momentum.dropna()

print("Moving Average (20 days):\n", ma_clean.head())
print("\nVolatility:\n", vol_clean.head())
print("\nMomentum:\n", momentum_clean.head())