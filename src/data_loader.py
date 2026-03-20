import yfinance as yf
import pandas as pd
import os


def download_stock_data(
    tickers,
    start_date="2015-01-01",
    end_date="2024-01-01"
):
    """
    Download historical stock price data from Yahoo Finance
    and return closing prices for all tickers.
    """

    all_data = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        # safely extract close column as dataframe
        close_df = df[["Close"]].copy()

        # rename column to ticker
        close_df.columns = [ticker]

        all_data.append(close_df)

    # merge all stocks together
    close_prices = pd.concat(all_data, axis=1)

    # remove rows with missing values
    close_prices = close_prices.dropna()

    return close_prices


def save_data(df, filepath="data/stock_prices.csv"):
    """
    Save dataframe to CSV
    """

    os.makedirs("data", exist_ok=True)

    df.to_csv(filepath)

    print(f"\nData saved to {filepath}")


def load_data(filepath="data/stock_prices.csv"):
    """
    Load stock data from CSV
    """

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    return df


if __name__ == "__main__":

    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA"
    ]

    print("Downloading stock data...\n")

    df = download_stock_data(tickers)

    print("\nPreview of downloaded data:\n")
    print(df.head())

    save_data(df)