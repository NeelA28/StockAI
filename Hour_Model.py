import yfinance as yf
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

def get_data(tickers, interval, days_back, chunk_size):
    final_data = []
    for ticker in tickers:

        all_data = []
        end_date = datetime.now() - timedelta(days=1)
        start_date = datetime.now() - timedelta(days=days_back)

        # Generate chunks of ~7 days
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_size), end_date)

            # print(f"Fetching {chunk_start} → {chunk_end}")

            chunk = yf.download(
                ticker,
                interval=interval,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                progress=False
            )

            if not chunk.empty:
                all_data.append(chunk)

            chunk_start = chunk_end  # move to next chunk

        # Combine all chunks into a single DataFrame
        full_data = pd.concat(all_data)
        full_data = full_data[~full_data.index.duplicated()]  # remove duplicates
        full_data = full_data.stack(level = 1)

        final_data.append(full_data)

    return pd.concat(final_data)

tickers = ['MSFT']
data = get_data(tickers, "1h", 729, 730)

X = data.drop(data.index[6::7])

y = data.drop(data.index[::7]).High

print(X.head(-10))
print(y.head(-10))

print(X.describe())
print(y.describe())
