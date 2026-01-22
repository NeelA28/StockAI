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

tickers = ['MSFT','AAPL']
data = get_data(tickers, "1m", 29, 7)

data_open = data['Open']

df = data_open.reset_index()
df['Date'] = df['Datetime'].dt.date
df['Hour'] = df['Datetime'].dt.strftime('%H:%M')

result = df.pivot(index=['Ticker','Date'], columns='Hour', values='Open')

result = result.dropna(how="all")
result = result.ffill(axis = 1)
result = result.bfill(axis = 1)

a = result.reset_index()   # keeps both Ticker and Date as columns

ticker_dummies = pd.get_dummies(a['Ticker'], prefix='Ticker')

df_encoded = pd.concat([a, ticker_dummies], axis=1)



high = data['High']

# Filter by time using the Datetime level
high = high[
    high.index.get_level_values('Datetime').time >= pd.to_datetime("15:00").time()
]
high = high[
    high.index.get_level_values('Datetime').time <= pd.to_datetime("19:00").time()
]

# Daily max per ticker
daily_targets = (
    high
    .groupby([
        high.index.get_level_values('Ticker'),
        high.index.get_level_values('Datetime').date
    ])
    .max()
)

daily_targets.index = daily_targets.index.set_names(['Ticker', 'Date'])

# --- Build X and y together ---
X_rows = []
y_vals = []

ticker_cols = df_encoded.columns[-len(tickers):].tolist()

for (ticker, date), row in result.iterrows():

    # Get target for this ticker/date
    if (ticker, date) not in daily_targets:
        continue

    target = daily_targets.loc[(ticker, date)]

    # Get one-hot encoding for this row
    encoded_row = df_encoded.loc[
        (df_encoded['Ticker'] == ticker) & (df_encoded['Date'] == date)
    ]

    if encoded_row.empty:
        continue

    ticker_one_hot = encoded_row[ticker_cols].iloc[0]

    # Generate 5 samples per day
    for j in range(5):
        start = 29 + 60 * j
        end = start + 60

        features = row.iloc[start:end]

        if len(features) != 60 or features.isna().any():
            continue

        X_rows.append(
            pd.concat([features, ticker_one_hot]).values
        )
        y_vals.append(target)

# --- Final datasets ---
X = pd.DataFrame(X_rows, columns=list(range(60)) + ticker_cols)
y = pd.Series(y_vals, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = XGBRegressor(
    n_estimators=500,  # small number of trees
    learning_rate=0.1,  # faster learning since few trees
    max_depth=3,  # shallow trees to avoid overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,  # L1 regularization
    reg_lambda=1,  # L2 regularization
    min_child_weight=3,
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()