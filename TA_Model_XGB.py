import yfinance as yf
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Parameters
ticker = "AAPL"
interval = "1m"
days_back = 29
chunk_size = 7  # Yahoo allows ~7 days for 1m data

end_date = datetime.now() - timedelta(days=1)
start_date = datetime.now() - timedelta(days=days_back)

all_data = []

# Generate chunks of ~7 days
chunk_start = start_date
while chunk_start < end_date:
    chunk_end = min(chunk_start + timedelta(days=chunk_size), end_date)

    print(f"Fetching {chunk_start} → {chunk_end}")

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
apple = full_data[~full_data.index.duplicated()]  # remove duplicates

apple_open = apple['Open']

df = apple_open.reset_index()
df['Date'] = df['Datetime'].dt.date
df['Hour'] = df['Datetime'].dt.strftime('%H:%M')

result = df.pivot(index='Date', columns='Hour', values=ticker)
result = result.iloc[:,1:]

result = result.dropna(how="all")
result = result.ffill(axis = 1)
result = result.bfill(axis = 1)

print(result.describe())
print(result.head(5))

X = []
y = []

for date in result.index:
    day_data = result.loc[date]

    for j in range(5):
        features = day_data[29+60*j:89+60*j].values
        if len(features) != 60:
            continue

        X.append(features)

        day_high = apple.loc[str(date)].between_time("15:00", "19:00")['High'].max()
        y.append(day_high)

X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = XGBRegressor(
    n_estimators=100,  # small number of trees
    learning_rate=0.2,  # faster learning since few trees
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

#print(X.head(15))
#print(y.head(15))
