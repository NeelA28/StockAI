import yfinance as yf
import pandas as pd
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

    #print(f"Fetching {chunk_start} → {chunk_end}")

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

final = []
for i in range(len(result)):
    for j in range(5):
        row = result.iloc[i,29+60*j:89+60*j]
        final.append(list(row))

X = pd.DataFrame(final, columns = range(60))

hourly_highs = apple['High'].resample('1h').max()

y = hourly_highs.between_time("15:00", "19:00")

y = y.dropna().reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#print(X.head(15))
#print(y.head(15))
