import pandas
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

apple = yf.download("AAPL",
    start="2024-01-1",
    end="2025-01-1",
    interval="1h")

apple_open = apple['Open']

df = apple_open.reset_index()
df['Date'] = df['Datetime'].dt.date
df['Hour'] = df['Datetime'].dt.strftime('%H:%M')

result = df.pivot(index='Date', columns='Hour', values='AAPL')
result = result.iloc[:,1:]
result = result.dropna(how="all")
result = result.ffill(axis = 1)

y = result.iloc[:,-1]
X = result.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(X_train.count())
print(X_test.count())

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)








