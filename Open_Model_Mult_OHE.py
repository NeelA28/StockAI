import yfinance as yf
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Parameters

interval = "1h"
days_back = 729
X = []
y = []

end_date = datetime.now() - timedelta(days=1)
start_date = datetime.now() - timedelta(days=days_back)

def get_data(tickers):

    for ticker in tickers:
        full_data = yf.download(
            ticker,
            interval=interval,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )

        apple = full_data[~full_data.index.duplicated()]  # remove duplicates
        tickers = apple.columns.get_level_values(1)
        apple["ticker"] = tickers[0]
        apple.columns = apple.columns.get_level_values(0)

        for i in range(len(apple)):
            if not apple.iloc[i].isna().any():
                if (i + 1) % 7 != 0:
                    X.append(apple.iloc[i, :])
                if i % 7 != 0:
                    y.append(apple.iloc[i, 1])
        X.pop()

get_data(["AAPL", "MSFT", "AMZN"])

X = pd.DataFrame(X)
X = pd.get_dummies(X)

print(X.head(-5))

y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = XGBRegressor(
    n_estimators=1000,  # small number of trees
    learning_rate=0.01,  # faster learning since few trees
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
