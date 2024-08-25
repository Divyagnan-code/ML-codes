import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use("ggplot")

data = pd.read_csv("GOOGLWIKI.csv")  # enter the csv file of stock prices
df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.fillna(-99999, inplace=True)
# engineer some relevant features for better predictions
df["HL_PCT"] = ((df['Adj. High'] - df['Adj. Close']) / df["Adj. Close"]) * 100
df["PCT_Change"] = ((df['Adj. Close'] - df['Adj. Open']) / df["Adj. Open"]) * 100
df = df[["HL_PCT", "PCT_Change", "Adj. Close", "Adj. Volume"]]  # choose the most valid columns
close_col = "Adj. Close"
forecast_shift = int(math.ceil(0.01 * len(df[close_col])))  # After how many days you wanna predict the price
df["label"] = df[close_col].shift(-forecast_shift)
print(df.head())
X = np.array(df.drop(["label"], axis=1))
preprocessing.scale(X)
X_after = X[-forecast_shift:]
X = X[:-forecast_shift]
df.dropna(inplace=True)
y = np.array(df["label"])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
linear_model = LinearRegression()  # if your data set is so large change the n_jobs argument to adjust threading to improve processing time
linear_model.fit(train_X, train_y)
with open("stock_price_prediction.pickle", "wb") as f:
    pickle.dump(linear_model, f)

pickle_in = open("stock_price_prediction.pickle", "rb")
linear_model = pickle.load(pickle_in)
acc = linear_model.score(test_X, test_y)
forecast_set = linear_model.predict(X_after)
print(forecast_set, acc, forecast_shift)

df["forecast"] = np.nan
last_date = df.index[-1]
next_unix = last_date.timestamp()
one_day = 86400
next_day = next_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]  # allocate the dates as indices and allocates nan to all columns except for the last (forecast).

df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
