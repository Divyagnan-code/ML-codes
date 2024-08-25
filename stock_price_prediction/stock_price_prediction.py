import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import math
import numpy as np

data = pd.read_csv("GOOGLWIKI.csv")  # enter the csv file of stock prices
df = pd.DataFrame(data)
df.fillna(-99999, inplace=True)
# engineer some relevant features for better predictions
df["HL_PCT"] = ((df['Adj. High'] - df['Adj. Close']) / df["Adj. Close"]) * 100
df["PCT_Change"] = ((df['Adj. Close'] - df['Adj. Open']) / df["Adj. Open"]) * 100
df = df[["HL_PCT", "PCT_Change", "Adj. Close", "Adj. Volume"]]  # choose the most valid columns
close_col = "Adj. Close"
forecast_shift = int(math.ceil(0.01 * len(df[close_col])))  # After how many days you wanna predict the price
df["label"] = df[close_col].shift(-forecast_shift)
df.dropna(inplace=True)
print(df.head())
X = np.array(df.drop(["label"], axis=1))
y = np.array(df["label"])
preprocessing.scale(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
linear_model = LinearRegression()  # if your data set is so large change the n_jobs argument to adjust threading to improve processing time
before = time.time_ns()
linear_model.fit(train_X, train_y)
after = time.time_ns()
print(f"time to train: {after - before}")
acc = linear_model.score(test_X, test_y)
print(acc)
