import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("iris.data")
df = pd.DataFrame(data)
print(df.head())
X = np.array(df.drop(["class"], axis=1))
Y = np.array(df['class'].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 3}))
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(train_x, train_y)
acc = clf.score(test_x, test_y)
print(acc)
