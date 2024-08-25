import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
import sklearn.neighbors
df = pd.DataFrame(pd.read_csv("iris.data"))
df.replace({"Iris-virginica": 3, "Iris-versicolor": 2, "Iris-setosa": 1}, inplace=True)
X = df.drop(["class"], axis=1)
Y = df["class"]
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)


def euclidian_distance(i1, i2):
    return math.sqrt(sum((i1 - i2) ** 2))


def KNN(train_x, test_x, train_y, test_y, k=4):
    predictions = []
    x = np.array(train_x)
    y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    for instance in test_x:
        votes = {}
        for i in range(len(train_x)):
            distance = euclidian_distance(instance, train_x[i])
            votes[distance] = train_y[i]
        ascending_votes = sorted(votes.keys())
        classes = {1: 0, 2: 0, 3: 0}
        for i in range(k):
            class_label = votes[ascending_votes[i]]
            classes[class_label] += 1
        prediction = max(classes, key=lambda k: classes[k])
        predictions.append(prediction)

    def calculate_accuracy(pred, y):
        count = 0
        for i in range(len(y)):
            if pred[i] == y[i]:
                count += 1
        return (count/len(y))*100
    acc = calculate_accuracy(predictions, test_y)
    return acc, predictions


train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
acc, predictions = KNN(train_x, test_x, train_y, test_y, k=4)
print(acc)
for i in range(len(test_y)):
    print(f"predicted:{predictions[i]}, original: {test_y[i]}")

plt.scatter(predictions, test_y)
plt.show()
