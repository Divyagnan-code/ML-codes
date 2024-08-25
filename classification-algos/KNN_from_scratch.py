import numpy as np
import math
import pandas as pd
from matplotlib import style

style.use("fivethirtyeight")

data = pd.read_csv("iris.data")
df = pd.DataFrame(data)
df.replace({"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}, inplace=True)
X = np.array(df.drop(["class"], axis=1))
Y = np.array(df["class"])


def K_nearest(input_features, X, Y, k=4):
    # Create a list to store distances and their corresponding classes
    distances = []
    for i in range(len(X)):
        # Calculate the Euclidean distance between input_features and X[i]
        distance = np.sqrt(np.sum((input_features - X[i]) ** 2))
        distances.append((distance, Y[i]))

    # Sort the distances and classes by distance
    sorted_distances = sorted(distances, key=lambda x: x[0])

    # Count the votes for the k nearest neighbors
    classes = {1: 0, 2: 0, 3: 0}
    for i in range(k):
        _, class_label = sorted_distances[i]
        classes[class_label] += 1

    # Return the class with the maximum votes
    return max(classes, key=lambda k: classes[k])


prediction = K_nearest([8.7, 3.0, 5.8, 2.1], X, Y, k=4)
if prediction == 1:
    print("Iris-setosa")
elif prediction == 2:
    print("Iris-versicolor")
elif prediction == 3:
    print("Iris-virginica")

#  note Have to implement the score functinality by applying one more outer for loop for all the given inputs
