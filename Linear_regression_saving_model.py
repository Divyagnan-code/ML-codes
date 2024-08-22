import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn.model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", delimiter=";")  # replace with the file with data in it
data = data[["G1", "G2", "famrel", "Dalc", "Walc", "health", "absences", "failures", "G3"]]
print(data.head())
predict = "G3"
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])
train_x, x_test, train_y, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1, random_state=44)


def train():  # call the function only the first time you are running the code to save the best performing model
    for _ in range(100):  # tune the number of iteration to look at different models according to your convenience
        best = 0.9
        train_x, x_test, train_y, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
        linear = linear_model.LinearRegression()
        linear.fit(train_x, train_y)
        acc = linear.score(x_test, y_test)
        if acc > best:
            best = acc
            with open("student.pickle", "wb") as f:
                pickle.dump(linear, f)
            print("model saved: " + str(acc))


linear = pickle.load(open("student.pickle", "rb"))
acc = linear.score(x_test, y_test)
print(f"model accuracy: {acc}")
predictions = linear.predict(x_test)
style.use("ggplot")
for i in range(len(predictions)):
    print(f"prediction: {predictions[i]}\n input: {x_test[i]}\n output: {y_test[i]}")
n_features = len(x_test[0])
n_rows = int(np.ceil(np.sqrt(n_features)))
n_cols = n_rows
fig, axs = plt.subplots(n_rows, n_cols)
axs = axs.flatten()
for i in range(n_features):
    axs[i].scatter(x_test[:, i], y_test)
    axs[i].set_xlabel(data.columns[i])
    axs[i].set_ylabel("final score")
plt.tight_layout()
plt.show()
