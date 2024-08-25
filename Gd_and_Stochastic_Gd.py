import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, Y, test_x, test_y, lr=0.001, n_iter=1000, conv=0.001):
        self.lr = lr
        self.iter = n_iter
        self.conv = conv
        self.X = X
        self.Y = Y
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]
        self.weights = np.zeros(self.n_features)  # initialize a random value not zero
        self.bias = 0
        self.test_x = test_x
        self.test_y = test_y
        self.cost_history = []
        self.final_hypothesis = None

    def compute_cost(self, predictions):
        return np.mean((predictions - self.Y) ** 2)

    def train_GD(self):
        self.cost_history = []
        for _ in range(self.iter):
            hyp_old = np.dot(self.X, self.weights) + self.bias
            error = hyp_old - self.Y
            self.weights -= self.lr * np.dot(self.X.T, error) / self.n_samples
            self.bias -= self.lr * np.mean(error)
            hyp_new = np.dot(self.X, self.weights) + self.bias
            cost = self.compute_cost(hyp_new)
            self.cost_history.append(cost)

        self.final_hypothesis = np.dot(self.X, self.weights) + self.bias
        return self.weights, self.bias

    def train_SD(self):
        self.cost_history = []
        for _ in range(self.iter):
            instance = random.randint(0, self.n_samples - 1)
            prediction = np.dot(self.X[instance], self.weights) + self.bias
            error = prediction - self.Y[instance]
            self.weights -= self.lr * error * self.X[instance]
            self.bias -= self.lr * error
            hyp_new = np.dot(self.X, self.weights) + self.bias
            cost = self.compute_cost(hyp_new)
            self.cost_history.append(cost)

        self.final_hypothesis = np.dot(self.X, self.weights) + self.bias
        return self.weights, self.bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def test(self):
        predictions = self.predict(self.test_x)
        mse = np.mean((predictions - self.test_y) ** 2)
        print(f"Mean Squared Error: {mse}")
        return mse

    def plot_comparison(self, gd_hypothesis, sd_hypothesis, gd_cost, sd_cost):
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.scatter(self.X, self.Y, label='Original Data')
        plt.plot(self.X, gd_hypothesis, color='red', label='GD Hypothesis')
        plt.title('Gradient Descent Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.scatter(self.X, self.Y, label='Original Data')
        plt.plot(self.X, sd_hypothesis, color='blue', label='SGD Hypothesis')
        plt.title('Stochastic Gradient Descent Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(gd_cost, color='red')
        plt.title('Gradient Descent Cost Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(sd_cost, color='blue')
        plt.title('Stochastic Gradient Descent Cost Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("studentperformance.csv", delimiter=',')  # change the path to csv file
    data.sample(frac=1, random_state=42)
    split_ratio = random.randint(5, 10)*0.1
    split_index = int(split_ratio * len(data))
    print(split_index)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    train_x = np.array(train_data.iloc[:, 1:])
    train_y = np.array(train_data.iloc[:, 0])
    test_x = np.array(test_data.iloc[:, 1:])
    test_y = np.array(test_data.iloc[:, 0])
    linear_gd = LinearRegression(train_x, train_y, test_x, test_y, lr=0.001, n_iter=1000, conv=0.001)
    linear_gd.train_GD()
    gd_hypothesis = linear_gd.final_hypothesis
    linear_sd = LinearRegression(train_x, train_y, test_x, test_y, lr=0.001, n_iter=1000, conv=0.001)
    linear_sd.train_SD()
    sd_hypothesis = linear_sd.final_hypothesis
    linear_gd.plot_comparison(gd_hypothesis, sd_hypothesis, linear_gd.cost_history, linear_sd.cost_history)
    print("Gradient Descent:")
    linear_gd.test()
    print("\nStochastic Gradient Descent:")
    linear_sd.test()
