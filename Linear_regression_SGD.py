import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Linear_Regression_SGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.mse_history = []

    # Step 1: Load and preprocess the data
    def load_and_preprocess_data(self, filename):
        data = pd.read_csv(filename)
        # Handle missing data by replacing missing values with column means
        data.fillna(data.mean(), inplace=True)
        return data

    # Step 2: Normalize the data
    def normalize(self, data):
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1]
        normalized_features = (features - features.mean()) / features.std()
        return pd.concat([normalized_features, target], axis=1)

    # Step 3: Split the data into train, validation, and test sets
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        np.random.seed(42)
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        train_size = int(train_ratio * len(shuffled_data))
        val_size = int(val_ratio * len(shuffled_data))

        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]

        return train_data, val_data, test_data

    # Step 4: Initialize parameters randomly
    def initialize_parameters(self, n_features):
        np.random.seed(42)
        self.weights = np.random.randn(n_features)
        self.bias = 0.0  # Initialize bias to 0

    # Step 5: Compute Mean Squared Error (MSE)
    def compute_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Step 6: Train the model using Stochastic Gradient Descent
    def train(self, X_train, y_train):
        m = len(y_train)
        self.mse_history = []

        for iteration in range(self.iterations):
            # Shuffle training data
            np.random.seed(iteration)
            indices = np.random.permutation(m)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(m):
                # Single sample
                X_i = X_train_shuffled[i].reshape(1, -1)
                y_i = y_train_shuffled[i]

                # Predicted value
                y_pred = np.dot(X_i, self.weights) + self.bias

                # Compute gradient for weights and bias
                weight_gradient = 2 * X_i.T * (y_pred - y_i)
                bias_gradient = 2 * (y_pred - y_i)

                # Update weights and bias
                self.weights -= self.learning_rate * weight_gradient.flatten()
                self.bias -= self.learning_rate * bias_gradient

            # Calculate and store MSE after each iteration
            y_pred_all = np.dot(X_train, self.weights) + self.bias
            mse = self.compute_mse(y_train, y_pred_all)
            self.mse_history.append(mse)

        print("Training complete using Stochastic Gradient Descent.")

    # Step 7: Validate the model
    def validate(self, X_val, y_val):
        y_pred = np.dot(X_val, self.weights) + self.bias
        mse_val = self.compute_mse(y_val, y_pred)
        print(f"Validation MSE: {mse_val}")
        return mse_val

    # Step 8: Test the model
    def test(self, X_test, y_test):
        y_pred = np.dot(X_test, self.weights) + self.bias
        mse_test = self.compute_mse(y_test, y_pred)
        print(f"Test MSE: {mse_test}")
        return mse_test

    # Step 9: Plot the training MSE history
    def plot_training_curve(self):
        plt.plot(self.mse_history)
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('Training MSE over iterations (SGD)')
        plt.show()


# Step 10: Main logic for SGD
def main_sgd():
    # Initialize the model
    model = Linear_Regression_SGD(learning_rate=0.01, iterations=1000)

    # Load and preprocess the data
    data = model.load_and_preprocess_data('data.csv')

    # Normalize the data
    data = model.normalize(data)

    # Split the data into train, validation, and test sets
    train_data, val_data, test_data = model.split_data(data)

    # Separate features and target for each set
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Initialize parameters
    model.initialize_parameters(X_train.shape[1])

    # Train the model using Stochastic Gradient Descent
    model.train(X_train, y_train)

    # Validate the model
    model.validate(X_val, y_val)

    # Test the model
    model.test(X_test, y_test)

    # Plot the training MSE history
    model.plot_training_curve()


# Call the main function for SGD
if __name__ == "__main__":
    main_sgd()
