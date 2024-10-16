import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Logistic_Regression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.nll_history = []

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

    # Step 5: Sigmoid function (hypothesis)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Step 6: Negative Log-Likelihood (NLL) loss
    def compute_nll(self, y_true, y_pred):
        # Clip predictions to avoid log(0) and overflow
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Step 7: Confusion Matrix
    def confusion_matrix(self, y_true, y_pred, threshold=0.5):
        y_pred_class = (y_pred >= threshold).astype(int)
        true_positive = np.sum((y_pred_class == 1) & (y_true == 1))
        true_negative = np.sum((y_pred_class == 0) & (y_true == 0))
        false_positive = np.sum((y_pred_class == 1) & (y_true == 0))
        false_negative = np.sum((y_pred_class == 0) & (y_true == 1))

        print(f"Confusion Matrix:")
        print(f"TP: {true_positive}, FP: {false_positive}")
        print(f"TN: {true_negative}, FN: {false_negative}")

        return true_positive, false_positive, true_negative, false_negative

    # Step 8: Train the model using Negative Log Likelihood
    def train(self, X_train, y_train):
        m = len(y_train)
        self.nll_history = []

        for iteration in range(self.iterations):
            # Compute predictions (hypothesis)
            y_pred = self.sigmoid(np.dot(X_train, self.weights) + self.bias)

            # Calculate gradients
            dw = (1 / m) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / m) * np.sum(y_pred - y_train)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate and store NLL after each iteration
            nll = self.compute_nll(y_train, y_pred)
            self.nll_history.append(nll)

        print("Training complete using Negative Log Likelihood.")

        # Confusion Matrix on Training Data
        print("Confusion Matrix on Training Data:")
        self.confusion_matrix(y_train, y_pred)

    # Step 9: Validate the model
    def validate(self, X_val, y_val):
        y_pred = self.sigmoid(np.dot(X_val, self.weights) + self.bias)
        nll_val = self.compute_nll(y_val, y_pred)
        print(f"Validation NLL: {nll_val}")

        # Confusion Matrix on Validation Data
        print("Confusion Matrix on Validation Data:")
        self.confusion_matrix(y_val, y_pred)

        return nll_val

    # Step 10: Test the model
    def test(self, X_test, y_test):
        y_pred = self.sigmoid(np.dot(X_test, self.weights) + self.bias)
        nll_test = self.compute_nll(y_test, y_pred)
        print(f"Test NLL: {nll_test}")

        # Confusion Matrix on Test Data
        print("Confusion Matrix on Test Data:")
        self.confusion_matrix(y_test, y_pred)

        return nll_test

    # Step 11: Plot the training Negative Log Likelihood (NLL) history
    def plot_training_curve(self):
        plt.plot(self.nll_history)
        plt.xlabel('Iterations')
        plt.ylabel('Negative Log Likelihood')
        plt.title('Training NLL over iterations (Logistic Regression)')
        plt.show()


# Step 12: Main logic for Logistic Regression
def main_logistic():
    # Initialize the model
    model = Logistic_Regression(learning_rate=0.01, iterations=1000)

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

    # Train the model using Negative Log Likelihood
    model.train(X_train, y_train)

    # Validate the model
    model.validate(X_val, y_val)

    # Test the model
    model.test(X_test, y_test)

    # Plot the training NLL history
    model.plot_training_curve()


# Call the main function for Logistic Regression
if __name__ == "__main__":
    main_logistic()
