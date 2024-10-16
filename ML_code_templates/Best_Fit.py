import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Linear_Regression_BestFit:
    def __init__(self):
        self.weights = None
        self.bias = None

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

    # Step 4: Train the model using Ordinary Least Squares (Best Fit)
    def train(self, X_train, y_train):
        # Use the Normal Equation: theta = (X^T * X)^(-1) * X^T * y
        X_train_with_bias = np.c_[np.ones(X_train.shape[0]), X_train]  # Adding bias as a column of 1s
        theta = np.linalg.inv(X_train_with_bias.T.dot(X_train_with_bias)).dot(X_train_with_bias.T).dot(y_train)

        # Separate weights and bias
        self.bias = theta[0]
        self.weights = theta[1:]
        print("Training complete using Best Fit Slope.")

    # Step 5: Validate the model
    def validate(self, X_val, y_val):
        y_pred = np.dot(X_val, self.weights) + self.bias
        mse_val = np.mean((y_val - y_pred) ** 2)
        print(f"Validation MSE: {mse_val}")
        return mse_val

    # Step 6: Test the model
    def test(self, X_test, y_test):
        y_pred = np.dot(X_test, self.weights) + self.bias
        mse_test = np.mean((y_test - y_pred) ** 2)
        print(f"Test MSE: {mse_test}")
        return mse_test


# Step 7: Main logic for Best Fit Slope
def main_bestfit():
    # Initialize the model
    model = Linear_Regression_BestFit()

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

    # Train the model using Best Fit Slope
    model.train(X_train, y_train)

    # Validate the model
    model.validate(X_val, y_val)

    # Test the model
    model.test(X_test, y_test)


# Call the main function for Best Fit Slope
if __name__ == "__main__":
    main_bestfit()
