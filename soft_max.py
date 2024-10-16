import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Multiclass_Logistic_Regression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    # Step 1: Load and preprocess the data
    def load_and_preprocess_data(self, filename):
        data = pd.read_csv(filename)
        # Handle missing data by replacing missing values with column means
        data.fillna(data.mean(), inplace=True)
        return data

    # Step 2: Normalize the data
    def normalize(self, data):
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1].astype(int)  # Ensure target is of integer type
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
    def initialize_parameters(self, n_features, n_classes):
        np.random.seed(42)
        self.weights = np.random.randn(n_classes, n_features)
        self.bias = np.zeros(n_classes)  # Initialize bias to zero for each class

    # Step 5: Softmax function
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Step 6: Cross-entropy loss
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    # Step 7: Confusion Matrix
    def confusion_matrix(self, y_true, y_pred):
        conf_matrix = np.zeros((np.unique(y_true).size, np.unique(y_true).size), dtype=int)
        for i in range(len(y_true)):
            conf_matrix[y_true[i], y_pred[i]] += 1

        print("Confusion Matrix:")
        print(conf_matrix)

        return conf_matrix

    # Step 8: Train the model using Softmax regression
    def train(self, X_train, y_train):
        m = len(y_train)
        self.loss_history = []

        for iteration in range(self.iterations):
            # Compute predictions (hypothesis)
            linear_model = np.dot(X_train, self.weights.T) + self.bias
            y_pred = self.softmax(linear_model)

            # Calculate gradients
            y_one_hot = np.zeros((m, np.max(y_train) + 1))
            y_one_hot[range(m), y_train] = 1  # Convert to one-hot encoding
            dw = (1 / m) * np.dot((y_pred - y_one_hot).T, X_train)
            db = (1 / m) * np.sum(y_pred - y_one_hot, axis=0)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate and store loss after each iteration
            loss = self.compute_loss(y_train, y_pred)
            self.loss_history.append(loss)

        print("Training complete using Softmax regression.")

        # Confusion Matrix on Training Data
        print("Confusion Matrix on Training Data:")
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.confusion_matrix(y_train, y_pred_classes)

    # Step 9: Validate the model
    def validate(self, X_val, y_val):
        linear_model = np.dot(X_val, self.weights.T) + self.bias
        y_pred = self.softmax(linear_model)
        loss_val = self.compute_loss(y_val, np.argmax(y_pred, axis=1))

        print(f"Validation Loss: {loss_val}")

        # Confusion Matrix on Validation Data
        print("Confusion Matrix on Validation Data:")
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.confusion_matrix(y_val, y_pred_classes)

        return loss_val

    # Step 10: Test the model
    def test(self, X_test, y_test):
        linear_model = np.dot(X_test, self.weights.T) + self.bias
        y_pred = self.softmax(linear_model)
        loss_test = self.compute_loss(y_test, np.argmax(y_pred, axis=1))

        print(f"Test Loss: {loss_test}")

        # Confusion Matrix on Test Data
        print("Confusion Matrix on Test Data:")
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.confusion_matrix(y_test, y_pred_classes)

        return loss_test

    # Step 11: Plot the training loss history
    def plot_training_curve(self):
        plt.plot(self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss over iterations (Multiclass Logistic Regression)')
        plt.show()


# Step 12: Main logic for Multiclass Logistic Regression
def main_multiclass():
    # Initialize the model
    model = Multiclass_Logistic_Regression(learning_rate=0.01, iterations=1000)

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
    model.initialize_parameters(X_train.shape[1], len(np.unique(y_train)))

    # Train the model using Softmax regression
    model.train(X_train, y_train)

    # Validate the model
    model.validate(X_val, y_val)

    # Test the model
    model.test(X_test, y_test)

    # Plot the training loss history
    model.plot_training_curve()


# Call the main function for Multiclass Logistic Regression
if __name__ == "__main__":
    main_multiclass()
