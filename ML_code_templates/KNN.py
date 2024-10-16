import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    # Step 1: Load and preprocess the data
    def load_and_preprocess_data(self, filename):
        data = pd.read_csv(filename)
        # Handle missing data by replacing missing values with column means
        data.fillna(data.mean(), inplace=True)
        return data

    # Step 2: Normalize the data
    def normalize(self, data):
        return (data - data.mean()) / data.std()

    # Step 3: Calculate Euclidean distance
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Step 4: Fit the model
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Step 5: Predict the label of a single instance
    def predict_single(self, x):
        # Calculate distances from the instance to all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Retrieve the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    # Step 6: Predict labels for a dataset
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    # Step 7: Calculate accuracy
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    # Step 8: Confusion matrix
    def confusion_matrix(self, y_true, y_pred):
        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        matrix = pd.DataFrame(0, index=unique_classes, columns=unique_classes)

        for true, pred in zip(y_true, y_pred):
            matrix.loc[true, pred] += 1

        return matrix


# Step 9: Main logic for K-Nearest Neighbors
def main_knn():
    # Initialize the KNN model
    knn_model = KNearestNeighbors(k=3)

    # Load and preprocess the data
    data = knn_model.load_and_preprocess_data('data.csv')

    # Separate features and target
    X = data.iloc[:, :-1].values  # All columns except the last
    y = data.iloc[:, -1].values  # Last column

    # Normalize the features
    X_normalized = knn_model.normalize(X)

    # Split the data into train, validation, and test sets
    np.random.seed(42)
    indices = np.random.permutation(len(X_normalized))
    train_size = int(0.7 * len(X_normalized))
    validation_size = int(0.15 * len(X_normalized))

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]

    X_train, y_train = X_normalized[train_indices], y[train_indices]
    X_validation, y_validation = X_normalized[validation_indices], y[validation_indices]
    X_test, y_test = X_normalized[test_indices], y[test_indices]

    # Fit the model
    knn_model.fit(X_train, y_train)

    # Validate the model
    y_validation_pred = knn_model.predict(X_validation)
    validation_accuracy = knn_model.accuracy(y_validation, y_validation_pred)
    print(f'Validation Accuracy: {validation_accuracy:.2f}')
    print('Confusion Matrix for Validation:')
    print(knn_model.confusion_matrix(y_validation, y_validation_pred))

    # Test the model
    y_test_pred = knn_model.predict(X_test)
    test_accuracy = knn_model.accuracy(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    print('Confusion Matrix for Test:')
    print(knn_model.confusion_matrix(y_test, y_test_pred))


# Call the main function for K-Nearest Neighbors
if __name__ == "__main__":
    main_knn()
