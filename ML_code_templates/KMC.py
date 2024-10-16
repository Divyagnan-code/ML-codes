import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeansClustering:
    def __init__(self, max_iterations=100, tolerance=1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.inertia_history = []

    # Step 1: Load and preprocess the data
    def load_and_preprocess_data(self, filename):
        data = pd.read_csv(filename)
        # Handle missing data by replacing missing values with column means
        data.fillna(data.mean(), inplace=True)
        return data

    # Step 2: Normalize the data
    def normalize(self, data):
        normalized_data = (data - data.mean()) / data.std()
        return normalized_data

    # Step 3: Calculate Euclidean distance
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Step 4: Initialize centroids randomly
    def initialize_centroids(self, X, k):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], size=k, replace=False)
        return X[random_indices]

    # Step 5: Assign clusters based on the nearest centroid
    def assign_clusters(self, X):
        distances = np.array([self.euclidean_distance(X[i], self.centroids[j]) for i in range(X.shape[0]) for j in
                              range(self.centroids.shape[0])])
        distances = distances.reshape(X.shape[0], self.centroids.shape[0])
        return np.argmin(distances, axis=1)

    # Step 6: Update centroids
    def update_centroids(self, X):
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.centroids.shape[0])])
        return new_centroids

    # Step 7: Fit the model
    def fit(self, X, k):
        self.centroids = self.initialize_centroids(X, k)

        for iteration in range(self.max_iterations):
            previous_centroids = self.centroids.copy()
            self.labels = self.assign_clusters(X)
            self.centroids = self.update_centroids(X)

            # Check for convergence
            if np.linalg.norm(self.centroids - previous_centroids) < self.tolerance:
                break

        inertia = np.sum((X - self.centroids[self.labels]) ** 2)
        self.inertia_history.append(inertia)

        print(f"K-Means clustering completed for k={k}. Inertia: {inertia}")

    # Step 8: Calculate custom Silhouette Coefficient
    def silhouette_coefficient(self, X):
        if self.labels is None:
            raise ValueError("Labels have not been assigned. Please run fit() first.")

        silhouette_scores = []
        unique_labels = np.unique(self.labels)

        for i in unique_labels:
            cluster_points = X[self.labels == i]
            if len(cluster_points) < 2:
                continue  # Skip clusters with less than 2 points

            # Compute the mean distance from all points in the cluster to the centroid of the cluster
            a = np.mean([self.euclidean_distance(p, np.median(cluster_points, axis=0)) for p in cluster_points])

            # Compute the mean distance to the centroids of other clusters
            b = []
            for j in unique_labels:
                if i != j:
                    other_cluster_points = X[self.labels == j]
                    b.append(np.mean(
                        [self.euclidean_distance(p, np.median(other_cluster_points, axis=0)) for p in cluster_points]))

            # Use the minimum b value if multiple clusters are present
            b = np.min(b) if b else 0

            # Compute silhouette score for the current cluster
            if b == 0:
                silhouette_scores.append(0)
            else:
                score = (b - a) / max(a, b)
                silhouette_scores.append(score)

        return np.mean(silhouette_scores)

    # Step 9: Plotting the Elbow Method
    def plot_elbow(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.inertia_history) + 1), self.inertia_history, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Within-cluster sum of squares (Inertia)')
        plt.xticks(range(1, len(self.inertia_history) + 1))
        plt.grid()
        plt.show()


# Step 10: Main logic for K-Means Clustering
def main_kmeans():
    # Initialize the KMeans model
    kmeans_model = KMeansClustering(max_iterations=100, tolerance=1e-4)

    # Load and preprocess the data
    data = kmeans_model.load_and_preprocess_data('data.csv')

    # Normalize the data
    normalized_data = kmeans_model.normalize(data)

    # Run K-Means for multiple values of k
    max_k = 10  # You can change this value based on your needs
    for k in range(1, max_k + 1):
        kmeans_model.fit(normalized_data.values, k)
        silhouette = kmeans_model.silhouette_coefficient(normalized_data.values)
        print(f'Silhouette Coefficient for k={k}: {silhouette}')

    # Plot the elbow method
    kmeans_model.plot_elbow()


# Call the main function for K-Means Clustering
if __name__ == "__main__":
    main_kmeans()
