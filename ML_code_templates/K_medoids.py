import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMedoidsClustering:
    def __init__(self, max_iterations=100, tolerance=1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.medoids = None
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

    # Step 4: Initialize medoids randomly
    def initialize_medoids(self, X, k):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], size=k, replace=False)
        return X[random_indices]

    # Step 5: Assign clusters based on the nearest medoid
    def assign_clusters(self, X):
        distances = np.array(
            [[self.euclidean_distance(X[i], self.medoids[j]) for j in range(self.medoids.shape[0])] for i in
             range(X.shape[0])])
        return np.argmin(distances, axis=1)

    # Step 6: Update medoids
    def update_medoids(self, X):
        new_medoids = np.copy(self.medoids)
        for i in range(self.medoids.shape[0]):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                # Compute pairwise distances
                distances = np.array(
                    [np.sum([self.euclidean_distance(p1, p2) for p2 in cluster_points]) for p1 in cluster_points])
                # Select the point with the smallest total distance as the new medoid
                new_medoids[i] = cluster_points[np.argmin(distances)]
        return new_medoids

    # Step 7: Fit the model
    def fit(self, X, k):
        self.medoids = self.initialize_medoids(X, k)

        for iteration in range(self.max_iterations):
            previous_medoids = self.medoids.copy()
            self.labels = self.assign_clusters(X)
            self.medoids = self.update_medoids(X)

            # Check for convergence
            if np.all(self.medoids == previous_medoids):
                break

        inertia = np.sum([self.euclidean_distance(X[i], self.medoids[self.labels[i]]) ** 2 for i in range(X.shape[0])])
        self.inertia_history.append(inertia)

        print(f"K-Medoids clustering completed for k={k}. Inertia: {inertia}")

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

            # Compute the mean distance from all points in the cluster to the medoid of the cluster
            a = np.mean([self.euclidean_distance(p, self.medoids[i]) for p in cluster_points])

            # Compute the mean distance to the medoids of other clusters
            b = []
            for j in unique_labels:
                if i != j:
                    other_cluster_points = X[self.labels == j]
                    b.append(np.mean([self.euclidean_distance(p, self.medoids[j]) for p in cluster_points]))

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


# Step 10: Main logic for K-Medoids Clustering
def main_kmedoids():
    # Initialize the KMedoids model
    kmedoids_model = KMedoidsClustering(max_iterations=100, tolerance=1e-4)

    # Load and preprocess the data
    data = kmedoids_model.load_and_preprocess_data('data.csv')

    # Normalize the data
    normalized_data = kmedoids_model.normalize(data)

    # Run K-Medoids for multiple values of k
    max_k = 10  # You can change this value based on your needs
    for k in range(1, max_k + 1):
        kmedoids_model.fit(normalized_data.values, k)
        silhouette = kmedoids_model.silhouette_coefficient(normalized_data.values)
        print(f'Silhouette Coefficient for k={k}: {silhouette}')

    # Plot the elbow method
    kmedoids_model.plot_elbow()


# Call the main function for K-Medoids Clustering
if __name__ == "__main__":
    main_kmedoids()
