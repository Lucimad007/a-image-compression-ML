import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):

    np.random.seed(42)  # For reproducibility
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids

def assign_clusters(X, centroids):
    # assign each point to its centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    # recompute centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def has_converged(old_centroids, new_centroids, tol=1e-4):
    return np.all(np.linalg.norm(new_centroids - old_centroids, axis=1) < tol)

def plot_progress(X, centroids, labels, iteration):
    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        points = X[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=30, label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering - Iteration {iteration}')
    plt.legend()
    plt.show()

def kmeans(X, k, max_iters=300, tol=1e-4, plot=False):
    centroids = initialize_centroids(X, k)

    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if plot:
            plot_progress(X, centroids, labels, i + 1)

        if has_converged(centroids, new_centroids, tol):
            break

        centroids = new_centroids

    return centroids, labels

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(10000, 2)

    # Run K-Means with plotting enabled
    k = 7
    centroids, labels = kmeans(X, k, plot=True)

    print("Final Centroids:")
    print(centroids)
