import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def initialize_centroids(X, k):
    np.random.seed(42)  # For reproducibility
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
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

def log_l2_norm(X, centroids, labels, iteration):

    l2_norm = np.sum(np.linalg.norm(X - centroids[labels], axis=1))
    with open("L2_norm_log.txt", "a") as f:
        f.write(f"Iteration {iteration}: Total L2 norm = {l2_norm:.6f}\n")

def kmeans(X, k, max_iters=300, tol=1e-4, plot=False):
    centroids = initialize_centroids(X, k)
    
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        # Log the L2 norm between all data points and their assigned centroids
        log_l2_norm(X, new_centroids, labels, i + 1)
        
        if plot:
            plot_progress(X, centroids, labels, i + 1)
        
        if has_converged(centroids, new_centroids, tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return image

def preprocess_image(image):
    return image.reshape(-1, image.shape[2])

def reconstruct_image(centroids, labels, image_shape):
    reconstructed_image = centroids[labels].reshape(image_shape)
    return reconstructed_image

def save_image(image, output_path):

    Image.fromarray(image.astype(np.uint8)).save(output_path)

def kmeans_image(image_path, k, output_path, max_iters=300, tol=1e-4, plot=False):

    image = load_image(image_path)
    X = preprocess_image(image)
    centroids, labels = kmeans(X, k, max_iters, tol, plot)
    clustered_image = reconstruct_image(centroids, labels, image.shape)
    
    save_image(clustered_image, output_path)
    print(f"Clustered image saved to {output_path}")

if __name__ == "__main__":
    # Run K-Means on an image and save the result
    image_path = 'lena.png'  
    output_path = 'lena_quantized.png' 
    k = 16
    outPlot = True
    kmeans_image(image_path, k, output_path, plot=False)
    
    # Optionally display the original and clustered images
    if(outPlot):
      original_image = load_image(image_path)
      clustered_image = load_image(output_path)
      
      plt.figure(figsize=(12, 6))
      
      plt.subplot(1, 2, 1)
      plt.imshow(original_image)
      plt.title('Original Image')
      
      plt.subplot(1, 2, 2)
      plt.imshow(clustered_image)
      plt.title(f'Clustered Image (k={k})')
      
      plt.show()
