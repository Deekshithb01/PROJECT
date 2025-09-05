import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import mean_squared_error
import time

def create_complex_terrain(x, y):
    """Create a synthetic complex terrain function for the ground truth."""
    return np.sin(0.5 * x) * np.cos(0.5 * y) + 0.3 * np.sin(2 * x) * np.cos(1.5 * y)

def generate_sample_data(n_points=1000, noise_level=0.05, sample_ratio=0.15):
    """
    Generate sample data with a complex underlying terrain and sparse, noisy measurements.
    """
    np.random.seed(42)
    x = np.random.uniform(-3, 3, n_points)
    y = np.random.uniform(-3, 3, n_points)
    coordinates = np.column_stack([x, y])

    # Create true values based on the complex terrain function
    true_values = create_complex_terrain(x, y)

    # Add Gaussian noise to simulate measurement error
    noisy_values = true_values + noise_level * np.random.randn(n_points)

    # Create a mask for sparse measurements (only sample_ratio of points are known)
    known_mask = np.random.rand(n_points) < sample_ratio
    measured_values = noisy_values.copy()
    measured_values[~known_mask] = np.nan

    return coordinates, true_values, measured_values, known_mask

def build_graph_laplacian(coordinates, method='delaunay', k_neighbors=8):
    """Build the graph Laplacian matrix L = D - A using the specified method."""
    n_points = coordinates.shape[0]

    if method == 'knn':
        A = kneighbors_graph(coordinates, n_neighbors=k_neighbors, mode='connectivity')
        A = A + A.T  # Make symmetric
        A.data = np.ones_like(A.data) # Use binary weights
    elif method == 'delaunay':
        tri = Delaunay(coordinates)
        A = lil_matrix((n_points, n_points))
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    A[simplex[i], simplex[j]] = 1
                    A[simplex[j], simplex[i]] = 1
        A = A.tocsr()
    else:
        raise ValueError("Method must be 'knn' or 'delaunay'")

    # Create Degree matrix D
    degrees = np.array(A.sum(axis=1)).flatten()
    D = diags(degrees)

    # Compute Laplacian L = D - A
    L = D - A

    return L

def biharmonic_interpolation(coordinates, measured_values, known_mask, n_eigenvectors=60, graph_method='delaunay'):
    """Perform biharmonic interpolation using graph eigenfunctions."""

    # Build graph Laplacian and the biharmonic operator B = L^2
    L = build_graph_laplacian(coordinates, method=graph_method)
    B = L @ L

    # Compute the smallest magnitude eigenvectors of the biharmonic operator
    print(f"Computing {n_eigenvectors} smallest eigenvectors...")
    start_time = time.time()
    eigenvalues, eigenvectors = eigsh(B, k=n_eigenvectors, which='SM')
    print(f"Eigen decomposition took {time.time() - start_time:.2f} seconds")

    # Extract known values and corresponding eigenvector rows
    known_indices = np.where(known_mask)[0]
    known_vals = measured_values[known_indices]
    Phi_known = eigenvectors[known_indices, :]

    # Solve the least squares problem: Phi_known * coefficients = known_vals
    coefficients, _, _, _ = np.linalg.lstsq(Phi_known, known_vals, rcond=None)

    # Reconstruct the full field using the solved coefficients
    interpolated_values = eigenvectors @ coefficients

    return interpolated_values

def plot_results(coordinates, true_values, measured_values, interpolated_values, known_mask):
    """Plot the results for visual comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=true_values, cmap='viridis', s=30)
    axes[0, 0].set_title('True Underlying Field')

    known_coords = coordinates[known_mask]
    known_vals = measured_values[known_mask]
    axes[0, 1].scatter(known_coords[:, 0], known_coords[:, 1], c=known_vals, cmap='viridis', s=30)
    axes[0, 1].set_title(f'Sparse Measurements ({np.sum(known_mask)} points)')

    axes[1, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=interpolated_values, cmap='viridis', s=30)
    axes[1, 0].set_title('Biharmonic Interpolation')

    error = interpolated_values - true_values
    sc = axes[1, 1].scatter(coordinates[:, 0], coordinates[:, 1], c=error, cmap='RdBu_r', s=30, vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Interpolation Error')
    plt.colorbar(sc, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the demonstration."""
    coordinates, true_values, measured_values, known_mask = generate_sample_data()

    interpolated_values = biharmonic_interpolation(
        coordinates, measured_values, known_mask
    )

    plot_results(coordinates, true_values, measured_values, interpolated_values, known_mask)

if __name__ == "__main__":
    main()
