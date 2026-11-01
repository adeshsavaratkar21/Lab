"""
Experiment 6: Competitive Learning Networks
File: som.py
Description: Kohonen Self-Organizing Map (SOM) implementation using pure Python.
             Demonstrates unsupervised clustering, Best Matching Unit (BMU) competition,
             topological neighborhood updates, and learning rate decay.
"""

import math
import random
from typing import List, Tuple


class KohonenSOM:
    def __init__(self, input_dim: int, num_clusters: int, lr: float = 0.5, seed: int = 42):
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.initial_lr = lr
        self.lr = lr
        random.seed(seed)

        # Initialize cluster weights randomly or with small values
        self.weights = [[random.uniform(0.1, 0.9) for _ in range(input_dim)] for _ in range(num_clusters)]

    def euclidean_distance_sq(self, x: List[float], w: List[float]) -> float:
        """Computes squared Euclidean distance between input x and weight w."""
        return sum((xi - wi) ** 2 for xi, wi in zip(x, w))

    def find_bmu(self, x: List[float]) -> Tuple[int, float]:
        """Finds Best Matching Unit (BMU) with minimum Euclidean distance."""
        best_cluster = 0
        min_dist = float('inf')

        for j in range(self.num_clusters):
            dist = self.euclidean_distance_sq(x, self.weights[j])
            if dist < min_dist:
                min_dist = dist
                best_cluster = j

        return best_cluster, math.sqrt(min_dist)

    def train(self, data: List[List[float]], epochs: int = 100, radius: float = 1.0):
        """
        Trains Kohonen SOM over specified epochs.
        Updates winner (and immediate topological neighbors).
        """
        print(f"\n--- Training Kohonen SOM ({self.num_clusters} Clusters, {epochs} Epochs) ---")
        print("Initial Cluster Weights:")
        for j, w in enumerate(self.weights):
            print(f"  Cluster {j}: {[round(val, 4) for val in w]}")
        print("-" * 65)

        for epoch in range(1, epochs + 1):
            # Linearly decay learning rate
            self.lr = self.initial_lr * (1.0 - (epoch - 1) / epochs)

            for x_vec in data:
                bmu, _ = self.find_bmu(x_vec)

                # Update BMU and topological neighbors
                for j in range(self.num_clusters):
                    dist_to_bmu = abs(j - bmu)  # 1D linear topology
                    if dist_to_bmu <= radius:
                        # Influence factor h(j, bmu)
                        influence = math.exp(-(dist_to_bmu ** 2) / (2 * (max(0.1, radius) ** 2)))
                        for i in range(self.input_dim):
                            self.weights[j][i] += self.lr * influence * (x_vec[i] - self.weights[j][i])

            # Decay radius gradually
            radius = max(0.0, radius * 0.98)

        print("Final Cluster Weights after Convergence:")
        for j, w in enumerate(self.weights):
            print(f"  Cluster {j}: {[round(val, 4) for val in w]}")
        print("-" * 65)

    def predict_clusters(self, data: List[List[float]]) -> List[int]:
        """Assigns each sample to its nearest cluster."""
        assignments = []
        for x_vec in data:
            bmu, _ = self.find_bmu(x_vec)
            assignments.append(bmu)
        return assignments


def demo_som():
    print("=" * 65)
    print("         KOHONEN SELF-ORGANIZING MAP (SOM) CLUSTERING         ")
    print("=" * 65)

    # Standard textbook 4-vector dataset (4-dimensional feature vectors)
    # Vectors represent 2 distinct conceptual categories
    dataset = [
        [1.0, 1.0, 0.0, 0.0],  # Pattern A1
        [0.0, 0.0, 0.0, 1.0],  # Pattern B1
        [1.0, 0.0, 0.0, 0.0],  # Pattern A2
        [0.0, 0.0, 1.0, 1.0],  # Pattern B2
    ]
    labels = ["Pattern A1", "Pattern B1", "Pattern A2", "Pattern B2"]

    som = KohonenSOM(input_dim=4, num_clusters=2, lr=0.6, seed=10)
    som.train(dataset, epochs=200, radius=1.0)

    print("\nCluster Assignment Results:")
    print(f"{'Sample Label':<14} | {'Input Vector':<22} | {'Assigned Cluster':<18}")
    print("-" * 65)
    assignments = som.predict_clusters(dataset)
    for label, x_vec, cluster in zip(labels, dataset, assignments):
        vec_str = str(x_vec)
        print(f"{label:<14} | {vec_str:<22} | Cluster {cluster}")
    print("=" * 65)


if __name__ == "__main__":
    demo_som()
