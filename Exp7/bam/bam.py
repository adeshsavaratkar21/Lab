"""
Experiment 7: Associative Memory Networks
File: bam.py
Description: Bidirectional Associative Memory (BAM) network implemented in pure Python.
             Stores bipolar pattern pairs (X, Y) using correlation matrix W = sum(X_k^T * Y_k).
             Demonstrates forward recall (X -> Y), backward recall (Y -> X),
             Lyapunov energy function, and noise-tolerant pattern reconstruction.
"""

from typing import List, Tuple


def sign_activation(net: List[int], prev_state: List[int]) -> List[int]:
    """
    Threshold activation with memory retention:
    y = +1 if net > 0
    y = -1 if net < 0
    y = prev_state if net == 0
    """
    out = []
    for val, prev in zip(net, prev_state):
        if val > 0:
            out.append(1)
        elif val < 0:
            out.append(-1)
        else:
            out.append(prev)
    return out


class BidirectionalAssociativeMemory:
    def __init__(self, x_dim: int, y_dim: int):
        self.x_dim = x_dim
        self.y_dim = y_dim
        # Initialize weight matrix W of size x_dim x y_dim
        self.W = [[0] * y_dim for _ in range(x_dim)]

    def compute_weights(self, pairs: List[Tuple[List[int], List[int]]]):
        """
        Calculates correlation weight matrix: W = sum(X_k^T * Y_k)
        """
        self.W = [[0] * self.y_dim for _ in range(self.x_dim)]
        for X, Y in pairs:
            for i in range(self.x_dim):
                for j in range(self.y_dim):
                    self.W[i][j] += X[i] * Y[j]

    def forward_recall(self, X: List[int], max_iter: int = 10) -> Tuple[List[int], int]:
        """
        Forward recall: X -> Y
        Iterates until convergence: Y = f(X * W), X' = f(Y * W^T)
        """
        curr_y = [0] * self.y_dim
        curr_x = list(X)

        for step in range(1, max_iter + 1):
            # Compute net_y = curr_x * W
            net_y = [sum(curr_x[i] * self.W[i][j] for i in range(self.x_dim)) for j in range(self.y_dim)]
            next_y = sign_activation(net_y, curr_y)

            # Check backward stability
            net_x = [sum(next_y[j] * self.W[i][j] for j in range(self.y_dim)) for i in range(self.x_dim)]
            next_x = sign_activation(net_x, curr_x)

            if next_y == curr_y and next_x == curr_x:
                return next_y, step

            curr_y = next_y
            curr_x = next_x

        return curr_y, max_iter

    def backward_recall(self, Y: List[int], max_iter: int = 10) -> Tuple[List[int], int]:
        """
        Backward recall: Y -> X
        X = f(Y * W^T)
        """
        curr_x = [0] * self.x_dim
        curr_y = list(Y)

        for step in range(1, max_iter + 1):
            # Compute net_x = curr_y * W^T
            net_x = [sum(curr_y[j] * self.W[i][j] for j in range(self.y_dim)) for i in range(self.x_dim)]
            next_x = sign_activation(net_x, curr_x)

            # Check forward stability
            net_y = [sum(next_x[i] * self.W[i][j] for i in range(self.x_dim)) for j in range(self.y_dim)]
            next_y = sign_activation(net_y, curr_y)

            if next_x == curr_x and next_y == curr_y:
                return next_x, step

            curr_x = next_x
            curr_y = next_y

        return curr_x, max_iter

    def energy(self, X: List[int], Y: List[int]) -> float:
        """Lyapunov Energy: E = -X * W * Y^T"""
        e = 0
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                e -= X[i] * self.W[i][j] * Y[j]
        return float(e)

    def print_weight_matrix(self):
        print(f"BAM Weight Matrix W ({self.x_dim}x{self.y_dim}):")
        for row in self.W:
            print("  [" + ", ".join(f"{v:3d}" for v in row) + "]")


def demo_bam():
    print("=" * 65)
    print("      BIDIRECTIONAL ASSOCIATIVE MEMORY (BAM) NETWORK          ")
    print("=" * 65)

    # 2 Bipolar pattern pairs to store: (X in {-1, 1}^4, Y in {-1, 1}^2)
    pairs = [
        ([1, 1, -1, -1], [1, -1]),   # Pair 1: (X1, Y1)
        ([-1, -1, 1, 1], [-1, 1])    # Pair 2: (X2, Y2)
    ]

    bam = BidirectionalAssociativeMemory(x_dim=4, y_dim=2)
    bam.compute_weights(pairs)
    bam.print_weight_matrix()

    # 1. Test Perfect Recall
    print("\n1. Testing Pristine Recall (Both Directions):")
    print("-" * 65)
    for idx, (X, Y) in enumerate(pairs, 1):
        recalled_y, steps_fwd = bam.forward_recall(X)
        recalled_x, steps_bwd = bam.backward_recall(Y)
        e = bam.energy(X, Y)

        print(f"Pair {idx}:")
        print(f"  Input  X = {X}")
        print(f"  Target Y = {Y}")
        print(f"  Forward Recall  (X -> Y): {recalled_y} (Steps: {steps_fwd}) -> {'MATCH' if recalled_y == Y else 'MISMATCH'}")
        print(f"  Backward Recall (Y -> X): {recalled_x} (Steps: {steps_bwd}) -> {'MATCH' if recalled_x == X else 'MISMATCH'}")
        print(f"  Lyapunov Energy E(X, Y) : {e}")
        print("-" * 65)

    # 2. Test Noise Tolerance
    print("\n2. Testing Noise Tolerance & Error Correction:")
    print("Corrupting 1 bit in X1: [1, -1, -1, -1] instead of [1, 1, -1, -1]")
    noisy_x = [1, -1, -1, -1]
    recovered_y, steps = bam.forward_recall(noisy_x)
    print(f"  Noisy Input X : {noisy_x}")
    print(f"  Recovered Y   : {recovered_y} (Expected: [1, -1]) -> {'SUCCESS' if recovered_y == [1, -1] else 'FAILED'}")

    print("\nCorrupting 1 bit in X2: [-1, -1, -1, 1] instead of [-1, -1, 1, 1]")
    noisy_x2 = [-1, -1, -1, 1]
    recovered_y2, steps = bam.forward_recall(noisy_x2)
    print(f"  Noisy Input X : {noisy_x2}")
    print(f"  Recovered Y   : {recovered_y2} (Expected: [-1, 1]) -> {'SUCCESS' if recovered_y2 == [-1, 1] else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    demo_bam()

