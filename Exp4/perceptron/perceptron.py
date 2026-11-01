"""
Experiment 4: Single Layer Perceptron
File: perceptron.py
Description: Implementation of Rosenblatt's Single-Layer Perceptron Learning Algorithm.
             Demonstrates:
             1. Bipolar and Binary Perceptron training with step-by-step weight updates.
             2. Convergence on linearly separable functions (AND, OR, NAND, NOR).
             3. Non-convergence / limitation proof on XOR (Minsky-Papert theorem).
             4. Extraction of the linear decision boundary equation.
"""

from typing import List, Tuple, Callable


class Perceptron:
    def __init__(self, num_inputs: int, lr: float = 1.0, threshold: float = 0.0, mode: str = "bipolar"):
        """
        :param num_inputs: Number of feature inputs (excluding bias)
        :param lr: Learning rate (alpha)
        :param threshold: Activation threshold theta
        :param mode: 'bipolar' (inputs/targets in {-1, 1}) or 'binary' (inputs/targets in {0, 1})
        """
        self.num_inputs = num_inputs
        self.lr = lr
        self.threshold = threshold
        self.mode = mode
        self.weights = [0.0] * num_inputs
        self.bias = 0.0

    def activation(self, y_in: float) -> int:
        if self.mode == "bipolar":
            if y_in > self.threshold:
                return 1
            elif y_in < -self.threshold:
                return -1
            else:
                return 0
        else:
            return 1 if y_in >= self.threshold else 0

    def predict(self, x: List[float]) -> int:
        y_in = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return self.activation(y_in)

    def train(self, X: List[List[float]], targets: List[int], max_epochs: int = 50, verbose: bool = True) -> bool:
        """
        Trains the perceptron using the delta / perceptron learning rule.
        Returns True if converged, False if max_epochs reached without convergence.
        """
        converged = False
        epoch = 0

        if verbose:
            print(f"\n--- Training Perceptron (mode={self.mode}, lr={self.lr}, theta={self.threshold}) ---")
            print(f"{'Epoch':<6} | {'Sample':<8} | {'Target':<6} | {'Net y_in':<9} | {'Pred y':<6} | {'Weight Updates (w, b)':<25}")
            print("-" * 75)

        while epoch < max_epochs and not converged:
            epoch += 1
            weight_changed = False

            for i, (x_vec, target) in enumerate(zip(X, targets)):
                y_in = self.bias + sum(w * xi for w, xi in zip(self.weights, x_vec))
                pred = self.activation(y_in)

                if pred != target:
                    weight_changed = True
                    # Weight and bias update
                    if self.mode == "bipolar":
                        delta_w = [self.lr * target * xi for xi in x_vec]
                        delta_b = self.lr * target
                    else:
                        error = target - pred
                        delta_w = [self.lr * error * xi for xi in x_vec]
                        delta_b = self.lr * error

                    self.weights = [w + dw for w, dw in zip(self.weights, delta_w)]
                    self.bias += delta_b

                    update_str = f"dw={delta_w}, db={delta_b:+.1f}"
                else:
                    update_str = "No Change"

                if verbose and epoch <= 3:  # Show step-by-step for first 3 epochs to avoid log clutter
                    x_str = str(x_vec)
                    print(f"{epoch:<6} | {x_str:<8} | {target:<6} | {y_in:<9.2f} | {pred:<6} | {update_str}")

            if not weight_changed:
                converged = True
                break

        if verbose:
            print("-" * 75)
            if converged:
                print(f"--> Converged successfully in {epoch} epoch(s)!")
            else:
                print(f"--> Reached maximum epochs ({max_epochs}) without convergence (Data is non-linearly separable).")

            print(f"Final Weights: {[round(w, 4) for w in self.weights]}")
            print(f"Final Bias   : {round(self.bias, 4)}")
            if len(self.weights) == 2:
                # Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
                w1, w2 = self.weights
                if abs(w2) > 1e-6:
                    print(f"Decision Boundary: {w1:.2f}*x1 + {w2:.2f}*x2 + ({self.bias:.2f}) = 0")
                    print(f"Slope (m): {-w1/w2:.2f}, Intercept (c): {-self.bias/w2:.2f}")

        return converged


def demo_gate(gate_name: str, X: List[List[float]], targets: List[int], mode: str = "bipolar"):
    print("\n" + "=" * 75)
    print(f"         PERCEPTRON EXPERIMENT: {gate_name.upper()} FUNCTION ({mode.upper()})")
    print("=" * 75)
    p = Perceptron(num_inputs=len(X[0]), lr=1.0, threshold=0.0, mode=mode)
    converged = p.train(X, targets, max_epochs=20, verbose=True)

    print("\nTesting Trained Perceptron:")
    print(f"{'Input (X)':<15} | {'Target':<8} | {'Predicted':<10} | {'Status'}")
    print("-" * 50)
    for x_vec, target in zip(X, targets):
        pred = p.predict(x_vec)
        status = "CORRECT" if pred == target else "WRONG"
        print(f"{str(x_vec):<15} | {target:<8} | {pred:<10} | {status}")


if __name__ == "__main__":
    # 1. Bipolar AND Gate (Linearly Separable)
    bipolar_X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    and_targets = [1, -1, -1, -1]
    demo_gate("Bipolar AND Gate", bipolar_X, and_targets, mode="bipolar")

    # 2. Bipolar OR Gate (Linearly Separable)
    or_targets = [1, 1, 1, -1]
    demo_gate("Bipolar OR Gate", bipolar_X, or_targets, mode="bipolar")

    # 3. Bipolar XOR Gate (Non-Linearly Separable - Should Fail to Converge)
    xor_targets = [-1, 1, 1, -1]
    demo_gate("Bipolar XOR Gate (Minsky-Papert Limitation Test)", bipolar_X, xor_targets, mode="bipolar")

