"""
Experiment 3: McCulloch-Pitts (MP) Neuron & Binary Adders
File: mp_adders.py
Description: Implementation of McCulloch-Pitts (MP) Neuron model to simulate:
             1. Basic logic gates (AND, OR, NOT, AND-NOT)
             2. Non-linear XOR gate using multi-layer MP neurons
             3. Half Adder (Sum = A XOR B, Carry = A AND B)
             4. Full Adder (Sum = A XOR B XOR Cin, Carry = Majority(A, B, Cin))
"""

from typing import List, Tuple, Optional


class MPNeuron:
    """
    McCulloch-Pitts (MP) Neuron Model:
    - Excitatory inputs contribute positively towards reaching the threshold theta.
    - Inhibitory inputs, if active (= 1), immediately suppress the neuron firing (output = 0).
    """
    def __init__(self, weights: List[int], threshold: int, inhibitory_indices: Optional[List[int]] = None):
        self.weights = weights
        self.threshold = threshold
        self.inhibitory_indices = inhibitory_indices if inhibitory_indices else []

    def activate(self, inputs: List[int]) -> int:
        # Check inhibitory inputs: any active inhibitor vetoes firing
        for idx in self.inhibitory_indices:
            if inputs[idx] == 1:
                return 0

        # Calculate net input from excitatory connections
        net_input = sum(w * x for i, (w, x) in enumerate(zip(self.weights, inputs)) if i not in self.inhibitory_indices)
        return 1 if net_input >= self.threshold else 0


# -------------------------------------------------------------
# Basic Logic Gate Implementations using MP Neurons
# -------------------------------------------------------------

def mp_and(x1: int, x2: int) -> int:
    """AND Gate: w = [1, 1], theta = 2"""
    neuron = MPNeuron(weights=[1, 1], threshold=2)
    return neuron.activate([x1, x2])


def mp_or(x1: int, x2: int) -> int:
    """OR Gate: w = [1, 1], theta = 1"""
    neuron = MPNeuron(weights=[1, 1], threshold=1)
    return neuron.activate([x1, x2])


def mp_not(x: int) -> int:
    """NOT Gate: 1 inhibitory input, theta = 0"""
    neuron = MPNeuron(weights=[0], threshold=0, inhibitory_indices=[0])
    return neuron.activate([x])


def mp_and_not(x1: int, x2: int) -> int:
    """AND-NOT (x1 AND NOT x2): x1 excitatory (w=1), x2 inhibitory, theta = 1"""
    neuron = MPNeuron(weights=[1, 0], threshold=1, inhibitory_indices=[1])
    return neuron.activate([x1, x2])


def mp_xor(x1: int, x2: int) -> int:
    """
    XOR Gate: Non-linearly separable.
    Implemented as a 2-layer network:
    z1 = x1 AND NOT x2
    z2 = NOT x1 AND x2
    y  = z1 OR z2
    """
    z1 = mp_and_not(x1, x2)  # x1 * ~x2
    z2 = mp_and_not(x2, x1)  # ~x1 * x2
    return mp_or(z1, z2)


# -------------------------------------------------------------
# Arithmetic Circuits using MP Neurons
# -------------------------------------------------------------

def half_adder(a: int, b: int) -> Tuple[int, int]:
    """
    Half Adder Circuit:
    - Sum   = A XOR B (using MP XOR network)
    - Carry = A AND B (using MP AND neuron)
    """
    s = mp_xor(a, b)
    c = mp_and(a, b)
    return s, c


def full_adder(a: int, b: int, cin: int) -> Tuple[int, int]:
    """
    Full Adder Circuit:
    - Sum   = A XOR B XOR Cin (Cascaded MP XOR)
    - Carry = Majority function of 3 inputs:
              MP Neuron with weights = [1, 1, 1] and threshold theta = 2
              (fires when at least 2 out of 3 inputs are 1).
    """
    # Sum: Cascaded XOR
    sum_ab = mp_xor(a, b)
    s = mp_xor(sum_ab, cin)

    # Carry: Majority MP Neuron (theta = 2, w = [1, 1, 1])
    carry_neuron = MPNeuron(weights=[1, 1, 1], threshold=2)
    c_out = carry_neuron.activate([a, b, cin])

    return s, c_out


def display_truth_tables():
    """Prints formatted truth tables for all gates and adders."""
    print("=" * 60)
    print("       McCULLOCH-PITTS (MP) NEURON TRUTH TABLES       ")
    print("=" * 60)

    # 1. Basic Gates
    print("\n1. Basic Logic Gates Verification:")
    print("-" * 50)
    print(f"{'X1':<5} | {'X2':<5} | {'AND':<6} | {'OR':<6} | {'NOT(X1)':<8} | {'XOR':<6}")
    print("-" * 50)
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"{x1:<5} | {x2:<5} | {mp_and(x1, x2):<6} | {mp_or(x1, x2):<6} | {mp_not(x1):<8} | {mp_xor(x1, x2):<6}")
    print("-" * 50)

    # 2. Half Adder
    print("\n2. Half Adder Circuit:")
    print("Formula: Sum = A XOR B, Carry = A AND B")
    print("-" * 45)
    print(f"{'A':<5} | {'B':<5} | {'Sum (S)':<10} | {'Carry (C)':<10}")
    print("-" * 45)
    for a in [0, 1]:
        for b in [0, 1]:
            s, c = half_adder(a, b)
            print(f"{a:<5} | {b:<5} | {s:<10} | {c:<10}")
    print("-" * 45)

    # 3. Full Adder
    print("\n3. Full Adder Circuit:")
    print("Formula: Sum = A XOR B XOR Cin, Carry = Majority(A, B, Cin) [theta=2]")
    print("-" * 55)
    print(f"{'A':<5} | {'B':<5} | {'Cin':<5} | {'Sum (S)':<10} | {'Carry (Cout)':<12}")
    print("-" * 55)
    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                s, cout = full_adder(a, b, cin)
                print(f"{a:<5} | {b:<5} | {cin:<5} | {s:<10} | {cout:<12}")
    print("-" * 55)


if __name__ == "__main__":
    display_truth_tables()
