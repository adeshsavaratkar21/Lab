# AISC Lab ESE (Artificial Intelligence & Soft Computing)

This repository contains the complete implementation of lab experiments for **Artificial Intelligence and Soft Computing (AISC)** prepared for the End Semester Examination (ESE).

---

## 📁 Repository Structure

```text
AISC_Lab_ESE/
├── Exp01_Search/
│   ├── water_jug_bfs.c
│   ├── water_jug_dfs.c
│   └── puzzle_astar.c
├── Exp02_Agents/
│   └── vacuum_agent.py
├── Exp03_MP_Neuron/
│   └── mp_adders.py
├── Exp04_Perceptron/
│   └── perceptron.py
├── Exp05_BPN/
│   └── backprop.py
├── Exp06_SOM_LVQ/
│   ├── som.py
│   └── lvq.py
├── Exp07_BAM/
│   └── bam.py
├── Exp08_Fuzzy_Sets/
│   └── fuzzy_operations.py
├── Exp09_FLC/
│   └── fuzzy_controller.py
└── Exp10_ANFIS/
    └── anfis_model.py
```

---

## 📚 Experiment Overview & Execution Guide

### Exp 01: Uninformed & Informed Search Algorithms (`C`)
* **`water_jug_bfs.c`**: Solves the classic Water Jug Problem using **Breadth-First Search (BFS)** to guarantee the shortest sequence of steps to reach the target amount.
  ```bash
  gcc AISC_Lab_ESE/Exp01_Search/water_jug_bfs.c -o water_jug_bfs
  ./water_jug_bfs
  ```
* **`water_jug_dfs.c`**: Solves the Water Jug Problem using **Depth-First Search (DFS)** with backtracking.
  ```bash
  gcc AISC_Lab_ESE/Exp01_Search/water_jug_dfs.c -o water_jug_dfs
  ./water_jug_dfs
  ```
* **`puzzle_astar.c`**: Solves the **8-Puzzle Problem** using the **A\* Search Algorithm** with Manhattan Distance heuristic ($f(n) = g(n) + h(n)$). Reconstructs step-by-step tile board movements.
  ```bash
  gcc AISC_Lab_ESE/Exp01_Search/puzzle_astar.c -o puzzle_astar
  ./puzzle_astar
  ```

---

### Exp 02: Intelligent Agents (`Python`)
* **`vacuum_agent.py`**:
  * PEAS Framework Specification (Performance, Environment, Actuators, Sensors).
  * Implementation of **Simple Reflex Agent** and **Model-Based Reflex Agent**.
  * Step-by-step simulation across 2-room and multi-room environments with full performance score tracking.
  ```bash
  python AISC_Lab_ESE/Exp02_Agents/vacuum_agent.py
  ```

---

### Exp 03: McCulloch-Pitts (MP) Neuron & Binary Adders (`Python`)
* **`mp_adders.py`**:
  * McCulloch-Pitts (MP) neuron model with excitatory and inhibitory connections.
  * Basic logic gates (AND, OR, NOT, AND-NOT).
  * Multi-layer MP network for XOR.
  * Arithmetic circuits: **Half Adder** ($S = A \oplus B, C = A \cdot B$) and **Full Adder** (Sum using cascaded MP XOR, Carry using 3-input MP majority neuron $\theta=2$).
  ```bash
  python AISC_Lab_ESE/Exp03_MP_Neuron/mp_adders.py
  ```

---

### Exp 04: Single Layer Perceptron (`Python`)
* **`perceptron.py`**:
  * Rosenblatt's Single-Layer Perceptron learning algorithm with step activation.
  * Linear separability testing on logic functions (AND, OR).
  * Decision boundary equation extraction ($w_1 x_1 + w_2 x_2 + b = 0$).
  * Non-convergence demonstration on XOR (Minsky-Papert theorem).
  ```bash
  python AISC_Lab_ESE/Exp04_Perceptron/perceptron.py
  ```

---

### Exp 05: Backpropagation Neural Network (BPN) (`Python`)
* **`backprop.py`**:
  * Multi-Layer Perceptron (MLP) with Sigmoid activation function.
  * Error backpropagation algorithm with gradient descent and momentum.
  * Solves the non-linearly separable **XOR Problem** with zero external dependencies.
  ```bash
  python AISC_Lab_ESE/Exp05_BPN/backprop.py
  ```

---

### Exp 06: Competitive Learning Networks (SOM & LVQ) (`Python`)
* **`som.py`**: **Kohonen Self-Organizing Map (SOM)** for unsupervised clustering, Best Matching Unit (BMU) selection, and topological neighbourhood updates.
  ```bash
  python AISC_Lab_ESE/Exp06_SOM_LVQ/som.py
  ```
* **`lvq.py`**: **Learning Vector Quantization (LVQ)** for supervised competitive classification with reward/penalty prototype weight vector updates.
  ```bash
  python AISC_Lab_ESE/Exp06_SOM_LVQ/lvq.py
  ```

---

### Exp 07: Bidirectional Associative Memory (BAM) (`Python`)
* **`bam.py`**:
  * Bipolar hetero-associative recurrent memory network.
  * Correlation weight matrix calculation: $W = \sum X_k^T Y_k$.
  * Bidirectional recall: Forward ($X \to Y$) and Backward ($Y \to X$).
  * Lyapunov energy function evaluation and reconstruction from corrupted/noisy inputs.
  ```bash
  python AISC_Lab_ESE/Exp07_BAM/bam.py
  ```

---

### Exp 08: Fuzzy Sets & Relations (`Python`)
* **`fuzzy_operations.py`**:
  * Standard operations: Union, Intersection, Complement.
  * Algebraic & Bounded operations: Algebraic Sum, Algebraic Product, Bounded Sum, Bounded Difference.
  * Cartesian Product of fuzzy sets.
  * Fuzzy Relations Composition: **Max-Min Composition** ($R \circ S$) and **Max-Product Composition** ($R \bullet S$).
  * Verification of De Morgan's Laws.
  ```bash
  python AISC_Lab_ESE/Exp08_Fuzzy_Sets/fuzzy_operations.py
  ```

---

### Exp 09: Fuzzy Logic Controller (FLC) (`Python`)
* **`fuzzy_controller.py`**:
  * Complete Mamdani Fuzzy Inference System (Washing Machine Controller).
  * Inputs: Dirt level (0-100%) and Grease level (0-100%).
  * Output: Wash time (0-60 minutes).
  * Triangular & Trapezoidal membership functions, min-max inference engine, and **Centroid Defuzzification (COG)**.
  ```bash
  python AISC_Lab_ESE/Exp09_FLC/fuzzy_controller.py
  ```

---

### Exp 10: Adaptive Neuro-Fuzzy Inference System (ANFIS) (`Python`)
* **`anfis_model.py`**:
  * 5-Layer ANFIS architecture based on Jang (1993) using first-order Takagi-Sugeno-Kang (TSK) FIS.
  * Layer 1 (Premise Gaussians) $\to$ Layer 2 (Firing Strengths) $\to$ Layer 3 (Normalized Strengths) $\to$ Layer 4 (Consequent Polynomials) $\to$ Layer 5 (Overall Output).
  * Gradient descent training for non-linear function approximation.
  ```bash
  python AISC_Lab_ESE/Exp10_ANFIS/anfis_model.py
  ```
