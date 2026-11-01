# AISC Lab (Artificial Intelligence & Soft Computing)

This repository contains the complete implementation of lab experiments for **Artificial Intelligence and Soft Computing (AISC)**.

---

## 📁 Repository Structure

```text
Lab/
├── Exp1/
│   ├── bfs.c
│   ├── dfs.c
│   └── puzzle.c
├── Exp2/
│   └── vacuum_agent.py
├── Exp3/
│   └── mp_adders.py
├── Exp4/
│   ├── perceptron.py
│   └── adaline.py
├── Exp5/
│   └── backprop.py
├── Exp6/
│   ├── som.py
│   └── lvq.py
├── Exp7/
│   └── bam.py
├── Exp8/
│   └── fuzzy_operations.py
├── Exp9/
│   └── fuzzy_controller.py
└── Exp10/
    └── anfis_model.py
```

---

## 📚 Experiment Details & Execution Guide

### Exp1: Search Algorithms (`C`)
* **`Exp1/bfs.c`**: Water Jug Problem solved using Breadth-First Search (BFS) with queue traversal.
  ```bash
  gcc Exp1/bfs.c -o bfs && ./bfs
  ```
* **`Exp1/dfs.c`**: Water Jug Problem solved using Depth-First Search (DFS) with stack traversal.
  ```bash
  gcc Exp1/dfs.c -o dfs && ./dfs
  ```
* **`Exp1/puzzle.c`**: 8-Puzzle solver using A* search algorithm with Manhattan Distance heuristic.
  ```bash
  gcc Exp1/puzzle.c -o puzzle && ./puzzle
  ```

---

### Exp2: Intelligent Agents (`Python`)
* **`Exp2/vacuum_agent.py`**: PEAS framework, Simple Reflex Agent, and Model-Based Reflex Agent with room state memory and performance tracking.
  ```bash
  python Exp2/vacuum_agent.py
  ```

---

### Exp3: McCulloch-Pitts (MP) Neuron & Binary Adders (`Python`)
* **`Exp3/mp_adders.py`**: MP Neuron model simulating logic gates, multi-layer MP XOR, Half Adder, and Full Adder.
  ```bash
  python Exp3/mp_adders.py
  ```

---

### Exp4: Single Layer Perceptron & Adaline Network (`Python`)
* **`Exp4/perceptron.py`**: Perceptron learning rule for Bipolar AND gate with epoch-by-epoch weights & decision boundary table.
  ```bash
  python Exp4/perceptron.py
  ```
* **`Exp4/adaline.py`**: Adaline Network using Delta Rule (Widrow-Hoff) for Bipolar OR gate with continuous error and MSE calculation.
  ```bash
  python Exp4/adaline.py
  ```

---

### Exp5: Backpropagation Neural Network (BPN) (`Python`)
* **`Exp5/backprop.py`**: Multi-Layer Perceptron (MLP) trained with backpropagation, gradient descent, and momentum solving XOR.
  ```bash
  python Exp5/backprop.py
  ```

---

### Exp6: Competitive Learning (SOM & LVQ) (`Python`)
* **`Exp6/som.py`**: Kohonen Self-Organizing Map (SOM) for unsupervised clustering with BMU selection and decaying neighbourhood.
  ```bash
  python Exp6/som.py
  ```
* **`Exp6/lvq.py`**: Learning Vector Quantization (LVQ) for supervised competitive classification with prototype weight updates.
  ```bash
  python Exp6/lvq.py
  ```

---

### Exp7: Bidirectional Associative Memory (BAM) (`Python`)
* **`Exp7/bam.py`**: BAM network storing bipolar pattern pairs ($W = \sum X_k^T Y_k$), bidirectional recall ($X \leftrightarrow Y$), and noise tolerance.
  ```bash
  python Exp7/bam.py
  ```

---

### Exp8: Fuzzy Sets & Relations (`Python`)
* **`Exp8/fuzzy_operations.py`**: Standard and algebraic fuzzy set operations, Cartesian product, Max-Min and Max-Product composition, and De Morgan's laws.
  ```bash
  python Exp8/fuzzy_operations.py
  ```

---

### Exp9: Fuzzy Logic Controller (FLC) (`Python`)
* **`Exp9/fuzzy_controller.py`**: Mamdani Fuzzy Inference System for Washing Machine wash time controller using Centroid Defuzzification (COG).
  ```bash
  python Exp9/fuzzy_controller.py
  ```

---

### Exp10: Adaptive Neuro-Fuzzy Inference System (ANFIS) (`Python`)
* **`Exp10/anfis_model.py`**: 5-Layer ANFIS architecture with first-order Takagi-Sugeno-Kang (TSK) fuzzy inference model and parameter tuning.
  ```bash
  python Exp10/anfis_model.py
  ```
