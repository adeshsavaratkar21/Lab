# AISC Lab (Artificial Intelligence & Soft Computing)

This repository contains the complete implementation of lab experiments for **Artificial Intelligence and Soft Computing (AISC)**.

---

## 📁 Repository Structure

```text
Lab/
├── Exp1/
│   ├── bfs/
│   │   └── bfs.c
│   ├── dfs/
│   │   └── dfs.c
│   └── puzzle/
│       └── puzzle.c
├── Exp2/
│   └── vacuum/
│       └── vacuum.py
├── Exp3/
│   └── adders/
│       └── adders.py
├── Exp4/
│   └── perceptron/
│       └── perceptron.py
├── Exp5/
│   └── backprop/
│       └── backprop.py
├── Exp6/
│   ├── som/
│   │   └── som.py
│   └── lvq/
│       └── lvq.py
├── Exp7/
│   └── bam/
│       └── bam.py
├── Exp8/
│   └── fuzzy/
│       └── fuzzy.py
├── Exp9/
│   └── controller/
│       └── controller.py
└── Exp10/
    └── anfis/
        └── anfis.py
```

---

## 📚 Experiment Details & Execution Guide

### Exp1: Search Algorithms (`C`)
* **`bfs/bfs.c`**: Water Jug Problem solved using Breadth-First Search (BFS) with queue traversal.
  ```bash
  cd Exp1/bfs
  gcc bfs.c -o bfs && ./bfs
  ```
* **`dfs/dfs.c`**: Water Jug Problem solved using Depth-First Search (DFS) with stack traversal.
  ```bash
  cd Exp1/dfs
  gcc dfs.c -o dfs && ./dfs
  ```
* **`puzzle/puzzle.c`**: 8-Puzzle solver using A* search algorithm with Manhattan Distance heuristic.
  ```bash
  cd Exp1/puzzle
  gcc puzzle.c -o puzzle && ./puzzle
  ```

---

### Exp2: Intelligent Agents (`Python`)
* **`vacuum/vacuum.py`**: PEAS framework, Simple Reflex Agent, and Model-Based Reflex Agent with room state memory and performance tracking.
  ```bash
  python Exp2/vacuum/vacuum.py
  ```

---

### Exp3: McCulloch-Pitts (MP) Neuron & Binary Adders (`Python`)
* **`adders/adders.py`**: MP Neuron model simulating logic gates, multi-layer MP XOR, Half Adder, and Full Adder.
  ```bash
  python Exp3/adders/adders.py
  ```

---

### Exp4: Single Layer Perceptron (`Python`)
* **`perceptron/perceptron.py`**: Rosenblatt's Perceptron learning algorithm, decision boundary, and linear separability vs XOR limitation.
  ```bash
  python Exp4/perceptron/perceptron.py
  ```

---

### Exp5: Backpropagation Neural Network (BPN) (`Python`)
* **`backprop/backprop.py`**: Multi-Layer Perceptron (MLP) trained with backpropagation, gradient descent, and momentum solving XOR.
  ```bash
  python Exp5/backprop/backprop.py
  ```

---

### Exp6: Competitive Learning (SOM & LVQ) (`Python`)
* **`som/som.py`**: Kohonen Self-Organizing Map (SOM) for unsupervised clustering with BMU selection and decaying neighbourhood.
  ```bash
  python Exp6/som/som.py
  ```
* **`lvq/lvq.py`**: Learning Vector Quantization (LVQ) for supervised competitive classification with prototype weight updates.
  ```bash
  python Exp6/lvq/lvq.py
  ```

---

### Exp7: Bidirectional Associative Memory (BAM) (`Python`)
* **`bam/bam.py`**: BAM network storing bipolar pattern pairs ($W = \sum X_k^T Y_k$), bidirectional recall ($X \leftrightarrow Y$), and noise tolerance.
  ```bash
  python Exp7/bam/bam.py
  ```

---

### Exp8: Fuzzy Sets & Relations (`Python`)
* **`fuzzy/fuzzy.py`**: Standard and algebraic fuzzy set operations, Cartesian product, Max-Min and Max-Product composition, and De Morgan's laws.
  ```bash
  python Exp8/fuzzy/fuzzy.py
  ```

---

### Exp9: Fuzzy Logic Controller (FLC) (`Python`)
* **`controller/controller.py`**: Mamdani Fuzzy Inference System for Washing Machine wash time controller using Centroid Defuzzification (COG).
  ```bash
  python Exp9/controller/controller.py
  ```

---

### Exp10: Adaptive Neuro-Fuzzy Inference System (ANFIS) (`Python`)
* **`anfis/anfis.py`**: 5-Layer ANFIS architecture with first-order Takagi-Sugeno-Kang (TSK) fuzzy inference model and parameter tuning.
  ```bash
  python Exp10/anfis/anfis.py
  ```
