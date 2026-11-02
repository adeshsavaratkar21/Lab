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
│   ├── autoassociative.py
│   ├── heteroassociative.py
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
* **`Exp5/backprop.py`**: Step-by-step Backpropagation algorithm tracing forward pass (hidden & output layers), error gradients ($\delta_k$, $\delta_j$), and updated weights/biases table.
  ```bash
  python Exp5/backprop.py
  ```

---

### Exp6: Competitive Learning (SOM & LVQ) (`Python`)
* **`Exp6/som.py`**: Kohonen Self-Organizing Feature Map (KSOM) simulation computing squared Euclidean distance and updating winner cluster units for input patterns.
  ```bash
  python Exp6/som.py
  ```
* **`Exp6/lvq.py`**: Learning Vector Quantization (LVQ) simulation updating class reference prototypes with reward/penalty rules.
  ```bash
  python Exp6/lvq.py
  ```

---

### Exp7: Associative Memory Networks (`Python`)
* **`Exp7/autoassociative.py`** (or `auto.py`): Autoassociative memory network storing bipolar vector `s = [-1, 1, 1, 1]` via outer product weight matrix $W = s^T s$, with testing for exact recall, missing entries, and mistaken entries.
  ```bash
  python Exp7/autoassociative.py
  # or: python Exp7/auto.py
  ```
* **`Exp7/heteroassociative.py`** (or `hetero.py` / `hetro.py`): Heteroassociative memory network storing 4 binary pattern pairs $(S \to T)$ with step-by-step outer product accumulation and binary step recall.
  ```bash
  python Exp7/heteroassociative.py
  # or: python Exp7/hetero.py
  ```
* **`Exp7/bam.py`**: Bidirectional Associative Memory (BAM) network storing $5 \times 3$ character grids ('E' and 'F') with bipolar target associations and forward recall verification.
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
