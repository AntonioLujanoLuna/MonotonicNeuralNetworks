# Monotonic Neural Networks

This repository provides an extensive framework for implementing and studying **Monotonic Neural Networks**, which are essential in applications requiring outputs to preserve monotonic relationships with respect to certain input features. These networks are highly useful in domains like finance, healthcare, and safety-critical systems, where domain knowledge often dictates predictable trends between inputs and outputs.

## Table of Contents

1. [Introduction](#introduction)
2. [Monotonic Neural Networks Taxonomy](#monotonic-neural-networks-taxonomy)
   - [Structure-Based Monotonicity](#structure-based-monotonicity)
   - [Training-Induced Monotonicity](#training-induced-monotonicity)
3. [Methodology](#methodology)
---

## Introduction

Monotonic Neural Networks enforce monotonic relationships between input features and outputs. For example:

- **Finance:** Loan approvals increase with income.
- **Healthcare:** Higher dosage leads to stronger effects (within limits).
- **Real Estate:** House prices rise with square footage.

By enforcing monotonicity, MNNs improve **interpretability**, **trustworthiness**, and **generalization** in domains where monotonic trends are expected.

---

## Monotonic Neural Networks Taxonomy

This taxonomy categorizes MNN approaches into **Structure-Based Monotonicity** and **Training-Induced Monotonicity**:

### Structure-Based Monotonicity

1. **Weight Constraints:** Use non-negative weights in combination with monotonic activation functions (e.g., sigmoid or ReLU). 
2. **Min-Max Architectures:** Ensure monotonicity through concave-convex combinations of linear neurons.
3. **Lattice-Based Approaches:** Use multidimensional grids with monotonic constraints (e.g., Deep Lattice Networks).
4. **Extensions of Classic Neural Networks:** Add monotonic constraints to Radial Basis Function networks and Extreme Learning Machines.

### Training-Induced Monotonicity

1. **Monotonic Learning:** Modify the loss function to penalize violations of monotonicity during training. Examples include:
   - Monotonicity hints.
   - Point-wise monotonic loss functions.
2. **Verified Monotonic Networks:** Certify monotonicity during or after training:
   - Certified Monotonic Neural Networks (CMNNs) use Mixed Integer Linear Programming.
   - COMET uses counterexample-guided training with monotonic envelopes.

---

## Methodology

This repository implements and benchmark several of these proposals using Python, NumPy and PyTorch.
