# Federated Learning for Breast Cancer Diagnosis

This repository evaluates **federated learning (FL)** methods for privacy-preserving breast cancer diagnosis under realistic **non-IID clinical settings**. The study compares **FedAvg** and **FedProx** across multiple model architectures with and without **differential privacy (DP)**.

---

## Repository Structure
,,,
CNN.ipynb # 1D CNN federated experiments
MLP.ipynb # Multilayer Perceptron experiments
Random_Forest.ipynb # Random Forest baseline and FL experiments
TABNET.ipynb # TabNet federated learning experiments
FL_.ipynb # Shared FL utilities and configuration
,,,

---

##  Models

- Random Forest (RF)
- Multilayer Perceptron (MLP)
- 1D Convolutional Neural Network (CNN)
- TabNet

Each model is evaluated under:
- Centralized training
- FedAvg
- FedAvg + Differential Privacy
- FedProx
- FedProx + Differential Privacy

---

##  Dataset

- Breast Cancer Wisconsin Diagnostic (WDBC)
- 569 samples, 30 features
- Binary classification (Benign / Malignant)

Data is split into **10 fixed federated clients** with unequal sizes and class distributions.

---

## Experimental Setup

- Non-IID client partitioning
- Distribution shift via Gaussian noise
- Client drift via feature perturbations
- Differential privacy using gradient clipping and Gaussian noise
- 50 trials per experiment with a shared global test set

---

## Evaluation

Metrics:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score

Statistical validation:
- Paired t-tests
- Wilcoxon signed-rank tests
- Friedman tests

---

##  Key Results

- CNN models show the highest robustness under FL and DP
- FedProx improves stability under client drift
- Differential privacy causes modest degradation for neural models and larger impact for TabNet

---

## Usage

Open any model notebook and run all cells:

