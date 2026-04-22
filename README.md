# Self-Pruning Neural Network

## Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary connections during training. Instead of manually pruning weights, the model uses a **gating mechanism** to identify and suppress unimportant weights automatically.

The goal is to demonstrate that neural networks are **over-parameterized** and can be significantly compressed without major loss in accuracy.

---

## Key Idea

Each weight in the network is multiplied by a learnable gate:

```
effective_weight = weight × sigmoid(gate_score)
```

* Gate ≈ 1 → connection is important
* Gate ≈ 0 → connection is pruned

This allows the network to **learn which weights to keep and which to remove**.

---

## Loss Function

The training objective combines classification performance with sparsity:

```
Total Loss = CrossEntropy + λ × SparsityLoss
```

* **CrossEntropy Loss** → ensures prediction accuracy
* **Sparsity Loss (L1 on gates)** → encourages many gates to go to 0

This leads to a sparse network where only important connections remain.

---

## Model Architecture

* Feedforward Neural Network (MLP)
* Custom `PrunableLinear` layers
* ReLU activations
* Dataset: **CIFAR-10**

---

## Training Details

* Optimizer: Adam
* Separate learning rates:

  * Weights: `1e-3`
  * Gates: `2e-2` (higher for effective pruning)
* Epochs: 50
* Batch size: 128

---

## Results

| Lambda (λ) | Accuracy (%) | Accuracy Drop | Sparsity (%) |
| ---------- | ------------ | ------------- | ------------ |
| 0.0        | 55.61        | +0.00         | 64.3         |
| 0.01       | 54.83        | -0.78         | 68.8         |
| 0.05       | **55.58**    | **-0.03**     | **72.8**     |
| 0.1        | 55.34        | -0.27         | 75.4         |

---

## Best Configuration

We select:

```
λ = 0.05
```

### Why?

* Maintains **almost identical accuracy** to baseline
* Achieves **~73% pruning**
* Provides the best **balance between performance and compression**

---

## Key Observations

* The model can remove **70–75% of weights** with minimal accuracy loss
* Even without explicit pruning (λ = 0), some sparsity emerges naturally
* Increasing λ increases sparsity but may slightly reduce accuracy

---

## Gate Distribution

The learned gate values show a **bimodal distribution**:

* Large spike near **0** → pruned weights
* Cluster near **1** → important weights

This confirms that the model successfully learns to separate useful and redundant connections.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training

```bash
python code.py
```

---

## Outputs

The script generates:

* Accuracy vs Epoch plots
* Sparsity vs Epoch plots
* Loss breakdown
* Gate distribution histogram
* Lambda vs Accuracy/Sparsity plots

---

## Conclusion

This project demonstrates that:

* Neural networks are highly over-parameterized
* A self-pruning mechanism can effectively remove redundant weights
* Significant compression (~70%+) is possible without degrading performance

---

## Future Work

* Structured pruning (neurons/channels)
* Applying pruning to CNNs (e.g., ResNet)
* Deployment for efficient inference

---

## Final Statement

> A large portion of neural network parameters are redundant, and self-pruning provides an effective way to reduce model complexity while preserving performance.

---
