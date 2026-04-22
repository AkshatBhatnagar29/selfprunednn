# selfprunednn
# Self Pruning Neural Network. Final Report

## 1.

Neural networks are a part of deep learning systems these days.. They often have too many parameters, which means they use up too much computer power and memory. We can use a technique called pruning to get rid of the weights and make the models work better.

In this project we made a *self pruning neural network** that can get rid of its own extra connections while it is being trained. It does this using a mechanism called a gate.

## 2. Methodology

### 2.1 Prunable Linear Layer

of using the usual linear layer we made a custom **Prunable Linear** layer. Each weight has a parameter called a **gate score** that it can learn. When the network is working:

* It calculates the gate values using the *gate score**.

* It then uses these gate values to decide which weights are important.

* If a gate value is close to 0 the connection is basically removed.

This way the model can **figure out which connections are important**.

### 2.2 Sparsity Regularization

To help the network get rid of weights we changed the way it calculates its loss. We added a part to the loss function:

```

Total Loss = CrossEntropyLoss + λ × SparsityLoss

```

**SparsityLoss** is the average of all the gate values. Since gate values are between 0 and 1 making this loss smaller makes many gate values go to 0.

### Why L1 Encourages Sparsity

The L1 norm punishes all values equally. It makes many of them go to exactly 0. This makes the network **sparse** meaning only the important connections are left.

### 2.3 Training Setup

* We used the CIFAR-10 dataset.

* Our model was a neural network with **prunable layers**.

* We used the Adam optimizer.

* We used learning rates for the weights and the gates.

We trained the model with λ values to see how it affects the trade-off between accuracy and sparsity.

## 3. Results

### 3.1 Summary Table

| Lambda (λ) | Test Accuracy (%) | Accuracy Drop | Sparsity @ 0.01 (%) |

| ---------- | ----------------- | ------------- | ------------------- |

0.0        | 55.61             | +0.00         | 64.3                |

| 0.01       | 54.83             | -0.78         | 68.8                |

| 0.05       | **55.58**         | -0.03          72.8                |

| 0.1        | 55.34             | -0.27         | **75.4**            |

### 3.2 Observations

* With **λ = 0** the model is already pretty **sparse** (~64%).

* When we increase λ the model gets more **sparse**.

* The model does not lose accuracy even when it is very **sparse**.

* At **λ = 0.05** the model is almost as accurate as before. It has removed ~73% of its weights.

### 3.3 Best Trade-off

The best balance between accuracy and sparsity is when:

```

λ = 0.05

```

* Accuracy: 55.58%

* Sparsity: 72.8%

This shows that a big part of the network is not needed.

## 4. Analysis

### 4.1 Effect of λ (Lambda)

* Low λ: The network does not prune much. It is very accurate.

* Medium λ: The network prunes some so it is a balance between accuracy and sparsity.

* High λ: The network prunes a lot so it is very sparse. Not as accurate.

λ controls how much the network prunes.

### 4.2 Role of Threshold

After training we use a threshold (like 0.01) to decide which weights to keep:

* If the gate value is less than the threshold the weight is removed.

* If the gate value is greater than or equal to the threshold the weight is kept.

This makes the gate values into a *binary decision**.

### 4.3 Pareto Trade-off

The model shows a strong trade-off between accuracy and sparsity:

* We can remove up to **75% of the weights**.

*. The model is still very accurate.

This shows that the network has a lot of weights.

## 5. Visualization Insights

### Accuracy vs Lambda

* The accuracy stays about the same for λ values.

* This means the model is very robust.

### Sparsity vs Lambda

* The sparsity increases as λ increases.

* This shows that the L1 regularization is working well.

### Gate Distribution

* Many gate values are close to 0.

* Some gate values are close to 1.

This shows that the model can clearly tell which connections are important.

## 6.

This project shows that a **self pruning neural network** can learn to remove its extra connections during training.

The main points are:

* The gate mechanism lets the network prune itself.

* L1 regularization makes the network sparse.

* We can remove a lot of weights (~70-75%) without losing accuracy.

* The model can find its optimal sparse structure.

## 7. Future Work

* We can try pruning neurons or channels.

* We can apply pruning to layers.

* We can use models, for real-time inference.

## 8. Final Statement

This experiment shows that:

> Neural networks have a lot of parameters and a self-pruning mechanism can make them smaller and more efficient without losing much accuracy.
