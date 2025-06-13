# Autoregressive Models (Image-Based)

## Overview
Autoregressive Models generate data sequentially, modeling conditional probabilities (e.g., pixel by pixel). Introduced by van den Oord et al. in 2016, they produce high-quality outputs but are slow due to sequential processing.

**Key Strengths**:
- Explicit probability modeling.
- High-quality details.
- Flexible for sequential data.

**Key Weaknesses**:
- Slow generation.
- High memory usage.
- Less competitive for images.

**Seminal Papers**:
- van den Oord et al., 2016, "Pixel Recurrent Neural Networks" [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)
- van den Oord et al., 2016, "Conditional Image Generation with PixelCNN" [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)
- Sutskever et al., 2011, "Autoregressive Models for Sequences" [arXiv:1106.6246](https://arxiv.org/abs/1106.6246)

## Architecture
![Autoregressive Model Architecture](../images/autoregressive_architecture.png)
[Download Autoregressive Model Architecture](../images/autoregressive_architecture.png)

## Algorithm
```pseudocode
Algorithm: Train_Autoregressive
Input: Dataset D, image size S, learning rate lr, epochs E
Output: Trained autoregressive model

1. Initialize model M with random weights
2. For each epoch in E:
    a. For each batch in D:
        i. Flatten images into sequences
        ii. Predict pixel probabilities: p(x_i | x_1, ..., x_{i-1})
        iii. Compute cross-entropy loss: L = -sum(log p(x_i))
        iv. Update M weights to minimize L
3. Return M
```

## Code
[Download Autoregressive Code](../code/autoregressive_mnist.py)

This implementation generates MNIST digits. Run it with:
```bash
python code/autoregressive_mnist.py
```