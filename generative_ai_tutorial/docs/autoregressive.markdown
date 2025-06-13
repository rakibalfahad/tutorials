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

## Theoretical Background

Autoregressive models decompose the joint distribution of the data into a product of conditional distributions, following the chain rule of probability:

$p(x) = \prod_{i=1}^{n} p(x_i | x_1, x_2, ..., x_{i-1})$

where $x_i$ represents the $i$-th element of the data (e.g., a pixel in an image). 

### PixelRNN and PixelCNN

For image generation, two main autoregressive architectures emerged:

1. **PixelRNN**: Uses LSTM cells with 2D recurrent connections to model the dependency of a pixel on previous pixels. As Chen et al. (2018) note, "PixelRNNs effectively capture long-range dependencies but are computationally intensive due to sequential processing" [1].

2. **PixelCNN**: Uses masked convolutions to ensure each pixel depends only on previously generated pixels. According to Salimans et al. (2017), "PixelCNNs offer faster parallel training than RNNs while maintaining high sample quality" [2].

### Masked Convolutions

The key innovation in PixelCNN is the use of masked convolutions, which van den Oord et al. explain as: "We mask the convolution kernel so that a pixel only depends on other pixels above and to the left of it" [3]. This creates an autoregressive constraint while allowing for efficient parallelization during training.

### Improvements and Variants

Numerous improvements have been proposed:

- **Gated PixelCNN** (van den Oord et al., 2016): Incorporates gating mechanisms similar to those in LSTMs to improve expressiveness.
- **PixelCNN++** (Salimans et al., 2017): Incorporates a discretized mixture of logistics to model the pixel distributions, significantly improving quality.
- **PixelSNAIL** (Chen et al., 2018): Combines the parallelism of PixelCNN with the expressivity of RNNs using self-attention.

### Mathematical Formulation

For a typical PixelCNN model, the loss function is the negative log-likelihood:

$L = -\sum_{i} \log p(x_i | x_{<i}; \theta)$

where $\theta$ represents the model parameters and $x_{<i}$ represents all pixels that come before pixel $i$ in the raster scan ordering.

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

## Generation Process

The generation process for autoregressive models is inherently sequential:

```pseudocode
Algorithm: Generate_Image_Autoregressive
Input: Trained model M, image size S
Output: Generated image G

1. Initialize empty image G of size S
2. For each pixel position (i,j) in raster scan order:
    a. Use model to predict probability distribution: p = M(G_current)
    b. Sample next pixel value from p
    c. Place sampled value at position (i,j) in G
3. Return completed image G
```

## Applications

Autoregressive models have found applications beyond image generation:

- **Audio synthesis**: WaveNet (van den Oord et al., 2016) revolutionized audio generation using autoregressive modeling [4].
- **Video prediction**: Predicting future frames given past frames (Kalchbrenner et al., 2017) [5].
- **Compression**: Bits Back with ANS (Townsend et al., 2019) leverages autoregressive models for neural compression [6].

## Code
[Download Autoregressive Code](../code/autoregressive_mnist.py)

This implementation generates MNIST digits. Run it with:
```bash
python code/autoregressive_mnist.py
```

## Additional References

[1] Chen, X., Mishra, N., Rohaninejad, M., & Abbeel, P. (2018). "PixelSNAIL: An Improved Autoregressive Generative Model" [arXiv:1712.09763](https://arxiv.org/abs/1712.09763)

[2] Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications" [arXiv:1701.05517](https://arxiv.org/abs/1701.05517)

[3] van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). "Pixel Recurrent Neural Networks" [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)

[4] van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). "WaveNet: A Generative Model for Raw Audio" [arXiv:1609.03499](https://arxiv.org/abs/1609.03499)

[5] Kalchbrenner, N., van den Oord, A., Simonyan, K., Danihelka, I., Vinyals, O., Graves, A., & Kavukcuoglu, K. (2017). "Video Pixel Networks" [arXiv:1610.00527](https://arxiv.org/abs/1610.00527)

[6] Townsend, J., Bird, T., & Barber, D. (2019). "Practical lossless compression with latent variables using bits back coding" [arXiv:1901.04866](https://arxiv.org/abs/1901.04866)