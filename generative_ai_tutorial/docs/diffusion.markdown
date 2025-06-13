# Diffusion Models

## Overview
Diffusion Models, popularized by Ho et al. in 2020, generate data by iteratively denoising random noise. They outperform GANs in image quality but are slow due to multiple denoising steps. They are used in tools like Stable Diffusion.

**Key Strengths**:
- High-quality outputs.
- Stable training.
- Supports conditional generation.

**Key Weaknesses**:
- Slow inference.
- High computational cost.
- Complex implementation.

**Seminal Papers**:
- Ho et al., 2020, "Denoising Diffusion Probabilistic Models" [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Rombach et al., 2022, "High-Resolution Image Synthesis with Latent Diffusion Models" [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Dhariwal & Nichol, 2021, "Cascaded Diffusion Models" [arXiv:2106.15282](https://arxiv.org/abs/2106.15282)

## Architecture
![Diffusion Model Architecture](../images/diffusion_architecture.png)
[Download Diffusion Model Architecture](../images/diffusion_architecture.png)

## Algorithm
```pseudocode
Algorithm: Diffusion_Model_Training
Input: Dataset D, noise steps T, learning rate lr, epochs E
Output: Trained denoising model U

1. Initialize denoising model U
2. For each epoch in E:
    a. For each batch in D:
        i. Sample image x from D
        ii. Sample t from [1, T]
        iii. Add noise to x: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        iv. Predict noise: epsilon_pred = U(x_t, t)
        v. Compute loss: mse_loss = mean((epsilon - epsilon_pred)^2)
        vi. Update U weights to minimize mse_loss
3. Return U
```

## Code
[Download Diffusion Model Code](../code/diffusion_mnist.py)

This implementation generates MNIST digits. Run it with:
```bash
python code/diffusion_mnist.py
```