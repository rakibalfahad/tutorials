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
- Dhariwal & Nichol, 2021, "Diffusion Models Beat GANs on Image Synthesis" [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)

## Theoretical Background

Diffusion models belong to a class of generative models inspired by non-equilibrium thermodynamics. They consist of two processes: a forward diffusion process that gradually adds noise to data, and a reverse process that learns to denoise.

### Forward Diffusion Process

The forward process gradually transforms a data point $x_0$ into pure noise over $T$ timesteps by adding Gaussian noise according to a variance schedule $\beta_t$:

$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$

As Sohl-Dickstein et al. (2015) showed in their seminal work on diffusion models, "This process can be understood as gradually destroying structure in a data distribution through the successive application of Markov diffusion kernels" [1].

### Closed-Form Sampling

A key mathematical advantage of diffusion models is that the distribution of $x_t$ given $x_0$ can be computed in closed form:

$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i$.

### Reverse Process

The core idea is to learn the reverse process, which gradually denoises a pure noise sample back into a data sample. As Ho et al. (2020) explain, "The reverse process is a Markov chain that progressively removes noise, and our algorithm is designed to learn this reverse process" [2].

The reverse process is defined as:

$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

### Training Objective

The training objective for diffusion models is derived from variational inference. According to Ho et al. (2020), "We can rewrite the variational bound as a reweighted objective that resembles denoising score matching with multiple noise levels" [2].

This leads to the simplified objective:

$L_{simple} = \mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$

where $\epsilon$ is the noise added to create $x_t$, and $\epsilon_\theta$ is the model's prediction of this noise.

### Sampling Strategies

Several techniques have been developed to improve sampling from diffusion models:

- **DDIM** (Song et al., 2021): Enables non-Markovian sampling paths for faster generation [3].
- **Classifier Guidance** (Dhariwal & Nichol, 2021): Uses a classifier to guide the reverse process for higher quality samples [4].
- **Latent Diffusion** (Rombach et al., 2022): Performs diffusion in a compressed latent space to reduce computational cost [5].

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

## Generation Process

The generation process starts from random noise and iteratively denoises it:

```pseudocode
Algorithm: Diffusion_Model_Sampling
Input: Trained model U, noise steps T, batch size B
Output: Generated samples x_0

1. Sample x_T from standard normal distribution N(0, I) of size B
2. For t from T down to 1:
    a. If t > 1:
        i. Sample noise z from N(0, I)
    else:
        i. Set z = 0
    b. Predict noise: epsilon = U(x_t, t)
    c. Compute mean: mu = (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon) / sqrt(alpha_t)
    d. Update x_(t-1) = mu + sqrt(beta_t) * z
3. Return x_0
```

## Applications

Diffusion models have rapidly found applications in:

- **Text-to-Image Generation**: DALL-E 2 (Ramesh et al., 2022) and Stable Diffusion (Rombach et al., 2022) use diffusion models to generate images from text descriptions [6].
- **Image Editing**: Tools like InstructPix2Pix (Brooks et al., 2022) use diffusion models for semantic image editing [7].
- **3D Generation**: DreamFusion (Poole et al., 2022) uses 2D diffusion models to guide 3D synthesis [8].
- **Audio Synthesis**: AudioLM (Borsos et al., 2022) applies diffusion principles to generate high-quality audio [9].

## Code
[Download Diffusion Model Code](../code/diffusion_mnist.py)

This implementation generates MNIST digits. Run it with:
```bash
python code/diffusion_mnist.py
```

## Additional References

[1] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)

[2] Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models" [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[3] Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models" [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

[4] Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis" [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)

[5] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models" [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

[6] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents" [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)

[7] Brooks, T., Holynski, A., & Efros, A. A. (2022). "InstructPix2Pix: Learning to Follow Image Editing Instructions" [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)

[8] Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). "DreamFusion: Text-to-3D using 2D Diffusion" [arXiv:2209.14988](https://arxiv.org/abs/2209.14988)

[9] Borsos, Z., Marinier, R., Vincent, D., Kharitonov, E., Pietquin, O., Sharifi, M., ... & Tagliasacchi, M. (2022). "AudioLM: a Language Modeling Approach to Audio Generation" [arXiv:2209.03143](https://arxiv.org/abs/2209.03143)