# Variational Autoencoders (VAEs)

## Overview
Variational Autoencoders (VAEs), introduced by Kingma and Welling in 2013, learn a probabilistic latent space for data generation. An encoder maps inputs to a latent distribution, and a decoder reconstructs or generates data. VAEs use the reparameterization trick for training, balancing reconstruction and regularization. They produce smooth outputs but may be blurry compared to GANs.

**Key Strengths**:
- Structured latent space for interpolation.
- Stable training.
- Suitable for denoising and synthetic data.

**Key Weaknesses**:
- Blurry outputs.
- Limited for complex distributions.
- Less realistic than GANs.

**Seminal Papers**:
- Kingma & Welling, 2013, "Auto-Encoding Variational Bayes" [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Higgins et al., 2017, "β-VAE: Learning Basic Visual Concepts" [arXiv:1606.05579](https://arxiv.org/abs/1606.05579)
- Rezende, Mohamed & Wierstra, 2014, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models" [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)

## Theoretical Background

VAEs are deep latent variable models based on variational inference principles, providing a principled way to learn complex probability distributions.

### Probabilistic Foundations

A VAE models the joint distribution $p(x, z)$ where $x$ is the observed data and $z$ is a latent variable. The generative process is:

$p(x, z) = p(z)p(x|z)$

where $p(z)$ is typically a standard normal prior $\mathcal{N}(0, I)$ and $p(x|z)$ is a complex distribution (often Gaussian) parameterized by a neural network (the decoder).

As Kingma and Welling (2013) explain, "The true posterior $p(z|x)$ is intractable due to the complex relationship between $x$ and $z$ through the decoder network" [1].

### Variational Inference

To address this intractability, VAEs use variational inference, approximating the posterior with a simpler distribution $q(z|x)$, typically a diagonal Gaussian parameterized by an encoder network:

$q(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)I)$

According to Rezende et al. (2014), "This approach allows us to derive a variational lower bound on the log-likelihood that can be optimized using stochastic gradient descent" [2].

### Evidence Lower Bound (ELBO)

The training objective for VAEs is the Evidence Lower Bound (ELBO):

$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$

where $\theta$ are the decoder parameters and $\phi$ are the encoder parameters.

Kingma and Welling note, "The first term is a reconstruction term that encourages the decoder to learn to reconstruct the data, while the second term is a regularization term that pushes the approximate posterior to be close to the prior" [1].

### Reparameterization Trick

A key innovation in VAEs is the reparameterization trick, which enables backpropagation through the sampling process:

$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

As Kingma and Welling explain, "This trick allows us to backpropagate through the sampling process by outsourcing the stochasticity to an independent noise variable $\epsilon$" [1].

### Disentangled Representations

Higgins et al. (2017) introduced β-VAE, which modifies the objective to encourage disentangled representations:

$\mathcal{L}(\theta, \phi; x, \beta) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z))$

The authors state, "By increasing the weight $\beta$ on the KL divergence term, we can learn latent representations where different dimensions correspond to different factors of variation in the data" [3].

## Architecture
![VAE Architecture](../images/vae_architecture.png)
[Download VAE Architecture](../images/vae_architecture.png)

## Algorithm
```pseudocode
Algorithm: Train_VAE
Input: Dataset D, latent dimension z_dim, learning rate lr, epochs E
Output: Trained VAE model

1. Initialize encoder E and decoder D with random weights
2. For each epoch in E:
    a. For each batch in D:
        i. Sample data x from D
        ii. Encode: (mu, logvar) = E(x)
        iii. Reparameterize: z = mu + epsilon * exp(0.5 * logvar), epsilon ~ N(0,1)
        iv. Decode: x_recon = D(z)
        v. Compute loss:
            L = BCE(x_recon, x) + KL(-0.5 * (1 + logvar - mu^2 - exp(logvar)))
        vi. Update E and D weights to minimize L
3. Return VAE
```

## Advanced Variants

Several important VAE variants have been developed:

1. **Conditional VAE (CVAE)** (Sohn et al., 2015): "Incorporates conditional information into both the encoder and decoder, allowing for controlled generation" [4].

2. **VQ-VAE** (van den Oord et al., 2017): "Uses vector quantization in the latent space to learn discrete representations, improving sample quality" [5].

3. **NVAE** (Vahdat & Kautz, 2020): "Employs a hierarchical architecture with normalizing flows to produce high-quality images" [6].

4. **VQGAN** (Esser et al., 2021): "Combines VQ-VAE with an adversarial loss for high-fidelity image synthesis" [7].

## Sampling and Generation

VAEs enable various generation approaches:

```pseudocode
Algorithm: Generate_Sample_VAE
Input: Trained decoder D, latent dimension z_dim
Output: Generated sample x

1. Sample z from N(0, I) of dimension z_dim
2. Generate sample: x = D(z)
3. Return x
```

Interpolation in latent space is straightforward:

```pseudocode
Algorithm: Interpolate_VAE
Input: Trained encoder E and decoder D, samples x1 and x2, steps n
Output: Interpolation sequence X

1. Encode: z1 = E(x1), z2 = E(x2)
2. Initialize empty sequence X
3. For i from 0 to n:
    a. Compute interpolation factor: t = i/n
    b. Interpolate: z = (1-t)*z1 + t*z2
    c. Generate: x = D(z)
    d. Add x to X
4. Return X
```

## Applications

VAEs have found numerous applications:

- **Image Generation**: Generating novel images with controllable attributes.
- **Anomaly Detection**: Identifying outliers by measuring reconstruction error.
- **Drug Discovery**: Exploring chemical space for new molecular structures (Gómez-Bombarelli et al., 2018) [8].
- **Text Generation**: Bowman et al. (2016) applied VAEs to natural language [9].
- **Representation Learning**: Learning meaningful features for downstream tasks.

## Code
[Download VAE Code](../code/vae_mnist.py)

This implementation generates MNIST digits. Run it with:
```bash
python code/vae_mnist.py
```

## Additional References

[1] Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes" [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

[2] Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models" [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)

[3] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" [ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl)

[4] Sohn, K., Lee, H., & Yan, X. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models" [arXiv:1511.06406](https://arxiv.org/abs/1511.06406)

[5] van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). "Neural Discrete Representation Learning" [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

[6] Vahdat, A., & Kautz, J. (2020). "NVAE: A Deep Hierarchical Variational Autoencoder" [arXiv:2007.03898](https://arxiv.org/abs/2007.03898)

[7] Esser, P., Rombach, R., & Ommer, B. (2021). "Taming Transformers for High-Resolution Image Synthesis" [arXiv:2012.09841](https://arxiv.org/abs/2012.09841)

[8] Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., ... & Aspuru-Guzik, A. (2018). "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" [ACS Central Science, 4(2), 268-276](https://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00572)

[9] Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2016). "Generating Sentences from a Continuous Space" [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)