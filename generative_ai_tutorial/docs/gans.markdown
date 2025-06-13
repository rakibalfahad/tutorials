# Generative Adversarial Networks (GANs)

## Overview
Generative Adversarial Networks (GANs), introduced by Ian Goodfellow et al. in 2014, consist of a **generator** that creates synthetic data from noise and a **discriminator** that distinguishes real from fake data. They train adversarially, converging when generated data mimics the real distribution. GANs excel in producing sharp images (e.g., faces) but face challenges like training instability and mode collapse.

**Key Strengths**:
- Sharp, realistic outputs.
- Flexible for images, audio, etc.
- Enables style transfer and data augmentation.

**Key Weaknesses**:
- Unstable training.
- Mode collapse.
- High computational cost.

**Seminal Papers**:
- Goodfellow et al., 2014, "Generative Adversarial Nets" [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Radford et al., 2015, "Unsupervised Representation Learning with DCGANs" [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Karras et al., 2019, "A Style-Based Generator Architecture for GANs" [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)

## Architecture
![GAN Architecture](../images/gan_architecture.png)
[Download GAN Architecture](../images/gan_architecture.png)

## Algorithm
```pseudocode
Algorithm: Train_GAN
Input: Real dataset D, noise dimension z_dim, learning rate lr, epochs E
Output: Trained generator G and discriminator D

1. Initialize G and D with random weights
2. For each epoch in E:
    a. For each batch in D:
        i. Sample real data x from D
        ii. Sample noise z from N(0,1)
        iii. Generate fake data: x_fake = G(z)
        iv. Compute discriminator loss:
            L_D = -mean(log(D(x))) - mean(log(1 - D(x_fake)))
        v. Update D weights to minimize L_D
        vi. Compute generator loss:
            L_G = -mean(log(D(x_fake)))
        vii. Update G weights to minimize L_G
3. Return G, D
```

## Code
[Download GAN Code](../code/gan_mnist.py)

This implementation uses PyTorch to generate MNIST digits. Run it with:
```bash
python code/gan_mnist.py
```