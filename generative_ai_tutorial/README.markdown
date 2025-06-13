# Comprehensive Tutorial on Generative AI

This repository contains a comprehensive tutorial on five key Generative AI (GenAI) modeling approaches: Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Transformers, Diffusion Models, and Autoregressive Models. Each model is explained with an overview, algorithm, PyTorch implementation (with GPU support), and a block diagram.

## Overview
Generative AI creates new content (e.g., images, text) by learning patterns from data. This tutorial covers:
- **Theory**: Detailed explanations and seminal papers.
- **Algorithms**: Pseudocode for training.
- **Code**: PyTorch implementations for MNIST (images) or simulated text.
- **Diagrams**: Visual architectures (downloadable PNGs).

## Prerequisites
- **Hardware**: GPU (NVIDIA CUDA-compatible) recommended.
- **Software**:
  - Python 3.8+
  - PyTorch 2.0+ (`pip install torch torchvision`)
  - Libraries: `matplotlib`, `numpy`, `transformers`, `diffusers` (`pip install matplotlib numpy transformers diffusers`)
- **Dataset**: MNIST for images (auto-downloaded). Simulated text for Transformers.
- **GPU Check**:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

## Chapters
1. [Generative Adversarial Networks (GANs)](docs/gans.markdown)
2. [Variational Autoencoders (VAEs)](docs/vaes.markdown)
3. [Transformers (Autoregressive for Text)](docs/transformers.markdown)
4. [Diffusion Models](docs/diffusion.markdown)
5. [Autoregressive Models (Image-Based)](docs/autoregressive.markdown)

## Repository Structure
- `docs/`: Markdown files for each chapter.
- `code/`: PyTorch implementations (`*.py`).
- `images/`: Architecture diagrams (`*.png`).

## Usage
- **Run Code**: `python code/gan_mnist.py`
- **View Diagrams**: Download from `images/` (linked in chapters).
- **Read Tutorial**: Start with this README, then explore chapters in `docs/`.

## Contributing
Feel free to submit issues or pull requests to improve the tutorial!