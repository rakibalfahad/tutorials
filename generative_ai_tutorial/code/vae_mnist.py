"""
Variational Autoencoder (VAE) for MNIST generation.
Implementation based on Kingma & Welling, 2013, "Auto-Encoding Variational Bayes"

This model demonstrates the key components of VAEs:
1. Encoder: Maps inputs to a latent distribution (mean and log-variance)
2. Reparameterization: Samples from the latent distribution
3. Decoder: Reconstructs inputs from latent samples
4. Loss: Balances reconstruction quality and latent space regularization

References:
- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes" [arXiv:1312.6114]
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and
  Approximate Inference in Deep Generative Models"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 20    # Dimension of the latent space
hidden_dim = 400   # Dimension of the hidden layers
image_dim = 784    # Flattened image dimension (28x28)
batch_size = 128
lr = 0.001
epochs = 10
beta = 1.0         # Weight for KL divergence term (β-VAE when β ≠ 1)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# VAE model
class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST digits.
    
    Architecture:
    - Encoder: Maps input images to latent mean and log-variance
    - Reparameterization: Samples from the latent distribution
    - Decoder: Reconstructs images from latent samples
    """
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder network (q(z|x))
        self.encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Mean and log-variance projections
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder network (p(x|z))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Sigmoid()  # Output in range [0, 1] for MNIST
        )
    
    def encode(self, x):
        """
        Encode input images to latent distribution parameters.
        
        Args:
            x: Input images [batch_size, image_dim]
            
        Returns:
            Tuple of (mean, log-variance) of the latent distribution
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ ⊙ ε
        
        Samples from the latent distribution using the reparameterization trick
        to allow backpropagation through the sampling process.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
            
        Returns:
            Samples from the latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent samples to reconstructed images.
        
        Args:
            z: Latent samples [batch_size, latent_dim]
            
        Returns:
            Reconstructed images [batch_size, image_dim]
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Tuple of (reconstructed_images, mean, log-variance)
        """
        # Flatten input
        x_flat = x.view(-1, image_dim)
        
        # Encode
        mu, logvar = self.encode(x_flat)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar

# VAE loss function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function = reconstruction loss + β * KL divergence
    
    Args:
        recon_x: Reconstructed images [batch_size, image_dim]
        x: Original images [batch_size, channels, height, width]
        mu: Mean of the latent distribution
        logvar: Log-variance of the latent distribution
        beta: Weight for the KL divergence term (β-VAE)
        
    Returns:
        Total loss (scalar)
    """
    # Binary cross entropy for reconstruction loss
    # Sum over all pixels
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, image_dim), reduction='sum'
    )
    
    # KL divergence: -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with β weighting for KL term
    return BCE + beta * KLD, BCE, KLD

# Initialize model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
print("Starting training...")
train_losses = []
bce_losses = []
kld_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    bce_epoch = 0
    kld_epoch = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss, bce, kld = vae_loss(recon_batch, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track losses
        train_loss += loss.item()
        bce_epoch += bce.item()
        kld_epoch += kld.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item() / len(data):.4f} "
                  f"(BCE: {bce.item() / len(data):.4f}, KLD: {kld.item() / len(data):.4f})")
    
    # Average losses for the epoch
    avg_loss = train_loss / len(dataloader.dataset)
    avg_bce = bce_epoch / len(dataloader.dataset)
    avg_kld = kld_epoch / len(dataloader.dataset)
    
    train_losses.append(avg_loss)
    bce_losses.append(avg_bce)
    kld_losses.append(avg_kld)
    
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f} "
          f"(BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})")
    
    # Visualize samples and reconstructions
    if epoch % 2 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            # Generate random samples
            z = torch.randn(16, latent_dim).to(device)
            samples = model.decode(z).view(-1, 28, 28).cpu()
            
            # Get reconstructions of real images
            test_data = next(iter(dataloader))[0][:8].to(device)
            recon, _, _ = model(test_data)
            
            # Combine original and reconstructed images
            comparison = torch.cat([
                test_data.view(-1, 28, 28).cpu(),
                recon.view(-1, 28, 28).cpu()
            ])
            
            # Plot generated samples
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(samples[i], cmap="gray")
                plt.axis("off")
            plt.savefig(f"vae_samples_epoch_{epoch}.png")
            plt.close()
            
            # Plot reconstructions (original vs. reconstructed)
            plt.figure(figsize=(10, 4))
            for i in range(16):
                plt.subplot(2, 8, i + 1)
                plt.imshow(comparison[i], cmap="gray")
                if i < 8:
                    plt.title("Original")
                else:
                    plt.title("Reconstructed")
                plt.axis("off")
            plt.savefig(f"vae_reconstruction_epoch_{epoch}.png")
            plt.close()

# Plot training losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Total Loss')
plt.title("Total VAE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bce_losses, label='BCE')
plt.plot(kld_losses, label='KLD')
plt.title("BCE and KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("vae_training_loss.png")
plt.close()

print("Training finished!")

# Visualize latent space
model.eval()
with torch.no_grad():
    # Get a batch of test data and encode to latent space
    test_data, test_labels = next(iter(dataloader))
    test_data = test_data.to(device)
    mu, _ = model.encode(test_data.view(-1, image_dim))
    z = mu.cpu().numpy()
    labels = test_labels.numpy()
    
    # Plot 2D visualization of latent space (first 2 dimensions)
    plt.figure(figsize=(10, 8))
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title("Latent Space Visualization")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.savefig("vae_latent_space.png")
    plt.close()
    
    # Generate images from a grid in latent space (2D manifold)
    n = 15  # Grid size
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Linearly spaced coordinates for the 2D grid
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    # Generate images for each grid point
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.zeros(1, latent_dim).to(device)
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi
            
            # Set the rest of the latent dimensions to 0
            # This shows how the first two dimensions affect the generated images
            
            # Decode and reshape
            x_decoded = model.decode(z_sample)
            digit = x_decoded.view(28, 28).cpu().numpy()
            
            # Place in the grid
            figure[i * digit_size:(i + 1) * digit_size,
                   j * digit_size:(j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.title("Latent Space 2D Manifold")
    plt.savefig("vae_latent_manifold.png")
    plt.show()

print("Visualization complete. Check the saved PNG files for results.")