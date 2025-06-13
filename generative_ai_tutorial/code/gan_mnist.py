"""
Generative Adversarial Network (GAN) for MNIST generation.
Implementation based on Goodfellow et al., 2014, "Generative Adversarial Nets"

This model demonstrates the adversarial training process between:
1. Generator: Creates synthetic data from random noise
2. Discriminator: Distinguishes real from fake data

References:
- Goodfellow, I., et al. (2014). "Generative Adversarial Nets" [arXiv:1406.2661]
- Radford, A., et al. (2015). "Unsupervised Representation Learning with DCGANs"
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
latent_dim = 100    # Dimension of the noise vector
hidden_dim = 256    # Hidden dimension for the networks
image_dim = 784     # Flattened image dimension (28x28)
batch_size = 64
lr = 0.0002         # Learning rate (from DCGAN paper)
beta1 = 0.5         # Adam optimizer beta1 (from DCGAN paper)
epochs = 10

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    """
    Generator network that transforms random noise into synthetic images.
    
    Architecture:
    - Input: Random noise vector z
    - Output: Synthetic image in the same format as real data
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input layer: latent_dim -> hidden_dim
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            
            # Hidden layer: hidden_dim -> hidden_dim*2
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            
            # Output layer: hidden_dim*2 -> image_dim
            nn.Linear(hidden_dim * 2, image_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z: Random noise vector [batch_size, latent_dim]
            
        Returns:
            Generated images [batch_size, image_dim]
        """
        return self.model(z)

# Discriminator network
class Discriminator(nn.Module):
    """
    Discriminator network that classifies images as real or fake.
    
    Architecture:
    - Input: Image (real or generated)
    - Output: Probability that the image is real
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input layer: image_dim -> hidden_dim*2
            nn.Linear(image_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layer: hidden_dim*2 -> hidden_dim
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer: hidden_dim -> 1
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, img):
        """
        Forward pass through the discriminator.
        
        Args:
            img: Images (real or generated) [batch_size, image_dim]
            
        Returns:
            Probability that each image is real [batch_size, 1]
        """
        return self.model(img)

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Use Adam optimizer with parameters recommended in DCGAN paper
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Binary cross entropy loss
criterion = nn.BCELoss()

# Training loop
print("Starting training...")
d_losses, g_losses = [], []
for epoch in range(epochs):
    epoch_d_loss, epoch_g_loss = 0, 0
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()
        
        # Train on real images
        real_output = discriminator(real_imgs)
        d_real_loss = criterion(real_output, real_labels)
        
        # Train on fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        fake_output = discriminator(fake_imgs.detach())  # Detach to avoid training G on these labels
        d_fake_loss = criterion(fake_output, fake_labels)
        
        # Combine losses and update
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        epoch_d_loss += d_loss.item()
        
        # ---------------------
        # Train Generator
        # ---------------------
        g_optimizer.zero_grad()
        
        # Generator tries to fool the discriminator
        fake_output = discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)  # Generator wants D to think its outputs are real
        g_loss.backward()
        g_optimizer.step()
        epoch_g_loss += g_loss.item()
        
        # Log progress
        if i % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Record losses
    avg_d_loss = epoch_d_loss / len(dataloader)
    avg_g_loss = epoch_g_loss / len(dataloader)
    d_losses.append(avg_d_loss)
    g_losses.append(avg_g_loss)
    print(f"Epoch [{epoch}/{epochs}] Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
    
    # Generate and visualize samples
    if epoch % 2 == 0 or epoch == epochs - 1:
        print("Generating samples...")
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            fake_imgs = generator(z).view(-1, 28, 28).cpu()
            
            # Convert from [-1, 1] to [0, 1] for visualization
            fake_imgs = (fake_imgs + 1) / 2
            
            plt.figure(figsize=(4, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(fake_imgs[j], cmap="gray")
                plt.axis("off")
            plt.savefig(f"gan_samples_epoch_{epoch}.png")
            plt.close()

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.title("GAN Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_training_loss.png")
plt.close()

print("Training finished!")

# Generate final samples and visualize latent space interpolation
with torch.no_grad():
    # Generate random samples
    z = torch.randn(16, latent_dim).to(device)
    fake_imgs = generator(z).view(-1, 28, 28).cpu()
    fake_imgs = (fake_imgs + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    plt.figure(figsize=(4, 4))
    for j in range(16):
        plt.subplot(4, 4, j + 1)
        plt.imshow(fake_imgs[j], cmap="gray")
        plt.axis("off")
    plt.savefig("gan_final_samples.png")
    plt.show()
    
    # Latent space interpolation
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    n_steps = 10
    interp_imgs = []
    for i in range(n_steps + 1):
        alpha = i / n_steps
        z_interp = z1 * (1 - alpha) + z2 * alpha
        img = generator(z_interp).view(28, 28).cpu()
        img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        interp_imgs.append(img)
    
    plt.figure(figsize=(15, 2))
    for i, img in enumerate(interp_imgs):
        plt.subplot(1, n_steps + 1, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.savefig("gan_latent_interpolation.png")
    plt.show()

print("Visualization complete. Check the saved PNG files for results.")