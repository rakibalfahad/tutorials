"""
Diffusion Model for MNIST generation.
Implementation based on Ho et al., 2020, "Denoising Diffusion Probabilistic Models"

This model demonstrates the core principles of diffusion models:
1. Forward process: Gradually adding noise to data
2. Reverse process: Learning to denoise and recover the original data

References:
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models" [arXiv:2006.11239]
- Sohl-Dickstein, J., et al. (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
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
image_size = 28
channels = 1
batch_size = 128
lr = 0.0002
epochs = 10
timesteps = 1000  # Number of diffusion steps
beta_start = 0.0001  # Initial noise level
beta_end = 0.02  # Final noise level

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Precompute the diffusion noise schedule and sampling parameters
# These values determine how noise is added during the forward process
# and how denoising works during the reverse process
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# Simple U-Net model for noise prediction
class SimpleUNet(nn.Module):
    """
    A simplified U-Net architecture for predicting noise in diffusion models.
    
    This model takes a noisy image and timestep as input and predicts the noise
    that was added to the original image.
    """
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Downsampling path
        self.down1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        
        # Time embedding (maps diffusion timestep to feature space)
        self.time_emb = nn.Linear(1, 128)
        
        # Upsampling path
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, channels, 3, padding=1)
        
        # Activation and upsampling
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x, t):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Noisy images [batch_size, channels, height, width]
            t: Timesteps [batch_size]
            
        Returns:
            Predicted noise at the given timestep
        """
        # Time embedding
        t_emb = self.time_emb(t.view(-1, 1)).view(-1, 128, 1, 1)
        
        # Downsampling
        h1 = self.relu(self.down1(x))
        h2 = self.relu(self.down2(h1)) + t_emb  # Add time embedding
        
        # Upsampling
        h3 = self.relu(self.up1(self.upsample(h2)))
        return self.up2(h3)

# Function to add noise to images at a specified timestep
def add_noise(x, t):
    """
    Add noise to images according to the diffusion schedule.
    
    This implements the forward diffusion process q(x_t | x_0).
    
    Args:
        x: Original images [batch_size, channels, height, width]
        t: Timesteps [batch_size]
        
    Returns:
        Tuple of (noisy_images, noise)
    """
    noise = torch.randn_like(x).to(device)
    noisy_images = (
        sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x + 
        sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
    )
    return noisy_images, noise

# Initialize model and optimizer
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
print("Starting training...")
loss_history = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        
        # Sample random timesteps for each image
        t = torch.randint(0, timesteps, (images.size(0),)).to(device)
        
        # Add noise to the images
        noisy_images, noise = add_noise(images, t)
        
        # Predict the noise
        optimizer.zero_grad()
        predicted_noise = model(noisy_images, t.float() / timesteps)
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")
    
    # Generate samples using the trained model
    if epoch % 2 == 0 or epoch == epochs - 1:
        print("Generating samples...")
        model.eval()
        with torch.no_grad():
            # Start from random noise (x_T)
            x = torch.randn(16, channels, image_size, image_size).to(device)
            
            # Storage for visualization of the reverse process
            if epoch == epochs - 1:
                denoising_process = [x.cpu().clone()]
                timesteps_to_save = [800, 600, 400, 200, 0]
            
            # Reverse diffusion process (iterative denoising)
            for t in reversed(range(timesteps)):
                # Normalized timestep
                t_tensor = torch.full((16,), t, device=device).float() / timesteps
                
                # Predict noise
                predicted_noise = model(x, t_tensor)
                
                # If last timestep, no noise is added
                if t > 0:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                
                # Denoising step formula
                x = (1 / torch.sqrt(alphas[t])) * (
                    x - (betas[t] / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise
                ) + torch.sqrt(betas[t]) * z
                
                # Save intermediate steps for visualization
                if epoch == epochs - 1 and t in timesteps_to_save:
                    denoising_process.append(x.cpu().clone())
            
            # Clamp values to valid image range
            samples = x.clamp(0, 1).cpu()
            
            # Plot generated samples
            plt.figure(figsize=(4, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(samples[j][0], cmap="gray")
                plt.axis("off")
            plt.savefig(f"diffusion_samples_epoch_{epoch}.png")
            plt.close()
            
            # Visualize the denoising process for the final epoch
            if epoch == epochs - 1:
                plt.figure(figsize=(15, 4))
                for i, img in enumerate(denoising_process):
                    plt.subplot(1, len(denoising_process), i + 1)
                    plt.imshow(img[0][0], cmap="gray")
                    if i == 0:
                        plt.title("Pure noise")
                    elif i == len(denoising_process) - 1:
                        plt.title("Final")
                    else:
                        plt.title(f"Step {timesteps_to_save[i-1]}")
                    plt.axis("off")
                plt.savefig("diffusion_process.png")
                plt.close()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("diffusion_training_loss.png")
plt.close()

print("Training finished!")

# Generate a final batch of samples
model.eval()
with torch.no_grad():
    # Start from random noise
    x = torch.randn(16, channels, image_size, image_size).to(device)
    
    # Reverse diffusion process (iterative denoising)
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((16,), t, device=device).float() / timesteps
        predicted_noise = model(x, t_tensor)
        
        # Apply denoising formula
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alphas[t])) * (
            x - (betas[t] / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise
        ) + torch.sqrt(betas[t]) * z
    
    # Clamp and display
    samples = x.clamp(0, 1).cpu()
    plt.figure(figsize=(4, 4))
    for j in range(16):
        plt.subplot(4, 4, j + 1)
        plt.imshow(samples[j][0], cmap="gray")
        plt.axis("off")
    plt.savefig("diffusion_final_samples.png")
    plt.show()

print("Visualization complete. Check the saved PNG files for results.")