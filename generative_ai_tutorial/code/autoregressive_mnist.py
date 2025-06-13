"""
Autoregressive Model for MNIST generation.
Implementation based on PixelRNN concepts from van den Oord et al., 2016.
This model uses LSTM cells to predict each pixel sequentially based on previously generated pixels.

Reference:
van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016).
"Pixel Recurrent Neural Networks" [arXiv:1601.06759]
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
pixel_values = 256  # 8-bit grayscale (0-255)
hidden_dim = 256
batch_size = 64
lr = 0.001
epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Autoregressive model
class Autoregressive(nn.Module):
    """
    LSTM-based autoregressive model that predicts each pixel sequentially.
    Similar to PixelRNN but using a simple LSTM for demonstration purposes.
    """
    def __init__(self):
        super(Autoregressive, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pixel_values)
    
    def forward(self, x):
        """
        Forward pass through the autoregressive model.
        For each position, uses all previous pixels to predict the next one.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
            
        Returns:
            Tensor of logits for each pixel position
        """
        batch_size, seq_len = x.size()
        outputs = []
        h = None
        for t in range(seq_len):
            input_t = x[:, t].unsqueeze(-1)
            out, h = self.lstm(input_t, h)
            out = self.fc(out)
            outputs.append(out)
        return torch.stack(outputs, dim=1)

# Initialize model and optimizer
model = Autoregressive().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
print("Starting training...")
loss_history = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, (images, _) in enumerate(dataloader):
        # Convert to long format for classification and flatten
        images = (images * 255).long().view(-1, image_size * image_size).to(device)
        
        optimizer.zero_grad()
        # We normalize inputs for the model but keep targets as integers
        outputs = model(images.float() / 255)
        
        # Cross-entropy loss for pixel-wise prediction
        loss = criterion(outputs.view(-1, pixel_values), images.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")
    
    # Generate samples
    if epoch % 2 == 0 or epoch == epochs - 1:
        print("Generating samples...")
        model.eval()
        with torch.no_grad():
            # Start with all zeros and generate pixel by pixel
            sample = torch.zeros(16, image_size * image_size).to(device)
            for t in range(image_size * image_size):
                # Get predictions for next pixel
                out = model(sample[:, :t+1].float() / 255)[:, t]
                probs = torch.softmax(out, dim=-1)
                # Sample from the predicted distribution
                sample[:, t] = torch.multinomial(probs, 1).squeeze()
            
            # Reshape and normalize for visualization
            samples = sample.view(-1, image_size, image_size).cpu() / 255
            plt.figure(figsize=(4, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(samples[j], cmap="gray")
                plt.axis("off")
            plt.savefig(f"autoregressive_samples_epoch_{epoch}.png")
            plt.close()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("autoregressive_training_loss.png")
plt.close()

print("Training finished!")

# Generate a final batch of samples
model.eval()
with torch.no_grad():
    # Unconditional generation - starting from zeros
    sample = torch.zeros(16, image_size * image_size).to(device)
    
    # Visualize the generation process for the first image
    progress_images = []
    progress_image = torch.zeros(image_size, image_size)
    progress_images.append(progress_image.clone())
    
    # Generate pixels sequentially (in raster scan order)
    for t in range(image_size * image_size):
        out = model(sample[:, :t+1].float() / 255)[:, t]
        probs = torch.softmax(out, dim=-1)
        next_pixel = torch.multinomial(probs, 1).squeeze()
        sample[:, t] = next_pixel
        
        # Record progress for the first image
        if t % (image_size * 4) == 0:
            progress_image = sample[0, :t+1].cpu().view(image_size, image_size) / 255
            progress_image = torch.cat([
                progress_image, 
                torch.zeros(image_size, image_size - (t+1) % image_size if (t+1) % image_size != 0 else 0)
            ], dim=1)
            progress_images.append(progress_image.clone())
    
    # Reshape and display final samples
    samples = sample.view(-1, image_size, image_size).cpu() / 255
    plt.figure(figsize=(4, 4))
    for j in range(16):
        plt.subplot(4, 4, j + 1)
        plt.imshow(samples[j], cmap="gray")
        plt.axis("off")
    plt.savefig("autoregressive_final_samples.png")
    plt.show()
    
    # Display generation progress
    plt.figure(figsize=(12, 4))
    for i, img in enumerate(progress_images):
        if i >= 8:  # Show at most 8 steps
            break
        plt.subplot(1, min(8, len(progress_images)), i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Step {i}")
        plt.axis("off")
    plt.savefig("autoregressive_generation_progress.png")
    plt.show()
    
print("Visualization complete. Check the saved PNG files for results.")