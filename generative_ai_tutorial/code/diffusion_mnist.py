import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set random seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 28
channels = 1
batch_size = 128
lr = 0.0002
epochs = 10
timesteps = 1000
beta_start = 0.0001
beta_end = 0.02

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Noise schedule
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# Simple U-Net
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.down1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.time_emb = nn.Linear(1, 128)
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x, t):
        t_emb = self.time_emb(t.view(-1, 1)).view(-1, 128, 1, 1)
        h1 = self.relu(self.down1(x))
        h2 = self.relu(self.down2(h1)) + t_emb
        h3 = self.relu(self.up1(self.upsample(h2)))
        return self.up2(h3)

# Noise addition
def add_noise(x, t):
    noise = torch.randn_like(x).to(device)
    return (sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x +
            sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise), noise

# Initialize model and optimizer
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        t = torch.randint(0, timesteps, (images.size(0),)).to(device)
        noisy_images, noise = add_noise(images, t)
        
        optimizer.zero_grad()
        predicted_noise = model(noisy_images, t.float() / timesteps)
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {epoch_loss / len(dataloader):.4f}")
    
    # Generate samples
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            x = torch.randn(16, channels, image_size, image_size).to(device)
            for t in reversed(range(timesteps)):
                t_tensor = torch.full((16,), t, device=device).float() / timesteps
                predicted_noise = model(x, t_tensor)
                x = (1 / torch.sqrt(alphas[t])) * (
                    x - (betas[t] / sqrt_one_minus_alphas_cumprod[t]) * predicted_noise
                ) + torch.sqrt(betas[t]) * torch.randn_like(x)
            samples = x.clamp(0, 1).cpu()
            plt.figure(figsize=(4, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(samples[j][0], cmap="gray")
                plt.axis("off")
            plt.show()

print("Training finished!")