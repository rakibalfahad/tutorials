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
latent_dim = 100
hidden_dim = 256
image_dim = 784  # 28x28
batch_size = 64
lr = 0.0002
epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, image_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        real_output = discriminator(real_imgs)
        d_real_loss = criterion(real_output, real_labels)
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        fake_output = discriminator(fake_imgs.detach())
        d_fake_loss = criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        if i % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Visualize samples
    if epoch % 2 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            fake_imgs = generator(z).view(-1, 28, 28).cpu()
            plt.figure(figsize=(4, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(fake_imgs[j], cmap="gray")
                plt.axis("off")
            plt.show()

print("Training finished!")