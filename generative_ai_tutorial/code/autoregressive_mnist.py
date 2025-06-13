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
pixel_values = 256  # Binarized to 0-255
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
    def __init__(self):
        super(Autoregressive, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pixel_values)
    
    def forward(self, x):
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
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = (images * 255).long().view(-1, image_size * image_size).to(device)
        optimizer.zero_grad()
        outputs = model(images.float() / 255)
        loss = criterion(outputs.view(-1, pixel_values), images.view(-1))
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
            sample = torch.zeros(16, image_size * image_size).to(device)
            for t in range(image_size * image_size):
                out = model(sample[:, :t+1].float() / 255)[:, t]
                probs = torch.softmax(out, dim=-1)
                sample[:, t] = torch.multinomial(probs, 1).squeeze()
            samples = sample.view(-1, image_size, image_size).cpu() / 255
            plt.figure(figsize=(4, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow(samples[j], cmap="gray")
                plt.axis("off")
            plt.show()

print("Training finished!")