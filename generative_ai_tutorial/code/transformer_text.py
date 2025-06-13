import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set random seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 100  # Simplified vocab for demo
seq_length = 20
embed_dim = 128
num_heads = 2
num_layers = 2
batch_size = 64
lr = 0.001
epochs = 10

# Simple dataset (simulated text)
class TextDataset(Dataset):
    def __init__(self, length=1000):
        self.data = torch.randint(0, vocab_size, (length, seq_length + 1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # Input and target (next token)

dataset = TextDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(device)
        out = self.transformer(x, memory=x, tgt_mask=mask)
        out = out.transpose(0, 1)
        return self.fc(out)

# Initialize model and optimizer
model = SimpleTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Generate sample
model.eval()
with torch.no_grad():
    input_seq = torch.randint(0, vocab_size, (1, seq_length)).to(device)
    output = model(input_seq)
    predicted_tokens = torch.argmax(output, dim=-1)
    print(f"Sample input: {input_seq[0].tolist()}")
    print(f"Sample output: {predicted_tokens[0].tolist()}")

print("Training finished!")