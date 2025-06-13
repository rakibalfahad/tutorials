"""
Transformer Model for text generation.
Implementation based on Vaswani et al., 2017, "Attention is All You Need"

This model demonstrates the core principles of transformer-based autoregressive text generation:
1. Self-attention for modeling token relationships
2. Autoregressive decoding for text generation

References:
- Vaswani, A., et al. (2017). "Attention is All You Need" [arXiv:1706.03762]
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math

# Set random seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
vocab_size = 100       # Simplified vocabulary for demo
seq_length = 20        # Length of sequences
embed_dim = 128        # Embedding dimension
num_heads = 4          # Number of attention heads
num_layers = 2         # Number of transformer layers
ff_dim = embed_dim * 4 # Feed-forward hidden dimension
dropout_rate = 0.1     # Dropout rate
batch_size = 64
lr = 0.001
epochs = 10

# Simple dataset for simulated text
class TextDataset(Dataset):
    """
    Synthetic text dataset for demonstration purposes.
    
    Creates random sequences of integers representing tokens.
    Each item returns an input sequence and the target (next token) sequence.
    """
    def __init__(self, length=1000, seed=42):
        np.random.seed(seed)
        # Create sequences with some pattern (e.g., token i often followed by token i+1)
        base = np.random.randint(0, vocab_size // 2, (length, seq_length // 2))
        derived = (base + 1) % vocab_size
        # Interleave to create some predictable patterns
        data = np.zeros((length, seq_length + 1), dtype=np.int64)
        for i in range(length):
            for j in range(seq_length // 2):
                data[i, j*2] = base[i, j]
                data[i, j*2+1] = derived[i, j] if np.random.random() > 0.3 else np.random.randint(0, vocab_size)
            # Add one extra token for the final prediction
            data[i, -1] = (data[i, -2] + 1) % vocab_size if np.random.random() > 0.3 else np.random.randint(0, vocab_size)
        
        self.data = torch.from_numpy(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # Input and target (next token prediction)

dataset = TextDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Positional Encoding
class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from the original Transformer paper.
    
    This adds positional information to the token embeddings since transformers
    don't have any inherent notion of position.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register buffer makes this tensor part of the module state but not a parameter
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0), :]

# Transformer model with decoder only (like GPT)
class TransformerDecoder(nn.Module):
    """
    Autoregressive Transformer for text generation.
    
    Uses a decoder-only architecture similar to GPT, where each token 
    can only attend to previous tokens in the sequence.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Output projection to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.nhead = nhead
        
    def forward(self, src):
        """
        Forward pass through the transformer.
        
        Args:
            src: Input sequence [batch_size, seq_len]
            
        Returns:
            Output logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        # Create causal mask to ensure autoregressive property
        seq_len = src.size(1)
        src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Embed tokens and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)  # Scale by sqrt(d_model)
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        
        # Pass through transformer layers
        output = self.transformer_decoder(src, memory=src, tgt_mask=src_mask)
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Project to vocabulary
        return self.output_layer(output)
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a causal attention mask.
        
        Each position can only attend to previous positions in the sequence.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, start_tokens, max_length):
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Initial tokens to condition on [batch_size, prefix_len]
            max_length: Maximum length of generated sequence
            
        Returns:
            Generated token sequence [batch_size, max_length]
        """
        self.eval()
        with torch.no_grad():
            batch_size = start_tokens.size(0)
            cur_seq = start_tokens
            
            for _ in range(max_length - start_tokens.size(1)):
                # Get predictions
                logits = self(cur_seq)
                
                # Take the last token prediction for each sequence
                next_token_logits = logits[:, -1, :]
                
                # Sample from the distribution (or take argmax)
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                cur_seq = torch.cat([cur_seq, next_token], dim=1)
            
            return cur_seq

# Initialize model and optimizer
model = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=embed_dim,
    nhead=num_heads,
    num_layers=num_layers,
    dim_feedforward=ff_dim,
    dropout=dropout_rate
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
print("Starting training...")
losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        
        # Backward pass
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
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("transformer_training_loss.png")
plt.close()

# Generate sample sequences
model.eval()
with torch.no_grad():
    # Start with random prefix tokens
    prefix_length = 5
    num_samples = 5
    
    # Generate multiple sequences
    for sample_idx in range(num_samples):
        input_seq = torch.randint(0, vocab_size, (1, prefix_length)).to(device)
        generated = model.generate(input_seq, max_length=seq_length)
        
        print(f"\nSample {sample_idx+1}:")
        print(f"Prefix:     {input_seq[0][:prefix_length].tolist()}")
        print(f"Completion: {generated[0][prefix_length:].tolist()}")
        print(f"Full sequence: {generated[0].tolist()}")
    
    # Demonstrate how the model uses context
    print("\nDemonstrating context awareness:")
    test_seq = dataset.data[0, :prefix_length].unsqueeze(0).to(device)
    print(f"Context: {test_seq[0].tolist()}")
    
    # Generate multiple continuations to show sampling variation
    for i in range(3):
        completion = model.generate(test_seq, max_length=seq_length)
        print(f"Completion {i+1}: {completion[0][prefix_length:].tolist()}")

# Visualize attention weights for a sample sequence
def visualize_attention(model, input_seq):
    """
    Visualize attention weights from the model.
    
    This requires adding hooks to extract attention weights.
    """
    # This is a placeholder for attention visualization
    # In a full implementation, you would need to add hooks to the model
    # to extract attention weights during forward pass
    
    # For simplicity, we'll just print a message
    print("\nAttention visualization would require modifying the model with hooks.")
    print("This would typically show how different tokens attend to each other.")

# Simulate attention visualization
sample_input = torch.randint(0, vocab_size, (1, seq_length)).to(device)
visualize_attention(model, sample_input)

print("\nTraining finished!")
print("Check the saved PNG files for training loss visualization.")