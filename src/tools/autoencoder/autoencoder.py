import torch

import torch.nn as nn
import torch.nn.functional as F

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, vocab_size):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits

    def decode(self, z, idx_to_char):
        logits = self.forward(z)
        probs = F.softmax(logits, dim=-1)
        _, indices = torch.max(probs, dim=-1)
        smiles = ''.join([idx_to_char[idx.item()] for idx in indices])
        return smiles

# Example usage:
# latent_dim = 128
# hidden_dim = 256
# vocab_size = len(idx_to_char)
# decoder = VAEDecoder(latent_dim, hidden_dim, vocab_size)
# z = torch.randn(1, latent_dim)
# smiles_string = decoder.decode(z, idx_to_char)
# print(smiles_string)