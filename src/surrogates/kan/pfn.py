import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Define Synthetic Data Generator (Prior)
class PriorDataset(Dataset):
    """Generates synthetic tasks from a prior over functions (e.g., linear functions)."""
    def __init__(self, num_tasks=1000, num_samples_per_task=10, input_dim=1):
        self.num_tasks = num_tasks
        self.num_samples_per_task = num_samples_per_task
        self.input_dim = input_dim
        
        # Prior over function parameters (e.g., slope and intercept for linear functions)
        self.prior_slope = torch.distributions.Normal(0, 1)
        self.prior_intercept = torch.distributions.Normal(0, 1)
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        # Sample a function from the prior
        slope = self.prior_slope.sample()
        intercept = self.prior_intercept.sample()
        
        # Generate synthetic data for this task
        x = torch.rand(self.num_samples_per_task, self.input_dim) * 10 - 5  # x ∈ [-5, 5]
        y = slope * x + intercept + torch.randn_like(x) * 0.1  # y = slope*x + intercept + noise
        
        return x, y

# 2. Define PFN Architecture
class PriorFittedNetwork(nn.Module):
    """Processes context (prior data) and query points to predict outputs."""
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Encodes (x_i, y_i) pairs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),  # Combines encoded context + query x
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)  # Aggregates context
        
    def forward(self, context_x, context_y, query_x):
        # Encode context: (x_i, y_i) → embeddings
        context = torch.cat([context_x, context_y], dim=-1)  # Shape: [batch, num_context, input_dim+1]
        batch_size, num_context, _ = context.shape
        context_emb = self.encoder(context.view(-1, context.shape[-1])) # [batch*num_context, hidden_dim]
        context_emb = context_emb.view(batch_size, num_context, -1)  # [batch, num_context, hidden_dim]
        
        # Attention aggregation over context
        context_emb = context_emb.permute(1, 0, 2)  # [num_context, batch, hidden_dim]
        query_emb = self.encoder(torch.cat([query_x, torch.zeros_like(query_x)], dim=-1))  # Dummy y=0
        query_emb = query_emb.unsqueeze(0)  # [1, batch, hidden_dim]
        attn_output, _ = self.attention(query_emb, context_emb, context_emb)  # [1, batch, hidden_dim]
        aggregated = attn_output.squeeze(0)  # [batch, hidden_dim]
        
        # Decode: aggregated context + query x → predicted y
        combined = torch.cat([aggregated, query_x.squeeze(1)], dim=-1)
        pred_y = self.decoder(combined)
        return pred_y

# 3. Training Loop
def train_pfn(model, num_epochs=100, batch_size=32):
    dataset = PriorDataset(num_tasks=10000, num_samples_per_task=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for context_x, context_y in dataloader:
            # For each task, split into context and target (simulating few-shot learning)
            num_context = np.random.randint(3, 8)  # Random context size (3-7 samples)
            x_context = context_x[:, :num_context, :]
            y_context = context_y[:, :num_context, :]
            x_query = context_x[:, num_context:, :]
            y_query = context_y[:, num_context:, :]
            
            optimizer.zero_grad()
            pred_y = model(x_context, y_context, x_query)
            loss = F.mse_loss(pred_y, y_query.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

# 4. Inference
def predict(model, context_x, context_y, query_x):
    """Predict for a new task given a context (x, y) and query points."""
    with torch.no_grad():
        pred_y = model(context_x.unsqueeze(0), context_y.unsqueeze(0), query_x.unsqueeze(0))
    return pred_y.squeeze(0)

# Example Usage
if __name__ == "__main__":
    # Initialize and train
    pfn = PriorFittedNetwork(input_dim=1)
    train_pfn(pfn, num_epochs=50)
    
    # Test on a new task (e.g., y = 2x + 1 + noise)
    context_x = torch.tensor([[1.0], [2.0], [3.0]])
    context_y = 2 * context_x + 1 + torch.randn(3, 1) * 0.1
    query_x = torch.tensor([[4.0], [5.0]])
    pred_y = predict(pfn, context_x, context_y, query_x)
    print(f"Predicted y for x=4,5: {pred_y}")