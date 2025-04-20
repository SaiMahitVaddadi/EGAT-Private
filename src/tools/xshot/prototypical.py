import torch
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np




class PrototypicalNetwork(nn.Module):
    def __init__(self,embeddings,support_labels,bins=20):
        self.compute_prototypes(embeddings,support_labels,bins)


    def compute_prototypes(self, embeddings, support_labels, n_conditions):
        # Discretize continuous labels into N conditions
        bins = torch.linspace(support_labels.min(), support_labels.max(), n_conditions+1)
        condition_idx = torch.bucketize(support_labels, bins[:-1])
        
        prototypes = []
        for i in range(n_conditions):
            mask = (condition_idx == i)
            prototypes.append(embeddings[mask].mean(dim=0))
        
        return torch.stack(prototypes), bins



class ProtoGNN(nn.Module):
    def __init__(self, node_features, gnn_hidden, embedding_dim, n_conditions):
        super().__init__()
        self.gnn = GNNEmbedding(node_features, gnn_hidden, embedding_dim)
        self.predictor = PropertyPredictor(embedding_dim)
        self.n_conditions = n_conditions
        
    def forward(self, support_data, query_data):
        # Embed all graphs
        support_embeddings = self.gnn(support_data)
        query_embeddings = self.gnn(query_data)
        
        # Compute prototypes
        prototypes, bins = compute_prototypes(
            support_embeddings, 
            support_data.y, 
            self.n_conditions
        )
        
        # Predict query properties
        preds = self.predictor(query_embeddings, prototypes)
        
        return preds, bins

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for support_data, query_data in dataloader:
        support_data, query_data = support_data.to(device), query_data.to(device)
        
        optimizer.zero_grad()
        
        preds, _ = model(support_data, query_data)
        loss = nn.MSELoss()(preds, query_data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Configuration
config = {
    'node_features': 10,       # Number of node features in your graphs
    'gnn_hidden': 128,         # GNN hidden dimension
    'embedding_dim': 64,       # Final embedding dimension
    'n_conditions': 5,        # Number of prototype conditions
    'lr': 1e-3,               # Learning rate
    'n_way': 5,               # Number of conditions per episode
    'k_shot': 10,             # Support examples per condition
    'q_queries': 5            # Query examples per condition
}

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProtoGNN(
    config['node_features'],
    config['gnn_hidden'],
    config['embedding_dim'],
    config['n_conditions']
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


def predict(model, support_data, query_data, device):
    model.eval()
    with torch.no_grad():
        support_data, query_data = support_data.to(device), query_data.to(device)
        preds, bins = model(support_data, query_data)
    return preds.cpu().numpy(), bins.cpu().numpy()

