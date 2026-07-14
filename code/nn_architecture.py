import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(PyTorchMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.network(x)

#Helper Function to Prepare PyTorch DataLoaders ---
def make_loader(X, y, batch_size=64, shuffle=True, sample_weights=None):
    # Convert pandas DataFrame/Series to numpy arrays if they aren't already
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y
    
    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
    
    if sample_weights is not None:
        w_arr = sample_weights.values if hasattr(sample_weights, "values") else sample_weights
        w_tensor = torch.tensor(w_arr, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    else:
        dataset = TensorDataset(X_tensor, y_tensor)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#Training Loop ---
def train_pytorch_model(dataloader, input_dim, epochs=20, pos_weight=None, use_sample_weights=False,device=device):
    model = PyTorchMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Configure loss function
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    else:
        reduction = 'none' if use_sample_weights else 'mean'
        criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            if use_sample_weights:
                batch_X, batch_y, batch_w = [b.to(device) for b in batch]
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss = (loss * batch_w).mean() # Apply individual sample weights manually
            else:
                batch_X, batch_y = [b.to(device) for b in batch]
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                
            loss.backward()
            optimizer.step()
            
    return model

#Extract raw probabilities
def get_probabilities(model, X_data, device):
    model.eval()
    
    X_arr = X_data.values if hasattr(X_data, "values") else X_data
    
    X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs
